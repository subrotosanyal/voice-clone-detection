"""End-to-end test through the real FastAPI app's WebSocket path: enough
windows of real spoken audio sent over /v1/stream/{session_id} eventually
produce a live transcript that feeds the contextual third signal and the
intent detector — same signals test_transcription_api.py already proves
for the file-upload path, exercised here over the live streaming path
instead (see app/pipeline/live_transcription.py).

Uses macOS's `say` to generate a genuine (if synthetic-voice) spoken
fixture at test time — skipped gracefully wherever `say` isn't available.
"""
from __future__ import annotations

import itertools
import shutil
import subprocess
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.live_transcription import _TRANSCRIBE_EVERY_MS

SR = 16_000
WINDOW_MS = 2000
HOP_MS = 500
# Generous, not tight: this drives a real CPU-bound pipeline (AASIST +
# Parselmouth scoring every 500ms, plus a real Whisper transcription and
# intent classification running in the background) — machine speed
# genuinely varies (a slower/shared CI runner can take noticeably longer
# than a dev laptop), and this test must not flake just because a single
# run happened to be slow. A real bug (the feature never firing at all)
# still fails this within a finite, if generous, bound.
MAX_WAIT_S = 60


@pytest.fixture(autouse=True)
def _ensure_checkpoint_before_app_startup(aasist_checkpoint):
    return aasist_checkpoint


@pytest.fixture(scope="module")
def spoken_urgency_samples(tmp_path_factory) -> np.ndarray:
    if shutil.which("say") is None:
        pytest.skip("macOS 'say' not available — real-speech test skipped on this platform")

    tmp_dir = tmp_path_factory.mktemp("say_fixtures_ws")
    aiff_path = tmp_dir / "urgency.aiff"
    wav_path = tmp_dir / "urgency.wav"
    subprocess.run(
        ["say", "-o", str(aiff_path), "Please transfer the money immediately, it's urgent."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), "-ar", str(SR), "-ac", "1", str(wav_path)],
        check=True,
        capture_output=True,
    )
    samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    assert sr == SR
    # Loop the clip so there's comfortably more than enough audio to cross
    # _TRANSCRIBE_EVERY_MS + _TRANSCRIPTION_WINDOW_MS worth of windows.
    return np.tile(samples, 6)


def _trailing_windows(samples: np.ndarray) -> list[np.ndarray]:
    """Same overlapping-window shape app/ui/app.js's trailingWindow()
    produces client-side: a WINDOW_MS-long slice every HOP_MS."""
    hop = round(SR * HOP_MS / 1000)
    window = round(SR * WINDOW_MS / 1000)
    windows = []
    cursor = window
    while cursor <= samples.size:
        windows.append(samples[cursor - window : cursor])
        cursor += hop
    return windows


def test_live_transcript_eventually_feeds_intent_and_contextual_detectors(spoken_urgency_samples):
    windows = _trailing_windows(spoken_urgency_samples)
    session_id = "ws-live-transcription-test"
    hops_to_trigger = _TRANSCRIBE_EVERY_MS // HOP_MS
    deadline = time.monotonic() + MAX_WAIT_S

    with TestClient(app) as client, client.websocket_connect(f"/v1/stream/{session_id}") as ws:
        found_transcript = False
        found_intent = False
        # Cycle through the (finite) window list rather than stopping once
        # exhausted — a real live call keeps sending windows indefinitely,
        # and this test's job is to keep feeding the connection until the
        # background transcription completes or MAX_WAIT_S genuinely runs out.
        #
        # Transcript and intent are checked independently, not "intent must
        # already be done the moment transcript first appears" — a real fix
        # (2026-09-08) decoupled intent classification into its own
        # independently-scheduled task specifically so a slow intent call
        # no longer blocks the next transcription cycle; that means intent
        # now genuinely lags transcript by design, not a bug to paper over.
        for seq, window in enumerate(itertools.cycle(windows)):
            ws.send_json(
                {
                    "seq": seq,
                    "sample_rate": SR,
                    "pcm_f32": window.tolist(),
                    "window_start_ms": seq * HOP_MS,
                }
            )
            body = ws.receive_json()
            third_signal = next(c for c in body["components"] if c["name"] == "third_signal")
            if not found_transcript and "transcript" in third_signal["detail"]:
                found_transcript = True
                assert "transfer" in third_signal["detail"]["transcript"].lower()

            intent = next(c for c in body["components"] if c["name"] == "intent")
            if not found_intent and intent["abstained"] is False:
                found_intent = True

            if found_transcript and found_intent:
                break

            if time.monotonic() >= deadline:
                break

            if seq >= hops_to_trigger:
                # The background transcription/intent tasks were just
                # scheduled (or already running) on a real OS thread via
                # run_in_threadpool — give them real wall-clock time to
                # actually finish (observed inside the real deployed
                # container: ~3-5s for transcription, ~5s for intent,
                # more on a slower machine) between sends, rather than
                # racing through every remaining window before they can.
                time.sleep(0.5)

        assert found_transcript, (
            f"expected a live transcript to appear in third_signal's detail within {MAX_WAIT_S}s "
            "of looped spoken audio — none did"
        )
        assert found_intent, (
            f"expected the intent detector to stop abstaining within {MAX_WAIT_S}s "
            "of looped spoken audio — it never did"
        )

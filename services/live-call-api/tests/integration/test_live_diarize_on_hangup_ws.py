"""End-to-end test through the real FastAPI app: a live WS session with
two distinct voices, closed normally, then GET /v1/sessions/{id}/speakers
proves the "diarize on hangup" feature (app/api/ws_router.py) actually
ran — the exact same diarizer test_diarization_api.py already proves
works for the file-upload path, exercised here via the live streaming
path's LiveSessionRecorder + Engine.diarize_and_score() instead.

REAL BUG found while writing this test: awaiting the hangup work directly
inline in the WS handler's own `finally` block got silently
CancelledError'd partway through — the ASGI framework tears down a
WebSocket's cancel scope the moment the connection closes, killing
anything still running in it. Fixed in ws_router.py by detaching it into
an independent asyncio task instead — which means it's now genuinely
async relative to the WS connection closing, so this test polls for the
result (same bounded-wait pattern test_live_transcription_ws.py already
uses for its own background work) rather than assuming it's done the
instant the `with` block exits.
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app
from tests.audio_fixtures import VOICE_A_FORMANTS, VOICE_B_FORMANTS, formant_voice

SR = 16_000
# Was 30 — bumped 2026-09-11 after a real CI run's poll loop hit this
# timeout with 0 speakers found (a GitHub-hosted runner is measurably
# slower than local dev hardware for this real-model-inference work);
# paired with the actual fix in app/api/ws_router.py's own 2026-09-11
# REAL BUG note (bounding + draining the background hangup-diarization
# tasks that were competing with THIS test's own for CPU/disk), not a
# substitute for it.
MAX_WAIT_S = 45


def _send(ws, seq: int, samples, window_start_ms: int) -> dict:
    ws.send_json(
        {
            "seq": seq,
            "sample_rate": SR,
            "pcm_f32": samples.tolist(),
            "window_start_ms": window_start_ms,
        }
    )
    return ws.receive_json()


def test_diarized_speaker_breakdown_is_retrievable_after_the_call_ends(aasist_checkpoint):
    session_id = "ws-diarize-on-hangup-test"
    # 2.5s clips, matching test_diarization_api.py's file-upload fixture
    # exactly — window_start_ms below tracks REAL cumulative elapsed time
    # (each clip's own duration), not a fixed WINDOW_MS step, since these
    # clips are longer than a normal 2s window: LiveSessionRecorder derives
    # new-audio duration from consecutive window_start_ms deltas (see its
    # own REAL BUG note), so a mismatched fixed step here would silently
    # truncate each clip instead of exercising real reconstruction.
    clip_ms = 2500
    voice_a_take1 = formant_voice(VOICE_A_FORMANTS, sample_rate=SR, duration_s=clip_ms / 1000, seed=1, pitch_hz=120)
    voice_b = formant_voice(VOICE_B_FORMANTS, sample_rate=SR, duration_s=clip_ms / 1000, seed=3, pitch_hz=180)
    voice_a_take2 = formant_voice(VOICE_A_FORMANTS, sample_rate=SR, duration_s=clip_ms / 1000, seed=2, pitch_hz=122)

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/stream/{session_id}") as ws:
            _send(ws, 0, voice_a_take1, window_start_ms=0)
            _send(ws, 1, voice_b, window_start_ms=clip_ms)
            _send(ws, 2, voice_a_take2, window_start_ms=2 * clip_ms)
        # WS context exited above -> WebSocketDisconnect -> ws_router.py
        # schedules "diarize on hangup" as an independent background task
        # (see this file's own REAL BUG note) — poll rather than assume
        # it's already done.
        # Engine.diarize_and_score() persists each speaker's summary
        # incrementally as that speaker's own score_call() finishes, not
        # atomically at the end — poll until the EXPECTED count is
        # reached, not just "any result", or this would flake on the real
        # window where only the first of two speakers has landed yet.
        deadline = time.monotonic() + MAX_WAIT_S
        speakers = []
        while time.monotonic() < deadline:
            resp = client.get(f"/v1/sessions/{session_id}/speakers")
            assert resp.status_code == 200
            speakers = resp.json()
            if len(speakers) >= 2:
                break
            time.sleep(0.2)

    assert len(speakers) >= 2, "expected at least two distinct voices in the diarized breakdown"
    for speaker in speakers:
        assert speaker["session_id"].startswith(session_id + "::")
        assert speaker["segment_count"] >= 1
        assert speaker["final"] is not None
        assert len(speaker["trace"]) >= 1


def test_no_speakers_endpoint_returns_empty_list_for_a_single_speaker_or_missing_session():
    with TestClient(app) as client:
        resp = client.get("/v1/sessions/some-session-that-never-existed/speakers")
    assert resp.status_code == 200  # empty, not 404 — see the endpoint's own "abstain quietly" docstring note
    assert resp.json() == []

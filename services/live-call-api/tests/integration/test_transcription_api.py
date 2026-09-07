"""End-to-end test through the real FastAPI app: a real spoken recording
uploaded to POST /v1/score/file gets transcribed automatically, and the
contextual third signal picks up the urgency/financial-request language
from it — no urgency_keywords typed in by hand.

Uses macOS's `say` to generate a genuine (if synthetic-voice) spoken
fixture at test time — skipped gracefully wherever `say` isn't available.
"""
from __future__ import annotations

import io
import shutil
import subprocess

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

SR = 16_000


@pytest.fixture(autouse=True)
def _ensure_checkpoint_before_app_startup(aasist_checkpoint):
    return aasist_checkpoint


@pytest.fixture(scope="module")
def spoken_urgency_wav_bytes(tmp_path_factory) -> bytes:
    if shutil.which("say") is None:
        pytest.skip("macOS 'say' not available — real-speech test skipped on this platform")

    tmp_dir = tmp_path_factory.mktemp("say_fixtures")
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
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


def test_transcript_auto_feeds_contextual_urgency_detection(spoken_urgency_wav_bytes):
    with TestClient(app) as client:
        resp = client.post(
            "/v1/score/file",
            files={"file": ("call.wav", spoken_urgency_wav_bytes, "audio/wav")},
            # deliberately NOT supplying urgency_keywords/is_financial_request —
            # this is exactly what transcription is for: detecting it automatically
            data={"context": "{}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    third_signal = next(c for c in body["final"]["components"] if c["name"] == "third_signal")
    detail = third_signal["detail"]

    assert "transcript" in detail
    assert "transfer" in detail["transcript"].lower()
    assert detail["rules_fired"]["urgency_language"] is True
    assert detail["rules_fired"]["financial_request"] is True
    assert detail["is_financial_request_from_transcript"] is True

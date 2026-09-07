"""End-to-end test through the real FastAPI app: POST /v1/score/file with
diarize=true -> per-speaker sub-scoring, fanned out through the SAME
Engine.score_call() path as a normal whole-call score.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from tests.audio_fixtures import VOICE_A_FORMANTS, VOICE_B_FORMANTS, formant_voice

SR = 16_000


@pytest.fixture(autouse=True)
def _ensure_checkpoint_before_app_startup(aasist_checkpoint):
    return aasist_checkpoint


def _multi_speaker_wav_bytes() -> bytes:
    voice_a_take1 = formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120, duration_s=2.5)
    voice_b = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=2.5)
    voice_a_take2 = formant_voice(VOICE_A_FORMANTS, seed=2, pitch_hz=122, duration_s=2.5)
    full_call = np.concatenate([voice_a_take1, voice_b, voice_a_take2])
    buf = io.BytesIO()
    sf.write(buf, full_call, SR, format="WAV")
    return buf.getvalue()


def test_score_file_without_diarize_omits_speakers():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/score/file",
            files={"file": ("call.wav", _multi_speaker_wav_bytes(), "audio/wav")},
        )
    assert resp.status_code == 200
    assert resp.json()["speakers"] is None


def test_score_file_with_diarize_true_returns_per_speaker_breakdown():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/score/file",
            files={"file": ("call.wav", _multi_speaker_wav_bytes(), "audio/wav")},
            data={"diarize": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    speakers = body["speakers"]
    assert speakers is not None
    assert len(speakers) >= 2  # at least two distinct voices detected

    for speaker in speakers:
        assert speaker["session_id"].startswith(body["session_id"] + "::")
        assert speaker["segment_count"] >= 1
        assert speaker["total_duration_ms"] > 0
        assert 0.0 <= speaker["final"]["smoothed_score_0_100"] <= 100.0
        assert len(speaker["trace"]) > 0

    # each speaker's sub-session was really saved through the normal
    # history machinery, independently retrievable later
    for speaker in speakers:
        history_resp = client.get(f"/v1/sessions/{speaker['session_id']}")
        assert history_resp.status_code == 200
        assert history_resp.json()["window_count"] == len(speaker["trace"])

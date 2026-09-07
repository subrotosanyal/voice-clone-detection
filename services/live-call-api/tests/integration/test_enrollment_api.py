"""End-to-end test through the real FastAPI app: enroll -> score/file with
claimed_identity -> full breakdown reflects the real ECAPA-TDNN comparison.

Shares the app's real data/voiceprints.db (same accepted trade-off
docs/testing.md already documents for data/sessions.db) — every test uses
a uuid-suffixed identity so rows never collide across runs.
"""
from __future__ import annotations

import io
import uuid

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


def _wav_bytes(samples: np.ndarray, sample_rate: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def test_enroll_list_and_delete_round_trip():
    identity = f"test-{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        enroll_resp = client.post(
            "/v1/enroll",
            data={"identity": identity},
            files={"file": ("voice.wav", _wav_bytes(formant_voice(VOICE_A_FORMANTS, seed=1)), "audio/wav")},
        )
        assert enroll_resp.status_code == 200
        assert enroll_resp.json()["identity"] == identity

        list_resp = client.get("/v1/enrollments")
        assert list_resp.status_code == 200
        assert identity in {e["identity"] for e in list_resp.json()}

        delete_resp = client.delete(f"/v1/enrollments/{identity}")
        assert delete_resp.status_code == 200

        delete_again_resp = client.delete(f"/v1/enrollments/{identity}")
        assert delete_again_resp.status_code == 404


def test_score_file_with_claimed_identity_uses_real_voiceprint_comparison():
    identity = f"test-{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        client.post(
            "/v1/enroll",
            data={"identity": identity},
            files={
                "file": (
                    "voice.wav",
                    _wav_bytes(formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120)),
                    "audio/wav",
                )
            },
        )

        same_speaker_resp = client.post(
            "/v1/score/file",
            files={
                "file": (
                    "call.wav",
                    _wav_bytes(formant_voice(VOICE_A_FORMANTS, seed=2, pitch_hz=122, duration_s=2.5)),
                    "audio/wav",
                )
            },
            data={"context": f'{{"claimed_identity": "{identity}"}}'},
        )
        different_speaker_resp = client.post(
            "/v1/score/file",
            files={
                "file": (
                    "call.wav",
                    _wav_bytes(formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=2.5)),
                    "audio/wav",
                )
            },
            data={"context": f'{{"claimed_identity": "{identity}"}}'},
        )

        client.delete(f"/v1/enrollments/{identity}")

    assert same_speaker_resp.status_code == 200
    assert different_speaker_resp.status_code == 200

    def _consistency_component(body: dict) -> dict:
        return next(c for c in body["final"]["components"] if c["name"] == "third_signal")

    same_component = _consistency_component(same_speaker_resp.json())
    different_component = _consistency_component(different_speaker_resp.json())

    assert same_component["detail"]["active_mode"] == "consistency"
    assert different_component["detail"]["active_mode"] == "consistency"
    assert same_component["raw_score"] < different_component["raw_score"]


def test_delete_unknown_enrollment_returns_404():
    with TestClient(app) as client:
        resp = client.delete(f"/v1/enrollments/never-enrolled-{uuid.uuid4().hex[:8]}")
    assert resp.status_code == 404


def test_enroll_rejects_undecodable_audio():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/enroll",
            data={"identity": f"test-{uuid.uuid4().hex[:8]}"},
            files={"file": ("voice.wav", b"not actually audio", "audio/wav")},
        )
    assert resp.status_code == 400

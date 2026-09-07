"""End-to-end: score a file, then confirm it shows up in session history."""
from __future__ import annotations

import io
import uuid

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def _ensure_checkpoint_before_app_startup(aasist_checkpoint):
    return aasist_checkpoint


def _wav_bytes(seed: int = 1, duration_s: float = 2.5, sample_rate: int = 16_000) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    rng = np.random.default_rng(seed)
    signal = (0.4 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
    signal += rng.normal(0, 0.01, size=signal.size).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV")
    return buf.getvalue()


def test_scored_session_appears_in_history_and_can_be_replayed():
    session_id = f"test-{uuid.uuid4().hex[:10]}"
    with TestClient(app) as client:
        score_resp = client.post(
            "/v1/score/file",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"session_id": session_id},
        )
        assert score_resp.status_code == 200
        original = score_resp.json()

        list_resp = client.get("/v1/sessions")
        assert list_resp.status_code == 200
        ids = [s["session_id"] for s in list_resp.json()]
        assert session_id in ids

        detail_resp = client.get(f"/v1/sessions/{session_id}")
        assert detail_resp.status_code == 200
        replayed = detail_resp.json()

    assert replayed["window_count"] == original["window_count"]
    assert replayed["final"]["smoothed_score_0_100"] == original["final"]["smoothed_score_0_100"]
    assert replayed["final"]["band"] == original["final"]["band"]
    assert len(replayed["trace"]) == len(original["trace"])


def test_unknown_session_returns_404():
    with TestClient(app) as client:
        resp = client.get("/v1/sessions/this-session-does-not-exist")
    assert resp.status_code == 404


def test_delete_session_then_404s():
    session_id = f"test-{uuid.uuid4().hex[:10]}"
    with TestClient(app) as client:
        client.post("/v1/score/file", files={"file": ("clip.wav", _wav_bytes(), "audio/wav")}, data={"session_id": session_id})

        delete_resp = client.delete(f"/v1/sessions/{session_id}")
        assert delete_resp.status_code == 200

        second_delete = client.delete(f"/v1/sessions/{session_id}")
        assert second_delete.status_code == 404

        get_resp = client.get(f"/v1/sessions/{session_id}")
        assert get_resp.status_code == 404


def test_list_sessions_respects_limit_query_param():
    with TestClient(app) as client:
        for _ in range(3):
            client.post(
                "/v1/score/file",
                files={"file": ("clip.wav", _wav_bytes(seed=int(uuid.uuid4().int % 10_000)), "audio/wav")},
            )
        resp = client.get("/v1/sessions", params={"limit": 1})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

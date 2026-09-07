"""End-to-end test through the real FastAPI app: upload -> pipeline -> JSON.

Generates its own in-memory WAV fixture rather than depending on
scripts/gen_test_audio.py having been run — `pytest` should work from a
clean checkout with zero setup beyond `pip install -r requirements-dev.txt`
(plus internet access on first run, to fetch the AASIST checkpoint the
default config now points the acoustic detector at — see conftest.py).
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def _ensure_checkpoint_before_app_startup(aasist_checkpoint):
    """The app's default config loads the AASIST detector eagerly at
    startup (inside `TestClient(app)`'s lifespan). If the checkpoint isn't
    there yet, that raises FileNotFoundError deep inside app startup —
    not a clean pytest.skip. Depending on `aasist_checkpoint` here forces
    the fetch (or a graceful skip of this whole module if offline) before
    any test in this file constructs a TestClient."""
    return aasist_checkpoint


def _wav_bytes(seed: int = 1, duration_s: float = 2.5, sample_rate: int = 16_000) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    rng = np.random.default_rng(seed)
    signal = (0.4 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
    signal += rng.normal(0, 0.01, size=signal.size).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV")
    return buf.getvalue()


def test_healthz():
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_config_endpoint_reflects_loaded_formula():
    with TestClient(app) as client:
        resp = client.get("/v1/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "formula_version" in body
    assert {d["name"] for d in body["detectors"]} == {"acoustic", "prosodic", "third_signal"}


def test_score_file_end_to_end():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/score/file",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"context": '{"known_number": true, "hour_of_day": 14}'},
        )

    assert resp.status_code == 200
    body = resp.json()

    assert body["window_count"] > 0
    assert len(body["trace"]) == body["window_count"]

    final = body["final"]
    assert 0.0 <= final["smoothed_score_0_100"] <= 100.0
    assert final["band"] in {"low", "elevated", "high"}
    assert final["formula_version"]
    assert len(final["components"]) == 3
    names = {c["name"] for c in final["components"]}
    assert names == {"acoustic", "prosodic", "third_signal"}
    third_signal = next(c for c in final["components"] if c["name"] == "third_signal")
    assert third_signal["detail"]["detector_name"]  # the actual implementation, for tracing


def test_score_file_rejects_bad_context_json():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/score/file",
            files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            data={"context": "{not json"},
        )
    assert resp.status_code == 400


def test_score_file_rejects_undecodable_audio():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/score/file",
            files={"file": ("clip.wav", b"not actually audio", "audio/wav")},
        )
    assert resp.status_code == 400


def test_repeated_calls_with_same_input_are_reproducible():
    """Same audio, same context, no cross-call state (session_id differs
    each time by default, but nothing else should vary) -> identical scores.
    This is the reproducibility guarantee from docs/risk-model.md."""
    payload = _wav_bytes(seed=99)
    with TestClient(app) as client:
        r1 = client.post("/v1/score/file", files={"file": ("clip.wav", payload, "audio/wav")})
        r2 = client.post("/v1/score/file", files={"file": ("clip.wav", payload, "audio/wav")})

    assert r1.json()["final"]["raw_score_0_100"] == r2.json()["final"]["raw_score_0_100"]
    assert r1.json()["final"]["smoothed_score_0_100"] == r2.json()["final"]["smoothed_score_0_100"]

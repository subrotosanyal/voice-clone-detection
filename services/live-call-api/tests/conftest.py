"""Shared pytest fixtures.

Auto-fetches the AASIST checkpoint on first local test run if it's
missing (idempotent — see vendor/fetch_checkpoint.py), so `pip install &&
pytest` still "just works" from a clean checkout with internet access.
Tests that need it request the `aasist_checkpoint` fixture explicitly;
everything else (the 33 tests from before this detector existed) stays
fully offline and unaffected. If fetching fails (no network), only the
AASIST-dependent tests are skipped — not the whole suite.
"""
from __future__ import annotations

import pytest

from app.adapters.detectors.vendor.fetch_checkpoint import ensure_checkpoint


@pytest.fixture(scope="session")
def aasist_checkpoint() -> str:
    try:
        path = ensure_checkpoint()
    except Exception as exc:  # noqa: BLE001 — any fetch failure just skips
        pytest.skip(f"AASIST checkpoint unavailable (no network?): {exc}")
    return str(path)


@pytest.fixture(scope="session")
def ecapa_extractor():
    """Session-scoped so the ~80MB speechbrain/spkrec-ecapa-voxceleb model
    is loaded once for every test that needs it (voiceprint consistency,
    diarization), not once per test. Same graceful-skip-if-offline pattern
    as aasist_checkpoint above — tests that don't request this fixture
    stay fully offline and unaffected."""
    try:
        from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor

        return EcapaEmbeddingExtractor()
    except Exception as exc:  # noqa: BLE001 — any load failure just skips
        pytest.skip(f"ECAPA-TDNN model unavailable (no network?): {exc}")


@pytest.fixture(scope="session")
def phi3_llm_model_path() -> str:
    """Same idempotent fetch-on-first-use pattern as aasist_checkpoint
    above — see app/adapters/semantic_risk/fetch_llm_model.py. This one
    is a real, ~2.4GB download the first time (Microsoft's official
    Phi-3-mini-4k-instruct GGUF) — heavier than every other model fixture
    here, cached afterward like the others."""
    try:
        from app.adapters.semantic_risk.fetch_llm_model import ensure_model

        path = ensure_model()
    except Exception as exc:  # noqa: BLE001 — any fetch failure just skips
        pytest.skip(f"Phi-3-mini GGUF unavailable (no network?): {exc}")
    return str(path)

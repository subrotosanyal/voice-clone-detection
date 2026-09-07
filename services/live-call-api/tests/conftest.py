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

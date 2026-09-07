"""Confirms the browser dashboard is actually served, and doesn't get
shadowed by the API routes it's mounted alongside (see app/main.py's
comment on mount order)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def _ensure_checkpoint_before_app_startup(aasist_checkpoint):
    return aasist_checkpoint


def test_index_page_served_at_root():
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Satya-Vani" in resp.text


def test_app_js_served():
    with TestClient(app) as client:
        resp = client.get("/app.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    assert "score/file" in resp.text  # sanity: it references the real API path


def test_static_files_are_not_cached_without_revalidation():
    """A browser silently serving a stale app.js after a rebuild is a real
    failure mode we hit once already (a page open from before a UI change
    kept running the old script). no-cache still lets the browser cache
    the file, but forces revalidation every time — see NoCacheStaticFiles
    in app/main.py."""
    with TestClient(app) as client:
        assert client.get("/").headers["cache-control"] == "no-cache"
        assert client.get("/app.js").headers["cache-control"] == "no-cache"


def test_api_routes_are_not_shadowed_by_the_static_mount():
    """The dashboard is mounted at "/" — this guards against a future
    change accidentally reordering things so it swallows API routes."""
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/config").status_code == 200
        assert client.get("/docs").status_code == 200

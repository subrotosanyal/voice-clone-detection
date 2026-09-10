"""Confirms the browser dashboard is actually served, and doesn't get
shadowed by the API routes it's mounted alongside (see app/main.py's
comment on mount order)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
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


def test_index_page_has_no_base_path_by_default():
    """settings.base_path defaults to "" (see app/config.py) -- the page
    served locally / by the plain docker-compose.yml deployment must be
    byte-identical to before BASE_PATH existed: no leftover
    "__SATYA_VANI_BASE_PATH__" placeholder, and every asset/API reference
    still a bare root-relative path."""
    with TestClient(app) as client:
        resp = client.get("/")
    assert "__SATYA_VANI_BASE_PATH__" not in resp.text
    assert 'src="/app.js"' in resp.text
    assert 'href="/favicon.svg"' in resp.text
    assert 'window.__BASE_PATH__ = "";' in resp.text


def test_index_page_injects_a_configured_base_path():
    """REAL BUG found deploying path-routed on Traefik: without this, every
    fetch()/WebSocket URL app.js builds resolves against the domain root,
    not the path prefix Traefik routes this app under, and 404s on
    everything past the first HTML load (see
    deploy/portainer/satya-vani.yml's own long-form note). Confirms the
    "/" route actually substitutes settings.base_path into every place
    index.html references it."""
    original = settings.base_path
    settings.base_path = "/satya-vani"
    try:
        with TestClient(app) as client:
            resp = client.get("/")
    finally:
        settings.base_path = original
    assert "__SATYA_VANI_BASE_PATH__" not in resp.text
    assert 'src="/satya-vani/app.js"' in resp.text
    assert 'href="/satya-vani/favicon.svg"' in resp.text
    assert 'href="/satya-vani/docs"' in resp.text
    assert 'window.__BASE_PATH__ = "/satya-vani";' in resp.text


def test_index_html_path_is_also_templated():
    """A direct request for "/index.html" (rather than "/") must get the
    same substitution — otherwise it would serve the raw file straight off
    the static mount, with the literal "__SATYA_VANI_BASE_PATH__" placeholder
    still in it."""
    settings.base_path = "/satya-vani"
    try:
        with TestClient(app) as client:
            resp = client.get("/index.html")
    finally:
        settings.base_path = ""
    assert "__SATYA_VANI_BASE_PATH__" not in resp.text
    assert 'src="/satya-vani/app.js"' in resp.text

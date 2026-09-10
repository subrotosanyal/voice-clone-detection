"""FastAPI entrypoint — wires config, logging, the pipeline, and routes.

Run locally:      uvicorn app.main:app --reload
Run via compose:   docker compose up   (see repo root docker-compose.yml)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.adapters.detectors.voiceprint_consistency import VoiceprintConsistencyDetector
from app.adapters.history.sqlite_store import SqliteHistoryStore
from app.adapters.registry import build_pipeline
from app.api.enrollment_router import router as enrollment_router
from app.api.history_router import router as history_router
from app.api.http_router import router as http_router
from app.api.ws_router import drain_pending_hangup_diarizations
from app.api.ws_router import router as ws_router
from app.config import settings
from app.logging_setup import configure_logging, get_logger
from app.pipeline.engine import Engine

logger = get_logger(component="startup")


def _limit_cpu_threads() -> None:
    """REAL BUG found 2026-09-09 (live-mic parity investigation): torch's
    own intra-op/inter-op thread pools default to using every visible
    core, independent of the Dockerfile's OMP_NUM_THREADS/MKL_NUM_THREADS/
    OPENBLAS_NUM_THREADS env vars (those bound the underlying BLAS
    libraries torch calls into; torch's own parallelism knobs are
    separate and must be set explicitly — see PyTorch's own docs on CPU
    threading). On the live WS path, AASIST and ECAPA-TDNN (both torch)
    run every ~500ms, concurrently with each other and with the
    background transcription task — measured by hand: per-window
    latency inflated to 0.85-1.5s against the 500ms target. Matches the
    Dockerfile's own ENV thread-limiting note; called once at startup,
    before build_pipeline() loads any model, since PyTorch requires
    set_num_interop_threads() to run before any inter-op parallel work
    has started.

    settings.torch_num_threads/torch_interop_threads (default 2/1, see
    app/config.py) exist because this same cap also throttles the
    ONE-SHOT file-upload path, which never had the concurrency problem
    above — score_window() runs every detector sequentially, one file at
    a time. REAL ISSUE found deploying on a 16-core/64GB Unraid box: `top`
    showed uvicorn pinned at ~156% CPU while 14+ cores sat idle — self-
    imposed, not weak hardware. Raise these via env vars on a host with
    cores to spare; the default stays 2/1 so nothing changes anywhere
    that doesn't explicitly override them."""
    import torch

    torch.set_num_threads(settings.torch_num_threads)
    try:
        torch.set_num_interop_threads(settings.torch_interop_threads)
    except RuntimeError:
        # REAL BUG found running the test suite: torch only allows
        # set_num_interop_threads() to be called ONCE per process,
        # before any parallel work starts — raises otherwise. A single
        # test-suite process instantiates `TestClient(app)` (and this
        # lifespan) many times over; the first call succeeds and every
        # later one hits this exact RuntimeError. Harmless to ignore: it
        # only ever means interop parallelism is already configured
        # (either by an earlier call in this same process, same intent,
        # or because torch's own default already kicked in) — never a
        # sign the thread limit silently failed to apply.
        pass


class NoCacheStaticFiles(StaticFiles):
    """Adds `Cache-Control: no-cache` to every static response.

    Without this, browsers can silently keep serving an old cached
    index.html/app.js after a rebuild — the exact confusion that made the
    History tab look broken right after it was added to an already-open
    tab. `no-cache` (not `no-store`) still lets the browser use its cached
    copy, but only after revalidating via If-None-Match — cheap, and
    always correct. This is a UI served straight off disk, not a CDN
    asset — correctness matters far more than shaving one round trip.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    _limit_cpu_threads()
    pipeline = build_pipeline(settings.risk_config_path)
    history = SqliteHistoryStore(settings.history_db_path)
    app.state.pipeline = pipeline
    app.state.history = history
    app.state.engine = Engine(
        pipeline,
        history=history,
        transcriber=pipeline.transcriber,
        live_transcriber=pipeline.live_transcriber,
        intent_classifier=pipeline.intent_classifier,
        semantic_risk_classifier=pipeline.semantic_risk_classifier,
        live_speaker_embedder=pipeline.live_speaker_embedder,
    )

    # If the configured third_signal.consistency_class is the real
    # VoiceprintConsistencyDetector, expose that SAME instance to
    # app/api/enrollment_router.py — reusing its already-loaded ECAPA-TDNN
    # model rather than loading a second copy just for enrollment. Stays
    # None (enrollment endpoints 503) if the always-abstaining stub is
    # configured instead — see config/risk_formula.yaml's fallback comment.
    app.state.voiceprint_detector = _find_voiceprint_detector(pipeline.detectors)

    logger.info(
        "pipeline_loaded",
        formula_version=pipeline.config["formula_version"],
        detectors=[d["name"] for d in pipeline.config["detectors"]],
        third_signal_mode=next(
            d["params"]["mode"] for d in pipeline.config["detectors"] if d["name"] == "third_signal"
        ),
        voiceprint_enrollment_available=app.state.voiceprint_detector is not None,
        diarization_available=pipeline.diarizer is not None,
        transcription_available=pipeline.transcriber is not None,
        live_transcription_available=pipeline.live_transcriber is not None,
        intent_classification_available=pipeline.intent_classifier is not None,
        semantic_risk_classification_available=pipeline.semantic_risk_classifier is not None,
        live_diarization_available=pipeline.live_speaker_embedder is not None,
    )
    yield
    # REAL BUG found 2026-09-11 (CI): without this, ws_router.py's
    # detached "diarize on hangup" background tasks (see its own
    # 2026-09-11 REAL BUG note) silently outlive this app's own
    # lifespan — in this project's integration test suite specifically,
    # where many `with TestClient(app) as client:` blocks reuse the SAME
    # process, that let one test's hangup work keep running concurrently
    # with the NEXT test's, piling up and exhausting a CI runner's disk.
    # Bounded (see drain_pending_hangup_diarizations' own docstring): a
    # stuck task must not hang shutdown forever.
    await drain_pending_hangup_diarizations()


def _find_voiceprint_detector(detectors) -> VoiceprintConsistencyDetector | None:
    """Walks the built detector list for ThirdSignalRouter's
    consistency_detector, returning it only if it's the real
    VoiceprintConsistencyDetector (not the always-abstaining stub)."""
    for detector in detectors:
        consistency_detector = getattr(detector, "consistency_detector", None)
        if isinstance(consistency_detector, VoiceprintConsistencyDetector):
            return consistency_detector
    return None


app = FastAPI(
    title="Satya-Vani — Live Call Path",
    description="Pluggable real-time voice-spoof risk scoring. See /docs and the project README.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(http_router)
app.include_router(ws_router)
app.include_router(history_router)
app.include_router(enrollment_router)

_UI_DIR = Path(__file__).resolve().parent / "ui"


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def _serve_index() -> HTMLResponse:
    """Serves index.html with `settings.base_path` substituted for every
    "__SATYA_VANI_BASE_PATH__" placeholder — the one templated exception to
    the otherwise-static UI mount below.

    REAL BUG found writing this: the placeholder can't be "__BASE_PATH__"
    itself, because index.html also assigns that exact text as the JS
    global's own name (`window.__BASE_PATH__ = "..."`) — a naive
    str.replace() then mangles the identifier too, turning it into
    `window./satya-vani = "/satya-vani";` (caught by
    test_index_page_injects_a_configured_base_path). The placeholder and
    the JS global it fills in must not share a spelling.

    WHY: this app's UI/API/WebSocket code is all root-relative
    (fetch("/v1/..."), WebSocket to `${location.host}/v1/stream/...`),
    which only resolves correctly when served at its origin's root. When
    Traefik path-routes this app under a prefix on a shared domain (see
    deploy/portainer/satya-vani.yml), the BROWSER must be told to prefix
    every request it makes with that same prefix — Traefik only ever sees
    the request the browser actually sends, so there's nothing a reverse
    proxy alone can do to "put the prefix back" on a request the page
    built without it (REAL BUG found deploying: app.js/index.html
    404'd on every asset and API call once path-routed, exactly because
    of this). index.html gets `window.__BASE_PATH__` injected here;
    app/ui/app.js reads it and prefixes every fetch()/WebSocket URL with
    it. Traefik still strips the prefix back off before forwarding (see
    the stack file's stripprefix middleware), so this container itself
    never has to learn to serve anything but the plain root-relative
    paths it already does — `settings.base_path` defaults to "" (see
    app/config.py), so locally and via the plain docker-compose.yml
    deployment this substitutes to nothing and the page is
    byte-identical to before this existed.
    """
    html = (_UI_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("__SATYA_VANI_BASE_PATH__", settings.base_path)
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})


# Browser dashboard — mounted LAST and at the root, so it only catches
# paths the routers/routes above didn't already claim (/app.js,
# /favicon.svg, ...). The routes above (/, /index.html, /healthz, /v1/*,
# and FastAPI's own /docs, /redoc, /openapi.json) always win first. See
# app/ui/README or docs/architecture.md.
app.mount("/", NoCacheStaticFiles(directory="app/ui", html=True), name="ui")

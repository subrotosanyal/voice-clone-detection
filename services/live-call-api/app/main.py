"""FastAPI entrypoint — wires config, logging, the pipeline, and routes.

Run locally:      uvicorn app.main:app --reload
Run via compose:   docker compose up   (see repo root docker-compose.yml)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.adapters.detectors.voiceprint_consistency import VoiceprintConsistencyDetector
from app.adapters.history.sqlite_store import SqliteHistoryStore
from app.adapters.registry import build_pipeline
from app.api.enrollment_router import router as enrollment_router
from app.api.history_router import router as history_router
from app.api.http_router import router as http_router
from app.api.ws_router import router as ws_router
from app.config import settings
from app.logging_setup import configure_logging, get_logger
from app.pipeline.engine import Engine

logger = get_logger(component="startup")


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
    pipeline = build_pipeline(settings.risk_config_path)
    history = SqliteHistoryStore(settings.history_db_path)
    app.state.pipeline = pipeline
    app.state.history = history
    app.state.engine = Engine(
        pipeline, history=history, transcriber=pipeline.transcriber, intent_classifier=pipeline.intent_classifier
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
        intent_classification_available=pipeline.intent_classifier is not None,
    )
    yield


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

# Browser dashboard — mounted LAST and at the root, so it only catches
# paths the routers above didn't already claim (/, /app.js). The routes
# above (/healthz, /v1/*, and FastAPI's own /docs, /redoc, /openapi.json)
# always win first. See app/ui/README or docs/architecture.md.
app.mount("/", NoCacheStaticFiles(directory="app/ui", html=True), name="ui")

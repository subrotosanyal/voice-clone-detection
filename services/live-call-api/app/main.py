"""FastAPI entrypoint — wires config, logging, the pipeline, and routes.

Run locally:      uvicorn app.main:app --reload
Run via compose:   docker compose up   (see repo root docker-compose.yml)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.adapters.registry import build_pipeline
from app.api.http_router import router as http_router
from app.api.ws_router import router as ws_router
from app.config import settings
from app.logging_setup import configure_logging, get_logger
from app.pipeline.engine import Engine

logger = get_logger(component="startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    pipeline = build_pipeline(settings.risk_config_path)
    app.state.pipeline = pipeline
    app.state.engine = Engine(pipeline)
    logger.info(
        "pipeline_loaded",
        formula_version=pipeline.config["formula_version"],
        detectors=[d["name"] for d in pipeline.config["detectors"]],
        third_signal_mode=next(
            d["params"]["mode"] for d in pipeline.config["detectors"] if d["name"] == "third_signal"
        ),
    )
    yield


app = FastAPI(
    title="Satya-Vani — Live Call Path",
    description="Pluggable real-time voice-spoof risk scoring. See /docs and the project README.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(http_router)
app.include_router(ws_router)

# Browser dashboard — mounted LAST and at the root, so it only catches
# paths the routers above didn't already claim (/, /app.js). The routes
# above (/healthz, /v1/*, and FastAPI's own /docs, /redoc, /openapi.json)
# always win first. See app/ui/README or docs/architecture.md.
app.mount("/", StaticFiles(directory="app/ui", html=True), name="ui")

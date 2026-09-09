"""WebSocket streaming endpoint — the actual "live call" path.

A client (a browser capturing mic audio, or a telephony bridge) opens
/v1/stream/{session_id} and sends one JSON message per audio window,
matching StreamChunkIn. The service scores it immediately and sends back
one FusedScoreOut. State (the EMA smoothing) lives in Engine.sessions,
keyed by session_id, for as long as the process runs.

For debugging without writing a WebSocket client, prefer
POST /v1/score/file (app/api/http_router.py) — it exercises the exact same
Engine.score_window() path per window, just fed from a whole file instead
of a socket.

Live transcription (added 2026-09-08): a LiveTranscriptionBuffer,
scoped to this one connection, accumulates audio and periodically
transcribes it in the background — see app/pipeline/live_transcription.py
for how and its own honesty note on the lag this introduces. Its latest
completed result (if any) is merged into every subsequent window's
context, which is also what lets the already-active intent detector
(config/risk_formula.yaml's `intent` entry, weight 0.15) start
contributing on the live path too, not just file uploads — same
detector, same known calibration caveat (see docs/risk-model.md,
"Intent detection"), now also fed a (lagging) live transcript instead of
always abstaining for lack of one.

Live diarization (added 2026-09-09): a LiveSpeakerTracker, ALSO scoped
to this one connection, incrementally attributes each window to a
speaker as the call progresses — see app/pipeline/live_diarization.py
for how, including its own honesty note on what's calibrated and what
isn't. Unlike live transcription, this does NOT merge into `context`
(no detector reads it, it doesn't change the score) — its result is
attached directly onto the outgoing FusedScore as `live_speaker`, purely
informational. Silently produces nothing (fused.live_speaker stays None)
when `live_diarization:` isn't configured in risk_formula.yaml, same
"omit the config section to disable" shape as transcription/intent.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.schemas import FusedScoreOut, StreamChunkIn
from app.domain.models import AudioWindow
from app.logging_setup import get_logger
from app.pipeline.live_diarization import LiveSpeakerTracker
from app.pipeline.live_transcription import LiveTranscriptionBuffer

router = APIRouter()
logger = get_logger(component="ws_router")


@router.websocket("/v1/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    engine = websocket.app.state.engine
    logger.info("ws_session_opened", session_id=session_id)

    live_transcription = LiveTranscriptionBuffer(
        transcriber=engine.transcriber,
        intent_classifier=engine.intent_classifier,
        semantic_risk_classifier=engine.semantic_risk_classifier,
        hop_ms=engine.pipeline.config["windowing"]["hop_ms"],
    )
    live_diarization_params = engine.pipeline.config.get("live_diarization", {}).get("params", {})
    live_speaker_tracker = LiveSpeakerTracker(
        embedder=engine.live_speaker_embedder,
        floor_rms=live_diarization_params.get("floor_rms", 1e-4),
        similarity_threshold=live_diarization_params.get("similarity_threshold", 0.75),
        ema_alpha=live_diarization_params.get("ema_alpha", 0.1),
        max_speakers=live_diarization_params.get("max_speakers", 8),
    )

    try:
        while True:
            raw = await websocket.receive_json()
            try:
                chunk = StreamChunkIn(**raw)
            except ValidationError as exc:
                await websocket.send_json({"error": "invalid_chunk", "detail": exc.errors()})
                continue

            window = AudioWindow(
                session_id=session_id,
                seq=chunk.seq,
                sample_rate=chunk.sample_rate,
                samples=np.asarray(chunk.pcm_f32, dtype=np.float32),
                window_start_ms=chunk.window_start_ms,
            )
            context = chunk.context.model_dump(exclude_none=True) if chunk.context else {}

            live_transcription.ingest(window.samples, window.sample_rate)
            # setdefault, not unconditional assignment: never overwrite an
            # explicit caller-supplied value, same "never overwrite" rule
            # Engine.score_call() already applies to a caller-supplied
            # transcript on the file-upload path.
            for key, value in live_transcription.latest_context.items():
                context.setdefault(key, value)

            # CPU-bound (AASIST + Parselmouth, maybe ECAPA-TDNN) — offload
            # so this one session's per-window inference doesn't stall
            # every other concurrent WS session or HTTP request on the
            # single event loop. See http_router.py's score_file() for the
            # same reasoning. Diarization's embedding extraction is its
            # own separate CPU-bound forward pass — same reasoning, own
            # threadpool call, not bundled into engine.score_window (which
            # knows nothing about live_speaker — see this module's own
            # docstring on why that stays a WS-router-only concern).
            live_speaker = await run_in_threadpool(live_speaker_tracker.ingest, window.samples, window.sample_rate)
            fused = await run_in_threadpool(engine.score_window, window, context)
            if live_speaker is not None:
                fused = dataclasses.replace(fused, live_speaker=live_speaker)
            await websocket.send_json(FusedScoreOut.from_domain(fused).model_dump(mode="json"))

    except WebSocketDisconnect:
        logger.info("ws_session_closed", session_id=session_id)
    finally:
        await live_transcription.aclose()

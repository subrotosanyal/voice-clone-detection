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
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.schemas import FusedScoreOut, StreamChunkIn
from app.domain.models import AudioWindow
from app.logging_setup import get_logger
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
        hop_ms=engine.pipeline.config["windowing"]["hop_ms"],
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
            # same reasoning.
            fused = await run_in_threadpool(engine.score_window, window, context)
            await websocket.send_json(FusedScoreOut.from_domain(fused).model_dump(mode="json"))

    except WebSocketDisconnect:
        logger.info("ws_session_closed", session_id=session_id)
    finally:
        await live_transcription.aclose()

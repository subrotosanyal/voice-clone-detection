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
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.schemas import FusedScoreOut, StreamChunkIn
from app.domain.models import AudioWindow
from app.logging_setup import get_logger

router = APIRouter()
logger = get_logger(component="ws_router")


@router.websocket("/v1/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    engine = websocket.app.state.engine
    logger.info("ws_session_opened", session_id=session_id)

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
            fused = engine.score_window(window, context)
            await websocket.send_json(FusedScoreOut.from_domain(fused).model_dump(mode="json"))

    except WebSocketDisconnect:
        logger.info("ws_session_closed", session_id=session_id)

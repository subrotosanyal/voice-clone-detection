"""REST endpoints — the easiest way to poke this service.

    curl -F "file=@samples/genuine_tone.wav" \\
         -F 'context={"known_number": true}' \\
         http://localhost:8000/v1/score/file | jq

This is the recommended path for debugging and for tests: no WebSocket
client needed, one process, one HTTP call, a full JSON breakdown back.
See docs/testing.md.
"""
from __future__ import annotations

import io
import json
import uuid

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.schemas import CallContext, FusedScoreOut, ScoreFileResponse

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/config")
def get_config(request: Request) -> dict:
    """Returns the exact risk-formula config this running instance loaded —
    the fastest way to check "what formula is actually live right now"
    without reading a file on disk somewhere."""
    return request.app.state.pipeline.config


@router.post("/v1/score/file", response_model=ScoreFileResponse)
async def score_file(
    request: Request,
    file: UploadFile = File(...),
    context: str = Form(default="{}"),
    session_id: str | None = Form(default=None),
) -> ScoreFileResponse:
    request_context = _parse_context(context)
    session_id = session_id or f"file-{uuid.uuid4().hex[:12]}"

    raw_bytes = await file.read()
    try:
        samples, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller verbatim
        raise HTTPException(status_code=400, detail=f"could not decode audio file: {exc}") from exc

    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)  # downmix to mono

    fused_scores = request.app.state.engine.score_call(
        session_id=session_id,
        samples=samples,
        sample_rate=int(sample_rate),
        context=request_context.model_dump(exclude_none=True),
    )

    if not fused_scores:
        raise HTTPException(status_code=422, detail="audio file produced zero windows (empty or too short)")

    trace = [FusedScoreOut.from_domain(f) for f in fused_scores]
    return ScoreFileResponse(
        session_id=session_id,
        window_count=len(trace),
        final=trace[-1],
        trace=trace,
    )


def _parse_context(raw: str) -> CallContext:
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"context is not valid JSON: {exc}") from exc
    return CallContext(**data)

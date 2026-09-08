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
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import CallContext, FusedScoreOut, ScoreFileResponse, SpeakerScoreOut

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
    diarize: bool = Form(default=False),
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
    sample_rate = int(sample_rate)

    context_dict = request_context.model_dump(exclude_none=True)

    # Transcribe once, HERE, rather than letting Engine.score_call() do it
    # implicitly — because when diarize=true this endpoint calls score_call
    # again once per detected speaker (see _diarize_and_score below), and
    # score_call's own transcription guard only skips re-transcribing when
    # the context it's given already has a `transcript`. Without this, a
    # call with N detected speakers ran Whisper N+1 times (once for the
    # whole file, then once more per speaker from a context that didn't
    # carry the transcript forward) — a real bug that made diarized calls
    # with several speakers take dramatically longer than a plain score,
    # compounded further whenever the diarizer over-counted speakers (see
    # embedding_cluster_diarizer.py's CALIBRATION HISTORY note). Populating
    # it once here means every score_call below — whole-call and every
    # per-speaker one — shares the same transcript instead of each
    # re-running ASR from scratch.
    transcriber = request.app.state.engine.transcriber
    if transcriber is not None and not context_dict.get("transcript"):
        transcript_result = await run_in_threadpool(transcriber.transcribe, samples, sample_rate)
        if transcript_result.text:
            context_dict["transcript"] = transcript_result.text

    # Scoring a whole file runs every 2s window through AASIST + Parselmouth
    # + (maybe) ECAPA-TDNN, in a plain synchronous call — genuinely CPU-
    # bound, and can take a while for a long recording. Running it directly
    # in this async handler would block uvicorn's single event loop for
    # that whole time, starving every other concurrent request (even a
    # trivial GET /healthz) until it finishes. run_in_threadpool moves it
    # to a worker thread so the event loop stays free to serve other
    # requests/WebSocket sessions while this one is still crunching.
    fused_scores = await run_in_threadpool(
        request.app.state.engine.score_call,
        session_id=session_id,
        samples=samples,
        sample_rate=sample_rate,
        context=context_dict,
    )

    if not fused_scores:
        raise HTTPException(status_code=422, detail="audio file produced zero windows (empty or too short)")

    trace = [FusedScoreOut.from_domain(f) for f in fused_scores]

    speakers: list[SpeakerScoreOut] | None = None
    if diarize:
        speakers = await _diarize_and_score(
            request=request,
            base_session_id=session_id,
            samples=samples,
            sample_rate=sample_rate,
            context=context_dict,
        )

    return ScoreFileResponse(
        session_id=session_id,
        window_count=len(trace),
        final=trace[-1],
        trace=trace,
        speakers=speakers,
    )


async def _diarize_and_score(
    request: Request,
    base_session_id: str,
    samples: np.ndarray,
    sample_rate: int,
    context: dict,
) -> list[SpeakerScoreOut]:
    """Splits `samples` by speaker (see app/ports/diarizer.py) and reruns
    each speaker's audio through the SAME Engine.score_call() path used for
    the whole-call result above — a derived session_id
    ("{base}::{speaker_label}") means every window still flows through the
    normal fusion + history-save machinery with no special-casing there.

    Both the diarizer itself (embedding extraction + clustering) and each
    speaker's score_call() are CPU-bound — run_in_threadpool for the same
    event-loop-blocking reason as score_file() above.
    """
    diarizer = getattr(request.app.state.pipeline, "diarizer", None)
    if diarizer is None:
        raise HTTPException(
            status_code=503,
            detail="diarization is not configured on this instance (no `diarization:` section in risk_formula.yaml)",
        )

    segments = await run_in_threadpool(diarizer.diarize, samples, sample_rate)
    if not segments:
        return []

    # Group segments by speaker_label, preserving chronological order of
    # first appearance, and concatenate each speaker's audio.
    speaker_order: list[str] = []
    speaker_chunks: dict[str, list[np.ndarray]] = {}
    speaker_duration_ms: dict[str, int] = {}
    for seg in segments:
        if seg.speaker_label not in speaker_chunks:
            speaker_chunks[seg.speaker_label] = []
            speaker_duration_ms[seg.speaker_label] = 0
            speaker_order.append(seg.speaker_label)
        start_sample = int(seg.start_ms / 1000 * sample_rate)
        end_sample = int(seg.end_ms / 1000 * sample_rate)
        speaker_chunks[seg.speaker_label].append(samples[start_sample:end_sample])
        speaker_duration_ms[seg.speaker_label] += seg.end_ms - seg.start_ms

    results: list[SpeakerScoreOut] = []
    for speaker_label in speaker_order:
        speaker_samples = np.concatenate(speaker_chunks[speaker_label])
        speaker_session_id = f"{base_session_id}::{speaker_label}"
        speaker_fused = await run_in_threadpool(
            request.app.state.engine.score_call,
            session_id=speaker_session_id,
            samples=speaker_samples,
            sample_rate=sample_rate,
            context=context,
        )
        if not speaker_fused:
            continue  # this speaker's total voiced audio was too short to window
        speaker_trace = [FusedScoreOut.from_domain(f) for f in speaker_fused]
        results.append(
            SpeakerScoreOut(
                speaker_label=speaker_label,
                session_id=speaker_session_id,
                segment_count=len(speaker_chunks[speaker_label]),
                total_duration_ms=speaker_duration_ms[speaker_label],
                final=speaker_trace[-1],
                trace=speaker_trace,
            )
        )
    return results


def _parse_context(raw: str) -> CallContext:
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"context is not valid JSON: {exc}") from exc
    return CallContext(**data)

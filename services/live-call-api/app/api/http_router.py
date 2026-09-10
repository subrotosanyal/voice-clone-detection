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
from app.domain.models import SpeakerCallResult

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
    #
    # REAL BUG found 2026-09-10: `if transcript_result.text:` guarding the
    # write below meant an EMPTY transcript (real speech genuinely wasn't
    # found — e.g. non-speech/synthetic audio, or silence) never set the
    # `transcript` key at all, so it looked identical to "never attempted"
    # to every downstream guard checking it — this method's own re-check,
    # PLUS engine.py's score_call() re-checking the SAME thing for the
    # whole-call score AND for every per-speaker score_call() under
    # diarize=true. A 2-speaker call with an empty transcript ran
    # transcription 1 (here) + 1 (whole-call) + 2 (one per speaker) = 4
    # times, not once (see test_diarized_multi_speaker_call_transcribes_
    # only_once). Always recording the attempt — context_dict["transcript"]
    # = transcript_result.text, even "" — fixes it; transcript_source/
    # transcript_language still only get set when text is non-empty, so
    # FusedScore.transcript_source/transcript_language stay correctly None
    # when no speech was found (see that dataclass's own documented
    # contract, unchanged).
    transcriber = request.app.state.engine.transcriber
    if transcriber is not None and "transcript" not in context_dict:
        transcript_result = await run_in_threadpool(transcriber.transcribe, samples, sample_rate)
        context_dict["transcript"] = transcript_result.text

    # Same one-time-per-call reasoning as transcription above: this is
    # enabled by default (config/risk_formula.yaml's `intent` detector —
    # see app/adapters/intent/zero_shot_intent_classifier.py's HONESTY
    # NOTE for the known calibration caveat that comes with it), and must
    # run once here, not once per speaker in the diarization fan-out below.
    intent_classifier = request.app.state.engine.intent_classifier
    if intent_classifier is not None and context_dict.get("transcript") and "intent_label" not in context_dict:
        intent_result = await run_in_threadpool(intent_classifier.classify, context_dict["transcript"])
        context_dict["intent_label"] = intent_result.top_label
        context_dict["intent_top_score"] = intent_result.top_score
        context_dict["intent_label_scores"] = intent_result.label_scores

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
    """Thin wrapper around Engine.diarize_and_score() (app/pipeline/
    engine.py) — that method used to be duplicated here; moved 2026-09-10
    so the live WebSocket path's "diarize on hangup" (app/api/ws_router.py)
    can share the exact same diarize-then-score-each-speaker logic instead
    of a second copy of it. This function's only remaining job is the
    HTTP-layer concerns: the 503 when diarization isn't configured, the
    threadpool offload, and mapping the engine's domain-level
    SpeakerCallResult onto this router's own SpeakerScoreOut schema.
    """
    diarizer = getattr(request.app.state.pipeline, "diarizer", None)
    if diarizer is None:
        raise HTTPException(
            status_code=503,
            detail="diarization is not configured on this instance (no `diarization:` section in risk_formula.yaml)",
        )

    speaker_results = await run_in_threadpool(
        request.app.state.engine.diarize_and_score,
        diarizer=diarizer,
        base_session_id=base_session_id,
        samples=samples,
        sample_rate=sample_rate,
        context=context,
    )
    return [_speaker_result_to_out(r) for r in speaker_results]


def _speaker_result_to_out(result: SpeakerCallResult) -> SpeakerScoreOut:
    trace = [FusedScoreOut.from_domain(f) for f in result.fused_scores]
    return SpeakerScoreOut(
        speaker_label=result.speaker_label,
        session_id=result.session_id,
        segment_count=result.segment_count,
        total_duration_ms=result.total_duration_ms,
        final=trace[-1],
        trace=trace,
    )


def _parse_context(raw: str) -> CallContext:
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"context is not valid JSON: {exc}") from exc
    return CallContext(**data)

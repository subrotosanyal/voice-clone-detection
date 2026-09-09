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
isn't. Its result is attached directly onto the outgoing FusedScore as
`live_speaker`, informational by itself — see "real-time per-speaker
fusion" below for where it DOES now feed a score.

Real-time per-speaker fusion (added 2026-09-10): once LiveSpeakerTracker
has detected a SECOND distinct speaker in this call, every window is
ALSO scored under a derived session_id ("{session_id}::{speaker_label}"),
attached as `speaker_score` on the outgoing FusedScoreOut. Deliberately
gated on speaker_count >= 2 — this doubles CPU cost (every acoustic
detector runs twice per window), a real trade-off only worth paying once
there's genuinely more than one voice to separate. HONEST LIMITATION:
`window.samples` for the per-speaker call is the SAME physical audio as
the whole-call score (LiveSpeakerTracker attributes the WHOLE window to
one dominant speaker, it doesn't separate the audio itself) — so a given
window's RAW acoustic score is identical between the whole-call and
per-speaker views. What genuinely differs is the SMOOTHED trajectory:
each speaker's derived session has its OWN independent EMA history (see
Engine.sessions / SessionStore), so one caller's calm baseline isn't
dragged down by a co-present agitated one, and vice versa — exactly the
gap live_diarization.py's own docstring named as "a future improvement,
not attempted [there]". Text-derived signals (third_signal's contextual
mode, intent, semantic_risk) are shared/call-wide either way, since
transcription runs on the whole mixed audio, not per-speaker-isolated
audio — only the acoustic-family detectors (acoustic, prosodic,
perth_watermark, phase_incoherence) meaningfully diverge per speaker.

Diarize on hangup (added 2026-09-10): a LiveSessionRecorder, ALSO scoped
to this one connection, reconstructs the call's whole, non-overlapping
audio — see app/pipeline/live_session_recorder.py for how. Once the
WebSocket closes, if a `diarization:` section is configured, that full
recording is run through the EXACT SAME Engine.diarize_and_score() the
file-upload path's POST /v1/score/file?diarize=true uses — a more
accurate, hindsight-benefiting per-speaker breakdown than the real-time
per-speaker fusion above (that one attributes on limited, causal,
per-window evidence only; this one gets the whole call and the file
diarizer's own pause-aware segmentation). Retrievable afterward via
GET /v1/sessions/{session_id}/speakers — NOT sent back over the socket
(it's already closed by the time this finishes; a client that wants it
immediately would need to send a graceful "done" message and wait before
actually closing its end, not implemented here). Best-effort: any failure
here is logged, never raised — a live call's real-time scoring must never
be retroactively "broken" by a hangup-time failure.
"""
from __future__ import annotations

import asyncio
import dataclasses

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.schemas import FusedScoreOut, StreamChunkIn
from app.domain.models import AudioWindow
from app.logging_setup import get_logger
from app.pipeline.live_diarization import LiveSpeakerTracker
from app.pipeline.live_session_recorder import LiveSessionRecorder
from app.pipeline.live_transcription import LiveTranscriptionBuffer

router = APIRouter()
logger = get_logger(component="ws_router")


@router.websocket("/v1/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    engine = websocket.app.state.engine
    logger.info("ws_session_opened", session_id=session_id)

    hop_ms = engine.pipeline.config["windowing"]["hop_ms"]
    live_transcription = LiveTranscriptionBuffer(
        transcriber=engine.transcriber,
        intent_classifier=engine.intent_classifier,
        semantic_risk_classifier=engine.semantic_risk_classifier,
        hop_ms=hop_ms,
    )
    live_diarization_params = engine.pipeline.config.get("live_diarization", {}).get("params", {})
    live_speaker_tracker = LiveSpeakerTracker(
        embedder=engine.live_speaker_embedder,
        floor_rms=live_diarization_params.get("floor_rms", 1e-4),
        similarity_threshold=live_diarization_params.get("similarity_threshold", 0.75),
        ema_alpha=live_diarization_params.get("ema_alpha", 0.1),
        max_speakers=live_diarization_params.get("max_speakers", 8),
    )
    # See this module's own "diarize on hangup" docstring note — always
    # recorded (cheap: just buffering), only ever USED at disconnect, and
    # only if a `diarization:` section is actually configured.
    session_recorder = LiveSessionRecorder(hop_ms=hop_ms)

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

            live_transcription.ingest(window.samples, window.sample_rate, window.window_start_ms)
            session_recorder.ingest(window.samples, window.sample_rate, window.window_start_ms)
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

            speaker_score_out = None
            if live_speaker is not None and live_speaker.speaker_count >= 2:
                # See this module's own "real-time per-speaker fusion"
                # docstring note for the honest scope/cost trade-off.
                speaker_window = dataclasses.replace(
                    window, session_id=f"{session_id}::{live_speaker.speaker_label}"
                )
                speaker_fused = await run_in_threadpool(engine.score_window, speaker_window, context)
                speaker_score_out = FusedScoreOut.from_domain(speaker_fused)

            await websocket.send_json(
                FusedScoreOut.from_domain(fused, speaker_score=speaker_score_out).model_dump(mode="json")
            )

    except WebSocketDisconnect:
        logger.info("ws_session_closed", session_id=session_id)
    finally:
        # Capture the best-known transcript-derived context BEFORE
        # aclose() (which cancels any in-flight transcription/intent/
        # semantic-risk task, but does not clear already-completed
        # results out of latest_context).
        final_context = _build_hangup_context(live_transcription.latest_context)
        await live_transcription.aclose()
        # REAL BUG found while testing this feature: awaiting
        # _diarize_on_hangup() directly here got silently CancelledError'd
        # partway through (confirmed via the ASGI framework's own cancel
        # scope, not this project's code) — the moment the WebSocket's
        # underlying connection is fully closed, Starlette/anyio tears
        # down the request's cancel scope, killing anything still running
        # in it, including work in this `finally` block, REGARDLESS of
        # whether it's "cleanup" in intent. A likely real production risk
        # too, not just a test artifact — the ASGI server has no way to
        # know this background work should outlive the connection unless
        # it's genuinely detached from that scope. Fixed by scheduling it
        # as an independent asyncio task instead of awaiting it inline —
        # see _schedule_diarize_on_hangup() below for how it's kept alive
        # (an unreferenced task can be silently garbage-collected).
        _schedule_diarize_on_hangup(engine, session_id, session_recorder, final_context)


# Kept alive here so asyncio doesn't garbage-collect an in-flight
# "fire and forget" task with no other referent — see the REAL BUG note
# above for why this can't just be awaited inline in the WS handler.
_pending_hangup_diarizations: set = set()


def _schedule_diarize_on_hangup(
    engine, session_id: str, session_recorder: "LiveSessionRecorder", context: dict
) -> None:
    task = asyncio.create_task(_diarize_on_hangup(engine, session_id, session_recorder, context))
    _pending_hangup_diarizations.add(task)
    task.add_done_callback(_pending_hangup_diarizations.discard)


def _build_hangup_context(latest_context: dict) -> dict:
    """Prefers the naive whole-session transcript (see
    live_transcription.py's own FULL-SESSION TRANSCRIPT note) over the
    last rolling ~8s chunk for the hangup-time per-speaker analysis —
    the most complete text available beats the most recent-but-partial
    text for a FINAL breakdown computed once, after the call already
    ended. Deliberately strips any already-computed intent_label/
    semantic_* fields so Engine.score_call() re-runs those classifiers
    fresh against the fuller text, rather than reusing results computed
    against just the last rolling chunk."""
    full_transcript = latest_context.get("full_session_transcript")
    if not full_transcript:
        return dict(latest_context)
    return {"transcript": full_transcript, "transcript_language": latest_context.get("transcript_language")}


async def _diarize_on_hangup(
    engine, session_id: str, session_recorder: LiveSessionRecorder, context: dict
) -> None:
    """See this module's own "diarize on hangup" docstring note. Never
    raises — a failure here must not be allowed to look like the live call
    itself failed, since by this point the call is already over and fully
    scored in real time regardless of what happens next."""
    diarizer = getattr(engine.pipeline, "diarizer", None)
    if diarizer is None:
        return
    full_audio = session_recorder.get_full_audio()
    if full_audio is None or session_recorder.sample_rate is None:
        return
    try:
        speaker_results = await run_in_threadpool(
            engine.diarize_and_score,
            diarizer=diarizer,
            base_session_id=session_id,
            samples=full_audio,
            sample_rate=session_recorder.sample_rate,
            context=context,
        )
        logger.info(
            "live_diarize_on_hangup_complete",
            session_id=session_id,
            speaker_count=len(speaker_results),
        )
    except Exception:  # noqa: BLE001 — best-effort, see this function's own docstring
        logger.exception("live_diarize_on_hangup_failed", session_id=session_id)

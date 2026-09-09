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

REAL BUG found 2026-09-11 (CI): the detached task from the note above
solved the CancelledError problem but introduced a resource-boundedness
one — it's genuinely fire-and-forget, with no cap on how many can run
their (expensive: AASIST + ECAPA + Whisper + Perth, per speaker) real
model inference at once. A GitHub Actions run doing many WS hangups in
quick succession (this project's own live-diarization integration test
shard) piled several of these up concurrently, exhausting the runner's
disk (observed: "Free space left: 91 MB") and ending in a native abort
(SIGABRT, exit 134) roughly 3 minutes after pytest's own visible tests
had already finished and reported results — the background tasks were
still running behind pytest's back. Two fixes, both below:
`_HANGUP_DIARIZE_SEMAPHORE` bounds how many run their heavy work at
once (real concurrency control, not just a timeout — a Python thread
handed to run_in_threadpool can't be forcibly cancelled once started,
so a timeout alone only stops US from waiting on it, not the thread
itself from continuing to consume resources); `drain_pending_hangup_
diarizations()` is awaited (bounded) by app/main.py's lifespan shutdown
so tasks from one test's WS session are actually finished (or given up
on) before the NEXT test's `with TestClient(app)` block starts, instead
of silently continuing to compete for CPU/disk with it.
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
        # Prefer the live-only transcriber (config/risk_formula.yaml's
        # `live_transcription:` section, see app/adapters/registry.py's
        # _build_live_transcriber) — see
        # app/adapters/transcription/whisperlive_transcriber.py's own
        # docstring for why the live path specifically needed its
        # transcriber moved out of this process (CPU contention with the
        # per-window detectors below). Falls back to the shared
        # `transcription:` transcriber when `live_transcription:` is
        # omitted (or the pipeline was built before this existed), same
        # "degrade, don't disable" shape as every other optional
        # feature's fallback in this codebase.
        transcriber=engine.live_transcriber or engine.transcriber,
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

# See this module's 2026-09-11 REAL BUG note above. A conservative
# default, not a measured-optimal one — each permit covers one full
# diarize_and_score() call (AASIST + ECAPA + Whisper + Perth per
# speaker), so even 2 concurrent ones is real CPU/memory/disk pressure;
# rebalance only with real production concurrency numbers in hand.
_HANGUP_DIARIZE_MAX_CONCURRENT = 2
_hangup_diarize_semaphore = asyncio.Semaphore(_HANGUP_DIARIZE_MAX_CONCURRENT)

# Bounds how long a single hangup diarization is WAITED on before its
# semaphore permit is released to unblock others queued behind it — see
# the REAL BUG note above for why this can't actually stop the
# underlying thread, only stop us from holding a permit for it forever.
_HANGUP_DIARIZE_TIMEOUT_S = 120.0

# How long app shutdown (app/main.py's lifespan) waits for in-flight
# hangup diarizations to finish before giving up and returning anyway —
# see drain_pending_hangup_diarizations() below.
_HANGUP_DRAIN_TIMEOUT_S = 30.0


def _schedule_diarize_on_hangup(
    engine, session_id: str, session_recorder: "LiveSessionRecorder", context: dict
) -> None:
    task = asyncio.create_task(_diarize_on_hangup(engine, session_id, session_recorder, context))
    _pending_hangup_diarizations.add(task)
    task.add_done_callback(_pending_hangup_diarizations.discard)


async def drain_pending_hangup_diarizations(timeout_s: float = _HANGUP_DRAIN_TIMEOUT_S) -> None:
    """Awaited by app/main.py's lifespan on shutdown. Bounded, not
    indefinite: a genuinely stuck task must not hang shutdown forever
    (a real production concern, not just a test one) — any task still
    running past the timeout is left to finish on its own (its thread
    can't be forcibly cancelled anyway, see this module's own REAL BUG
    note) and just logged, not waited on further. Without this, these
    fire-and-forget tasks silently outlive the "app" they were scheduled
    under — in this project's own integration test suite, where many
    `with TestClient(app) as client:` blocks reuse the SAME process
    (and therefore the same module-level state here) across tests, that
    let hangup work from an EARLIER test keep running concurrently with
    a LATER one, competing for the same limited CI runner resources."""
    if not _pending_hangup_diarizations:
        return
    pending = list(_pending_hangup_diarizations)
    _, still_pending = await asyncio.wait(pending, timeout=timeout_s)
    if still_pending:
        logger.warning(
            "hangup_diarizations_still_pending_at_shutdown",
            count=len(still_pending),
            timeout_s=timeout_s,
        )


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
    # See this module's 2026-09-11 REAL BUG note: bounds how many of
    # these run their real model inference at once, regardless of how
    # many WS sessions hang up close together.
    async with _hangup_diarize_semaphore:
        try:
            speaker_results = await asyncio.wait_for(
                run_in_threadpool(
                    engine.diarize_and_score,
                    diarizer=diarizer,
                    base_session_id=session_id,
                    samples=full_audio,
                    sample_rate=session_recorder.sample_rate,
                    context=context,
                ),
                timeout=_HANGUP_DIARIZE_TIMEOUT_S,
            )
            logger.info(
                "live_diarize_on_hangup_complete",
                session_id=session_id,
                speaker_count=len(speaker_results),
            )
        except TimeoutError:
            # The permit is released here even though the underlying
            # threadpool thread may still be running to completion in
            # the background — Python threads can't be forcibly killed.
            # This only bounds how long OTHER queued hangups wait on us.
            logger.error(
                "live_diarize_on_hangup_timed_out",
                session_id=session_id,
                timeout_s=_HANGUP_DIARIZE_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 — best-effort, see this function's own docstring
            logger.exception("live_diarize_on_hangup_failed", session_id=session_id)

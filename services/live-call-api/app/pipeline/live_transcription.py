"""LiveTranscriptionBuffer — periodic, off-hot-path transcription for the
live WebSocket path.

SCOPE: feeds the exact same thing Engine.score_call()'s one-shot
transcription does — app/adapters/detectors/contextual_rules.py's
urgency/financial/authority-claim keyword detection, and (if configured)
the intent classifier — just periodic instead of once, because a live
call has no natural "whole buffer" to transcribe until it ends.

HOW: app/api/ws_router.py calls ingest() once per incoming window. Each
call appends only the TRAILING new-audio slice of that window to a
rolling buffer (a window overlaps the previous one by construction — see
app/ui/app.js's WINDOW_MS/HOP_MS — so only the newest slice is actually
new audio). Every _TRANSCRIBE_EVERY_MS of new audio, fires a
transcription of the trailing _TRANSCRIPTION_WINDOW_MS of that buffer in
a background asyncio task (via run_in_threadpool, so it never blocks the
per-window score response ws_router.py awaits). The latest COMPLETED
transcript (and intent classification, if configured) is exposed via
`latest_context` for the caller to merge into each subsequent window's
context — never blocking on an in-flight transcription.

REAL BUG found and fixed 2026-09-10 (live-mic parity review): `ingest()`
used to always assume exactly `hop_ms` of new audio arrived per call —
true only if the client's actual send cadence exactly matches
config/risk_formula.yaml's windowing.hop_ms, an assumption this file's own
former HONESTY NOTE admitted could silently stretch/compress the
reconstructed timeline if violated (a slow client tick, a dropped/retried
WS frame, a browser tab backgrounded and throttled). Fixed by deriving
the ACTUAL new-audio duration from consecutive windows' own
`window_start_ms` field — data every chunk already carries (see
StreamChunkIn in app/api/schemas.py), so no new client-server handshake
was needed. `hop_ms` is now only a fallback for the very first window
(nothing to diff against yet) and a sanity-clamp ceiling (see
_MAX_PLAUSIBLE_GAP_MULTIPLE below) — not the primary source of truth.
Backward compatible: `window_start_ms` is optional; omitting it (as every
existing unit test does) reproduces the exact old fixed-hop behaviour.

REAL BUG found and fixed 2026-09-08 (user report: "I see it for the
first sentence, then nothing"): intent classification used to run
INSIDE the same task as transcription, sequentially after it, before
the next transcription cycle was allowed to start (the `already_running`
guard covered both steps as one unit). Measured by hand inside the
actual deployed container (not the faster local dev machine): Whisper
"small" takes ~3-5s for a realistic window, and the zero-shot intent
classifier takes a further ~5s on top of that — combined, ~8-10s per
cycle. A short live-mic session (10-20s) would complete exactly ONE
cycle and never start a second before the user stopped, reading as
"transcription stopped working" when it was actually just slower than
the session was long. Fixed by decoupling intent classification into
its OWN independently-scheduled task (`_intent_task`, separate from
`_transcribe_task`): a slow intent classification no longer blocks the
next transcription cycle from starting once ITS OWN new-audio threshold
is met. Also reduced _TRANSCRIPTION_WINDOW_MS (12s -> 8s) for a further,
real (measured) reduction in per-cycle Whisper latency.

FULL-SESSION TRANSCRIPT (added 2026-09-10): `latest_context["transcript"]`
above is deliberately a ROLLING trailing window (~8s) — the right shape
for scoring, since intent/semantic_risk/contextual keyword detection need
current not historical text. But that means anything said more than
~8-15s ago is invisible to those detectors on the live path (e.g. an
authority claim made 20s ago plus a financial request just now wouldn't
be seen as connected the way a file-upload's whole-call transcript would
catch it — see #2 Option B in the live-mic-parity plan for the fix to
THAT specific problem, which is separate from this). This class also now
accumulates every completed cycle's text into a naive, WHOLE-SESSION
`latest_context["full_session_transcript"]`, exposed for DISPLAY/audit
purposes on the dashboard — an operator glancing at a long call should be
able to see everything said, not just the last few seconds. HONEST
LIMITATION: because consecutive transcription cycles overlap (this class
re-transcribes a TRAILING window every cycle, not a disjoint one), naive
concatenation can repeat a phrase that straddled two cycles' boundaries.
Deliberately not de-duplicated — doing that reliably would need aligning
two independent Whisper decodes of overlapping-but-not-identical audio,
which isn't attempted here. This field is NEVER read by any detector
(scoring still uses only the rolling `transcript` above) — display only.

HONESTY NOTE: this is deliberately never real-time-exact. A transcript
lags real speech by up to _TRANSCRIBE_EVERY_MS plus however long Whisper
actually takes to run — measured inside the real deployed container
(not just a faster local dev machine): roughly 3-5s for an 8-12s window,
notably slower than earlier optimistic local-only measurements suggested
this HONESTY NOTE will keep, deliberately, so a future container-
performance regression is easy to compare against). The keyword/intent
signal is a further-delayed signal on top of that (its own ~5s, no longer
blocking transcription but still real, sequential latency for intent
specifically), and the very first _TRANSCRIBE_EVERY_MS of any call has no
transcript at all yet. The former hop-mismatch timeline risk (see the REAL
BUG note above) is fixed, not just documented, as of 2026-09-10.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np
from fastapi.concurrency import run_in_threadpool

from app.logging_setup import get_logger
from app.ports.intent_classifier import IntentClassifierPort
from app.ports.semantic_risk_classifier import SemanticRiskClassifierPort
from app.ports.transcriber import TranscriberPort

logger = get_logger(component="live_transcription")

_TRANSCRIBE_EVERY_MS = 4000  # re-transcribe once this much NEW audio has arrived
_TRANSCRIPTION_WINDOW_MS = 8000  # trailing window actually sent to Whisper (was 12000 — real, measured latency cut)
_MAX_BUFFER_MS = 15000  # hard cap so a long call's buffer doesn't grow unbounded
# Safety clamp for the window_start_ms-derived elapsed time (see this
# module's REAL BUG note, 2026-09-10): a client hiccup (a backgrounded
# tab, a reconnect) could otherwise report a huge single jump, dumping
# an implausible amount of "new" audio into the buffer at once. Capped
# at a generous multiple of a normal hop rather than hop_ms itself,
# since hop_ms varies by config.
_MAX_PLAUSIBLE_GAP_MULTIPLE = 10


class LiveTranscriptionBuffer:
    """One instance per WebSocket session (not shared, not safe to reuse
    across sessions) — mirrors SessionStore's "one instance per session_id"
    shape in app/pipeline/engine.py, just held by the caller instead of a
    dict keyed by session_id, since a WS connection's lifetime already
    matches one session's lifetime one-to-one."""

    def __init__(
        self,
        transcriber: Optional[TranscriberPort],
        intent_classifier: Optional[IntentClassifierPort],
        hop_ms: int,
        semantic_risk_classifier: Optional[SemanticRiskClassifierPort] = None,
    ) -> None:
        self._transcriber = transcriber
        self._intent_classifier = intent_classifier
        self._semantic_risk_classifier = semantic_risk_classifier
        self._hop_ms = hop_ms
        self._buffer: list[np.ndarray] = []
        self._buffer_ms = 0
        self._new_ms_since_transcription = 0
        # THREE independent tasks, not one — see the REAL BUG note above
        # the class docstring for why: intent classification (~5s
        # measured inside the real container) used to run inside the
        # same task as transcription, so a slow intent classification
        # blocked the next transcription cycle from starting. Same
        # reasoning extends to semantic risk classification (~1-1.5s CPU,
        # see local_llm_semantic_classifier.py) — each has its own
        # already-running guard, so a slow one never blocks the others.
        self._transcribe_task: Optional[asyncio.Task] = None
        self._intent_task: Optional[asyncio.Task] = None
        self._semantic_task: Optional[asyncio.Task] = None
        self._last_window_start_ms: Optional[int] = None
        # Naive whole-session concatenation, display-only — see this
        # class's own FULL-SESSION TRANSCRIPT docstring note for the
        # honest overlap-duplication caveat. Never read by a detector.
        self._full_transcript_parts: list[str] = []
        self.latest_context: dict[str, Any] = {}

    def ingest(self, samples: np.ndarray, sample_rate: int, window_start_ms: Optional[int] = None) -> None:
        """Call once per incoming window. No-ops (does nothing but still
        safe to call) when no transcriber is configured.

        `window_start_ms` (optional, backward compatible): when given, the
        actual new-audio duration is derived from the delta against the
        PREVIOUS call's `window_start_ms`, rather than assumed to always
        equal `hop_ms` — see this module's REAL BUG note (2026-09-10).
        Omitting it (every existing caller before this fix) reproduces the
        old fixed-hop behaviour exactly.
        """
        if self._transcriber is None:
            return

        elapsed_ms = self._hop_ms
        if window_start_ms is not None:
            if self._last_window_start_ms is not None:
                delta = window_start_ms - self._last_window_start_ms
                # Clamp: never negative (an out-of-order or duplicate
                # window_start_ms), never an implausible single jump (see
                # _MAX_PLAUSIBLE_GAP_MULTIPLE above).
                elapsed_ms = max(0, min(delta, self._hop_ms * _MAX_PLAUSIBLE_GAP_MULTIPLE))
            self._last_window_start_ms = window_start_ms

        if elapsed_ms <= 0:
            return

        new_samples = round(sample_rate * elapsed_ms / 1000)
        new_tail = samples[-new_samples:] if samples.size > new_samples else samples
        if new_tail.size == 0:
            return
        self._buffer.append(new_tail)
        self._buffer_ms += elapsed_ms
        self._new_ms_since_transcription += elapsed_ms

        while self._buffer_ms > _MAX_BUFFER_MS and len(self._buffer) > 1:
            dropped = self._buffer.pop(0)
            self._buffer_ms -= round(dropped.size / sample_rate * 1000)

        already_running = self._transcribe_task is not None and not self._transcribe_task.done()
        if self._new_ms_since_transcription >= _TRANSCRIBE_EVERY_MS and not already_running:
            self._new_ms_since_transcription = 0
            window_samples = np.concatenate(self._buffer)
            trailing_samples = round(sample_rate * _TRANSCRIPTION_WINDOW_MS / 1000)
            if window_samples.size > trailing_samples:
                window_samples = window_samples[-trailing_samples:]
            self._transcribe_task = asyncio.create_task(self._transcribe(window_samples, sample_rate))

    async def _transcribe(self, samples: np.ndarray, sample_rate: int) -> None:
        try:
            result = await run_in_threadpool(self._transcriber.transcribe, samples, sample_rate)
        except Exception:  # noqa: BLE001 — best-effort, must never break live scoring
            logger.exception("live_transcription_failed")
            return
        if not result.text:
            return

        logger.info(
            "live_transcript_computed",
            detector_name=result.detector_name,
            language=result.language,
            transcript_length=len(result.text),
        )
        self._full_transcript_parts.append(result.text)

        # MERGE, not replace — intent's fields (if any) complete
        # independently and shouldn't be wiped out by a transcript-only
        # update, or vice versa.
        self.latest_context = {
            **self.latest_context,
            "transcript": result.text,
            "transcript_language": result.language,
            "full_session_transcript": " ".join(self._full_transcript_parts),
        }

        if self._intent_classifier is not None:
            intent_already_running = self._intent_task is not None and not self._intent_task.done()
            if not intent_already_running:
                self._intent_task = asyncio.create_task(self._classify_intent(result.text))

        if self._semantic_risk_classifier is not None:
            semantic_already_running = self._semantic_task is not None and not self._semantic_task.done()
            if not semantic_already_running:
                self._semantic_task = asyncio.create_task(self._analyze_semantic_risk(result.text))

    async def _classify_intent(self, text: str) -> None:
        try:
            intent_result = await run_in_threadpool(self._intent_classifier.classify, text)
        except Exception:  # noqa: BLE001 — best-effort, must never break live scoring
            logger.exception("live_intent_classification_failed")
            return

        logger.info(
            "live_intent_classified",
            detector_name=intent_result.detector_name,
            top_label=intent_result.top_label,
            top_score=intent_result.top_score,
        )
        self.latest_context = {
            **self.latest_context,
            "intent_label": intent_result.top_label,
            "intent_top_score": intent_result.top_score,
            "intent_label_scores": intent_result.label_scores,
        }

    async def _analyze_semantic_risk(self, text: str) -> None:
        try:
            semantic_result = await run_in_threadpool(self._semantic_risk_classifier.analyze, text)
        except Exception:  # noqa: BLE001 — best-effort, must never break live scoring
            logger.exception("live_semantic_risk_analysis_failed")
            return

        logger.info(
            "live_semantic_risk_analyzed",
            detector_name=semantic_result.detector_name,
            urgency_level=semantic_result.urgency_level,
            financial_solicitation=semantic_result.financial_solicitation,
            authority_claim=semantic_result.authority_claim,
            isolation_request=semantic_result.isolation_request,
        )
        self.latest_context = {
            **self.latest_context,
            "semantic_urgency_level": semantic_result.urgency_level,
            "semantic_financial_solicitation": semantic_result.financial_solicitation,
            "semantic_authority_claim": semantic_result.authority_claim,
            "semantic_isolation_request": semantic_result.isolation_request,
            "semantic_reasoning": semantic_result.reasoning,
        }

    async def aclose(self) -> None:
        """Call when the WebSocket session ends, so an in-flight
        transcription/intent classification/semantic analysis doesn't
        keep running (and logging) after the client that would have used
        its result is already gone."""
        for task in (self._transcribe_task, self._intent_task, self._semantic_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

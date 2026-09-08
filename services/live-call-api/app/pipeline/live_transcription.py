"""LiveTranscriptionBuffer — periodic, off-hot-path transcription for the
live WebSocket path.

SCOPE: feeds the exact same thing Engine.score_call()'s one-shot
transcription does — app/adapters/detectors/contextual_rules.py's
urgency/financial/authority-claim keyword detection, and (if configured)
the intent classifier — just periodic instead of once, because a live
call has no natural "whole buffer" to transcribe until it ends.

HOW: app/api/ws_router.py calls ingest() once per incoming window. Each
call appends only the TRAILING hop-worth of that window to a rolling
buffer (a window overlaps the previous one by construction — see
app/ui/app.js's WINDOW_MS/HOP_MS — so only the newest hop's worth is
actually new audio; this assumes the client's hop matches
config/risk_formula.yaml's windowing.hop_ms, same assumption app.js's own
comment already makes). Every _TRANSCRIBE_EVERY_MS of new audio, fires a
transcription of the trailing _TRANSCRIPTION_WINDOW_MS of that buffer in
a background asyncio task (via run_in_threadpool, so it never blocks the
per-window score response ws_router.py awaits). The latest COMPLETED
transcript (and intent classification, if configured) is exposed via
`latest_context` for the caller to merge into each subsequent window's
context — never blocking on an in-flight transcription.

HONESTY NOTE: this is deliberately never real-time-exact. A transcript
lags real speech by up to _TRANSCRIBE_EVERY_MS plus however long Whisper
actually takes to run (observed: roughly 1-2s on CPU for a several-second
clip) — the keyword/intent signal it feeds is a delayed signal, not an
instant one, and the very first _TRANSCRIBE_EVERY_MS of any call has no
transcript at all yet. If the client's window hop doesn't match
windowing.hop_ms, the reconstructed buffer will be stretched or
compressed relative to real time — a documented limitation, not a silently
wrong one.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np
from fastapi.concurrency import run_in_threadpool

from app.logging_setup import get_logger
from app.ports.intent_classifier import IntentClassifierPort
from app.ports.transcriber import TranscriberPort

logger = get_logger(component="live_transcription")

_TRANSCRIBE_EVERY_MS = 4000  # re-transcribe once this much NEW audio has arrived
_TRANSCRIPTION_WINDOW_MS = 12000  # trailing window actually sent to Whisper
_MAX_BUFFER_MS = 15000  # hard cap so a long call's buffer doesn't grow unbounded


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
    ) -> None:
        self._transcriber = transcriber
        self._intent_classifier = intent_classifier
        self._hop_ms = hop_ms
        self._buffer: list[np.ndarray] = []
        self._buffer_ms = 0
        self._new_ms_since_transcription = 0
        self._task: Optional[asyncio.Task] = None
        self.latest_context: dict[str, Any] = {}

    def ingest(self, samples: np.ndarray, sample_rate: int) -> None:
        """Call once per incoming window. No-ops (does nothing but still
        safe to call) when no transcriber is configured."""
        if self._transcriber is None:
            return

        hop_samples = round(sample_rate * self._hop_ms / 1000)
        new_tail = samples[-hop_samples:] if samples.size > hop_samples else samples
        if new_tail.size == 0:
            return
        self._buffer.append(new_tail)
        self._buffer_ms += self._hop_ms
        self._new_ms_since_transcription += self._hop_ms

        while self._buffer_ms > _MAX_BUFFER_MS and len(self._buffer) > 1:
            dropped = self._buffer.pop(0)
            self._buffer_ms -= round(dropped.size / sample_rate * 1000)

        already_running = self._task is not None and not self._task.done()
        if self._new_ms_since_transcription >= _TRANSCRIBE_EVERY_MS and not already_running:
            self._new_ms_since_transcription = 0
            window_samples = np.concatenate(self._buffer)
            trailing_samples = round(sample_rate * _TRANSCRIPTION_WINDOW_MS / 1000)
            if window_samples.size > trailing_samples:
                window_samples = window_samples[-trailing_samples:]
            self._task = asyncio.create_task(self._transcribe(window_samples, sample_rate))

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
        context_update: dict[str, Any] = {
            "transcript": result.text,
            "transcript_language": result.language,
        }

        if self._intent_classifier is not None:
            try:
                intent_result = await run_in_threadpool(self._intent_classifier.classify, result.text)
                context_update["intent_label"] = intent_result.top_label
                context_update["intent_top_score"] = intent_result.top_score
                context_update["intent_label_scores"] = intent_result.label_scores
                logger.info(
                    "live_intent_classified",
                    detector_name=intent_result.detector_name,
                    top_label=intent_result.top_label,
                    top_score=intent_result.top_score,
                )
            except Exception:  # noqa: BLE001 — same best-effort reasoning as above
                logger.exception("live_intent_classification_failed")

        self.latest_context = context_update

    async def aclose(self) -> None:
        """Call when the WebSocket session ends, so an in-flight
        transcription doesn't keep running (and logging) after the client
        that would have used its result is already gone."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

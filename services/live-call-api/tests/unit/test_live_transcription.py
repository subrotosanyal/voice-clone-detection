"""Tests for LiveTranscriptionBuffer — fake transcriber/intent-classifier
objects (no real Whisper/mDeBERTa load), same fast/offline/deterministic
style as test_intent_risk.py. Async internals (ingest() schedules a
background asyncio.Task) are driven with plain asyncio.run() rather than
a pytest-asyncio dependency — this project avoids new dependencies where
a standard-library approach works just as well.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.domain.models import IntentClassificationResult, TranscriptResult
from app.pipeline.live_transcription import (
    _TRANSCRIBE_EVERY_MS,
    LiveTranscriptionBuffer,
)

SR = 16_000
HOP_MS = 500


class _FakeTranscriber:
    def __init__(self, text: str = "please transfer the money now", language: str = "en") -> None:
        self.text = text
        self.language = language
        self.calls = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        self.calls += 1
        return TranscriptResult(
            text=self.text, language=self.language, detector_name="fake_transcriber", detector_version="0.0"
        )


class _SlowFakeTranscriber(_FakeTranscriber):
    """Blocks (via a real thread-sleep, since transcribe() runs inside
    run_in_threadpool) long enough for a test to reliably observe the
    task as still in-flight."""

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        import time

        time.sleep(0.3)
        return super().transcribe(samples, sample_rate)


class _FakeIntentClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, text: str) -> IntentClassificationResult:
        self.calls += 1
        return IntentClassificationResult(
            text=text,
            top_label="requesting a money transfer or payment",
            top_score=0.9,
            label_scores={"requesting a money transfer or payment": 0.9, "ordinary conversation": 0.1},
            detector_name="fake_intent",
            detector_version="0.0",
        )


def _hop_samples(n_hops: int) -> np.ndarray:
    return np.random.default_rng(0).uniform(-1, 1, size=round(SR * HOP_MS / 1000) * n_hops).astype(np.float32)


def test_ingest_is_a_noop_without_a_transcriber():
    buf = LiveTranscriptionBuffer(transcriber=None, intent_classifier=None, hop_ms=HOP_MS)
    buf.ingest(_hop_samples(20), SR)
    assert buf.latest_context == {}
    assert buf._task is None


def test_transcription_fires_after_enough_new_audio_and_updates_latest_context():
    async def _run():
        transcriber = _FakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._task is not None
        await buf._task
        assert buf.latest_context["transcript"] == transcriber.text
        assert buf.latest_context["transcript_language"] == "en"
        assert transcriber.calls == 1

    asyncio.run(_run())


def test_intent_classifier_is_also_invoked_when_configured():
    async def _run():
        transcriber = _FakeTranscriber()
        intent_classifier = _FakeIntentClassifier()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=intent_classifier, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._task
        assert buf.latest_context["intent_label"] == "requesting a money transfer or payment"
        assert buf.latest_context["intent_top_score"] == pytest.approx(0.9)
        assert intent_classifier.calls == 1

    asyncio.run(_run())


def test_empty_transcript_does_not_update_latest_context():
    async def _run():
        transcriber = _FakeTranscriber(text="")
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._task
        assert buf.latest_context == {}

    asyncio.run(_run())


def test_a_second_transcription_does_not_start_while_one_is_in_flight():
    async def _run():
        transcriber = _SlowFakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        first_task = buf._task
        # feed enough more audio to cross the threshold again immediately —
        # must NOT start a second task while the first is still running
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._task is first_task
        await first_task
        assert transcriber.calls == 1

    asyncio.run(_run())


def test_aclose_cancels_an_in_flight_transcription():
    async def _run():
        transcriber = _SlowFakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._task is not None and not buf._task.done()
        await buf.aclose()  # must not raise
        assert buf._task.cancelled() or buf._task.done()

    asyncio.run(_run())

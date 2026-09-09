"""Tests for LiveTranscriptionBuffer — fake transcriber/intent-classifier
objects (no real Whisper/mDeBERTa load), same fast/offline/deterministic
style as test_intent_risk.py. Async internals (ingest() schedules
background asyncio.Tasks) are driven with plain asyncio.run() rather than
a pytest-asyncio dependency — this project avoids new dependencies where
a standard-library approach works just as well.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.domain.models import IntentClassificationResult, SemanticRiskAssessment, TranscriptResult
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


class _SlowFakeIntentClassifier(_FakeIntentClassifier):
    """Real bug regression fixture: intent classification measured ~5s
    inside the actual deployed container. This stands in for that real
    slowness to prove it no longer blocks the NEXT transcription cycle."""

    def classify(self, text: str) -> IntentClassificationResult:
        import time

        time.sleep(0.3)
        return super().classify(text)


class _FakeSemanticRiskClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, text: str) -> SemanticRiskAssessment:
        self.calls += 1
        return SemanticRiskAssessment(
            text=text,
            urgency_level=0.8,
            financial_solicitation=True,
            authority_claim=False,
            isolation_request=False,
            reasoning="fake reasoning",
            detector_name="fake_semantic_risk",
            detector_version="0.0",
        )


def _hop_samples(n_hops: int) -> np.ndarray:
    return np.random.default_rng(0).uniform(-1, 1, size=round(SR * HOP_MS / 1000) * n_hops).astype(np.float32)


def test_ingest_is_a_noop_without_a_transcriber():
    buf = LiveTranscriptionBuffer(transcriber=None, intent_classifier=None, hop_ms=HOP_MS)
    buf.ingest(_hop_samples(20), SR)
    assert buf.latest_context == {}
    assert buf._transcribe_task is None


def test_transcription_fires_after_enough_new_audio_and_updates_latest_context():
    async def _run():
        transcriber = _FakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._transcribe_task is not None
        await buf._transcribe_task
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
        await buf._transcribe_task
        assert buf._intent_task is not None
        await buf._intent_task
        assert buf.latest_context["intent_label"] == "requesting a money transfer or payment"
        assert buf.latest_context["intent_top_score"] == pytest.approx(0.9)
        assert intent_classifier.calls == 1
        # transcript fields must survive the intent update (merge, not replace)
        assert buf.latest_context["transcript"] == transcriber.text

    asyncio.run(_run())


def test_semantic_risk_classifier_is_also_invoked_when_configured():
    async def _run():
        transcriber = _FakeTranscriber()
        semantic_risk_classifier = _FakeSemanticRiskClassifier()
        buf = LiveTranscriptionBuffer(
            transcriber=transcriber,
            intent_classifier=None,
            hop_ms=HOP_MS,
            semantic_risk_classifier=semantic_risk_classifier,
        )
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._transcribe_task
        assert buf._semantic_task is not None
        await buf._semantic_task
        assert buf.latest_context["semantic_urgency_level"] == pytest.approx(0.8)
        assert buf.latest_context["semantic_financial_solicitation"] is True
        assert semantic_risk_classifier.calls == 1
        # transcript fields must survive the semantic update (merge, not replace)
        assert buf.latest_context["transcript"] == transcriber.text

    asyncio.run(_run())


def test_empty_transcript_does_not_update_latest_context():
    async def _run():
        transcriber = _FakeTranscriber(text="")
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._transcribe_task
        assert buf.latest_context == {}

    asyncio.run(_run())


def test_a_second_transcription_does_not_start_while_one_is_in_flight():
    async def _run():
        transcriber = _SlowFakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        first_task = buf._transcribe_task
        # feed enough more audio to cross the threshold again immediately —
        # must NOT start a second task while the first is still running
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._transcribe_task is first_task
        await first_task
        assert transcriber.calls == 1

    asyncio.run(_run())


def test_slow_intent_classification_does_not_block_the_next_transcription_cycle():
    """Regression test for a real bug found 2026-09-08 (user report: "I
    see it for the first sentence, then nothing"): intent classification
    used to run inside the SAME task as transcription, so a slow intent
    call (measured ~5s inside the real deployed container) blocked the
    next transcription cycle from starting for its whole duration. Proves
    the fix: a second transcription cycle starts and completes while the
    first cycle's intent classification is STILL in flight."""

    async def _run():
        transcriber = _FakeTranscriber()
        intent_classifier = _SlowFakeIntentClassifier()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=intent_classifier, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS

        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._transcribe_task  # first transcription completes, kicks off (slow) intent classification
        assert buf._intent_task is not None and not buf._intent_task.done()

        # Feed enough new audio to cross the threshold again WHILE the
        # first cycle's intent classification is still running.
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        second_transcribe_task = buf._transcribe_task
        assert second_transcribe_task is not None
        assert not buf._intent_task.done()  # intent still in flight — did NOT block this
        await second_transcribe_task
        assert transcriber.calls == 2  # the second cycle really did run, not skipped

        await buf._intent_task  # let the first cycle's intent classification finish too

    asyncio.run(_run())


def test_aclose_cancels_an_in_flight_transcription():
    async def _run():
        transcriber = _SlowFakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._transcribe_task is not None and not buf._transcribe_task.done()
        await buf.aclose()  # must not raise
        assert buf._transcribe_task.cancelled() or buf._transcribe_task.done()

    asyncio.run(_run())

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


def test_window_start_ms_derives_elapsed_time_instead_of_assuming_hop_ms():
    """Regression test for a real bug found 2026-09-10 (live-mic-parity
    review): ingest() used to always assume exactly hop_ms of new audio
    arrived per call, wrong whenever the client's actual send cadence
    doesn't match config hop_ms (a slow tick, a backgrounded tab). Passing
    window_start_ms derives the REAL elapsed time from consecutive deltas
    instead — here, two calls 2x HOP_MS apart should count as 2 hops of
    new audio, not 1."""
    buf = LiveTranscriptionBuffer(transcriber=_FakeTranscriber(), intent_classifier=None, hop_ms=HOP_MS)
    buf.ingest(_hop_samples(1), SR, window_start_ms=0)
    assert buf._buffer_ms == HOP_MS  # first call: no prior reference, falls back to hop_ms
    buf.ingest(_hop_samples(1), SR, window_start_ms=2 * HOP_MS)
    assert buf._buffer_ms == HOP_MS + 2 * HOP_MS


def test_window_start_ms_backward_compatible_when_omitted():
    """Every existing caller (and every other test in this file) omits
    window_start_ms — must reproduce the exact old fixed-hop-per-call
    behaviour, not silently change."""
    buf = LiveTranscriptionBuffer(transcriber=_FakeTranscriber(), intent_classifier=None, hop_ms=HOP_MS)
    buf.ingest(_hop_samples(1), SR)
    buf.ingest(_hop_samples(1), SR)
    assert buf._buffer_ms == 2 * HOP_MS


def test_window_start_ms_clamps_an_implausible_single_jump():
    """A backgrounded tab / reconnect could report a huge single gap —
    must not dump an unbounded amount of "new" audio into the buffer at
    once (see _MAX_PLAUSIBLE_GAP_MULTIPLE). The clamped jump is still
    large enough to cross _TRANSCRIBE_EVERY_MS, so this runs inside an
    event loop like the other cycle-triggering tests."""

    async def _run():
        buf = LiveTranscriptionBuffer(transcriber=_FakeTranscriber(), intent_classifier=None, hop_ms=HOP_MS)
        buf.ingest(_hop_samples(1), SR, window_start_ms=0)
        buf.ingest(_hop_samples(1), SR, window_start_ms=1_000_000)  # a huge, implausible gap
        assert buf._buffer_ms == HOP_MS + HOP_MS * 10  # clamped, not a ~1000s jump
        if buf._transcribe_task is not None:
            await buf._transcribe_task

    asyncio.run(_run())


def test_window_start_ms_clamps_a_negative_or_duplicate_delta_to_zero():
    """An out-of-order or duplicate window_start_ms must not add negative
    "new" audio or corrupt buffer accounting."""
    buf = LiveTranscriptionBuffer(transcriber=_FakeTranscriber(), intent_classifier=None, hop_ms=HOP_MS)
    buf.ingest(_hop_samples(1), SR, window_start_ms=1000)
    buf.ingest(_hop_samples(1), SR, window_start_ms=500)  # goes backwards
    assert buf._buffer_ms == HOP_MS  # the second call contributed nothing


def test_window_start_ms_zero_does_not_silently_discard_the_opening_of_a_call():
    """REAL BUG found and fixed 2026-09-11, via eval/wer_eval.py's own
    word-level comparison against file-upload on real audio: a client's
    own `window_start_ms = max(0, elapsed - WINDOW_MS)` (see app/ui/
    app.js) stays clamped at 0 for the first WINDOW_MS (2s by default) of
    ANY call — so several CONSECUTIVE calls report the identical
    window_start_ms=0, and the delta-based logic used to read that as
    "no new audio arrived" every time, silently dropping real audio.
    Measured real-world impact: ~1.5s missing from the very start of
    every live call, in every language, regardless of length."""
    buf = LiveTranscriptionBuffer(transcriber=_FakeTranscriber(), intent_classifier=None, hop_ms=HOP_MS)
    buf.ingest(_hop_samples(1), SR, window_start_ms=0)  # first call — no prior reference
    buf.ingest(_hop_samples(1), SR, window_start_ms=0)  # still ramping up — must NOT be treated as "no new audio"
    buf.ingest(_hop_samples(1), SR, window_start_ms=0)  # same
    assert buf._buffer_ms == 3 * HOP_MS, "each call during the client's clamped ramp-up period must still count as hop_ms of new audio"


def test_window_start_ms_zero_transitions_cleanly_to_delta_based_accounting():
    """Once real elapsed time exceeds WINDOW_MS, window_start_ms starts
    advancing normally — the buffer accounting must pick back up exactly
    where the ramp-up period left off, not double-count or gap."""
    buf = LiveTranscriptionBuffer(transcriber=_FakeTranscriber(), intent_classifier=None, hop_ms=HOP_MS)
    buf.ingest(_hop_samples(1), SR, window_start_ms=0)
    buf.ingest(_hop_samples(1), SR, window_start_ms=0)
    buf.ingest(_hop_samples(1), SR, window_start_ms=HOP_MS)  # ramp-up over, normal delta-based accounting resumes
    assert buf._buffer_ms == 2 * HOP_MS + HOP_MS  # 2 ramp-up calls + one normal HOP_MS delta


def test_full_session_transcript_accumulates_across_cycles():
    """#2 Option A: a naive whole-session concatenation, for display/audit
    only — proves it grows across cycles rather than only holding the
    latest rolling chunk like `transcript` does."""

    async def _run():
        transcriber = _FakeTranscriber(text="first chunk")
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._transcribe_task
        assert buf.latest_context["full_session_transcript"] == "first chunk"

        transcriber.text = "second chunk"
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._transcribe_task
        assert buf.latest_context["transcript"] == "second chunk"  # rolling: latest only
        assert buf.latest_context["full_session_transcript"] == "first chunk second chunk"  # cumulative

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


def test_flush_transcribes_a_call_shorter_than_transcribe_every_ms():
    """The real gap eval/wer_eval.py's own WER measurement found: a call
    shorter than _TRANSCRIBE_EVERY_MS never crosses ingest()'s own
    threshold, so without flush() it would produce ZERO transcript,
    ever — regardless of how much was actually said."""

    async def _run():
        transcriber = _FakeTranscriber(text="a short call")
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        short_hops = (_TRANSCRIBE_EVERY_MS // HOP_MS) - 1  # deliberately under the normal threshold
        for _ in range(short_hops):
            buf.ingest(_hop_samples(1), SR)
        assert buf._transcribe_task is None, "must not have fired a normal cycle yet — that's the point of this test"
        assert buf.latest_context == {}

        await buf.flush()

        assert buf.latest_context["transcript"] == "a short call"
        assert transcriber.calls == 1

    asyncio.run(_run())


def test_flush_is_a_noop_when_nothing_was_ever_ingested():
    async def _run():
        transcriber = _FakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)

        await buf.flush()  # must not raise

        assert transcriber.calls == 0
        assert buf.latest_context == {}

    asyncio.run(_run())


def test_flush_is_a_noop_without_a_transcriber():
    async def _run():
        buf = LiveTranscriptionBuffer(transcriber=None, intent_classifier=None, hop_ms=HOP_MS)
        buf.ingest(_hop_samples(1), SR)  # no-ops without a transcriber anyway

        await buf.flush()  # must not raise (no _last_sample_rate/_buffer to work with)

    asyncio.run(_run())


def test_flush_does_not_re_transcribe_when_the_normal_cycle_already_covered_everything():
    """If ingest() already fired a cycle and no NEW audio arrived since,
    flush() must not fire a redundant second transcription."""

    async def _run():
        transcriber = _FakeTranscriber()
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._transcribe_task
        assert transcriber.calls == 1

        await buf.flush()

        assert transcriber.calls == 1, "nothing new since the last cycle — flush() must not re-transcribe"

    asyncio.run(_run())


def test_flush_waits_for_an_already_in_flight_transcription_instead_of_starting_a_second_one():
    async def _run():
        transcriber = _SlowFakeTranscriber(text="in flight result")
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._transcribe_task is not None and not buf._transcribe_task.done()

        await buf.flush()  # must wait for the in-flight cycle, not fire a second transcribe() call

        assert transcriber.calls == 1
        assert buf.latest_context["transcript"] == "in flight result"

    asyncio.run(_run())


def test_flush_catches_up_on_a_backlog_that_accumulated_while_a_cycle_was_in_flight():
    """REAL BUG found 2026-09-11 via eval/wer_eval.py's own real WER
    measurement: audio can keep accumulating (ingest() never blocks on
    an in-flight cycle) while a slow transcription is still running.
    flush() awaiting that first cycle and stopping — its original
    behavior — silently dropped whatever piled up in the meantime. A
    real, not simulation-only, risk: a call ending right as a slow
    cycle is still finishing hits this exact case."""

    async def _run():
        transcriber = _SlowFakeTranscriber(text="first cycle result")
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        assert buf._transcribe_task is not None and not buf._transcribe_task.done()

        # More audio arrives WHILE that first cycle is still running — not
        # awaited here, matching eval/wer_eval.py's own simulate_live_path()
        # (and a real call whose transcriber is slower than how fast more
        # audio keeps arriving).
        transcriber.text = "backlog result"
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
            await asyncio.sleep(0)
        assert buf._new_ms_since_transcription > 0, "audio must have accumulated without starting a second cycle"

        await buf.flush()

        assert transcriber.calls == 2, "flush() must fire a SECOND cycle for the backlog, not just await the first"
        assert buf.latest_context["transcript"] == "backlog result"

    asyncio.run(_run())


def test_flush_captures_trailing_audio_after_the_last_completed_cycle():
    """The other half of the real gap: even a call that DID cross the
    normal threshold at least once still loses whatever was said between
    the last completed cycle and the call ending, without a flush."""

    async def _run():
        transcriber = _FakeTranscriber(text="first cycle")
        buf = LiveTranscriptionBuffer(transcriber=transcriber, intent_classifier=None, hop_ms=HOP_MS)
        hops_needed = _TRANSCRIBE_EVERY_MS // HOP_MS
        for _ in range(hops_needed):
            buf.ingest(_hop_samples(1), SR)
        await buf._transcribe_task
        assert transcriber.calls == 1

        transcriber.text = "trailing words after the first cycle"
        buf.ingest(_hop_samples(1), SR)  # some new audio, but not enough to cross the threshold again

        await buf.flush()

        assert transcriber.calls == 2
        assert buf.latest_context["transcript"] == "trailing words after the first cycle"

    asyncio.run(_run())

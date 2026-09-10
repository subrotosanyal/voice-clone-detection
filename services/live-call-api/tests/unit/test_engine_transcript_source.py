"""Tests for FusedScore.transcript_source/transcript_language — added
2026-09-10 in answer to a real question ("how do I know if Whisper or
vexyl-stt was used?") whose only honest answer, until this, was "grep the
logs". See FusedScore's own docstring for the full account.

A fake transcriber (no real Whisper/vexyl-stt load) keeps this fast, same
"fake the expensive port, test the real logic" style as
test_engine_diarize_and_score.py's fake diarizer.
"""
from __future__ import annotations

import numpy as np

from app.adapters.detectors.acoustic_spectral_flatness import SpectralFlatnessDetector
from app.adapters.fusion.weighted_sum import WeightedSumFusion
from app.adapters.registry import Pipeline
from app.domain.models import TranscriptResult
from app.pipeline.engine import Engine

SR = 16_000


class _FakeTranscriber:
    def __init__(self, text: str, language: str, detector_name: str) -> None:
        self.text = text
        self.language = language
        self.detector_name = detector_name
        self.calls = 0

    def transcribe(self, samples, sample_rate):
        self.calls += 1
        return TranscriptResult(
            text=self.text,
            language=self.language,
            detector_name=self.detector_name,
            detector_version="1.0.0",
        )


def _pipeline() -> Pipeline:
    return Pipeline(
        config={
            "formula_version": "test",
            "detectors": [{"name": "acoustic", "weight": 1.0}],
            "fusion": {"smoothing_alpha": 1.0},
            "bands": {"low_max": 34, "elevated_max": 69},
            "windowing": {"window_ms": 1000, "hop_ms": 500},
        },
        detectors=[SpectralFlatnessDetector(floor_rms=0.0)],
        fusion=WeightedSumFusion(),
    )


def _loud_audio(seconds: float = 2.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_every_window_carries_the_transcriber_that_actually_ran():
    transcriber = _FakeTranscriber(text="namaste", language="hi", detector_name="vexyl_stt_transcriber")
    engine = Engine(pipeline=_pipeline(), transcriber=transcriber)

    results = engine.score_call("s1", _loud_audio(), SR, {})

    assert len(results) > 1, "need at least 2 windows to prove this isn't just the first one"
    assert transcriber.calls == 1, "the whole buffer is transcribed once, not per-window"
    for fused in results:
        assert fused.transcript_source == "vexyl_stt_transcriber"
        assert fused.transcript_language == "hi"


def test_reflects_whichever_transcriber_ran_not_a_hardcoded_name():
    """Same test as above, different transcriber identity — proves this
    isn't hardcoded to one name (the whole point: a single call can be
    routed to either Whisper or vexyl-stt, see routing_transcriber.py)."""
    transcriber = _FakeTranscriber(text="hello", language="en", detector_name="whisper_transcriber")
    engine = Engine(pipeline=_pipeline(), transcriber=transcriber)

    results = engine.score_call("s1", _loud_audio(), SR, {})

    assert all(fused.transcript_source == "whisper_transcriber" for fused in results)
    assert all(fused.transcript_language == "en" for fused in results)


def test_stays_none_when_no_transcriber_is_configured():
    engine = Engine(pipeline=_pipeline(), transcriber=None)

    results = engine.score_call("s1", _loud_audio(), SR, {})

    assert all(fused.transcript_source is None for fused in results)
    assert all(fused.transcript_language is None for fused in results)


def test_stays_none_when_the_transcriber_found_no_speech():
    """An empty transcript must not populate transcript_source either —
    score_call()'s own guard (`if transcript_result.text:`) never merges
    transcript_source/transcript_language into context in that case, so
    there's nothing informative to report."""
    transcriber = _FakeTranscriber(text="", language=None, detector_name="whisper_transcriber")
    engine = Engine(pipeline=_pipeline(), transcriber=transcriber)

    results = engine.score_call("s1", _loud_audio(), SR, {})

    assert all(fused.transcript_source is None for fused in results)


def test_a_caller_supplied_transcript_never_gets_a_transcriber_identity():
    """context already carrying a transcript (a caller-supplied one, e.g.
    from the live path's rolling window) short-circuits score_call()'s own
    transcriber.transcribe() call entirely — transcript_source must stay
    None, not silently claim credit for a transcript this transcriber
    never produced."""
    transcriber = _FakeTranscriber(text="ignored", language="hi", detector_name="vexyl_stt_transcriber")
    engine = Engine(pipeline=_pipeline(), transcriber=transcriber)

    results = engine.score_call("s1", _loud_audio(), SR, {"transcript": "already have one"})

    assert transcriber.calls == 0
    assert all(fused.transcript_source is None for fused in results)

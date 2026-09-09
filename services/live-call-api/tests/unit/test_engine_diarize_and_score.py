"""Tests for Engine.diarize_and_score() — the shared diarize-then-score-
each-speaker logic moved out of http_router.py 2026-09-10 so the live
WebSocket path's "diarize on hangup" (app/api/ws_router.py) can reuse it.
A fake diarizer (no real ECAPA-TDNN/embedding_cluster_diarizer.py load)
keeps this fast, same "fake the expensive port, test the real logic"
style as test_live_transcription.py.
"""
from __future__ import annotations

import numpy as np

from app.adapters.detectors.acoustic_spectral_flatness import SpectralFlatnessDetector
from app.adapters.fusion.weighted_sum import WeightedSumFusion
from app.adapters.history.sqlite_store import SqliteHistoryStore
from app.adapters.registry import Pipeline
from app.domain.models import SpeakerSegment
from app.pipeline.engine import Engine

SR = 16_000


class _FakeDiarizer:
    def __init__(self, segments: list[SpeakerSegment]) -> None:
        self.segments = segments
        self.calls = 0

    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        self.calls += 1
        return self.segments


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


def _loud_audio(seconds: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_returns_empty_list_when_diarizer_finds_no_segments():
    engine = Engine(pipeline=_pipeline())
    diarizer = _FakeDiarizer(segments=[])

    results = engine.diarize_and_score(diarizer, "s1", _loud_audio(2.0), SR, {})

    assert results == []


def test_splits_and_scores_each_speaker_independently():
    engine = Engine(pipeline=_pipeline())
    diarizer = _FakeDiarizer(
        segments=[
            SpeakerSegment(speaker_label="speaker_1", start_ms=0, end_ms=1000),
            SpeakerSegment(speaker_label="speaker_2", start_ms=1000, end_ms=2000),
            SpeakerSegment(speaker_label="speaker_1", start_ms=2000, end_ms=3000),
        ]
    )
    audio = _loud_audio(3.0)

    results = engine.diarize_and_score(diarizer, "base1", audio, SR, {})

    labels = [r.speaker_label for r in results]
    assert labels == ["speaker_1", "speaker_2"]  # first-seen order, deduplicated

    speaker_1 = next(r for r in results if r.speaker_label == "speaker_1")
    assert speaker_1.session_id == "base1::speaker_1"
    assert speaker_1.segment_count == 2  # two non-contiguous segments
    assert speaker_1.total_duration_ms == 2000  # 1000 + 1000
    assert len(speaker_1.fused_scores) > 0

    speaker_2 = next(r for r in results if r.speaker_label == "speaker_2")
    assert speaker_2.session_id == "base1::speaker_2"
    assert speaker_2.segment_count == 1
    assert speaker_2.total_duration_ms == 1000


def test_persists_a_speaker_summary_per_speaker(tmp_path):
    history = SqliteHistoryStore(str(tmp_path / "sessions.db"))
    engine = Engine(pipeline=_pipeline(), history=history)
    diarizer = _FakeDiarizer(segments=[SpeakerSegment(speaker_label="speaker_1", start_ms=0, end_ms=2000)])

    engine.diarize_and_score(diarizer, "base2", _loud_audio(2.0), SR, {})

    summaries = history.list_speaker_summaries("base2")
    assert len(summaries) == 1
    assert summaries[0].speaker_label == "speaker_1"
    assert summaries[0].speaker_session_id == "base2::speaker_1"
    # And the speaker's own trace is separately retrievable, same as any
    # other session — proves score_call() really did persist it.
    assert history.get_session("base2::speaker_1") != []


def test_a_speaker_with_zero_audio_is_skipped():
    """A degenerate zero-length segment (see app/pipeline/windowing.py's
    own "if total == 0: return") produces zero FusedScores — must not
    appear in the results at all, not crash."""
    engine = Engine(pipeline=_pipeline())
    diarizer = _FakeDiarizer(
        segments=[
            SpeakerSegment(speaker_label="speaker_1", start_ms=0, end_ms=2000),
            SpeakerSegment(speaker_label="speaker_2", start_ms=2000, end_ms=2000),  # zero-length
        ]
    )

    results = engine.diarize_and_score(diarizer, "base3", _loud_audio(2.0), SR, {})

    assert [r.speaker_label for r in results] == ["speaker_1"]

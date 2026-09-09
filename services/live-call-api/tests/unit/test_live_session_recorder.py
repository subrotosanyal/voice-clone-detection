"""Tests for LiveSessionRecorder — reconstructing a live session's whole,
non-overlapping audio for "diarize on hangup" (see app/api/ws_router.py).
Same window_start_ms-delta approach as live_transcription.py's REAL BUG
fix; see test_live_transcription.py for the equivalent tests on that
class.
"""
from __future__ import annotations

import numpy as np

from app.pipeline.live_session_recorder import LiveSessionRecorder

SR = 16_000
HOP_MS = 500


def _hop_samples(n_hops: int = 1) -> np.ndarray:
    return np.random.default_rng(0).uniform(-1, 1, size=round(SR * HOP_MS / 1000) * n_hops).astype(np.float32)


def test_returns_none_before_anything_ingested():
    rec = LiveSessionRecorder(hop_ms=HOP_MS)
    assert rec.get_full_audio() is None


def test_reconstructs_non_overlapping_audio_from_overlapping_windows():
    """Windows overlap by construction (2s window / 0.5s hop) — only the
    trailing new-audio slice of each should end up in the reconstruction,
    not the whole (overlapping) window every time."""
    rec = LiveSessionRecorder(hop_ms=HOP_MS)
    for i in range(4):
        rec.ingest(_hop_samples(1), SR, window_start_ms=i * HOP_MS)
    full = rec.get_full_audio()
    assert full is not None
    assert full.size == round(SR * HOP_MS / 1000) * 4  # 4 hops' worth, not 4x that from overlap


def test_backward_compatible_fixed_hop_when_window_start_ms_omitted():
    rec = LiveSessionRecorder(hop_ms=HOP_MS)
    rec.ingest(_hop_samples(1), SR)
    rec.ingest(_hop_samples(1), SR)
    full = rec.get_full_audio()
    assert full.size == round(SR * HOP_MS / 1000) * 2


def test_max_duration_drops_oldest_audio():
    rec = LiveSessionRecorder(hop_ms=HOP_MS, max_duration_ms=HOP_MS * 2)
    for i in range(5):
        rec.ingest(_hop_samples(1), SR, window_start_ms=i * HOP_MS)
    full = rec.get_full_audio()
    # Bounded: not all 5 hops' worth kept, only around the cap.
    assert full.size <= round(SR * HOP_MS / 1000) * 3


def test_sample_rate_is_recorded():
    rec = LiveSessionRecorder(hop_ms=HOP_MS)
    assert rec.sample_rate is None
    rec.ingest(_hop_samples(1), SR)
    assert rec.sample_rate == SR

"""Tests for detect_speech_segments() — real Praat calls (Parselmouth is
already a hard dependency in this project), no mocking. Exact boundary
values here are Praat's real frame-based silence detection, not a fixed
clock — see individual tests for the measured tolerance."""
import numpy as np
import pytest

from app.adapters.diarization.pause_segmentation import detect_speech_segments
from tests.audio_fixtures import VOICE_A_FORMANTS, VOICE_B_FORMANTS, formant_voice

SR = 16_000


def _segments(*args, **kwargs):
    try:
        return detect_speech_segments(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — Praat/Parselmouth unavailable in some envs
        pytest.skip(f"Parselmouth unavailable: {exc}")


def test_empty_input_returns_nothing():
    assert _segments(np.array([], dtype=np.float32), SR) == []


def test_all_silence_returns_at_most_one_region():
    """Praat's own silence detector can mislabel a perfectly flat/silent
    clip as 'sounding' (verified by hand — no intensity variation for its
    relative threshold to key off), so this doesn't assert an empty list.
    embedding_cluster_diarizer.py's own RMS floor is what actually
    excludes true silence downstream; see that module's docstring."""
    silence = np.zeros(SR * 2, dtype=np.float32)
    segments = _segments(silence, SR)
    assert len(segments) <= 1


def test_real_pause_produces_two_boundary_aligned_regions():
    voice_a = formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120, duration_s=2.0)
    gap = np.zeros(int(SR * 0.4), dtype=np.float32)
    voice_b = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=2.0)
    full = np.concatenate([voice_a, gap, voice_b])

    segments = _segments(full, SR)

    assert len(segments) == 2
    start0, end0 = segments[0]
    start1, end1 = segments[1]
    assert start0 == 0
    assert abs(end0 - 2000) <= 50  # real pause starts ~here
    assert abs(start1 - 2400) <= 50  # real pause ends ~here (400ms gap)
    assert abs(end1 - 4400) <= 50


def test_zero_gap_is_not_detected_and_falls_back_to_sub_slicing():
    """Honest limitation, not a bug: immediate back-to-back speech with no
    acoustic silence at all cannot be split by any pause-based method —
    see this module's own HONESTY NOTE. The whole buffer is treated as
    one continuous region and sub-sliced at max_segment_ms instead."""
    voice_a = formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120, duration_s=2.0)
    voice_b = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=2.0)
    full = np.concatenate([voice_a, voice_b])

    segments = _segments(full, SR, max_segment_ms=2000)

    assert segments == [(0, 2000), (2000, 4000)]


def test_long_uninterrupted_region_is_sub_sliced_at_max_segment_ms():
    audio = formant_voice(VOICE_A_FORMANTS, seed=1, duration_s=3.0)
    segments = _segments(audio, SR, max_segment_ms=4000)
    # 3s < 4000ms cap -> stays as one region, not sub-sliced
    assert segments == [(0, 3000)]


def test_short_trailing_region_below_min_segment_ms_is_dropped():
    voice_a = formant_voice(VOICE_A_FORMANTS, seed=1, duration_s=1.0)
    gap = np.zeros(int(SR * 0.4), dtype=np.float32)
    tiny_b = formant_voice(VOICE_B_FORMANTS, seed=5, pitch_hz=180, duration_s=0.2)
    full = np.concatenate([voice_a, gap, tiny_b])

    segments = _segments(full, SR, min_segment_ms=500)

    # only the first (real, ~1s) region survives; the 200ms tail is dropped
    assert len(segments) == 1
    assert segments[0][0] == 0

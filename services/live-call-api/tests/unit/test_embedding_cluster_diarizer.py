import numpy as np
import pytest

from app.adapters.diarization.embedding_cluster_diarizer import EmbeddingClusterDiarizer
from tests.audio_fixtures import VOICE_A_FORMANTS, VOICE_B_FORMANTS, formant_voice

SR = 16_000


def _diarizer(**kwargs) -> EmbeddingClusterDiarizer:
    try:
        return EmbeddingClusterDiarizer(**kwargs)
    except Exception as exc:  # noqa: BLE001 — model load can fail offline
        pytest.skip(f"ECAPA-TDNN model unavailable (no network?): {exc}")


def test_returns_empty_for_silence():
    diarizer = _diarizer()
    silence = np.zeros(SR * 3, dtype=np.float32)

    assert diarizer.diarize(silence, SR) == []


def test_single_speaker_recording_is_one_segment():
    diarizer = _diarizer(max_segment_ms=1000)
    audio = formant_voice(VOICE_A_FORMANTS, seed=1, duration_s=3.0)

    segments = diarizer.diarize(audio, SR)

    assert len(segments) == 1
    assert segments[0].speaker_label == "speaker_1"
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 3000


def test_separates_two_alternating_speakers_and_reidentifies_the_first():
    """A-B-A pattern: the diarizer must (1) find at least two distinct
    speaker labels, and (2) label the third segment the SAME as the first
    (re-identification), not a new third speaker — proving this is real
    clustering by voice similarity, not just "a new label every segment".
    No real silence gap between the three takes (concatenated directly),
    so pause detection finds no boundary here — the whole buffer is one
    continuous "sounding" region, sub-sliced at max_segment_ms exactly
    like the old fixed-grid behaviour."""
    diarizer = _diarizer(max_segment_ms=2000)
    voice_a_take1 = formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120, duration_s=2.0)
    voice_b = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=2.0)
    voice_a_take2 = formant_voice(VOICE_A_FORMANTS, seed=2, pitch_hz=122, duration_s=2.0)
    full_call = np.concatenate([voice_a_take1, voice_b, voice_a_take2])

    segments = diarizer.diarize(full_call, SR)

    assert len(segments) == 3
    labels = [s.speaker_label for s in segments]
    assert labels[0] == labels[2]  # same speaker re-identified
    assert labels[0] != labels[1]  # genuinely a different speaker in the middle
    assert [(s.start_ms, s.end_ms) for s in segments] == [(0, 2000), (2000, 4000), (4000, 6000)]


def test_skips_silent_gap_between_speakers():
    """A real 1s silence gap between the two voices — this is exactly the
    case pause-aware segmentation (see pause_segmentation.py) is meant to
    detect. Boundaries are allowed a small tolerance here (unlike the
    fixed-grid tests above): Praat's frame-based silence detection doesn't
    land on the exact millisecond a fixed clock would — measured by hand,
    the second segment here starts at 1992ms, not a clean 2000ms."""
    diarizer = _diarizer(max_segment_ms=1000, floor_rms=1e-3)
    voice_a = formant_voice(VOICE_A_FORMANTS, seed=1, duration_s=1.0)
    silence = np.zeros(SR * 1, dtype=np.float32)
    voice_b = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=1.0)
    full_call = np.concatenate([voice_a, silence, voice_b])

    segments = diarizer.diarize(full_call, SR)

    # exactly 2 voiced segments — the silent middle second contributes none
    assert len(segments) == 2
    assert abs(segments[0].end_ms - 1000) <= 50
    assert abs(segments[1].start_ms - 2000) <= 50


def test_one_speakers_natural_variation_is_not_over_counted_as_several():
    """Regression test for the real over-counting bug fixed 2026-09-08 (see
    embedding_cluster_diarizer.py's CALIBRATION HISTORY note): one person's
    voice naturally drifts a little segment to segment (breath, background
    noise, prosody) — formant_voice's seed/pitch_hz jitter stands in for
    that here, since no real labeled multi-speaker corpus exists in this
    repo. With the class's DEFAULT max_segment_ms/distance_threshold (not
    overridden, unlike the other tests above), six such takes of the SAME
    voice must still cluster as one speaker, not several. No real silence
    between takes, so this exercises the sub-slicing fallback path, not
    pause detection."""
    diarizer = _diarizer()  # class defaults: max_segment_ms=2500, distance_threshold=0.4
    takes = [
        formant_voice(VOICE_A_FORMANTS, seed=seed, pitch_hz=pitch, duration_s=2.5)
        for seed, pitch in [(1, 120), (2, 122), (3, 118), (4, 126), (5, 115), (6, 130)]
    ]
    full_call = np.concatenate(takes)

    segments = diarizer.diarize(full_call, SR)

    labels = {s.speaker_label for s in segments}
    assert labels == {"speaker_1"}, f"expected one speaker across all natural variation, got {labels}"


def test_max_speakers_caps_pathological_over_counting():
    """Regression test for the SAME over-counting bug class, but as a
    safety net independent of whatever distance_threshold turns out to be
    wrong about on some future real recording: force every segment into
    its own cluster (distance_threshold=0.0 — nothing is ever close enough
    to merge) and confirm max_speakers still bounds the result, rather than
    reporting one "speaker" per segment (the exact "speaker_124"-style
    failure a real user hit before this cap existed)."""
    diarizer = _diarizer(max_segment_ms=1000, distance_threshold=0.0, max_speakers=3)
    # 10 one-second segments, all the same voice — with distance_threshold=0.0
    # every one of them would become its own singleton cluster if the cap
    # didn't exist, i.e. 10 "speakers" for one person talking continuously.
    full_call = np.concatenate(
        [formant_voice(VOICE_A_FORMANTS, seed=i, duration_s=1.0) for i in range(10)]
    )

    segments = diarizer.diarize(full_call, SR)

    labels = {s.speaker_label for s in segments}
    assert len(labels) <= 3, f"expected at most max_speakers=3 distinct speakers, got {len(labels)}: {labels}"


def test_real_speaker_change_mid_recording_is_correctly_split_at_the_pause():
    """New coverage for pause-aware segmentation itself: two voices
    separated by a real, but brief and realistic, 300ms pause (not the 1s
    gap the older silence test above uses) must still be split into two
    correctly-attributed segments, not one straddled segment. This is the
    exact scenario the whole pause-aware rewrite (2026-09-08) targets —
    see pause_segmentation.py's HONESTY NOTE for the measured difference
    between a straddled segment's embedding and either real speaker's."""
    diarizer = _diarizer(max_segment_ms=4000)  # long enough that only pause detection can split this
    voice_a = formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120, duration_s=2.0)
    pause = np.zeros(int(SR * 0.3), dtype=np.float32)
    voice_b = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=2.0)
    full_call = np.concatenate([voice_a, pause, voice_b])

    segments = diarizer.diarize(full_call, SR)

    assert len(segments) == 2
    labels = [s.speaker_label for s in segments]
    assert labels[0] != labels[1]

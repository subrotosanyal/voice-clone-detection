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
    diarizer = _diarizer(segment_ms=1000)
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
    clustering by voice similarity, not just "a new label every segment"."""
    diarizer = _diarizer(segment_ms=2000)
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
    diarizer = _diarizer(segment_ms=1000, floor_rms=1e-3)
    voice_a = formant_voice(VOICE_A_FORMANTS, seed=1, duration_s=1.0)
    silence = np.zeros(SR * 1, dtype=np.float32)
    voice_b = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180, duration_s=1.0)
    full_call = np.concatenate([voice_a, silence, voice_b])

    segments = diarizer.diarize(full_call, SR)

    # exactly 2 voiced segments — the silent middle second contributes none
    assert len(segments) == 2
    assert segments[0].end_ms == 1000
    assert segments[1].start_ms == 2000

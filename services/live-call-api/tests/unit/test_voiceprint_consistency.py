import numpy as np
import pytest

from app.adapters.detectors.voiceprint_consistency import VoiceprintConsistencyDetector
from app.domain.models import AudioWindow
from tests.audio_fixtures import VOICE_A_FORMANTS, VOICE_B_FORMANTS, formant_voice

SR = 16_000


def _window(samples: np.ndarray) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=SR, samples=samples, window_start_ms=0)


def _detector(tmp_path, **kwargs) -> VoiceprintConsistencyDetector:
    try:
        return VoiceprintConsistencyDetector(
            enrollment_db_path=str(tmp_path / "voiceprints.db"), **kwargs
        )
    except Exception as exc:  # noqa: BLE001 — model load can fail offline
        pytest.skip(f"ECAPA-TDNN model unavailable (no network?): {exc}")


def test_abstains_without_claimed_identity(tmp_path):
    detector = _detector(tmp_path)
    audio = formant_voice(VOICE_A_FORMANTS, seed=1)

    result = detector.score(_window(audio), context={})

    assert result.score is None
    assert "claimed_identity" in result.abstain_reason


def test_abstains_without_enrollment(tmp_path):
    detector = _detector(tmp_path)
    audio = formant_voice(VOICE_A_FORMANTS, seed=1)

    result = detector.score(_window(audio), context={"claimed_identity": "alice"})

    assert result.score is None
    assert "enroll" in result.abstain_reason


def test_abstains_on_near_silence(tmp_path):
    detector = _detector(tmp_path, floor_rms=1e-3)
    detector.enroll("alice", formant_voice(VOICE_A_FORMANTS, seed=1), SR)
    silence = np.zeros(SR * 2, dtype=np.float32)

    result = detector.score(_window(silence), context={"claimed_identity": "alice"})

    assert result.score is None
    assert "silent" in result.abstain_reason


def test_same_speaker_scores_lower_risk_than_different_speaker(tmp_path):
    """The core correctness property: enrolling voice A, then presenting
    another take of voice A, must score LOWER risk than presenting voice B
    while still claiming to be A. Uses formant_voice (see
    tests/audio_fixtures.py), not pure tones — pure tones don't carry
    enough timbre difference for the embedding model to separate."""
    detector = _detector(tmp_path)
    enroll_audio = formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120)
    detector.enroll("alice", enroll_audio, SR)

    same_speaker_audio = formant_voice(VOICE_A_FORMANTS, seed=2, pitch_hz=122)
    different_speaker_audio = formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180)

    same_result = detector.score(_window(same_speaker_audio), context={"claimed_identity": "alice"})
    different_result = detector.score(
        _window(different_speaker_audio), context={"claimed_identity": "alice"}
    )

    assert same_result.score is not None and different_result.score is not None
    assert same_result.detail["cosine_similarity"] > different_result.detail["cosine_similarity"]
    assert same_result.score < different_result.score


def test_enroll_list_and_delete_round_trip(tmp_path):
    detector = _detector(tmp_path)
    detector.enroll("alice", formant_voice(VOICE_A_FORMANTS, seed=1), SR)

    summaries = detector.list_enrollments()
    assert [s.identity for s in summaries] == ["alice"]

    assert detector.delete_enrollment("alice") is True
    assert detector.list_enrollments() == []
    assert detector.delete_enrollment("alice") is False

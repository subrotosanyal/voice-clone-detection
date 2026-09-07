import numpy as np

from app.adapters.detectors.prosody_pitch_variance import PitchVarianceDetector, _estimate_pitch_hz
from app.domain.models import AudioWindow

SR = 16_000


def _window(samples: np.ndarray) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=SR, samples=samples, window_start_ms=0)


def test_pitch_estimate_recovers_known_frequency():
    t = np.linspace(0, 0.05, int(SR * 0.05), endpoint=False)  # 50ms frame
    frame = (np.sin(2 * np.pi * 150 * t)).astype(np.float32)

    hz = _estimate_pitch_hz(frame, SR)

    assert hz is not None
    assert abs(hz - 150.0) < 10.0  # autocorrelation lag quantisation tolerance


def test_pitch_estimate_none_for_silence():
    frame = np.zeros(int(SR * 0.05), dtype=np.float32)
    assert _estimate_pitch_hz(frame, SR) is None


def test_detector_abstains_when_too_few_voiced_frames():
    detector = PitchVarianceDetector()
    silence = np.zeros(int(SR * 0.5), dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.score is None
    assert result.abstain_reason is not None


def test_detector_scores_a_steady_tone():
    detector = PitchVarianceDetector()
    t = np.linspace(0, 1.0, SR, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

    result = detector.score(_window(tone), context={})

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert result.detail["voiced_frames"] >= 3
    # a perfectly steady tone has ~zero pitch variance -> should read as
    # maximally "regular", i.e. near the top of the configured scale.
    assert result.detail["coefficient_of_variation"] < 0.05

import numpy as np

from app.adapters.detectors.acoustic_spectral_flatness import SpectralFlatnessDetector, spectral_flatness
from app.domain.models import AudioWindow

SR = 16_000


def _window(samples: np.ndarray) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=SR, samples=samples, window_start_ms=0)


def test_pure_tone_has_low_flatness():
    t = np.linspace(0, 1.0, SR, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert spectral_flatness(tone) < 0.05


def test_white_noise_has_high_flatness():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.3, SR).astype(np.float32)
    assert spectral_flatness(noise) > 0.3


def test_detector_abstains_on_near_silence():
    detector = SpectralFlatnessDetector(floor_rms=1e-3)
    silence = np.zeros(SR, dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.score is None
    assert result.abstain_reason is not None


def test_detector_returns_score_and_detail_for_real_audio():
    detector = SpectralFlatnessDetector()
    t = np.linspace(0, 1.0, SR, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    result = detector.score(_window(tone), context={})

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert "spectral_flatness" in result.detail

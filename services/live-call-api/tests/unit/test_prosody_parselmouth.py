import numpy as np

from app.adapters.detectors.prosody_parselmouth import ParselmouthProsodyDetector
from app.domain.models import AudioWindow

SR = 16_000


def _window(samples: np.ndarray) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=SR, samples=samples, window_start_ms=0)


def test_abstains_on_near_silence():
    detector = ParselmouthProsodyDetector(floor_rms=1e-3)
    silence = np.zeros(SR * 2, dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.score is None
    assert result.abstain_reason is not None


def test_pure_tone_scores_maximal_risk():
    """A perfectly periodic pure tone has ~zero jitter/shimmer and very
    high HNR — the extreme case this detector's heuristic (see its
    docstring's HONESTY NOTE) treats as most synthetic-suspicious."""
    detector = ParselmouthProsodyDetector()
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    pure_tone = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

    result = detector.score(_window(pure_tone), context={})

    assert result.score is not None
    assert result.score > 0.95
    assert result.detail["jitter_local"] < 1e-3
    assert result.detail["shimmer_local"] < 1e-3
    assert result.detail["hnr_db"] > detector.hnr_ceiling


def test_wobbly_noisy_signal_scores_lower_risk_than_pure_tone():
    """A pitch-wobbling, noise-added signal has real jitter/shimmer and a
    bounded HNR — it should score meaningfully lower risk than a pure tone,
    exercising the actual Praat-derived numbers (not a constant)."""
    detector = ParselmouthProsodyDetector()
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    rng = np.random.default_rng(0)
    freq_wobble = 150 + 8 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(freq_wobble) / SR
    wobbly = (0.4 * np.sin(phase) + rng.normal(0, 0.01, t.size)).astype(np.float32)
    pure_tone = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

    wobbly_result = detector.score(_window(wobbly), context={})
    tone_result = detector.score(_window(pure_tone), context={})

    assert wobbly_result.score is not None
    assert wobbly_result.detail["jitter_local"] > tone_result.detail["jitter_local"]
    assert wobbly_result.score < tone_result.score


def test_score_is_always_within_unit_bounds():
    detector = ParselmouthProsodyDetector()
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    rng = np.random.default_rng(7)
    signal = (0.3 * np.sin(2 * np.pi * 200 * t) + rng.normal(0, 0.05, t.size)).astype(np.float32)

    result = detector.score(_window(signal), context={})

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

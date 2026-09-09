import numpy as np
import pytest

from app.adapters.detectors import prosody_parselmouth
from app.adapters.detectors.prosody_parselmouth import ParselmouthProsodyDetector, ParselmouthProsodyMLDetector
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


def test_abstains_cleanly_instead_of_leaking_a_raw_praat_error_on_a_short_window():
    """Regression test for a real bug found 2026-09-09 (via a real external
    deepfake sample whose trailing window was ~26.7ms): Praat's own
    pitch-period extraction needs at least 3 periods of _MIN_PITCH_HZ
    (70Hz) visible, i.e. ~42.9ms — a shorter window was already caught
    (no crash) but abstained with Praat's raw, jargon-heavy exception
    text ("minimum pitch must not be less than 112.5Hz...") instead of a
    clean reason. Loud (not silent) so this exercises the NEW length
    guard specifically, not the existing near-silence one above."""
    detector = ParselmouthProsodyDetector()
    # 20ms at 16kHz = 320 samples — below both the ~42.9ms Praat minimum
    # and this detector's own 60ms (_MIN_DURATION_S) guard.
    short_loud = np.random.default_rng(0).uniform(-0.5, 0.5, size=320).astype(np.float32)

    result = detector.score(_window(short_loud), context={})

    assert result.score is None
    assert result.abstain_reason is not None
    assert "short" in result.abstain_reason
    assert "112.5" not in result.abstain_reason  # no raw Praat internals leaking through


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


# --- ParselmouthProsodyMLDetector — deployed 2026-09-09, see
# app/adapters/detectors/prosody_parselmouth.py's own module docstring
# ("2026-09-09: trained-classifier upgrade") for the real calibration
# numbers behind this. Reuses the same _extract_features() abstain logic
# as ParselmouthProsodyDetector (tested above), so these tests focus on
# what's actually new: the trained score mapping. ------------------------


def test_ml_detector_abstains_on_near_silence():
    """Same shared _extract_features() path as ParselmouthProsodyDetector
    — confirms the refactor into a shared helper didn't lose this."""
    detector = ParselmouthProsodyMLDetector(floor_rms=1e-3)
    silence = np.zeros(SR * 2, dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.score is None
    assert result.abstain_reason is not None


def test_ml_detector_abstains_cleanly_on_short_window():
    detector = ParselmouthProsodyMLDetector()
    short_loud = np.random.default_rng(0).uniform(-0.5, 0.5, size=320).astype(np.float32)

    result = detector.score(_window(short_loud), context={})

    assert result.score is None
    assert result.abstain_reason is not None
    assert "short" in result.abstain_reason


def test_ml_detector_score_is_always_within_unit_bounds():
    detector = ParselmouthProsodyMLDetector()
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    rng = np.random.default_rng(7)
    signal = (0.3 * np.sin(2 * np.pi * 200 * t) + rng.normal(0, 0.05, t.size)).astype(np.float32)

    result = detector.score(_window(signal), context={})

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


def test_ml_detector_reproduces_the_exact_fitted_logistic_regression(monkeypatch):
    """Pins the hardcoded _ML_SCALER_MEAN/_ML_SCALER_SCALE/_ML_COEF/
    _ML_INTERCEPT constants against a hand-computed sigmoid, independent
    of Praat entirely (monkeypatches _measure_voice_quality with a fixed
    triple) — catches an accidental edit to those constants going
    forward. Values are the exact means from eval/indian_language/
    results/prosodic_calibration.json's genuine_train distribution, which
    a real (not fabricated) genuine example should therefore land near
    the classifier's own decision boundary for."""
    monkeypatch.setattr(
        prosody_parselmouth,
        "_measure_voice_quality",
        lambda samples, sample_rate: (0.0203, 0.096, 13.4),  # ~genuine-train means
    )
    detector = ParselmouthProsodyMLDetector()
    loud_enough = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = detector.score(_window(loud_enough), context={})

    z = [
        (0.0203 - 0.022166341806446492) / 0.005116343520411652,
        (0.096 - 0.09576296062763698) / 0.02495170502399299,
        (13.4 - 13.469464679917822) / 3.1039219121944606,
    ]
    coef = (0.3018977933589121, -0.31844813946535616, 0.03738590949551451)
    logit = sum(c * v for c, v in zip(coef, z)) + -0.02643924021858421
    expected = 1.0 / (1.0 + np.exp(-logit))

    assert result.score is not None
    assert result.score == pytest.approx(expected, abs=1e-9)


def test_ml_detector_and_heuristic_detector_can_disagree(monkeypatch):
    """The whole point of the 2026-09-09 upgrade (see module docstring):
    the two detectors are not the same formula wearing different code,
    and CAN disagree on the same input. This specific (jitter, shimmer,
    hnr) triple is a real, verified case (jitter/hnr both comfortably
    inside ParselmouthProsodyDetector's "normal" range, so its heuristic
    averages down to low risk despite low shimmer alone) where the
    heuristic reads LOW risk (0.238, jitter_risk=hnr_risk=0 dilutes the
    average) but the trained model reads HIGH risk (0.718) — confirmed
    by hand-computing both formulas before writing this assertion, not
    guessed."""
    monkeypatch.setattr(
        prosody_parselmouth,
        "_measure_voice_quality",
        lambda samples, sample_rate: (0.02, 0.01, 13.0),
    )
    heuristic = ParselmouthProsodyDetector()
    ml = ParselmouthProsodyMLDetector()
    loud_enough = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    heuristic_result = heuristic.score(_window(loud_enough), context={})
    ml_result = ml.score(_window(loud_enough), context={})

    assert heuristic_result.score < 0.5
    assert ml_result.score > 0.5

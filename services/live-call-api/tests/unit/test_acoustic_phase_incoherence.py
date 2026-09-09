"""Tests for PhaseIncoherenceDetector.

See the detector's own module docstring for the real corpus numbers this
was calibrated from (eval/indian_language/data/, resampled to a common
16kHz after a real sample-rate-confound bug was found and fixed during
that validation). These tests exercise the deterministic DSP mechanics on
synthetic signals — a pure tone as the maximally phase-coherent extreme,
same pattern as test_prosody_parselmouth.py's pure-tone test for jitter/
shimmer/HNR — not a re-run of the corpus validation itself.
"""
from __future__ import annotations

import numpy as np

from app.adapters.detectors.acoustic_phase_incoherence import PhaseIncoherenceDetector
from app.domain.models import AudioWindow

SR = 16_000


def _window(samples: np.ndarray, sample_rate: int = SR) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=sample_rate, samples=samples, window_start_ms=0)


def test_abstains_on_near_silence():
    detector = PhaseIncoherenceDetector()
    silence = np.zeros(SR * 2, dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.score is None
    assert result.abstain_reason is not None


def test_abstains_on_too_short_window():
    detector = PhaseIncoherenceDetector()
    # 300 samples at 16kHz is loud but far too short for a reliable
    # multi-frame phase-delta estimate (needs _N_FFT + 4*_HOP_LENGTH).
    short_loud = np.random.default_rng(0).uniform(-0.5, 0.5, size=300).astype(np.float32)

    result = detector.score(_window(short_loud), context={})

    assert result.score is None
    assert "short" in result.abstain_reason


def test_pure_tone_is_maximally_phase_coherent_and_scores_high_risk():
    """A pure sine tone's phase advances by an exactly constant amount every
    STFT frame — the maximally phase-coherent (minimally incoherent) case,
    and exactly the pattern this detector's heuristic (see its docstring's
    HONESTY NOTE and VERIFIED BY HAND section) treats as most synthetic-
    suspicious."""
    detector = PhaseIncoherenceDetector()
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    pure_tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    result = detector.score(_window(pure_tone), context={})

    assert result.score is not None
    assert result.detail["phase_incoherence"] < 0.1
    assert result.score > 0.9


def test_noisy_signal_is_more_phase_incoherent_than_pure_tone():
    detector = PhaseIncoherenceDetector()
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    rng = np.random.default_rng(1)
    pure_tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noisy = (0.3 * rng.standard_normal(t.size)).astype(np.float32)

    tone_result = detector.score(_window(pure_tone), context={})
    noisy_result = detector.score(_window(noisy), context={})

    # The score mapping's floor/span (0.50-0.60) is calibrated against real
    # speech's measured range (~0.5-0.72 — see the module docstring); a
    # synthetic tone and white noise both sit well below that window on the
    # raw metric, so both legitimately clip to the same max risk score.
    # The metric itself (not the clipped score) is the real assertion here.
    assert noisy_result.detail["phase_incoherence"] > tone_result.detail["phase_incoherence"]
    assert noisy_result.score <= tone_result.score


def test_resamples_non_16k_windows_instead_of_measuring_the_recording_chain():
    """Regression test for the REAL BUG found while validating this
    detector (see its module docstring): comparing raw spectral features
    across different native sample rates measures the recording chain, not
    the speech content. The same underlying tone content, delivered at two
    different sample rates, must resample to a common rate internally and
    produce a similar phase_incoherence — not the wildly different values
    a naive same-n_fft comparison across sample rates would give."""
    detector = PhaseIncoherenceDetector()
    duration_s = 2.0

    t_16k = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    tone_16k = (0.4 * np.sin(2 * np.pi * 220 * t_16k)).astype(np.float32)

    other_sr = 48_000
    t_other = np.linspace(0, duration_s, int(other_sr * duration_s), endpoint=False)
    tone_other = (0.4 * np.sin(2 * np.pi * 220 * t_other)).astype(np.float32)

    result_16k = detector.score(_window(tone_16k, sample_rate=SR), context={})
    result_other = detector.score(_window(tone_other, sample_rate=other_sr), context={})

    assert result_16k.score is not None
    assert result_other.score is not None
    assert abs(result_16k.detail["phase_incoherence"] - result_other.detail["phase_incoherence"]) < 0.05


def test_score_is_always_within_unit_bounds():
    detector = PhaseIncoherenceDetector()
    rng = np.random.default_rng(7)
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * 200 * t) + rng.normal(0, 0.05, t.size)).astype(np.float32)

    result = detector.score(_window(signal), context={})

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


def test_result_carries_detector_identity():
    detector = PhaseIncoherenceDetector()
    silence = np.zeros(SR * 2, dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.detector_name == "acoustic_phase_incoherence"
    assert result.detector_version == "0.1.0-stft-circvar"

"""PhaseIncoherenceDetector — STFT phase-coherence, a CPU-cheap complement
to AASIST.

WHY THIS EXISTS: neural vocoders (HiFi-GAN and similar, underlying both
XTTS-v2 and Chatterbox — see eval/indian_language/) typically reconstruct
a magnitude spectrogram and either discard phase or estimate it crudely,
producing MORE internally-consistent (coherent) frame-to-frame phase than
a real human vocal tract, which is a noisier physical process. This
detector measures that: the circular variance of frame-to-frame phase
deltas per frequency bin (magnitude-weighted), a real, deterministic DSP
computation — not a trained classifier.

ORIGIN: this started from a user-supplied "AdvancedAcousticProber"
snippet proposing phase-incoherence, spectral-flux and cepstral features
together. That snippet had real bugs (assumed int16 PCM on already-
normalized float32 samples; a `cepstrum = np.fft.textual = ...` typo that
silently monkey-patched numpy.fft itself; called plain `np.var` on wrapped
phase angles while claiming "circular variance"; used HIGH-quefrency
cepstral energy as "vocal tract purity", backwards from standard cepstral
analysis where the vocal-tract envelope is the LOW-quefrency band) and
asserted specific risk thresholds as "hand-calibrated" with nothing behind
that claim. None of that survived contact with real data — see below.

VERIFIED BY HAND, not assumed (2026-09-09) — measured against this
project's own eval/indian_language/data/ corpus (200 genuine Hindi
utterances from IndicTTS-Hindi, 40 XTTS-v2-cloned, 40 held-out Chatterbox-
cloned, none used in AASIST's own fine-tuning):

- REAL BUG found and fixed during this validation: the genuine corpus is
  48kHz PCM16; both synthetic corpora are 24kHz (XTTS-v2 PCM16, Chatterbox
  FLOAT). Computing spectral/cepstral features straight off each file's
  native sample rate measured the RECORDING CHAIN, not genuine-vs-
  synthetic content — a classic anti-spoofing shortcut-learning pitfall.
  That naive comparison produced absurd "effect sizes" (Cohen's d up to
  8.0, AUC of 1.000/0.000) driven entirely by the sample-rate mismatch;
  every one of those numbers vanished (d=-0.08, p=0.71) once every file
  was resampled to a common 16kHz — matching what acoustic_aasist.py's own
  `_MODEL_SAMPLE_RATE` already does — before feature extraction. This is
  why `score()` below resamples internally rather than trusting
  `window.sample_rate` to already be 16kHz.
- With that fixed, phase_incoherence itself holds up: genuine mean=0.644
  (std=0.033, n=200) vs. XTTS-v2 mean=0.620 (Cohen's d=+0.70, AUC=0.68,
  Mann-Whitney p<0.001) and vs. HELD-OUT Chatterbox mean=0.559 (Cohen's
  d=+2.56, AUC=0.96, p<0.0001) — genuine speech has reliably HIGHER phase
  incoherence than either synthesis system, moderately against XTTS-v2,
  strongly against a system never used to tune this detector.
- Two other features from the original snippet (RMS-normalized spectral
  flux variance, and both directions of the cepstral quefrency split) were
  measured the same way and did NOT hold up post-resampling-fix (flux:
  p=0.86 vs. XTTS-v2; cepstral ratio: p=0.71 vs. XTTS-v2, p=0.41 vs.
  Chatterbox) — deliberately left out of this detector rather than shipped
  on the strength of the pre-fix numbers. Do not re-add them without
  re-measuring the same way.

HONESTY NOTE — read this before trusting any output of this class:
- This is a heuristic SCORE MAPPING calibrated against the population
  statistics above, not a trained classifier — same caveat as
  prosody_parselmouth.py and prosody_pitch_variance.py.
- It is meaningfully WEAKER against XTTS-v2 (d=0.70, AUC=0.68 — not a
  reliable standalone signal on its own) than against Chatterbox (d=2.56,
  AUC=0.96). Do not read a low score here as "not Chatterbox-like" and a
  high score as "definitely synthetic" without the rest of the fused
  signal; this is one input among several, exactly per DetectorPort's
  contract.
- Chatterbox's corpus files are FLOAT-encoded, unlike genuine/XTTS-v2's
  PCM16 — a residual encoding-vs-content confound this analysis has not
  fully ruled out for Chatterbox specifically. It's the same reason the
  cepstral features (which ARE quantization-noise-sensitive) were dropped.
  It's less likely to explain the phase result, both because the XTTS-v2
  effect (same PCM16 encoding as genuine) already replicates the same
  direction on its own, and because 16-bit quantization noise is phase-
  random rather than phase-*coherent* — but this has not been separately
  verified with a matched-encoding Chatterbox re-render, so treat the
  Chatterbox number as directionally supportive, not independently proven.
"""
from __future__ import annotations

import numpy as np

from app.domain.models import AudioWindow, DetectorResult

_MODEL_SAMPLE_RATE = 16_000
_N_FFT = 512
_HOP_LENGTH = 128
# Needs enough frames for a handful of phase deltas to be meaningful, not
# just 1-2 — see the "too few frames" abstain path below.
_MIN_FRAMES = 5

# Calibrated against the measured genuine population above (mean=0.644,
# std=0.033, p10=0.593, p25=0.627): below this floor is increasingly rare
# for genuine speech; risk reaches 1.0 by the low end of Chatterbox's own
# measured range (min=0.497) — see the module docstring's VERIFIED BY HAND
# section for the numbers this was calibrated from, not a guess.
_INCOHERENCE_FLOOR = 0.60
_INCOHERENCE_RISK_SPAN = 0.10


class PhaseIncoherenceDetector:
    """STFT phase-coherence, mapped to a heuristic risk score."""

    name = "acoustic_phase_incoherence"
    version = "0.1.0-stft-circvar"

    def __init__(
        self,
        floor_rms: float = 1e-4,
        incoherence_floor: float = _INCOHERENCE_FLOOR,
        incoherence_risk_span: float = _INCOHERENCE_RISK_SPAN,
    ) -> None:
        self.floor_rms = floor_rms
        self.incoherence_floor = incoherence_floor
        self.incoherence_risk_span = incoherence_risk_span

    def score(self, window: AudioWindow, context: dict) -> DetectorResult:
        samples = window.samples
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        if rms < self.floor_rms:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"rms": rms, "floor_rms": self.floor_rms},
                abstain_reason="window is near-silent — nothing to measure",
            )

        sample_rate = window.sample_rate
        if sample_rate != _MODEL_SAMPLE_RATE:
            # See the module docstring's REAL BUG note: comparing raw
            # spectral features across different native sample rates
            # measures the recording chain, not the speech content.
            samples = _resample(samples, sample_rate, _MODEL_SAMPLE_RATE)
            sample_rate = _MODEL_SAMPLE_RATE

        min_samples = _N_FFT + (_MIN_FRAMES - 1) * _HOP_LENGTH
        if samples.size < min_samples:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"rms": rms, "window_samples": int(samples.size), "min_samples_needed": min_samples},
                abstain_reason=(
                    f"window too short ({samples.size} samples at {sample_rate}Hz) for a reliable "
                    f"phase-coherence estimate — needs at least {min_samples}"
                ),
            )

        phase_incoherence = _phase_incoherence(samples.astype(np.float32))

        risk = np.clip(
            (self.incoherence_floor - phase_incoherence) / self.incoherence_risk_span, 0.0, 1.0
        )
        raw_score = float(risk)

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={
                "phase_incoherence": phase_incoherence,
                "incoherence_floor_reference": self.incoherence_floor,
                "rms": rms,
                "explanation": _build_explanation(phase_incoherence, self.incoherence_floor, raw_score),
            },
        )


def _phase_incoherence(samples: np.ndarray) -> float:
    """Magnitude-weighted circular variance of frame-to-frame phase deltas.

    Circular variance of a set of angles theta is 1 - |mean(e^{i*theta})|,
    in [0, 1] — 0 means every delta points the same direction (perfectly
    coherent phase evolution), 1 means uniformly scattered (incoherent).
    Weighting by each bin's mean magnitude keeps near-silent, phase-noise-
    dominated frequency bins from swamping the bins that actually carry
    speech energy.
    """
    from scipy.signal import stft

    _, _, z = stft(samples, nperseg=_N_FFT, noverlap=_N_FFT - _HOP_LENGTH, boundary=None, padded=False)
    magnitude = np.abs(z)
    phase = np.angle(z)

    phase_deltas = np.diff(phase, axis=1)
    mean_vector = np.mean(np.exp(1j * phase_deltas), axis=1)
    per_bin_circular_variance = 1.0 - np.abs(mean_vector)

    bin_weight = np.mean(magnitude[:, 1:], axis=1)
    weight_sum = np.sum(bin_weight)
    if weight_sum <= 0:
        return float(np.mean(per_bin_circular_variance))
    bin_weight = bin_weight / weight_sum

    return float(np.sum(per_bin_circular_variance * bin_weight))


def _build_explanation(phase_incoherence: float, floor: float, risk: float) -> str:
    if risk <= 0.0:
        return (
            f"Phase incoherence ({phase_incoherence:.3f}) is at or above the {floor:.2f} natural-speech "
            "reference — no phase-based risk factor detected."
        )
    return (
        f"Phase incoherence ({phase_incoherence:.3f}) is below the {floor:.2f} natural-speech reference — "
        "unusually phase-coherent audio can indicate a neural vocoder, though this is meaningfully weaker "
        "evidence against some synthesis systems than others (see this detector's HONESTY NOTE) and should "
        "not be read in isolation."
    )


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(samples, up, down).astype(np.float32)

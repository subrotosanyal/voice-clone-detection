"""Prosodic detector — v0 baseline.

HONESTY NOTE: same caveat as acoustic_spectral_flatness.py — this is a real,
deterministic DSP computation (autocorrelation-based pitch estimation, per
sub-frame, then its coefficient of variation across the window), not a
validated indicator of synthetic prosody on its own. Replace with a proper
front-end (Parselmouth/Praat, or a learned prosody model) behind this same
DetectorPort before trusting the output — see the blueprint's §02 table.
"""
from __future__ import annotations

import numpy as np

from app.domain.models import AudioWindow, DetectorResult

_MIN_HZ = 70.0
_MAX_HZ = 400.0


def _estimate_pitch_hz(frame: np.ndarray, sample_rate: int) -> float | None:
    """Autocorrelation pitch estimate for one short sub-frame. Returns None
    if the frame is too quiet or no plausible periodicity is found."""
    if frame.size < 32:
        return None
    frame = frame - np.mean(frame)
    energy = np.sum(frame**2)
    if energy < 1e-9:
        return None

    corr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]
    min_lag = int(sample_rate / _MAX_HZ)
    max_lag = min(int(sample_rate / _MIN_HZ), corr.size - 1)
    if max_lag <= min_lag:
        return None

    search = corr[min_lag:max_lag]
    if search.size == 0 or np.max(search) <= 0:
        return None
    peak_lag = min_lag + int(np.argmax(search))
    if corr[0] <= 0:
        return None
    periodicity = corr[peak_lag] / corr[0]
    if periodicity < 0.3:  # too aperiodic to trust as voiced pitch
        return None
    return sample_rate / peak_lag


class PitchVarianceDetector:
    """Coefficient of variation of frame-level pitch across the window."""

    name = "prosody_pitch_variance"
    version = "0.1.0-baseline"

    def __init__(self, frame_ms: int = 40, hop_ms: int = 20, scale: float = 2.5) -> None:
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.scale = scale

    def score(self, window: AudioWindow, context: dict) -> DetectorResult:
        sr = window.sample_rate
        frame_len = int(sr * self.frame_ms / 1000)
        hop_len = int(sr * self.hop_ms / 1000)
        samples = window.samples

        pitches: list[float] = []
        for start in range(0, max(samples.size - frame_len, 0) + 1, hop_len):
            frame = samples[start : start + frame_len]
            hz = _estimate_pitch_hz(frame, sr)
            if hz is not None:
                pitches.append(hz)

        if len(pitches) < 3:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"voiced_frames": len(pitches)},
                abstain_reason="too few voiced frames to estimate pitch variance",
            )

        pitches_arr = np.array(pitches)
        mean_hz = float(np.mean(pitches_arr))
        std_hz = float(np.std(pitches_arr))
        coeff_variation = std_hz / mean_hz if mean_hz > 0 else 0.0

        # Overly regular pitch (low variance) is the heuristic "suspicious"
        # direction here: raw_score rises as coeff_variation falls below a
        # natural-speech-ish midpoint. This is a coarse, documented guess —
        # see the module docstring.
        natural_midpoint = 0.18
        raw_score = float(np.clip((natural_midpoint - coeff_variation) / natural_midpoint, 0.0, 1.0))
        raw_score = float(np.clip(raw_score * self.scale, 0.0, 1.0))

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={
                "mean_pitch_hz": mean_hz,
                "pitch_std_hz": std_hz,
                "coefficient_of_variation": coeff_variation,
                "voiced_frames": len(pitches),
            },
        )

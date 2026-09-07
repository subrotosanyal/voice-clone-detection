"""Acoustic detector — v0 baseline.

HONESTY NOTE (read this before trusting any output of this class):
This is a placeholder so the pipeline has a real, deterministic acoustic
signal to compute end to end — it is NOT a validated spoof/deepfake
detector. Spectral flatness (the ratio of the geometric mean to the
arithmetic mean of the power spectrum) is a real, well-understood DSP
measurement, but there is no published evidence backing it as a synthetic-
speech indicator on its own. Before this system's acoustic score should be
trusted for anything, swap this class for a wrapper around a published,
pretrained countermeasure (AASIST, RawGAT-ST/RawNet2 — see the "Component
→ open-source candidate" table in the project blueprint, §02) behind the
same DetectorPort interface. Nothing else in the pipeline needs to change
when you do that — that's the point of the port.
"""
from __future__ import annotations

import numpy as np

from app.domain.models import AudioWindow, DetectorResult

_EPS = 1e-12


def spectral_flatness(samples: np.ndarray) -> float:
    """Wiener entropy: geometric_mean(power_spectrum) / arithmetic_mean(power_spectrum).

    Returns a value in [0, 1]. Near 0 for tonal/harmonic signals (a clean
    sine tone, voiced speech), near 1 for noise-like/flat spectra (white
    noise, silence-floor hiss).
    """
    if samples.size == 0:
        return 0.0
    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    spectrum = spectrum[spectrum > 0]
    if spectrum.size == 0:
        return 0.0
    geometric_mean = np.exp(np.mean(np.log(spectrum + _EPS)))
    arithmetic_mean = np.mean(spectrum)
    return float(np.clip(geometric_mean / (arithmetic_mean + _EPS), 0.0, 1.0))


class SpectralFlatnessDetector:
    """Maps spectral flatness to a [0,1] "acoustic" risk contribution."""

    name = "acoustic_spectral_flatness"
    version = "0.1.0-baseline"

    def __init__(self, scale: float = 1.0, floor_rms: float = 1e-4) -> None:
        self.scale = scale
        self.floor_rms = floor_rms

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

        flatness = spectral_flatness(samples)
        raw_score = float(np.clip(flatness * self.scale, 0.0, 1.0))
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={"spectral_flatness": flatness, "rms": rms, "scale": self.scale},
        )

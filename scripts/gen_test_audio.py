#!/usr/bin/env python3
"""Generates synthetic WAV fixtures for smoke-testing the live-call pipeline.

IMPORTANT: these are NOT real voice recordings and NOT deepfake samples —
they're plain sine tones and white noise, useful only for exercising the
plumbing (does a file go in, does a score come out, does it look sane).
They tell you nothing about real detection accuracy. For that, see the
project blueprint's §07/§08 on evaluation datasets (ASVspoof, In-the-Wild,
WaveFake) and the held-out-generator discipline in §04.

Usage:
    python scripts/gen_test_audio.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
DURATION_S = 4.0
OUT_DIR = Path(__file__).resolve().parent.parent / "services" / "live-call-api" / "samples"


def _tone_with_natural_wobble(seed: int) -> np.ndarray:
    """A decaying tone with a wandering pitch and a little noise — meant to
    exercise the pipeline with *some* natural-ish pitch variance, not to
    represent a validated "genuine speech" fixture."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S), endpoint=False)
    base_hz = 150.0
    wobble = 8.0 * np.sin(2 * np.pi * 0.6 * t) + rng.normal(0, 1.5, size=t.size).cumsum() * 0.002
    instantaneous_hz = base_hz + wobble
    phase = 2 * np.pi * np.cumsum(instantaneous_hz) / SAMPLE_RATE
    signal = 0.5 * np.sin(phase)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t) ** 2
    signal = signal * envelope
    signal += rng.normal(0, 0.01, size=signal.size)
    return signal.astype(np.float32)


def _white_noise(seed: int) -> np.ndarray:
    """Flat-spectrum noise — a stand-in "suspicious" clip for the acoustic
    baseline detector's spectral-flatness signal, not a real spoof sample."""
    rng = np.random.default_rng(seed)
    n_samples = int(SAMPLE_RATE * DURATION_S)
    signal = rng.normal(0, 0.3, size=n_samples).astype(np.float32)
    return signal


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    genuine_path = OUT_DIR / "genuine_tone.wav"
    sf.write(genuine_path, _tone_with_natural_wobble(seed=42), SAMPLE_RATE)
    print(f"wrote {genuine_path}")

    noisy_path = OUT_DIR / "noisy_clip.wav"
    sf.write(noisy_path, _white_noise(seed=7), SAMPLE_RATE)
    print(f"wrote {noisy_path}")


if __name__ == "__main__":
    main()

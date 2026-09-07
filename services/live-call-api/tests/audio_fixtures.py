"""Shared synthetic-audio helpers for tests that need signals a speaker-
embedding model can actually tell apart.

Plain sine tones (used elsewhere in this test suite, e.g.
test_acoustic_detector.py) carry almost no vocal-tract/timbre information,
so ECAPA-TDNN embeddings of two different-frequency tones end up nearly as
similar as two embeddings of the *same* tone — verified by hand while
building voiceprint_consistency.py: two pure tones at 150Hz vs 400+800Hz
still cosine-similarity'd at ~0.93. `formant_voice()` instead generates a
pulsed source (crude glottal-pulse analogue) filtered through a few
band-pass "formants", which gives genuinely different spectral envelopes
between calls with different `formants` — enough for the embedding model to
separate "same voice, two takes" from "different voice" with a real margin
(observed ~0.87 vs ~0.55 cosine similarity for the two presets below, vs.
undistinguishable for pure tones). Still synthetic, not real speech — see
scripts/gen_test_audio.py's docstring for the project's standing caveat on
what synthetic fixtures can and can't prove.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

# Two presets with clearly different formant structure and pitch, standing
# in for "two different speakers" in tests.
VOICE_A_FORMANTS = (700, 1200, 2500)
VOICE_B_FORMANTS = (300, 2200, 3000)


def formant_voice(
    formants: tuple[int, ...],
    sample_rate: int = 16_000,
    duration_s: float = 2.0,
    seed: int = 0,
    pitch_hz: float = 120.0,
) -> np.ndarray:
    n = int(sample_rate * duration_s)
    pulse = np.zeros(n, dtype=np.float64)
    period = max(int(sample_rate / pitch_hz), 1)
    pulse[::period] = 1.0
    rng = np.random.default_rng(seed)
    pulse = pulse + rng.normal(0, 0.02, n)  # a little jitter/breath noise

    signal = np.zeros(n, dtype=np.float64)
    nyquist = sample_rate / 2
    for center_hz in formants:
        low = max(center_hz - 100, 20) / nyquist
        high = min(center_hz + 100, sample_rate / 2 - 10) / nyquist
        sos = butter(4, [low, high], btype="band", output="sos")
        signal += sosfilt(sos, pulse)

    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak * 0.5
    return signal.astype(np.float32)

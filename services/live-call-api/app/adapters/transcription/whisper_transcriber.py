"""WhisperTranscriber — real speech-to-text via OpenAI's Whisper.

WHY WHISPER: MIT licensed, confirmed ungated on HuggingFace (`gated:
false`), genuinely multilingual (English, Hindi, and Marathi all verified
by hand — see below), and runs CPU-only, matching every other model this
project uses. No auth token needed to fetch it, same "docker build just
works" requirement applied to every other model choice here.

MODEL SIZE — verified by hand, not assumed: the smaller "base" model
mis-transcribed real Hindi speech into Urdu script (a known Whisper
quirk — the two languages are spoken near-identically but written in
different scripts, and "base" isn't reliable enough to keep them
straight). The "small" model, tested on the same audio, correctly
detected the language and produced accurate Devanagari text — the
smallest size that gave usable Hindi/Marathi output when this was checked.
Still fast on CPU (~1-2s for a several-second clip once loaded).

WHAT THIS ISN'T: a general-purpose transcription feature. It exists only
to feed app/adapters/detectors/contextual_rules.py's urgency/financial-
request keyword detection (see
app/adapters/transcription/urgency_language.py) — the transcript itself
isn't scored directly, and nothing here does sentiment analysis in the
generic (positive/negative movie-review) sense. See docs/risk-model.md
for why a transparent keyword list was chosen over a black-box sentiment
model for this purpose.

REPRODUCIBILITY FIX (2026-09-08) — real bug, found and fixed: Whisper's
`transcribe()` defaults to a TEMPERATURE FALLBACK tuple
(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), not a single fixed temperature. On audio
with no real speech content (silence, a pure tone, background noise —
exactly this project's synthetic test fixtures), the initial greedy
(temperature=0.0) pass can fail Whisper's own internal quality gates
(avg_logprob / compression_ratio thresholds), triggering a fallback to
temperature>0 SAMPLING — which draws from PyTorch's global, unseeded
RNG. Verified by hand: this made transcription genuinely non-
deterministic call to call on the exact same audio within one process
(the RNG state differs depending on what else ran earlier), producing
either an empty transcript or a hallucinated one (observed:
"MMMMMMMMMMMMMM" — a real Whisper hallucination artifact on non-speech
audio, not a parsing bug). That hallucinated text then fed the intent
classifier, which scored it as high-risk, changing the final fused score
call to call — the exact mechanism behind a real intermittent CI
failure (test_repeated_calls_with_same_input_are_reproducible). Fix:
pin `temperature=0.0` (a single float, not a tuple) below, which
disables the sampling fallback entirely and forces pure greedy decoding
— deterministic by construction, regardless of RNG state. This does
trade away Whisper's one real recovery mechanism for a genuinely
low-confidence greedy pass on real difficult audio (it will no longer
retry at higher temperature), but reproducibility is the higher-priority
guarantee for a system whose whole design point is explainable, replay-
identical scoring — see docs/risk-model.md's "Reproducibility" section.
"""
from __future__ import annotations

from math import gcd

import numpy as np

from app.domain.models import TranscriptResult

_MODEL_SAMPLE_RATE = 16_000
_DOWNLOAD_ROOT = "app/adapters/transcription/.cache"


class WhisperTranscriber:
    name = "whisper_transcriber"
    version = "whisper-small"

    def __init__(self, model_size: str = "small", floor_rms: float = 1e-4) -> None:
        import whisper

        self.model_size = model_size
        self.floor_rms = floor_rms
        self._model = whisper.load_model(model_size, download_root=_DOWNLOAD_ROOT)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        if rms < self.floor_rms:
            return TranscriptResult(text="", language=None, detector_name=self.name, detector_version=self.version)

        if sample_rate != _MODEL_SAMPLE_RATE:
            samples = _resample(samples, sample_rate, _MODEL_SAMPLE_RATE)

        # temperature=0.0 (a single float, not Whisper's default fallback
        # tuple) forces pure greedy decoding — see this module's
        # REPRODUCIBILITY FIX docstring note for why the default is
        # genuinely non-deterministic on non-speech audio.
        result = self._model.transcribe(samples.astype(np.float32), fp16=False, temperature=0.0)
        return TranscriptResult(
            text=result["text"].strip(),
            language=result.get("language"),
            detector_name=self.name,
            detector_version=self.version,
        )


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(samples, up, down).astype(np.float32)

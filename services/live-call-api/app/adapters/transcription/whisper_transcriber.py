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

        result = self._model.transcribe(samples.astype(np.float32), fp16=False)
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

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

REAL BUG found and fixed 2026-09-09, via dogfooding real external audio
(IndieFake Dataset's public demo clips) through the deployed service:
a very short window crashed the WHOLE request with an uncaught
`RuntimeError: cannot reshape tensor of 0 elements into shape
[1, 0, 12, -1] because the unspecified dimension size -1 can be any
value and is ambiguous` inside whisper/model.py's
MultiHeadAttention.qkv_attention — not abstained, an actual HTTP 500.
The "12" in that shape is this project's "small" model's attention head
count, and the "0" is a zero-length context dimension: some internal
Whisper framing step (mel/STFT windowing, or a segment slice inside its
own `transcribe()` seek loop) produced a zero-frame sequence for a
too-short input, which cascades into that ambiguous reshape once it
reaches attention. Same class of bug as the one already fixed in
app/adapters/detectors/perth_watermark.py (see its own "REAL BUG found
and fixed 2026-09-09" docstring note) — a real trailing partial window
from windowing.py "is still yielded" even when much shorter than the
configured window length, which is a normal occurrence whenever a
file's duration isn't an exact multiple of the hop length, not a
contrived edge case. Fixed the same way: abstain instead of crashing,
with a conservative safety margin above the ~20-60ms window lengths
observed to trigger it.

REAL BUG found and fixed 2026-09-09 (second instance, different
trigger), via a real user-supplied ~4-minute MP3 conversation recording
uploaded through the live dashboard: the SAME crash
(`RuntimeError: cannot reshape tensor of 0 elements...`) recurred even
though the WHOLE buffer was comfortably longer than `_MIN_DURATION_S`
below — because this time the zero-length slice happens INSIDE
Whisper's own internal `transcribe()` seek loop, on one of the many
~30-second segments it internally splits a long recording into, not on
our own windowing.py's trailing partial window. `_MIN_DURATION_S` only
guards the length of the WHOLE buffer passed in; it can't guard against
whatever real-audio characteristic (a silence gap landing on an
internal segment boundary, in this case) makes ONE of Whisper's own
internal segments zero-length. Rather than trying to out-guess
Whisper's internal segmentation, the fix is the same "best-effort,
never break the whole request" guarantee the LIVE path's
`_transcribe()` already gives (see app/pipeline/live_transcription.py)
but the file-upload path (`Engine.score_call()`'s synchronous call
here) did not: catch ANY exception from the real model call and fall
back to an empty transcript, same as "no speech found" — transcription
failing is always recoverable (the rest of the pipeline still scores
the call, just without transcript-derived signals), a crash never is.
"""
from __future__ import annotations

from math import gcd

import numpy as np

from app.domain.models import TranscriptResult
from app.logging_setup import get_logger

logger = get_logger(component="whisper_transcriber")

_MODEL_SAMPLE_RATE = 16_000
_DOWNLOAD_ROOT = "app/adapters/transcription/.cache"

# See the REAL BUG docstring note above: a window this short can hit a
# zero-frame sequence somewhere inside Whisper's own framing, which
# crashes attention's reshape rather than producing an empty transcript.
# 100ms is a conservative safety margin above the ~20-60ms window
# lengths observed to trigger it — comfortably below any window length
# that could carry a recognisable word, so nothing usable is lost.
_MIN_DURATION_S = 0.1


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

        duration_s = samples.size / sample_rate if sample_rate else 0.0
        if duration_s < _MIN_DURATION_S:
            # See the REAL BUG note above the class docstring: Whisper's own
            # framing crashes (uncaught RuntimeError, an actual HTTP 500) on
            # a window this short — this is not hypothetical, a real
            # trailing partial window from windowing.py triggered it.
            return TranscriptResult(text="", language=None, detector_name=self.name, detector_version=self.version)

        if sample_rate != _MODEL_SAMPLE_RATE:
            samples = _resample(samples, sample_rate, _MODEL_SAMPLE_RATE)

        # temperature=0.0 (a single float, not Whisper's default fallback
        # tuple) forces pure greedy decoding — see this module's
        # REPRODUCIBILITY FIX docstring note for why the default is
        # genuinely non-deterministic on non-speech audio.
        try:
            result = self._model.transcribe(samples.astype(np.float32), fp16=False, temperature=0.0)
        except Exception:  # noqa: BLE001 — see the second REAL BUG note above the
            # class docstring: Whisper's own internal segmentation can hit a
            # zero-length slice on real, long audio in a way _MIN_DURATION_S
            # above can't guard against. Best-effort, same as every other
            # "abstain rather than crash" detector in this project —
            # transcription failing is recoverable, a crash isn't.
            logger.exception("whisper_transcription_failed", duration_s=duration_s)
            return TranscriptResult(text="", language=None, detector_name=self.name, detector_version=self.version)

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

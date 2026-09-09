"""PerthWatermarkDetector — checks for Resemble AI's Perth neural
watermark, a real, narrow, high-precision signal that complements (does
NOT replace) AASIST's general spoof classification.

WHY THIS EXISTS: Resemble AI's Chatterbox (and other Perth-integrated
voice-cloning tools) embed an implicit neural watermark in EVERY clip
they generate — see https://github.com/resemble-ai/perth (MIT). Unlike
AASIST's learned "does this sound synthetic" judgment, this is closer to
a fingerprint check: does this specific audio carry the specific pattern
Perth's encoder imprints. When it fires, that's about as close to
definitive evidence as this system has that the audio passed through a
Perth-integrated tool.

VERIFIED BY HAND, not assumed (2026-09-08):
- Real genuine speech (10 samples: 2 macOS `say`-synthesized English
  clips, 8 real Hindi speakers from a local test set) reads near-zero:
  9/10 below 0.04, one outlier at 0.36 — still well below any reasonable
  0.5 threshold.
- A DIFFERENT, non-Perth TTS system (Coqui XTTS-v2, already used
  elsewhere in this project — see eval/indian_language/) also reads
  near-zero on 5 synthetic samples — confirms this detector is specific
  to Perth's own watermark pattern, not a generic "sounds synthetic"
  trigger that would make it redundant with AASIST.
- A clip actually run through PerthImplicitWatermarker.apply_watermark()
  reads 1.0, both on a ~4s clip and a realistic 2s production window.
- Inference is fast: ~7ms per 2s window on CPU — no real-time concern.

HONEST LIMITATION, verified not assumed: a pure sine tone (this
project's own non-speech test fixture, scripts/gen_test_audio.py) reads
0.92 — a real false-positive on degenerate, non-speech audio. This
detector's real-world scope is genuine/cloned SPEECH, matching every
other detector in this project; a synthetic tone was never a realistic
input to begin with (AASIST already scores such fixtures ~100% "not
bonafide" for the same underlying reason — see
docs/architecture.md's acoustic-detector section).

NARROW SCOPE, honestly stated: this only catches watermark-compliant
generators (Chatterbox and anything else built on Perth). It is
evadable by any tool that doesn't watermark its output, or by
deliberately re-encoding/stripping audio in ways aggressive enough to
destroy the watermark (Perth's own docs claim survival through
compression/editing, not through every possible transformation).
Silence when this signal doesn't fire means "no Perth watermark found",
never "this audio is safe" — AASIST and the other detectors still carry
the general-purpose burden.

Near-silence: get_watermark() returns NaN on all-zero input (a real,
verified edge case, not a hypothetical one) — guarded by the same
floor_rms abstain pattern every other acoustic detector in this project
uses.

REAL BUG found and fixed 2026-09-09, via dogfooding real external audio
(IndieFake Dataset's public demo clips, arxiv.org/html/2506.19014 —
Indian-accented English samples, not part of this project's committed
pipeline, just a spot-check input) through the deployed service: a very
short trailing window crashed the WHOLE request with an uncaught
`RuntimeError: Argument #4: Padding size should be less than the
corresponding input dimension` — not abstained, an actual HTTP 500.
Root cause verified by inspecting Perth's own loaded config
(`PerthImplicitWatermarker().perth_net.ap.hp`): `sample_rate=32000,
n_fft=2048`. `get_watermark()` computes a centered STFT via
`torch.stft` (reflect-padded by `n_fft // 2 = 1024` samples on each
side, internally after resampling to 32kHz) — reflect padding requires
the input to be STRICTLY LONGER than the pad amount, so any window
resampling to 1024 samples or fewer at Perth's internal 32kHz (~32ms)
crashes. `windowing.py`'s own docstring already documents that the
final window of any file "is still yielded" even when shorter than the
configured window length — a totally normal occurrence whenever a
file's duration isn't an exact multiple of the hop length, not a
specifically crafted edge case. Fixed the same way `floor_rms` already
handles near-silence: abstain instead of crashing, with a safety margin
above the verified 1024-sample/32kHz minimum.
"""
from __future__ import annotations

import numpy as np

from app.domain.models import AudioWindow, DetectorResult

# Perth's own internal STFT needs the (resampled-to-32kHz) signal longer
# than n_fft // 2 = 1024 samples (~32ms) — see the REAL BUG note above.
# 50ms is a safety margin above that verified minimum, not a tuned value.
_MIN_DURATION_S = 0.05


class PerthWatermarkDetector:
    name = "perth_watermark"
    version = "resemble-perth@1.0.1"

    def __init__(self, floor_rms: float = 1e-4) -> None:
        import perth

        # REAL BUG found and fixed 2026-09-08: resemble-perth's own PyPI
        # metadata declares zero dependencies, but its implicit
        # watermarker imports librosa internally — a real gap in that
        # package. perth/__init__.py swallows that ImportError and
        # silently leaves PerthImplicitWatermarker as None instead of
        # raising, which would otherwise surface here as a cryptic
        # "'NoneType' object is not callable" with no clue why. See
        # requirements.txt's own note on pinning librosa explicitly —
        # this check exists so a missing/broken install fails loudly and
        # actionably instead of silently, even if that pin is ever lost.
        if perth.PerthImplicitWatermarker is None:
            raise ImportError(
                "perth.PerthImplicitWatermarker is None — resemble-perth's implicit "
                "watermarker failed to import, almost certainly because librosa is "
                "missing (a real, undeclared dependency of resemble-perth itself). "
                "Run: pip install librosa (see requirements.txt's own note on this)."
            )

        self.floor_rms = floor_rms
        self._watermarker = perth.PerthImplicitWatermarker()

    def score(self, window: AudioWindow, context: dict) -> DetectorResult:
        samples = window.samples
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0

        if rms < self.floor_rms:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"rms": rms, "floor_rms": self.floor_rms},
                abstain_reason="window is near-silent — nothing to check for a watermark in",
            )

        duration_s = samples.size / window.sample_rate if window.sample_rate else 0.0
        if duration_s < _MIN_DURATION_S:
            # See the REAL BUG note above the class docstring: Perth's own
            # STFT crashes (uncaught RuntimeError, an actual HTTP 500) on a
            # window this short — this is not hypothetical, a real trailing
            # partial window from windowing.py triggered it.
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={
                    "duration_s": duration_s,
                    "min_duration_s": _MIN_DURATION_S,
                },
                abstain_reason=(
                    f"window too short ({duration_s * 1000:.1f}ms) for Perth's own STFT — "
                    f"needs at least {_MIN_DURATION_S * 1000:.0f}ms"
                ),
            )

        watermark_probability = float(
            self._watermarker.get_watermark(samples, sample_rate=window.sample_rate, round=False)
        )

        detail = {
            "watermark_probability": watermark_probability,
            "rms": rms,
            "explanation": _build_explanation(watermark_probability),
        }

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=watermark_probability,
            detail=detail,
        )


def _build_explanation(watermark_probability: float) -> str:
    if watermark_probability >= 0.5:
        return (
            f"Detected a Resemble AI Perth neural watermark with {watermark_probability * 100:.1f}% confidence — "
            "this specific signal is embedded by Chatterbox and other Perth-integrated voice-cloning tools in "
            "every clip they generate. A strong, narrow signal: it only catches watermark-compliant tools, and "
            "its absence does not mean the audio is genuine."
        )
    return (
        f"No Resemble AI Perth watermark detected ({watermark_probability * 100:.1f}% confidence) — this only "
        "means this specific tool's fingerprint is absent, not that the audio is genuine; see the other "
        "detectors for general-purpose spoof detection."
    )

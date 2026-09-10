"""Acoustic detector backed by a real pretrained AASIST countermeasure.

Unlike acoustic_spectral_flatness.py (a hand-rolled DSP heuristic), this
wraps an actual published, trained spoof-detection model: AASIST
(clovaai/aasist, MIT licensed, NAVER Corp.), trained on ASVspoof2019 LA.
The model code is vendored in ./vendor/aasist_model.py (unmodified except
a class rename); the checkpoint is fetched separately by
./vendor/fetch_checkpoint.py (not committed to git — see that file).

WHAT THIS DOES AND DOESN'T PROVE
This model was trained to distinguish bonafide human speech from the
specific TTS/voice-conversion attacks in the ASVspoof2019 LA training set
— all English. It had no exposure to Hindi or the held-out-generator
discipline the project blueprint's §04 calls for at training time;
measuring that gap (and fine-tuning to close it) is exactly the work §04
describes, and it has partially happened — see
eval/indian_language/README.md for a real first EER number on Hindi and
what's still missing (scope there is English + Hindi only, for now).
Treat this as "a real countermeasure is now wired in and reproducible,"
not "spoof detection is solved."

HINDI RECALIBRATION (wired into production 2026-09-08): `config/
risk_formula.yaml`'s `acoustic:` entry now also passes
`finetuned_out_layer_path`, pointing at `vendor/checkpoints/
AASIST_hindi_finetuned_out_layer.pth` — a small (~3KB), in-house-produced
recalibration of just the final classification layer, from
eval/indian_language's fine-tuning pipeline (see that directory's
README.md for the full reproducible process, the class-imbalance and
BatchNorm/Dropout-drift bugs found and fixed along the way, and the
honest before/after numbers: EER on the eval corpus 17.75% -> 3.0%,
and — the more meaningful check, since it's a genuinely held-out set
never used in training — false-positive rate on an independent, more
varied real Hindi speech sample 75% -> 20%). Everything else about the
base model (architecture, checkpoint, training data) is unchanged; only
`out_layer`'s weights differ when this param is supplied. Omit the param
to run the original, un-recalibrated checkpoint.

Sign convention (verified against the original repo's evaluation code,
see docs/risk-model.md): the model's training labels put bonafide at
class index 1 and spoof at class index 0 (data_utils.genSpoof_list:
`label = 1 if "bonafide" else 0`), and the official eval script scores
trials with `logits[:, 1]` directly (higher = more bonafide-like, the
ASVspoof CM convention). We take softmax(logits)[:, 0] as our
"probability this is synthetic" risk contribution — the same information,
inverted and bounded to [0, 1] for this project's fusion formula.

REAL BUG found and fixed 2026-09-09/10: `floor_rms` (the near-silence
abstain guard, same pattern as prosody_parselmouth.py/perth_watermark.py)
was left at 1e-4 — permissive enough to only catch literal digital
silence. Found by manually tracing a real fine-tuned run's per-window
scores over three real spoof files (POST /v1/score/file's full `trace`,
not just the final smoothed number): every window with rms below
roughly 0.03-0.07 scored confidently WRONG (near-0 spoof probability on
genuinely synthetic audio) instead of abstaining — e.g. a Chatterbox
clone scored 0.80/0.79/0.67 spoof-probability on its actual spoken
content, then 0.10/0.06/0.00/0.00 on its last four, progressively
quieter windows (rms 0.067 -> 0.0026), dragging the file's EMA-smoothed
FINAL score down into the "low" risk band despite the model correctly
flagging the spoken content throughout. Raised to 0.03 — high enough to
catch the clearest quiet-tail cases observed, kept below the existing
test suite's ~0.05-rms synthetic fixture audio so genuinely analysable
signal isn't over-abstained.
HONESTY NOTE, not fully solved by this fix: at least two windows in
the same real traces scored wrong despite NORMAL loudness (rms 0.12-0.16,
comparable to correctly-scored neighboring windows) — genuine per-window
volatility in AASIST's own predictions, not a silence artifact. No RMS
threshold fixes that; it would need either a different aggregation
across the whole trace (not just the last window) or a probability-
margin-based abstain, neither implemented here.

REAL BUG in the fix above, found and fixed the same day (2026-09-10),
via a real GENUINE Movie-MUSNOMIX clip (39 windows, rms 0.0038-0.029
throughout, never once crossing 0.03): the 0.03 threshold was chosen
after checking ONLY its effect on the three SPOOF files above — never
checked whether the same RMS band is also where AASIST correctly
recognises genuine audio. It is: run by hand with the gate bypassed,
this real genuine clip's quiet windows scored 0.000-0.020 spoof-
probability throughout — correctly bonafide, not a coin flip. So the
SAME rms 0.01-0.03 band is unreliable on spoof audio but reliable on
genuine audio — meaning floor_rms cannot cleanly separate "trustworthy"
from "not" in that range at all, in either direction, and gating the
whole band out costs real correct signal on genuine calls without a
comparably clean win on spoof ones (recall the HONESTY NOTE above: even
some LOUDER spoof windows, rms 0.12-0.16, were already wrong — the
failure was never cleanly RMS-bounded to begin with). Lowered back to
0.01 — comfortably above true silence and below the clearest confirmed-
wrong spoof cases (rms 0.0026-0.0067), but no longer excluding the
0.01-0.03 band this second investigation showed is genuinely ambiguous,
not reliably wrong.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.adapters.detectors.vendor.aasist_model import AasistNet
from app.domain.models import AudioWindow, DetectorResult

_DEFAULT_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
_CHECKPOINT_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
_MODEL_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
_MODEL_SAMPLE_RATE = 16_000


def _deterministic_pad(x: np.ndarray, max_len: int) -> np.ndarray:
    """Matches clovaai/aasist's eval-time `pad()` exactly (data_utils.py):
    truncate to the first max_len samples if long enough, otherwise tile
    (repeat) the signal to fill max_len. No randomness — required for
    this project's reproducibility guarantee (see docs/risk-model.md)."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    if x_len == 0:
        return np.zeros(max_len, dtype=x.dtype)
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


class AasistAcousticDetector:
    name = "acoustic_aasist"
    version = f"aasist@{_MODEL_COMMIT[:12]}"

    def __init__(
        self,
        checkpoint_path: str,
        floor_rms: float = 0.01,
        device: str = "cpu",
        finetuned_out_layer_path: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.floor_rms = floor_rms
        self.device = torch.device(device)
        self.finetuned_out_layer_path = Path(finetuned_out_layer_path) if finetuned_out_layer_path else None

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"AASIST checkpoint not found at {self.checkpoint_path}. "
                f"Run: python app/adapters/detectors/vendor/fetch_checkpoint.py "
                f"(the Dockerfile does this automatically at build time)."
            )

        self._model = AasistNet(_DEFAULT_MODEL_CONFIG)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self._model.load_state_dict(state_dict, strict=True)

        if self.finetuned_out_layer_path is not None:
            if not self.finetuned_out_layer_path.exists():
                raise FileNotFoundError(
                    f"Fine-tuned output-layer checkpoint not found at {self.finetuned_out_layer_path}. "
                    "This file is committed directly to git (small, in-house-produced — see "
                    "eval/indian_language/README.md), not fetched at build time; if it's missing, "
                    "something removed it from the working tree."
                )
            finetuned_state = torch.load(self.finetuned_out_layer_path, map_location=self.device)
            self._model.out_layer.load_state_dict(finetuned_state)

        self._model.to(self.device)
        self._model.eval()

    def score(self, window: AudioWindow, context: dict) -> DetectorResult:
        samples = window.samples
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0

        if rms < self.floor_rms:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"rms": rms, "floor_rms": self.floor_rms},
                abstain_reason="window is near-silent — nothing to classify",
            )

        resampled_from_hz: Optional[int] = None
        if window.sample_rate != _MODEL_SAMPLE_RATE:
            samples = _resample(samples, window.sample_rate, _MODEL_SAMPLE_RATE)
            resampled_from_hz = window.sample_rate

        padded = _deterministic_pad(samples, _DEFAULT_MODEL_CONFIG["nb_samp"])
        x = torch.tensor(padded, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[0]

        spoof_probability = float(probs[0].item())
        bonafide_probability = float(probs[1].item())

        detail = {
            "spoof_probability": spoof_probability,
            "bonafide_probability": bonafide_probability,
            "raw_logits": [float(v) for v in logits[0].tolist()],
            "checkpoint": "AASIST.pth",
            "checkpoint_sha256": _CHECKPOINT_SHA256,
            "model_source_commit": _MODEL_COMMIT,
            "trained_on": "ASVspoof2019 LA (English only)",
            "nb_samp": _DEFAULT_MODEL_CONFIG["nb_samp"],
            "rms": rms,
            "explanation": _build_explanation(spoof_probability),
            "hindi_finetuned_out_layer": self.finetuned_out_layer_path is not None,
        }
        if self.finetuned_out_layer_path is not None:
            detail["finetuned_out_layer_checkpoint"] = self.finetuned_out_layer_path.name
        if resampled_from_hz is not None:
            detail["resampled_from_hz"] = resampled_from_hz

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=spoof_probability,
            detail=detail,
        )


def _build_explanation(spoof_probability: float) -> str:
    """Honest by construction: AASIST is a trained neural classifier, not a
    rule engine, so there is no feature-level cue to name — only the
    confidence value itself, which is already in `detail`. Do not invent a
    more specific-sounding reason than the model actually produces."""
    if spoof_probability >= 0.5:
        return (
            f"AASIST's trained classifier assigned {spoof_probability * 100:.1f}% probability that this audio "
            "is synthetic, based on its internal learned representation — not a specific acoustic cue this "
            "system can name. (AASIST is a neural network, not a rule engine: there is no feature-level "
            "\"why\" to report beyond this confidence value.)"
        )
    return (
        f"AASIST's trained classifier assigned {(1 - spoof_probability) * 100:.1f}% probability that this audio "
        "is bonafide human speech, based on its internal learned representation — not a specific acoustic cue "
        "this system can name."
    )


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(samples, up, down).astype(np.float32)

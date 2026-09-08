"""Prosodic detector — Parselmouth (Praat) upgrade of prosody_pitch_variance.py.

WHAT'S ACTUALLY REAL HERE vs. prosody_pitch_variance.py's autocorrelation
pitch estimate: jitter, shimmer, and harmonics-to-noise ratio (HNR) are
computed by Praat's own validated acoustic-phonetic algorithms (via
Parselmouth, the official Python binding — GPLv3, source
https://github.com/YannickJadoul/Parselmouth), not a hand-rolled DSP
formula. These are the standard voice-quality measures used in clinical
voice-pathology research (Praat itself is the long-standing reference tool
in that field) — a meaningfully more validated FRONT END than the previous
autocorrelation coefficient-of-variation heuristic.

HONESTY NOTE (same spirit as prosody_pitch_variance.py and
voiceprint_consistency.py's caveats — do not let "uses Praat" read as "is a
trained spoof classifier"): this is still a heuristic SCORE MAPPING, not a
trained classifier. The three thresholds below (`_JITTER_FLOOR`,
`_SHIMMER_FLOOR`, `_HNR_CEILING`) encode one specific, defensible-but-
unvalidated hypothesis — that unnaturally LOW jitter/shimmer and
unnaturally HIGH HNR (a voice that is "too smooth/too clean") is
prosodically synthetic-suspicious — mirroring the same "too regular is
suspicious" direction prosody_pitch_variance.py already used for pitch
variance. It has not been validated against a labeled genuine-vs-synthetic
corpus. Replace the score mapping with a trained model behind this same
DetectorPort before trusting the output in a real deployment — the
underlying jitter/shimmer/HNR features Praat computes are the validated
part; the risk thresholds are not.
"""
from __future__ import annotations

import threading

import numpy as np

from app.domain.models import AudioWindow, DetectorResult

# Praat's C++ core predates any concept of being called from multiple
# threads at once (it was originally a single-threaded desktop app), and
# Parselmouth doesn't document it as thread-safe. Now that
# app/api/http_router.py and ws_router.py run detectors via
# run_in_threadpool (so one slow request doesn't block the whole async
# event loop), concurrent calls into Praat are a real possibility — this
# lock serialises them process-wide. Cheap: one _measure_voice_quality()
# call is tens of milliseconds.
_PRAAT_LOCK = threading.Lock()

_MIN_PITCH_HZ = 70.0
_MAX_PITCH_HZ = 400.0

# Placeholder "healthy/natural voice" reference points — see HONESTY NOTE.
# Clinical voice-quality literature commonly cites ~1% jitter and ~3-4%
# shimmer as typical upper bounds for a healthy natural voice; we use those
# as the floor below which we call a voice "unnaturally smooth".
_JITTER_FLOOR = 0.010  # 1.0% local jitter
_SHIMMER_FLOOR = 0.035  # 3.5% local shimmer
_HNR_CEILING = 20.0  # dB; natural conversational speech rarely exceeds this


class ParselmouthProsodyDetector:
    """Praat-derived jitter/shimmer/HNR, mapped to a heuristic risk score."""

    name = "prosody_parselmouth"
    version = "0.1.0-praat"

    def __init__(
        self,
        floor_rms: float = 1e-4,
        jitter_floor: float = _JITTER_FLOOR,
        shimmer_floor: float = _SHIMMER_FLOOR,
        hnr_ceiling: float = _HNR_CEILING,
    ) -> None:
        self.floor_rms = floor_rms
        self.jitter_floor = jitter_floor
        self.shimmer_floor = shimmer_floor
        self.hnr_ceiling = hnr_ceiling

    def score(self, window: AudioWindow, context: dict) -> DetectorResult:
        samples = window.samples
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        if rms < self.floor_rms:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"rms": rms},
                abstain_reason="window is near-silent — nothing to analyse",
            )

        try:
            jitter, shimmer, hnr_db = _measure_voice_quality(samples, window.sample_rate)
        except Exception as exc:  # noqa: BLE001 — Praat raises on "no voiced frames" etc.
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"rms": rms},
                abstain_reason=f"Praat could not extract voice-quality measures: {exc}",
            )

        if any(np.isnan(v) for v in (jitter, shimmer, hnr_db)):
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"rms": rms},
                abstain_reason="no reliably-voiced pitch periods found in this window",
            )

        jitter_risk = np.clip((self.jitter_floor - jitter) / self.jitter_floor, 0.0, 1.0)
        shimmer_risk = np.clip((self.shimmer_floor - shimmer) / self.shimmer_floor, 0.0, 1.0)
        hnr_span = 30.0 - self.hnr_ceiling
        hnr_risk = np.clip((hnr_db - self.hnr_ceiling) / hnr_span, 0.0, 1.0) if hnr_span > 0 else 0.0

        raw_score = float(np.clip((jitter_risk + shimmer_risk + hnr_risk) / 3.0, 0.0, 1.0))

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={
                "jitter_local": jitter,
                "shimmer_local": shimmer,
                "hnr_db": hnr_db,
                "jitter_floor_reference": self.jitter_floor,
                "shimmer_floor_reference": self.shimmer_floor,
                "hnr_ceiling_reference": self.hnr_ceiling,
                "rms": rms,
                "explanation": _build_explanation(
                    jitter, shimmer, hnr_db, jitter_risk, shimmer_risk, hnr_risk,
                    self.jitter_floor, self.shimmer_floor, self.hnr_ceiling,
                ),
            },
        )


def _build_explanation(
    jitter: float, shimmer: float, hnr_db: float,
    jitter_risk: float, shimmer_risk: float, hnr_risk: float,
    jitter_floor: float, shimmer_floor: float, hnr_ceiling: float,
) -> str:
    """Names whichever of the three sub-scores actually contributed, with
    the real measured value next to its reference — every clause here is
    a fact already in `detail`, not a new inference."""
    candidates = [
        (jitter_risk, f"jitter is unusually low ({jitter * 100:.2f}% vs. a {jitter_floor * 100:.1f}% natural-voice reference)"),
        (shimmer_risk, f"shimmer is unusually low ({shimmer * 100:.2f}% vs. a {shimmer_floor * 100:.1f}% natural-voice reference)"),
        (hnr_risk, f"harmonics-to-noise ratio is unusually high ({hnr_db:.1f}dB vs. a {hnr_ceiling:.0f}dB natural-voice reference)"),
    ]
    contributing = [desc for risk, desc in candidates if risk > 0.0]
    if not contributing:
        return "Voice quality (jitter, shimmer, HNR) is within the natural-voice reference range — no prosodic risk factors detected."
    return (
        "Elevated because " + "; ".join(contributing) + " — an unusually smooth/regular voice can indicate "
        "synthetic speech, though this jitter/shimmer/HNR-to-risk mapping is an unvalidated heuristic, not a "
        "trained classifier (see this detector's HONESTY NOTE)."
    )


def _measure_voice_quality(samples: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """Returns (jitter_local, shimmer_local, hnr_mean_db) via Praat, called
    through Parselmouth's `praat.call` bridge — the same commands Praat's
    own "Voice report" script uses."""
    import parselmouth
    from parselmouth.praat import call

    with _PRAAT_LOCK:
        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate)

        point_process = call(sound, "To PointProcess (periodic, cc)", _MIN_PITCH_HZ, _MAX_PITCH_HZ)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, _MIN_PITCH_HZ, 0.1, 1.0)
        hnr_mean_db = call(harmonicity, "Get mean", 0, 0)

    return float(jitter_local), float(shimmer_local), float(hnr_mean_db)

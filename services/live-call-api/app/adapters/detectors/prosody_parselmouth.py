"""Prosodic detectors — Parselmouth (Praat) upgrade of prosody_pitch_variance.py.

WHAT'S ACTUALLY REAL HERE vs. prosody_pitch_variance.py's autocorrelation
pitch estimate: jitter, shimmer, and harmonics-to-noise ratio (HNR) are
computed by Praat's own validated acoustic-phonetic algorithms (via
Parselmouth, the official Python binding — GPLv3, source
https://github.com/YannickJadoul/Parselmouth), not a hand-rolled DSP
formula. These are the standard voice-quality measures used in clinical
voice-pathology research (Praat itself is the long-standing reference tool
in that field) — a meaningfully more validated FRONT END than the previous
autocorrelation coefficient-of-variation heuristic. Both classes below
share that same extraction (`_extract_features`); they differ only in
how the three numbers are turned into a risk score.

2026-09-09: trained-classifier upgrade. `ParselmouthProsodyDetector`'s own
HONESTY NOTE always said its three hand-tuned thresholds were an
unvalidated hypothesis, to be replaced once real labeled data existed.
`eval/indian_language/scripts/calibrate_prosodic_thresholds.py` finally
ran that validation against 1423 genuine + 340 spoof Hindi examples:
`ParselmouthProsodyDetector`'s thresholds scored exactly 50% balanced
accuracy — chance level — on both trained-on and held-out data (100%
genuine accuracy, 0% spoof accuracy: they never fire against real audio).
`ParselmouthProsodyMLDetector` below is the fix: the SAME three Praat
features, combined by a logistic regression instead of three independent
floors/ceiling, reaching 82.5% balanced accuracy on genuinely held-out
data (Chatterbox-cloned spoof + local speakers' own reserved test clips,
neither used in fitting) — see `results/prosodic_calibration.json` and
README.md's "Prosodic detector calibration" section for the full numbers
and honest caveats (140 held-out examples is still a calibration-scale
sample; the learned shimmer coefficient runs opposite
`ParselmouthProsodyDetector`'s "low shimmer is suspicious" hypothesis on
this specific cloning system's artifacts). The trained model's
coefficients are hardcoded as plain constants below (sigmoid of a linear
combination on standardized features) rather than loading a serialized
sklearn model — this project has no other runtime dependency on
scikit-learn, and the whole computation is 3 numbers' worth of
arithmetic, which keeps it traceable to plain arithmetic the same way
run_held_out_eval.py's EER computation is (see that script's own
comment) rather than adding a new production dependency for it.
`ParselmouthProsodyDetector` is kept, unchanged, not deleted — same
"kept rather than deleted so this doesn't happen again" discipline as
every other REAL BUG note in this codebase.
"""
from __future__ import annotations

import numpy as np

from app.adapters.praat_lock import PRAAT_LOCK
from app.domain.models import AudioWindow, DetectorResult

_MIN_PITCH_HZ = 70.0
_MAX_PITCH_HZ = 400.0

# Praat's own pitch-period extraction needs at least 3 periods of
# _MIN_PITCH_HZ visible in the window (3 / 70Hz ≈ 42.9ms) — a real,
# verified minimum, not a guess: caught 2026-09-09 via a real external
# deepfake sample (IndieFake Dataset's public demo clips) whose trailing
# window was ~26.7ms; the exception message read "minimum pitch must not
# be less than 112.5Hz" (3 / 112.5Hz ≈ 26.7ms — Praat naming the pitch
# floor THIS window's actual length would have supported). Same root
# cause class as the short-window crash already found and fixed in
# perth_watermark.py that same day — here it was already caught (the
# broad except below), just with an unfriendly raw Praat message leaking
# through instead of a clean abstain reason. 60ms is a safety margin
# above the verified 42.9ms minimum.
_MIN_DURATION_S = 0.06

# Placeholder "healthy/natural voice" reference points for
# ParselmouthProsodyDetector — see that class's own HONESTY NOTE, and the
# module docstring's "2026-09-09: trained-classifier upgrade" note on why
# ParselmouthProsodyMLDetector exists instead of just retuning these.
# Clinical voice-quality literature commonly cites ~1% jitter and ~3-4%
# shimmer as typical upper bounds for a healthy natural voice; we use those
# as the floor below which we call a voice "unnaturally smooth".
_JITTER_FLOOR = 0.010  # 1.0% local jitter
_SHIMMER_FLOOR = 0.035  # 3.5% local shimmer
_HNR_CEILING = 20.0  # dB; natural conversational speech rarely exceeds this

# ParselmouthProsodyMLDetector's trained parameters — a StandardScaler +
# LogisticRegression(class_weight="balanced") fit on jitter/shimmer/HNR,
# see eval/indian_language/scripts/calibrate_prosodic_thresholds.py and
# results/prosodic_calibration.json (2026-09-09 run) for exactly how.
# Order throughout: (jitter_local, shimmer_local, hnr_db).
_ML_SCALER_MEAN = (0.022166341806446492, 0.09576296062763698, 13.469464679917822)
_ML_SCALER_SCALE = (0.005116343520411652, 0.02495170502399299, 3.1039219121944606)
_ML_COEF = (0.3018977933589121, -0.31844813946535616, 0.03738590949551451)
_ML_INTERCEPT = -0.02643924021858421


def _extract_features(window: AudioWindow, floor_rms: float) -> tuple[dict, str | None]:
    """Shared by both detector classes below: the real Praat extraction
    plus every abstain check (near-silent, too-short, Praat exception,
    unvoiced/NaN) — identical for both, since they differ only in how
    the extracted numbers become a risk score, not in when to abstain.
    Returns (detail_dict, abstain_reason) — abstain_reason is None on
    success, in which case detail_dict has "jitter_local"/"shimmer_local"/
    "hnr_db"/"rms"; on abstain, detail_dict has whatever was computed
    before the abstain point (at least "rms")."""
    samples = window.samples
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    if rms < floor_rms:
        return {"rms": rms}, "window is near-silent — nothing to analyse"

    duration_s = samples.size / window.sample_rate if window.sample_rate else 0.0
    if duration_s < _MIN_DURATION_S:
        return (
            {"rms": rms, "duration_s": duration_s, "min_duration_s": _MIN_DURATION_S},
            f"window too short ({duration_s * 1000:.1f}ms) for Praat's own pitch-period "
            f"extraction — needs at least {_MIN_DURATION_S * 1000:.0f}ms",
        )

    try:
        jitter, shimmer, hnr_db = _measure_voice_quality(samples, window.sample_rate)
    except Exception as exc:  # noqa: BLE001 — Praat raises on "no voiced frames" etc.
        return {"rms": rms}, f"Praat could not extract voice-quality measures: {exc}"

    if any(np.isnan(v) for v in (jitter, shimmer, hnr_db)):
        return {"rms": rms}, "no reliably-voiced pitch periods found in this window"

    return {"jitter_local": jitter, "shimmer_local": shimmer, "hnr_db": hnr_db, "rms": rms}, None


class ParselmouthProsodyDetector:
    """Praat-derived jitter/shimmer/HNR, mapped to a heuristic risk score.

    HONESTY NOTE (same spirit as prosody_pitch_variance.py and
    voiceprint_consistency.py's caveats — do not let "uses Praat" read as
    "is a trained spoof classifier"): this is a heuristic SCORE MAPPING,
    not a trained classifier. The three thresholds below (`_JITTER_FLOOR`,
    `_SHIMMER_FLOOR`, `_HNR_CEILING`) encode one specific, defensible-but-
    unvalidated hypothesis — that unnaturally LOW jitter/shimmer and
    unnaturally HIGH HNR (a voice that is "too smooth/too clean") is
    prosodically synthetic-suspicious. It was validated 2026-09-09 (see
    module docstring) and scored exactly chance level against real data —
    kept here, unchanged, as a documented historical baseline; see
    `ParselmouthProsodyMLDetector` below for what's actually deployed.
    """

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
        detail, abstain_reason = _extract_features(window, self.floor_rms)
        if abstain_reason is not None:
            return DetectorResult(
                detector_name=self.name, detector_version=self.version,
                score=None, detail=detail, abstain_reason=abstain_reason,
            )
        jitter, shimmer, hnr_db = detail["jitter_local"], detail["shimmer_local"], detail["hnr_db"]

        jitter_risk = np.clip((self.jitter_floor - jitter) / self.jitter_floor, 0.0, 1.0)
        shimmer_risk = np.clip((self.shimmer_floor - shimmer) / self.shimmer_floor, 0.0, 1.0)
        hnr_span = 30.0 - self.hnr_ceiling
        hnr_risk = np.clip((hnr_db - self.hnr_ceiling) / hnr_span, 0.0, 1.0) if hnr_span > 0 else 0.0

        raw_score = float(np.clip((jitter_risk + shimmer_risk + hnr_risk) / 3.0, 0.0, 1.0))

        detail.update(
            {
                "jitter_floor_reference": self.jitter_floor,
                "shimmer_floor_reference": self.shimmer_floor,
                "hnr_ceiling_reference": self.hnr_ceiling,
                "explanation": _build_explanation(
                    jitter, shimmer, hnr_db, jitter_risk, shimmer_risk, hnr_risk,
                    self.jitter_floor, self.shimmer_floor, self.hnr_ceiling,
                ),
            }
        )
        return DetectorResult(
            detector_name=self.name, detector_version=self.version, score=raw_score, detail=detail,
        )


class ParselmouthProsodyMLDetector:
    """Same Praat jitter/shimmer/HNR extraction as ParselmouthProsodyDetector
    (via the shared `_extract_features` helper), scored by a trained
    logistic regression instead of three independent hand-tuned
    thresholds — see the module docstring's "2026-09-09: trained-
    classifier upgrade" note for the real calibration numbers and honest
    caveats behind this. This is what `config/risk_formula.yaml`'s
    `prosodic:` entry actually points at as of that date.
    """

    name = "prosody_parselmouth_ml"
    version = "0.2.0-logreg"

    def __init__(self, floor_rms: float = 1e-4) -> None:
        self.floor_rms = floor_rms

    def score(self, window: AudioWindow, context: dict) -> DetectorResult:
        detail, abstain_reason = _extract_features(window, self.floor_rms)
        if abstain_reason is not None:
            return DetectorResult(
                detector_name=self.name, detector_version=self.version,
                score=None, detail=detail, abstain_reason=abstain_reason,
            )
        jitter, shimmer, hnr_db = detail["jitter_local"], detail["shimmer_local"], detail["hnr_db"]

        raw_features = (jitter, shimmer, hnr_db)
        standardized = [
            (v - mean) / scale for v, mean, scale in zip(raw_features, _ML_SCALER_MEAN, _ML_SCALER_SCALE)
        ]
        logit = sum(c * z for c, z in zip(_ML_COEF, standardized)) + _ML_INTERCEPT
        raw_score = float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid -> spoof probability

        detail["explanation"] = (
            f"A logistic regression trained on jitter/shimmer/HNR assigned {raw_score * 100:.1f}% "
            "spoof probability (82.5% balanced accuracy on genuinely held-out data — see "
            "eval/indian_language/README.md's \"Prosodic detector calibration\" section for the "
            "full numbers and honest caveats)."
        )
        return DetectorResult(
            detector_name=self.name, detector_version=self.version, score=raw_score, detail=detail,
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

    with PRAAT_LOCK:
        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate)

        point_process = call(sound, "To PointProcess (periodic, cc)", _MIN_PITCH_HZ, _MAX_PITCH_HZ)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, _MIN_PITCH_HZ, 0.1, 1.0)
        hnr_mean_db = call(harmonicity, "Get mean", 0, 0)

    return float(jitter_local), float(shimmer_local), float(hnr_mean_db)

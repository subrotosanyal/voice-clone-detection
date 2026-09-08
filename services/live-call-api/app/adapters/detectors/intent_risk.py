"""IntentRiskDetector — reads a PRE-COMPUTED zero-shot intent classification
out of `context`, does not call any model itself.

WHY PRE-COMPUTED, NOT COMPUTED HERE: this detector's score() runs once per
2-second window, same as every other DetectorPort. A zero-shot NLI
forward pass is expensive — calling it once per window would mean a
30-second call (15 windows) paying for 15 classifications of the SAME
transcript, the exact class of redundant-expensive-computation bug fixed
2026-09-08 for transcription-during-diarization (see http_router.py's
comment on that). Instead, Engine.score_call() (and http_router.py, for
the diarization fan-out) computes the classification ONCE per call and
merges `intent_label`/`intent_top_score`/`intent_label_scores` into
`context` before windowing — this detector only ever reads those,
matching the same "transcribe once, read many times" pattern already
established for `context["transcript"]`.

STATUS: this detector is real and correct, but is NOT present in
config/risk_formula.yaml's active `detectors:` list by default — see
app/adapters/intent/zero_shot_intent_classifier.py's HONESTY NOTE for the
real calibration problem found before shipping this live. Wiring this in
requires both an `intent_classification:` config section (to populate the
context fields this detector reads) AND an `intent` entry in `detectors:`.

RISK MAPPING: `raw_score = 1.0 - label_scores["ordinary conversation"]` —
deliberately reads only the negative class's own score, not "whichever
risky label won", so the mapping stays a single, named, reproducible
number rather than an arbitrary combination of the four risky candidates.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.models import AudioWindow, DetectorResult

_ORDINARY_LABEL = "ordinary conversation"


class IntentRiskDetector:
    name = "intent_risk"
    version = "0.1.0-zero-shot-mdeberta"

    def __init__(self, ordinary_label: str = _ORDINARY_LABEL) -> None:
        self.ordinary_label = ordinary_label

    def score(self, window: AudioWindow, context: dict[str, Any]) -> DetectorResult:
        label_scores = context.get("intent_label_scores")
        top_label = context.get("intent_label")
        if not label_scores or not top_label:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={},
                abstain_reason=(
                    "no transcript-derived intent classification available for this call "
                    "(needs both a transcriber and intent_classification configured)"
                ),
            )

        ordinary_score = float(label_scores.get(self.ordinary_label, 0.0))
        raw_score = float(np.clip(1.0 - ordinary_score, 0.0, 1.0))

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={
                "top_label": top_label,
                "top_score": context.get("intent_top_score"),
                "label_scores": label_scores,
                "explanation": _build_explanation(top_label, ordinary_score, label_scores, self.ordinary_label),
            },
        )


def _build_explanation(top_label: str, ordinary_score: float, label_scores: dict, ordinary_label: str) -> str:
    if top_label == ordinary_label:
        return (
            f"Zero-shot classification found this transcript most consistent with "
            f"'{ordinary_label}' ({ordinary_score * 100:.0f}%) — no fraud-relevant intent detected."
        )
    top_score = label_scores.get(top_label, 0.0)
    return (
        f"Zero-shot classification found this transcript most consistent with '{top_label}' "
        f"({top_score * 100:.0f}%), and only {ordinary_score * 100:.0f}% consistent with "
        f"'{ordinary_label}'."
    )

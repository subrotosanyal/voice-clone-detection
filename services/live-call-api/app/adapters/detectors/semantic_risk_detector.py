"""SemanticRiskDetector — reads a PRE-COMPUTED SemanticRiskAssessment out
of `context`, does not call the LLM itself.

WHY PRE-COMPUTED, NOT COMPUTED HERE: same reasoning as
app/adapters/detectors/intent_risk.py — this detector's score() runs
once per 2-second window, but a local LLM forward pass (~1-1.5s CPU,
measured by hand — see local_llm_semantic_classifier.py) is far too
expensive to repeat per window. Engine.score_call() computes the
assessment ONCE per call and merges `semantic_urgency_level`/
`semantic_financial_solicitation`/`semantic_authority_claim`/
`semantic_isolation_request`/`semantic_reasoning` into `context` before
windowing, matching the exact "compute once, read many times" pattern
already established for `context["transcript"]` and
`context["intent_label"]`.

RISK MAPPING, a starting point — NOT validated against a labeled corpus
(same honesty standard as every other heuristic mapping in this
project): `raw_score = max(urgency_level, flags_present / 3)`, where
flags_present counts however many of financial_solicitation/
authority_claim/isolation_request are True. Taking the MAX (not an
average) means either a high manufactured-urgency reading on its own,
OR enough concrete tactics present on their own, can drive the score —
deliberately NOT requiring both to agree, since a scam script can lean
on either style. This intentionally does NOT replicate
contextual_rules.py's own combined authority+financial "pressure rule"
— that rule works off literal keyword matches; this one works off the
LLM's semantic reading of the whole transcript, a genuinely different
(complementary, not competing) signal.

STATUS: see app/ports/semantic_risk_classifier.py's own STATUS note for
whether/how this is wired into the active formula as of this writing.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.models import AudioWindow, DetectorResult


class SemanticRiskDetector:
    name = "semantic_risk"
    version = "0.1.0-phi3-mini-semantic"

    def score(self, window: AudioWindow, context: dict[str, Any]) -> DetectorResult:
        urgency_level = context.get("semantic_urgency_level")
        if urgency_level is None:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={},
                abstain_reason=(
                    "no transcript-derived semantic risk assessment available for this call "
                    "(needs both a transcriber and semantic_risk_classification configured)"
                ),
            )

        financial_solicitation = bool(context.get("semantic_financial_solicitation", False))
        authority_claim = bool(context.get("semantic_authority_claim", False))
        isolation_request = bool(context.get("semantic_isolation_request", False))
        flags_present = sum([financial_solicitation, authority_claim, isolation_request])

        raw_score = float(np.clip(max(float(urgency_level), flags_present / 3.0), 0.0, 1.0))
        reasoning = context.get("semantic_reasoning", "")

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={
                "urgency_level": urgency_level,
                "financial_solicitation": financial_solicitation,
                "authority_claim": authority_claim,
                "isolation_request": isolation_request,
                "reasoning": reasoning,
                "explanation": _build_explanation(
                    urgency_level, financial_solicitation, authority_claim, isolation_request, reasoning
                ),
            },
        )


def _build_explanation(
    urgency_level: float,
    financial_solicitation: bool,
    authority_claim: bool,
    isolation_request: bool,
    reasoning: str,
) -> str:
    tactics = []
    if financial_solicitation:
        tactics.append("a financial request")
    if authority_claim:
        tactics.append("a claimed authority")
    if isolation_request:
        tactics.append("a request not to hang up or tell anyone")

    if not tactics and urgency_level < 0.3:
        return f"No social-engineering tactics detected by the local LLM's reading of this transcript. ({reasoning})"

    tactics_str = ", ".join(tactics) if tactics else "no specific tactic flagged"
    return (
        f"The local LLM's reading of this transcript found {tactics_str}, with a manufactured-urgency "
        f"level of {urgency_level * 100:.0f}%. ({reasoning})"
    )

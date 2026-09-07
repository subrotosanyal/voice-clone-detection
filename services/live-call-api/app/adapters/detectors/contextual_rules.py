"""Contextual detector — the cheap, no-enrollment "third signal" mode.

Purely rule-based: no audio, no ML, no enrollment store required. Reads
call metadata from `context` (a plain dict — see the field list below) and
returns a fully explainable score where every point comes from a named
rule that either fired or didn't. This is the lowest-effort of the two
"third signal" options described in the blueprint's §03 — see
consistency_stub.py for the other one.

Expected `context` fields (all optional; a missing field just doesn't
contribute):
  known_number: bool          — is the caller's number on file for the
                                 claimed identity?
  hour_of_day: int             — 0-23, local time of the call
  is_financial_request: bool   — is a transfer/payment being requested?
  urgency_keywords: list[str]  — words from a transcript indicating
                                 pressure/urgency (e.g. "immediately",
                                 "right now", "don't tell anyone")
"""
from __future__ import annotations

from typing import Any

from app.domain.models import AudioWindow, DetectorResult

_UNUSUAL_HOUR_START = 22  # 10pm
_UNUSUAL_HOUR_END = 7  # 7am


class ContextualRulesDetector:
    """Deterministic rule engine over call metadata."""

    name = "contextual_rules"
    version = "0.1.0"

    def __init__(
        self,
        weight_unknown_number: float = 0.30,
        weight_unusual_hour: float = 0.20,
        weight_financial_request: float = 0.30,
        weight_urgency_language: float = 0.20,
    ) -> None:
        self.weight_unknown_number = weight_unknown_number
        self.weight_unusual_hour = weight_unusual_hour
        self.weight_financial_request = weight_financial_request
        self.weight_urgency_language = weight_urgency_language

    def score(self, window: AudioWindow, context: dict[str, Any]) -> DetectorResult:
        if not context:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={},
                abstain_reason="no call metadata supplied for this session",
            )

        fired: dict[str, bool] = {}
        total = 0.0

        known_number = context.get("known_number")
        if known_number is False:
            fired["unknown_number"] = True
            total += self.weight_unknown_number
        elif known_number is True:
            fired["unknown_number"] = False

        hour = context.get("hour_of_day")
        if isinstance(hour, int):
            unusual = hour >= _UNUSUAL_HOUR_START or hour < _UNUSUAL_HOUR_END
            fired["unusual_hour"] = unusual
            if unusual:
                total += self.weight_unusual_hour

        is_financial = context.get("is_financial_request")
        if isinstance(is_financial, bool):
            fired["financial_request"] = is_financial
            if is_financial:
                total += self.weight_financial_request

        urgency_keywords = context.get("urgency_keywords") or []
        has_urgency = len(urgency_keywords) > 0
        fired["urgency_language"] = has_urgency
        if has_urgency:
            total += self.weight_urgency_language

        if not fired:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"context_keys_seen": list(context.keys())},
                abstain_reason="none of the recognised metadata fields were present",
            )

        raw_score = min(total, 1.0)
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={"rules_fired": fired, "urgency_keywords": urgency_keywords},
        )

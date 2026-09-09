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
                                 (manually supplied; auto-detection below
                                 can also set this)
  urgency_keywords: list[str]  — words a caller manually supplies as
                                 indicating pressure/urgency
  authority_claim: bool        — is the caller claiming to be a bank,
                                 police, or government official? (manually
                                 supplied; auto-detection below can also
                                 set this)
  transcript: str              — a real transcript (see
                                 app/adapters/transcription/
                                 whisper_transcriber.py), set automatically
                                 by Engine.score_call() when a transcriber
                                 is configured — not something a caller
                                 needs to supply by hand. Scanned for the
                                 same class of urgency/financial-request/
                                 authority-claim language via
                                 app/adapters/transcription/
                                 urgency_language.py, and merged with (not
                                 replacing) any manually-supplied
                                 urgency_keywords/is_financial_request/
                                 authority_claim.
"""
from __future__ import annotations

from typing import Any

from app.adapters.transcription.urgency_language import detect_urgency_signals
from app.domain.models import AudioWindow, DetectorResult

_UNUSUAL_HOUR_START = 22  # 10pm
_UNUSUAL_HOUR_END = 7  # 7am


class ContextualRulesDetector:
    """Deterministic rule engine over call metadata."""

    name = "contextual_rules"
    version = "0.2.0"

    def __init__(
        self,
        weight_unknown_number: float = 0.30,
        weight_unusual_hour: float = 0.20,
        weight_financial_request: float = 0.30,
        weight_urgency_language: float = 0.20,
        weight_authority_claim: float = 0.25,
        weight_combined_authority_financial_pressure: float = 0.25,
    ) -> None:
        self.weight_unknown_number = weight_unknown_number
        self.weight_unusual_hour = weight_unusual_hour
        self.weight_financial_request = weight_financial_request
        self.weight_urgency_language = weight_urgency_language
        self.weight_authority_claim = weight_authority_claim
        self.weight_combined_authority_financial_pressure = weight_combined_authority_financial_pressure

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

        # Auto-detected from a real transcript (if Engine.score_call() set
        # one — see this module's docstring), merged with anything manually
        # supplied, never replacing it. Every field below distinguishes the
        # two sources so a result stays traceable to either "the caller
        # said so" or "the transcript actually contains this word".
        transcript = context.get("transcript") or ""
        transcript_urgency_keywords, transcript_is_financial, transcript_authority_keywords = (
            detect_urgency_signals(transcript)
        )

        manual_is_financial = context.get("is_financial_request")
        is_financial = bool(manual_is_financial) or transcript_is_financial
        if manual_is_financial is not None or transcript_is_financial:
            fired["financial_request"] = is_financial
            if is_financial:
                total += self.weight_financial_request

        manual_urgency_keywords = context.get("urgency_keywords") or []
        urgency_keywords = list(manual_urgency_keywords)
        for phrase in transcript_urgency_keywords:
            if phrase not in urgency_keywords:
                urgency_keywords.append(phrase)
        has_urgency = len(urgency_keywords) > 0
        fired["urgency_language"] = has_urgency
        if has_urgency:
            total += self.weight_urgency_language

        # Authority claim — the caller asserting they're a bank, police, or
        # government official. Weaker evidence alone than the rules above
        # (a real bank call also says "this is your bank"); its real
        # weight comes from the combined rule right below.
        manual_authority_claim = context.get("authority_claim")
        is_authority_claim = bool(manual_authority_claim) or bool(transcript_authority_keywords)
        if manual_authority_claim is not None or transcript_authority_keywords:
            fired["authority_claim"] = is_authority_claim
            if is_authority_claim:
                total += self.weight_authority_claim

        # Combined pattern: an authority claim paired with a financial
        # request is the classic fraud script ("this is your bank, your
        # account has been compromised, transfer your funds now") — named
        # and weighted as its own explicit rule, on top of the two
        # component rules above, so this specific combination is visibly
        # flagged rather than silently folded into a bigger number.
        if is_authority_claim and is_financial:
            fired["combined_authority_financial_pressure"] = True
            total += self.weight_combined_authority_financial_pressure

        if not fired:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"context_keys_seen": list(context.keys())},
                abstain_reason="none of the recognised metadata fields were present",
            )

        raw_score = min(total, 1.0)
        detail: dict[str, Any] = {
            "rules_fired": fired,
            "urgency_keywords": urgency_keywords,
            "authority_keywords": transcript_authority_keywords,
        }
        if transcript:
            detail["transcript"] = transcript
            detail["urgency_keywords_manual"] = manual_urgency_keywords
            detail["urgency_keywords_from_transcript"] = transcript_urgency_keywords
            detail["is_financial_request_from_transcript"] = transcript_is_financial
            detail["authority_keywords_from_transcript"] = transcript_authority_keywords
            # Live-mic path only (app/pipeline/live_transcription.py) — the
            # naive whole-session concatenation, for display/audit only.
            # File-upload's `transcript` already covers the whole call, so
            # this is simply absent there. See that module's own docstring
            # for the honest overlap-duplication caveat.
            full_session_transcript = context.get("full_session_transcript")
            if full_session_transcript:
                detail["full_session_transcript"] = full_session_transcript
        detail["explanation"] = _build_explanation(
            fired, urgency_keywords, transcript_authority_keywords, hour if isinstance(hour, int) else None
        )
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail=detail,
        )


def _build_explanation(
    fired: dict[str, bool], urgency_keywords: list[str], authority_keywords: list[str], hour: int | None
) -> str:
    """Turns `fired` (+ the specific values that triggered it) into one
    plain-language sentence — every clause here names a concrete,
    reproducible fact already in `detail`, nothing inferred beyond it."""
    reasons: list[str] = []
    if fired.get("unknown_number"):
        reasons.append("the caller's number is not on file")
    if fired.get("unusual_hour"):
        reasons.append(f"the call was placed at an unusual hour ({hour:02d}:00)" if hour is not None else "the call was placed at an unusual hour")
    if fired.get("financial_request"):
        reasons.append("the request appears to be financial")
    if fired.get("urgency_language"):
        if urgency_keywords:
            quoted = ", ".join(f'"{kw}"' for kw in urgency_keywords)
            reasons.append(f"urgency language was detected ({quoted})")
        else:
            reasons.append("urgency language was detected")
    if fired.get("authority_claim"):
        if authority_keywords:
            quoted = ", ".join(f'"{kw}"' for kw in authority_keywords)
            reasons.append(f"the caller claimed authority ({quoted})")
        else:
            reasons.append("the caller claimed authority (bank/police/government)")
    if fired.get("combined_authority_financial_pressure"):
        reasons.append(
            "this is a particularly high-risk combination: an authority claim paired with a financial request"
        )
    if not reasons:
        return "No contextual risk factors detected: known number, ordinary hour, no financial, urgency, or authority-claim language found."
    return "Elevated because " + "; ".join(reasons) + "."

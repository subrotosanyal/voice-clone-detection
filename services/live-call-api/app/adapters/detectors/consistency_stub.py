"""Consistency detector — the other "third signal" mode. NOT YET WIRED UP.

This is a deliberate stub, not a shortcut: it always abstains. Implementing
it for real means adding an enrollment flow (record a voiceprint, extract
an embedding — the blueprint's §02 candidate is SpeechBrain's ECAPA-TDNN —
and store it, consented and TTL'd, per §06's ethics section) plus a lookup
here by claimed identity. Until that enrollment store exists, honestly
abstaining is the only correct behaviour — this is exactly the fallback
path config/risk_formula.yaml's `auto` mode exists to handle gracefully
(see app/adapters/fusion/weighted_sum.py).

When you do wire this up, look at the SASV Challenge 2022 baseline
(AASIST + ECAPA-TDNN score fusion) referenced in the blueprint's §03 as a
working reference implementation of this exact signal.
"""
from __future__ import annotations

from typing import Any

from app.domain.models import AudioWindow, DetectorResult


class ConsistencyStubDetector:
    """Always abstains — no voiceprint enrollment store exists yet."""

    name = "consistency_stub"
    version = "0.0.0-not-implemented"

    def score(self, window: AudioWindow, context: dict[str, Any]) -> DetectorResult:
        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=None,
            detail={"claimed_identity": context.get("claimed_identity")},
            abstain_reason=(
                "consistency mode is not implemented yet — no enrollment "
                "store exists to compare a live embedding against. Configure "
                "third_signal.mode: contextual (or auto, which falls back "
                "to contextual) until this is built."
            ),
        )

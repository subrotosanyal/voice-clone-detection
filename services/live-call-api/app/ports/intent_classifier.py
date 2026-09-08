"""The IntentClassifier port.

Classifies a call's transcript against a fixed, explicit list of
fraud-relevant candidate intents (see app/adapters/intent/
zero_shot_intent_classifier.py for the actual labels) — meant to
complement, not replace, the keyword-based urgency/financial-request/
authority-claim detection in app/adapters/transcription/urgency_language.py,
by generalising beyond exact phrase matches. Not a DetectorPort itself:
like TranscriberPort, this produces a classification result that a
detector (app/adapters/detectors/intent_risk.py) reads out of context,
not a risk score directly.

STATUS: built and verified (real model, real license, real language
coverage — see docs/risk-model.md's "Intent detection" section), but NOT
wired into config/risk_formula.yaml's active `detectors:` list by
default. A real calibration problem was found before shipping this live:
see zero_shot_intent_classifier.py's own honesty note.

Scope note: same as TranscriberPort — file-upload path only. Depends on a
transcript existing, so it's meaningless on the live streaming path.
"""
from __future__ import annotations

from typing import Protocol

from app.domain.models import IntentClassificationResult


class IntentClassifierPort(Protocol):
    name: str
    version: str

    def classify(self, text: str) -> IntentClassificationResult:
        """Returns an IntentClassificationResult for the given transcript
        text. Callers should not call this per-window — like
        TranscriberPort, it's meant to run once per call; see
        Engine.score_call()'s docstring for why."""
        ...

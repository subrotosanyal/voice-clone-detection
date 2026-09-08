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

STATUS: built, verified, and enabled by default (see docs/risk-model.md's
"Intent detection" section) — with a real, documented calibration
problem that was surfaced in full before enabling, not fixed. See
zero_shot_intent_classifier.py's own honesty note for what mitigates the
risk (a low fusion weight, full UI transparency) and what doesn't (the
model still misjudges ordinary conversation).

Scope note: depends on a transcript existing — the file-upload path
provides one up front (Engine.score_call()); the live WebSocket path
(added 2026-09-08) now does too, via app/pipeline/live_transcription.py's
periodic background transcription, so this is no longer file-upload-only.
The live path's classification necessarily lags real speech (see that
module's own honesty note on the delay) and abstains entirely until the
first background transcription completes — same abstain-on-no-transcript
behaviour app/adapters/detectors/intent_risk.py already had, just now
reachable from both call paths instead of one.
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

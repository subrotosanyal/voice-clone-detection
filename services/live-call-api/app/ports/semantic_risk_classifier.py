"""The SemanticRiskClassifier port.

Merges what were discussed as two separate ideas (a smarter replacement
for the zero-shot intent classifier, and a "script divergence"/semantic-
drift tracker for social-engineering patterns) into one: a generative
LLM reads the call's transcript and produces a richer structured
judgment than either existing mechanism alone — see
app/domain/models.py's SemanticRiskAssessment for the exact fields and
why `isolation_request` in particular is a genuinely new signal.

NOT a DetectorPort: like TranscriberPort and IntentClassifierPort, this
produces a result a detector would read out of context, not a risk
score directly — see Engine.score_call()'s "compute once per call, read
many times" pattern that TranscriberPort/IntentClassifierPort already
use, which this is meant to slot into the same way once wired in.

STATUS (2026-09-09): wired into Engine.score_call(), the registry, and
config/risk_formula.yaml's active `detectors:` list (`semantic_risk`,
weight 0.15) — enabled only after real, measured evidence, not an
assumed improvement: run by hand against the exact sentence
`intent`'s own zero-shot classifier is documented to misjudge, this
correctly scored it as non-suspicious, and correctly flagged two real
scam scripts (including the isolation tactic neither `intent` nor the
keyword-based contextual rules capture). See app/adapters/semantic_risk/
local_llm_semantic_classifier.py's own HONESTY NOTE and
config/risk_formula.yaml's `semantic_risk` entry for the exact numbers
and reasoning behind the starting weight. Runs ALONGSIDE `intent`, not
as a replacement for it — see that config entry for why.
"""
from __future__ import annotations

from typing import Protocol

from app.domain.models import SemanticRiskAssessment


class SemanticRiskClassifierPort(Protocol):
    name: str
    version: str

    def analyze(self, text: str) -> SemanticRiskAssessment:
        """Returns a SemanticRiskAssessment for the given transcript
        text. Like IntentClassifierPort.classify(), meant to run once
        per call, not per-window — see Engine.score_call()'s docstring
        for why (this is an expensive, non-real-time-per-window
        operation, same reasoning as transcription and intent
        classification)."""
        ...

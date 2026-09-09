"""Tests for LocalLLMSemanticClassifier.

Two groups: fast, offline, deterministic tests of the JSON-extraction and
abstain/fallback logic (a fake `_llm` injected directly, bypassing
_ensure_loaded() and the real model entirely — same "fake the model
call, test the plumbing" style as test_live_transcription.py's fake
transcriber/intent classifier), and a REAL-model comparison against
zero_shot_intent_classifier.py's own documented false positive — the
whole point of building this component in the first place. See
local_llm_semantic_classifier.py's own HONESTY NOTE for the full
reasoning before changing the prompt or the risk mapping.
"""
from __future__ import annotations

import pytest

from app.adapters.semantic_risk.local_llm_semantic_classifier import (
    LocalLLMSemanticClassifier,
    _extract_json,
)


class _FakeLlama:
    """Mimics llama_cpp.Llama's __call__ return shape closely enough for
    LocalLLMSemanticClassifier.analyze() to consume — no real model."""

    def __init__(self, completion_text: str) -> None:
        self.completion_text = completion_text
        self.calls = 0

    def __call__(self, prompt: str, **kwargs) -> dict:
        self.calls += 1
        return {"choices": [{"text": self.completion_text}]}


def _classifier_with_fake_llm(completion_text: str) -> LocalLLMSemanticClassifier:
    clf = LocalLLMSemanticClassifier()
    clf._llm = _FakeLlama(completion_text)  # bypasses _ensure_loaded() entirely
    return clf


# --- _extract_json (pure function, no model at all) -------------------


def test_extract_json_parses_a_clean_object():
    result = _extract_json('{"urgency_level": 0.5, "financial_solicitation": true}')
    assert result == {"urgency_level": 0.5, "financial_solicitation": True}


def test_extract_json_pulls_the_object_out_of_surrounding_text():
    """Small quantized models don't always emit ONLY the JSON — this
    must still find it, not require the whole completion to be clean."""
    result = _extract_json('Sure, here is the analysis:\n{"urgency_level": 0.2}\nLet me know if you need more.')
    assert result == {"urgency_level": 0.2}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("not json at all") is None


def test_extract_json_returns_none_on_malformed_braces():
    assert _extract_json('{"urgency_level": 0.5,}') is None  # trailing comma, invalid JSON


# --- LocalLLMSemanticClassifier.analyze(), fake model ------------------


def test_empty_transcript_abstains_without_calling_the_model():
    clf = _classifier_with_fake_llm("irrelevant")
    result = clf.analyze("")
    assert result.urgency_level == 0.0
    assert clf._llm.calls == 0  # never even reaches the model for empty text


def test_valid_json_completion_is_parsed_correctly():
    clf = _classifier_with_fake_llm(
        '{"urgency_level": 0.8, "financial_solicitation": true, '
        '"authority_claim": true, "isolation_request": false, '
        '"reasoning": "bank scam"}'
    )
    result = clf.analyze("some transcript")
    assert result.urgency_level == 0.8
    assert result.financial_solicitation is True
    assert result.authority_claim is True
    assert result.isolation_request is False
    assert result.reasoning == "bank scam"


def test_malformed_completion_falls_back_to_neutral_not_a_crash():
    """FAILURE MODE note in the class docstring: never let an
    infrastructure failure (bad JSON) manufacture a false positive."""
    clf = _classifier_with_fake_llm("I cannot help with that request.")
    result = clf.analyze("some transcript")
    assert result.urgency_level == 0.0
    assert result.financial_solicitation is False
    assert "could not parse" in result.reasoning


def test_missing_required_field_falls_back_to_neutral():
    clf = _classifier_with_fake_llm('{"urgency_level": 0.5}')  # missing financial_solicitation etc.
    result = clf.analyze("some transcript")
    assert result.urgency_level == 0.0
    assert "could not parse" in result.reasoning


# --- Real model: the actual evaluation that justified enabling this ----


@pytest.fixture(scope="session")
def real_classifier(phi3_llm_model_path) -> LocalLLMSemanticClassifier:
    clf = LocalLLMSemanticClassifier(model_path=phi3_llm_model_path)
    clf._ensure_loaded()  # force the load now, so a failure skips cleanly
    return clf


def test_real_model_correctly_handles_the_exact_sentence_mdeberta_gets_wrong(real_classifier):
    """THE evaluation that justified enabling this alongside `intent`:
    zero_shot_intent_classifier.py's own HONESTY NOTE documents this
    EXACT sentence scoring 38.9% "creating urgency" against only 4.5%
    "ordinary conversation" — a severe false positive. This must not
    reproduce that failure."""
    result = real_classifier.analyze("Hey, are we still on for dinner tonight? I was thinking Italian.")
    assert result.urgency_level < 0.3
    assert result.financial_solicitation is False
    assert result.authority_claim is False


def test_real_model_correctly_flags_an_otp_phishing_script(real_classifier):
    result = real_classifier.analyze(
        "Sir this is inspector from cyber crime cell, you need to share the OTP "
        "right now or you will be arrested."
    )
    assert result.urgency_level > 0.5
    assert result.authority_claim is True


def test_real_model_correctly_flags_a_bank_scam_with_isolation_tactic(real_classifier):
    """The genuinely NEW signal this component adds over both `intent`
    and the keyword-based contextual rules: isolation_request."""
    result = real_classifier.analyze(
        "Sir, this is your bank calling, your account has been compromised, please "
        "transfer your funds to this safe account immediately, and please don't hang "
        "up the phone or tell anyone."
    )
    assert result.urgency_level > 0.5
    assert result.financial_solicitation is True
    assert result.authority_claim is True
    assert result.isolation_request is True

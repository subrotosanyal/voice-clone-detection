"""Tests for SemanticRiskDetector — reads PRE-COMPUTED context, calls no
model itself (see the class's own docstring for why), same
fast/deterministic/offline pattern as test_intent_risk.py."""
import numpy as np
import pytest

from app.adapters.detectors.semantic_risk_detector import SemanticRiskDetector
from app.domain.models import AudioWindow

WINDOW = AudioWindow(session_id="s1", seq=0, sample_rate=16_000, samples=np.zeros(1), window_start_ms=0)


def test_abstains_without_precomputed_semantic_assessment():
    detector = SemanticRiskDetector()
    result = detector.score(WINDOW, context={})
    assert result.score is None
    assert "semantic" in result.abstain_reason


def test_benign_conversation_scores_low_risk():
    detector = SemanticRiskDetector()
    result = detector.score(
        WINDOW,
        context={
            "semantic_urgency_level": 0.0,
            "semantic_financial_solicitation": False,
            "semantic_authority_claim": False,
            "semantic_isolation_request": False,
            "semantic_reasoning": "Conversation is a dinner plan, not suspicious.",
        },
    )
    assert result.score == 0.0
    assert "No social-engineering tactics" in result.detail["explanation"]


def test_high_urgency_alone_drives_the_score():
    """urgency_level on its own (no flags true) should still be able to
    drive risk — the max(), not requiring both signals to agree, is the
    whole point of the mapping (see the detector's own docstring)."""
    detector = SemanticRiskDetector()
    result = detector.score(
        WINDOW,
        context={
            "semantic_urgency_level": 0.9,
            "semantic_financial_solicitation": False,
            "semantic_authority_claim": False,
            "semantic_isolation_request": False,
            "semantic_reasoning": "Manufactured urgency without a concrete ask.",
        },
    )
    assert result.score == pytest.approx(0.9)


def test_all_three_flags_present_drives_the_score_even_with_low_urgency():
    detector = SemanticRiskDetector()
    result = detector.score(
        WINDOW,
        context={
            "semantic_urgency_level": 0.1,
            "semantic_financial_solicitation": True,
            "semantic_authority_claim": True,
            "semantic_isolation_request": True,
            "semantic_reasoning": "Scam call with urgency, authority claim, and isolation tactic.",
        },
    )
    assert result.score == pytest.approx(1.0)  # 3/3 flags
    assert "a financial request" in result.detail["explanation"]
    assert "a claimed authority" in result.detail["explanation"]
    assert "a request not to hang up or tell anyone" in result.detail["explanation"]


def test_detail_carries_every_raw_field_for_explainability():
    detector = SemanticRiskDetector()
    result = detector.score(
        WINDOW,
        context={
            "semantic_urgency_level": 0.5,
            "semantic_financial_solicitation": True,
            "semantic_authority_claim": False,
            "semantic_isolation_request": False,
            "semantic_reasoning": "some reason",
        },
    )
    assert result.detail["urgency_level"] == 0.5
    assert result.detail["financial_solicitation"] is True
    assert result.detail["authority_claim"] is False
    assert result.detail["isolation_request"] is False
    assert result.detail["reasoning"] == "some reason"

"""Tests for IntentRiskDetector — reads PRE-COMPUTED context, calls no
model itself (see the class's own docstring for why), so these are fast,
deterministic, offline unit tests over literal context dicts, same
pattern as test_contextual_rules.py."""
import numpy as np
import pytest

from app.adapters.detectors.intent_risk import IntentRiskDetector
from app.domain.models import AudioWindow

WINDOW = AudioWindow(session_id="s1", seq=0, sample_rate=16_000, samples=np.zeros(1), window_start_ms=0)


def test_abstains_without_precomputed_intent_classification():
    detector = IntentRiskDetector()
    result = detector.score(WINDOW, context={})
    assert result.score is None
    assert "intent" in result.abstain_reason


def test_ordinary_top_label_scores_low_risk():
    detector = IntentRiskDetector()
    result = detector.score(
        WINDOW,
        context={
            "intent_label": "ordinary conversation",
            "intent_top_score": 0.8,
            "intent_label_scores": {
                "requesting a money transfer or payment": 0.05,
                "requesting an OTP or PIN": 0.03,
                "impersonating a bank or government official": 0.02,
                "creating urgency or time pressure": 0.1,
                "ordinary conversation": 0.8,
            },
        },
    )
    assert result.score == pytest.approx(0.2)  # 1.0 - 0.8, modulo float precision


def test_risky_top_label_scores_high_risk():
    detector = IntentRiskDetector()
    result = detector.score(
        WINDOW,
        context={
            "intent_label": "requesting a money transfer or payment",
            "intent_top_score": 0.86,
            "intent_label_scores": {
                "requesting a money transfer or payment": 0.86,
                "requesting an OTP or PIN": 0.01,
                "impersonating a bank or government official": 0.03,
                "creating urgency or time pressure": 0.09,
                "ordinary conversation": 0.01,
            },
        },
    )
    assert result.score == pytest.approx(0.99)  # 1.0 - 0.01
    assert "requesting a money transfer or payment" in result.detail["explanation"]


def test_missing_ordinary_label_in_scores_defaults_to_zero():
    """Defensive: if a caller ever passes a label_scores dict that doesn't
    even include the ordinary-conversation key (e.g. a customised
    candidate list), risk should default to maximal, not crash."""
    detector = IntentRiskDetector()
    result = detector.score(
        WINDOW,
        context={
            "intent_label": "requesting an OTP or PIN",
            "intent_top_score": 0.7,
            "intent_label_scores": {"requesting an OTP or PIN": 0.7, "other": 0.3},
        },
    )
    assert result.score == 1.0

import numpy as np

from app.adapters.detectors.contextual_rules import ContextualRulesDetector
from app.domain.models import AudioWindow

WINDOW = AudioWindow(session_id="s1", seq=0, sample_rate=16_000, samples=np.zeros(1), window_start_ms=0)


def test_abstains_with_no_context():
    detector = ContextualRulesDetector()
    result = detector.score(WINDOW, context={})
    assert result.score is None


def test_all_rules_off_scores_zero():
    detector = ContextualRulesDetector()
    result = detector.score(
        WINDOW,
        context={"known_number": True, "hour_of_day": 14, "is_financial_request": False, "urgency_keywords": []},
    )
    assert result.score == 0.0
    assert result.detail["rules_fired"]["unknown_number"] is False
    assert result.detail["rules_fired"]["unusual_hour"] is False
    assert result.detail["rules_fired"]["financial_request"] is False
    assert result.detail["rules_fired"]["urgency_language"] is False


def test_all_rules_on_scores_capped_at_one():
    detector = ContextualRulesDetector()
    result = detector.score(
        WINDOW,
        context={
            "known_number": False,
            "hour_of_day": 2,
            "is_financial_request": True,
            "urgency_keywords": ["immediately", "don't tell anyone"],
        },
    )
    assert result.score == 1.0  # 0.30 + 0.20 + 0.30 + 0.20 = 1.00 exactly


def test_partial_rules_sum_correctly():
    detector = ContextualRulesDetector()
    result = detector.score(
        WINDOW,
        context={"known_number": False, "is_financial_request": True},
    )
    assert result.score == 0.60  # 0.30 (unknown number) + 0.30 (financial)


def test_unusual_hour_boundaries():
    detector = ContextualRulesDetector()
    late = detector.score(WINDOW, context={"hour_of_day": 23})
    early = detector.score(WINDOW, context={"hour_of_day": 6})
    business = detector.score(WINDOW, context={"hour_of_day": 12})

    assert late.detail["rules_fired"]["unusual_hour"] is True
    assert early.detail["rules_fired"]["unusual_hour"] is True
    assert business.detail["rules_fired"]["unusual_hour"] is False

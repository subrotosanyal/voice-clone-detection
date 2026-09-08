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


def test_transcript_auto_detects_urgency_and_financial_request():
    """A transcript (set by Engine.score_call() when a transcriber is
    configured — see app/adapters/transcription/) is scanned the same way
    a manually-supplied urgency_keywords list is, without the caller
    needing to type anything by hand."""
    detector = ContextualRulesDetector()
    result = detector.score(
        WINDOW,
        context={"transcript": "Please transfer the money immediately, it's urgent."},
    )
    assert result.score is not None
    assert result.detail["rules_fired"]["financial_request"] is True
    assert result.detail["rules_fired"]["urgency_language"] is True
    assert "immediately" in result.detail["urgency_keywords"]
    assert "urgent" in result.detail["urgency_keywords"]
    assert result.detail["is_financial_request_from_transcript"] is True


def test_transcript_merges_with_manually_supplied_fields_not_replacing_them():
    detector = ContextualRulesDetector()
    result = detector.score(
        WINDOW,
        context={
            "urgency_keywords": ["don't tell anyone"],
            "transcript": "please respond immediately",
        },
    )
    # both the manual keyword and the transcript-derived one are present
    assert "don't tell anyone" in result.detail["urgency_keywords"]
    assert "immediately" in result.detail["urgency_keywords"]
    assert result.detail["urgency_keywords_manual"] == ["don't tell anyone"]
    assert result.detail["urgency_keywords_from_transcript"] == ["immediately"]


def test_empty_transcript_does_not_add_a_transcript_detail():
    detector = ContextualRulesDetector()
    result = detector.score(WINDOW, context={"known_number": True, "transcript": ""})
    assert "transcript" not in result.detail


def test_transcript_with_no_urgency_language_does_not_fire_the_rule():
    detector = ContextualRulesDetector()
    result = detector.score(WINDOW, context={"transcript": "Good morning, how are you today?"})
    assert result.detail["rules_fired"]["urgency_language"] is False
    # financial_request isn't reported at all here — no manual value was
    # supplied and the transcript didn't trigger it either, matching the
    # pre-existing convention (see test_abstains_with_no_context): a rule
    # only appears in rules_fired when there's something to report.
    assert "financial_request" not in result.detail["rules_fired"]


def test_manual_authority_claim_fires_its_own_rule():
    detector = ContextualRulesDetector()
    result = detector.score(WINDOW, context={"authority_claim": True})
    assert result.detail["rules_fired"]["authority_claim"] is True
    assert result.score == 0.25  # weight_authority_claim, alone


def test_transcript_auto_detects_authority_claim():
    # Deliberately avoids the word "bank" — it's also a financial keyword,
    # which would make this a bad example of "authority claim alone".
    detector = ContextualRulesDetector()
    result = detector.score(
        WINDOW, context={"transcript": "This is the police calling about a noise complaint at your address."}
    )
    assert result.detail["rules_fired"]["authority_claim"] is True
    assert "police" in result.detail["authority_keywords_from_transcript"]
    # no financial ask here — the combined pattern must NOT fire on an
    # authority claim alone, only on authority + financial together.
    assert "combined_authority_financial_pressure" not in result.detail["rules_fired"]


def test_authority_claim_and_financial_request_together_fires_combined_pattern():
    """The classic fraud script — 'this is your bank, your account has
    been compromised, transfer your funds now' — should score higher than
    the two component rules alone would sum to, via the explicit combined
    rule, not just financial_request + authority_claim added up."""
    detector = ContextualRulesDetector()
    result = detector.score(
        WINDOW,
        context={
            "transcript": "This is your bank calling. Your account has been compromised. "
            "Please transfer your funds to a safe account immediately."
        },
    )
    fired = result.detail["rules_fired"]
    assert fired["authority_claim"] is True
    assert fired["financial_request"] is True
    assert fired["combined_authority_financial_pressure"] is True
    # 0.25 (authority) + 0.30 (financial) + 0.20 (urgency, "immediately") +
    # 0.25 (combined) = 1.00, capped
    assert result.score == 1.0


def test_financial_request_without_authority_claim_does_not_fire_combined_pattern():
    detector = ContextualRulesDetector()
    result = detector.score(WINDOW, context={"is_financial_request": True})
    assert "combined_authority_financial_pressure" not in result.detail["rules_fired"]

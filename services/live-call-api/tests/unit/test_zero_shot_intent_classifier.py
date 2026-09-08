"""Tests for ZeroShotIntentClassifier.

Real model, real network fetch on first use (skipped gracefully if
unavailable, same pattern as test_whisper_transcriber.py). These tests
verify the class works mechanically AND honestly document the real
calibration finding that keeps this detector out of the active formula —
see zero_shot_intent_classifier.py's own HONESTY NOTE, and
test_known_limitation_ordinary_conversation_is_not_reliably_detected
below, which captures the actual, current, verified behaviour rather than
asserting a result the model doesn't actually produce.
"""
from __future__ import annotations

import pytest

from app.adapters.intent.zero_shot_intent_classifier import (
    _CANDIDATE_LABELS,
    _ORDINARY_LABEL,
    ZeroShotIntentClassifier,
)


@pytest.fixture(scope="session")
def classifier() -> ZeroShotIntentClassifier:
    try:
        clf = ZeroShotIntentClassifier()
        clf._ensure_loaded()  # force the download/load now, so a failure skips cleanly
        return clf
    except Exception as exc:  # noqa: BLE001 — model download can fail offline
        pytest.skip(f"zero-shot intent model unavailable (no network?): {exc}")


def test_result_structure_is_a_full_probability_distribution(classifier):
    result = classifier.classify("Hello, how can I help you today?")
    assert result.detector_name == "zero_shot_intent_classifier"
    assert set(result.label_scores.keys()) == set(_CANDIDATE_LABELS)
    assert result.top_label in _CANDIDATE_LABELS
    assert result.top_score == max(result.label_scores.values())
    # multi_label=False (the default) normalises scores across all
    # candidates — they should sum to ~1, not be independent probabilities.
    assert abs(sum(result.label_scores.values()) - 1.0) < 1e-3


def test_clear_fraud_script_is_classified_as_financial_request(classifier):
    result = classifier.classify(
        "This is your bank calling. Your account has been compromised. "
        "Please transfer your funds to a safe account immediately, and do not tell anyone."
    )
    assert result.top_label == "requesting a money transfer or payment"
    assert result.label_scores[_ORDINARY_LABEL] < 0.1


def test_clear_impersonation_script_scores_low_on_ordinary_conversation(classifier):
    result = classifier.classify(
        "Sir this is inspector from cyber crime cell, you need to share the OTP "
        "right now or you will be arrested."
    )
    assert result.label_scores[_ORDINARY_LABEL] < 0.1


def test_known_limitation_ordinary_conversation_is_not_reliably_detected(classifier):
    """KNOWN LIMITATION, not a regression to fix quietly: a completely
    benign sentence should score high on 'ordinary conversation' and it
    verifiably does not, under this model + this candidate label set.
    This is exactly why IntentRiskDetector is not wired into the active
    risk formula — see zero_shot_intent_classifier.py's HONESTY NOTE. This
    test exists so a future attempt to fix the calibration (a rebalanced
    candidate list, a different model, hypothesis_template tuning) has a
    concrete, reproducible check to run against — flip the assertion
    direction once it's actually fixed, don't delete it."""
    result = classifier.classify("Hey, are we still on for dinner tonight? I was thinking Italian.")
    # Documenting the actual (bad) behaviour, not the desired one: as of
    # this writing, "ordinary conversation" is NOT the top label for this
    # obviously-ordinary sentence.
    assert result.top_label != _ORDINARY_LABEL
    assert result.label_scores[_ORDINARY_LABEL] < 0.5

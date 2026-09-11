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


def test_concurrent_ensure_loaded_calls_load_the_model_only_once(monkeypatch):
    """Regression test for a real bug found 2026-09-11, via a home-lab
    production log (`ImportError: cannot import name 'pipeline' from
    'transformers'`, raised inside Engine.diarize_and_score()'s per-speaker
    scoring): _ensure_loaded() used to be a plain, unsynchronized
    `if self._pipeline is None:` check — safe only as long as a single
    shared ZeroShotIntentClassifier instance (see app/main.py's one-time
    app.state.engine construction) was never asked to classify() from more
    than one thread at once. That stopped being true once
    Engine.diarize_and_score() started scoring different speakers
    CONCURRENTLY via a thread pool — the exact same race already found and
    fixed in local_llm_semantic_classifier.py's own _ensure_loaded() (see
    test_local_llm_semantic_classifier.py's own
    test_concurrent_ensure_loaded_calls_load_the_model_only_once, which
    this test mirrors), just missed here at the time. Proves the fix:
    double-checked locking around the load."""
    import sys
    import threading
    import time
    import types

    load_count = 0
    max_concurrent_loads = 0
    current_loads = 0
    counter_lock = threading.Lock()

    fake_transformers_module = types.ModuleType("transformers")

    def _slow_fake_pipeline(*args, **kwargs):
        nonlocal load_count, max_concurrent_loads, current_loads
        with counter_lock:
            current_loads += 1
            max_concurrent_loads = max(max_concurrent_loads, current_loads)
            load_count += 1
        time.sleep(0.1)  # long enough for a second thread to reach the check first
        with counter_lock:
            current_loads -= 1
        return "fake-pipeline-object"

    fake_transformers_module.pipeline = _slow_fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers_module)

    clf = ZeroShotIntentClassifier()

    threads = [threading.Thread(target=clf._ensure_loaded) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert load_count == 1, f"model was loaded {load_count} times, not once"
    assert max_concurrent_loads == 1, "two threads loaded the model concurrently — not actually serialized"
    assert clf._pipeline == "fake-pipeline-object"

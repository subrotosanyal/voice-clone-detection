"""Tests for SessionStore.apply_sticky_flags() — see its own docstring in
app/pipeline/engine.py for the real live-mic problem this fixes (a
rolling transcript window makes contextual_rules/semantic_risk forget a
signal once it ages out).

Two levels: SessionStore in isolation (fast, no detector involved), and
Engine.score_window() end-to-end with the REAL ContextualRulesDetector,
proving the combined authority+financial pressure rule fires across two
separate windows whose transcripts individually mention only one half —
exactly the case a live call's rolling window would otherwise miss.
"""
from __future__ import annotations

import numpy as np

from app.adapters.detectors.contextual_rules import ContextualRulesDetector
from app.adapters.fusion.weighted_sum import WeightedSumFusion
from app.adapters.registry import Pipeline
from app.domain.models import AudioWindow
from app.pipeline.engine import Engine, SessionStore

SR = 16_000


def _window(seq: int) -> AudioWindow:
    samples = np.zeros(SR, dtype=np.float32)
    return AudioWindow(session_id="s1", seq=seq, sample_rate=SR, samples=samples, window_start_ms=seq * 2000)


def _pipeline() -> Pipeline:
    return Pipeline(
        config={
            "formula_version": "test",
            "detectors": [{"name": "contextual_rules", "weight": 1.0}],
            "fusion": {"smoothing_alpha": 1.0},
            "bands": {"low_max": 34, "elevated_max": 69},
        },
        detectors=[ContextualRulesDetector()],
        fusion=WeightedSumFusion(),
    )


# ---------- SessionStore in isolation ----------


def test_boolean_flag_stays_true_once_observed():
    store = SessionStore()
    store.apply_sticky_flags("s1", {"authority_claim": True})
    merged = store.apply_sticky_flags("s1", {})  # next window: no longer mentioned
    assert merged["authority_claim"] is True


def test_urgency_level_tracks_a_running_max():
    store = SessionStore()
    store.apply_sticky_flags("s1", {"semantic_urgency_level": 0.8})
    merged = store.apply_sticky_flags("s1", {"semantic_urgency_level": 0.2})  # a later, calmer window
    assert merged["semantic_urgency_level"] == 0.8


def test_urgency_keywords_union_without_duplicates():
    store = SessionStore()
    store.apply_sticky_flags("s1", {"urgency_keywords": ["urgent"]})
    merged = store.apply_sticky_flags("s1", {"urgency_keywords": ["urgent", "immediately"]})
    assert merged["urgency_keywords"] == ["urgent", "immediately"]


def test_transcript_derived_authority_claim_is_re_derived_and_sticky():
    """No explicit `authority_claim` field at all — derived from the
    transcript itself via the same detect_urgency_signals() contextual_
    rules.py uses, and still sticks once seen."""
    store = SessionStore()
    store.apply_sticky_flags("s1", {"transcript": "this is your bank calling"})
    merged = store.apply_sticky_flags("s1", {"transcript": "please confirm your identity"})
    assert merged["authority_claim"] is True


def test_reset_clears_sticky_state():
    store = SessionStore()
    store.apply_sticky_flags("s1", {"authority_claim": True})
    store.reset("s1")
    merged = store.apply_sticky_flags("s1", {})
    assert merged["authority_claim"] is False


def test_different_sessions_do_not_share_sticky_state():
    store = SessionStore()
    store.apply_sticky_flags("s1", {"authority_claim": True})
    merged_other = store.apply_sticky_flags("s2", {})
    assert merged_other["authority_claim"] is False


def test_file_upload_shape_is_a_no_op():
    """File-upload's context is already whole-call from window 1 — merging
    it with itself must not change anything."""
    store = SessionStore()
    context = {"authority_claim": True, "is_financial_request": True}
    merged = store.apply_sticky_flags("s1", context)
    assert merged["authority_claim"] is True
    assert merged["is_financial_request"] is True


# ---------- End-to-end through Engine.score_window() + the real detector ----------


def test_combined_pressure_rule_fires_across_two_windows_on_a_live_style_session():
    """The exact fraud-relevant case this was built for: an authority
    claim in an EARLY window's (rolling, live-style) transcript, a
    financial request in a LATER window whose transcript no longer
    mentions the authority claim at all — contextual_rules.py's own
    combined_authority_financial_pressure rule must still fire on the
    later window, because the authority claim is sticky for the session."""
    engine = Engine(pipeline=_pipeline())

    # "police" is an authority-claim phrase with no financial keyword in
    # it (unlike "this is your bank", which would trip BOTH at once and
    # defeat the point of this test — see urgency_language.py's own lists).
    first = engine.score_window(_window(0), {"transcript": "this call is from the police"})
    first_contextual = next(c for c in first.components if c.name == "contextual_rules")
    assert first_contextual.detail["rules_fired"].get("combined_authority_financial_pressure") is not True

    # A later window: the rolling transcript has moved on, no longer
    # mentions the police at all — only a financial request now.
    second = engine.score_window(_window(1), {"transcript": "please transfer the money now"})
    second_contextual = next(c for c in second.components if c.name == "contextual_rules")
    assert second_contextual.detail["rules_fired"]["combined_authority_financial_pressure"] is True
    assert second_contextual.detail["rules_fired"]["authority_claim"] is True


def test_file_upload_single_context_still_scores_normally():
    """Sanity check: the sticky mechanism must not break the ordinary
    file-upload shape, where the SAME whole-call context is passed to
    every window unchanged."""
    engine = Engine(pipeline=_pipeline())
    context = {"transcript": "this is your bank calling, please transfer the money now"}

    first = engine.score_window(_window(0), context)
    second = engine.score_window(_window(1), context)

    for fused in (first, second):
        contextual = next(c for c in fused.components if c.name == "contextual_rules")
        assert contextual.detail["rules_fired"]["combined_authority_financial_pressure"] is True

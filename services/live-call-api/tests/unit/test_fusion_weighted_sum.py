from app.adapters.fusion.weighted_sum import WeightedSumFusion
from app.domain.models import Band, DetectorResult

CONFIG = {
    "formula_version": "test-formula-v1",
    "detectors": [
        {"name": "acoustic", "weight": 0.6},
        {"name": "prosodic", "weight": 0.2},
        {"name": "third_signal", "weight": 0.2, "params": {"mode": "contextual"}},
    ],
    "fusion": {"smoothing_alpha": 0.5},
    "bands": {"low_max": 34, "elevated_max": 69},
}


def _result(name: str, score) -> DetectorResult:
    return DetectorResult(detector_name=name, detector_version="test", score=score)


def test_all_signals_present_weighted_sum_is_exact():
    fusion = WeightedSumFusion()
    results = [_result("acoustic", 0.5), _result("prosodic", 0.5), _result("third_signal", 0.5)]

    fused = fusion.fuse("s1", 0, 0, results, previous_component_smoothed={}, config=CONFIG)

    # all three at 0.5 -> weighted sum is 0.5 regardless of weight split
    assert fused.raw_score_0_100 == 50.0
    assert fused.smoothed_score_0_100 == 50.0  # no previous score -> smoothed == raw for every component


def test_formula_version_comes_from_config_not_the_fusion_class():
    """Regression test for a real bug: FusedScore.formula_version used to
    be stamped from WeightedSumFusion.version (a hardcoded class
    attribute, "0.1.0") instead of config["formula_version"] — meaning
    every score was mislabeled with the same fixed string regardless of
    which config actually produced it, contradicting docs/risk-model.md's
    documented reproducibility promise. CONFIG's formula_version here is
    deliberately different from WeightedSumFusion.version so this would
    fail loudly if the bug ever came back."""
    fusion = WeightedSumFusion()
    assert CONFIG["formula_version"] != fusion.version
    results = [_result("acoustic", 0.5), _result("prosodic", 0.5), _result("third_signal", 0.5)]

    fused = fusion.fuse("s1", 0, 0, results, previous_component_smoothed={}, config=CONFIG)

    assert fused.formula_version == CONFIG["formula_version"]


def test_abstained_signal_with_no_history_is_renormalised_not_zeroed():
    fusion = WeightedSumFusion()
    # third_signal abstains on its very first window (no prior history to
    # carry forward) — acoustic=1.0, prosodic=1.0 should renormalise to
    # weight 0.75/0.25 of each other and still reach 1.0*100, NOT be
    # dragged down by treating the abstained signal as a zero.
    results = [_result("acoustic", 1.0), _result("prosodic", 1.0), _result("third_signal", None)]

    fused = fusion.fuse("s1", 0, 0, results, previous_component_smoothed={}, config=CONFIG)

    assert fused.raw_score_0_100 == 100.0
    assert fused.smoothed_score_0_100 == 100.0
    third_signal_component = next(c for c in fused.components if c.name == "third_signal")
    assert third_signal_component.abstained is True
    assert third_signal_component.smoothed_score is None  # never scored, nothing to carry forward
    assert third_signal_component.weight_effective == 0.0
    assert third_signal_component.contribution == 0.0


def test_per_component_ema_moves_toward_new_raw_score():
    fusion = WeightedSumFusion()
    results = [_result("acoustic", 1.0), _result("prosodic", 1.0), _result("third_signal", 1.0)]
    previous = {"acoustic": 0.0, "prosodic": 0.0, "third_signal": 0.0}

    fused = fusion.fuse("s1", 1, 500, results, previous_component_smoothed=previous, config=CONFIG)

    # alpha=0.5: each component's own smoothed value = 0.5*1.0 + 0.5*0.0 = 0.5
    acoustic = next(c for c in fused.components if c.name == "acoustic")
    assert acoustic.smoothed_score == 0.5
    # combined: still 0.5 regardless of weight split, since every component moved the same way
    assert fused.smoothed_score_0_100 == 50.0
    # raw_score_0_100 is untouched by any of this — it's this window alone
    assert fused.raw_score_0_100 == 100.0


def test_abstaining_component_with_history_carries_its_smoothed_value_forward():
    """The actual real bug this whole redesign fixes (2026-09-10, found via
    a real ~53s movie-dialogue clip): a component that abstains for ONE
    window must not simply vanish from the formula if it has real recent
    history — it should keep contributing its last known smoothed value
    until it either scores again or the session ends."""
    fusion = WeightedSumFusion()
    # acoustic has been reading ~0.9 (confident spoof) for a while; this
    # window it abstains (e.g. a quiet trailing slice) but should still
    # count at its carried-forward 0.9, not drop out and let prosodic take
    # over 100% of the weight.
    results = [_result("acoustic", None), _result("prosodic", 0.5), _result("third_signal", 0.5)]
    previous = {"acoustic": 0.9, "prosodic": 0.5, "third_signal": 0.5}

    fused = fusion.fuse("s1", 5, 2500, results, previous_component_smoothed=previous, config=CONFIG)

    acoustic = next(c for c in fused.components if c.name == "acoustic")
    assert acoustic.abstained is True  # honest about THIS window
    assert acoustic.smoothed_score == 0.9  # but still carries its real history
    assert acoustic.weight_effective > 0.0  # and still counts in the formula
    # all three components have a usable (if carried-forward) value, so
    # the weighted sum is exactly 0.6*0.9 + 0.2*0.5 + 0.2*0.5 = 0.74
    assert fused.smoothed_score_0_100 == 74.0
    # raw_score_0_100, by contrast, only sees THIS window's raw readings —
    # acoustic is genuinely absent from it, renormalised across the other two
    assert fused.raw_score_0_100 == 50.0


def test_bands_are_inclusive_at_boundaries():
    fusion = WeightedSumFusion()
    zero_results = [_result("acoustic", 0.0), _result("prosodic", 0.0), _result("third_signal", 0.0)]
    high_results = [_result("acoustic", 1.0), _result("prosodic", 1.0), _result("third_signal", 1.0)]

    low = fusion.fuse("s1", 0, 0, zero_results, previous_component_smoothed={}, config=CONFIG)
    high = fusion.fuse("s1", 0, 0, high_results, previous_component_smoothed={}, config=CONFIG)

    assert low.band == Band.LOW
    assert high.band == Band.HIGH


def test_component_contributions_sum_to_smoothed_score():
    """Contributions are now computed from each component's SMOOTHED value
    (the number that actually determines the recommended action), not the
    raw per-window one — see WeightedSumFusion's own docstring for why."""
    fusion = WeightedSumFusion()
    results = [_result("acoustic", 0.9), _result("prosodic", 0.3), _result("third_signal", 0.1)]

    fused = fusion.fuse("s1", 0, 0, results, previous_component_smoothed={}, config=CONFIG)

    total_contribution = sum(c.contribution for c in fused.components)
    assert abs(total_contribution * 100 - fused.smoothed_score_0_100) < 0.01


def test_raw_score_reflects_only_this_windows_readings():
    """raw_score_0_100 is deliberately NOT derived from the smoothed
    per-component values — it's this window alone, unsmoothed, so a real
    instantaneous spike is still visible even while smoothed_score_0_100
    (driven by history) hasn't caught up to it yet."""
    fusion = WeightedSumFusion()
    results = [_result("acoustic", 1.0), _result("prosodic", 1.0), _result("third_signal", 1.0)]
    previous = {"acoustic": 0.0, "prosodic": 0.0, "third_signal": 0.0}

    fused = fusion.fuse("s1", 1, 500, results, previous_component_smoothed=previous, config=CONFIG)

    assert fused.raw_score_0_100 == 100.0  # this window alone
    assert fused.smoothed_score_0_100 == 50.0  # dragged toward history (alpha=0.5)

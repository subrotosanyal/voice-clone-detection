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

    fused = fusion.fuse("s1", 0, 0, results, previous_smoothed_score=None, config=CONFIG)

    # all three at 0.5 -> weighted sum is 0.5 regardless of weight split
    assert fused.raw_score_0_100 == 50.0
    assert fused.smoothed_score_0_100 == 50.0  # no previous score -> raw == smoothed


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

    fused = fusion.fuse("s1", 0, 0, results, previous_smoothed_score=None, config=CONFIG)

    assert fused.formula_version == CONFIG["formula_version"]


def test_abstained_signal_is_renormalised_not_zeroed():
    fusion = WeightedSumFusion()
    # third_signal abstains; acoustic=1.0, prosodic=1.0 should renormalise
    # to weight 0.75/0.25 of each other and still reach 1.0*100, NOT be
    # dragged down by treating the abstained signal as a zero.
    results = [_result("acoustic", 1.0), _result("prosodic", 1.0), _result("third_signal", None)]

    fused = fusion.fuse("s1", 0, 0, results, previous_smoothed_score=None, config=CONFIG)

    assert fused.raw_score_0_100 == 100.0
    third_signal_component = next(c for c in fused.components if c.name == "third_signal")
    assert third_signal_component.abstained is True
    assert third_signal_component.weight_effective == 0.0
    assert third_signal_component.contribution == 0.0


def test_ema_smoothing_moves_toward_new_raw_score():
    fusion = WeightedSumFusion()
    results = [_result("acoustic", 1.0), _result("prosodic", 1.0), _result("third_signal", 1.0)]

    fused = fusion.fuse("s1", 1, 500, results, previous_smoothed_score=0.0, config=CONFIG)

    # alpha=0.5: smoothed = 0.5*100 + 0.5*0 = 50.0
    assert fused.smoothed_score_0_100 == 50.0


def test_bands_are_inclusive_at_boundaries():
    fusion = WeightedSumFusion()
    zero_results = [_result("acoustic", 0.0), _result("prosodic", 0.0), _result("third_signal", 0.0)]
    high_results = [_result("acoustic", 1.0), _result("prosodic", 1.0), _result("third_signal", 1.0)]

    low = fusion.fuse("s1", 0, 0, zero_results, previous_smoothed_score=None, config=CONFIG)
    high = fusion.fuse("s1", 0, 0, high_results, previous_smoothed_score=None, config=CONFIG)

    assert low.band == Band.LOW
    assert high.band == Band.HIGH


def test_component_contributions_sum_to_raw_score():
    fusion = WeightedSumFusion()
    results = [_result("acoustic", 0.9), _result("prosodic", 0.3), _result("third_signal", 0.1)]

    fused = fusion.fuse("s1", 0, 0, results, previous_smoothed_score=None, config=CONFIG)

    total_contribution = sum(c.contribution for c in fused.components)
    assert abs(total_contribution * 100 - fused.raw_score_0_100) < 0.01

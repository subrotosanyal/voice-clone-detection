"""WeightedSumFusion — the default risk formula.

    fused = sum(weight_effective[i] * score[i]) for every non-abstaining
            detector i, where weight_effective renormalises the configured
            weights of the detectors that DID produce a score, so they
            always sum to 1.0 — an abstained signal never silently drags
            the score toward zero.

Then: EMA-smoothed across the session's windows, banded per the
configured thresholds. Every number in the returned FusedScore is derived
only from `results`, `previous_smoothed_score`, and `config` — nothing
hidden, nothing time-dependent — so re-running fuse() with the same three
inputs always reproduces the same FusedScore.

To change the formula itself (different smoothing, non-linear combination,
a learned meta-model), write a new class against FusionPort and point
`fusion.class` in config/risk_formula.yaml at it. This file is not special
— it's just the default.
"""
from __future__ import annotations

from typing import Any, Optional

from app.domain.models import Band, ComponentContribution, DetectorResult, FusedScore


class WeightedSumFusion:
    version = "0.1.0"

    def fuse(
        self,
        session_id: str,
        seq: int,
        window_start_ms: int,
        results: list[DetectorResult],
        previous_smoothed_score: Optional[float],
        config: dict[str, Any],
    ) -> FusedScore:
        detector_entries = config["detectors"]
        if len(detector_entries) != len(results):
            raise ValueError(
                f"fuse() received {len(results)} detector results but config lists "
                f"{len(detector_entries)} detectors — the engine must call every "
                f"configured detector, in config order, exactly once per window."
            )
        # Matched by POSITION, not by name-string parsing: a detector's own
        # `.name` (e.g. "acoustic_spectral_flatness") need not match its
        # config entry name (e.g. "acoustic"), and a composite detector like
        # ThirdSignalRouter tags its result name with the active sub-mode
        # (e.g. "third_signal[auto:...]") — position is the only thing both
        # sides agree on. This relies on Engine.score_window() calling
        # self.pipeline.detectors (built from config["detectors"] in order)
        # and passing results through in that same order, unchanged.
        active = [
            (r, entry["name"], float(entry["weight"])) for r, entry in zip(results, detector_entries)
        ]
        configured_weight_sum = sum(w for _, _, w in active) or 1.0

        non_abstained_weight = sum(w for r, _, w in active if r.score is not None)
        renorm_factor = (1.0 / non_abstained_weight) if non_abstained_weight > 0 else 0.0

        components: list[ComponentContribution] = []
        raw_score = 0.0
        for result, config_name, configured_weight in active:
            abstained = result.score is None
            weight_effective = 0.0 if abstained else configured_weight * renorm_factor
            contribution = 0.0 if abstained else weight_effective * result.score
            raw_score += contribution
            components.append(
                ComponentContribution(
                    # `name` is the stable config-entry name (what you'd set
                    # `weight:` on in risk_formula.yaml) — the underlying
                    # implementation's own name/version is preserved in
                    # `detail` so a score is still fully traceable to the
                    # exact code that produced it.
                    name=config_name,
                    raw_score=result.score,
                    weight_configured=configured_weight / configured_weight_sum,
                    weight_effective=weight_effective,
                    contribution=contribution,
                    abstained=abstained,
                    detail={
                        "detector_name": result.detector_name,
                        "detector_version": result.detector_version,
                        "abstain_reason": result.abstain_reason,
                        **result.detail,
                    },
                )
            )

        raw_score_0_100 = round(raw_score * 100, 2)

        smoothing_alpha = float(config["fusion"].get("smoothing_alpha", 0.35))
        if previous_smoothed_score is None:
            smoothed_score_0_100 = raw_score_0_100
        else:
            smoothed_score_0_100 = round(
                smoothing_alpha * raw_score_0_100 + (1 - smoothing_alpha) * previous_smoothed_score, 2
            )

        bands_cfg = config["bands"]
        if smoothed_score_0_100 <= bands_cfg["low_max"]:
            band = Band.LOW
            recommended_action = "None — monitor silently."
        elif smoothed_score_0_100 <= bands_cfg["elevated_max"]:
            band = Band.ELEVATED
            recommended_action = "Show a quiet on-screen advisory. Do not interrupt the call."
        else:
            band = Band.HIGH
            recommended_action = (
                "Block the pending sensitive action. Require call-back on a "
                "known number or a second approver."
            )

        third_signal_mode = "unknown"
        for d in config["detectors"]:
            if d["name"] == "third_signal":
                third_signal_mode = d.get("params", {}).get("mode", "unknown")

        return FusedScore(
            session_id=session_id,
            seq=seq,
            window_start_ms=window_start_ms,
            raw_score_0_100=raw_score_0_100,
            smoothed_score_0_100=smoothed_score_0_100,
            band=band,
            # The CONFIG's formula_version (config/risk_formula.yaml), not
            # self.version (this fusion CLASS's own code version — a
            # different thing, same distinction as a detector's own
            # detector_version vs. this field). This was a real bug: it
            # used to read self.version here, so every FusedScore was
            # stamped "0.1.0" regardless of the actual config version,
            # contradicting docs/risk-model.md's documented reproducibility
            # promise that formula_version ties a score back to the exact
            # config that produced it.
            formula_version=config["formula_version"],
            third_signal_mode=third_signal_mode,
            components=components,
            recommended_action=recommended_action,
        )

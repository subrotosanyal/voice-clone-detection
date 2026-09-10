"""WeightedSumFusion — the default risk formula.

    fused = sum(weight_effective[i] * smoothed_score[i]) for every
            component that has EVER produced a score this session, where
            weight_effective renormalises the configured weights of the
            components that currently HAVE a usable value, so they
            always sum to 1.0 — a component with no value yet never
            silently drags the score toward zero.

Then: banded per the configured thresholds. Every number in the returned
FusedScore is derived only from `results`, `previous_component_smoothed`,
and `config` — nothing hidden, nothing time-dependent — so re-running
fuse() with the same three inputs always reproduces the same FusedScore.

PER-COMPONENT EMA, added 2026-09-10 (real bug this fixes — reported by a
user testing a real ~53s movie-dialogue clip through the dashboard):
smoothing used to happen ONLY on the final combined number
(`smoothed_score_0_100`) — each component's own `raw_score` was always
just that single window's instantaneous reading, and a component that
abstained for one window dropped OUT of the weighted sum entirely for
that window (its configured weight renormalised away across whoever
else was still active), even if it had scored confidently on every
prior window. For a real per-window dashboard display, this meant
"what does this detector currently think" was answered by whichever
window happened to be showing — often the LAST one, which for a short
trailing clip can be an unrepresentative near-silent tail (see
acoustic_aasist.py's and prosody_parselmouth.py's own REAL BUG notes on
exactly that failure mode). Fixed by moving the EMA to each component
individually: a component that scores this window updates its own
smoothed value (`alpha * raw + (1-alpha) * previous`, or just `raw` on
its first-ever score); a component that abstains this window CARRIES
FORWARD its last smoothed value unchanged, rather than vanishing. The
fused `smoothed_score_0_100` (the number that actually determines
`band`/`recommended_action`) is now the weighted combination of these
per-component smoothed values directly — i.e. each component's own
displayed number genuinely IS what fed the formula, not a separate,
disconnected instantaneous reading. `raw_score_0_100` is kept, unchanged
in meaning, as the "this window alone, before any smoothing" figure —
still useful for seeing an instantaneous spike, just no longer what
drives the recommended action.

To change the formula itself (different smoothing, non-linear combination,
a learned meta-model), write a new class against FusionPort and point
`fusion.class` in config/risk_formula.yaml at it. This file is not special
— it's just the default.
"""
from __future__ import annotations

from typing import Any

from app.domain.models import Band, ComponentContribution, DetectorResult, FusedScore


class WeightedSumFusion:
    version = "0.2.0"

    def fuse(
        self,
        session_id: str,
        seq: int,
        window_start_ms: int,
        results: list[DetectorResult],
        previous_component_smoothed: dict[str, float],
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

        smoothing_alpha = float(config["fusion"].get("smoothing_alpha", 0.35))

        # --- this window's raw (unsmoothed) combination — unchanged from
        # before 2026-09-10, still "what did just this instant say" -------
        raw_active_weight = sum(w for r, _, w in active if r.score is not None)
        raw_renorm_factor = (1.0 / raw_active_weight) if raw_active_weight > 0 else 0.0
        raw_score = sum(
            (w * raw_renorm_factor) * r.score for r, _, w in active if r.score is not None
        )
        raw_score_0_100 = round(raw_score * 100, 2)

        # --- per-component smoothed values (EMA, carried forward across
        # abstains) — this is the NEW state that actually drives the fused
        # smoothed_score_0_100 below ----------------------------------------
        component_smoothed: dict[str, float] = {}
        for result, config_name, _ in active:
            previous = previous_component_smoothed.get(config_name)
            if result.score is not None:
                component_smoothed[config_name] = (
                    result.score if previous is None else smoothing_alpha * result.score + (1 - smoothing_alpha) * previous
                )
            elif previous is not None:
                component_smoothed[config_name] = previous  # abstained this window — carry forward, don't drop out
            # else: never scored yet this session AND abstaining now — stays absent (no key)

        usable_weight = sum(w for _, config_name, w in active if config_name in component_smoothed)
        smoothed_renorm_factor = (1.0 / usable_weight) if usable_weight > 0 else 0.0

        components: list[ComponentContribution] = []
        smoothed_total = 0.0
        for result, config_name, configured_weight in active:
            smoothed_value = component_smoothed.get(config_name)
            has_value = smoothed_value is not None
            weight_effective = (configured_weight * smoothed_renorm_factor) if has_value else 0.0
            contribution = (weight_effective * smoothed_value) if has_value else 0.0
            smoothed_total += contribution
            components.append(
                ComponentContribution(
                    # `name` is the stable config-entry name (what you'd set
                    # `weight:` on in risk_formula.yaml) — the underlying
                    # implementation's own name/version is preserved in
                    # `detail` so a score is still fully traceable to the
                    # exact code that produced it.
                    name=config_name,
                    raw_score=result.score,
                    smoothed_score=smoothed_value,
                    weight_configured=configured_weight / configured_weight_sum,
                    weight_effective=weight_effective,
                    contribution=contribution,
                    abstained=result.score is None,
                    detail={
                        "detector_name": result.detector_name,
                        "detector_version": result.detector_version,
                        "abstain_reason": result.abstain_reason,
                        **result.detail,
                    },
                )
            )

        smoothed_score_0_100 = round(smoothed_total * 100, 2)

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

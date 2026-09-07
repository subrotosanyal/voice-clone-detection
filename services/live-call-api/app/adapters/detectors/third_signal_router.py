"""ThirdSignalRouter — the config-selectable pluggable risk model.

This is itself a DetectorPort implementation (the fusion engine and
registry treat it like any other detector — no special-casing anywhere
else in the codebase). It composes two other DetectorPort implementations
— contextual and consistency — and picks between them per `mode`:

  mode: "contextual"   always use the contextual rules detector.
  mode: "consistency"  always use the consistency (voiceprint) detector.
  mode: "auto"         (recommended default) try consistency first; if it
                       abstains — e.g. no enrollment on file — fall back to
                       contextual rather than abstaining entirely.

Pros/cons of each mode are documented in the project blueprint, §03. Set
`third_signal.params.mode` in config/risk_formula.yaml to switch — no code
change required.
"""
from __future__ import annotations

from typing import Any

from app.domain.models import AudioWindow, DetectorResult
from app.ports.detector import DetectorPort

_VALID_MODES = {"contextual", "consistency", "auto"}


class ThirdSignalRouter:
    name = "third_signal"
    version = "0.1.0"

    def __init__(
        self,
        contextual_detector: DetectorPort,
        consistency_detector: DetectorPort,
        mode: str = "auto",
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"third_signal mode must be one of {_VALID_MODES}, got {mode!r}")
        self.contextual_detector = contextual_detector
        self.consistency_detector = consistency_detector
        self.mode = mode

    def score(self, window: AudioWindow, context: dict[str, Any]) -> DetectorResult:
        if self.mode == "contextual":
            return self._tag(self.contextual_detector.score(window, context), "contextual")

        if self.mode == "consistency":
            return self._tag(self.consistency_detector.score(window, context), "consistency")

        # auto: prefer consistency, fall back to contextual on abstain
        consistency_result = self.consistency_detector.score(window, context)
        if consistency_result.score is not None:
            return self._tag(consistency_result, "consistency")

        contextual_result = self.contextual_detector.score(window, context)
        tagged = self._tag(contextual_result, "contextual")
        return DetectorResult(
            detector_name=tagged.detector_name,
            detector_version=tagged.detector_version,
            score=tagged.score,
            detail={**tagged.detail, "fell_back_from": "consistency", "fallback_reason": consistency_result.abstain_reason},
            abstain_reason=tagged.abstain_reason,
        )

    @staticmethod
    def _tag(result: DetectorResult, active_mode: str) -> DetectorResult:
        return DetectorResult(
            detector_name=f"third_signal[{active_mode}:{result.detector_name}]",
            detector_version=result.detector_version,
            score=result.score,
            detail={"active_mode": active_mode, **result.detail},
            abstain_reason=result.abstain_reason,
        )

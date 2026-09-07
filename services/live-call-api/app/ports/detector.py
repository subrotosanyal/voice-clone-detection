"""The Detector port.

Any signal that wants to contribute a component score to the fused risk
score implements this interface. The pipeline engine never imports a
concrete detector class directly — it only ever talks to this Protocol,
and instantiates concrete detectors by dotted path from
config/risk_formula.yaml (see app/adapters/registry.py).

To add a new risk factor:
  1. Write a class implementing DetectorPort (see app/adapters/detectors/
     for examples — contextual_rules.py is the simplest one to copy).
  2. Register it under `detectors:` in config/risk_formula.yaml with a
     name and a weight.
  3. Nothing else changes — the fusion step discovers it automatically.
"""
from __future__ import annotations

from typing import Any, Protocol

from app.domain.models import AudioWindow, DetectorResult


class DetectorPort(Protocol):
    """A pluggable risk signal.

    Implementations MUST be deterministic for the same (window, context,
    detector_version) — that's what makes a fused score reproducible.
    Anything that depends on wall-clock time, randomness, or external
    network state breaks that guarantee and must not be hidden inside a
    detector; if a detector genuinely needs non-deterministic state (e.g. a
    remote model endpoint), pin and record a model/version id in `detail`.
    """

    name: str
    version: str

    def score(self, window: AudioWindow, context: dict[str, Any]) -> DetectorResult:
        """Score one audio window.

        `context` carries session-level metadata that isn't audio itself —
        e.g. claimed caller identity, whether the number is known, the
        hour of day, whether a financial action is pending. A detector
        that doesn't need it (most audio-based ones) should ignore it.

        Return a DetectorResult with score=None (and abstain_reason set)
        rather than guessing, whenever the detector genuinely has nothing
        to say for this window — the fusion step is built to handle that.
        """
        ...

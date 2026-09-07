"""Builds detectors and the fusion strategy from config/risk_formula.yaml.

This is the ONE place in the codebase that turns a dotted-path string into
a live Python object. Everywhere else talks only to DetectorPort /
FusionPort. That's what makes adding a factor or swapping the formula a
config-only change — see docs/architecture.md, "Adding a new risk factor".
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.adapters.detectors.third_signal_router import ThirdSignalRouter
from app.ports.detector import DetectorPort
from app.ports.diarizer import DiarizerPort
from app.ports.fusion import FusionPort


def _load_class(dotted_path: str) -> type:
    """'pkg.module:ClassName' -> the class object."""
    module_path, _, class_name = dotted_path.partition(":")
    if not class_name:
        raise ValueError(f"expected 'module.path:ClassName', got {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_detector(entry: dict[str, Any]) -> DetectorPort:
    params = dict(entry.get("params", {}))

    if entry["name"] == "third_signal":
        # Special construction only because this one composite detector
        # needs two other detectors built first — it is still just a
        # DetectorPort as far as everything downstream is concerned.
        mode = params.pop("mode", "auto")
        contextual_cls = _load_class(params.pop("contextual_class"))
        consistency_cls = _load_class(params.pop("consistency_class"))
        return ThirdSignalRouter(
            contextual_detector=contextual_cls(**params.pop("contextual_params", {})),
            consistency_detector=consistency_cls(**params.pop("consistency_params", {})),
            mode=mode,
        )

    cls = _load_class(entry["class"])
    return cls(**params)


def _build_fusion(config: dict[str, Any]) -> FusionPort:
    cls = _load_class(config["fusion"]["class"])
    return cls()


def _build_diarizer(config: dict[str, Any]) -> DiarizerPort | None:
    """Diarization is optional — older configs (or a stripped-down local
    override) without a `diarization:` section simply don't get one, and
    POST /v1/score/file's diarize=true option becomes unavailable rather
    than erroring at startup."""
    diarization_cfg = config.get("diarization")
    if diarization_cfg is None:
        return None
    cls = _load_class(diarization_cfg["class"])
    return cls(**diarization_cfg.get("params", {}))


@dataclass(frozen=True)
class Pipeline:
    """Everything the engine needs, built once from config and reused."""

    config: dict[str, Any]
    detectors: list[DetectorPort]
    fusion: FusionPort
    diarizer: DiarizerPort | None = None


def build_pipeline(config_path: str | Path) -> Pipeline:
    config = load_config(config_path)
    detectors = [_build_detector(entry) for entry in config["detectors"]]
    fusion = _build_fusion(config)
    diarizer = _build_diarizer(config)
    return Pipeline(config=config, detectors=detectors, fusion=fusion, diarizer=diarizer)

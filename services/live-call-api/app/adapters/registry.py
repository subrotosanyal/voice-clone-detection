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
from app.ports.intent_classifier import IntentClassifierPort
from app.ports.semantic_risk_classifier import SemanticRiskClassifierPort
from app.ports.transcriber import TranscriberPort


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


def _build_transcriber(config: dict[str, Any]) -> TranscriberPort | None:
    """Transcription is optional, same reasoning as diarization above — no
    `transcription:` section means Engine.score_call() just never sets
    context["transcript"], and ContextualRulesDetector falls back to
    whatever urgency_keywords/is_financial_request a caller supplies by
    hand (its original, pre-transcription behaviour)."""
    transcription_cfg = config.get("transcription")
    if transcription_cfg is None:
        return None
    cls = _load_class(transcription_cfg["class"])
    return cls(**transcription_cfg.get("params", {}))


def _build_intent_classifier(config: dict[str, Any]) -> IntentClassifierPort | None:
    """Intent classification is optional, same reasoning as transcription
    above — no `intent_classification:` section means Engine.score_call()
    never sets context["intent_label"], and IntentRiskDetector (if it were
    ever added to `detectors:` — it is NOT there by default) would simply
    abstain. See app/adapters/intent/zero_shot_intent_classifier.py's
    HONESTY NOTE before ever adding that detector entry."""
    intent_cfg = config.get("intent_classification")
    if intent_cfg is None:
        return None
    cls = _load_class(intent_cfg["class"])
    return cls(**intent_cfg.get("params", {}))


def _build_semantic_risk_classifier(config: dict[str, Any]) -> SemanticRiskClassifierPort | None:
    """Semantic risk classification (app/adapters/semantic_risk/) is
    optional, same "omit the section to disable" reasoning as
    transcription/intent_classification above — no
    `semantic_risk_classification:` section means Engine.score_call()
    never sets context["semantic_urgency_level"], and SemanticRiskDetector
    (if it's in `detectors:` — see that entry's own comment for the
    current weight/status) simply abstains. See
    app/adapters/semantic_risk/local_llm_semantic_classifier.py's own
    HONESTY NOTE before changing the weight here."""
    semantic_cfg = config.get("semantic_risk_classification")
    if semantic_cfg is None:
        return None
    cls = _load_class(semantic_cfg["class"])
    return cls(**semantic_cfg.get("params", {}))


def _build_live_speaker_embedder(config: dict[str, Any]):
    """Live diarization (app/pipeline/live_diarization.py) is optional,
    same "omit the section to disable" convention as transcription/
    intent_classification above — no `live_diarization:` section means
    ws_router.py's LiveSpeakerTracker gets embedder=None and every window
    quietly has no `live_speaker` attached (same "abstain quietly" shape
    every other optional feature in this project already has).

    Returns only the shared, expensive-to-load EcapaEmbeddingExtractor —
    NOT a LiveSpeakerTracker itself, since that class holds PER-SESSION
    state (speaker centroids) and must be instantiated fresh per
    WebSocket connection, not once at startup. ws_router.py reads the
    other tuning params (similarity_threshold, ema_alpha, max_speakers,
    floor_rms) straight out of `live_diarization.params` itself when it
    constructs each session's tracker — no need to thread them through
    Pipeline/Engine, they're plain primitives, not an expensive model.

    Deliberately builds its OWN EcapaEmbeddingExtractor instance rather
    than reusing `diarizer`'s internal one (if `diarization:` is also
    configured) — see live_diarization.py's own docstring for why that's
    an honest trade-off (a second copy of the model in memory when both
    features are enabled), not an oversight.
    """
    live_diarization_cfg = config.get("live_diarization")
    if live_diarization_cfg is None:
        return None
    from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor

    params = live_diarization_cfg.get("params", {})
    return EcapaEmbeddingExtractor(source=params.get("embedding_model_source", "speechbrain/spkrec-ecapa-voxceleb"))


@dataclass(frozen=True)
class Pipeline:
    """Everything the engine needs, built once from config and reused."""

    config: dict[str, Any]
    detectors: list[DetectorPort]
    fusion: FusionPort
    diarizer: DiarizerPort | None = None
    transcriber: TranscriberPort | None = None
    intent_classifier: IntentClassifierPort | None = None
    semantic_risk_classifier: SemanticRiskClassifierPort | None = None
    live_speaker_embedder: Any = None  # Optional[EcapaEmbeddingExtractor], lazily typed to avoid an eager speechbrain import


def build_pipeline(config_path: str | Path) -> Pipeline:
    config = load_config(config_path)
    detectors = [_build_detector(entry) for entry in config["detectors"]]
    fusion = _build_fusion(config)
    diarizer = _build_diarizer(config)
    transcriber = _build_transcriber(config)
    intent_classifier = _build_intent_classifier(config)
    semantic_risk_classifier = _build_semantic_risk_classifier(config)
    live_speaker_embedder = _build_live_speaker_embedder(config)
    return Pipeline(
        config=config,
        detectors=detectors,
        fusion=fusion,
        diarizer=diarizer,
        transcriber=transcriber,
        intent_classifier=intent_classifier,
        semantic_risk_classifier=semantic_risk_classifier,
        live_speaker_embedder=live_speaker_embedder,
    )

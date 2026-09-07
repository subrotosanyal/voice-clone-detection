"""Orchestrates one window through: detectors -> fusion -> log.

This is the only place that knows about ALL of: windowing, the detector
list, the fusion strategy, and per-session smoothing state. Everything it
calls is a port; everything it produces is logged in full before it's
returned, so the log is always at least as complete as the API response —
you can debug from the logs alone, without re-running anything.

"Parallel" in the blueprint's sense means the three detectors are
independent of each other, not that this v0 necessarily runs them on
separate threads — they're small, fast, numpy-bound computations, and
running them in a plain loop keeps this trivially debuggable (a stack
trace points at exactly one detector). If a future detector is slow
(a real network-bound model call), thread/process them behind the same
DetectorPort — nothing else here needs to change.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from app.adapters.registry import Pipeline
from app.domain.models import AudioWindow, FusedScore
from app.logging_setup import get_logger
from app.pipeline.windowing import make_windows
from app.ports.history_store import HistoryStorePort

logger = get_logger(component="pipeline_engine")


class SessionStore:
    """In-memory per-session smoothing state.

    Deliberately not persisted anywhere yet — a restart resets every
    session's EMA to a fresh start. That's fine for a single-process
    demo; swap for Redis (keyed by session_id) the moment more than one
    API replica needs to share session state. See docs/architecture.md,
    "What's not built yet".
    """

    def __init__(self) -> None:
        self._last_smoothed: dict[str, float] = {}

    def get_previous(self, session_id: str) -> Optional[float]:
        return self._last_smoothed.get(session_id)

    def update(self, session_id: str, smoothed_score: float) -> None:
        self._last_smoothed[session_id] = smoothed_score

    def reset(self, session_id: str) -> None:
        self._last_smoothed.pop(session_id, None)


class Engine:
    def __init__(
        self,
        pipeline: Pipeline,
        session_store: Optional[SessionStore] = None,
        history: Optional[HistoryStorePort] = None,
    ) -> None:
        self.pipeline = pipeline
        self.sessions = session_store or SessionStore()
        self.history = history

    def score_window(self, window: AudioWindow, context: dict[str, Any]) -> FusedScore:
        results = [detector.score(window, context) for detector in self.pipeline.detectors]

        previous = self.sessions.get_previous(window.session_id)
        fused = self.pipeline.fusion.fuse(
            session_id=window.session_id,
            seq=window.seq,
            window_start_ms=window.window_start_ms,
            results=results,
            previous_smoothed_score=previous,
            config=self.pipeline.config,
        )
        self.sessions.update(window.session_id, fused.smoothed_score_0_100)

        logger.info(
            "risk_score_computed",
            session_id=fused.session_id,
            seq=fused.seq,
            window_start_ms=fused.window_start_ms,
            raw_score=fused.raw_score_0_100,
            smoothed_score=fused.smoothed_score_0_100,
            band=fused.band.value,
            formula_version=fused.formula_version,
            third_signal_mode=fused.third_signal_mode,
            components=[
                {
                    "name": c.name,
                    "raw_score": c.raw_score,
                    "weight_configured": c.weight_configured,
                    "weight_effective": c.weight_effective,
                    "contribution": c.contribution,
                    "abstained": c.abstained,
                    "detail": c.detail,
                }
                for c in fused.components
            ],
        )

        if self.history is not None:
            try:
                self.history.save(fused)
            except Exception:  # noqa: BLE001 — history is best-effort, never blocks live scoring
                logger.exception("history_save_failed", session_id=fused.session_id, seq=fused.seq)

        return fused

    def score_windows(
        self, windows: Iterable[AudioWindow], context: dict[str, Any]
    ) -> list[FusedScore]:
        return [self.score_window(w, context) for w in windows]

    def score_call(
        self,
        session_id: str,
        samples,
        sample_rate: int,
        context: dict[str, Any],
    ) -> list[FusedScore]:
        """Convenience: window a whole buffer and score every window.

        Used by the one-shot REST endpoint (POST /v1/score/file) and by
        tests — the natural entrypoint when you have a complete recording
        rather than a live stream.
        """
        self.sessions.reset(session_id)
        windowing_cfg = self.pipeline.config["windowing"]
        windows = make_windows(
            session_id=session_id,
            samples=samples,
            sample_rate=sample_rate,
            window_ms=windowing_cfg["window_ms"],
            hop_ms=windowing_cfg["hop_ms"],
        )
        return self.score_windows(windows, context)

"""The History Store port.

Persists every FusedScore so a session can be looked up and replayed
later — from a browser tab, a script, or an incident review — without
having to grep container logs for it. This is deliberately a separate
port from FusionPort/DetectorPort: fusion is about producing a score;
this is about remembering one that was already produced. A detector or
the fusion strategy never touches this — only Engine does, once per
window, after fusion.

Writes here must never be able to break the live scoring path: if
storage is down, the engine logs it and keeps serving scores. See
Engine.score_window() in app/pipeline/engine.py.
"""
from __future__ import annotations

from typing import Protocol

from app.domain.models import FusedScore, SessionSummary


class HistoryStorePort(Protocol):
    def save(self, fused: FusedScore) -> None:
        """Persist one window's FusedScore."""
        ...

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        """Most recent sessions first."""
        ...

    def get_session(self, session_id: str) -> list[FusedScore]:
        """Every window for one session, in seq order. Empty list if unknown."""
        ...

    def delete_session(self, session_id: str) -> bool:
        """Returns True if the session existed and was deleted."""
        ...

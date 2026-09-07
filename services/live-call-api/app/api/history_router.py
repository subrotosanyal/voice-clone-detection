"""Session history — list, replay, and delete past sessions.

Every window scored by the engine (file upload or live stream) is
persisted by app/pipeline/engine.py right after fusion. These routes only
ever read from / delete via HistoryStorePort — see
app/adapters/history/sqlite_store.py for the default implementation and
docs/architecture.md, "Session history", for why SQLite is enough for now.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import FusedScoreOut, SessionDetailResponse, SessionSummaryOut

router = APIRouter(prefix="/v1/sessions", tags=["history"])


@router.get("", response_model=list[SessionSummaryOut])
def list_sessions(request: Request, limit: int = 50) -> list[SessionSummaryOut]:
    summaries = request.app.state.history.list_sessions(limit=limit)
    return [SessionSummaryOut.from_domain(s) for s in summaries]


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(request: Request, session_id: str) -> SessionDetailResponse:
    trace = request.app.state.history.get_session(session_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"no session found with id {session_id!r}")
    trace_out = [FusedScoreOut.from_domain(f) for f in trace]
    return SessionDetailResponse(
        session_id=session_id,
        window_count=len(trace_out),
        final=trace_out[-1],
        trace=trace_out,
    )


@router.delete("/{session_id}")
def delete_session(request: Request, session_id: str) -> dict[str, str]:
    existed = request.app.state.history.delete_session(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail=f"no session found with id {session_id!r}")
    return {"deleted": session_id}

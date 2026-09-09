"""Session history — list, replay, and delete past sessions.

Every window scored by the engine (file upload or live stream) is
persisted by app/pipeline/engine.py right after fusion. These routes only
ever read from / delete via HistoryStorePort — see
app/adapters/history/sqlite_store.py for the default implementation and
docs/architecture.md, "Session history", for why SQLite is enough for now.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import FusedScoreOut, SessionDetailResponse, SessionSummaryOut, SpeakerScoreOut

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


@router.get("/{session_id}/speakers", response_model=list[SpeakerScoreOut])
def get_session_speakers(request: Request, session_id: str) -> list[SpeakerScoreOut]:
    """A per-speaker breakdown for `session_id`, if one was ever computed
    — via POST /v1/score/file?diarize=true (immediate), or the live
    WebSocket path's "diarize on hangup" (app/api/ws_router.py, computed
    once the call ends). Empty list (not 404) when none exists: a session
    with no diarized breakdown is a perfectly normal, common case (never
    requested, no diarizer configured, or only one speaker detected), not
    an error — same "abstain quietly" shape the rest of this project uses.

    See Engine.diarize_and_score() (app/pipeline/engine.py) for how these
    get persisted: a small HistoryStorePort.speaker_call_summaries record
    per speaker (segment_count/total_duration_ms — the one part not
    re-derivable from fused_scores alone), then each speaker's own full
    trace is read back the same way any other session's is.
    """
    summaries = request.app.state.history.list_speaker_summaries(session_id)
    results: list[SpeakerScoreOut] = []
    for s in summaries:
        trace = request.app.state.history.get_session(s.speaker_session_id)
        if not trace:
            continue  # summary exists but the trace was separately deleted — skip, don't error
        trace_out = [FusedScoreOut.from_domain(f) for f in trace]
        results.append(
            SpeakerScoreOut(
                speaker_label=s.speaker_label,
                session_id=s.speaker_session_id,
                segment_count=s.segment_count,
                total_duration_ms=s.total_duration_ms,
                final=trace_out[-1],
                trace=trace_out,
            )
        )
    return results

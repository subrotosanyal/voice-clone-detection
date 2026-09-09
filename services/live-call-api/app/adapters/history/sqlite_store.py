"""SqliteHistoryStore — the default HistoryStorePort implementation.

One file, no server process, no extra dependency (sqlite3 is in the
Python standard library). That's enough for a single-instance demo
service; it stops being enough the moment more than one API replica
needs to share history, or query volume grows past what one file on one
disk can serve — at that point, swap this for a Postgres-backed
implementation of the same HistoryStorePort. Nothing outside app/main.py's
wiring needs to change — Engine and the API routes only ever talk to
HistoryStorePort.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.domain.models import Band, ComponentContribution, FusedScore, SessionSummary, SpeakerCallSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fused_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    window_start_ms INTEGER NOT NULL,
    raw_score REAL NOT NULL,
    smoothed_score REAL NOT NULL,
    band TEXT NOT NULL,
    formula_version TEXT NOT NULL,
    third_signal_mode TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    components_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fused_scores_session ON fused_scores(session_id, seq);

-- See HistoryStorePort.save_speaker_summary()'s own docstring: the one
-- part of a diarized speaker's result (app/domain/models.py's
-- SpeakerCallResult) that isn't re-derivable from fused_scores alone.
CREATE TABLE IF NOT EXISTS speaker_call_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_session_id TEXT NOT NULL,
    speaker_label TEXT NOT NULL,
    speaker_session_id TEXT NOT NULL,
    segment_count INTEGER NOT NULL,
    total_duration_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_speaker_summaries_base ON speaker_call_summaries(base_session_id, id);
"""


class SqliteHistoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # A single shared connection + lock is deliberately simple: SQLite
        # serialises writes anyway, and this service's whole point is a
        # low write-rate demo tool, not a high-throughput store.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, fused: FusedScore) -> None:
        components_json = json.dumps(
            [
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
            ]
        )
        with self._lock:
            self._conn.execute(
                """INSERT INTO fused_scores
                   (session_id, seq, window_start_ms, raw_score, smoothed_score, band,
                    formula_version, third_signal_mode, recommended_action, components_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fused.session_id,
                    fused.seq,
                    fused.window_start_ms,
                    fused.raw_score_0_100,
                    fused.smoothed_score_0_100,
                    fused.band.value,
                    fused.formula_version,
                    fused.third_signal_mode,
                    fused.recommended_action,
                    components_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    fs.session_id,
                    COUNT(*) AS window_count,
                    MIN(fs.created_at) AS started_at,
                    MAX(fs.created_at) AS ended_at,
                    (SELECT smoothed_score FROM fused_scores fs2
                     WHERE fs2.session_id = fs.session_id ORDER BY fs2.seq DESC LIMIT 1) AS final_score,
                    (SELECT band FROM fused_scores fs2
                     WHERE fs2.session_id = fs.session_id ORDER BY fs2.seq DESC LIMIT 1) AS final_band
                FROM fused_scores fs
                GROUP BY fs.session_id
                ORDER BY ended_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            SessionSummary(
                session_id=r[0],
                window_count=r[1],
                started_at=r[2],
                ended_at=r[3],
                final_smoothed_score=r[4],
                final_band=Band(r[5]),
            )
            for r in rows
        ]

    def get_session(self, session_id: str) -> list[FusedScore]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT session_id, seq, window_start_ms, raw_score, smoothed_score, band,
                          formula_version, third_signal_mode, recommended_action, components_json
                   FROM fused_scores WHERE session_id = ? ORDER BY seq ASC""",
                (session_id,),
            ).fetchall()
        return [_row_to_fused_score(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM fused_scores WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def save_speaker_summary(self, summary: SpeakerCallSummary) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO speaker_call_summaries
                   (base_session_id, speaker_label, speaker_session_id, segment_count,
                    total_duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    summary.base_session_id,
                    summary.speaker_label,
                    summary.speaker_session_id,
                    summary.segment_count,
                    summary.total_duration_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def list_speaker_summaries(self, base_session_id: str) -> list[SpeakerCallSummary]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT base_session_id, speaker_label, speaker_session_id, segment_count, total_duration_ms
                   FROM speaker_call_summaries WHERE base_session_id = ? ORDER BY id ASC""",
                (base_session_id,),
            ).fetchall()
        return [
            SpeakerCallSummary(
                base_session_id=r[0],
                speaker_label=r[1],
                speaker_session_id=r[2],
                segment_count=r[3],
                total_duration_ms=r[4],
            )
            for r in rows
        ]


def _row_to_fused_score(row: tuple) -> FusedScore:
    (
        session_id,
        seq,
        window_start_ms,
        raw_score,
        smoothed_score,
        band,
        formula_version,
        third_signal_mode,
        recommended_action,
        components_json,
    ) = row
    components = [
        ComponentContribution(
            name=c["name"],
            raw_score=c["raw_score"],
            weight_configured=c["weight_configured"],
            weight_effective=c["weight_effective"],
            contribution=c["contribution"],
            abstained=c["abstained"],
            detail=c["detail"],
        )
        for c in json.loads(components_json)
    ]
    return FusedScore(
        session_id=session_id,
        seq=seq,
        window_start_ms=window_start_ms,
        raw_score_0_100=raw_score,
        smoothed_score_0_100=smoothed_score,
        band=Band(band),
        formula_version=formula_version,
        third_signal_mode=third_signal_mode,
        components=components,
        recommended_action=recommended_action,
    )

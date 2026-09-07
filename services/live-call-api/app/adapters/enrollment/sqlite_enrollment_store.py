"""SqliteEnrollmentStore — the default EnrollmentStorePort implementation.

Same rationale as app/adapters/history/sqlite_store.py: one file, stdlib
sqlite3, no extra dependency, fine for a single-instance demo service.

Schema — one row per enrolled identity:
    identity          TEXT PRIMARY KEY   -- caller-claimed identity string,
                                          -- e.g. "alice@example.com" or an
                                          -- account/customer id — whatever
                                          -- the calling context's
                                          -- claimed_identity field uses
    embedding_json    TEXT NOT NULL      -- JSON list[float], 192 numbers
                                          -- (ECAPA-TDNN's embedding dim)
    embedding_model    TEXT NOT NULL     -- e.g. "speechbrain/spkrec-ecapa-voxceleb"
                                          -- — recorded so a future model
                                          -- swap can detect stale embeddings
    enrolled_at        TEXT NOT NULL     -- ISO 8601 UTC, set once
    updated_at         TEXT NOT NULL     -- ISO 8601 UTC, bumped on re-enroll

Only the derived embedding vector is stored — never raw audio. Re-enrolling
an identity overwrites its row (INSERT ... ON CONFLICT DO UPDATE), so there
is at most one voiceprint per identity at any time.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.domain.models import EnrollmentSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voiceprints (
    identity TEXT PRIMARY KEY,
    embedding_json TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    enrolled_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class SqliteEnrollmentStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def enroll(self, identity: str, embedding: np.ndarray, embedding_model: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        embedding_json = json.dumps([float(v) for v in embedding.tolist()])
        with self._lock:
            self._conn.execute(
                """INSERT INTO voiceprints (identity, embedding_json, embedding_model, enrolled_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(identity) DO UPDATE SET
                       embedding_json = excluded.embedding_json,
                       embedding_model = excluded.embedding_model,
                       updated_at = excluded.updated_at""",
                (identity, embedding_json, embedding_model, now, now),
            )
            self._conn.commit()

    def get_embedding(self, identity: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding_json FROM voiceprints WHERE identity = ?", (identity,)
            ).fetchone()
        if row is None:
            return None
        return np.array(json.loads(row[0]), dtype=np.float32)

    def list_enrollments(self) -> list[EnrollmentSummary]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT identity, embedding_model, enrolled_at, updated_at FROM voiceprints ORDER BY updated_at DESC"
            ).fetchall()
        return [
            EnrollmentSummary(identity=r[0], embedding_model=r[1], enrolled_at=r[2], updated_at=r[3])
            for r in rows
        ]

    def delete(self, identity: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM voiceprints WHERE identity = ?", (identity,))
            self._conn.commit()
            return cur.rowcount > 0

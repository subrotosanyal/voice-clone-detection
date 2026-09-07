"""The Enrollment Store port.

Persists enrolled voiceprints (identity -> speaker embedding) so the
voiceprint-consistency detector can compare a live call's embedding against
one recorded earlier with the caller's consent. Deliberately a separate
port from HistoryStorePort — enrollment is long-lived reference data an
operator manages explicitly (enroll/list/delete), not an append-only log of
past calls.

Consent and retention are policy, not code, but this port is shaped to make
the honest policy easy: `delete` is a hard delete (no soft-delete/undo), and
nothing here stores raw audio — only the derived embedding vector, which
cannot be played back as speech.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from app.domain.models import EnrollmentSummary


class EnrollmentStorePort(Protocol):
    def enroll(self, identity: str, embedding: np.ndarray, embedding_model: str) -> None:
        """Store (or overwrite) the voiceprint for `identity`."""
        ...

    def get_embedding(self, identity: str) -> Optional[np.ndarray]:
        """The enrolled embedding for `identity`, or None if never enrolled."""
        ...

    def list_enrollments(self) -> list[EnrollmentSummary]:
        """Every enrolled identity's metadata (never the embedding itself)."""
        ...

    def delete(self, identity: str) -> bool:
        """Returns True if `identity` existed and was deleted."""
        ...

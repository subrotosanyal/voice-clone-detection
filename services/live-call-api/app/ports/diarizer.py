"""The Diarizer port.

"Who spoke when" — segments a whole recording into contiguous stretches
attributed to distinct speaker clusters. Deliberately a separate port from
DetectorPort: a diarizer doesn't produce a risk score, it produces a
partition of the audio that the rest of the pipeline (Engine.score_call)
can then be run against once per speaker — see
app/adapters/diarization/embedding_cluster_diarizer.py and the fan-out
logic in app/api/http_router.py's POST /v1/score/file (diarize=true).

Scope note: diarization is currently wired into the file-upload path only.
Real-time diarization on the live WebSocket stream is a substantially
harder problem (speaker clusters have to be discovered incrementally, from
partial audio) and is not implemented — see docs/architecture.md, "What's
not built yet".
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.domain.models import SpeakerSegment


class DiarizerPort(Protocol):
    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        """Returns contiguous speaker segments covering the whole input,
        in chronological order. A single-speaker recording still returns
        one (or more, if energy gaps split it) segment(s), all with the
        same speaker_label."""
        ...

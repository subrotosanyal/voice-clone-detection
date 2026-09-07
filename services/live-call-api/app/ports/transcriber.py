"""The Transcriber port.

Converts a call's audio into text — used only to feed the contextual third
signal's urgency/financial-request keyword detection (see
app/adapters/detectors/contextual_rules.py), never for anything else. Not
a DetectorPort: a transcriber doesn't produce a risk score, it produces
text that ContextualRulesDetector reads out of `context["transcript"]`.

Scope note: wired into the file-upload path only (see
app/pipeline/engine.py's score_call()), same reasoning as diarization
(app/ports/diarizer.py) — transcribing a live, still-arriving stream
incrementally is a substantially harder problem, deferred for now.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.domain.models import TranscriptResult


class TranscriberPort(Protocol):
    name: str
    version: str

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        """Returns a TranscriptResult for the whole given audio buffer.
        Returns empty text (not an error) when no speech is found."""
        ...

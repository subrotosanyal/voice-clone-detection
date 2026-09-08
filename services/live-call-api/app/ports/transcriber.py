"""The Transcriber port.

Converts a call's audio into text — used only to feed the contextual third
signal's urgency/financial-request keyword detection (see
app/adapters/detectors/contextual_rules.py), never for anything else. Not
a DetectorPort: a transcriber doesn't produce a risk score, it produces
text that ContextualRulesDetector reads out of `context["transcript"]`.

Scope note: the file-upload path (app/pipeline/engine.py's score_call())
transcribes the whole buffer once, up front. The live WebSocket path
(added 2026-09-08, see app/pipeline/live_transcription.py) instead
transcribes periodically in the background as audio accumulates — a
different calling pattern, but the same port; nothing here changed to
support it. Diarization (app/ports/diarizer.py) remains file-upload-only
for the harder reason stated there (incremental speaker clustering from
partial audio); transcription's earlier "substantially harder" framing
for live no longer applies now that periodic re-transcription is built.
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

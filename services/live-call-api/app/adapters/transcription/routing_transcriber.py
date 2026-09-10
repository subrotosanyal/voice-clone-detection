"""RoutingTranscriber — language-gated dispatch between an Indic-language
transcriber (vexyl_stt_transcriber.py's VexylSttTranscriber) and a general
fallback (whisper_transcriber.py's WhisperTranscriber), so this project
gets vexyl-stt's real accuracy/speed advantage on Hindi/Marathi/etc.
without vexyl-stt's real failure mode on English (see
vexyl_stt_transcriber.py's own WHY note for both — measured by hand on 6
real/synthesized clips, not assumed).

THE PROBLEM THIS SOLVES: neither transcriber gives you a language signal
in a form the other can use BEFORE transcribing. vexyl-stt needs the
language up front (part of its WS handshake); Whisper only reports the
language it detected AFTER a full autoregressive decode — and
TranscriberPort.transcribe(samples, sample_rate) carries no language hint
at all (checked: no caller in this codebase supplies one). So a router
can't decide "which transcriber" from a detected language without first
paying for a transcription — UNLESS the detection step is cheap. Whisper
exposes exactly that: `WhisperTranscriber.detect_language()` runs ONE
encoder forward pass (no decoding loop), a small fraction of the cost of
a full transcribe() — see that method's own docstring.

DISPATCH:
  1. Cheap LID via `lid_transcriber.detect_language()` (duck-typed — see
     below).
  2. If the detected language is in `indic_languages` AND the Indic path
     returns non-empty text, use that result.
  3. Otherwise (undetected language, a language outside
     `indic_languages`, or the Indic path came back empty — e.g. vexyl-stt
     is unreachable) — fall back to `fallback_transcriber.transcribe()`.
     Same "abstain rather than lose the call" discipline as every other
     best-effort adapter here: a language-ID miss or a transient
     vexyl-stt outage degrades to the previous (fallback-only) behaviour,
     never to silence.

REAL GAP found and fixed while wiring config/risk_formula.yaml's LIVE
`live_transcription:` example, before it ever shipped: on the file-upload
path, `fallback_transcriber` is WhisperTranscriber, which loads its own
model in-process and can do cheap LID directly. On the LIVE path,
`fallback_transcriber` is WhisperLiveTranscriber — a WS client with no
local model at all, so it has nothing to run detect_language() against
(collabora/WhisperLive's own protocol only returns a language as part of
a full transcribe cycle, no cheap LID-only mode). Reusing
`fallback_transcriber` for LID unconditionally, as an earlier version of
this class did, would have made live routing crash the moment it was
enabled. Fixed with a THIRD, optional constructor param, `lid_transcriber`
— defaults to `fallback_transcriber` (the file-upload case, unchanged),
but on the live path must be given its own in-process WhisperTranscriber
instance dedicated to LID only (see config/risk_formula.yaml's own
`live_transcription:` example for exactly how). That does mean one more
loaded Whisper model in memory on the live path specifically — but only
for a single cheap encoder pass per window, not the repeated full
autoregressive decode cycles WhisperLive was built to move out of this
process (see whisperlive_transcriber.py's own docstring for that
original bug) — a real trade-off, not the same problem recurring.

`lid_transcriber` (or `fallback_transcriber`, if `lid_transcriber` isn't
given) must be duck-type compatible with WhisperTranscriber (a
`detect_language(samples, sample_rate) -> str | None` method) — checked
at construction time with an actionable error, not discovered at call
time. `indic_transcriber` must implement `transcribe_lang(samples,
sample_rate, lang) -> TranscriptResult` alongside standard
TranscriberPort methods — see VexylSttTranscriber.transcribe_lang() for
why the language-aware entry point is a separate method from
transcribe().

SCOPE NOTE: this class itself never imports/constructs its two
sub-transcribers by dotted path — app/adapters/registry.py remains "the
ONE place in the codebase that turns a dotted-path string into a live
Python object" (see that module's own top-of-file docstring); this class
only composes whatever two already-built TranscriberPort-shaped objects
it's handed, same pattern as ThirdSignalRouter composing two already-
built detectors."""
from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from app.domain.models import TranscriptResult
from app.logging_setup import get_logger

logger = get_logger(component="routing_transcriber")

# vexyl-stt's own supported set (see vexyl_stt_transcriber.py's
# SUPPORTED_LANGUAGES) — kept as this module's own default so a config
# that omits `indic_languages` still routes correctly, but always
# overridable per-deployment via config/risk_formula.yaml's `params:`.
_DEFAULT_INDIC_LANGUAGES = frozenset(
    {"ml", "hi", "ta", "te", "kn", "bn", "gu", "mr", "pa", "or", "as", "ur", "sa", "ne"}
)


class _LanguageDetectingTranscriber(Protocol):
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult: ...
    def detect_language(self, samples: np.ndarray, sample_rate: int) -> Optional[str]: ...


class _LanguageAwareTranscriber(Protocol):
    def transcribe_lang(self, samples: np.ndarray, sample_rate: int, lang: str) -> TranscriptResult: ...


class RoutingTranscriber:
    name = "routing_transcriber"
    version = "1.0.0"

    def __init__(
        self,
        indic_transcriber: _LanguageAwareTranscriber,
        fallback_transcriber: Any,
        indic_languages: Optional[list[str]] = None,
        lid_transcriber: Optional[_LanguageDetectingTranscriber] = None,
    ) -> None:
        self.indic_transcriber = indic_transcriber
        self.fallback_transcriber = fallback_transcriber
        self.lid_transcriber = lid_transcriber if lid_transcriber is not None else fallback_transcriber
        if not hasattr(self.lid_transcriber, "detect_language"):
            raise TypeError(
                "RoutingTranscriber needs a language-ID source with a detect_language() "
                f"method; {type(self.lid_transcriber).__name__} (from fallback_transcriber, "
                "since no lid_transcriber was given) doesn't have one — this is expected for "
                "WhisperLiveTranscriber on the live path (it has no local model to run LID "
                "against). Pass `lid_class`/`lid_params` explicitly in config, pointing at an "
                "in-process app.adapters.transcription.whisper_transcriber:WhisperTranscriber "
                "instance dedicated to language ID — see config/risk_formula.yaml's own "
                "live_transcription: example."
            )
        self.indic_languages = frozenset(indic_languages) if indic_languages is not None else _DEFAULT_INDIC_LANGUAGES

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        if samples.size == 0:
            return self._empty()

        detected_lang = self.lid_transcriber.detect_language(samples, sample_rate)

        if detected_lang in self.indic_languages:
            result = self.indic_transcriber.transcribe_lang(samples, sample_rate, detected_lang)
            if result.text:
                logger.info("routed_to_indic_transcriber", detected_language=detected_lang)
                return result
            # Indic path came back empty (server down, timed out, or
            # genuinely no speech) — fall through to the fallback rather
            # than trust an empty result from the language we thought
            # was right; the fallback's own auto-detect re-checks from
            # scratch, so a real "no speech" case still comes back empty.
            logger.warning(
                "indic_transcriber_returned_empty_falling_back",
                detected_language=detected_lang,
            )
        else:
            logger.info("routed_to_fallback_transcriber", detected_language=detected_lang)

        return self.fallback_transcriber.transcribe(samples, sample_rate)

    def _empty(self) -> TranscriptResult:
        return TranscriptResult(text="", language=None, detector_name=self.name, detector_version=self.version)

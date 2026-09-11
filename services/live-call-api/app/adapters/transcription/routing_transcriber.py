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

# Added 2026-09-11: Hindi and Urdu are close to the same SPOKEN language
# (Hindustani) — they diverge mainly in script (Devanagari vs.
# Perso-Arabic), which is exactly why whisper_transcriber.py's own
# docstring documents real Hindi speech getting mis-transcribed as Urdu
# script. This project's downstream keyword lists
# (app/adapters/transcription/urgency_language.py) only cover en/hi/mr —
# there is no Urdu keyword list to match against, so a "correctly"
# Urdu-labeled transcript is exactly as useless to
# ContextualRulesDetector as a Hindi one mis-written in Urdu script;
# there was no working Urdu-language capability this alias gives up.
# Treating a detected "ur" as "hi" here directs the (near-identical)
# audio toward the script the keyword lists can actually match, for
# whichever transcriber — vexyl-stt or Whisper — ends up handling it.
#
# HONEST LIMITATION: this only helps where a language CODE is used as an
# input to a decision (routing, or telling a language-aware transcriber
# which language to use) — it does NOT un-mangle text Whisper has
# already independently decided to write in Urdu script during its own
# auto-detecting full decode (see whisper_transcriber.py's own docstring
# and RoutingTranscriber's module-level WHY note); that needs the
# resolved language forced into the decode itself, not relabeled after
# the fact.
_LANGUAGE_ALIASES = {"ur": "hi"}


def _normalize_detected_language(lang: Optional[str]) -> Optional[str]:
    return _LANGUAGE_ALIASES.get(lang, lang)


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

        detected_lang = _normalize_detected_language(self.lid_transcriber.detect_language(samples, sample_rate))
        return self._route(samples, sample_rate, detected_lang, lambda: self.indic_transcriber)

    def _route(self, samples: np.ndarray, sample_rate: int, detected_lang: Optional[str], indic_provider: Any) -> TranscriptResult:
        """Shared by transcribe() (one-shot, fresh `indic_transcriber` every
        call) and _RoutingSession.transcribe() (persistent, session-scoped
        indic session handed in instead) — the actual gate/fallback
        decision is identical either way, only WHICH indic transcriber
        object gets called differs.

        `indic_provider` is a zero-arg CALLABLE, not an already-resolved
        object — REAL BUG found writing _RoutingSession's own test suite:
        resolving it eagerly (before checking `detected_lang` is actually
        Indic) opened a persistent vexyl-stt connection for every live
        session regardless of its language, including English ones that
        would never use it. Calling `indic_provider()` only inside the
        branch that's actually going to use it fixes that."""
        if detected_lang in self.indic_languages:
            indic = indic_provider()
            result = indic.transcribe_lang(samples, sample_rate, detected_lang)
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

    def create_session(self) -> "_RoutingSession":
        """Returns a per-live-session TranscriberPort-shaped wrapper — see
        _RoutingSession's own docstring for why this exists (added
        2026-09-11, re-enabling live-path Indic routing after the
        2026-09-10 revert documented above). This RoutingTranscriber
        instance itself is built ONCE by app/adapters/registry.py and
        shared/reused across every concurrent live session — it stays
        stateless and thread-safe exactly as before; only the returned
        session object carries per-call-sequence state (the cached
        language decision, the persistent indic connection), and it must
        not be shared across sessions. Caller (app/api/ws_router.py, via
        LiveTranscriptionBuffer) owns the session object's lifetime —
        call `.close()` when the WS session ends."""
        return _RoutingSession(self)

    def _empty(self) -> TranscriptResult:
        return TranscriptResult(text="", language=None, detector_name=self.name, detector_version=self.version)


class _RoutingSession:
    """Per-live-session TranscriberPort-shaped wrapper around a shared
    RoutingTranscriber — see RoutingTranscriber.create_session()'s own
    docstring for why this exists.

    Fixes the two real, specifically-identified costs that made live-path
    routing regress on 2026-09-10 (see this module's docstring and
    config/risk_formula.yaml's own revert comment):

    1. LANGUAGE DECIDED ONCE (once enough audio has accumulated — see
       _MIN_CONFIDENT_LID_DURATION_S below), not re-detected every ~4s
       cycle forever. A live call's language essentially never changes
       mid-conversation, so paying for `lid_transcriber.detect_language()`
       (a full extra Whisper encoder forward pass, competing for the same
       CPU/thread pool as AASIST/Parselmouth/ECAPA on every window) on
       every cycle of a long call was pure waste, not a correctness
       requirement.
    2. ONE PERSISTENT indic connection per session (when the configured
       indic_transcriber supports it — see VexylSttTranscriber.
       open_session()), not a fresh WS connection opened and torn down
       every cycle. Falls back to the old one-shot-per-call behaviour
       (still correct, just not connection-persistent) if the configured
       indic_transcriber doesn't implement `open_session()`, or if
       opening the persistent connection fails — logged once, not
       retried every cycle (a downed vexyl-stt shouldn't cost a fresh
       connect attempt on every single window).

    REAL GAP found and fixed 2026-09-11, via eval/wer_eval.py's own WER
    measurement against labeled real Hindi audio: caching the FIRST
    cycle's language decision forever means a single wrong guess is
    permanent for the whole call — proven, not hypothetical: a genuine
    Hindi clip's first ~4s of audio was misdetected as German, silently
    routing the entire session to the wrong transcriber. Whisper's own
    LID is already known-unreliable for Hindi specifically in this
    codebase (whisper_transcriber.py's own docstring: the "base" model
    mis-transcribed real Hindi into Urdu script — a related confusion).
    Before this fix, `_RoutingSession` had no defense against that at
    all. `_MIN_CONFIDENT_LID_DURATION_S` fixes it CHEAPLY, not by adding
    extra LID passes: a decision is only cached once a cycle hands this
    session at least that much audio (in practice, one extra cycle on a
    call long enough to matter — the buffer naturally grows toward
    LiveTranscriptionBuffer's own _TRANSCRIPTION_WINDOW_MS trailing cap
    within a cycle or two); every cycle before that stays "provisional" —
    still routed with its own best-effort LID result, just not treated as
    final, so a short call (or a call that ends before ever reaching this
    threshold) still works exactly as before, just without ever paying
    the "locked in wrong" risk."""

    # Sentinel distinguishing "checked once, indic_transcriber doesn't
    # support persistent sessions (or opening one failed)" from "haven't
    # tried yet" — avoids retrying open_session() every cycle when it's
    # simply not supported/reachable.
    _NO_PERSISTENT_SESSION = object()

    # Chosen to match LiveTranscriptionBuffer's own _TRANSCRIPTION_WINDOW_MS
    # (8000ms) BY CONVENTION, not an import (routing_transcriber.py has no
    # dependency on the live_transcription module, and shouldn't gain one
    # just for this) — a cycle handing this session that much audio has
    # already reached the live path's own natural "as much context as it
    # ever accumulates per cycle" ceiling, the most audio LID will ever
    # realistically see here.
    _MIN_CONFIDENT_LID_DURATION_S = 8.0

    def __init__(self, parent: RoutingTranscriber) -> None:
        self._parent = parent
        self._decided_lang: Optional[str] = None
        self._lang_decided = False
        self._indic_session: Any = None

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        if samples.size == 0:
            return self._parent._empty()

        if self._lang_decided:
            lang = self._decided_lang
        else:
            lang = _normalize_detected_language(self._parent.lid_transcriber.detect_language(samples, sample_rate))
            duration_s = samples.size / sample_rate
            if duration_s >= self._MIN_CONFIDENT_LID_DURATION_S:
                self._decided_lang = lang
                self._lang_decided = True
                logger.info("live_session_language_decided", detected_language=lang, duration_s=round(duration_s, 2))
            else:
                # Not yet enough audio to trust this as final — use it for
                # THIS cycle only (still routes somewhere reasonable right
                # now), keep re-deciding on the next cycle instead of
                # locking in a low-context guess.
                logger.info(
                    "live_session_language_provisional", detected_language=lang, duration_s=round(duration_s, 2)
                )

        # Lazy — _route() only calls this when `lang` is actually Indic,
        # so an English/non-Indic session never opens the persistent
        # connection at all (see _route's own note on this).
        return self._parent._route(samples, sample_rate, lang, self._indic_for_this_call)

    def _indic_for_this_call(self) -> Any:
        if self._indic_session is None:
            opener = getattr(self._parent.indic_transcriber, "open_session", None)
            if opener is None:
                self._indic_session = self._NO_PERSISTENT_SESSION
            else:
                try:
                    self._indic_session = opener()
                except Exception:  # noqa: BLE001 — best-effort; fall back to one-shot rather than crash live scoring
                    logger.exception("live_session_indic_open_failed")
                    self._indic_session = self._NO_PERSISTENT_SESSION
        if self._indic_session is self._NO_PERSISTENT_SESSION:
            return self._parent.indic_transcriber
        return self._indic_session

    def close(self) -> None:
        """Call when the WS session ends (see LiveTranscriptionBuffer/
        ws_router.py) so the persistent indic connection, if one was
        opened, doesn't linger."""
        if self._indic_session is not None and self._indic_session is not self._NO_PERSISTENT_SESSION:
            closer = getattr(self._indic_session, "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass

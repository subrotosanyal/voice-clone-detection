"""Tests for RoutingTranscriber — the language-gated dispatch between an
Indic-language transcriber and a general fallback (see that module's own
docstring for the real problem this solves and the REAL GAP its
`lid_transcriber` param fixes).

Uses simple stub objects rather than the real VexylSttTranscriber/
WhisperTranscriber classes — RoutingTranscriber only depends on the
methods named in its own Protocol definitions (transcribe_lang,
transcribe, detect_language), so a stub implementing just those is a
faithful, cheap substitute; the real adapters have their own dedicated
test files (test_vexyl_stt_transcriber.py, test_whisper_transcriber.py)."""
from __future__ import annotations

import numpy as np
import pytest

from app.adapters.transcription.routing_transcriber import RoutingTranscriber, _normalize_detected_language
from app.domain.models import TranscriptResult

SR = 16_000


class _StubIndicTranscriber:
    def __init__(self, text: str = "namaste") -> None:
        self.text = text
        self.calls: list[tuple] = []

    def transcribe_lang(self, samples, sample_rate, lang):
        self.calls.append((sample_rate, lang))
        return TranscriptResult(
            text=self.text, language=lang, detector_name="stub_indic", detector_version="1.0.0"
        )


class _StubFallbackTranscriber:
    def __init__(self, text: str = "hello", detected_lang: str = "en") -> None:
        self.text = text
        self.detected_lang = detected_lang
        self.transcribe_calls = 0
        self.detect_calls = 0

    def detect_language(self, samples, sample_rate):
        self.detect_calls += 1
        return self.detected_lang

    def transcribe(self, samples, sample_rate):
        self.transcribe_calls += 1
        return TranscriptResult(
            text=self.text, language=self.detected_lang, detector_name="stub_fallback", detector_version="1.0.0"
        )


class _StubFallbackWithoutDetectLanguage:
    """Stands in for WhisperLiveTranscriber — a real TranscriberPort that
    has no local model and therefore no detect_language() at all."""

    def transcribe(self, samples, sample_rate):
        return TranscriptResult(text="", language=None, detector_name="stub", detector_version="1.0.0")


def _samples() -> np.ndarray:
    return np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)


def _confident_samples() -> np.ndarray:
    """9s of audio — safely past _RoutingSession._MIN_CONFIDENT_LID_DURATION_S
    (8.0s), so a call using this locks in its language decision on the
    first cycle, same as this test file's pre-2026-09-11 assumption."""
    return np.random.default_rng(0).uniform(-0.5, 0.5, size=SR * 9).astype(np.float32)


def test_routes_a_supported_indic_language_to_the_indic_transcriber():
    indic = _StubIndicTranscriber(text="namaste duniya")
    fallback = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback)

    result = router.transcribe(_samples(), SR)

    assert result.text == "namaste duniya"
    assert indic.calls == [(SR, "hi")]
    assert fallback.transcribe_calls == 0, "must not also call the fallback's full transcribe"


def test_routes_english_to_the_fallback_transcriber():
    indic = _StubIndicTranscriber()
    fallback = _StubFallbackTranscriber(text="hello world", detected_lang="en")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback)

    result = router.transcribe(_samples(), SR)

    assert result.text == "hello world"
    assert fallback.transcribe_calls == 1
    assert indic.calls == [], "must not call the Indic transcriber for a non-Indic language"


def test_normalize_detected_language_treats_urdu_as_hindi():
    """Hindi and Urdu are close to the same spoken language, differing
    mainly in script — and this project's downstream keyword lists
    (urgency_language.py) only cover en/hi/mr, so a "correctly"
    Urdu-labeled transcript was never usable here anyway. See
    routing_transcriber.py's own comment on _LANGUAGE_ALIASES."""
    assert _normalize_detected_language("ur") == "hi"


def test_normalize_detected_language_leaves_other_languages_alone():
    assert _normalize_detected_language("hi") == "hi"
    assert _normalize_detected_language("en") == "en"
    assert _normalize_detected_language("de") == "de"
    assert _normalize_detected_language(None) is None


def test_transcribe_routes_a_detected_urdu_call_to_indic_as_hindi():
    """End-to-end: a one-shot RoutingTranscriber.transcribe() call must
    route "ur" the same way it routes "hi" — to the indic transcriber,
    with "hi" (not "ur") as the language actually asked for."""
    indic = _StubIndicTranscriber(text="hindi text")
    fallback = _StubFallbackTranscriber(detected_lang="ur")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback)

    result = router.transcribe(_samples(), SR)

    assert result.text == "hindi text"
    assert indic.calls == [(SR, "hi")], "must ask the indic transcriber for Hindi, not Urdu"


def test_session_routes_a_detected_urdu_call_to_indic_as_hindi():
    """Same alias, through the per-live-session path — a session that
    decides "ur" (provisional or locked-in) must transcribe as Hindi."""
    indic = _StubIndicTranscriberWithSessions(text="hindi text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="ur")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    result = session.transcribe(_confident_samples(), SR)

    assert result.text == "hindi text"
    assert indic.last_session.calls == [(SR, "hi")], "must ask the indic transcriber for Hindi, not Urdu"


def test_routes_an_unsupported_indic_language_to_the_fallback():
    """A language detected but not in indic_languages (e.g. explicitly
    narrowed for a deployment) must still fall back, not error."""
    indic = _StubIndicTranscriber()
    fallback = _StubFallbackTranscriber(detected_lang="ta")
    router = RoutingTranscriber(
        indic_transcriber=indic, fallback_transcriber=fallback, indic_languages=["hi", "mr"]
    )

    router.transcribe(_samples(), SR)

    assert indic.calls == []
    assert fallback.transcribe_calls == 1


def test_falls_back_when_indic_transcriber_returns_empty_text():
    """The Indic path being unreachable (server down) or the detected
    language being wrong must not lose the call — the fallback's own
    auto-detect gets a real second chance."""
    indic = _StubIndicTranscriber(text="")  # simulates vexyl-stt unreachable/timed out
    fallback = _StubFallbackTranscriber(text="real content", detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback)

    result = router.transcribe(_samples(), SR)

    assert result.text == "real content"
    assert fallback.transcribe_calls == 1


def test_empty_samples_abstain_without_detecting_or_transcribing():
    indic = _StubIndicTranscriber()
    fallback = _StubFallbackTranscriber()
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback)

    result = router.transcribe(np.array([], dtype=np.float32), SR)

    assert result.text == ""
    assert fallback.detect_calls == 0
    assert fallback.transcribe_calls == 0


def test_uses_a_separate_lid_transcriber_when_given():
    """Real gap this fixes: on the live path, fallback_transcriber
    (WhisperLiveTranscriber) has no local model to run detect_language()
    against — a separate lid_transcriber must be used instead, and the
    fallback itself must never be asked for language ID."""
    indic = _StubIndicTranscriber(text="marathi text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="mr")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)

    result = router.transcribe(_samples(), SR)

    assert result.text == "marathi text"
    assert lid.detect_calls == 1


def test_raises_an_actionable_error_if_fallback_cannot_detect_language_and_no_lid_given():
    indic = _StubIndicTranscriber()
    fallback = _StubFallbackWithoutDetectLanguage()

    with pytest.raises(TypeError, match="detect_language"):
        RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback)


class _StubIndicSession:
    """Stands in for VexylSttSession — a stub's persistent-connection
    handle, tracking how many cycles ran on it and whether it was closed."""

    def __init__(self, text: str = "namaste") -> None:
        self.text = text
        self.calls: list[tuple] = []
        self.closed = False

    def transcribe_lang(self, samples, sample_rate, lang):
        self.calls.append((sample_rate, lang))
        return TranscriptResult(text=self.text, language=lang, detector_name="stub_indic_session", detector_version="1.0.0")

    def close(self):
        self.closed = True


class _StubIndicTranscriberWithSessions(_StubIndicTranscriber):
    """Stands in for VexylSttTranscriber once open_session() exists —
    records how many times a persistent connection was actually opened,
    so a test can assert it happens at most once per live session."""

    def __init__(self, text: str = "namaste") -> None:
        super().__init__(text=text)
        self.sessions_opened = 0
        self.last_session: _StubIndicSession | None = None

    def open_session(self):
        self.sessions_opened += 1
        self.last_session = _StubIndicSession(text=self.text)
        return self.last_session


def test_create_session_decides_language_once_not_every_call():
    """The real regression this fixes (see this module's own docstring,
    2026-09-10 revert): re-running detect_language() every ~4s cycle was
    one of the two costs that made live-path routing fall behind. A
    session must pay for LID exactly once (once confident — see
    _MIN_CONFIDENT_LID_DURATION_S — which 9s-long cycles satisfy
    immediately), then reuse that decision."""
    indic = _StubIndicTranscriberWithSessions(text="marathi cycle text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="mr")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    for _ in range(5):
        result = session.transcribe(_confident_samples(), SR)
        assert result.text == "marathi cycle text"

    assert lid.detect_calls == 1, "language must be decided ONCE, not once per cycle"


def test_short_cycles_stay_provisional_and_keep_re_deciding_language():
    """REAL GAP fixed 2026-09-11: caching a language decision from a
    short, low-context cycle risks locking in a wrong guess (found via
    eval/wer_eval.py — a real Hindi clip's first ~4s was misdetected as
    German) for the rest of the call. Cycles shorter than
    _MIN_CONFIDENT_LID_DURATION_S must re-run LID every time, not just
    once — still routing correctly each time, just never locking in."""
    indic = _StubIndicTranscriberWithSessions(text="hindi text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    for _ in range(4):
        result = session.transcribe(_samples(), SR)  # 1s each — well under the 8.0s confidence threshold
        assert result.text == "hindi text"

    assert lid.detect_calls == 4, "short cycles must keep re-deciding, never lock in a low-context guess"


def test_a_confident_cycle_locks_in_even_if_earlier_cycles_were_provisional():
    """The common real-world shape: a call's first cycle or two are short
    (provisional), then the buffer grows toward its natural trailing-
    window cap and a later cycle is long enough to lock in — from that
    point on, LID must stop running."""
    indic = _StubIndicTranscriberWithSessions(text="hindi text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    session.transcribe(_samples(), SR)  # provisional (1s)
    session.transcribe(_samples(), SR)  # provisional (1s)
    assert lid.detect_calls == 2

    session.transcribe(_confident_samples(), SR)  # 9s — locks in
    assert lid.detect_calls == 3

    session.transcribe(_samples(), SR)  # short again, but already locked in
    session.transcribe(_samples(), SR)
    assert lid.detect_calls == 3, "once locked in, later cycles must not re-run LID regardless of their own length"


def test_a_wrong_provisional_guess_does_not_stick_once_corrected():
    """Directly reproduces the real scenario found via eval/wer_eval.py:
    a short, low-context first cycle misdetects the language; a later,
    more-confident cycle must be free to correct it rather than being
    stuck with the first (wrong, uncached) guess."""
    indic = _StubIndicTranscriberWithSessions(text="corrected hindi text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="de")  # wrong first guess, same as the real incident
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    first = session.transcribe(_samples(), SR)  # provisional, routed (wrongly) to fallback
    assert first.text == ""  # _StubFallbackWithoutDetectLanguage's own empty default

    lid.detected_lang = "hi"  # more context next cycle corrects the earlier misdetection
    second = session.transcribe(_confident_samples(), SR)  # confident — locks in "hi" this time

    assert second.text == "corrected hindi text"
    assert indic.last_session.calls == [(SR, "hi")], "the corrected language must actually reach the indic transcriber"


def test_create_session_reuses_one_persistent_indic_connection():
    """The other real regression cost: a fresh connection per cycle. A
    session must open the indic transcriber's persistent connection at
    most once, then reuse it for every subsequent cycle."""
    indic = _StubIndicTranscriberWithSessions(text="hindi cycle text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    for _ in range(3):
        session.transcribe(_samples(), SR)

    assert indic.sessions_opened == 1, "must open the persistent connection at most once per session"
    assert len(indic.last_session.calls) == 3, "every cycle after that must reuse the same connection"


def test_session_close_closes_the_persistent_indic_connection():
    indic = _StubIndicTranscriberWithSessions(text="text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()
    session.transcribe(_samples(), SR)

    session.close()

    assert indic.last_session.closed is True


def test_session_close_before_any_call_is_a_safe_noop():
    indic = _StubIndicTranscriberWithSessions()
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    session.close()  # must not raise even though nothing was ever opened


def test_session_falls_back_to_one_shot_when_indic_transcriber_has_no_open_session():
    """A plain VexylSttTranscriber-shaped stub with no open_session() must
    still work correctly under create_session() — just without connection
    persistence (the old one-shot-per-call behaviour), not an error."""
    indic = _StubIndicTranscriber(text="one shot text")  # no open_session()
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    result = session.transcribe(_samples(), SR)

    assert result.text == "one shot text"
    assert indic.calls == [(SR, "hi")]


def test_session_falls_back_to_whisperlive_when_persistent_indic_result_is_empty():
    """Same safety net as the one-shot RoutingTranscriber.transcribe():
    a language decided as Indic but an empty result this cycle (server
    down, timed out, or genuine silence) must still fall through to the
    fallback transcriber, not go silent."""
    indic = _StubIndicTranscriberWithSessions(text="")
    fallback = _StubFallbackWithoutDetectLanguage()
    fallback.transcribe = lambda samples, sample_rate: TranscriptResult(  # type: ignore[method-assign]
        text="fallback caught it", language=None, detector_name="stub", detector_version="1.0.0"
    )
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)
    session = router.create_session()

    result = session.transcribe(_samples(), SR)

    assert result.text == "fallback caught it"


def test_session_stays_stateless_on_the_shared_parent_router():
    """RoutingTranscriber itself (built once, shared across every
    concurrent live session by app/adapters/registry.py) must stay
    reusable — creating and using a session must not mutate the parent's
    own state, so a second, independent session for a different call
    starts with its own fresh language decision."""
    indic = _StubIndicTranscriberWithSessions(text="text")
    fallback = _StubFallbackWithoutDetectLanguage()
    lid = _StubFallbackTranscriber(detected_lang="hi")
    router = RoutingTranscriber(indic_transcriber=indic, fallback_transcriber=fallback, lid_transcriber=lid)

    session_a = router.create_session()
    session_a.transcribe(_samples(), SR)
    lid.detected_lang = "en"  # a second, different call
    session_b = router.create_session()
    session_b.transcribe(_samples(), SR)

    assert lid.detect_calls == 2, "each session must independently pay for its own LID pass"
    assert indic.sessions_opened == 1, "session_b (English) must never have opened an indic connection at all"


def test_default_indic_languages_cover_vexyl_stts_own_supported_set():
    """Regression guard against the two lists silently drifting apart —
    see routing_transcriber.py's own _DEFAULT_INDIC_LANGUAGES comment."""
    from app.adapters.transcription.vexyl_stt_transcriber import SUPPORTED_LANGUAGES
    from app.adapters.transcription.routing_transcriber import _DEFAULT_INDIC_LANGUAGES

    assert _DEFAULT_INDIC_LANGUAGES == SUPPORTED_LANGUAGES

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

from app.adapters.transcription.routing_transcriber import RoutingTranscriber
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


def test_default_indic_languages_cover_vexyl_stts_own_supported_set():
    """Regression guard against the two lists silently drifting apart —
    see routing_transcriber.py's own _DEFAULT_INDIC_LANGUAGES comment."""
    from app.adapters.transcription.vexyl_stt_transcriber import SUPPORTED_LANGUAGES
    from app.adapters.transcription.routing_transcriber import _DEFAULT_INDIC_LANGUAGES

    assert _DEFAULT_INDIC_LANGUAGES == SUPPORTED_LANGUAGES

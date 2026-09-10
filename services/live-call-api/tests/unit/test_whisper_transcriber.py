"""Tests for WhisperTranscriber.

Verified by hand while building this: real spoken English ("Please
transfer the money immediately, it's urgent.") synthesized with macOS's
`say` and fed through this exact class produced the correct text, correct
detected language, and correct urgency-keyword matches (see the
class's own module docstring for the "base" vs "small" model-size finding
that came out of that same verification).

The "real speech" tests here use macOS's `say` command to generate a
genuine (if synthetic-voice) spoken fixture at test time — no network, no
new dependency, and it's real intelligible speech Whisper actually has to
transcribe, not a synthetic tone. Skipped gracefully wherever `say` isn't
available (any non-macOS CI/Docker environment) — the near-silence and
non-speech-tone tests below stay platform-independent and always run.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from app.adapters.transcription.whisper_transcriber import WhisperTranscriber

SR = 16_000


@pytest.fixture(scope="session")
def transcriber() -> WhisperTranscriber:
    try:
        return WhisperTranscriber(model_size="small")
    except Exception as exc:  # noqa: BLE001 — model download can fail offline
        pytest.skip(f"Whisper model unavailable (no network?): {exc}")


@pytest.fixture(scope="session")
def spoken_urgency_wav(tmp_path_factory) -> np.ndarray:
    if shutil.which("say") is None:
        pytest.skip("macOS 'say' not available — real-speech test skipped on this platform")

    tmp_dir = tmp_path_factory.mktemp("say_fixtures")
    aiff_path = tmp_dir / "urgency.aiff"
    wav_path = tmp_dir / "urgency.wav"
    subprocess.run(
        ["say", "-o", str(aiff_path), "Please transfer the money immediately, it's urgent."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), "-ar", str(SR), "-ac", "1", str(wav_path)],
        check=True,
        capture_output=True,
    )
    samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    assert sr == SR
    return samples


def test_abstains_on_near_silence(transcriber):
    silence = np.zeros(SR * 2, dtype=np.float32)
    result = transcriber.transcribe(silence, SR)
    assert result.text == ""
    assert result.language is None


def test_non_speech_tone_produces_empty_transcript(transcriber):
    samples, sr = sf.read("samples/genuine_tone.wav", dtype="float32", always_2d=False)
    result = transcriber.transcribe(samples, sr)
    assert result.text == ""  # correct behaviour: nothing to abstain from being wrong about


def test_transcribes_real_speech_and_detects_language(transcriber, spoken_urgency_wav):
    result = transcriber.transcribe(spoken_urgency_wav, SR)
    assert result.language == "en"
    lowered = result.text.lower()
    assert "transfer" in lowered
    assert "money" in lowered
    assert "urgent" in lowered or "immediately" in lowered


def test_result_carries_detector_identity(transcriber):
    silence = np.zeros(SR * 2, dtype=np.float32)
    result = transcriber.transcribe(silence, SR)
    assert result.detector_name == "whisper_transcriber"
    assert result.detector_version == "whisper-small"


def test_abstains_instead_of_crashing_on_a_very_short_window(transcriber):
    """Regression test for a real bug found 2026-09-09 (via dogfooding real
    external audio — IndieFake Dataset's public demo clips — through the
    deployed service): a very short window crashed the WHOLE request with an
    uncaught `RuntimeError: cannot reshape tensor of 0 elements into shape
    [1, 0, 12, -1] because the unspecified dimension size -1 can be any
    value and is ambiguous` inside whisper/model.py's
    MultiHeadAttention.qkv_attention — an actual HTTP 500, not a graceful
    abstain. A window this short is not a contrived edge case: windowing.py's
    own docstring documents that the final window of any file "is still
    yielded" even when shorter than the configured window length, which
    happens for ANY file whose duration isn't an exact multiple of the hop
    length. Loud (not silent) so this exercises the NEW length guard
    specifically, not the existing near-silence one above. Same pattern as
    test_perth_watermark.py's counterpart regression test for the same class
    of bug in PerthWatermarkDetector."""
    # 600 samples at 16kHz = 37.5ms — within the 300-900 sample range
    # observed to trigger the crash, and below this module's own
    # _MIN_DURATION_S (100ms) guard.
    short_loud = np.random.default_rng(0).uniform(-0.5, 0.5, size=600).astype(np.float32)

    result = transcriber.transcribe(short_loud, SR)

    assert result.text == ""
    assert result.language is None


def test_falls_back_to_empty_transcript_instead_of_crashing_on_model_failure(transcriber, monkeypatch):
    """Regression test for a real bug found 2026-09-09 (second instance,
    different trigger than the short-window one above): a real user-
    uploaded ~4-minute MP3 conversation recording crashed the WHOLE
    request with the SAME uncaught RuntimeError as the short-window bug
    — but this time from INSIDE Whisper's own internal transcribe() seek
    loop on a long, comfortably-above-_MIN_DURATION_S buffer, which that
    guard can't catch. Reproducing the exact upstream trigger isn't
    reliable (it didn't even reproduce locally on macOS — only inside
    the actual Linux container, likely a libsndfile/MP3-decode
    difference) — this instead verifies OUR OWN fallback behaviour
    directly, by making the real model object raise, same as monkeypatch
    patterns already used elsewhere in this suite (e.g.
    test_perth_watermark.py's test_raises_actionable_error_if_watermarker_is_none)."""
    monkeypatch.setattr(
        transcriber._model,
        "transcribe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cannot reshape tensor of 0 elements...")),
    )
    loud_speech_length = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR * 5).astype(np.float32)

    result = transcriber.transcribe(loud_speech_length, SR)

    assert result.text == ""
    assert result.language is None


def test_discards_a_degenerate_repetition_loop_transcript(transcriber, monkeypatch):
    """Regression test for a real bug found 2026-09-10, via real Movie-
    MUSNOMIX movie-dialogue clips fed through the deployed dashboard: on
    hard audio, greedy decoding (temperature=0.0, see this module's
    REPRODUCIBILITY FIX note) can lock into a repetition loop
    ("तो तो तो तो..." repeated 30+ times, observed verbatim) that Whisper's
    own model is confident about (a real clip measured avg_logprob -0.16)
    even though it's nonsense — and that nonsense then fed Intent's
    zero-shot classifier, which scored it 96% "creating urgency", a
    fabricated signal from garbage text. Whisper's own compression_ratio
    metric (already computed per segment regardless of temperature mode)
    catches this cleanly: the real degenerate clip measured 21.71, a
    real correctly-transcribed clip from the same corpus measured 1.40 —
    this test pins the discard behavior at that threshold with a fake
    model, no real audio needed."""
    monkeypatch.setattr(
        transcriber._model,
        "transcribe",
        lambda *a, **k: {
            "text": " तो तो तो तो तो तो तो तो तो",
            "language": "hi",
            "segments": [
                {
                    "text": " तो तो तो तो तो तो तो तो तो",
                    "compression_ratio": 21.71,
                    "avg_logprob": -0.16,
                    "no_speech_prob": 0.02,
                }
            ],
        },
    )
    loud_enough = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(loud_enough, SR)

    assert result.text == ""


def test_keeps_good_segments_and_drops_only_the_degenerate_one(transcriber, monkeypatch):
    """Regression test for a real bug found 2026-09-10, the same day as
    the fix above: the FIRST version discarded the WHOLE transcript if
    ANY segment exceeded the compression_ratio threshold — wrong for a
    real 52.9s movie-dialogue clip that was genuinely mostly-coherent
    Hindi speech (11 segments) with ONE brief glitch in a single segment
    (a repeated-diacritic artifact, "जाँँँँँँँ", compression_ratio 2.83
    vs. the 2.4 threshold): that all-or-nothing check threw away the
    other 10 good segments too, silently blinding Intent/semantic-risk on
    a call that actually had real, usable transcript content. This test
    reproduces the shape of that failure with 3 segments (2 good, 1
    degenerate) and asserts only the degenerate one is dropped."""
    monkeypatch.setattr(
        transcriber._model,
        "transcribe",
        lambda *a, **k: {
            "text": "मेरा गोला हार गया जाँँँँँँँ तुमारे मावी मागने से क्या होता है",
            "language": "hi",
            "segments": [
                {"text": "मेरा गोला हार गया ", "compression_ratio": 1.9, "avg_logprob": -0.3, "no_speech_prob": 0.01},
                {"text": "जाँँँँँँँ ", "compression_ratio": 2.83, "avg_logprob": -0.2, "no_speech_prob": 0.02},
                {
                    "text": "तुमारे मावी मागने से क्या होता है",
                    "compression_ratio": 1.7,
                    "avg_logprob": -0.4,
                    "no_speech_prob": 0.01,
                },
            ],
        },
    )
    loud_enough = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(loud_enough, SR)

    assert "मेरा गोला हार गया" in result.text
    assert "तुमारे मावी मागने से क्या होता है" in result.text
    assert "जाँँँँँँँ" not in result.text


def test_keeps_a_normal_transcript_with_a_low_compression_ratio(transcriber, monkeypatch):
    """The other half of the fix above: a real, non-degenerate transcript
    (compression_ratio well under Whisper's own 2.4 default threshold —
    1.40 measured on a real correctly-transcribed clip) must still pass
    through untouched, not get discarded by an over-eager check."""
    monkeypatch.setattr(
        transcriber._model,
        "transcribe",
        lambda *a, **k: {
            "text": "निक्केई पिनेंसिल ताइम्स की मुल कम्पनी भी है",
            "language": "hi",
            "segments": [
                {
                    "text": "निक्केई पिनेंसिल ताइम्स की मुल कम्पनी भी है",
                    "compression_ratio": 1.40,
                    "avg_logprob": -0.51,
                    "no_speech_prob": 0.01,
                }
            ],
        },
    )
    loud_enough = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(loud_enough, SR)

    assert result.text == "निक्केई पिनेंसिल ताइम्स की मुल कम्पनी भी है"


def test_repeated_calls_on_non_speech_audio_are_deterministic(transcriber):
    """Regression test for a real intermittent bug found 2026-09-08: without
    a fixed temperature, Whisper's default fallback-to-sampling behaviour on
    non-speech audio (this exact tone fixture) produced a different result
    call to call within the same process — sometimes an empty transcript,
    sometimes a hallucinated one ("MMMMMMMMMMMMMM", observed by hand) — which
    then fed the intent classifier and changed the final fused score
    non-deterministically (see this test's counterpart in
    tests/integration/test_api_score_file.py, and WhisperTranscriber's own
    REPRODUCIBILITY FIX docstring note). temperature=0.0 forces greedy
    decoding, which must give the identical result every time."""
    samples, sr = sf.read("samples/genuine_tone.wav", dtype="float32", always_2d=False)
    results = [transcriber.transcribe(samples, sr) for _ in range(5)]
    texts = {r.text for r in results}
    assert len(texts) == 1, f"expected identical transcript every call, got {texts}"


class _SlowFakeModel:
    """Stands in for the real whisper model — no network/model-download
    dependency, fast, deterministic. Sleeps briefly inside transcribe()
    (the same place the real REAL BUG's concurrent decode() calls
    collided) so a concurrency test can reliably observe two calls
    overlapping if they aren't actually serialized."""

    def __init__(self) -> None:
        self.max_concurrent = 0
        self._current = 0
        self._current_lock = threading.Lock()

    def transcribe(self, samples, fp16, temperature):
        with self._current_lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
        time.sleep(0.05)
        with self._current_lock:
            self._current -= 1
        return {"text": "fake transcript", "language": "en"}


def _transcriber_with_fake_model(model) -> WhisperTranscriber:
    """Bypasses __init__ (no real model download) — sets up exactly what
    transcribe() actually reads."""
    t = WhisperTranscriber.__new__(WhisperTranscriber)
    t.model_size = "small"
    t.floor_rms = 1e-4
    t._model = model
    t._lock = threading.Lock()
    return t


def test_concurrent_transcribe_calls_are_serialized_not_overlapping():
    """Regression test for a real bug found 2026-09-10 (building "diarize
    on hangup"): two concurrent decode() calls into the SAME shared
    Whisper model instance corrupted its kv_cache
    (KeyError inside whisper/model.py's cross-attention) — the same class
    of non-thread-safe-shared-model bug this project already guards
    against for Parselmouth (app/adapters/praat_lock.py) and ECAPA-TDNN
    (EcapaEmbeddingExtractor's own lock). Proves the fix: two threads
    calling transcribe() at the same time never actually overlap inside
    the model call."""
    model = _SlowFakeModel()
    transcriber = _transcriber_with_fake_model(model)
    loud_samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    threads = [threading.Thread(target=transcriber.transcribe, args=(loud_samples, SR)) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert model.max_concurrent == 1, "two transcribe() calls overlapped inside the model — not actually serialized"

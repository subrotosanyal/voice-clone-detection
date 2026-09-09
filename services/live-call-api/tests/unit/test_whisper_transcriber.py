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

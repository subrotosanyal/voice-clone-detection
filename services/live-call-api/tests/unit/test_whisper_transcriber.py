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

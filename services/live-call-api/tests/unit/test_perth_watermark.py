"""Tests for PerthWatermarkDetector — real model, real audio, no mocking.

Uses macOS's `say` to generate a genuine (if synthetic-voice) spoken
fixture at test time — skipped gracefully wherever `say` isn't available,
same pattern as test_whisper_transcriber.py. The "watermarked" case is
produced by running the SAME PerthImplicitWatermarker's own
apply_watermark() on that clip — a real, not simulated, watermark.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from app.adapters.detectors.perth_watermark import PerthWatermarkDetector
from app.domain.models import AudioWindow

SR = 16_000


def _window(samples: np.ndarray) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=SR, samples=samples, window_start_ms=0)


@pytest.fixture(scope="module")
def spoken_samples(tmp_path_factory) -> np.ndarray:
    if shutil.which("say") is None:
        pytest.skip("macOS 'say' not available — real-speech test skipped on this platform")

    tmp_dir = tmp_path_factory.mktemp("say_fixtures_perth")
    aiff_path = tmp_dir / "speech.aiff"
    wav_path = tmp_dir / "speech.wav"
    subprocess.run(
        ["say", "-o", str(aiff_path), "This is a test of the Perth audio watermarking system."],
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
    return samples[: SR * 2]  # a realistic 2s production window


@pytest.fixture(scope="module")
def detector() -> PerthWatermarkDetector:
    try:
        return PerthWatermarkDetector()
    except Exception as exc:  # noqa: BLE001 — model load can fail offline
        pytest.skip(f"Perth model unavailable: {exc}")


def test_raises_actionable_error_if_watermarker_is_none(monkeypatch):
    """Regression test for a real bug found 2026-09-08: resemble-perth
    silently sets PerthImplicitWatermarker to None if its own (undeclared)
    librosa dependency fails to import, rather than raising. Verify our
    own defensive check catches that and fails loudly instead."""
    import perth

    monkeypatch.setattr(perth, "PerthImplicitWatermarker", None)

    with pytest.raises(ImportError, match="librosa"):
        PerthWatermarkDetector()


def test_abstains_on_near_silence(detector):
    silence = np.zeros(SR * 2, dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.score is None
    assert result.abstain_reason is not None


def test_abstains_instead_of_crashing_on_a_very_short_window(detector):
    """Regression test for a real bug found 2026-09-09 (via dogfooding real
    external audio — IndieFake Dataset's public demo clips — through the
    deployed service): Perth's own STFT (n_fft=2048, reflect-padded by 1024
    samples internally at its 32kHz working rate) raises an uncaught
    RuntimeError on a window shorter than that pad amount — an actual HTTP
    500, not a graceful abstain. A window this short is not a contrived
    edge case: windowing.py's own docstring documents that the final
    window of any file "is still yielded" even when shorter than the
    configured window length, which happens for ANY file whose duration
    isn't an exact multiple of the hop length. Loud (not silent) so this
    exercises the NEW length guard specifically, not the existing
    near-silence one above."""
    # 20ms at 16kHz = 320 samples — comfortably below both the 1024-sample
    # crash threshold at Perth's internal 32kHz rate and this detector's
    # own 50ms (_MIN_DURATION_S) guard.
    short_loud = np.random.default_rng(0).uniform(-0.5, 0.5, size=320).astype(np.float32)

    result = detector.score(_window(short_loud), context={})

    assert result.score is None
    assert result.abstain_reason is not None
    assert "short" in result.abstain_reason


def test_scores_low_on_genuine_unwatermarked_speech(detector, spoken_samples):
    result = detector.score(_window(spoken_samples), context={})

    assert result.score is not None
    assert result.score < 0.5
    assert result.detail["watermark_probability"] == result.score
    assert "explanation" in result.detail


def test_scores_high_on_actually_watermarked_audio(detector, spoken_samples):
    watermarked = detector._watermarker.apply_watermark(spoken_samples, sample_rate=SR)

    result = detector.score(_window(watermarked.astype(np.float32)), context={})

    assert result.score is not None
    assert result.score > 0.9


def test_returns_full_detail_breakdown(detector, spoken_samples):
    result = detector.score(_window(spoken_samples), context={})

    assert 0.0 <= result.score <= 1.0
    assert "rms" in result.detail
    assert result.detector_name == "perth_watermark"

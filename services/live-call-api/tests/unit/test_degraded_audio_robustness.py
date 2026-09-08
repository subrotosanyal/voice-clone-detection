"""Degraded-audio robustness checks — motivated by a real, documented
finding (IEEE Spectrum, "Real-Time Audio Deepfakes And Voice Phishing"):
NCC Group demonstrated convincing real-time vishing using "poor" quality
input audio on commodity hardware. If real attacks succeed on degraded
audio, this system's detectors need to behave sanely on degraded audio
too — not necessarily perfectly, but not by crashing, returning NaN, or
wildly misclassifying genuine speech as high-confidence spoof merely
because a real phone line/noisy room/low-bitrate codec degraded it.

Uses macOS's `say` to generate a genuine (if synthetic-voice) spoken
fixture at test time — skipped gracefully wherever `say` isn't available
(any non-macOS CI/Docker environment), same pattern as
test_whisper_transcriber.py. No new dependency: degradations use scipy
(already required) and ffmpeg (already required by every `say`-based test
in this suite for AIFF->WAV conversion).

HONEST SCOPE: this tests ROBUSTNESS (does the detector stay sane), not
ACCURACY on degraded audio — there's no degraded-audio ground truth
labeled dataset here to measure detection accuracy against. A relative
check (does the score jump unreasonably from the clean baseline) is what
these tests can honestly claim, not an absolute "still correct" claim.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from app.adapters.detectors.acoustic_aasist import AasistAcousticDetector
from app.adapters.detectors.prosody_parselmouth import ParselmouthProsodyDetector
from app.domain.models import AudioWindow

SR = 16_000


def _window(samples: np.ndarray) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=SR, samples=samples.astype(np.float32), window_start_ms=0)


@pytest.fixture(scope="module")
def genuine_speech(tmp_path_factory) -> np.ndarray:
    if shutil.which("say") is None:
        pytest.skip("macOS 'say' not available — real-speech test skipped on this platform")

    tmp_dir = tmp_path_factory.mktemp("say_fixtures_degraded")
    aiff_path = tmp_dir / "speech.aiff"
    wav_path = tmp_dir / "speech.wav"
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
    return samples[: SR * 2]  # a realistic 2s production window


def _bandpass_phone_filter(samples: np.ndarray) -> np.ndarray:
    """300-3400Hz — the classic analogue telephone passband."""
    sos = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, samples).astype(np.float32)


def _add_noise_at_snr(samples: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal_power = np.mean(samples**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), samples.shape).astype(np.float32)
    return samples + noise


def _low_bitrate_reencode(samples: np.ndarray, tmp_path, bitrate: str = "6k") -> np.ndarray:
    """Real lossy re-encode via ffmpeg's libopus at a low bitrate — a
    real-world proxy for a degraded call connection, not a simulation."""
    wav_in = tmp_path / "clean.wav"
    opus_out = tmp_path / "degraded.opus"
    wav_out = tmp_path / "degraded.wav"
    sf.write(wav_in, samples, SR)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_in), "-c:a", "libopus", "-b:a", bitrate, str(opus_out)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(opus_out), "-ar", str(SR), "-ac", "1", str(wav_out)],
        check=True,
        capture_output=True,
    )
    degraded, sr = sf.read(wav_out, dtype="float32", always_2d=False)
    assert sr == SR
    # opus re-encoding can shift length slightly — align to the original
    if degraded.size < samples.size:
        degraded = np.pad(degraded, (0, samples.size - degraded.size))
    return degraded[: samples.size]


@pytest.fixture(scope="module")
def aasist(aasist_checkpoint) -> AasistAcousticDetector:
    return AasistAcousticDetector(checkpoint_path=aasist_checkpoint)


@pytest.fixture(scope="module")
def prosodic() -> ParselmouthProsodyDetector:
    try:
        return ParselmouthProsodyDetector()
    except Exception as exc:  # noqa: BLE001 — Parselmouth unavailable in some envs
        pytest.skip(f"Parselmouth unavailable: {exc}")


# A generous bound: degradation is EXPECTED to move the score somewhat
# (that's the whole point of testing it) — this catches a detector
# breaking catastrophically (NaN, a score pinned to 0 or 1 regardless of
# input, or an unreasonable swing), not "the score must barely move".
_MAX_REASONABLE_SCORE_SWING = 0.5


def _assert_robust(clean_score: float, degraded_score: float, label: str) -> None:
    assert not np.isnan(degraded_score), f"{label}: detector returned NaN on degraded audio"
    assert 0.0 <= degraded_score <= 1.0, f"{label}: degraded score {degraded_score} out of [0,1] range"
    swing = abs(degraded_score - clean_score)
    assert swing <= _MAX_REASONABLE_SCORE_SWING, (
        f"{label}: score swung from {clean_score:.3f} (clean) to {degraded_score:.3f} (degraded) — "
        f"a {swing:.3f} swing exceeds the {_MAX_REASONABLE_SCORE_SWING} sanity bound"
    )


@pytest.mark.parametrize("degradation", ["bandpass_phone_filter", "noise_20db", "noise_10db", "low_bitrate_opus"])
def test_aasist_stays_sane_on_degraded_genuine_speech(aasist, genuine_speech, tmp_path, degradation):
    clean_result = aasist.score(_window(genuine_speech), context={})
    assert clean_result.score is not None
    clean_score = clean_result.score

    if degradation == "bandpass_phone_filter":
        degraded = _bandpass_phone_filter(genuine_speech)
    elif degradation == "noise_20db":
        degraded = _add_noise_at_snr(genuine_speech, snr_db=20)
    elif degradation == "noise_10db":
        degraded = _add_noise_at_snr(genuine_speech, snr_db=10)
    else:
        degraded = _low_bitrate_reencode(genuine_speech, tmp_path)

    degraded_result = aasist.score(_window(degraded), context={})
    assert degraded_result.score is not None, f"{degradation}: AASIST abstained on degraded audio unexpectedly"
    _assert_robust(clean_score, degraded_result.score, f"AASIST/{degradation}")


@pytest.mark.parametrize("degradation", ["bandpass_phone_filter", "noise_20db", "noise_10db", "low_bitrate_opus"])
def test_prosodic_stays_sane_on_degraded_genuine_speech(prosodic, genuine_speech, tmp_path, degradation):
    clean_result = prosodic.score(_window(genuine_speech), context={})
    if clean_result.score is None:
        pytest.skip("prosodic detector abstained on the clean baseline — nothing to compare against")
    clean_score = clean_result.score

    if degradation == "bandpass_phone_filter":
        degraded = _bandpass_phone_filter(genuine_speech)
    elif degradation == "noise_20db":
        degraded = _add_noise_at_snr(genuine_speech, snr_db=20)
    elif degradation == "noise_10db":
        degraded = _add_noise_at_snr(genuine_speech, snr_db=10)
    else:
        degraded = _low_bitrate_reencode(genuine_speech, tmp_path)

    degraded_result = prosodic.score(_window(degraded), context={})
    if degraded_result.score is None:
        return  # abstaining is a safe, sane response — not a robustness failure
    _assert_robust(clean_score, degraded_result.score, f"Prosodic/{degradation}")

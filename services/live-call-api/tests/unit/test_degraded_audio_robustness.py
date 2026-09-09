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
import urllib.error
import urllib.request

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


def _window_at(samples: np.ndarray, sample_rate: int) -> AudioWindow:
    """Same as _window() above but for a sample rate OTHER than this
    file's fixed SR=16kHz — needed for known_deepfake_speech below, which
    is fetched at its own native rate (24kHz), not resampled beforehand
    (AasistAcousticDetector resamples internally; mislabeling the rate
    here would skip that step and silently corrupt the test)."""
    return AudioWindow(session_id="s1", seq=0, sample_rate=sample_rate, samples=samples, window_start_ms=0)


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


_INDIEFAKE_SAMPLE_URL = "https://indie-fake-dataset.netlify.app/audios/sadhguru_deepfake.wav"
_HINDI_FINETUNED_OUT_LAYER = "app/adapters/detectors/vendor/checkpoints/AASIST_hindi_finetuned_out_layer.pth"


@pytest.fixture(scope="module")
def aasist_as_deployed(aasist_checkpoint) -> AasistAcousticDetector:
    """The `aasist` fixture above does NOT include the Hindi-recalibrated
    output layer config/risk_formula.yaml actually wires into production
    (finetuned_out_layer_path) — verified by hand this matters a lot for
    the test below: the base checkpoint scored the same real deepfake
    window at spoof_probability=0.999 (correct, and unmoved by
    degradation), while the actually-deployed recalibrated checkpoint
    scored the SAME window at 0.22 (far more marginal) and DID move under
    degradation. Real production behaviour needs this exact config."""
    return AasistAcousticDetector(
        checkpoint_path=aasist_checkpoint, finetuned_out_layer_path=_HINDI_FINETUNED_OUT_LAYER
    )


def _reverb_and_lossy_reencode(samples: np.ndarray, sample_rate: int, tmp_path) -> np.ndarray:
    """Real room reverb (ffmpeg's `aecho`) + a real lossy AAC re-encode —
    a realistic proxy for actual real-world recording/distribution
    conditions (e.g. a YouTube-sourced clip), not a synthetic simulation.
    Distinct from _low_bitrate_reencode above (opus at a very low
    bitrate — a degraded CALL connection): this is a milder but more
    acoustically realistic kind of degradation, closer to what a public
    video's audio track would actually carry."""
    wav_in = tmp_path / "clean_kd.wav"
    m4a_out = tmp_path / "degraded_kd.m4a"
    wav_out = tmp_path / "degraded_kd.wav"
    sf.write(wav_in, samples, sample_rate)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(wav_in),
            "-af", "aecho=0.8:0.7:40|60:0.35|0.2,acompressor",
            "-ar", str(sample_rate), "-ac", "1", "-c:a", "aac", "-b:a", "96k", str(m4a_out),
        ],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(m4a_out), "-ar", str(sample_rate), "-ac", "1", str(wav_out)],
        check=True, capture_output=True,
    )
    degraded, out_sr = sf.read(wav_out, dtype="float32", always_2d=False)
    assert out_sr == sample_rate
    if degraded.size < samples.size:
        degraded = np.pad(degraded, (0, samples.size - degraded.size))
    return degraded[: samples.size].astype(np.float32)


@pytest.fixture(scope="module")
def known_deepfake_speech(tmp_path_factory) -> tuple[np.ndarray, int]:
    """A REAL, modern-commercial-TTS-generated deepfake sample — deliberately
    NOT a macOS `say` fixture: verified by hand that `say`'s concatenative/
    formant synthesis is SO obviously synthetic (spoof_probability ~0.999
    on the base checkpoint) that no realistic degradation can move it. This
    regression test needs a genuinely BORDERLINE case, which is exactly
    what exposed the real finding below.

    One public demo clip from the IndieFake Dataset (Kumar, Verma, More —
    "IndieFake Dataset: A Benchmark Dataset for Audio Deepfake Detection",
    arxiv.org/html/2506.19014), licensed CC BY 4.0 (site footer: "Licensed
    under Creative Commons Attribution 4.0 International License") — NOT
    the full dataset, which is request-gated and not adopted into this
    project (see docs/risk-model.md's note on why). Fetched at test time
    rather than committed to git (avoids a binary-audio commit for one
    test), skipping gracefully if unreachable — same "network-optional,
    skip don't fail" convention as every other externally-sourced fixture
    in this project (see conftest.py's aasist_checkpoint)."""
    tmp_dir = tmp_path_factory.mktemp("indiefake_sample")
    wav_path = tmp_dir / "known_deepfake.wav"
    try:
        urllib.request.urlretrieve(_INDIEFAKE_SAMPLE_URL, wav_path)
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"IndieFake Dataset demo sample unreachable (no network?): {exc}")
    samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32)
    return samples, sr


def test_known_limitation_degrading_a_real_deepfake_lowers_its_spoof_score(
    aasist_as_deployed, known_deepfake_speech, tmp_path
):
    """KNOWN LIMITATION, not a regression to fix quietly — same "document
    the actual bad behaviour, flip the assertion once it's fixed"
    convention as test_zero_shot_intent_classifier.py's own known-
    limitation test. Found 2026-09-09 while investigating why a real
    external deepfake sample scored LOW risk in the live dashboard: room
    reverb + a lossy AAC re-encode (simulating realistic real-world
    channel conditions, e.g. a YouTube-sourced clip) consistently LOWERS
    AASIST's spoof-probability on this KNOWN, unambiguously synthetic
    clip — the opposite of what a robust countermeasure should do, and a
    CAUSAL (not just correlational) confirmation of the exact real-world
    risk already motivating this test file (NCC Group's vishing finding:
    poor audio quality helps real attacks evade detection).

    Specific to the DEPLOYED (Hindi-recalibrated) checkpoint, verified by
    hand — the base checkpoint scored this same window at 0.999 (correct)
    and was unmoved by this degradation; the recalibration that reduced
    Hindi false positives appears to have also made the decision more
    marginal, and it's exactly that marginal state degradation can push
    the wrong way. Verified by hand across 4 different IndieFake speakers
    before writing this test — all 4 showed the same direction under this
    degradation.

    Flip this assertion once AASIST's real-world robustness genuinely
    improves (a better fine-tune, a newer checkpoint, or an ensemble with
    a detector that doesn't share this failure mode) — don't delete it."""
    if shutil.which("ffmpeg") is None:
        # REAL BUG found 2026-09-09 via a real CI run: every OTHER
        # ffmpeg-dependent test in this file is downstream of the
        # `genuine_speech` fixture's own `shutil.which("say") is None`
        # skip (macOS-only, so it always skips first on Linux CI
        # runners) — meaning ffmpeg's presence was never actually
        # independently verified there. This test's fixture
        # (known_deepfake_speech) doesn't depend on `say` at all, so it
        # was the first to reach a bare ffmpeg call on a runner that,
        # it turns out, doesn't have ffmpeg installed either. Skip
        # explicitly instead of a confusing FileNotFoundError.
        pytest.skip("ffmpeg not available — needed for the reverb+AAC degradation")
    samples, sr = known_deepfake_speech
    window_samples = samples[: sr * 2] if samples.size > sr * 2 else samples

    clean_result = aasist_as_deployed.score(_window_at(window_samples, sr), context={})
    assert clean_result.score is not None

    degraded = _reverb_and_lossy_reencode(window_samples, sr, tmp_path)
    degraded_result = aasist_as_deployed.score(_window_at(degraded, sr), context={})
    assert degraded_result.score is not None

    assert degraded_result.score < clean_result.score, (
        f"expected the KNOWN LIMITATION to still reproduce (degraded score should read "
        f"LOWER than clean) — clean={clean_result.score:.3f} degraded={degraded_result.score:.3f}. "
        "If this now FAILS because degraded >= clean, that's genuinely good news: AASIST's "
        "real-world robustness may have improved — investigate and, if confirmed, flip this "
        "assertion rather than deleting the test."
    )


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

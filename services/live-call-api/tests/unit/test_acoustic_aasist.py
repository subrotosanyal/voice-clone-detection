import numpy as np
import pytest

from app.adapters.detectors.acoustic_aasist import AasistAcousticDetector, _deterministic_pad
from app.domain.models import AudioWindow

SR = 16_000


def _window(samples: np.ndarray, sample_rate: int = SR) -> AudioWindow:
    return AudioWindow(session_id="s1", seq=0, sample_rate=sample_rate, samples=samples, window_start_ms=0)


def test_deterministic_pad_truncates_when_long_enough():
    x = np.arange(70_000, dtype=np.float32)
    padded = _deterministic_pad(x, 64_600)
    assert padded.shape[0] == 64_600
    np.testing.assert_array_equal(padded, x[:64_600])  # first samples, not random offset


def test_deterministic_pad_tiles_when_short():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    padded = _deterministic_pad(x, 10)
    assert padded.shape[0] == 10
    np.testing.assert_array_equal(padded, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1])


def test_missing_checkpoint_raises_with_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_checkpoint.py"):
        AasistAcousticDetector(checkpoint_path=str(tmp_path / "does_not_exist.pth"))


def test_abstains_on_near_silence(aasist_checkpoint):
    detector = AasistAcousticDetector(checkpoint_path=aasist_checkpoint, floor_rms=1e-3)
    silence = np.zeros(SR * 2, dtype=np.float32)

    result = detector.score(_window(silence), context={})

    assert result.score is None
    assert result.abstain_reason is not None


def test_scores_real_audio_and_returns_full_breakdown(aasist_checkpoint):
    detector = AasistAcousticDetector(checkpoint_path=aasist_checkpoint)
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.05, SR * 2).astype(np.float32)

    result = detector.score(_window(audio), context={})

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0
    assert abs(result.detail["spoof_probability"] + result.detail["bonafide_probability"] - 1.0) < 1e-5
    assert result.detail["spoof_probability"] == result.score
    assert len(result.detail["raw_logits"]) == 2
    assert result.detail["checkpoint_sha256"]
    assert "resampled_from_hz" not in result.detail  # already 16kHz


def test_is_deterministic_across_repeated_calls(aasist_checkpoint):
    detector = AasistAcousticDetector(checkpoint_path=aasist_checkpoint)
    rng = np.random.default_rng(1)
    audio = rng.normal(0, 0.05, SR * 2).astype(np.float32)
    window = _window(audio)

    r1 = detector.score(window, context={})
    r2 = detector.score(window, context={})

    assert r1.score == r2.score
    assert r1.detail["raw_logits"] == r2.detail["raw_logits"]


def test_resamples_non_16k_audio(aasist_checkpoint):
    detector = AasistAcousticDetector(checkpoint_path=aasist_checkpoint)
    rng = np.random.default_rng(2)
    audio_8k = rng.normal(0, 0.05, 8_000 * 2).astype(np.float32)  # 2s at 8kHz

    result = detector.score(_window(audio_8k, sample_rate=8_000), context={})

    assert result.score is not None
    assert result.detail["resampled_from_hz"] == 8_000


_FINETUNED_OUT_LAYER = "app/adapters/detectors/vendor/checkpoints/AASIST_hindi_finetuned_out_layer.pth"


def test_without_finetuned_param_detail_flag_is_false(aasist_checkpoint):
    detector = AasistAcousticDetector(checkpoint_path=aasist_checkpoint)
    rng = np.random.default_rng(3)
    audio = rng.normal(0, 0.05, SR * 2).astype(np.float32)

    result = detector.score(_window(audio), context={})

    assert result.detail["hindi_finetuned_out_layer"] is False
    assert "finetuned_out_layer_checkpoint" not in result.detail


def test_finetuned_param_loads_and_flags_detail(aasist_checkpoint):
    detector = AasistAcousticDetector(
        checkpoint_path=aasist_checkpoint, finetuned_out_layer_path=_FINETUNED_OUT_LAYER
    )
    rng = np.random.default_rng(3)
    audio = rng.normal(0, 0.05, SR * 2).astype(np.float32)

    result = detector.score(_window(audio), context={})

    assert result.score is not None
    assert result.detail["hindi_finetuned_out_layer"] is True
    assert result.detail["finetuned_out_layer_checkpoint"] == "AASIST_hindi_finetuned_out_layer.pth"


def test_finetuned_out_layer_actually_changes_the_score(aasist_checkpoint):
    """Not just an interface test — proves the fine-tuned weights are
    genuinely swapped in, not silently ignored: the same audio must score
    differently through the recalibrated out_layer than through the
    original one."""
    rng = np.random.default_rng(4)
    audio = rng.normal(0, 0.05, SR * 2).astype(np.float32)
    window = _window(audio)

    base_detector = AasistAcousticDetector(checkpoint_path=aasist_checkpoint)
    finetuned_detector = AasistAcousticDetector(
        checkpoint_path=aasist_checkpoint, finetuned_out_layer_path=_FINETUNED_OUT_LAYER
    )

    base_score = base_detector.score(window, context={}).score
    finetuned_score = finetuned_detector.score(window, context={}).score

    assert base_score != finetuned_score


def test_missing_finetuned_out_layer_file_raises_with_actionable_message(aasist_checkpoint, tmp_path):
    with pytest.raises(FileNotFoundError, match="eval/indian_language"):
        AasistAcousticDetector(
            checkpoint_path=aasist_checkpoint,
            finetuned_out_layer_path=str(tmp_path / "does_not_exist.pth"),
        )

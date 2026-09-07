import numpy as np

from app.pipeline.windowing import make_windows


def test_exact_multiple_produces_expected_window_count():
    sr = 16_000
    duration_s = 3.0
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)

    windows = list(make_windows("s1", samples, sr, window_ms=2000, hop_ms=500))

    # 2s window, 0.5s hop, over 3.0s of audio: starts at 0.0,0.5,...,2.5 => 6 windows,
    # then the loop condition (start < total) stops once start=3.0s == total.
    assert len(windows) == 6
    assert [w.window_start_ms for w in windows] == [0, 500, 1000, 1500, 2000, 2500]


def test_final_partial_window_is_still_yielded():
    sr = 16_000
    samples = np.zeros(int(sr * 2.2), dtype=np.float32)  # 2.2s

    windows = list(make_windows("s1", samples, sr, window_ms=2000, hop_ms=500))

    assert windows[-1].samples.shape[0] < int(sr * 2.0)
    assert windows[-1].samples.shape[0] > 0


def test_seq_increments_from_start_seq():
    sr = 16_000
    samples = np.zeros(int(sr * 1.0), dtype=np.float32)

    windows = list(make_windows("s1", samples, sr, window_ms=200, hop_ms=200, start_seq=10))

    assert [w.seq for w in windows] == list(range(10, 10 + len(windows)))


def test_empty_buffer_yields_nothing():
    sr = 16_000
    windows = list(make_windows("s1", np.zeros(0, dtype=np.float32), sr, window_ms=2000, hop_ms=500))
    assert windows == []

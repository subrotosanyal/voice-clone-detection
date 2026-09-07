import numpy as np
import pytest

from app.adapters.detectors.third_signal_router import ThirdSignalRouter
from app.domain.models import AudioWindow, DetectorResult

WINDOW = AudioWindow(session_id="s1", seq=0, sample_rate=16_000, samples=np.zeros(1), window_start_ms=0)


class _FixedDetector:
    def __init__(self, name: str, score, abstain_reason=None):
        self.name = name
        self.version = "test"
        self._score = score
        self._abstain_reason = abstain_reason

    def score(self, window, context):
        return DetectorResult(
            detector_name=self.name, detector_version=self.version, score=self._score, abstain_reason=self._abstain_reason
        )


def test_mode_contextual_ignores_consistency():
    router = ThirdSignalRouter(
        contextual_detector=_FixedDetector("ctx", 0.4),
        consistency_detector=_FixedDetector("cons", 0.9),
        mode="contextual",
    )
    result = router.score(WINDOW, context={})
    assert result.score == 0.4
    assert result.detail["active_mode"] == "contextual"


def test_mode_consistency_ignores_contextual():
    router = ThirdSignalRouter(
        contextual_detector=_FixedDetector("ctx", 0.4),
        consistency_detector=_FixedDetector("cons", 0.9),
        mode="consistency",
    )
    result = router.score(WINDOW, context={})
    assert result.score == 0.9
    assert result.detail["active_mode"] == "consistency"


def test_mode_auto_prefers_consistency_when_available():
    router = ThirdSignalRouter(
        contextual_detector=_FixedDetector("ctx", 0.4),
        consistency_detector=_FixedDetector("cons", 0.9),
        mode="auto",
    )
    result = router.score(WINDOW, context={})
    assert result.score == 0.9
    assert result.detail["active_mode"] == "consistency"


def test_mode_auto_falls_back_to_contextual_on_abstain():
    router = ThirdSignalRouter(
        contextual_detector=_FixedDetector("ctx", 0.4),
        consistency_detector=_FixedDetector("cons", None, abstain_reason="no enrollment"),
        mode="auto",
    )
    result = router.score(WINDOW, context={})
    assert result.score == 0.4
    assert result.detail["active_mode"] == "contextual"
    assert result.detail["fell_back_from"] == "consistency"


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        ThirdSignalRouter(
            contextual_detector=_FixedDetector("ctx", 0.4),
            consistency_detector=_FixedDetector("cons", 0.9),
            mode="bogus",
        )

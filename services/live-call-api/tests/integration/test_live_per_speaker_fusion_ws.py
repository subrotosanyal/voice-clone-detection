"""End-to-end test through the real FastAPI app's WebSocket path:
real-time per-speaker fusion (app/api/ws_router.py) — once a second
distinct voice is detected, every window's response should ALSO carry a
`speaker_score`, scoped to whichever speaker is currently talking, with
its own derived session_id and independent EMA smoothing history. Same
real-ECAPA-TDNN fixture style as test_live_diarization_ws.py.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from tests.audio_fixtures import VOICE_A_FORMANTS, VOICE_B_FORMANTS, formant_voice

SR = 16_000
WINDOW_MS = 2000


def _send(ws, seq: int, samples) -> dict:
    ws.send_json(
        {
            "seq": seq,
            "sample_rate": SR,
            "pcm_f32": samples.tolist(),
            "window_start_ms": seq * WINDOW_MS,
        }
    )
    return ws.receive_json()


def test_speaker_score_absent_for_a_single_speaker_present_once_a_second_appears(aasist_checkpoint):
    session_id = "ws-per-speaker-fusion-test"
    voice_a_take1 = formant_voice(VOICE_A_FORMANTS, sample_rate=SR, duration_s=WINDOW_MS / 1000, seed=1, pitch_hz=120)
    voice_a_take2 = formant_voice(VOICE_A_FORMANTS, sample_rate=SR, duration_s=WINDOW_MS / 1000, seed=2, pitch_hz=121)
    voice_b_take1 = formant_voice(VOICE_B_FORMANTS, sample_rate=SR, duration_s=WINDOW_MS / 1000, seed=3, pitch_hz=180)

    with TestClient(app) as client, client.websocket_connect(f"/v1/stream/{session_id}") as ws:
        body1 = _send(ws, 0, voice_a_take1)  # only speaker so far — no speaker_score yet
        body2 = _send(ws, 1, voice_a_take2)  # still only one speaker
        body3 = _send(ws, 2, voice_b_take1)  # second speaker appears — speaker_score should now be present

    assert body1["speaker_score"] is None
    assert body2["speaker_score"] is None

    assert body3["live_speaker"]["speaker_count"] == 2
    assert body3["speaker_score"] is not None
    assert body3["speaker_score"]["session_id"] == f"{session_id}::{body3['live_speaker']['speaker_label']}"
    # The whole-call score and the per-speaker score are genuinely
    # different sessions with independent EMA history — different
    # session_id proves they're not the same object/state.
    assert body3["speaker_score"]["session_id"] != body3["session_id"]

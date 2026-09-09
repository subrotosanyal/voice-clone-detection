"""End-to-end test through the real FastAPI app's WebSocket path: real
`live_diarization` config (config/risk_formula.yaml), real ECAPA-TDNN
embeddings, proving `live_speaker` actually appears on the WS response
and tracks speakers correctly across windows — same "exercise the real
app, not a mock" style as test_live_transcription_ws.py, just for
app/pipeline/live_diarization.py instead.

Uses tests/audio_fixtures.py's formant_voice — real embedding-model
input, deterministic, no macOS `say` dependency (unlike the live
transcription test, this doesn't need real speech content, just real
enough timbre for ECAPA-TDNN to tell voices apart).
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


def test_live_speaker_appears_on_the_ws_response_and_tracks_across_windows(aasist_checkpoint):
    session_id = "ws-live-diarization-test"
    voice_a_take1 = formant_voice(VOICE_A_FORMANTS, sample_rate=SR, duration_s=WINDOW_MS / 1000, seed=1, pitch_hz=120)
    voice_b_take1 = formant_voice(VOICE_B_FORMANTS, sample_rate=SR, duration_s=WINDOW_MS / 1000, seed=3, pitch_hz=180)
    voice_a_take2 = formant_voice(VOICE_A_FORMANTS, sample_rate=SR, duration_s=WINDOW_MS / 1000, seed=2, pitch_hz=122)

    with TestClient(app) as client, client.websocket_connect(f"/v1/stream/{session_id}") as ws:
        body1 = _send(ws, 0, voice_a_take1)
        body2 = _send(ws, 1, voice_b_take1)
        body3 = _send(ws, 2, voice_a_take2)

    assert body1["live_speaker"] is not None, (
        "expected live_speaker on the WS response — config/risk_formula.yaml's "
        "live_diarization section should be active by default"
    )
    assert body1["live_speaker"]["speaker_label"] == "speaker_1"
    assert body1["live_speaker"]["is_new_speaker"] is True
    assert body1["live_speaker"]["speaker_count"] == 1

    assert body2["live_speaker"]["speaker_label"] == "speaker_2"
    assert body2["live_speaker"]["is_new_speaker"] is True
    assert body2["live_speaker"]["speaker_count"] == 2

    # The real correctness property: a second take of voice A is reunited
    # with speaker_1, not minted as a new speaker_3.
    assert body3["live_speaker"]["speaker_label"] == "speaker_1"
    assert body3["live_speaker"]["is_new_speaker"] is False
    assert body3["live_speaker"]["speaker_count"] == 2

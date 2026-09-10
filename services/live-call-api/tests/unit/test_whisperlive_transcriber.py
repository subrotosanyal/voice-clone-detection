"""Tests for WhisperLiveTranscriber — the LIVE path's bridge to a
collabora/WhisperLive server (see that module's own docstring for why).

Fakes `websockets.sync.client.connect` throughout — no real WhisperLive
server, same "fake the expensive/external port, test the real logic"
style used elsewhere in this suite (e.g. test_fetch_llm_model.py fakes
hf_hub_download(), test_whisper_transcriber.py fakes the model object).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.adapters.transcription import whisperlive_transcriber as wl_module
from app.adapters.transcription.whisperlive_transcriber import WhisperLiveTranscriber

SR = 16_000


class _FakeWS:
    """Stands in for websockets.sync.client's connection object — a
    scripted queue of incoming server messages, and a record of
    everything sent, so a test can assert on both without a real
    server."""

    def __init__(self, incoming: list) -> None:
        self._incoming = list(incoming)
        self.sent: list = []

    def send(self, data):
        self.sent.append(data)

    def recv(self, timeout=None):
        if not self._incoming:
            raise TimeoutError("fake server: no more scripted messages")
        return self._incoming.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_connect(incoming: list):
    ws = _FakeWS(incoming)

    def _connect(uri, open_timeout=None):
        return ws

    return _connect, ws


def _server_ready() -> str:
    return json.dumps({"message": "SERVER_READY", "backend": "faster_whisper"})


def _segments(*texts: str) -> str:
    return json.dumps({"segments": [{"text": t, "completed": True} for t in texts]})


def _language(code: str) -> str:
    return json.dumps({"language": code, "language_prob": 0.98})


def _disconnect() -> str:
    return json.dumps({"message": "DISCONNECT"})


@pytest.fixture
def transcriber() -> WhisperLiveTranscriber:
    return WhisperLiveTranscriber(host="whisper-live", port=9090, model="small")


def test_happy_path_returns_concatenated_segments_and_language(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect(
        [_server_ready(), _language("en"), _segments("please transfer", "the money now"), _disconnect()]
    )
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == "please transfer the money now"
    assert result.language == "en"
    assert result.detector_name == "whisperlive_transcriber"
    assert result.detector_version == "whisperlive@small"


def test_sends_handshake_before_any_audio(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect([_server_ready(), _segments("hi"), _disconnect()])
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    transcriber.transcribe(samples, SR)

    first_sent = ws.sent[0]
    assert isinstance(first_sent, str)
    handshake = json.loads(first_sent)
    assert handshake["task"] == "transcribe"
    assert handshake["model"] == "small"
    assert handshake["use_vad"] is True
    assert "uid" in handshake and handshake["uid"]
    # Every remaining sent item until END_OF_AUDIO is raw binary PCM.
    audio_frames = ws.sent[1:-1]
    assert all(isinstance(f, (bytes, bytearray)) for f in audio_frames)
    assert ws.sent[-1] == b"END_OF_AUDIO"


def test_two_calls_use_different_uids(transcriber, monkeypatch):
    """Each transcribe() call is an independent ~8s trailing window, not a
    continuation of a prior stream — see this module's own CALLING
    PATTERN docstring note — so every call must open a fresh handshake,
    not reuse a uid a previous call already used."""
    seen_uids = []

    def _connect(uri, open_timeout=None):
        ws = _FakeWS([_server_ready(), _segments("ok"), _disconnect()])
        real_send = ws.send

        def _send(data):
            if isinstance(data, str):
                seen_uids.append(json.loads(data)["uid"])
            real_send(data)

        ws.send = _send
        return ws

    monkeypatch.setattr(wl_module, "connect", _connect)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    transcriber.transcribe(samples, SR)
    transcriber.transcribe(samples, SR)

    assert len(seen_uids) == 2
    assert seen_uids[0] != seen_uids[1]


def test_abstains_when_server_never_ready(transcriber, monkeypatch):
    """No SERVER_READY within connect_timeout_s — must abstain (empty
    transcript), never send audio, never raise."""
    connect_fn, ws = _fake_connect([])  # recv() raises TimeoutError immediately
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == ""
    assert result.language is None
    assert ws.sent == [ws.sent[0]], "must not send audio if the server never reported SERVER_READY"


def test_never_raises_when_connection_fails(transcriber, monkeypatch):
    def _connect(uri, open_timeout=None):
        raise ConnectionRefusedError("whisper-live not up yet")

    monkeypatch.setattr(wl_module, "connect", _connect)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == ""
    assert result.language is None


def test_empty_samples_abstain_without_connecting(transcriber, monkeypatch):
    def _connect(uri, open_timeout=None):
        raise AssertionError("must not connect for an empty buffer")

    monkeypatch.setattr(wl_module, "connect", _connect)

    result = transcriber.transcribe(np.array([], dtype=np.float32), SR)

    assert result.text == ""


def test_captures_segments_even_when_disconnect_never_arrives(transcriber, monkeypatch):
    """Regression test for the REAL BUG this module's own docstring
    documents (2026-09-10): the real collabora/whisper-live server closes
    the connection almost immediately after END_OF_AUDIO WITHOUT ever
    sending a DISCONNECT message in the normal (non-timeout) case — the
    old code, which only stopped collecting on DISCONNECT or a full
    result_timeout_s of silence AFTER sending END_OF_AUDIO, got nothing
    in this exact scenario. No _disconnect() in this script at all —
    matching the real server's actual behaviour, not the old assumption."""
    connect_fn, ws = _fake_connect([_server_ready(), _language("hi"), _segments("namaste")])
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == "namaste"
    assert result.language == "hi"


def test_end_of_audio_is_sent_after_segments_are_collected_not_before(transcriber, monkeypatch):
    """The actual fix: END_OF_AUDIO must be the LAST thing sent, after
    whatever segments arrive — sending it first is what let the real
    server's connection-close race lose the result (see this module's
    own REAL BUG note)."""
    connect_fn, ws = _fake_connect([_server_ready(), _segments("please transfer the money")])
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == "please transfer the money"
    assert ws.sent[-1] == b"END_OF_AUDIO"


def test_a_failed_end_of_audio_send_does_not_lose_an_already_collected_result(transcriber, monkeypatch):
    """The connection can legitimately already be gone (server-side
    closed) by the time we try to send END_OF_AUDIO, now that it's sent
    LAST — that send failing must not raise past transcribe() or discard
    whatever was already collected."""
    connect_fn, ws = _fake_connect([_server_ready(), _segments("ok")])
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    real_send = ws.send

    def _send_that_fails_on_eof(data):
        if data == b"END_OF_AUDIO":
            raise ConnectionError("connection already closed by server")
        real_send(data)

    ws.send = _send_that_fails_on_eof
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == "ok"


def test_stops_collecting_promptly_once_the_scripted_queue_runs_dry(transcriber, monkeypatch):
    """Guards against the bounded-deadline collection loop turning into a
    busy-wait for the full result_timeout_s (20s default) once there's
    nothing left to read — would make every test using this fixture slow
    without this. A fake connection's recv() raises immediately (no real
    delay) once its queue is empty, so this should return well under a
    second if the loop exits on the first empty read rather than retrying
    until a real wall-clock deadline elapses."""
    import time

    connect_fn, ws = _fake_connect([_server_ready(), _segments("done")])
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    start = time.monotonic()
    result = transcriber.transcribe(samples, SR)
    elapsed = time.monotonic() - start

    assert result.text == "done"
    assert elapsed < 2.0, f"took {elapsed:.2f}s — collection loop may be busy-waiting instead of exiting promptly"


def test_resamples_non_16k_audio_before_sending(transcriber, monkeypatch):
    """Confirms the resample path runs (real scipy resample_poly, no
    mock) rather than sending audio at the wrong sample rate — the
    server always expects 16kHz."""
    connect_fn, ws = _fake_connect([_server_ready(), _segments("ok"), _disconnect()])
    monkeypatch.setattr(wl_module, "connect", connect_fn)
    samples_48k = np.random.default_rng(0).uniform(-0.5, 0.5, size=48_000).astype(np.float32)

    result = transcriber.transcribe(samples_48k, 48_000)

    assert result.text == "ok"
    total_audio_bytes = sum(len(f) for f in ws.sent[1:-1])
    # float32 = 4 bytes/sample; resampled 48k->16k halves-ish the sample count.
    assert total_audio_bytes == pytest.approx(16_000 * 4, rel=0.05)

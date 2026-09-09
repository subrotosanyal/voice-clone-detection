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

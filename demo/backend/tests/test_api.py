"""Backend REST API and audio WebSocket with the fake ASR and a scripted VAD (no GPU)."""

import time
from dataclasses import replace

import numpy as np
from fastapi.testclient import TestClient

from backend.config import Config
from backend.server import FRAME_HEADER, create_app
from backend.timing import FRAME_SAMPLES
from backend.vad.silero import VADEvent

from .fakes import FakeASR, ScriptedVAD


def frame_bytes(seq: int) -> bytes:
    return FRAME_HEADER.pack(seq, time.time() * 1000) + np.zeros(FRAME_SAMPLES, "<i2").tobytes()


def test_session_api_and_log(tmp_path):
    cfg = replace(Config(), runs_dir=str(tmp_path / "runs"))
    vad = ScriptedVAD(
        {1: VADEvent("start", 0, FRAME_SAMPLES), 8: VADEvent("end", 4 * FRAME_SAMPLES, 4 * FRAME_SAMPLES)}
    )
    app = create_app(cfg, asr_engine=FakeASR(time.monotonic_ns), vad=vad)

    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        before = client.get("/api/session").json()
        assert before["session"] is None and before["asr"] == {"engine": "fake"}
        assert before["config"]["vad"]["min_silence_ms"] == cfg.vad.min_silence_ms
        assert client.get("/api/session/log").status_code == 404

        with client.websocket_connect("/ws/audio") as ws:
            for seq in range(12):
                ws.send_bytes(frame_bytes(seq))
            ws.send_json({"type": "stop"})
            events = []
            while not events or events[-1]["type"] != "session_end":
                events.append(ws.receive_json())

        info = client.get("/api/session").json()["session"]
        assert info["closed"] and not info["active"]
        assert info["audio_s"] == 12 * FRAME_SAMPLES / 16000
        assert [row["reason"] for row in info["segments"]] == ["vad_end"]
        assert info["summary"]["vad_fire_to_final"]["count"] == 1

        log = client.get("/api/session/log")
        assert log.status_code == 200
        lines = log.text.strip().splitlines()
        assert len(lines) > 12 and '"type":"session_end"' in lines[-1]


def test_bad_frame_size_reports_error(tmp_path):
    cfg = replace(Config(), runs_dir=str(tmp_path / "runs"))
    app = create_app(cfg, asr_engine=FakeASR(time.monotonic_ns), vad=ScriptedVAD({}))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as ws:
            assert ws.receive_json()["type"] == "session_start"
            ws.send_bytes(b"\x00" * 10)
            assert ws.receive_json() == {"type": "error", "message": "bad frame size 10"}
            ws.send_json({"type": "stop"})


def test_bad_text_message_reports_error_and_session_continues(tmp_path):
    cfg = replace(Config(), runs_dir=str(tmp_path / "runs"))
    app = create_app(cfg, asr_engine=FakeASR(time.monotonic_ns), vad=ScriptedVAD({}))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as ws:
            assert ws.receive_json()["type"] == "session_start"
            ws.send_text("not json")
            assert ws.receive_json()["message"].startswith("bad message: JSONDecodeError")
            ws.send_json({"type": "rtt"})
            assert ws.receive_json() == {"type": "error", "message": "bad message: KeyError('rtt_ms')"}
            ws.send_json({"type": "ping", "id": 1, "client_ts": 0})
            assert ws.receive_json()["type"] == "pong"
            ws.send_json({"type": "stop"})

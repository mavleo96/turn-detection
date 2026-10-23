"""Headless end-to-end check: stream a WAV over /ws/audio in real time; check events, REST API and JSONL."""

import time
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import Config
from backend.eventlog import read_events
from backend.server import FRAME_HEADER, create_app
from backend.timing import FRAME_SAMPLES, SAMPLE_RATE

pytestmark = pytest.mark.gpu


def encode(seq: int, frame: np.ndarray) -> bytes:
    pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
    return FRAME_HEADER.pack(seq, time.time() * 1000) + pcm


def test_stream_wav_over_websocket(tmp_path, asr_engine, silero_vad, speech_wav):
    cfg = replace(Config(), runs_dir=str(tmp_path / "runs"))
    audio = np.concatenate([speech_wav[: 6 * SAMPLE_RATE], np.zeros(SAMPLE_RATE, np.float32)])
    frame_s = FRAME_SAMPLES / SAMPLE_RATE

    app = create_app(cfg, asr_engine=asr_engine, vad=silero_vad)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as ws:
            # A second client is rejected while this session is active.
            with client.websocket_connect("/ws/audio") as ws2:
                assert ws2.receive_json()["type"] == "error"

            t0 = time.perf_counter()
            for seq in range(len(audio) // FRAME_SAMPLES):
                frame = audio[seq * FRAME_SAMPLES : (seq + 1) * FRAME_SAMPLES]
                while time.perf_counter() < t0 + seq * frame_s:  # real-time pacing
                    time.sleep(0.001)
                ws.send_bytes(encode(seq, frame))
            ws.send_json({"type": "ping", "id": 1, "client_ts": 0})
            ws.send_json({"type": "stop"})

            events = []
            while not events or events[-1]["type"] != "session_end":
                events.append(ws.receive_json())

        info = client.get("/api/session").json()["session"]
        assert client.get("/api/session/log").status_code == 200

    types = [e["type"] for e in events]
    assert types[0] == "session_start"
    assert "vad_start" in types and "vad_end" in types and "asr_partial" in types and "pong" in types
    finals = [e for e in events if e["type"] == "asr_final" and e["reason"] == "vad_end"]
    assert finals and finals[0]["text"]
    # In real time, the user-perceived end-to-final latency is bounded by silence + flush.
    assert 0 < finals[0]["speech_end_to_final_ms"] < 1000
    assert info["closed"] and info["segments"][0]["text"] == finals[0]["text"]

    logs = list((tmp_path / "runs").glob("*.jsonl"))
    assert len(logs) == 1
    logged = read_events(logs[0])
    assert sum(e["type"] == "frame" for e in logged) == len(audio) // FRAME_SAMPLES
    assert logged[-1]["type"] == "session_end"

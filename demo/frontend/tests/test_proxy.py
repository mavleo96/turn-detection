"""The frontend's audio WebSocket proxy, against a fake backend served on a local port."""

import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from frontend.app import create_app
from frontend.config import FrontendConfig


def fake_backend() -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"type": "session_start"})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                if msg.get("bytes") is not None:
                    await ws.send_json({"type": "got_bytes", "n": len(msg["bytes"])})
                elif msg.get("text") == '{"type": "stop"}':
                    await ws.send_json({"type": "session_end"})
                    await ws.close()
                    return
                else:
                    await ws.send_json({"type": "got_text", "text": msg["text"]})
        except WebSocketDisconnect:
            return

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def backend_url():
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(fake_backend(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "fake backend did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_proxy_relays_both_directions(backend_url):
    app = create_app(FrontendConfig(backend_url=backend_url))
    with TestClient(app) as client, client.websocket_connect("/ws/audio") as ws:
        assert ws.receive_json() == {"type": "session_start"}
        ws.send_bytes(b"\x01" * 1036)
        assert ws.receive_json() == {"type": "got_bytes", "n": 1036}
        ws.send_text('{"type": "ping"}')
        assert ws.receive_json() == {"type": "got_text", "text": '{"type": "ping"}'}
        ws.send_text('{"type": "stop"}')
        assert ws.receive_json() == {"type": "session_end"}


def test_proxy_reports_backend_down():
    app = create_app(FrontendConfig(backend_url=f"http://127.0.0.1:{free_port()}"))
    with TestClient(app) as client, client.websocket_connect("/ws/audio") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error" and "backend unavailable" in msg["message"]

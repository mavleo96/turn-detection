"""Backend: FastAPI with the audio WebSocket and a small REST API. No UI. One live session at a time.

Run (from demo/):  python -m backend.server --config configs/demo.yaml

WebSocket /ws/audio, client -> backend:
  binary  [seq: uint32 LE][client_ts_ms: float64 LE][512 x int16 LE]   one 32 ms frame of 16 kHz PCM
  text    {"type": "ping", "id": n, "client_ts": ms} | {"type": "rtt", "id": n, "rtt_ms": x} | {"type": "stop"}
backend -> client: JSON pipeline events (see pipeline.py) plus {"type": "pong"} and {"type": "error"}.

REST:
  GET /api/health        {"status": "ok"}
  GET /api/session       active or most recent session: summary, segment rows, config, ASR info
  GET /api/session/log   that session's JSONL event log
"""

import argparse
import asyncio
import json
import secrets
import struct
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import Config, load_config
from .eventlog import EventLog
from .pipeline import Session
from .timing import FRAME_SAMPLES, SAMPLE_RATE

FRAME_HEADER = struct.Struct("<Id")
FRAME_BYTES = FRAME_HEADER.size + FRAME_SAMPLES * 2
# Pipeline events forwarded to the client (everything is logged).
CLIENT_EVENTS = {
    "session_start",
    "vad_start",
    "vad_end",
    "asr_partial",
    "asr_final",
    "network",
    "error",
    "session_end",
}


class AppState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vad = None
        self.asr = None
        self.session: Session | None = None  # active session
        self.last_session: Session | None = None  # active or most recent
        self.last_log_path: Path | None = None


def create_app(cfg: Config, asr_engine=None, vad=None) -> FastAPI:
    """Build the app. `asr_engine` / `vad` can be injected (tests); otherwise they load at startup."""
    runs_dir = Path(cfg.runs_dir).resolve()
    state = AppState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if vad is None:
            from .vad.silero import SileroVAD

            state.vad = SileroVAD(cfg.vad)
        else:
            state.vad = vad
        if asr_engine is None:
            from .asr.nemotron_hf import NemotronHFEngine

            state.asr = await asyncio.to_thread(NemotronHFEngine, cfg.asr)
        else:
            state.asr = asr_engine
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.demo = state

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/session")
    async def session_info():
        s = state.last_session
        info = {"config": cfg.to_dict(), "asr": state.asr.info() if state.asr else None, "session": None}
        if s is not None:
            info["session"] = {
                "session_id": s.session_id,
                "active": s is state.session,
                "closed": s.closed,
                "audio_s": round(s.clock.num_frames * FRAME_SAMPLES / SAMPLE_RATE, 3),
                "summary": s.summary(),
                "segments": s.segment_rows(),
            }
        return info

    @app.get("/api/session/log")
    async def session_log():
        path = state.last_log_path
        if path is None or not path.exists():
            raise HTTPException(404, "no session log yet")
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket):
        await ws.accept()
        if state.session is not None:
            await ws.send_json({"type": "error", "message": "Another session is already active."})
            await ws.close(code=1013)
            return

        loop = asyncio.get_running_loop()
        outq: asyncio.Queue = asyncio.Queue()
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        log = EventLog(runs_dir / f"{session_id}.jsonl")

        def sink(event: dict) -> None:
            log.write(event)
            if event["type"] in CLIENT_EVENTS:
                loop.call_soon_threadsafe(outq.put_nowait, event)

        session = Session(session_id, cfg, state.vad, state.asr, sink)
        state.session = state.last_session = session
        state.last_log_path = log.path
        sender = asyncio.create_task(_send_events(ws, outq))
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    if len(data) != FRAME_BYTES:
                        await outq.put({"type": "error", "message": f"bad frame size {len(data)}"})
                        continue
                    seq, client_ts = FRAME_HEADER.unpack_from(data)
                    pcm = np.frombuffer(data, dtype="<i2", offset=FRAME_HEADER.size).astype(np.float32) / 32768.0
                    session.on_frame(seq, pcm, client_ts)
                elif msg.get("text") is not None:
                    try:
                        m = json.loads(msg["text"])
                        kind = m["type"]
                        rtt_ms = float(m["rtt_ms"]) if kind == "rtt" else None
                    except (ValueError, KeyError, TypeError) as e:
                        await outq.put({"type": "error", "message": f"bad message: {e!r}"})
                        continue
                    if kind == "ping":
                        await outq.put({"type": "pong", "id": m.get("id"), "client_ts": m.get("client_ts")})
                    elif kind == "rtt":
                        session.on_rtt(rtt_ms)
                    elif kind == "stop":
                        break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.write({"type": "error", "message": repr(e)})
            await outq.put({"type": "error", "message": repr(e)})
        finally:
            try:
                await asyncio.to_thread(session.close)
            finally:
                state.session = None
                log.close()
                outq.put_nowait(None)
                await sender
                try:
                    await ws.close()
                except RuntimeError:
                    pass  # already closed by the client

    return app


async def _send_events(ws: WebSocket, q: asyncio.Queue) -> None:
    while (event := await q.get()) is not None:
        try:
            await ws.send_json(event)
        except Exception:
            pass  # client gone; keep draining so the session can finish


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo backend: streaming ASR + VAD")
    ap.add_argument("--config", default="configs/demo.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, ws_max_size=1 << 20)


if __name__ == "__main__":
    main()

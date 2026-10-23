"""Frontend: serves the page (Gradio + static browser code) and proxies the audio WebSocket to the
backend, so the browser only talks to this process. No ML code; never imports the backend.

Run (from demo/, after the backend):  python -m frontend.app --config configs/demo.yaml
"""

import argparse
import asyncio
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .config import FrontendConfig, load_config
from .ui import build_ui

STATIC_DIR = Path(__file__).parent / "static"


def create_app(cfg: FrontendConfig) -> FastAPI:
    app = FastAPI()
    app.mount("/td-static", StaticFiles(directory=STATIC_DIR), name="td-static")

    @app.websocket("/ws/audio")
    async def ws_proxy(ws: WebSocket):
        await ws.accept()
        try:
            upstream = await connect(cfg.backend_ws_url, max_size=None, open_timeout=3)
        except Exception:
            await ws.send_json({"type": "error", "message": f"backend unavailable at {cfg.backend_url}"})
            await ws.close(code=1011)
            return

        async def browser_to_backend():
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if msg.get("bytes") is not None:
                        await upstream.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await upstream.send(msg["text"])
            except (WebSocketDisconnect, ConnectionClosed):
                pass
            finally:
                await upstream.close()  # browser gone: backend ends the session

        async def backend_to_browser():
            try:
                async for msg in upstream:
                    if isinstance(msg, bytes):
                        await ws.send_bytes(msg)
                    else:
                        await ws.send_text(msg)
            except (ConnectionClosed, RuntimeError):
                pass

        up = asyncio.create_task(browser_to_backend())
        await backend_to_browser()  # ends when the backend closes (after session_end) or the browser leaves
        up.cancel()
        try:
            await up
        except asyncio.CancelledError:
            pass
        try:
            await ws.close()
        except RuntimeError:
            pass  # already closed

    return gr.mount_gradio_app(
        app,
        build_ui(cfg),
        path="",
        head='<script src="/td-static/client.js" defer></script>',
        css_paths=[STATIC_DIR / "panel.css"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo frontend: browser UI + audio proxy to the backend")
    ap.add_argument("--config", default="configs/demo.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, ws_max_size=1 << 20)


if __name__ == "__main__":
    main()

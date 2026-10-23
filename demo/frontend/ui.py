"""Gradio layout. The live panel (mic, VAD indicator, transcript) is plain HTML driven by
static/client.js over the audio WebSocket; the tables poll the backend's REST API on a timer."""

import functools
import tempfile
from pathlib import Path

import gradio as gr
import httpx
import pandas as pd

from .config import FrontendConfig

LIVE_PANEL = """
<div id="td-panel">
  <div class="td-controls">
    <button id="td-start" class="td-btn td-primary">Start mic</button>
    <button id="td-stop" class="td-btn" disabled>Stop</button>
    <span id="td-status">idle</span>
  </div>
  <div class="td-row">
    <span id="td-vad" class="td-vad">silent</span>
    <span id="td-vad-lat" class="td-dim"></span>
    <span id="td-net" class="td-dim"></span>
  </div>
  <div id="td-transcript" class="td-transcript"></div>
</div>
"""

SEGMENT_COLUMNS = [
    "segment_id",
    "reason",
    "text",
    "speech_start_s",
    "speech_end_s",
    "vad_start_latency_ms",
    "vad_end_latency_ms",
    "vad_fire_to_final_ms",
    "speech_end_to_final_ms",
    "tokens",
    "token_perceived_p50_ms",
    "token_compute_p50_ms",
]
SUMMARY_COLUMNS = ["metric", "count", "mean", "p50", "p95", "min", "max"]


def config_markdown(info: dict) -> str:
    asr, vad = info["config"]["asr"], info["config"]["vad"]
    return (
        f"**ASR** `{asr['model_id']}` · chunk latency **{asr['chunk_latency_ms']} ms** · {asr['dtype']}  \n"
        f"**VAD** Silero · threshold {vad['threshold']} · min silence **{vad['min_silence_ms']} ms** · "
        f"speech pad {vad['speech_pad_ms']} ms"
    )


def session_tables(session: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Status line, summary table and segment table from the backend's `session` object."""
    summary = session["summary"]
    summary_df = pd.DataFrame(
        [[metric] + [stats[k] for k in SUMMARY_COLUMNS[1:]] for metric, stats in summary.items()],
        columns=SUMMARY_COLUMNS,
    )
    segments_df = pd.DataFrame(session["segments"], columns=SEGMENT_COLUMNS)
    status = "live" if session["active"] else "finished"
    line = f"Session `{session['session_id']}` ({status}) · {session['audio_s']:.1f} s audio"
    return line, summary_df, segments_df


@functools.cache
def log_dir() -> Path:
    """One folder per process for downloaded session logs."""
    return Path(tempfile.mkdtemp(prefix="demo-frontend-logs-"))


def build_ui(cfg: FrontendConfig) -> gr.Blocks:
    http = httpx.Client(base_url=cfg.backend_url, timeout=2.0)

    def refresh(last_sig):
        try:
            info = http.get("/api/session").json()
        except httpx.HTTPError:
            sig = "down"
            if sig == last_sig:
                return [gr.skip()] * 4 + [last_sig]
            msg = f"⚠️ Backend unavailable at `{cfg.backend_url}`"
            return msg, msg, gr.skip(), gr.skip(), sig
        s = info["session"]
        sig = "none" if s is None else (s["session_id"], len(s["segments"]), s["active"], int(s["audio_s"]))
        if sig == last_sig:
            return [gr.skip()] * 4 + [last_sig]
        if s is None:
            return config_markdown(info), "No session yet.", gr.skip(), gr.skip(), sig
        line, summary_df, segments_df = session_tables(s)
        return config_markdown(info), line, summary_df, segments_df, sig

    def fetch_log():
        try:
            r = http.get("/api/session/log")
        except httpx.HTTPError:
            raise gr.Error(f"Backend unavailable at {cfg.backend_url}")
        if r.status_code == 404:
            raise gr.Error("No session log yet.")
        r.raise_for_status()
        name = r.headers.get("content-disposition", "").partition("filename=")[2].strip('"') or "session.jsonl"
        path = log_dir() / Path(name).name
        path.write_bytes(r.content)
        return str(path)

    with gr.Blocks(title="ASR + VAD latency demo") as demo:
        gr.Markdown("## Streaming ASR + VAD latency demo")
        config_md = gr.Markdown("Connecting to backend…")
        gr.HTML(LIVE_PANEL, js_on_load=None)  # no default click->trigger handler
        session_md = gr.Markdown("No session yet.")
        summary = gr.Dataframe(headers=SUMMARY_COLUMNS, label="Session summary (ms)", interactive=False)
        segments = gr.Dataframe(headers=SEGMENT_COLUMNS, label="Segments", interactive=False, wrap=True)
        with gr.Row():
            get_log = gr.Button("Get session log (JSONL)", size="sm")
            log_file = gr.File(label="Session log", interactive=False)
        get_log.click(fetch_log, outputs=log_file)
        last_sig = gr.State(None)
        gr.Timer(0.5).tick(refresh, inputs=last_sig, outputs=[config_md, session_md, summary, segments, last_sig])
    return demo

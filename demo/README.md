# demo: streaming ASR + VAD latency harness

Live browser mic → streaming ASR + VAD, with per-stage latency measurement. It's a self-contained
project, separate from the turn-detection model in `../src/`. Run all commands below from this
`demo/` directory.

```
browser ──HTTP + /ws/audio──▶ frontend :7860 ──REST + /ws/audio proxy──▶ backend :8000
          (page, mic)          Gradio UI, static JS,                      VAD + ASR on GPU,
                               WebSocket proxy                            metrics, JSONL logs
```

- **Backend** (`backend/`): FastAPI only, with no UI code.
  - **ASR:** [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
    via HF Transformers, behind a pluggable interface (`backend/asr/base.py`).
  - **VAD:** Silero VAD. ASR and VAD see every frame. A VAD end-of-speech (default 200 ms of silence)
    flushes the ASR, which emits that segment's final transcript, resets, and keeps streaming.
- **Frontend** (`frontend/`): Gradio page plus browser JS (mic capture with an AudioWorklet). It
  proxies the audio WebSocket to the backend and polls the backend's REST API for stats. It has no
  ML dependencies and never imports the backend.

## Layout

```
demo/
  configs/demo.yaml    shared config: backend: {asr, vad, server, runs_dir}, frontend: {host, port, backend_url}
  backend/             server.py (WebSocket + REST), pipeline.py, asr/, vad/, timing.py, eventlog.py
    requirements.txt  tests/
  frontend/            app.py (page + WebSocket proxy), ui.py (Gradio layout), config.py, static/
    requirements.txt  tests/
  runs/                session logs written by the backend (gitignored)
  pytest.ini
```

## Setup (conda env `turn`)

```bash
cd demo
conda activate turn            # or use /venv/turn/bin/python directly
pip install -r backend/requirements.txt -r frontend/requirements.txt
```

The model (~2.5 GB of GPU memory) downloads to `$HF_HOME` on first backend start.

## Run

```bash
python -m backend.server --config configs/demo.yaml    # 127.0.0.1:8000, loads the model
python -m frontend.app   --config configs/demo.yaml    # 127.0.0.1:7860
```

From your own machine, tunnel the frontend port only and browse to http://localhost:7860. Browsers
only allow mic access on `localhost` or HTTPS.

```bash
ssh -p <ssh port> -L 7860:127.0.0.1:7860 root@<instance ip>
```

Supported chunk latencies are 160, 560 and 1120 ms. Only one session can be active at a time.

## Backend API

| Endpoint | Purpose |
|---|---|
| `WS /ws/audio` | binary frames `[seq u32][client_ts_ms f64][512 × int16]` and JSON `ping` / `rtt` / `stop` in; pipeline events out |
| `GET /api/health` | `{"status": "ok"}` |
| `GET /api/session` | active or most recent session: summary, segment rows, config, ASR info |
| `GET /api/session/log` | that session's JSONL event log |

## Metrics

All latencies are measured on the backend's monotonic clock against when the audio frames arrived at
the backend, so the frontend proxy hop is not included (except in `network_one_way`).

| Metric | Definition |
|---|---|
| `vad_start_latency` | VAD start fired − arrival of the frame holding VAD's (unpadded) start sample |
| `vad_end_latency` | VAD end fired − arrival of the last speech sample (includes the min-silence wait) |
| `token_perceived` | token emitted − arrival of the audio its encoder frame needs (includes lookahead wait) |
| `token_compute` | token emitted − arrival of the last sample of the chunk that produced it |
| `vad_fire_to_final` | final transcript emitted − VAD end fired (flush cost) |
| `speech_end_to_final` | final transcript emitted − arrival of the last speech sample (user-perceived) |
| `network_one_way` | browser round-trip / 2, through the frontend proxy |

## Tests

```bash
pytest -m "not gpu"                                         # backend + frontend unit tests
TURNDEMO_TEST_WAV=/path/to/speech_16k_mono.wav pytest -m gpu   # real model + Silero, end to end
```

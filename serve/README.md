# Inference API

FastAPI server for the EoT model: audio in, P(turn complete) out. Built for low latency under concurrent
load: a few workers run small batches in parallel, and when too many requests are waiting the server answers
503 at once instead of late. CPU by default; the Docker image is CPU-only.

| File | What it does |
|---|---|
| `inference.py` | `EoTInference`: loads a model directory (`save_pretrained` output: `<run>/best`), per-request preprocessing + features, batched forward pass, warm-up |
| `batcher.py` | `MicroBatcher`: queues requests; each of `workers` takes up to `max_batch` of them, waiting at most `max_wait_ms` after the first; rejects beyond `max_queue` |
| `app.py` | FastAPI app: `POST /predict`, `GET /health` |
| `__main__.py` | `python -m serve` (flags or environment variables) |
| `Dockerfile` | CPU image (`python:3.12-slim` + CPU torch) |
| `tests/` | API and batcher tests with a tiny random model (no network) |

Preprocessing is `turn_detection.data.audio`, the same code as training (minus augmentation): resample to
16 kHz, last 8 s, 8 kHz round trip if the model was trained with it (`config.narrowband`), loudness
normalized to the model's RMS level (`config.loudness_dbfs`, -30 dBFS), zero-padded on the left to 8 s,
then the model's own `WhisperFeatureExtractor`.

## Run locally

From the repo root, in the conda env `turn` (the package is already installed there with `pip install -e`):

```bash
python scripts/save_untrained.py                       # stand-in model until one is trained
python -m serve --model artifacts/models/untrained     # http://127.0.0.1:8000, docs at /docs
```

| Flag | Env var | Default | |
|---|---|---|---|
| `--model` | `MODEL_DIR` | (required) | a `save_pretrained` directory: model + feature extractor |
| `--host`, `--port` | `HOST`, `PORT` | `127.0.0.1`, `8000` | the demo backend also defaults to 8000: pick another port to run both |
| `--device` | `DEVICE` | `cpu` | `cuda` works too |
| `--workers` | `WORKERS` | 1 | forward passes run in parallel (torch releases the GIL) |
| `--max-batch` | `MAX_BATCH` | 4 | largest batch per forward pass |
| `--max-wait-ms` | `MAX_WAIT_MS` | 2 | longest a request waits for others to join its batch |
| `--max-queue` | `MAX_QUEUE` | 0 (no limit) | requests allowed to wait for a worker; beyond that, 503 |
| `--threads` | `TORCH_THREADS` | torch's default | intra-op threads per forward pass |
| `--access-log` | `ACCESS_LOG` | off | uvicorn's per-request log line |

## API

`POST /predict`: the request body is the audio, ending at the pause to classify (only the last 8 s are
used; at most 60 s accepted).

| Content-Type | Body |
|---|---|
| `audio/wav`, `audio/flac`, ... | an audio file at any sample rate; stereo is mixed down |
| `application/octet-stream` | raw little-endian float32 mono samples; rate from `?sample_rate=` (default 16000) |

```bash
curl -X POST --data-binary @clip.wav -H "Content-Type: audio/wav" http://127.0.0.1:8000/predict
# {"probability": 0.87, "complete": true, "threshold": 0.43}
```

`complete` is `probability >= threshold`, the model's threshold chosen on val during training. Every
response carries `Server-Timing: total;dur=<ms>`, the time spent in the server. Errors: 400 (empty,
undecodable, too long, NaN samples), 415 (other content types), 422 (bad `sample_rate`), 503 with
`Retry-After: 1` (more than `max_queue` requests waiting).

`GET /health`: status, device, threshold, the batching settings, and counters: `requests`, `batches`,
`mean_batch_size`, `queued` (waiting now) and `rejected` (503s so far).

## Docker

Build from the repo root (the image needs `pyproject.toml`, `src/` and `serve/`; see
`Dockerfile.dockerignore`). The model directory is mounted, not baked in:

```bash
docker build -f serve/Dockerfile -t eot-serve .
docker run --rm -p 8000:8000 -v "$PWD/artifacts/models/untrained:/model:ro" eot-serve
```

The image installs the CPU torch wheel, the pinned `serve/requirements.txt` (with uvloop and httptools for
uvicorn), and the `turn_detection` package core only (`pip install --no-deps .`: no datasets, augmentation
or wandb). It runs as a non-root user with a `HEALTHCHECK` on `/health`. Settings go in as environment
variables (`-e WORKERS=2 -e TORCH_THREADS=4`).

The image is built for the machine that builds it: on an Apple Silicon Mac that is `linux/arm64`. For an
x86 server, build on the server, or cross-build with
`docker buildx build --platform linux/amd64 -f serve/Dockerfile -t eot-serve .` (slow: emulated).

## Benchmark

`scripts/benchmark.py` load-tests a running server (local or Docker, any host) and reports throughput and
latency per step against a target (default p95 < 100 ms). It needs only httpx, numpy and soundfile:

```bash
python scripts/benchmark.py --url http://127.0.0.1:8000                   # 10 workers x 100 calls each
python scripts/benchmark.py --workers 1,5,10 --calls 200 --json results.json
python scripts/benchmark.py --rps 5,10,20 --duration 20                   # fixed arrival rates instead
```

- `--workers`: closed loop. Each worker sends its next request as soon as the last returns, `--calls`
  times. This finds the most the server can do; latency then includes queueing behind the other workers.
- `--rps`: open loop. Requests arrive at that rate (Poisson) whatever the server is doing, as real callers
  do; latency counts from the scheduled send time, so queueing can't hide. This is the right test for
  "is p95 under 100 ms at N requests per second".

Per step: OK / 503 / errors, req/s, p50/p90/p95/p99/max latency, the server's own time (`Server-Timing`:
client minus server is the network and client overhead) and the mean batch size.

## Performance and tuning

The forward pass (the Whisper Tiny encoder over 8 s) is nearly all of the work; preprocessing is ~1.5 ms
natively, ~4 ms in Docker. On CPU a batch of B costs about B times one clip, so big batches add latency
without adding throughput; parallel workers and threads add throughput until the cores run out. Measured
on an M3 MacBook (8 cores), untrained stand-in model, 6 s WAV requests:

| Where | Forward, batch 1 | Most req/s | 1 worker x 100 calls | 10 workers x 100 calls |
|---|---|---|---|---|
| macOS, natively (torch uses Accelerate) | ~13 ms | ~70–90 | – | – |
| Docker Desktop (Linux VM, `TORCH_THREADS=4`, `OMP_WAIT_POLICY=PASSIVE`) | ~40 ms | ~30 | p50 42 ms, p95 64 ms | p50 318 ms, p95 516 ms |

Docker Desktop on a Mac is a poor stand-in for a server: its Linux VM can't use Accelerate's matrix units
and effectively gets about 3 cores. One caller at a time is well under 100 ms; 10 back-to-back callers
need ~10x the throughput. Tried and not worth it there: bf16/fp16 (slower on this CPU), oneDNN bf16 fast
math (slower), ONNX Runtime fp32 (no faster), `torch.compile` (~1 ms). int8 quantization would cut the
compute but changes the outputs, so it needs checking against a trained model first.

Tune on the machine you deploy to: start one container, run `scripts/benchmark.py --workers 1,5,10`, and try
`TORCH_THREADS` (forward speed; more threads helped up to 4 in the Docker VM), `WORKERS` (parallel passes;
total threads `WORKERS x TORCH_THREADS` should not exceed the cores), and `OMP_WAIT_POLICY=PASSIVE` (less
spinning when the CPU is shared). Set `MAX_QUEUE` to keep accepted requests fast once the server is at
capacity (callers retry or fall back on 503); scale out with more containers behind a load balancer.

## Tests

```bash
pytest serve/tests      # or `pytest` from the repo root, which runs these too
```

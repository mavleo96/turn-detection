# Inference API

FastAPI server for the EoT model: audio in, P(turn complete) out. Built for low latency under concurrent
load: a few workers run small batches in parallel, and when too many requests are waiting the server answers
503 at once instead of late. The model runs in ONNX Runtime on CPU (fp32, exported by `scripts/export_onnx.py`),
so neither the server nor its image needs torch.

| File | What it does |
|---|---|
| `inference.py` | `EoTInference`: loads a checkpoint directory (`<run>/best` with `model.onnx`), per-request preprocessing + features, batched forward pass in ONNX Runtime, warm-up |
| `batcher.py` | `MicroBatcher`: queues requests; each of `workers` takes up to `max_batch` of them, waiting at most `max_wait_ms` after the first; rejects beyond `max_queue` |
| `app.py` | FastAPI app: `POST /predict`, `GET /health` |
| `__main__.py` | `python -m serve` (flags or environment variables) |
| `Dockerfile` | CPU image (`python:3.12-slim` + ONNX Runtime, no torch) |
| `tests/` | API and batcher tests with a tiny random model, exported to ONNX (no network) |

Preprocessing is `turn_detection.data.audio`, the same code as training (minus augmentation): resample to
16 kHz, the last `config.window_s` seconds (2, 4 or 8, as the model was trained), 8 kHz round trip if the
model was trained with it (`config.narrowband`), loudness normalized to the model's RMS level
(`config.loudness_dbfs`, -30 dBFS), zero-padded on the left to the window, then the model's own
`WhisperFeatureExtractor`. Audio files are decoded with soundfile, not training's torchcodec (which would put
torch and FFmpeg in the image); for 16 kHz mono both give the same samples.

## Run locally

From the repo root, in the training env (the package installed with `pip install -e`; the export needs torch):

```bash
# a trained checkpoint (W&B model artifact <run>-best), e.g. both-8s
wandb artifact get mavleo96-team/turn-detection/both-8s-5ep-20261024-225310-best:best --root artifacts/models/both-8s
python scripts/export_onnx.py artifacts/models/both-8s   # -> artifacts/models/both-8s/model.onnx, checked vs torch
python -m serve --model artifacts/models/both-8s       # http://127.0.0.1:8000, docs at /docs
```

The export writes the model plus its sigmoid as one fp32 `model.onnx`, with a dynamic batch
dimension, and fails if ONNX Runtime and torch disagree by more than 1e-4. The
server reads everything else from the same directory: `config.json` (threshold, window, preprocessing) and the
feature extractor.

| Flag | Env var | Default | |
|---|---|---|---|
| `--model` | `MODEL_DIR` | (required) | a checkpoint directory with `model.onnx`, `config.json` and the feature extractor |
| `--host`, `--port` | `HOST`, `PORT` | `127.0.0.1`, `8000` | |
| `--workers` | `WORKERS` | 1 | forward passes run in parallel (ONNX Runtime releases the GIL) |
| `--max-batch` | `MAX_BATCH` | 4 | largest batch per forward pass |
| `--max-wait-ms` | `MAX_WAIT_MS` | 2 | longest a request waits for others to join its batch |
| `--max-queue` | `MAX_QUEUE` | 0 (no limit) | requests allowed to wait for a worker; beyond that, 503 |
| `--threads` | `THREADS` | 0: one per core | ONNX Runtime's intra-op threads per forward pass |
| `--access-log` | `ACCESS_LOG` | off | uvicorn's per-request log line |

## API

`POST /predict`: the request body is the audio, ending at the pause to classify (only the model's last
`window_s` seconds are used; at most 60 s accepted).

| Content-Type | Body |
|---|---|
| `audio/wav`, `audio/flac`, ... | an audio file soundfile can read, at any sample rate; stereo is mixed down (channel mean) |
| `application/octet-stream` | raw little-endian float32 mono samples; rate from `?sample_rate=` (default 16000) |

```bash
curl -X POST --data-binary @clip.wav -H "Content-Type: audio/wav" http://127.0.0.1:8000/predict
# {"probability": 0.87, "complete": true, "threshold": 0.43}
```

`complete` is `probability >= threshold`, the model's threshold chosen on val during training. Every
response carries `Server-Timing: total;dur=<ms>`, the time spent in the server. Errors: 400 (empty,
undecodable, too long, NaN samples), 415 (other content types), 422 (bad `sample_rate`), 503 with
`Retry-After: 1` (more than `max_queue` requests waiting).

`GET /health`: status, threads, threshold, the batching settings, and counters: `requests`, `batches`,
`mean_batch_size`, `queued` (waiting now) and `rejected` (503s so far).

## Docker

Build from the repo root (the image needs `pyproject.toml`, `src/` and `serve/`; see
`Dockerfile.dockerignore`). The model directory is mounted, not baked in:

```bash
docker build -f serve/Dockerfile -t eot-serve .
docker run --rm -p 8000:8000 -v "$PWD/artifacts/models/both-8s:/model:ro" eot-serve
```

Export the checkpoint first (`model.onnx` in the mounted directory), and make its files readable by the image's
user (uid 1000): `chmod a+r artifacts/models/both-8s/*`. The image (~670 MB) installs the pinned
`serve/requirements.txt` (ONNX Runtime, with uvloop and httptools for uvicorn) and the `turn_detection` package
without dependencies (`pip install --no-deps .`: serve imports only `turn_detection.data`, numpy and soxr; no
torch, datasets, augmentation or wandb). It sets `OPENBLAS_NUM_THREADS=1`: numpy's OpenBLAS threads otherwise
busy-wait after the feature extractor's matmul and take the cores ONNX Runtime needs (87 ms instead of 42 ms per
request with 2 threads on 2 cores). It runs as a non-root user with a `HEALTHCHECK` on `/health`. Settings go
in as environment variables (`-e WORKERS=2 -e THREADS=1`).

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
natively, ~4–6 ms in Docker. On CPU a batch of B costs about B times one clip, so big batches add latency
without adding throughput; parallel workers and threads add throughput until the cores run out.

Measured in Docker on a 2-vCPU x86 Linux VM (AMD EPYC, AVX2 only), ONNX Runtime fp32, both-8s, 6 s WAV
requests, 50 calls per worker (the benchmark client shares the 2 cores):

| `THREADS` x `WORKERS`, `MAX_BATCH` | 1 worker: req/s, p50, p95 | 10 workers: req/s, p50, p95 |
|---|---|---|
| 2 x 1, 1 | 22.1, 43 ms, 53 ms | 22.4, 443 ms, 484 ms |
| 2 x 1, 4 | 19.7, 48 ms, 66 ms | 23.9, 376 ms, 527 ms |
| 1 x 2, 1 | 14.4, 69 ms, 75 ms | 26.7, 370 ms, 405 ms |
| 1 x 2, 4 | 14.1, 71 ms, 75 ms | 27.1, 370 ms, 453 ms |

On the same VM the torch image (untrained model, same compute) did 2 x 1: 19.1 req/s at p95 76 ms for one
caller, and at most 23.6 req/s (1 x 2, batch 1); without `OPENBLAS_NUM_THREADS=1`, which may have slowed it too.
Two threads per pass give the lowest latency for one caller; one thread per worker the most throughput.

Earlier, torch on an M3 MacBook (8 cores), untrained stand-in model:

| Where | Forward, batch 1 | Most req/s | 1 worker x 100 calls | 10 workers x 100 calls |
|---|---|---|---|---|
| macOS, natively (torch uses Accelerate) | ~13 ms | ~70–90 | – | – |
| Docker Desktop (Linux VM, `TORCH_THREADS=4`, `OMP_WAIT_POLICY=PASSIVE`) | ~40 ms | ~30 | p50 42 ms, p95 64 ms | p50 318 ms, p95 516 ms |

Docker Desktop on a Mac is a poor stand-in for a server: its Linux VM can't use Accelerate's matrix units
and effectively gets about 3 cores. One caller at a time is well under 100 ms; 10 back-to-back callers
need ~10x the throughput. Tried and not worth it there: bf16/fp16 (slower on this CPU), oneDNN bf16 fast
math (slower), ONNX Runtime fp32 (no faster on the Mac; ~15% faster than torch on x86 Linux), `torch.compile`
(~1 ms). int8 (ONNX Runtime dynamic quantization: ~1.5x faster than fp32 on the Linux VM) changes the outputs
and is not used.

Tune on the machine you deploy to: start one container, run `scripts/benchmark.py --workers 1,5,10`, and try
`THREADS` (forward speed) and `WORKERS` (parallel passes; total threads `WORKERS x THREADS` should not exceed
the cores). Set `MAX_QUEUE` to keep accepted requests fast once the server is at
capacity (callers retry or fall back on 503); scale out with more containers behind a load balancer.

## Tests

In the training env (the fixtures export a tiny model with `scripts/export_onnx.py`, which needs torch):

```bash
pytest serve/tests      # or `pytest` from the repo root, which runs these too
```

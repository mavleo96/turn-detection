# Inference API

FastAPI server for the EoT model: audio in, P(turn complete) out. Concurrent requests share one forward pass
through dynamic micro-batching. CPU by default; the Docker image is CPU-only.

| File | What it does |
|---|---|
| `inference.py` | `EoTInference`: loads a model directory (`save_pretrained` output: `<run>/best`), per-request preprocessing + features, batched forward pass |
| `batcher.py` | `MicroBatcher`: queues requests, runs up to `max_batch` together, waiting at most `max_wait_ms` after the first |
| `app.py` | FastAPI app: `POST /predict`, `GET /health` |
| `__main__.py` | `python -m serve` (flags or environment variables) |
| `Dockerfile` | CPU image (`python:3.12-slim` + CPU torch) |
| `tests/` | API and batcher tests with a tiny random model (no network) |

Preprocessing is `turn_detection.data.audio`, the same code as training (minus augmentation): resample to
16 kHz, last 8 s, 8 kHz round trip if the model was trained with it (`config.narrowband`), loudness
normalized to the model's RMS level (`config.loudness_dbfs`, -30 dBFS), zero-pad on the side it was trained
with (`config.padding`), then the model's own `WhisperFeatureExtractor`.

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
| `--max-batch` | `MAX_BATCH` | 32 | largest batch per forward pass |
| `--max-wait-ms` | `MAX_WAIT_MS` | 5 | longest a request waits for others to join its batch |
| `--threads` | `TORCH_THREADS` | torch's default | intra-op threads for the forward pass |

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

`complete` is `probability >= threshold`, the model's threshold chosen on val during training. Errors: 400
(empty, undecodable, too long, NaN samples), 415 (other content types), 422 (bad `sample_rate`).

`GET /health`: status, device, threshold, batching settings, and counters (`requests`, `batches`,
`mean_batch_size`) that show how much batching happens under load.

## Docker

Build from the repo root (the image needs `pyproject.toml`, `src/` and `serve/`; see
`Dockerfile.dockerignore`). The model directory is mounted, not baked in:

```bash
docker build -f serve/Dockerfile -t eot-serve .
docker run --rm -p 8000:8000 -v "$PWD/artifacts/models/untrained:/model:ro" eot-serve
```

The image installs the CPU torch wheel, the pinned `serve/requirements.txt`, and the `turn_detection`
package core only (`pip install --no-deps .`: no datasets, augmentation or wandb). It runs as a non-root
user with a `HEALTHCHECK` on `/health`. Settings go in as environment variables (`-e MAX_BATCH=16`).

## Tests

```bash
pytest serve/tests      # or `pytest` from the repo root, which runs these too
```

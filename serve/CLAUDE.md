# Notes for Claude: serve/

Context for working on the inference API. User-facing docs are in `README.md`; this is what isn't obvious
from the code or the README.

## Goal and status (2026-10-24)

- Goal: dockerized FastAPI endpoint, p95 < 100 ms, stress-tested under concurrent calls.
- Done: workers + small batches + `MAX_QUEUE` 503 shedding, warm-up, `Server-Timing`, uvloop/httptools,
  `scripts/benchmark.py` (`--workers N --calls M`, or `--rps`), `serve/tests` (15, pass in <1 s).
- Open: the Dockerfile has no tuned `TORCH_THREADS`/`WORKERS` defaults (deliberately: tune on the target
  machine); never benchmarked on x86 Linux; no trained model yet (only `artifacts/models/untrained`,
  from `python scripts/save_untrained.py`).

## Environment gotchas (the user's M3 Mac)

- **Port 8000 is taken by Cursor** on 127.0.0.1/::1 (a port forward). Docker `-p 8000:8000` still binds,
  but requests to 127.0.0.1:8000 reach Cursor ("Server disconnected without sending a response"). Map the
  container to another host port, e.g. `-p 18000:8000`, and pass `--url http://127.0.0.1:18000`.
- Docker Desktop must be running (`open -a Docker`); check with `docker info`.
- `ruff` isn't on PATH: lint with `pre-commit run --files <paths>` (ruff pinned in `.pre-commit-config.yaml`).
- `serve` isn't an installed package: from the repo root `python -m serve` works, but an ad-hoc script that
  imports `serve.*` needs `PYTHONPATH=.`. Don't name scratch scripts `profile.py` (shadows the stdlib
  module that torch imports).
- The conda env `turn` has the package installed editable (`pip install --no-deps -e .`) plus fastapi,
  uvicorn, uvloop, httptools, onnxruntime/onnx/onnxscript (installed while experimenting; not required).

## Performance facts (measured; untrained model, 6 s WAV requests)

- Forward pass (Whisper Tiny encoder, 400 positions) is ~all of the cost; ~7 GFLOP per clip, mostly
  `addmm`. Preprocessing ~1.4 ms natively, ~4 ms in Docker.
- CPU batching doesn't amortize: batch B costs ~B x batch 1 (superlinear from B=16). That's why
  `max_batch` went from 32 to 4; the old default meant ~1.2 s batches under load.
- macOS native: batch 1 ~13 ms (Accelerate/AMX), ~70–90 req/s ceiling regardless of replicas/threads.
- Docker Desktop VM: batch 1 ~40 ms, ~20–30 req/s ceiling. The VM can't use AMX and gets ~3 effective
  cores (8 parallel busy loops scale only ~2.8x). Best measured env: `TORCH_THREADS=4
  OMP_WAIT_POLICY=PASSIVE` (unloaded p50 ~40 ms, p95 ~62 ms). 10 workers x 100 calls: p95 ~516 ms.
- `TORCH_THREADS=1` alone isn't single-threaded: BLAS still threads. With `OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1` a pass takes ~80 ms.
- Multiple worker threads in one process: 25 -> ~33–37 clips/s in Docker, then flat; separate processes
  didn't scale better.
- Tried and rejected (don't retry without a new reason):
  - fp16 or bf16 weights on CPU: ~10x slower.
  - oneDNN bf16 fast math (`ONEDNN_DEFAULT_FPMATH_MODE=BF16`): slower, ~1% hidden-state drift.
  - ONNX Runtime fp32: 3–4x slower on macOS, no faster in Docker. The TorchScript exporter fails;
    `dynamo=True` works.
  - `torch.compile`: ~1 ms gain.
  - torch dynamic int8: the API is deprecated and there's no qengine on macOS.
  - ORT `quantize_dynamic`: fails in shape inference on the dynamo export.
- int8 is the remaining big compute lever, but it changes outputs: validate with `scripts/run_eval.py` on
  a trained model before using it.

## Benchmarking

- Benchmark the container, not the host process. The host numbers are ~3x better and misleading for
  deployment.
- Pattern: `docker run -d --name eot-bench -p 18000:8000 -e TORCH_THREADS=4 -e OMP_WAIT_POLICY=PASSIVE
  -v "$PWD/artifacts/models/untrained:/model:ro" eot-serve`, wait on `/health`, then
  `python scripts/benchmark.py --url http://127.0.0.1:18000 --workers 1,10 --calls 100`; `docker rm -f` after.
- The client shares the laptop's CPU with the Docker VM, so heavy steps are pessimistic. Client p50 minus
  server p50 (`srv50`) has been ~1.5–2 ms: the Python client is accurate at this scale.
- Explain closed loop (`--workers`: max throughput, latency includes queueing) vs open loop (`--rps`: the
  latency SLO at a given arrival rate) when reporting. The user thinks in "N workers x M calls".

## Repo conventions the user expects

- Commit messages: a subject line followed directly by `- ` bullets (no blank line), then the Co-Authored-By
  trailer. Work on branch `serve`; push only when asked.
- `.gitignore` ignores only the root `/tests/` (kept local by the user); `serve/tests/` is tracked.
- Comments and docstrings are dense, explain why, and match the existing style in `src/turn_detection`.

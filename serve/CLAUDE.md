# Notes for Claude: serve/

Context for working on the inference API. User-facing docs are in `README.md`; this is what isn't obvious
from the code or the README.

## Goal and status (2026-10-25)

- Goal: dockerized FastAPI endpoint, p95 < 100 ms, stress-tested under concurrent calls.
- Done: workers + small batches + `MAX_QUEUE` 503 shedding, warm-up, `Server-Timing`, uvloop/httptools,
  `scripts/benchmark.py` (closed loop, 1/2/4/10 workers x 100 calls), `serve/tests` (17, ~4 s; the fixtures export
  a tiny model, so they run in the training env, not with serve/requirements.txt alone).
- Main's refactor merged (2026-10-25): windows of 2 / 4 / 8 s (`config.window_s`, read per checkpoint);
  serve decodes with soundfile itself (main removed `audio.decode`: decoding is the caller's; training uses
  torchcodec, kept out of the image).
- Trained checkpoints exist (runs `both-{2,4,8}s`, `ttd-8s`, `smart-turn-8s` of 2026-10-24; test metrics in
  `results/eval/`), stored as W&B model artifacts `<run>-best`, not in git.
- ONNX Runtime only (2026-10-25): `scripts/export_onnx.py` writes `model.onnx` (fp32, sigmoid included) into
  the checkpoint dir; the server reads `config.json` as JSON (importing `turn_detection.model` imports torch);
  the image has no torch (~670 MB, was 1.74 GB). The user chose fp32 only: no int8, no fp16.
- Open: no tuned `THREADS`/`WORKERS` defaults in the Dockerfile (tune on the target machine).

## Environment gotchas

- On the 2-vCPU Linux box (AVX2 only, no GPU): the env is the repo's `.venv` (CPU torch and torchcodec, both from
  the PyTorch CPU index: PyPI's torchcodec is a CUDA build), FFmpeg from apt. Docker runs natively (no VM).
  Measured there in Docker, both-8s: torch ~52 ms batch 1, ceiling ~24 req/s; ONNX Runtime fp32 ~42 ms
  (2 threads), ceiling ~27 req/s (1 thread x 2 workers). Table in README.
- `OPENBLAS_NUM_THREADS=1` matters (set in the Dockerfile): numpy's OpenBLAS threads busy-wait after the
  feature extractor's matmul and starved ONNX Runtime (2 threads: 87 ms -> 42 ms per request). ORT's own
  spinning (`session.intra_op.allow_spinning`) made no difference once that was set.
- Export gotchas: `torch.onnx.export(..., external_data=False)` for one file (the default writes `.data`);
  int8 `quantize_dynamic` (not used) also needed `graph.value_info` cleared (stale shapes).

The user's M3 Mac:

- **Port 8000 is taken by Cursor** on 127.0.0.1/::1 (a port forward). Docker `-p 8000:8000` still binds,
  but requests to 127.0.0.1:8000 reach Cursor ("Server disconnected without sending a response"). Map the
  container to another host port, e.g. `-p 18000:8000`, and pass `--url http://127.0.0.1:18000`.
- Docker Desktop must be running (`open -a Docker`); check with `docker info`.
- `ruff` isn't on PATH: lint with `pre-commit run --files <paths>` (ruff pinned in `.pre-commit-config.yaml`).
- `serve` isn't an installed package: from the repo root `python -m serve` works, but an ad-hoc script that
  imports `serve.*` needs `PYTHONPATH=.`. Don't name scratch scripts `profile.py` (shadows the stdlib
  module that torch imports).
- The conda env `turn` has the package installed editable (`pip install --no-deps -e .`) plus fastapi,
  uvicorn, uvloop, httptools; onnxruntime/onnx/onnxscript are now needed (pinned in requirements.txt) for
  scripts/export_onnx.py and the serve tests.

## Performance facts (measured with torch, mostly on the M3 Mac; untrained model, 6 s WAV requests)

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
  - ONNX Runtime fp32 on the Mac: 3–4x slower on macOS, no faster in Docker Desktop (on x86 Linux it is
    ~15% faster than torch, hence serving with it). The TorchScript exporter fails; `dynamo=True` works.
  - `torch.compile`: ~1 ms gain.
  - torch dynamic int8: the API is deprecated and there's no qengine on macOS.
  - ORT `quantize_dynamic`: works on x86 once `graph.value_info` is cleared (~1.5x faster than fp32), but
    the user chose fp32 only.

## Benchmarking

- Benchmark the container, not the host process. The host numbers are ~3x better and misleading for
  deployment.
- Pattern: `docker run -d --name eot-bench -p 18000:8000 -e THREADS=2
  -v "$PWD/artifacts/models/both-8s:/model:ro" eot-serve`, wait on `/health`, then
  `python scripts/benchmark.py --model both-8s --url http://127.0.0.1:18000`; `docker rm -f` after. Results land
  in `results/benchmark/<model>-t<threads>-w<workers>-b<max_batch>/` (tracked).
- The client shares the laptop's CPU with the Docker VM, so heavy steps are pessimistic. Client p50 minus
  server p50 (`srv50`) has been ~1.5–2 ms: the Python client is accurate at this scale.
- The benchmark is closed loop only, fixed at 1/2/4/10 workers x 100 calls, no pass/fail (the user's choice: open
  loop, `--rps`, `--duration`, `--workers`/`--calls`, `--slo-ms` removed). `--model` is required: the server can't
  tell which checkpoint is mounted at /model. When reporting, note that with many workers latency includes
  queueing (saturated server).

## Repo conventions the user expects

- Commit messages: a subject line followed directly by `- ` bullets (no blank line), then the Co-Authored-By
  trailer. `serve/` is on `main` since the merge (2026-10-25; the `serve` branch is no longer used); commit and
  push only when asked.
- Both `tests/` (root) and `serve/tests/` are tracked, and the root `pytest.ini` runs both.
- Comments and docstrings are dense, explain why, and match the existing style in `src/turn_detection`.

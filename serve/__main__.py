"""Run the inference API (from the repo root):

    python -m serve --model artifacts/runs/<run>/best           # http://127.0.0.1:8000
    python -m serve --model artifacts/models/both-8s --workers 2 --threads 2 --max-queue 8

Every flag also reads an environment variable (used by the Dockerfile): MODEL_DIR, HOST, PORT,
MAX_BATCH, MAX_WAIT_MS, WORKERS, MAX_QUEUE, THREADS, ACCESS_LOG.
"""

import argparse
import os

import uvicorn

from .app import create_app
from .inference import EoTInference


def parse_args() -> argparse.Namespace:
    env = os.environ.get
    ap = argparse.ArgumentParser(description="EoT inference API")
    ap.add_argument("--model", default=env("MODEL_DIR"), help="a checkpoint directory with model.onnx (export_onnx.py)")
    ap.add_argument("--host", default=env("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("PORT", "8000")))
    ap.add_argument("--max-batch", type=int, default=int(env("MAX_BATCH", "4")))
    ap.add_argument("--max-wait-ms", type=float, default=float(env("MAX_WAIT_MS", "2")))
    ap.add_argument("--workers", type=int, default=int(env("WORKERS", "1")), help="forward passes run in parallel")
    ap.add_argument("--max-queue", type=int, default=int(env("MAX_QUEUE", "0")), help="503 beyond this (0: no limit)")
    ap.add_argument("--threads", type=int, default=int(env("THREADS", "0")), help="ONNX Runtime threads (0: per core)")
    ap.add_argument("--access-log", action="store_true", default=env("ACCESS_LOG", "") not in ("", "0", "false"))
    args = ap.parse_args()
    if not args.model:
        ap.error("--model (or MODEL_DIR) is required")
    return args


def main() -> None:
    args = parse_args()
    inference = EoTInference(args.model, threads=args.threads)
    app = create_app(
        inference,
        max_batch=args.max_batch,
        max_wait_ms=args.max_wait_ms,
        workers=args.workers,
        max_queue=args.max_queue,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=args.access_log)


if __name__ == "__main__":
    main()

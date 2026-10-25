"""Benchmark the inference API (serve/, e.g. its Docker image): throughput and latency under concurrent load.
Needs only httpx, numpy and soundfile, so it runs from any machine.

Closed loop: 1, 2, 4, then 10 workers, each sending its next request as soon as the last returns, 100 times.
With many workers the server is saturated: throughput is its capacity, and latency includes queueing behind
the other workers.

Run (from the repo root, with the server up, e.g. the Docker image on :18000; --model names the run, since the
server can't tell which checkpoint is mounted):
  python scripts/benchmark.py --model both-8s --url http://127.0.0.1:18000

Writes <out>/<model>-t<threads>-w<workers>-b<max_batch>/ (default out: results/benchmark/, tracked; the server
settings come from /health): results.md (the table, as printed) and results.json (server settings, arguments,
every step). Per step: OK / 503 (shed by the server) / other errors, achieved throughput, client latency
percentiles of the OK requests, the server's own time (Server-Timing), and the mean batch size (from /health).
"""

import argparse
import asyncio
import io
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
WORKERS = (1, 2, 4, 10)  # one step each
CALLS = 100  # per worker per step
OUT = Path(__file__).resolve().parents[1] / "results" / "benchmark"


def synthetic_clip(seconds: float, seed: int = 0) -> np.ndarray:
    """Speech-like test audio: noise with a syllable-rate envelope, trailing 0.5 s of silence (the pause)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    envelope = np.abs(np.sin(np.arange(n) * 2 * np.pi * 4 / SAMPLE_RATE))  # ~4 syllables/s
    x = rng.standard_normal(n) * envelope * 0.1
    x[-SAMPLE_RATE // 2 :] = 0
    return x.astype(np.float32)


def payload(args: argparse.Namespace) -> tuple[bytes, str]:
    """The request body and its Content-Type."""
    if args.audio:
        with open(args.audio, "rb") as f:
            return f.read(), "audio/wav" if args.audio.endswith(".wav") else "audio/flac"

    x = synthetic_clip(args.seconds)
    if args.raw:
        return x.astype("<f4").tobytes(), "application/octet-stream"
    buf = io.BytesIO()
    sf.write(buf, x, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


@dataclass
class Step:
    workers: int
    sent: int = 0
    ok: int = 0
    shed: int = 0  # 503
    errors: int = 0  # anything else: other statuses, timeouts, connection errors
    latencies_ms: list[float] = field(default_factory=list)  # OK requests, client side
    server_ms: list[float] = field(default_factory=list)  # OK requests, Server-Timing
    elapsed_s: float = 0.0
    mean_batch: float | None = None

    def summary(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([np.nan])
        srv = np.array(self.server_ms) if self.server_ms else np.array([np.nan])
        return {
            "workers": self.workers,
            "sent": self.sent,
            "ok": self.ok,
            "shed_503": self.shed,
            "errors": self.errors,
            "throughput_rps": self.ok / self.elapsed_s if self.elapsed_s else 0.0,
            "p50_ms": float(np.percentile(lat, 50)),
            "p90_ms": float(np.percentile(lat, 90)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "max_ms": float(np.max(lat)),
            "server_p50_ms": float(np.percentile(srv, 50)),
            "server_p95_ms": float(np.percentile(srv, 95)),
            "mean_batch": self.mean_batch,
        }


async def one_request(client: httpx.AsyncClient, url: str, body: bytes, headers: dict, step: Step):
    step.sent += 1
    start = time.perf_counter()
    try:
        r = await client.post(url, content=body, headers=headers)
    except httpx.HTTPError:
        step.errors += 1
        return
    latency = (time.perf_counter() - start) * 1000

    if r.status_code == 200:
        step.ok += 1
        step.latencies_ms.append(latency)
        timing = r.headers.get("server-timing", "")
        if "dur=" in timing:
            step.server_ms.append(float(timing.split("dur=")[1].split(",")[0]))
    elif r.status_code == 503:
        step.shed += 1
    else:
        step.errors += 1


async def closed_loop(client, url, body, headers, workers: int) -> Step:
    """`workers` workers, each making CALLS back-to-back calls."""
    step = Step(workers=workers)
    t0 = time.perf_counter()

    async def caller():
        for _ in range(CALLS):
            await one_request(client, url, body, headers, step)

    await asyncio.gather(*(caller() for _ in range(workers)))
    step.elapsed_s = time.perf_counter() - t0
    return step


async def health(client: httpx.AsyncClient, base: str) -> dict:
    return (await client.get(f"{base}/health")).json()


def row(s: dict) -> str:
    """One step as a markdown table row."""
    latencies = [f"{s[k]:.1f}" for k in ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "server_p50_ms")]
    batch = f"{s['mean_batch']:.2f}" if s["mean_batch"] else "-"
    values = [s["workers"], s["ok"], s["shed_503"], s["errors"], f"{s['throughput_rps']:.1f}", *latencies, batch]
    return "| " + " | ".join(map(str, values)) + " |"


async def run(args: argparse.Namespace) -> list[dict]:
    body, content_type = payload(args)
    headers = {"Content-Type": content_type}
    base = args.url.rstrip("/")
    url = f"{base}/predict"
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        info = await health(client, base)
        run_dir = args.out / f"{args.model}-t{info['threads']}-w{info['workers']}-b{info['max_batch']}"
        local = urlparse(base).hostname in ("127.0.0.1", "localhost", "::1")
        limit = f", limited to {info['cpu_limit']:g} cores" if info["cpu_limit"] else ""
        audio = args.audio or f"synthetic {args.seconds:g} s clip"
        lines = [
            f"# Benchmark: {args.model}",
            "",
            f"Run {datetime.now():%Y-%m-%d %H:%M}.",
            "",
            "## Setup",
            "",
            f"- Server: `serve/` at {base}, ONNX Runtime fp32 on CPU: {info['cpu']}, {info['cores']} cores usable"
            f"{limit}.",
            f"- Model: {args.model}, window {info['window_s']} s, threshold {info['threshold']:.4f}.",
            f"- Settings: threads {info['threads']} (0: one per core), workers {info['workers']}, max_batch "
            f"{info['max_batch']}, max_wait_ms {info['max_wait_ms']:g}, max_queue {info['max_queue']} (0: no limit).",
            "- Client: this script (Python, httpx), "
            + ("on the same machine, sharing its CPU with the server." if local else "on another machine."),
            f"- Requests: {audio}, {len(body) / 1024:.0f} KiB {content_type}; closed loop: each worker sends its next "
            f"request as soon as the last returns, {CALLS} calls per worker, steps of {', '.join(map(str, WORKERS))} "
            f"workers; {args.warmup} warm-up requests first, {args.pause:g} s pause between steps.",
            "",
            "## How to read it",
            "",
            "- Latency (ms) is client side: HTTP, decoding, preprocessing, queueing and the model; server p50 is the "
            "time inside the server (`Server-Timing`), so the difference is network and client overhead.",
            "- Once workers exceed what the server runs in parallel, it is saturated: req/s is its capacity and "
            "latency is mostly queueing (about workers / req/s), not slower requests.",
            "- On CPU a batch of B costs about B single passes, so batching (batch = mean batch size) adds little "
            "throughput.",
            "- The content of the audio doesn't change the cost: every request is cut or padded to the window.",
        ]
        if local:
            lines.append("- Client and server share the CPU: heavy steps are pessimistic for the server alone.")
        lines += [
            "",
            "## Results",
            "",
            "| workers | ok | 503 | err | req/s | p50 | p90 | p95 | p99 | max | server p50 | batch |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        print("\n".join(lines), flush=True)

        # Warm-up: a few sequential requests (connections, server caches)
        for _ in range(args.warmup):
            await client.post(url, content=body, headers=headers)

        results = []
        for workers in WORKERS:
            before = await health(client, base)
            step = await closed_loop(client, url, body, headers, workers)
            after = await health(client, base)
            batches = after["batches"] - before["batches"]
            step.mean_batch = (after["requests"] - before["requests"]) / batches if batches else None

            s = step.summary()
            results.append(s)
            lines.append(row(s))
            print(lines[-1], flush=True)
            await asyncio.sleep(args.pause)  # let the server drain between steps

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.md").write_text("\n".join(lines) + "\n")
        with open(run_dir / "results.json", "w") as f:
            json.dump({"server": info, "args": vars(args), "steps": results}, f, indent=2, default=str)  # Path -> str
        print(f"\nwrote {run_dir}")
        return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark the EoT inference API under concurrent load")
    ap.add_argument("--model", required=True, help="the served checkpoint's name, for the run directory (e.g. both-8s)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, default=OUT, help="results go to <out>/<model>-t<threads>-w<workers>-b<batch>/")
    ap.add_argument("--audio", help="a WAV/FLAC file to send (default: a synthetic clip)")
    ap.add_argument("--seconds", type=float, default=6, help="length of the synthetic clip")
    ap.add_argument("--raw", action="store_true", help="send raw float32 (application/octet-stream) instead of WAV")
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pause", type=float, default=2, help="seconds between steps")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

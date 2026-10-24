"""Benchmark the inference API (serve/, e.g. its Docker image): throughput and latency under concurrent load,
against a latency target. Needs only httpx, numpy and soundfile, so it runs from any machine.

Two load shapes, each run as a sweep of steps:
  --rps 5,10,20     open loop: requests arrive at that rate (Poisson), whether or not earlier ones are done,
                    as real callers do. Latency counts from the scheduled send time, so a client that falls
                    behind can't hide queueing (no coordinated omission).
  --workers 1,10    closed loop: that many workers, each sending its next request as soon as one returns,
                    --calls times each (default 100); or for --duration seconds with --calls 0.

Run (from the repo root, with the server up, e.g. the Docker image on :8000):
  python scripts/benchmark.py                                   # 10 workers x 100 calls each
  python scripts/benchmark.py --workers 1,5,10 --calls 200 --json results.json
  python scripts/benchmark.py --rps 5,10,20 --duration 20       # fixed arrival rates instead
  python scripts/benchmark.py --url http://host:8000 --audio clip.wav

Per step: requests sent, OK / 503 (shed by the server) / other errors, achieved throughput, client latency
percentiles of the OK requests, the server's own time (Server-Timing), and the mean batch size (from
/health). A step passes when its p95 (--percentile) is under --slo-ms and nothing but 503s failed; the
summary names the highest passing load.
"""

import argparse
import asyncio
import io
import json
import random
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


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
    load: str  # "rps=20" or "workers=8"
    sent: int = 0
    ok: int = 0
    shed: int = 0  # 503
    errors: int = 0  # anything else: other statuses, timeouts, connection errors
    latencies_ms: list[float] = field(default_factory=list)  # OK requests, client side
    server_ms: list[float] = field(default_factory=list)  # OK requests, Server-Timing
    elapsed_s: float = 0.0
    mean_batch: float | None = None
    late_sends: int = 0  # open loop: sends that started > 5 ms after schedule (client too slow)

    def summary(self, percentile: float, slo_ms: float) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([np.nan])
        srv = np.array(self.server_ms) if self.server_ms else np.array([np.nan])
        p = np.percentile(lat, percentile)
        return {
            "load": self.load,
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
            "late_sends": self.late_sends,
            "pass": bool(self.ok and p < slo_ms and self.errors == 0),
        }


async def one_request(client: httpx.AsyncClient, url: str, body: bytes, headers: dict, start: float, step: Step):
    step.sent += 1
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


async def open_loop(client, url, body, headers, rps: float, duration: float, seed: int) -> Step:
    step = Step(load=f"rps={rps:g}")
    rng = random.Random(seed)
    tasks = []
    t0 = time.perf_counter()
    scheduled = t0
    while True:
        scheduled += rng.expovariate(rps)  # Poisson arrivals
        if scheduled - t0 >= duration:
            break
        delay = scheduled - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.005:
            step.late_sends += 1
        tasks.append(asyncio.create_task(one_request(client, url, body, headers, scheduled, step)))
    await asyncio.gather(*tasks)
    step.elapsed_s = time.perf_counter() - t0
    return step


async def closed_loop(client, url, body, headers, workers: int, calls: int, duration: float) -> Step:
    """`workers` workers, each calling back to back: `calls` calls each, or until `duration` is up (calls 0)."""
    step = Step(load=f"workers={workers}")
    t0 = time.perf_counter()

    async def caller():
        done = 0
        while (done < calls) if calls else (time.perf_counter() - t0 < duration):
            await one_request(client, url, body, headers, time.perf_counter(), step)
            done += 1

    await asyncio.gather(*(caller() for _ in range(workers)))
    step.elapsed_s = time.perf_counter() - t0
    return step


async def health(client: httpx.AsyncClient, base: str) -> dict:
    return (await client.get(f"{base}/health")).json()


async def run(args: argparse.Namespace) -> list[dict]:
    body, content_type = payload(args)
    headers = {"Content-Type": content_type}
    base = args.url.rstrip("/")
    url = f"{base}/predict"
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        info = await health(client, base)
        print(f"server: {json.dumps(info)}")
        length = f"{args.calls} calls per worker" if args.calls else f"{args.duration:g} s"
        print(f"payload: {len(body) / 1024:.0f} KiB {content_type}; worker steps: {length}\n")

        # Warm-up: a few sequential requests (connections, server caches)
        for _ in range(args.warmup):
            await client.post(url, content=body, headers=headers)

        loads = [("rps", float(v)) for v in args.rps.split(",") if v] if args.rps else []
        loads += [("workers", int(v)) for v in args.workers.split(",") if v] if args.workers else []

        results = []
        header = (
            f"{'load':>16} {'ok':>6} {'503':>5} {'err':>4} {'req/s':>7} {'p50':>7} {'p90':>7} {'p95':>7} "
            f"{'p99':>7} {'max':>7} {'srv50':>6} {'batch':>5}  pass(p{args.percentile:g}<{args.slo_ms:g}ms)"
        )
        print(header)
        for i, (kind, value) in enumerate(loads):
            before = await health(client, base)
            if kind == "rps":
                step = await open_loop(client, url, body, headers, value, args.duration, seed=i)
            else:
                step = await closed_loop(client, url, body, headers, value, args.calls, args.duration)
            after = await health(client, base)
            batches = after["batches"] - before["batches"]
            step.mean_batch = (after["requests"] - before["requests"]) / batches if batches else None

            s = step.summary(args.percentile, args.slo_ms)
            results.append(s)
            batch = f"{s['mean_batch']:.2f}" if s["mean_batch"] else "-"
            print(
                f"{s['load']:>16} {s['ok']:>6} {s['shed_503']:>5} {s['errors']:>4} {s['throughput_rps']:>7.1f} "
                f"{s['p50_ms']:>7.1f} {s['p90_ms']:>7.1f} {s['p95_ms']:>7.1f} {s['p99_ms']:>7.1f} "
                f"{s['max_ms']:>7.1f} {s['server_p50_ms']:>6.1f} {batch:>5}  {'PASS' if s['pass'] else 'FAIL'}"
                + (f"  ({s['late_sends']} late sends: client-limited)" if s["late_sends"] > s["sent"] * 0.05 else ""),
                flush=True,
            )
            await asyncio.sleep(args.pause)  # let the server drain between steps

        passing = [s for s in results if s["pass"]]
        if passing:
            best = max(passing, key=lambda s: s["throughput_rps"])
            print(
                f"\nhighest passing load: {best['load']} -> {best['throughput_rps']:.1f} req/s at "
                f"p{args.percentile:g} {best[f'p{args.percentile:g}_ms']:.1f} ms"
            )
        else:
            print("\nno step met the target")

        if args.json:
            with open(args.json, "w") as f:
                json.dump({"server": info, "args": vars(args), "steps": results}, f, indent=2)
        return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark the EoT inference API under concurrent load")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rps", default="", help="open-loop request rates, comma-separated (e.g. 5,10,20)")
    ap.add_argument("--workers", default="", help="closed-loop worker counts, comma-separated (e.g. 1,5,10)")
    ap.add_argument("--calls", type=int, default=100, help="calls per worker per step (0: run for --duration)")
    ap.add_argument(
        "--duration", type=float, default=15, help="seconds per --rps step (and worker step with --calls 0)"
    )
    ap.add_argument("--audio", help="a WAV/FLAC file to send (default: a synthetic clip)")
    ap.add_argument("--seconds", type=float, default=6, help="length of the synthetic clip")
    ap.add_argument("--raw", action="store_true", help="send raw float32 (application/octet-stream) instead of WAV")
    ap.add_argument("--slo-ms", type=float, default=100)
    ap.add_argument("--percentile", type=float, default=95, choices=[50, 90, 95, 99])
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pause", type=float, default=2, help="seconds between steps")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    if not args.rps and not args.workers:
        args.workers = "10"
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

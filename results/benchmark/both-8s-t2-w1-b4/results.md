# Benchmark: both-8s

Run 2026-10-25 01:48.

## Setup

- Server: `serve/` at http://127.0.0.1:18000, ONNX Runtime fp32 on CPU: AMD EPYC 9555P 64-Core Processor, 2 cores usable.
- Model: both-8s, window 8 s, threshold 0.4321.
- Settings: threads 2 (0: one per core), workers 1, max_batch 4, max_wait_ms 2, max_queue 0 (0: no limit).
- Client: this script (Python, httpx), on the same machine, sharing its CPU with the server.
- Requests: synthetic 6 s clip, 188 KiB audio/wav; closed loop: each worker sends its next request as soon as the last returns, 100 calls per worker, steps of 1, 2, 4, 10 workers; 10 warm-up requests first, 2 s pause between steps.

## How to read it

- Latency (ms) is client side: HTTP, decoding, preprocessing, queueing and the model; server p50 is the time inside the server (`Server-Timing`), so the difference is network and client overhead.
- Once workers exceed what the server runs in parallel, it is saturated: req/s is its capacity and latency is mostly queueing (about workers / req/s), not slower requests.
- On CPU a batch of B costs about B single passes, so batching (batch = mean batch size) adds little throughput.
- The content of the audio doesn't change the cost: every request is cut or padded to the window.
- Client and server share the CPU: heavy steps are pessimistic for the server alone.

## Results

| workers | ok | 503 | err | req/s | p50 | p90 | p95 | p99 | max | server p50 | batch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 0 | 0 | 20.5 | 45.8 | 53.9 | 67.4 | 93.1 | 97.2 | 44.3 | 1.00 |
| 2 | 200 | 0 | 0 | 21.3 | 88.4 | 108.1 | 117.9 | 186.8 | 201.8 | 86.6 | 1.00 |
| 4 | 400 | 0 | 0 | 21.5 | 172.3 | 207.3 | 291.7 | 407.7 | 418.8 | 168.9 | 2.00 |
| 10 | 1000 | 0 | 0 | 23.3 | 472.3 | 525.4 | 563.2 | 689.4 | 767.7 | 460.4 | 3.94 |

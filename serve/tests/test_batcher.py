"""MicroBatcher: batching, parallel workers, load shedding, errors."""

import asyncio
import threading
import time

import numpy as np
import pytest

from serve.batcher import MicroBatcher, QueueFull


def run(coro):
    return asyncio.run(coro)


def test_results_match_inputs_and_batches_form():
    sizes = []

    def double(x):
        sizes.append(len(x))
        return x[:, 0] * 2

    async def main():
        b = MicroBatcher(double, max_batch=4, max_wait_ms=20)
        b.start()
        out = await asyncio.gather(*(b.submit(np.array([float(i)])) for i in range(8)))
        await b.stop()
        return out, b

    out, b = run(main())
    assert out == [2.0 * i for i in range(8)]
    assert b.requests == 8 and max(sizes) == 4 and b.batches == len(sizes)


def test_workers_run_batches_in_parallel():
    active, peak, lock = [0], [0], threading.Lock()

    def slow(x):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return x[:, 0]

    async def main(workers):
        b = MicroBatcher(slow, max_batch=1, max_wait_ms=0, workers=workers)
        b.start()
        t = time.perf_counter()
        await asyncio.gather(*(b.submit(np.zeros(1)) for _ in range(4)))
        await b.stop()
        return time.perf_counter() - t

    serial = run(main(1))
    peak[0] = 0
    parallel = run(main(4))
    assert peak[0] == 4
    assert parallel < serial / 2


def test_queue_full_is_rejected():
    release = threading.Event()

    def blocked(x):
        release.wait(5)
        return x[:, 0]

    async def main():
        b = MicroBatcher(blocked, max_batch=1, max_wait_ms=0, max_queue=2)
        b.start()
        first = asyncio.create_task(b.submit(np.zeros(1)))
        await asyncio.sleep(0.05)  # taken by the worker, which now blocks
        queued = [asyncio.create_task(b.submit(np.zeros(1))) for _ in range(2)]
        await asyncio.sleep(0.01)
        with pytest.raises(QueueFull):
            await b.submit(np.zeros(1))
        release.set()
        await asyncio.gather(first, *queued)
        await b.stop()
        return b

    b = run(main())
    assert b.rejected == 1 and b.requests == 3


def test_errors_fail_only_their_batch():
    def flaky(x):
        if (x < 0).any():
            raise RuntimeError("boom")
        return x[:, 0]

    async def main():
        b = MicroBatcher(flaky, max_batch=1, max_wait_ms=0)
        b.start()
        with pytest.raises(RuntimeError):
            await b.submit(np.array([-1.0]))
        ok = await b.submit(np.array([3.0]))
        await b.stop()
        return ok

    assert run(main()) == 3.0


def test_bad_settings():
    with pytest.raises(ValueError):
        MicroBatcher(lambda x: x, workers=0)
    with pytest.raises(ValueError):
        MicroBatcher(lambda x: x, max_queue=-1)

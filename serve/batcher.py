"""Dynamic micro-batching: concurrent requests share forward passes, run by a few workers in parallel.

Each request submits its features and awaits a future. Each of `workers` background tasks takes the first
waiting request, then keeps collecting until it has `max_batch` or `max_wait_ms` has passed since that first
one, runs the batch in a worker thread (so the event loop keeps accepting requests), and resolves every
future. Requests that arrive while the workers are busy queue up and form the next batches.

On CPU a forward pass costs about the same per clip at any batch size, so batching saves little; running a
few small batches at once (ONNX Runtime releases the GIL) is what adds throughput. When more than `max_queue`
requests are already waiting, `submit` raises QueueFull at once instead of queueing a request that could
only finish late (the API answers 503).
"""

import asyncio
from collections.abc import Callable

import numpy as np


class QueueFull(Exception):
    """More than `max_queue` requests are already waiting."""


class MicroBatcher:
    def __init__(
        self,
        run_batch: Callable[[np.ndarray], np.ndarray],
        max_batch: int = 4,
        max_wait_ms: float = 2.0,
        workers: int = 1,
        max_queue: int = 0,
    ):
        if max_batch < 1 or max_wait_ms < 0 or workers < 1 or max_queue < 0:
            raise ValueError("max_batch and workers must be >= 1, max_wait_ms and max_queue >= 0")
        self.run_batch = run_batch  # (B, ...) stacked inputs -> (B,) outputs
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000
        self.workers = workers
        self.max_queue = max_queue  # 0: unbounded
        self.queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []

        # Counters for /health (batch sizes show how well batching works under load)
        self.requests = 0
        self.batches = 0
        self.rejected = 0

    def start(self) -> None:
        self.tasks = [asyncio.create_task(self._loop()) for _ in range(self.workers)]

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def submit(self, item: np.ndarray) -> float:
        if self.max_queue and self.queue.qsize() >= self.max_queue:
            self.rejected += 1
            raise QueueFull
        future = asyncio.get_running_loop().create_future()
        self.queue.put_nowait((item, future))
        return await future

    async def _next_batch(self) -> list[tuple[np.ndarray, asyncio.Future]]:
        batch = [await self.queue.get()]
        deadline = asyncio.get_running_loop().time() + self.max_wait_s

        while len(batch) < self.max_batch:
            if not self.queue.empty():  # already waiting: take it without waiting
                batch.append(self.queue.get_nowait())
                continue

            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), timeout))
            except TimeoutError:
                break

        return batch

    async def _loop(self) -> None:
        while True:
            batch = await self._next_batch()
            batch = [(item, future) for item, future in batch if not future.done()]  # clients gone
            if not batch:
                continue

            inputs = np.stack([item for item, _ in batch])
            try:
                outputs = await asyncio.to_thread(self.run_batch, inputs)
            except Exception as error:  # fail this batch's requests, keep serving
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)
                continue

            self.requests += len(batch)
            self.batches += 1
            for (_, future), output in zip(batch, outputs):
                if not future.done():
                    future.set_result(float(output))

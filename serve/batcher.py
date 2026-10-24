"""Dynamic micro-batching: concurrent requests share one forward pass.

Each request submits its features and awaits a future. One background task takes the first waiting
request, then keeps collecting until it has `max_batch` or `max_wait_ms` has passed since that first one,
runs the batch in a worker thread (so the event loop keeps accepting requests), and resolves every future.
Requests that arrive while a batch runs queue up and form the next one, so batches grow with load.
"""

import asyncio
from collections.abc import Callable

import numpy as np


class MicroBatcher:
    def __init__(self, run_batch: Callable[[np.ndarray], np.ndarray], max_batch: int = 32, max_wait_ms: float = 5.0):
        if max_batch < 1 or max_wait_ms < 0:
            raise ValueError("max_batch must be >= 1 and max_wait_ms >= 0")
        self.run_batch = run_batch  # (B, ...) stacked inputs -> (B,) outputs
        self.max_batch = max_batch
        self.max_wait_s = max_wait_ms / 1000
        self.queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self.task: asyncio.Task | None = None

        # Counters for /health (batch sizes show how well batching works under load)
        self.requests = 0
        self.batches = 0

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def submit(self, item: np.ndarray) -> float:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((item, future))
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

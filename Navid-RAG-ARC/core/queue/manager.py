import asyncio
from typing import Any, Callable, Coroutine
from loguru import logger

class AsyncTaskQueue:
    def __init__(self, max_workers: int = 3):
        self.queue = asyncio.Queue()
        self.max_workers = max_workers
        self._workers = []
        self._initialized = False

    async def initialize(self):
        """Initialize workers asynchronously"""
        if not self._initialized:
            for _ in range(self.max_workers):
                worker = asyncio.create_task(self._worker())
                self._workers.append(worker)
            self._initialized = True

    async def _worker(self):
        """Worker process to handle queue items"""
        while True:
            try:
                func, args, kwargs, future = await self.queue.get()
                try:
                    result = await func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
            except Exception as e:
                logger.error(f"Worker error: {e}")

    async def enqueue(self, func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """Enqueue a task and return its future"""
        if not self._initialized:
            await self.initialize()
            
        future = asyncio.Future()
        await self.queue.put((func, args, kwargs, future))
        return await future

    async def shutdown(self):
        """Shutdown the task queue"""
        if self._initialized:
            # Wait for queue to be empty
            await self.queue.join()
            # Cancel all workers
            for worker in self._workers:
                worker.cancel()
            # Wait for all workers to finish
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
            self._initialized = False

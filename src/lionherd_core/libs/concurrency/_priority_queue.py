# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Priority queue implementation using lionherd_core concurrency primitives.

Provides async PriorityQueue using anyio primitives. Note that unlike
asyncio.PriorityQueue, the nowait methods (get_nowait/put_nowait) are async
because anyio.Condition requires async lock acquisition for thread safety.
"""

from __future__ import annotations

import heapq
from typing import Generic, TypeVar

from ._primitives import Condition

T = TypeVar("T")

__all__ = ("PriorityQueue", "QueueEmpty", "QueueFull")


class QueueEmpty(Exception):  # noqa: N818
    """Exception raised when queue.get_nowait() is called on empty queue."""


class QueueFull(Exception):  # noqa: N818
    """Exception raised when queue.put_nowait() is called on full queue."""


class PriorityQueue(Generic[T]):
    """Async priority queue using heapq + anyio concurrency primitives.

    Similar interface to asyncio.PriorityQueue but with key difference:
    get_nowait() and put_nowait() are async methods (must be awaited).
    This is required because anyio.Condition uses async lock acquisition.

    Use cases:
    - await q.get_nowait()  # Must await (unlike asyncio)
    - await q.put_nowait(item)  # Must await (unlike asyncio)
    - await q.get()  # Blocking get
    - await q.put(item)  # Blocking put

    Attributes:
        maxsize: Maximum queue size (0 = unlimited)
    """

    def __init__(self, maxsize: int = 0):
        """Initialize priority queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self.maxsize = maxsize
        self._queue: list[T] = []
        self._condition = Condition()

    async def put(self, item: T) -> None:
        """Put item into queue.

        Args:
            item: Item to add (should be tuple with priority as first element)
        """
        async with self._condition:
            # Wait if queue is full
            while self.maxsize > 0 and len(self._queue) >= self.maxsize:
                await self._condition.wait()

            heapq.heappush(self._queue, item)
            self._condition.notify()

    async def put_nowait(self, item: T) -> None:
        """Put item into queue without blocking (async method).

        Note: Unlike asyncio.PriorityQueue.put_nowait(), this is an async method
        and must be awaited. This is required because anyio.Condition uses async
        lock acquisition for thread safety.

        Args:
            item: Item to add (should be tuple with priority as first element)

        Raises:
            QueueFull: If queue is at maxsize
        """
        async with self._condition:
            if self.maxsize > 0 and len(self._queue) >= self.maxsize:
                raise QueueFull("Queue is full")

            heapq.heappush(self._queue, item)
            # Notify waiting getters that item is available
            self._condition.notify()

    async def get(self) -> T:
        """Get highest priority item from queue (blocking).

        Returns:
            Highest priority item (lowest value first)
        """
        async with self._condition:
            # Wait if queue is empty
            while not self._queue:
                await self._condition.wait()

            item = heapq.heappop(self._queue)
            self._condition.notify()
            return item

    async def get_nowait(self) -> T:
        """Get item without blocking (async method).

        Note: Unlike asyncio.PriorityQueue.get_nowait(), this is an async method
        and must be awaited. This is required because anyio.Condition uses async
        lock acquisition for thread safety.

        Returns:
            Highest priority item

        Raises:
            QueueEmpty: If queue is empty
        """
        async with self._condition:
            if not self._queue:
                raise QueueEmpty("Queue is empty")

            item = heapq.heappop(self._queue)
            # Notify waiting putters that space is available
            self._condition.notify()
            return item

    def qsize(self) -> int:
        """Get current queue size.

        Returns:
            Number of items in queue
        """
        return len(self._queue)

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue is empty
        """
        return len(self._queue) == 0

    def full(self) -> bool:
        """Check if queue is full.

        Returns:
            True if queue is at maxsize
        """
        return self.maxsize > 0 and len(self._queue) >= self.maxsize

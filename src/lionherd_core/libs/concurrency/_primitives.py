# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Self, TypeVar

import anyio
import anyio.abc

T = TypeVar("T")


__all__ = (
    "CapacityLimiter",
    "Condition",
    "Event",
    "Lock",
    "Queue",
    "Semaphore",
)


class Lock:
    """Async mutex lock."""

    __slots__ = ("_lock",)

    def __init__(self) -> None:
        self._lock = anyio.Lock()

    async def acquire(self) -> None:
        """Acquire lock."""
        await self._lock.acquire()

    def release(self) -> None:
        """Release lock."""
        self._lock.release()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class Semaphore:
    """Async semaphore."""

    __slots__ = ("_sem",)

    def __init__(self, initial_value: int) -> None:
        if initial_value < 0:
            raise ValueError("initial_value must be >= 0")
        self._sem = anyio.Semaphore(initial_value)

    async def acquire(self) -> None:
        """Acquire slot."""
        await self._sem.acquire()

    def release(self) -> None:
        """Release slot."""
        self._sem.release()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class CapacityLimiter:
    """Async capacity limiter."""

    __slots__ = ("_lim",)

    def __init__(self, total_tokens: float) -> None:
        """Initialize capacity limiter."""
        if total_tokens <= 0:
            raise ValueError("total_tokens must be > 0")
        self._lim = anyio.CapacityLimiter(total_tokens)

    async def acquire(self) -> None:
        """Acquire capacity."""
        await self._lim.acquire()

    def release(self) -> None:
        """Release capacity."""
        self._lim.release()

    @property
    def remaining_tokens(self) -> float:
        """Available capacity (deprecated)."""
        return self._lim.available_tokens

    @property
    def total_tokens(self) -> float:
        """Get capacity limit."""
        return self._lim.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: float) -> None:
        """Set capacity limit."""
        if value <= 0:
            raise ValueError("total_tokens must be > 0")
        self._lim.total_tokens = value

    @property
    def borrowed_tokens(self) -> float:
        """Get borrowed tokens."""
        return self._lim.borrowed_tokens

    @property
    def available_tokens(self) -> float:
        """Get available tokens."""
        return self._lim.available_tokens

    async def acquire_on_behalf_of(self, borrower: object) -> None:
        """Acquire for borrower."""
        await self._lim.acquire_on_behalf_of(borrower)

    def release_on_behalf_of(self, borrower: object) -> None:
        """Release for borrower."""
        self._lim.release_on_behalf_of(borrower)

    # Support idiomatic AnyIO usage: `async with limiter: ...`
    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


@dataclass(slots=True)
class Queue(Generic[T]):
    """Async FIFO queue."""

    _send: anyio.abc.ObjectSendStream[T]
    _recv: anyio.abc.ObjectReceiveStream[T]

    @classmethod
    def with_maxsize(cls, maxsize: int) -> Queue[T]:
        """Create queue with maxsize."""
        send, recv = anyio.create_memory_object_stream(maxsize)
        return cls(send, recv)

    async def put(self, item: T) -> None:
        """Put item."""
        await self._send.send(item)

    def put_nowait(self, item: T) -> None:
        """Put without blocking."""
        self._send.send_nowait(item)  # type: ignore[attr-defined]

    async def get(self) -> T:
        """Get item."""
        return await self._recv.receive()

    def get_nowait(self) -> T:
        """Get without blocking."""
        return self._recv.receive_nowait()  # type: ignore[attr-defined]

    async def close(self) -> None:
        """Close streams."""
        await self._send.aclose()
        await self._recv.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    @property
    def sender(self) -> anyio.abc.ObjectSendStream[T]:
        """Get send stream."""
        return self._send

    @property
    def receiver(self) -> anyio.abc.ObjectReceiveStream[T]:
        """Get receive stream."""
        return self._recv


class Event:
    """Async event for task signaling."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = anyio.Event()

    def set(self) -> None:
        """Set flag."""
        self._event.set()

    def is_set(self) -> bool:
        """Check if set."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Wait for flag."""
        await self._event.wait()

    def statistics(self) -> anyio.EventStatistics:
        """Get statistics."""
        return self._event.statistics()


class Condition:
    """Async condition variable."""

    __slots__ = ("_condition",)

    def __init__(self, lock: Lock | None = None) -> None:
        """Initialize condition variable."""
        _lock = lock._lock if lock else None
        self._condition = anyio.Condition(_lock)

    async def acquire(self) -> None:
        """Acquire lock."""
        await self._condition.acquire()

    def release(self) -> None:
        """Release lock."""
        self._condition.release()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    async def wait(self) -> None:
        """Wait until notified."""
        await self._condition.wait()

    def notify(self, n: int = 1) -> None:
        """Wake n tasks."""
        self._condition.notify(n)

    def notify_all(self) -> None:
        """Wake all tasks."""
        self._condition.notify_all()

    def statistics(self) -> anyio.ConditionStatistics:
        """Get statistics."""
        return self._condition.statistics()

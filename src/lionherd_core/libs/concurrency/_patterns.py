# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import TypeVar

import anyio

# ExceptionGroup is built-in in Python 3.11+
from ._cancel import effective_deadline, move_on_at
from ._errors import non_cancel_subgroup
from ._primitives import CapacityLimiter
from ._task import create_task_group
from ._utils import current_time

T = TypeVar("T")
R = TypeVar("R")


__all__ = (
    "CompletionStream",
    "bounded_map",
    "gather",
    "race",
    "retry",
)


async def gather(*aws: Awaitable[T], return_exceptions: bool = False) -> list[T | BaseException]:
    """Run awaitables concurrently."""
    if not aws:
        return []

    results: list[T | BaseException | None] = [None] * len(aws)

    async def _runner(idx: int, aw: Awaitable[T]) -> None:
        try:
            results[idx] = await aw
        except BaseException as exc:
            results[idx] = exc
            if not return_exceptions:
                raise  # Propagate to the TaskGroup

    try:
        async with create_task_group() as tg:
            for i, aw in enumerate(aws):
                tg.start_soon(_runner, i, aw)
    except ExceptionGroup as eg:
        if not return_exceptions:
            # Separate cancellations from real failures while preserving structure/tracebacks
            rest = non_cancel_subgroup(eg)
            if rest is not None:
                raise rest
            # All were cancellations -> propagate cancellation group
            raise

    return results  # type: ignore


async def race(*aws: Awaitable[T]) -> T:
    """Return first completion."""
    if not aws:
        raise ValueError("race() requires at least one awaitable")
    send, recv = anyio.create_memory_object_stream(0)

    async def _runner(aw: Awaitable[T]) -> None:
        try:
            res = await aw
            await send.send((True, res))
        except BaseException as exc:
            await send.send((False, exc))

    async with send, recv, create_task_group() as tg:
        for aw in aws:
            tg.start_soon(_runner, aw)
        ok, payload = await recv.receive()
        tg.cancel_scope.cancel()

    # Raise outside the TaskGroup context to avoid ExceptionGroup wrapping
    if ok:
        return payload  # type: ignore[return-value]
    raise payload  # type: ignore[misc]


async def bounded_map(
    func: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    limit: int,
    return_exceptions: bool = False,
) -> list[R | BaseException]:
    """Apply async function to items with concurrency limit."""
    if limit <= 0:
        raise ValueError("limit must be >= 1")

    seq = list(items)
    if not seq:
        return []

    out: list[R | BaseException | None] = [None] * len(seq)
    limiter = CapacityLimiter(limit)

    async def _runner(i: int, x: T) -> None:
        async with limiter:
            try:
                out[i] = await func(x)
            except BaseException as exc:
                out[i] = exc
                if not return_exceptions:
                    raise  # Propagate to the TaskGroup

    try:
        async with create_task_group() as tg:
            for i, x in enumerate(seq):
                tg.start_soon(_runner, i, x)
    except ExceptionGroup as eg:
        if not return_exceptions:
            # Separate cancellations from real failures while preserving structure/tracebacks
            rest = non_cancel_subgroup(eg)
            if rest is not None:
                raise rest
            # All were cancellations -> propagate cancellation group
            raise

    return out  # type: ignore


class CompletionStream:
    """Async completion stream with structured concurrency and explicit lifecycle."""

    def __init__(self, aws: Sequence[Awaitable[T]], *, limit: int | None = None):
        self.aws = aws
        self.limit = limit
        self._task_group: anyio.abc.TaskGroup | None = None
        self._send: anyio.abc.ObjectSendStream[tuple[int, T]] | None = None
        self._recv: anyio.abc.ObjectReceiveStream[tuple[int, T]] | None = None
        self._completed_count = 0
        self._total_count = len(aws)

    async def __aenter__(self):
        n = len(self.aws)
        self._send, self._recv = anyio.create_memory_object_stream(n)
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()

        limiter = CapacityLimiter(self.limit) if self.limit else None

        async def _runner(i: int, aw: Awaitable[T]) -> None:
            if limiter:
                await limiter.acquire()
            try:
                res = await aw
                try:
                    assert self._send is not None
                    await self._send.send((i, res))  # type: ignore[arg-type]
                except anyio.ClosedResourceError:
                    # Stream was closed (e.g., early break from iteration)
                    # Swallow the error gracefully
                    pass
            finally:
                if limiter:
                    limiter.release()

        # Start all tasks
        for i, aw in enumerate(self.aws):
            self._task_group.start_soon(_runner, i, aw)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel remaining tasks and clean up
        if self._task_group:
            await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
        if self._send:
            await self._send.aclose()
        if self._recv:
            await self._recv.aclose()
        return False

    def __aiter__(self):
        if not self._recv:
            raise RuntimeError("CompletionStream must be used as async context manager")
        return self

    async def __anext__(self):
        if self._completed_count >= self._total_count:
            raise StopAsyncIteration

        try:
            result = await self._recv.receive()
            self._completed_count += 1
            return result
        except anyio.EndOfStream:
            raise StopAsyncIteration


async def retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.1,
) -> T:
    """Deadline-aware exponential backoff retry."""
    attempt = 0
    deadline = effective_deadline()
    while True:
        try:
            return await fn()
        except retry_on:
            attempt += 1
            if attempt >= attempts:
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                import random

                delay *= 1 + random.random() * jitter

            # Cap by ambient deadline if one exists
            if deadline is not None:
                remaining = deadline - current_time()
                if remaining <= 0:
                    # Out of time; surface the last error
                    raise
                # Use move_on_at to avoid TOCTOU race between deadline check and sleep
                with move_on_at(deadline):
                    await anyio.sleep(delay)
                # If we were cancelled by deadline, re-raise the original exception
                if current_time() >= deadline:
                    raise
            else:
                await anyio.sleep(delay)

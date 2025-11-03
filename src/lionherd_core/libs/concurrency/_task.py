# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import anyio
import anyio.abc

T = TypeVar("T")
R = TypeVar("R")

__all__ = (
    "TaskGroup",
    "create_task_group",
)


class TaskGroup:
    """Structured concurrency task group."""

    __slots__ = ("_tg",)

    def __init__(self, tg: anyio.abc.TaskGroup) -> None:
        self._tg = tg

    @property
    def cancel_scope(self) -> anyio.CancelScope:
        """Cancel scope for task group lifetime."""
        return self._tg.cancel_scope

    def start_soon(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        name: str | None = None,
    ) -> None:
        """Start task without waiting for initialization."""
        self._tg.start_soon(func, *args, name=name)

    async def start(
        self,
        func: Callable[..., Awaitable[R]],
        *args: Any,
        name: str | None = None,
    ) -> R:
        """Start task and wait for initialization."""
        return await self._tg.start(func, *args, name=name)


@asynccontextmanager
async def create_task_group() -> AsyncIterator[TaskGroup]:
    """Create task group for structured concurrency."""
    async with anyio.create_task_group() as tg:
        yield TaskGroup(tg)

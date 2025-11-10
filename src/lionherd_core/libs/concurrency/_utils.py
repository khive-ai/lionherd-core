# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from functools import cache, partial
from typing import Any, ParamSpec, TypeVar

import anyio
import anyio.to_thread

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

__all__ = ("current_time", "is_coro_func", "run_sync", "sleep")


@cache
def _is_coro_func(func: Callable[..., Any]) -> bool:
    """Cached check for coroutine function, handles partials."""
    # Unwrap partials to get the underlying function
    while isinstance(func, partial):
        func = func.func

    # Check if it's a coroutine function
    return inspect.iscoroutinefunction(func)


def is_coro_func(func: Callable[..., Any]) -> bool:
    """Check if function is coroutine function."""
    return _is_coro_func(func)


def current_time() -> float:
    """Get current time in seconds (monotonic clock)."""
    return anyio.current_time()


async def run_sync(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run synchronous function in thread pool.

    Preserves the function signature using ParamSpec, improving type inference.

    Args:
        func: Synchronous function to run in thread pool
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result of func
    """
    if kwargs:
        # anyio.to_thread.run_sync doesn't accept **kwargs, use partial
        func_with_kwargs = partial(func, **kwargs)
        return await anyio.to_thread.run_sync(func_with_kwargs, *args)
    return await anyio.to_thread.run_sync(func, *args)


async def sleep(seconds: float) -> None:
    """Sleep without blocking event loop."""
    await anyio.sleep(seconds)

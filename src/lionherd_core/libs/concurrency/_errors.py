# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

import anyio

T = TypeVar("T")
P = ParamSpec("P")


__all__ = (
    "get_cancelled_exc_class",
    "is_cancelled",
    "non_cancel_subgroup",
    "shield",
    "split_cancellation",
)


def get_cancelled_exc_class() -> type[BaseException]:
    """Return the backend-native cancellation exception class."""
    return anyio.get_cancelled_exc_class()


def is_cancelled(exc: BaseException) -> bool:
    """True if this is the backend-native cancellation exception."""
    return isinstance(exc, anyio.get_cancelled_exc_class())


async def shield(func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run ``func`` immune to outer cancellation.

    Args:
        func: Async function to shield from cancellation
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result of func
    """
    with anyio.CancelScope(shield=True):
        return await func(*args, **kwargs)  # type: ignore[return-value]


# -------- ExceptionGroup helpers (Python 3.11+) --------


def split_cancellation(
    eg: BaseExceptionGroup,
) -> tuple[BaseExceptionGroup | None, BaseExceptionGroup | None]:
    """Split an exception group into (cancel_subgroup, non_cancel_subgroup).

    Uses Python 3.11+ ExceptionGroup.split() to preserve structure, tracebacks,
    __cause__/__context__/__notes__.

    Args:
        eg: Exception group to split

    Returns:
        Tuple of (cancellation exceptions, non-cancellation exceptions).
        Either element may be None if that category is empty.
    """
    return eg.split(anyio.get_cancelled_exc_class())


def non_cancel_subgroup(eg: BaseExceptionGroup) -> BaseExceptionGroup | None:
    """Return subgroup without cancellations, or None if empty.

    Args:
        eg: Exception group to filter

    Returns:
        Subgroup containing only non-cancellation exceptions, or None.
    """
    _, rest = eg.split(anyio.get_cancelled_exc_class())
    return rest

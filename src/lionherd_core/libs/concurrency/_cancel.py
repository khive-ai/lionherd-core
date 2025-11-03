# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from math import isinf

import anyio

from ._utils import current_time

CancelScope = anyio.CancelScope


__all__ = (
    "CancelScope",
    "effective_deadline",
    "fail_after",
    "fail_at",
    "move_on_after",
    "move_on_at",
)


@contextmanager
def fail_after(seconds: float | None) -> Iterator[CancelScope]:
    """Create context with timeout that raises TimeoutError."""
    if seconds is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    with anyio.fail_after(seconds) as scope:
        yield scope


@contextmanager
def move_on_after(seconds: float | None) -> Iterator[CancelScope]:
    """Create context with timeout that silently cancels."""
    if seconds is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    with anyio.move_on_after(seconds) as scope:
        yield scope


@contextmanager
def fail_at(deadline: float | None) -> Iterator[CancelScope]:
    """Create context that raises TimeoutError at absolute deadline."""
    if deadline is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    now = current_time()
    seconds = max(0.0, deadline - now)
    with fail_after(seconds) as scope:
        yield scope


@contextmanager
def move_on_at(deadline: float | None) -> Iterator[CancelScope]:
    """Create context that silently cancels at absolute deadline."""
    if deadline is None:
        # No timeout, but still cancellable by outer scopes
        with CancelScope() as scope:
            yield scope
        return
    now = current_time()
    seconds = max(0.0, deadline - now)
    with anyio.move_on_after(seconds) as scope:
        yield scope


def effective_deadline() -> float | None:
    """Return the ambient effective deadline, or None if unlimited.

    AnyIO uses +inf to indicate "no deadline".
    """
    d = anyio.current_effective_deadline()
    return None if isinf(d) else d

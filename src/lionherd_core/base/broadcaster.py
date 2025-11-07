# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from ..libs.concurrency import is_coro_func

logger = logging.getLogger(__name__)

__all__ = ["Broadcaster"]


class Broadcaster:
    """Singleton pub/sub for O(1) memory event broadcasting.

    Memory Management:
        Uses weakref for automatic subscriber cleanup when callback objects are
        garbage collected. Prevents memory leaks in long-running services with
        dynamic agent/tenant lifecycles.
    """

    _instance: ClassVar[Broadcaster | None] = None
    _subscribers: ClassVar[
        list[weakref.ref[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]]
    ] = []
    _event_type: ClassVar[type]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def subscribe(cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]) -> None:
        """Add subscriber callback.

        Callbacks stored as weak references for automatic cleanup when callback
        objects are garbage collected. Prevents cross-tenant/session leaks.
        """
        # Check if callback already subscribed (compare actual callbacks, not weakrefs)
        for weak_ref in cls._subscribers:
            if weak_ref() is callback:
                return  # Already subscribed

        # Store as weakref for automatic cleanup
        weak_callback = weakref.ref(callback)
        cls._subscribers.append(weak_callback)

    @classmethod
    def unsubscribe(
        cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]
    ) -> None:
        """Remove subscriber callback."""
        # Find and remove weakref that points to this callback
        for weak_ref in list(cls._subscribers):
            if weak_ref() is callback:
                cls._subscribers.remove(weak_ref)
                return

    @classmethod
    async def broadcast(cls, event: Any) -> None:
        """Broadcast event to all subscribers."""
        if not isinstance(event, cls._event_type):
            raise ValueError(f"Event must be of type {cls._event_type.__name__}")

        # Resolve weak references and filter out dead ones
        callbacks = []
        alive_refs = []

        for weak_ref in cls._subscribers:
            callback = weak_ref()
            if callback is not None:
                callbacks.append(callback)
                alive_refs.append(weak_ref)

        # Update subscriber list to remove dead references
        cls._subscribers[:] = alive_refs  # In-place update for ClassVar

        # Broadcast to live callbacks
        for callback in callbacks:
            try:
                if is_coro_func(callback):
                    result = callback(event)
                    if result is not None:  # Coroutine functions return awaitable
                        await result
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}", exc_info=True)

    @classmethod
    def get_subscriber_count(cls) -> int:
        """Get live subscriber count (excludes garbage-collected callbacks)."""
        # Clean up dead references and count live ones
        alive_refs = []
        count = 0

        for ref in cls._subscribers:
            if ref() is not None:
                alive_refs.append(ref)
                count += 1

        # Update subscriber list to remove dead references
        cls._subscribers[:] = alive_refs  # In-place update for ClassVar

        return count

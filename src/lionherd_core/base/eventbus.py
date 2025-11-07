# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from ..libs.concurrency import gather

__all__ = ("EventBus", "Handler")

Handler = Callable[..., Awaitable[None]]


class EventBus:
    """In-process pub/sub with concurrent handler execution.

    Fire-and-forget: handlers run concurrently via gather(), exceptions suppressed.

    Memory Management:
        Uses weakref for automatic handler cleanup when handler objects are garbage collected.
        Prevents memory leaks in long-running services where handlers are registered dynamically.
    """

    def __init__(self) -> None:
        """Initialize with empty subscription registry."""
        # Store weak references to handlers for automatic cleanup
        self._subs: dict[str, list[weakref.ref[Handler]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe async handler to topic.

        Handlers are stored as weak references for automatic cleanup when
        handler objects are garbage collected. Prevents memory leaks in
        long-running services.
        """
        # Store weakref without callback - cleanup happens lazily
        weak_handler = weakref.ref(handler)
        self._subs[topic].append(weak_handler)

    def unsubscribe(self, topic: str, handler: Handler) -> bool:
        """Unsubscribe handler from topic. Returns True if found and removed."""
        if topic not in self._subs:
            return False

        # Find and remove weakref that points to this handler
        for weak_ref in list(self._subs[topic]):
            if weak_ref() is handler:
                self._subs[topic].remove(weak_ref)
                return True
        return False

    def _cleanup_dead_refs(self, topic: str) -> list[Handler]:
        """Remove garbage-collected handlers and return list of live handlers.

        Lazily cleans up dead weakrefs during normal operations (emit/handler_count).
        Updates subscription list in-place to remove dead references.

        Returns:
            List of live handler callables (weakrefs resolved).
        """
        weak_refs = self._subs[topic]
        handlers = []
        alive_refs = []

        for weak_ref in weak_refs:
            handler = weak_ref()
            if handler is not None:
                handlers.append(handler)
                alive_refs.append(weak_ref)

        # Update subscription list to remove dead references
        self._subs[topic] = alive_refs
        return handlers

    async def emit(self, topic: str, *args: Any, **kwargs: Any) -> None:
        """Emit event to all subscribers.

        Handlers run concurrently via gather(), exceptions suppressed.
        Dead weakrefs are lazily cleaned up during emission.
        """
        if topic not in self._subs:
            return

        handlers = self._cleanup_dead_refs(topic)
        if not handlers:
            return

        # Run all handlers concurrently, suppress exceptions
        await gather(*(h(*args, **kwargs) for h in handlers), return_exceptions=True)

    def clear(self, topic: str | None = None) -> None:
        """Clear subscriptions for topic (or all if None)."""
        if topic is None:
            self._subs.clear()
        else:
            self._subs.pop(topic, None)

    def topics(self) -> list[str]:
        """Get list of all registered topics."""
        return list(self._subs.keys())

    def handler_count(self, topic: str) -> int:
        """Get number of live handlers for topic (excludes garbage-collected handlers)."""
        if topic not in self._subs:
            return 0

        handlers = self._cleanup_dead_refs(topic)
        return len(handlers)

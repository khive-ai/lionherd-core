# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from ..libs.concurrency import gather

__all__ = ("EventBus", "Handler")

Handler = Callable[..., Awaitable[None]]


class EventBus:
    """In-process pub/sub with concurrent handler execution.

    Fire-and-forget: handlers run concurrently via gather(), exceptions suppressed.
    """

    def __init__(self) -> None:
        """Initialize with empty subscription registry."""
        self._subs: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe async handler to topic."""
        self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> bool:
        """Unsubscribe handler from topic. Returns True if found and removed."""
        if topic in self._subs and handler in self._subs[topic]:
            self._subs[topic].remove(handler)
            return True
        return False

    async def emit(self, topic: str, *args: Any, **kwargs: Any) -> None:
        """Emit event to all subscribers. Handlers run concurrently, exceptions suppressed."""
        handlers = self._subs.get(topic, [])
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
        """Get number of handlers for topic."""
        return len(self._subs.get(topic, []))

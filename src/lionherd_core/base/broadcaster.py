# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from ..libs.concurrency import is_coro_func

logger = logging.getLogger(__name__)

__all__ = ["Broadcaster"]


class Broadcaster:
    """Singleton pub/sub for O(1) memory event broadcasting."""

    _instance: ClassVar[Broadcaster | None] = None
    _subscribers: ClassVar[list[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]] = []
    _event_type: ClassVar[type]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def subscribe(cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]) -> None:
        """Add subscriber callback."""
        if callback not in cls._subscribers:
            cls._subscribers.append(callback)

    @classmethod
    def unsubscribe(
        cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]
    ) -> None:
        """Remove subscriber callback."""
        if callback in cls._subscribers:
            cls._subscribers.remove(callback)

    @classmethod
    async def broadcast(cls, event: Any) -> None:
        """Broadcast event to all subscribers."""
        if not isinstance(event, cls._event_type):
            raise ValueError(f"Event must be of type {cls._event_type.__name__}")

        for callback in cls._subscribers:
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
        """Get subscriber count."""
        return len(cls._subscribers)

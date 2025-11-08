"""Test Event status race condition - Issue #26.

The race: Multiple concurrent invoke() calls execute _invoke() multiple times
instead of once, causing duplicate execution, double API calls, and double charges.

This test demonstrates the TOCTOU (Time-Of-Check-Time-Of-Use) footgun.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import Field

from lionherd_core.base.event import Event, EventStatus


@dataclass
class ExecutionTracker:
    """Fixture-scoped execution tracking for race condition tests.

    Provides isolated state management per test, preventing cross-test pollution
    and ensuring automatic cleanup.
    """

    _counts: dict[str, int] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def register(self, key: str) -> None:
        """Register a new tracking key with initial count and lock."""
        self._counts[key] = 0
        self._locks[key] = asyncio.Lock()

    def get_count(self, key: str) -> int:
        """Get current execution count for a key."""
        return self._counts.get(key, 0)

    async def increment(self, key: str) -> int:
        """Atomically increment and return count for a key."""
        async with self._locks[key]:
            self._counts[key] += 1
            return self._counts[key]


@pytest.fixture
def execution_tracker() -> ExecutionTracker:
    """Provide execution tracking for race condition tests.

    Automatically cleans up after each test to prevent state leaks.
    """
    tracker = ExecutionTracker()
    yield tracker
    # Cleanup is automatic via dataclass field factories


class CountingEvent(Event):
    """Test event that tracks execution count via injected tracker."""

    tracker: Any = Field(default=None, exclude=True)
    counter_key: str = Field(default="default", exclude=True)

    def model_post_init(self, __context) -> None:
        """Initialize tracking after Pydantic validation."""
        super().model_post_init(__context)
        if self.tracker is not None:
            key = str(self.id)
            self.counter_key = key
            self.tracker.register(key)

    @property
    def execution_count(self) -> int:
        """Get execution count for this event."""
        if self.tracker is None:
            return 0
        return self.tracker.get_count(self.counter_key)

    async def _invoke(self):
        """Track execution count and simulate work."""
        # Simulate async work (creates race window)
        await asyncio.sleep(0.01)

        # Increment counter (the resource that should only be touched once)
        if self.tracker is not None:
            count = await self.tracker.increment(self.counter_key)
        else:
            count = 1

        return f"result_{count}"


@pytest.mark.asyncio
async def test_concurrent_invoke_executes_once(execution_tracker):
    """Multiple concurrent invoke() calls should execute _invoke() exactly once.

    WITHOUT fix: Both calls execute _invoke() → execution_count = 2
    WITH fix: Second call waits or returns cached result → execution_count = 1
    """
    event = CountingEvent(tracker=execution_tracker)

    # Sanity check - starts in PENDING
    assert event.status == EventStatus.PENDING
    assert event.execution_count == 0

    # Launch 10 concurrent invoke() calls
    results = await asyncio.gather(*[event.invoke() for _ in range(10)])

    # CRITICAL: _invoke() should execute exactly once
    assert event.execution_count == 1, (
        f"Expected 1 execution, got {event.execution_count}. "
        f"Race condition: multiple concurrent invoke() calls executed _invoke() multiple times."
    )

    # All calls should return the same result
    assert all(r == results[0] for r in results), f"Results differ: {results}"

    # Event should be COMPLETED
    assert event.status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_invoke_returns_cached_result_after_completion(execution_tracker):
    """After first execution completes, subsequent invoke() should return cached result."""
    event = CountingEvent(tracker=execution_tracker)

    # First execution
    result1 = await event.invoke()
    assert event.execution_count == 1
    assert event.status == EventStatus.COMPLETED

    # Second invoke() should NOT re-execute
    result2 = await event.invoke()
    assert event.execution_count == 1  # Still 1, not 2
    assert result2 == result1  # Same result returned

    # Third invoke() - verify idempotency
    result3 = await event.invoke()
    assert event.execution_count == 1
    assert result3 == result1


@pytest.mark.asyncio
async def test_racing_invoke_calls_high_concurrency(execution_tracker):
    """Stress test: 100 concurrent invoke() calls should still execute once."""
    event = CountingEvent(tracker=execution_tracker)

    # Launch 100 concurrent calls
    results = await asyncio.gather(*[event.invoke() for _ in range(100)])

    # Only one execution
    assert event.execution_count == 1, (
        f"Race condition under high concurrency: {event.execution_count} executions"
    )

    # All return same result
    assert len(set(results)) == 1, f"Got different results: {set(results)}"


@pytest.mark.asyncio
async def test_invoke_idempotency_with_delay(execution_tracker):
    """invoke() after completion should be instant (no re-execution delay)."""
    event = CountingEvent(tracker=execution_tracker)

    # First invoke (takes ~10ms due to sleep)
    await event.invoke()
    assert event.execution_count == 1

    # Subsequent invoke should be instant (no sleep)
    import time

    start = time.perf_counter()
    await event.invoke()
    duration = time.perf_counter() - start

    # Should return instantly (<1ms), not re-execute (which takes 10ms)
    assert duration < 0.005, (
        f"invoke() after completion took {duration * 1000:.1f}ms. "
        f"Expected instant return of cached result."
    )
    assert event.execution_count == 1  # Still 1

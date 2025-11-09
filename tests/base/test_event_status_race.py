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


class TimeoutEvent(Event):
    """Test event that times out during execution."""

    tracker: Any = Field(default=None, exclude=True)
    counter_key: str = Field(default="default", exclude=True)
    timeout: float = Field(default=0.001)  # Very short timeout

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
        """Simulate work that exceeds timeout."""
        if self.tracker is not None:
            await self.tracker.increment(self.counter_key)

        # Sleep longer than timeout to trigger timeout
        await asyncio.sleep(self.timeout * 10)
        return "should_not_reach"


class FailingEvent(Event):
    """Test event that raises an exception during execution."""

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
        """Raise an exception during execution."""
        # Increment counter before raising
        if self.tracker is not None:
            await self.tracker.increment(self.counter_key)

        # Simulate some async work before failing
        await asyncio.sleep(0.01)
        raise ValueError("Intentional test failure")


class SlowEvent(Event):
    """Test event with slow execution for cancellation testing."""

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
        """Simulate slow work that can be cancelled."""
        if self.tracker is not None:
            await self.tracker.increment(self.counter_key)

        # Long sleep to allow cancellation
        await asyncio.sleep(1.0)
        return "should_not_complete"


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


@pytest.mark.asyncio
async def test_concurrent_invoke_with_timeout_race(execution_tracker):
    """Multiple concurrent invoke() calls during timeout should execute once.

    Verifies that @async_synchronized prevents duplicate execution even when
    the first execution times out.
    """
    from lionherd_core.types._sentinel import Unset

    event = TimeoutEvent(tracker=execution_tracker, timeout=0.001)

    # Sanity check - starts in PENDING
    assert event.status == EventStatus.PENDING
    assert event.execution_count == 0

    # Launch 10 concurrent calls
    results = await asyncio.gather(*[event.invoke() for _ in range(10)])

    # CRITICAL: Only one execution (that times out)
    assert event.execution_count == 1, (
        f"Expected 1 execution, got {event.execution_count}. "
        f"Race condition: multiple concurrent invoke() calls executed _invoke() multiple times."
    )

    # All callers should get None or Unset (timeout results in no meaningful response)
    # First caller may get None (cached response), waiters get Unset
    assert all(r is None or r is Unset for r in results), f"Expected None or Unset, got: {results}"

    # Event should be CANCELLED (timeout)
    assert event.status == EventStatus.CANCELLED


@pytest.mark.asyncio
async def test_concurrent_invoke_with_exception_race(execution_tracker):
    """Multiple concurrent invoke() calls when _invoke() raises should execute once.

    Verifies that @async_synchronized prevents duplicate execution even when
    the first execution raises an exception.
    """
    from lionherd_core.types._sentinel import Unset

    event = FailingEvent(tracker=execution_tracker)

    # Sanity check - starts in PENDING
    assert event.status == EventStatus.PENDING
    assert event.execution_count == 0

    # Launch 10 concurrent calls
    results = await asyncio.gather(*[event.invoke() for _ in range(10)])

    # CRITICAL: Only one execution (that fails)
    assert event.execution_count == 1, (
        f"Expected 1 execution, got {event.execution_count}. "
        f"Race condition: multiple concurrent invoke() calls executed _invoke() multiple times."
    )

    # All callers should get None or Unset (error results in no meaningful response)
    # First caller may get None (cached response), waiters get Unset
    assert all(r is None or r is Unset for r in results), f"Expected None or Unset, got: {results}"

    # Event should be FAILED
    assert event.status == EventStatus.FAILED

    # Error should be captured
    assert event.execution.error is not None
    assert "Intentional test failure" in str(event.execution.error)


@pytest.mark.asyncio
async def test_concurrent_invoke_with_cancellation_race(execution_tracker):
    """Multiple concurrent invoke() calls during cancellation should execute once.

    Verifies that @async_synchronized prevents duplicate execution even when
    tasks are cancelled mid-execution.
    """
    event = SlowEvent(tracker=execution_tracker)

    # Sanity check - starts in PENDING
    assert event.status == EventStatus.PENDING
    assert event.execution_count == 0

    # Launch 10 concurrent tasks
    tasks = [asyncio.create_task(event.invoke()) for _ in range(10)]

    # Let execution start
    await asyncio.sleep(0.02)

    # Verify execution started
    assert event.execution_count == 1, "Execution should have started"

    # Cancel all tasks
    for task in tasks:
        task.cancel()

    # Wait for cancellation (gather with return_exceptions to catch CancelledError)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # CRITICAL: Only one execution started (then cancelled)
    assert event.execution_count == 1, (
        f"Expected 1 execution, got {event.execution_count}. "
        f"Race condition: multiple concurrent invoke() calls executed _invoke() multiple times."
    )

    # All tasks should be cancelled
    assert all(isinstance(r, asyncio.CancelledError) for r in results), (
        f"Expected all CancelledError, got: {results}"
    )

    # Event should be CANCELLED
    assert event.status == EventStatus.CANCELLED

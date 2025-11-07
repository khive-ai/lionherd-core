"""Test Event status race condition - Issue #26.

The race: Multiple concurrent invoke() calls execute _invoke() multiple times
instead of once, causing duplicate execution, double API calls, and double charges.

This test demonstrates the TOCTOU (Time-Of-Check-Time-Of-Use) footgun.
"""

import asyncio

import pytest
from pydantic import Field

from lionherd_core.base.event import Event, EventStatus

# Module-level tracking to avoid Pydantic field issues
_execution_counts = {}
_execution_locks = {}


class CountingEvent(Event):
    """Test event that tracks execution count via module-level dict."""

    counter_key: str = Field(default="default", exclude=True)

    def model_post_init(self, __context) -> None:
        """Initialize tracking after Pydantic validation."""
        super().model_post_init(__context)
        key = str(self.id)
        self.counter_key = key
        _execution_counts[key] = 0
        _execution_locks[key] = asyncio.Lock()

    @property
    def execution_count(self) -> int:
        """Get execution count for this event."""
        return _execution_counts.get(self.counter_key, 0)

    async def _invoke(self):
        """Track execution count and simulate work."""
        # Simulate async work (creates race window)
        await asyncio.sleep(0.01)

        # Increment counter (the resource that should only be touched once)
        async with _execution_locks[self.counter_key]:
            _execution_counts[self.counter_key] += 1
            count = _execution_counts[self.counter_key]

        return f"result_{count}"


@pytest.mark.asyncio
async def test_concurrent_invoke_executes_once():
    """Multiple concurrent invoke() calls should execute _invoke() exactly once.

    WITHOUT fix: Both calls execute _invoke() → execution_count = 2
    WITH fix: Second call waits or returns cached result → execution_count = 1
    """
    event = CountingEvent()

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
async def test_invoke_returns_cached_result_after_completion():
    """After first execution completes, subsequent invoke() should return cached result."""
    event = CountingEvent()

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
async def test_racing_invoke_calls_high_concurrency():
    """Stress test: 100 concurrent invoke() calls should still execute once."""
    event = CountingEvent()

    # Launch 100 concurrent calls
    results = await asyncio.gather(*[event.invoke() for _ in range(100)])

    # Only one execution
    assert event.execution_count == 1, (
        f"Race condition under high concurrency: {event.execution_count} executions"
    )

    # All return same result
    assert len(set(results)) == 1, f"Got different results: {set(results)}"


@pytest.mark.asyncio
async def test_invoke_idempotency_with_delay():
    """invoke() after completion should be instant (no re-execution delay)."""
    event = CountingEvent()

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

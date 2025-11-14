# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Security and robustness tests for Processor (P0 bug fixes)."""

import math

import pytest

from lionherd_core.base.event import Event, EventStatus
from lionherd_core.base.pile import Pile
from lionherd_core.base.processor import Processor
from lionherd_core.errors import QueueFullError


class SecTestEvent(Event):
    """Test event for security tests."""

    return_value: str | None = None

    async def _invoke(self):
        return self.return_value


class SecTestProcessor(Processor):
    """Test processor for security tests."""

    event_type = SecTestEvent


# ==================== Queue Size Limit Tests ====================


@pytest.mark.asyncio
async def test_processor_enforces_max_queue_size():
    """Test queue rejects events when max_queue_size exceeded (DoS protection).

    Security Fix: Previously no queue limit, allowing unbounded memory growth.
    """
    pile = Pile[Event]()
    proc = SecTestProcessor(
        queue_capacity=10,
        capacity_refresh_time=0.1,
        pile=pile,
        max_queue_size=5,  # Small limit for testing
    )

    # Add events up to limit
    events = [SecTestEvent(return_value=f"event_{i}") for i in range(5)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)  # Should succeed

    assert proc.queue.qsize() == 5

    # 6th event should raise QueueFullError
    overflow_event = SecTestEvent(return_value="overflow")
    pile.add(overflow_event)

    with pytest.raises(QueueFullError, match=r"Queue size .* exceeds max"):
        await proc.enqueue(overflow_event.id)


@pytest.mark.asyncio
async def test_processor_queue_limit_default_1000():
    """Test default max_queue_size is 1000."""
    pile = Pile[Event]()
    proc = SecTestProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    assert proc.max_queue_size == 1000


# ==================== Priority Validation Tests ====================


@pytest.mark.asyncio
async def test_processor_rejects_nan_priority():
    """Test enqueue rejects NaN priority (heap corruption prevention).

    Security Fix: NaN priority breaks heap invariants, allowing queue manipulation.
    """
    pile = Pile[Event]()
    proc = SecTestProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    event = SecTestEvent(return_value="test")
    pile.add(event)

    with pytest.raises(ValueError, match="Priority must be finite and not NaN"):
        await proc.enqueue(event.id, priority=float("nan"))


@pytest.mark.asyncio
async def test_processor_rejects_inf_priority():
    """Test enqueue rejects Inf priority (queue manipulation prevention).

    Security Fix: Inf priority allows malicious events to monopolize queue.
    """
    pile = Pile[Event]()
    proc = SecTestProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    event = SecTestEvent(return_value="test")
    pile.add(event)

    # Positive infinity
    with pytest.raises(ValueError, match="Priority must be finite"):
        await proc.enqueue(event.id, priority=float("inf"))

    # Negative infinity
    with pytest.raises(ValueError, match="Priority must be finite"):
        await proc.enqueue(event.id, priority=float("-inf"))


# ==================== Denial Retry Limit Tests ====================


@pytest.mark.asyncio
async def test_processor_aborts_after_3_permission_denials():
    """Test events aborted after 3 permission denials (infinite loop prevention).

    Security Fix: Previously denied events requeued infinitely, causing unbounded
    queue growth and CPU consumption.
    """

    class DenyingProcessor(SecTestProcessor):
        """Processor that always denies permission."""

        async def request_permission(self, **kwargs):
            return False  # Always deny

    pile = Pile[Event]()
    from unittest.mock import AsyncMock

    executor_mock = AsyncMock()
    executor_mock._update_progression = AsyncMock()

    proc = DenyingProcessor(
        queue_capacity=10, capacity_refresh_time=0.1, pile=pile, executor=executor_mock
    )

    event = SecTestEvent(return_value="denied")
    pile.add(event)
    await proc.enqueue(event.id)

    # Process 3 times - each time denied and requeued
    await proc.process()  # 1st denial - requeue
    assert proc.queue.qsize() == 1
    assert event.id in proc._denial_counts
    assert proc._denial_counts[event.id] == 1

    await proc.process()  # 2nd denial - requeue
    assert proc.queue.qsize() == 1
    assert proc._denial_counts[event.id] == 2

    await proc.process()  # 3rd denial - ABORT
    assert proc.queue.qsize() == 0  # Not requeued
    assert event.id not in proc._denial_counts  # Cleaned up

    # Verify event was aborted
    executor_mock._update_progression.assert_called_with(event, EventStatus.ABORTED)


@pytest.mark.asyncio
async def test_processor_denial_backoff_increases_priority():
    """Test denied events get lower priority on retry (exponential backoff).

    Prevents denial storms from blocking queue.
    """

    class DenyFirstProcessor(SecTestProcessor):
        """Processor that denies first N times."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.deny_count = 0

        async def request_permission(self, **kwargs):
            self.deny_count += 1
            return self.deny_count > 2  # Deny first 2, allow after

    pile = Pile[Event]()
    proc = DenyFirstProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    event = SecTestEvent(return_value="test")
    pile.add(event)

    original_priority = 100.0
    await proc.enqueue(event.id, priority=original_priority)

    # 1st denial
    await proc.process()
    priority1, _ = await proc.queue.get()
    assert priority1 == original_priority + 1.0  # +1s backoff

    # Re-enqueue for next test
    await proc.queue.put((priority1, event.id))

    # 2nd denial
    await proc.process()
    priority2, _ = await proc.queue.get()
    assert priority2 == priority1 + 2.0  # +2s backoff (total +3s)


# ==================== Missing Event Handling Tests ====================


@pytest.mark.asyncio
async def test_processor_handles_event_removed_from_pile():
    """Test processor skips events removed from pile while in queue (robustness).

    Edge Case: Event deleted from pile after enqueue but before process.
    """
    pile = Pile[Event]()
    proc = SecTestProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    event1 = SecTestEvent(return_value="keep")
    event2 = SecTestEvent(return_value="removed")
    pile.add(event1)
    pile.add(event2)

    await proc.enqueue(event1.id, priority=1.0)
    await proc.enqueue(event2.id, priority=2.0)

    # Remove event2 from pile (simulates external deletion)
    pile.remove(event2.id)

    # Process should skip missing event without crashing
    await proc.process()

    # event1 should be processed, event2 skipped
    assert event1.execution.status == EventStatus.COMPLETED
    # event2 no longer in pile, can't check status


# ==================== Bounds Validation Tests ====================


@pytest.mark.asyncio
async def test_processor_validates_queue_capacity_upper_bound():
    """Test queue_capacity <= 10000 (prevent unbounded batches)."""
    pile = Pile[Event]()

    with pytest.raises(ValueError, match="Queue capacity must be <= 10000"):
        SecTestProcessor(queue_capacity=20000, capacity_refresh_time=0.1, pile=pile)


@pytest.mark.asyncio
async def test_processor_validates_refresh_time_bounds():
    """Test capacity_refresh_time in [0.01, 3600] (prevent hot loop/starvation)."""
    pile = Pile[Event]()

    # Too low - CPU hot loop
    with pytest.raises(ValueError, match=r"Capacity refresh time must be >= 0\.01s"):
        SecTestProcessor(queue_capacity=10, capacity_refresh_time=0.001, pile=pile)

    # Too high - starvation
    with pytest.raises(ValueError, match=r"Capacity refresh time must be <= 3600s"):
        SecTestProcessor(queue_capacity=10, capacity_refresh_time=7200, pile=pile)


@pytest.mark.asyncio
async def test_processor_validates_concurrency_limit_positive():
    """Test concurrency_limit >= 1."""
    pile = Pile[Event]()

    with pytest.raises(ValueError, match="Concurrency limit must be >= 1"):
        SecTestProcessor(
            queue_capacity=10, capacity_refresh_time=0.1, pile=pile, concurrency_limit=0
        )


# ==================== Cleanup Memory Leak Tests ====================


@pytest.mark.asyncio
async def test_cleanup_events_removes_denial_tracking():
    """Test cleanup_events() cleans up processor denial counts (C1 fix).

    Security Fix: Prevents memory leak where denial_counts accumulate forever
    when events are manually removed via cleanup_events().
    """
    from lionherd_core.base.processor import Executor

    class CleanupTestProcessor(SecTestProcessor):
        """Processor that denies first time."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.deny_first = True

        async def request_permission(self, **kwargs):
            if self.deny_first:
                self.deny_first = False
                return False
            return True

    class CleanupTestExecutor(Executor):
        processor_type = CleanupTestProcessor

    executor = CleanupTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1}
    )
    await executor.start()

    # Create event and deny it once
    event = SecTestEvent(return_value="test")
    await executor.append(event)

    # Process - will be denied once and requeued
    await executor.processor.process()

    # Verify denial tracked
    assert event.id in executor.processor._denial_counts
    assert executor.processor._denial_counts[event.id] == 1

    # Manually clean up the event (simulates logging + cleanup)
    removed = await executor.cleanup_events([EventStatus.PENDING])

    # C1 FIX VERIFICATION: Denial tracking should be cleaned up
    assert removed == 1
    assert event.id not in executor.processor._denial_counts, (
        "Memory leak: denial_counts not cleaned up by cleanup_events()"
    )


@pytest.mark.asyncio
async def test_cleanup_events_uses_pile_locks():
    """Test cleanup_events() uses Pile's async locks to prevent race conditions (C2 fix).

    Security Fix: Prevents TOCTOU race where cleanup_events() modifies progressions
    concurrently with _update_progression(), causing data corruption.

    Uses Pile's built-in async context manager locks instead of custom lock.
    """
    from lionherd_core.base.processor import Executor

    executor = Executor.__new__(Executor)  # Create without __init__ to set processor_type
    executor.processor_type = SecTestProcessor
    executor.__init__(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    # Create and complete an event
    event = SecTestEvent(return_value="test")
    await executor.append(event)
    event.execution.status = EventStatus.COMPLETED
    await executor._update_progression(event, EventStatus.COMPLETED)

    # Verify event in COMPLETED progression
    assert event.id in executor.states.get_progression("completed")

    # C2 FIX VERIFICATION: cleanup_events should use Pile locks (no custom _progression_lock)
    assert not hasattr(executor, "_progression_lock"), (
        "Executor should NOT have custom _progression_lock, should use Pile locks"
    )

    # Clean up and verify removal
    removed = await executor.cleanup_events([EventStatus.COMPLETED])

    assert removed == 1
    assert event.id not in executor.states.get_progression("completed"), (
        "Event should be removed by cleanup_events()"
    )

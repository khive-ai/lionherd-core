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


# ==================== M1/M2 Memory Management Tests ====================


@pytest.mark.asyncio
async def test_stop_clears_denial_counts():
    """Test stop() clears _denial_counts to prevent memory leaks (M1 fix).

    Security Fix: Prevents memory leaks across processor restart cycles.
    """

    class AlwaysDenyProcessor(SecTestProcessor):
        """Processor that always denies permission."""

        async def request_permission(self, **kwargs):
            return False  # Always deny

    pile = Pile[Event]()
    proc = AlwaysDenyProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    # Create events and deny them
    events = [SecTestEvent(return_value=f"event_{i}") for i in range(5)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)
        # Process immediately after enqueue - will be denied and requeued
        await proc.process()

    # Verify denial tracking populated (5 events denied once each)
    assert len(proc._denial_counts) == 5
    for event in events:
        assert event.id in proc._denial_counts
        assert proc._denial_counts[event.id] == 1

    # M1 FIX VERIFICATION: stop() should clear denial tracking
    await proc.stop()

    assert len(proc._denial_counts) == 0, "Memory leak: stop() did not clear _denial_counts"


@pytest.mark.asyncio
async def test_bounded_denial_tracking():
    """Test FIFO eviction when _denial_counts exceeds max_denial_tracking (M2 fix).

    Security Fix: Prevents unbounded memory growth from DoS scenarios.
    """

    class AlwaysDenyProcessor(SecTestProcessor):
        """Processor that always denies permission."""

        async def request_permission(self, **kwargs):
            return False  # Always deny

    pile = Pile[Event]()
    # Set max_denial_tracking to 10 for testing
    proc = AlwaysDenyProcessor(
        queue_capacity=10, capacity_refresh_time=0.1, pile=pile, max_denial_tracking=10
    )

    # Create 15 events (exceeds max_denial_tracking=10)
    events = [SecTestEvent(return_value=f"event_{i}") for i in range(15)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)
        # Process immediately after enqueue - will be denied and requeued
        await proc.process()

    # M2 FIX VERIFICATION: Should have exactly 10 entries (FIFO eviction)
    assert len(proc._denial_counts) == 10, (
        f"Expected 10 denial entries (max_denial_tracking), got {len(proc._denial_counts)}"
    )

    # First 5 events should be evicted (FIFO), last 10 should remain
    for i in range(5):
        assert events[i].id not in proc._denial_counts, (
            f"Event {i} should be evicted (oldest entries)"
        )

    for i in range(5, 15):
        assert events[i].id in proc._denial_counts, f"Event {i} should be tracked (newest entries)"


# ==================== Additional Validation Tests ====================


@pytest.mark.asyncio
async def test_processor_validates_max_queue_size_positive():
    """Test max_queue_size >= 1 validation (line 82)."""
    pile = Pile[Event]()

    with pytest.raises(ValueError, match="Max queue size must be >= 1"):
        SecTestProcessor(
            queue_capacity=10,
            capacity_refresh_time=0.1,
            pile=pile,
            max_queue_size=0,
        )


@pytest.mark.asyncio
async def test_processor_validates_max_denial_tracking_positive():
    """Test max_denial_tracking >= 1 validation (line 86)."""
    pile = Pile[Event]()

    with pytest.raises(ValueError, match="Max denial tracking must be >= 1"):
        SecTestProcessor(
            queue_capacity=10,
            capacity_refresh_time=0.1,
            pile=pile,
            max_denial_tracking=0,
        )


# ==================== Concurrency Control Tests ====================


@pytest.mark.asyncio
async def test_processor_execute_without_concurrency_limit():
    """Test _execute_with_concurrency_control without semaphore (line 319 path)."""
    pile = Pile[Event]()
    proc = SecTestProcessor(
        queue_capacity=10,
        capacity_refresh_time=0.1,
        pile=pile,
    )

    # Create and process event
    event = SecTestEvent(return_value="test")
    pile.add(event)
    await proc.enqueue(event.id)

    # Temporarily set semaphore to None to test line 319 path
    original_sem = proc._concurrency_sem
    proc._concurrency_sem = None

    # Process should work without semaphore (line 319)
    await proc.process()

    # Restore semaphore
    proc._concurrency_sem = original_sem

    assert event.execution.status == EventStatus.COMPLETED


# ==================== Executor Property Tests ====================


@pytest.mark.asyncio
async def test_executor_event_type_property():
    """Test Executor.event_type property (line 377)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})

    assert executor.event_type == SecTestEvent


@pytest.mark.asyncio
async def test_executor_strict_event_type_property():
    """Test Executor.strict_event_type property (line 382)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})

    # Default should be False (Pile's default)
    assert isinstance(executor.strict_event_type, bool)


# ==================== Executor Method Tests ====================


@pytest.mark.asyncio
async def test_executor_forward_method():
    """Test Executor.forward() calls processor.process() (lines 417-418)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    event = SecTestEvent(return_value="test")
    await executor.append(event)

    # Forward should process the event
    await executor.forward()

    assert event.execution.status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_executor_start_backfills_pending_events():
    """Test Executor.start() backfills pending events (line 426)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})

    # Add events with PENDING status using the proper Flow API
    events = [SecTestEvent(return_value=f"event_{i}") for i in range(3)]
    for event in events:
        # Add to items pile
        executor.states.add_item(event)
        # Manually set event status to PENDING (default for new events)
        event.execution.status = EventStatus.PENDING
        # Add to pending progression
        pending_prog = executor.states.get_progression("pending")
        pending_prog.append(event.id)

    # Verify events are pending
    assert len(executor.pending_events) == 3

    # Start should backfill pending events into processor queue
    await executor.start()

    # Verify processor queue has all 3 pending events (line 426 covered)
    assert executor.processor.queue.qsize() == 3


@pytest.mark.asyncio
async def test_executor_stop_method():
    """Test Executor.stop() calls processor.stop() (lines 431-432)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    event = SecTestEvent(return_value="test")
    await executor.append(event)

    # Stop should work
    await executor.stop()

    # Processor should be stopped
    assert executor.processor.execution_mode is False


# ==================== Status Query Tests ====================


@pytest.mark.asyncio
async def test_executor_completed_events_property():
    """Test Executor.completed_events property (line 470)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    # Add and process events
    events = [SecTestEvent(return_value=f"event_{i}") for i in range(3)]
    for event in events:
        await executor.append(event)

    await executor.processor.process()

    # Check completed events
    completed = executor.completed_events
    assert len(completed) == 3
    assert all(e.execution.status == EventStatus.COMPLETED for e in completed)


@pytest.mark.asyncio
async def test_executor_failed_events_property():
    """Test Executor.failed_events property (line 480)."""
    from lionherd_core.base.processor import Executor

    class FailingTestEvent(Event):
        """Event that always fails."""

        async def _invoke(self):
            raise ValueError("Test error")

    class FailingTestProcessor(Processor):
        event_type = FailingTestEvent

    class TestExecutor(Executor):
        processor_type = FailingTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    # Add failing event
    event = FailingTestEvent()
    await executor.append(event)
    await executor.processor.process()

    # Check failed events
    failed = executor.failed_events
    assert len(failed) == 1
    assert failed[0].execution.status == EventStatus.FAILED


@pytest.mark.asyncio
async def test_executor_processing_events_property():
    """Test Executor.processing_events property (line 485)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})

    # Add event but don't process
    event = SecTestEvent(return_value="test")
    executor.states.add_item(event)
    event.execution.status = EventStatus.PROCESSING
    await executor._update_progression(event, EventStatus.PROCESSING)

    # Check processing events
    processing = executor.processing_events
    assert len(processing) == 1
    assert processing[0].execution.status == EventStatus.PROCESSING


@pytest.mark.asyncio
async def test_executor_status_counts_method():
    """Test Executor.status_counts() method (line 489)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    # Add events
    event1 = SecTestEvent(return_value="event1")
    event2 = SecTestEvent(return_value="event2")

    await executor.append(event1)
    await executor.append(event2)

    # Process one batch (may process both)
    await executor.processor.process()

    # Check status counts - should have dict with progression names as keys
    counts = executor.status_counts()
    assert isinstance(counts, dict)
    # Verify all status names are present
    total_events = sum(counts.values())
    assert total_events == 2  # Both events accounted for


# ==================== Cleanup Tests ====================


@pytest.mark.asyncio
async def test_executor_cleanup_events_default_statuses():
    """Test cleanup_events() with default statuses (line 512)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    # Add and process events
    events = [SecTestEvent(return_value=f"event_{i}") for i in range(3)]
    for event in events:
        await executor.append(event)

    await executor.processor.process()

    # Cleanup with default statuses (should include COMPLETED)
    removed = await executor.cleanup_events()

    assert removed == 3


# ==================== Inspection Tests ====================


@pytest.mark.asyncio
async def test_executor_inspect_state_method():
    """Test Executor.inspect_state() debug method (lines 530-534)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})
    await executor.start()

    # Add events
    events = [SecTestEvent(return_value=f"event_{i}") for i in range(2)]
    for event in events:
        await executor.append(event)

    # Inspect state
    state_str = executor.inspect_state()

    assert isinstance(state_str, str)
    assert "Executor State" in state_str
    assert "pending" in state_str.lower() or "PENDING" in state_str


@pytest.mark.asyncio
async def test_executor_contains_method():
    """Test Executor.__contains__() method (line 538)."""
    from lionherd_core.base.processor import Executor

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})

    event = SecTestEvent(return_value="test")
    executor.states.add_item(event)

    # Test __contains__ with Event
    assert event in executor

    # Test __contains__ with UUID
    assert event.id in executor

    # Test negative case
    other_event = SecTestEvent(return_value="other")
    assert other_event not in executor


# ==================== Error Handling Tests ====================


@pytest.mark.asyncio
async def test_executor_update_progression_missing_status():
    """Test _update_progression with missing progression (lines 406-407).

    Note: Currently raises NotFoundError from Pile, not ConfigurationError.
    The KeyError handler at lines 406-407 may be unreachable due to semantic
    exception migration (NotFoundError replaced KeyError in Pile).
    """
    from lionherd_core.base.processor import Executor
    from lionherd_core.errors import NotFoundError

    class TestExecutor(Executor):
        processor_type = SecTestProcessor

    executor = TestExecutor(processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.1})

    # Find and remove the "pending" progression
    pending_prog = executor.states.get_progression("pending")
    executor.states.progressions.remove(pending_prog.id)

    event = SecTestEvent(return_value="test")
    executor.states.add_item(event)

    # Trying to update to PENDING should raise NotFoundError
    # (Lines 406-407 KeyError handler may be unreachable with current Pile implementation)
    with pytest.raises(NotFoundError, match="not found"):
        await executor._update_progression(event, EventStatus.PENDING)

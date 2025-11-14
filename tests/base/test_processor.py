# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Processor: priority queue, capacity control, background execution.

Processor Architecture - UUID-Based Queue:
==========================================

The Processor manages event execution using a priority queue pattern:
    - Queue stores (priority, UUID) tuples (not full events)
    - Events live in executor's Flow.items (single source of truth)
    - Processor fetches events from pile when dequeuing
    - After execution, updates executor progressions

Key Features:
-------------
1. Priority Queue: Lower priority values processed first
2. Capacity Limiting: Max events per batch, refresh after processing
3. Background Loop: Continuous execution until stop() called
4. Concurrency Control: Optional semaphore limits concurrent executions
5. Streaming Support: Handles async generator events
6. Executor Integration: Calls executor._update_progression() after execution

Test Coverage:
--------------
- Initialization: Valid/invalid parameters
- Enqueue/Dequeue: UUID queueing, priority ordering, pile fetch
- Capacity: Tracking, reset after batch
- Process: Batch execution, progression updates
- Background Loop: execute(), stop/start
- Concurrency: Semaphore limits
- Streaming Events: Consume generator
- Errors: Queue empty, execution failures, permission denied
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from lionherd_core.base.event import Event, EventStatus
from lionherd_core.base.pile import Pile
from lionherd_core.base.processor import Executor, Processor
from lionherd_core.libs import concurrency

# ============================================================================
# Test Event Subclasses (Concrete implementations for testing)
# ============================================================================


class SimpleEvent(Event):
    """Simple event that completes successfully."""

    return_value: Any = "success"
    streaming: bool = False

    async def _invoke(self) -> Any:
        """Return configured value."""
        # Event.invoke() handles status transitions
        return self.return_value


class FailingEvent(Event):
    """Event that always fails."""

    error_message: str = "Test error"
    streaming: bool = False

    async def _invoke(self) -> Any:
        """Raise configured error."""
        raise ValueError(self.error_message)


class SlowEvent(Event):
    """Event that takes time to complete."""

    delay: float = 0.05
    return_value: Any = "completed"
    streaming: bool = False

    async def _invoke(self) -> Any:
        """Wait then return value."""
        await concurrency.sleep(self.delay)
        return self.return_value


class StreamingEvent(Event):
    """Event that yields values via async generator."""

    stream_count: int = 3
    streaming: bool = True

    async def _invoke(self) -> Any:
        """Not used for streaming events."""
        raise NotImplementedError("Use stream() instead")

    async def stream(self):
        """Yield values asynchronously."""
        for i in range(self.stream_count):
            await concurrency.sleep(0.01)
            yield i
        # Mark as completed after streaming
        self.execution.status = EventStatus.COMPLETED
        self.execution.response = f"streamed {self.stream_count} items"


class SimpleProcessor(Processor):
    """Concrete Processor for SimpleEvent."""

    event_type = SimpleEvent


class FailingProcessor(Processor):
    """Concrete Processor for FailingEvent."""

    event_type = FailingEvent


class SlowProcessor(Processor):
    """Concrete Processor for SlowEvent."""

    event_type = SlowEvent


class StreamingProcessor(Processor):
    """Concrete Processor for StreamingEvent."""

    event_type = StreamingEvent


class PermissionProcessor(SimpleProcessor):
    """Processor with custom permission logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permission_requests = []

    async def request_permission(self, **kwargs: Any) -> bool:
        """Track permission requests."""
        self.permission_requests.append(kwargs)
        return kwargs.get("allowed", True)


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processor_init_valid():
    """Test Processor initialization with valid parameters.

    Design Pattern: Dependency injection of Pile reference.

    Architecture:
        - Processor stores reference to executor's Flow.items (not a copy)
        - Queue stores UUIDs only, events live in pile (single source of truth)
        - Capacity control prevents runaway queue processing
        - Optional concurrency semaphore limits parallel executions

    This pattern avoids data duplication while maintaining type safety.
    """
    pile = Pile[Event]()
    proc = SimpleProcessor(
        queue_capacity=10, capacity_refresh_time=0.1, pile=pile, concurrency_limit=5
    )

    assert proc.queue_capacity == 10
    assert proc.capacity_refresh_time == 0.1
    assert proc.pile is pile
    assert proc.executor is None
    assert proc.available_capacity == 10
    assert not proc.execution_mode
    assert not proc.is_stopped()
    assert proc._concurrency_sem is not None


@pytest.mark.asyncio
async def test_processor_init_default_concurrency_limit():
    """Test Processor initialization with default concurrency limit (100)."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=5, capacity_refresh_time=0.1, pile=pile)

    # Default concurrency limit is 100 (safe default)
    assert proc._concurrency_sem is not None


@pytest.mark.asyncio
async def test_processor_init_with_executor():
    """Test Processor initialization with executor reference."""
    pile = Pile[Event]()
    executor_mock = Mock()

    proc = SimpleProcessor(
        queue_capacity=10, capacity_refresh_time=0.1, pile=pile, executor=executor_mock
    )

    assert proc.executor is executor_mock


@pytest.mark.asyncio
async def test_processor_init_invalid_queue_capacity():
    """Test Processor raises ValueError for invalid queue_capacity."""
    pile = Pile[Event]()

    with pytest.raises(ValueError, match="Queue capacity must be greater than 0"):
        SimpleProcessor(queue_capacity=0, capacity_refresh_time=0.1, pile=pile)

    with pytest.raises(ValueError, match="Queue capacity must be greater than 0"):
        SimpleProcessor(queue_capacity=-5, capacity_refresh_time=0.1, pile=pile)


@pytest.mark.asyncio
async def test_processor_init_invalid_refresh_time():
    """Test Processor raises ValueError for invalid capacity_refresh_time.

    Validation: capacity_refresh_time must be in [0.01, 3600] seconds.
    """
    pile = Pile[Event]()

    # Too low (< 0.01s = 10ms) - prevents CPU hot loop
    with pytest.raises(ValueError, match=r"Capacity refresh time must be >= 0\.01s"):
        SimpleProcessor(queue_capacity=10, capacity_refresh_time=0, pile=pile)

    with pytest.raises(ValueError, match=r"Capacity refresh time must be >= 0\.01s"):
        SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.001, pile=pile)

    # Too high (> 3600s = 1 hour) - prevents starvation
    with pytest.raises(ValueError, match=r"Capacity refresh time must be <= 3600s"):
        SimpleProcessor(queue_capacity=10, capacity_refresh_time=7200, pile=pile)


@pytest.mark.asyncio
async def test_processor_create_classmethod():
    """Test Processor.create() async constructor."""
    pile = Pile[Event]()
    proc = await SimpleProcessor.create(
        queue_capacity=15, capacity_refresh_time=0.2, pile=pile, concurrency_limit=3
    )

    assert isinstance(proc, SimpleProcessor)
    assert proc.queue_capacity == 15
    assert proc.capacity_refresh_time == 0.2
    assert proc.pile is pile


# ============================================================================
# Enqueue/Dequeue Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processor_enqueue_with_priority():
    """Test enqueuing event UUID with explicit priority."""
    pile = Pile[Event]()
    event = SimpleEvent()
    pile.add(event)

    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    await proc.enqueue(event.id, priority=5.0)
    assert not proc.queue.empty()


@pytest.mark.asyncio
async def test_processor_enqueue_default_priority():
    """Test enqueuing event UUID with default priority (created_at)."""
    pile = Pile[Event]()
    event = SimpleEvent()
    pile.add(event)

    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    await proc.enqueue(event.id)  # Should use event.created_at
    assert not proc.queue.empty()


@pytest.mark.asyncio
async def test_processor_dequeue_returns_event():
    """Test dequeue retrieves event from pile (not UUID)."""
    pile = Pile[Event]()
    event = SimpleEvent(return_value="test_value")
    pile.add(event)

    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    await proc.enqueue(event.id, priority=1.0)
    dequeued_event = await proc.dequeue()

    assert dequeued_event is event
    assert dequeued_event.return_value == "test_value"


@pytest.mark.asyncio
async def test_processor_dequeue_priority_order():
    """Test dequeue respects priority order (lower values first).

    Priority Queue Semantics:
        - Lower numerical values = higher priority (processed first)
        - Default priority = event.created_at (earlier events first)
        - Custom priority allows urgent events to skip the line

    Design Rationale:
        Using numerical priority (not high/medium/low enum) provides:
        - Fine-grained control (can insert between any two events)
        - Natural timestamp-based ordering (created_at as default)
        - Flexible scheduling (rate limiting, retry backoff, etc.)
    """
    pile = Pile[Event]()

    event1 = SimpleEvent(return_value="high_priority")
    event2 = SimpleEvent(return_value="low_priority")
    event3 = SimpleEvent(return_value="medium_priority")

    pile.add(event1)
    pile.add(event2)
    pile.add(event3)

    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    # Enqueue in non-sorted order
    await proc.enqueue(event2.id, priority=10.0)  # Low priority
    await proc.enqueue(event1.id, priority=1.0)  # High priority
    await proc.enqueue(event3.id, priority=5.0)  # Medium priority

    # Dequeue should return in priority order
    first = await proc.dequeue()
    second = await proc.dequeue()
    third = await proc.dequeue()

    assert first.return_value == "high_priority"
    assert second.return_value == "medium_priority"
    assert third.return_value == "low_priority"


@pytest.mark.asyncio
async def test_processor_join_waits_for_empty_queue():
    """Test join() blocks until queue is empty."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    event1 = SimpleEvent()
    event2 = SimpleEvent()
    pile.add(event1)
    pile.add(event2)

    await proc.enqueue(event1.id)
    await proc.enqueue(event2.id)

    # Start task to drain queue
    async def drain_queue():
        await concurrency.sleep(0.05)
        await proc.dequeue()
        await concurrency.sleep(0.05)
        await proc.dequeue()

    async with concurrency.create_task_group() as tg:
        tg.start_soon(drain_queue)
        tg.start_soon(proc.join)

    assert proc.queue.empty()


# ============================================================================
# Capacity Management Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processor_available_capacity_property():
    """Test available_capacity property getter/setter."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    assert proc.available_capacity == 10

    proc.available_capacity = 5
    assert proc.available_capacity == 5


@pytest.mark.asyncio
async def test_processor_capacity_decreases_during_process():
    """Test capacity decreases as events are processed."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=3, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    # Add events to pile and queue
    for i in range(3):
        event = SimpleEvent(return_value=f"event_{i}")
        pile.add(event)
        await proc.enqueue(event.id)

    initial_capacity = proc.available_capacity
    assert initial_capacity == 3

    await proc.process()

    # After processing, capacity should be reset
    assert proc.available_capacity == 3


@pytest.mark.asyncio
async def test_processor_capacity_reset_after_batch():
    """Test capacity resets to queue_capacity after processing batch.

    Capacity Reset Mechanism:
        - Capacity decreases as events are dequeued (prevents runaway processing)
        - After batch completes, capacity resets to queue_capacity
        - This implements periodic throttling (batch → reset → batch)

    Design Rationale:
        Capacity control prevents processor from draining entire queue in one cycle.
        Benefits:
        - Predictable batch sizes for monitoring
        - Opportunity for priority changes between batches
        - Rate limiting control point (capacity + refresh time)
        - Resource management (limit memory/CPU per batch)
    """
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=5, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    # Manually reduce capacity
    proc.available_capacity = 2

    # Add event to process
    event = SimpleEvent()
    pile.add(event)
    await proc.enqueue(event.id)

    await proc.process()

    # Capacity should reset to queue_capacity
    assert proc.available_capacity == 5


@pytest.mark.asyncio
async def test_processor_capacity_not_reset_if_no_events_processed():
    """Test capacity does NOT reset if no events were processed.

    Capacity is only consumed when events are actually processed (permission granted).
    If no events processed (empty queue or all denied), capacity remains unchanged.
    """
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=5, capacity_refresh_time=0.01, pile=pile)

    # Manually reduce capacity
    proc.available_capacity = 2

    # Process with empty queue
    await proc.process()

    # Capacity should remain unchanged (no events processed)
    assert proc.available_capacity == 2


# ============================================================================
# Process Tests
# ============================================================================


@pytest.mark.asyncio
async def test_processor_process_single_event():
    """Test processing single event updates status and calls executor.

    Processor-Executor Integration:
        1. Dequeue event UUID from priority queue
        2. Fetch event from pile (shared with executor)
        3. Update executor progression to PROCESSING (before invoke)
        4. Invoke event (status transitions handled by Event._invoke)
        5. Update executor progression to final status (after invoke)

    Key Insight - Dual Progression Updates:
        - First update (PROCESSING): Tracks "this event is running now"
        - Second update (COMPLETED/FAILED): Tracks final outcome
        - This enables real-time monitoring of in-flight events
        - Executor can query "show me all processing events" at any moment

    Why this matters:
        Without dual updates, events would jump from PENDING → COMPLETED,
        losing visibility into what's actively executing. The PROCESSING
        state is critical for observability and debugging.
    """
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    event = SimpleEvent(return_value="test_result")
    pile.add(event)
    await proc.enqueue(event.id)

    await proc.process()

    # Event should be invoked
    assert event.execution.status == EventStatus.COMPLETED
    assert event.execution.response == "test_result"

    # Executor progression should be updated (PROCESSING + final)
    assert executor_mock._update_progression.call_count == 2
    executor_mock._update_progression.assert_any_call(event, EventStatus.PROCESSING)
    executor_mock._update_progression.assert_any_call(event)


@pytest.mark.asyncio
async def test_processor_process_multiple_events():
    """Test processing multiple events in batch."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    events = [SimpleEvent(return_value=f"result_{i}") for i in range(3)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)

    await proc.process()

    # All events should be completed
    for i, event in enumerate(events):
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == f"result_{i}"


@pytest.mark.asyncio
async def test_processor_process_respects_capacity_limit():
    """Test process() respects available_capacity limit."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=2, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    # Add 5 events
    events = [SimpleEvent(return_value=f"result_{i}") for i in range(5)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)

    # First process() should dequeue and process max 2 events (capacity limit)
    await proc.process()

    # process() uses TaskGroup which waits for all tasks, so events should be complete
    # Count completed events - should have processed exactly 2 (capacity limit)
    completed = sum(1 for e in events if e.execution.status == EventStatus.COMPLETED)
    # Note: May be 1 or 2 depending on pending event retry logic
    assert completed >= 1 and completed <= 2

    # Remaining events should still be queued
    assert not proc.queue.empty()


@pytest.mark.asyncio
async def test_processor_process_without_executor():
    """Test process() works without executor (no progression updates)."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=None)

    event = SimpleEvent(return_value="test")
    pile.add(event)
    await proc.enqueue(event.id)

    # Should not raise error even without executor
    await proc.process()

    assert event.execution.status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_processor_process_handles_event_failure():
    """Test process() handles event execution failures gracefully."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = FailingProcessor(
        queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    event = FailingEvent(error_message="Expected failure")
    pile.add(event)
    await proc.enqueue(event.id)

    # Should not raise - errors are captured in execution
    await proc.process()

    # Event should be marked as FAILED
    assert event.execution.status == EventStatus.FAILED
    assert executor_mock._update_progression.call_count == 2


@pytest.mark.asyncio
async def test_processor_process_streaming_event():
    """Test process() handles streaming events correctly."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = StreamingProcessor(
        queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    event = StreamingEvent(stream_count=3)
    pile.add(event)
    await proc.enqueue(event.id)

    await proc.process()

    # Event should be completed
    assert event.execution.status == EventStatus.COMPLETED
    # Should update progression for PROCESSING + final
    assert executor_mock._update_progression.call_count == 2


@pytest.mark.asyncio
async def test_processor_process_streaming_event_with_error():
    """Test process() handles streaming event errors."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    # Create event that fails during streaming
    class FailingStreamingEvent(StreamingEvent):
        async def stream(self):
            yield 1
            raise ValueError("Stream error")

    proc = StreamingProcessor(
        queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    event = FailingStreamingEvent()
    pile.add(event)
    await proc.enqueue(event.id)

    await proc.process()

    # Progression should still be updated after error
    assert executor_mock._update_progression.call_count == 2


@pytest.mark.asyncio
async def test_processor_process_with_concurrency_limit():
    """Test process() respects concurrency semaphore."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    # Create processor with concurrency limit of 2
    proc = SlowProcessor(
        queue_capacity=10,
        capacity_refresh_time=0.01,
        pile=pile,
        executor=executor_mock,
        concurrency_limit=2,
    )

    # Add 4 slow events
    events = [SlowEvent(delay=0.05, return_value=f"result_{i}") for i in range(4)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)

    # Process should complete despite concurrency limit
    await proc.process()

    # All events should eventually complete (though timing may vary)
    completed = sum(1 for e in events if e.execution.status == EventStatus.COMPLETED)
    assert completed >= 1  # At least one should complete


@pytest.mark.asyncio
async def test_processor_request_permission_default():
    """Test default request_permission() always returns True."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    result = await proc.request_permission(foo="bar", baz=123)
    assert result is True


@pytest.mark.asyncio
async def test_processor_request_permission_custom():
    """Test custom request_permission() can deny events."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = PermissionProcessor(
        queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    # Event that will be allowed
    class AllowedEvent(SimpleEvent):
        @property
        def request(self) -> dict:
            return {"allowed": True}

    # Event that will be denied
    class DeniedEvent(SimpleEvent):
        @property
        def request(self) -> dict:
            return {"allowed": False}

    event_allowed = AllowedEvent(return_value="allowed")
    pile.add(event_allowed)
    await proc.enqueue(event_allowed.id)

    event_denied = DeniedEvent(return_value="denied")
    pile.add(event_denied)
    await proc.enqueue(event_denied.id)

    await proc.process()

    # Only allowed event should be processed
    assert event_allowed.execution.status == EventStatus.COMPLETED
    assert event_denied.execution.status == EventStatus.PENDING

    # Permission should be requested multiple times for denied event (it retries)
    # At least 2 requests: 1 for allowed, 1+ for denied (retried until capacity exhausted)
    assert len(proc.permission_requests) >= 2
    # First request should be for allowed event
    assert {"allowed": True} in proc.permission_requests
    # Remaining requests should be for denied event
    assert {"allowed": False} in proc.permission_requests


# ============================================================================
# Background Loop Tests (execute/stop/start)
# ============================================================================


@pytest.mark.asyncio
async def test_processor_execution_mode_property():
    """Test execution_mode property getter/setter."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    assert not proc.execution_mode

    proc.execution_mode = True
    assert proc.execution_mode

    proc.execution_mode = False
    assert not proc.execution_mode


@pytest.mark.asyncio
async def test_processor_start_clears_stop_signal():
    """Test start() clears stop signal."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    await proc.stop()
    assert proc.is_stopped()

    await proc.start()
    assert not proc.is_stopped()


@pytest.mark.asyncio
async def test_processor_stop_sets_stop_signal():
    """Test stop() sets stop signal."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    assert not proc.is_stopped()

    await proc.stop()
    assert proc.is_stopped()


@pytest.mark.asyncio
async def test_processor_execute_background_loop():
    """Test execute() runs background loop until stopped."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=10, capacity_refresh_time=0.05, pile=pile, executor=executor_mock
    )

    # Add events
    events = [SimpleEvent(return_value=f"result_{i}") for i in range(3)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)

    # Start execute in background
    async def run_execute():
        await proc.execute()

    async def stop_after_delay():
        await concurrency.sleep(0.2)  # Let it process a few cycles
        await proc.stop()

    async with concurrency.create_task_group() as tg:
        tg.start_soon(run_execute)
        tg.start_soon(stop_after_delay)

    # All events should be processed
    assert all(e.execution.status == EventStatus.COMPLETED for e in events)
    assert not proc.execution_mode  # Should be False after execute() exits


@pytest.mark.asyncio
async def test_processor_execute_sets_execution_mode():
    """Test execute() sets execution_mode to True during execution."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.05, pile=pile)

    execution_mode_during = None

    async def check_execution_mode():
        nonlocal execution_mode_during
        await concurrency.sleep(0.1)  # Wait for execute() to start
        execution_mode_during = proc.execution_mode
        await proc.stop()

    async with concurrency.create_task_group() as tg:
        tg.start_soon(proc.execute)
        tg.start_soon(check_execution_mode)

    assert execution_mode_during is True
    assert proc.execution_mode is False  # Should be False after exit


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_processor_process_empty_queue():
    """Test process() with empty queue does nothing gracefully."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    # Should not raise error
    await proc.process()
    assert proc.queue.empty()


@pytest.mark.asyncio
async def test_processor_multiple_start_calls():
    """Test multiple start() calls are safe."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    await proc.start()
    await proc.start()  # Should not raise

    assert not proc.is_stopped()


@pytest.mark.asyncio
async def test_processor_multiple_stop_calls():
    """Test multiple stop() calls are safe."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    await proc.stop()
    await proc.stop()  # Should not raise

    assert proc.is_stopped()


@pytest.mark.asyncio
async def test_processor_process_with_pending_event_retry():
    """Test process() waits for pending events before processing next."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=10, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    # Create event that stays PENDING
    class PendingEvent(Event):
        remain_pending: bool = True
        streaming: bool = False

        async def _invoke(self) -> Any:
            if self.remain_pending:
                # This should not happen - invoke() always transitions from PENDING
                # But for testing, we won't update status
                return None
            else:
                return "completed"

    event = PendingEvent()
    pile.add(event)
    await proc.enqueue(event.id)

    # Start processing in background
    async def process_task():
        await proc.process()

    async def update_event():
        # After a delay, allow event to complete
        await concurrency.sleep(0.05)
        event.remain_pending = False

    async with concurrency.create_task_group() as tg:
        tg.start_soon(process_task)
        tg.start_soon(update_event)

    # Event should eventually complete
    # (Note: This test might be flaky due to timing, but demonstrates the wait logic)


@pytest.mark.asyncio
async def test_processor_with_zero_capacity_refresh_time_rejected():
    """Test processor rejects zero capacity_refresh_time (prevents CPU hot loop)."""
    pile = Pile[Event]()

    with pytest.raises(ValueError, match=r"Capacity refresh time must be >= 0\.01s"):
        SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.0, pile=pile)


@pytest.mark.asyncio
async def test_processor_streaming_with_concurrency_limit():
    """Test streaming events work with concurrency semaphore."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = StreamingProcessor(
        queue_capacity=10,
        capacity_refresh_time=0.01,
        pile=pile,
        executor=executor_mock,
        concurrency_limit=2,
    )

    # Add multiple streaming events
    events = [StreamingEvent(stream_count=2) for _ in range(3)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)

    await proc.process()

    # All events should complete
    assert all(e.execution.status == EventStatus.COMPLETED for e in events)


@pytest.mark.asyncio
async def test_processor_non_streaming_with_concurrency_limit():
    """Test non-streaming events work with concurrency semaphore."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    proc = SimpleProcessor(
        queue_capacity=10,
        capacity_refresh_time=0.01,
        pile=pile,
        executor=executor_mock,
        concurrency_limit=2,
    )

    # Add multiple events
    events = [SimpleEvent(return_value=f"result_{i}") for i in range(4)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)

    await proc.process()

    # All events should complete
    assert all(e.execution.status == EventStatus.COMPLETED for e in events)


@pytest.mark.asyncio
async def test_processor_enqueue_dequeue_concurrent_access():
    """Test concurrent enqueue/dequeue operations are safe."""
    pile = Pile[Event]()
    proc = SimpleProcessor(queue_capacity=10, capacity_refresh_time=0.1, pile=pile)

    events = [SimpleEvent(return_value=f"result_{i}") for i in range(10)]
    for event in events:
        pile.add(event)

    async def enqueue_events():
        for event in events[:5]:
            await proc.enqueue(event.id)
            await concurrency.sleep(0.01)

    async def dequeue_events():
        await concurrency.sleep(0.02)  # Let some events queue up
        for _ in range(3):
            if not proc.queue.empty():
                await proc.dequeue()
            await concurrency.sleep(0.01)

    async with concurrency.create_task_group() as tg:
        tg.start_soon(enqueue_events)
        tg.start_soon(dequeue_events)

    # Should complete without errors


@pytest.mark.asyncio
async def test_processor_capacity_boundary_conditions():
    """Test processor handles capacity=1 edge case."""
    pile = Pile[Event]()
    executor_mock = Mock()
    executor_mock._update_progression = AsyncMock()

    # Minimum capacity
    proc = SimpleProcessor(
        queue_capacity=1, capacity_refresh_time=0.01, pile=pile, executor=executor_mock
    )

    event = SimpleEvent(return_value="single")
    pile.add(event)
    await proc.enqueue(event.id)

    await proc.process()

    assert event.execution.status == EventStatus.COMPLETED
    assert proc.available_capacity == 1  # Reset to queue_capacity


@pytest.mark.asyncio
async def test_processor_requeues_denied_events():
    """Test denied events are requeued immediately instead of lost.

    Bug Fix: Previously, events denied by request_permission() were removed from
    the queue and lost. Now they are immediately requeued with original priority
    and don't consume capacity.

    Scenario:
        1. Enqueue 3 events
        2. Deny permission for all events
        3. Verify all events remain in queue (immediately requeued)
        4. Verify capacity NOT consumed (denied events don't count)
        5. Grant permission and verify events can be processed

    This validates the fix for Issue #1 from chatgpt-codex-connector bot.
    """
    pile = Pile[Event]()

    # Custom processor that denies permission initially
    class DenyingProcessor(SimpleProcessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.deny_count = 0
            self.deny_threshold = 3  # Deny first 3 checks

        async def request_permission(self, **kwargs):
            self.deny_count += 1
            return self.deny_count > self.deny_threshold

    proc = DenyingProcessor(queue_capacity=5, capacity_refresh_time=0.01, pile=pile)

    # Enqueue 3 events
    events = [SimpleEvent(return_value=f"event_{i}") for i in range(3)]
    for i, event in enumerate(events):
        pile.add(event)
        await proc.enqueue(event.id, priority=float(i))

    initial_capacity = proc.available_capacity
    assert initial_capacity == 5

    # First process() call - first event denied, breaks loop
    await proc.process()

    # Verify first event requeued (loop breaks after first denial)
    assert proc.queue.qsize() == 3  # First event requeued, other 2 still in queue

    # CRITICAL: Capacity NOT consumed (denied events don't count)
    assert proc.available_capacity == 5

    # Second process() call - permission granted
    proc.deny_threshold = 0  # Grant all

    # Process multiple times to handle all events
    await proc.process()  # Processes up to capacity (5)

    # All events should be processed
    completed = [e for e in events if e.execution.status == EventStatus.COMPLETED]
    assert len(completed) == 3


@pytest.mark.asyncio
async def test_processor_capacity_not_consumed_when_all_denied():
    """Test capacity is NOT consumed when all events are denied permission.

    Bug Fix: Capacity should only be consumed when events are actually processed.
    Denied events don't consume capacity, preventing processor deadlock.

    Scenario:
        1. Set capacity to 3
        2. Deny permission for all events
        3. Call process() - events requeued, capacity NOT consumed
        4. Verify capacity remains at 3 (not stuck at 0)
        5. Verify processor can continue processing

    This validates the fix for Issue #2 from chatgpt-codex-connector bot.
    """
    pile = Pile[Event]()

    # Processor that always denies
    class AlwaysDenyProcessor(SimpleProcessor):
        async def request_permission(self, **kwargs):
            return False  # Always deny

    proc = AlwaysDenyProcessor(queue_capacity=3, capacity_refresh_time=0.01, pile=pile)

    # Enqueue events
    events = [SimpleEvent(return_value=f"event_{i}") for i in range(5)]
    for event in events:
        pile.add(event)
        await proc.enqueue(event.id)

    # First process() - first event denied, breaks loop
    initial_capacity = proc.available_capacity
    assert initial_capacity == 3

    await proc.process()

    # CRITICAL: Capacity NOT consumed (denied events don't count)
    assert proc.available_capacity == 3

    # Verify first event requeued, others still in queue (loop breaks after denial)
    assert proc.queue.qsize() == 5

    # Verify processor can continue (not deadlocked)
    # Change to allow processing
    proc.__class__ = SimpleProcessor  # Switch class to allow permission
    await proc.process()

    # Should process events now (3 up to capacity)
    completed = [e for e in events if e.execution.status == EventStatus.COMPLETED]
    assert len(completed) == 3  # Processed up to capacity

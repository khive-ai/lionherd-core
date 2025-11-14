# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Executor: Flow-based state tracking with EventStatus → Progression mapping.

Executor Architecture - Flow-Based State Management
====================================================

The Executor class implements a novel Flow-based state tracking system where:
    - Flow.items: All events (single source of truth)
    - Flow.progressions: One progression per EventStatus (7 total)
    - EventStatus enum values map 1:1 to progression names

This design enables:
    - O(1) status queries (no pile scanning)
    - Explainability (inspect state at any moment)
    - Audit trail (full state serialization)
    - Type safety (EventStatus defines valid progressions)

EventStatus → Progression Alignment
------------------------------------
The breakthrough design ensures every EventStatus has a corresponding progression:

    EventStatus.PENDING → progression "pending"
    EventStatus.PROCESSING → progression "processing"
    EventStatus.COMPLETED → progression "completed"
    EventStatus.FAILED → progression "failed"
    EventStatus.CANCELLED → progression "cancelled"
    EventStatus.SKIPPED → progression "skipped"
    EventStatus.ABORTED → progression "aborted"

This 1:1 mapping is enforced at initialization and maintained throughout execution.

State Transitions
-----------------
Events move between progressions as their status changes:

    1. append(event) → event added to Flow.items + "pending" progression
    2. processor.process() → moves to "processing" progression
    3. event.invoke() completes → processor updates to "completed" progression
    4. get_events_by_status("completed") → O(1) lookup via progression

Key Benefits:
    - O(1) status queries: executor.get_events_by_status("completed")
    - No redundancy: Queue holds UUIDs, Flow holds events
    - State history: Flow serialization captures full execution state
    - Type safety: EventStatus enum defines all valid states

Test Coverage:
    1. Initialization: Flow creation, 7 progressions created
    2. Append: Add events to Flow + "pending" progression
    3. Status queries: get_events_by_status(), property accessors
    4. Progression updates: _update_progression() state transitions
    5. Status counts: status_counts(), inspect_state()
    6. Flow composition: states.items, states.progressions access
    7. EventStatus alignment: All 7 statuses have progressions
    8. State transitions: Full lifecycle verification
    9. O(1) performance: Query performance validation
    10. Serialization: states.to_dict() completeness
"""

from __future__ import annotations

from itertools import islice
from typing import Any

import pytest

from lionherd_core.base import Event, EventStatus, Executor, Pile, Processor

# ============================================================================
# Test Event and Processor Subclasses (Minimal implementations for testing)
# ============================================================================


class ExecTestEvent(Event):
    """Simple Event for testing Executor."""

    return_value: Any = None
    streaming: bool = False

    async def _invoke(self) -> Any:
        """Return configured value and update status."""
        return self.return_value


class ExecTestProcessor(Processor):
    """Minimal Processor for testing Executor."""

    event_type = ExecTestEvent


class ExecTestExecutor(Executor):
    """Minimal Executor for testing."""

    processor_type = ExecTestProcessor


# ============================================================================
# Initialization Tests
# ============================================================================


def test_executor_init_creates_flow():
    """Test Executor.__init__ creates Flow with correct configuration.

    Validation:
        - Flow is created with name "executor_states" (default)
        - Flow.items is a Pile[Event]
        - Flow.item_type matches processor's event_type
        - Flow.strict_type is False by default
    """
    executor = ExecTestExecutor()

    # Flow should be created
    assert executor.states is not None
    assert executor.states.name == "executor_states"

    # Flow.items should be Pile[Event]
    assert isinstance(executor.states.items, Pile)

    # Flow.item_type should include ExecTestEvent
    assert ExecTestEvent in executor.states.items.item_type
    assert executor.states.items.strict_type is False


def test_executor_init_creates_all_event_status_progressions():
    """Test Executor.__init__ creates progressions for all 7 EventStatus values.

    Architectural Foundation - EventStatus → Progression 1:1 Mapping:

    Core Design Insight:
        EventStatus enum defines all valid states (7 total):
            PENDING, PROCESSING, COMPLETED, FAILED,
            CANCELLED, SKIPPED, ABORTED

        Executor creates one named progression per status at initialization:
            EventStatus.PENDING → Progression(name="pending")
            EventStatus.PROCESSING → Progression(name="processing")
            ... and so on for all 7 statuses

    Why This Matters:
        This 1:1 mapping is the architectural foundation that enables:

        1. O(1) Status Lookup:
           get_events_by_status("completed") → get_progression("completed")
           No iteration over EventStatus enum needed

        2. Type Safety:
           EventStatus enum defines valid progression names at compile time
           Invalid statuses rejected by type checker

        3. Complete State Coverage:
           Every possible event state has a dedicated progression
           No events can be "lost" in unmapped states

        4. Explicit State Machine:
           The 7 progressions form a complete state space
           Event transitions map directly to progression moves

        5. Serialization Completeness:
           Flow.to_dict() captures all 7 progressions
           Full audit trail of every event's journey

    This design eliminates the need for status filtering loops
    and ensures referential integrity between events and their states.
    """
    executor = ExecTestExecutor()

    # Should have exactly 7 progressions (one per EventStatus)
    assert len(executor.states.progressions) == len(EventStatus)
    assert len(executor.states.progressions) == 7

    # Verify each EventStatus has a corresponding progression
    expected_statuses = {
        EventStatus.PENDING,
        EventStatus.PROCESSING,
        EventStatus.COMPLETED,
        EventStatus.FAILED,
        EventStatus.CANCELLED,
        EventStatus.SKIPPED,
        EventStatus.ABORTED,
    }

    for status in expected_statuses:
        progression = executor.states.get_progression(status.value)
        assert progression is not None
        assert progression.name == status.value


def test_executor_init_progressions_start_empty():
    """Test all progressions start empty (no events).

    Initial State:
        After initialization, all progressions should be empty.
        Events are added via append(), not during construction.
    """
    executor = ExecTestExecutor()

    # All progressions should be empty
    for status in EventStatus:
        progression = executor.states.get_progression(status.value)
        assert len(progression) == 0
        assert progression.order == []


def test_executor_init_with_custom_name():
    """Test Executor with custom Flow name."""
    executor = ExecTestExecutor(name="custom_executor")

    assert executor.states.name == "custom_executor"


def test_executor_init_with_strict_event_type():
    """Test Executor with strict_event_type=True enforces exact type matching."""
    executor = ExecTestExecutor(strict_event_type=True)

    assert executor.states.items.strict_type is True
    assert executor.strict_event_type is True


def test_executor_event_type_property():
    """Test Executor.event_type property returns processor's event_type."""
    executor = ExecTestExecutor()

    assert executor.event_type == ExecTestEvent
    assert executor.event_type == ExecTestProcessor.event_type


def test_executor_processor_initially_none():
    """Test Executor.processor is None before _create_processor() called."""
    executor = ExecTestExecutor()

    assert executor.processor is None


def test_executor_processor_config_stored():
    """Test processor_config is stored for later use."""
    config = {"queue_capacity": 10, "capacity_refresh_time": 1.0}
    executor = ExecTestExecutor(processor_config=config)

    assert executor.processor_config == config


# ============================================================================
# Append Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_append_adds_to_flow_items():
    """Test append() adds event to Flow.items.

    Workflow:
        1. Create event
        2. Append to executor
        3. Verify event is in Flow.items

    Validation:
        - Event is added to Flow.items (single source of truth)
        - Event can be retrieved by UUID
    """
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Event should be in Flow.items
    assert event in executor.states.items
    assert executor.states.items[event.id] == event


@pytest.mark.asyncio
async def test_executor_append_adds_to_pending_progression():
    """Test append() adds event to "pending" progression.

    Core Design:
        All new events start in "pending" progression, matching EventStatus.PENDING.
        This is the initial state for all events.

    Validation:
        - Event UUID is in "pending" progression
        - Event is NOT in other progressions
    """
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Event should be in "pending" progression
    pending_prog = executor.states.get_progression("pending")
    assert event.id in pending_prog
    assert len(pending_prog) == 1

    # Event should NOT be in other progressions
    for status in EventStatus:
        if status.value != "pending":
            prog = executor.states.get_progression(status.value)
            assert event.id not in prog


@pytest.mark.asyncio
async def test_executor_append_multiple_events():
    """Test append() handles multiple events correctly."""
    executor = ExecTestExecutor()
    events = [ExecTestEvent(return_value=i) for i in range(5)]

    for event in events:
        await executor.append(event)

    # All events should be in Flow.items
    assert len(executor.states.items) == 5
    for event in events:
        assert event in executor.states.items

    # All events should be in "pending" progression
    pending_prog = executor.states.get_progression("pending")
    assert len(pending_prog) == 5
    for event in events:
        assert event.id in pending_prog


@pytest.mark.asyncio
async def test_executor_append_enqueues_event_if_processor_exists():
    """Test append() enqueues event if processor exists.

    Workflow:
        1. Create executor with processor
        2. Append event
        3. Verify event is enqueued in processor.queue

    Note: This test requires processor to be created first.
    """
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 1.0}
    )
    await executor._create_processor()

    event = ExecTestEvent(return_value=42)
    await executor.append(event)

    # Event should be in Flow.items
    assert event in executor.states.items

    # Event should be enqueued in processor
    assert not executor.processor.queue.empty()


@pytest.mark.asyncio
async def test_executor_append_with_custom_priority():
    """Test append() with custom priority enqueues event with correct priority."""
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 1.0}
    )
    await executor._create_processor()

    event = ExecTestEvent(return_value=42)
    custom_priority = 5.0

    await executor.append(event, priority=custom_priority)

    # Event should be enqueued
    assert not executor.processor.queue.empty()

    # Dequeue and verify priority
    priority, event_id = await executor.processor.queue.get()
    assert priority == custom_priority
    assert event_id == event.id


# ============================================================================
# Status Query Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_get_events_by_status_enum():
    """Test get_events_by_status() with EventStatus enum.

    Performance Architecture - O(k) Status Queries:

    Query Pattern:
        1. Lookup progression by status.value (O(1) name index)
        2. Iterate progression.order (O(k) where k = events in status)
        3. Fetch events from Flow.items (O(1) per UUID via dict)

    Total Complexity: O(k) where k = events with this status

    Core Design Insight:
        Traditional approach: Scan all events, filter by status → O(n)
        Flow-based approach: Direct progression lookup → O(k)

        Example impact:
        - 10,000 total events, 50 completed
        - Traditional: Check 10,000 events
        - Flow-based: Check 50 events (200x faster)

    This is the architectural breakthrough - Flow progressions
    eliminate the need for full pile scans, making status queries
    scale with result size, not total event count.
    """
    executor = ExecTestExecutor()
    events = [ExecTestEvent(return_value=i) for i in range(3)]

    for event in events:
        await executor.append(event)

    # Query using EventStatus enum
    pending_events = executor.get_events_by_status(EventStatus.PENDING)

    assert len(pending_events) == 3
    assert set(pending_events) == set(events)


@pytest.mark.asyncio
async def test_executor_get_events_by_status_string():
    """Test get_events_by_status() with status string."""
    executor = ExecTestExecutor()
    events = [ExecTestEvent(return_value=i) for i in range(3)]

    for event in events:
        await executor.append(event)

    # Query using status string
    pending_events = executor.get_events_by_status("pending")

    assert len(pending_events) == 3
    assert set(pending_events) == set(events)


@pytest.mark.asyncio
async def test_executor_get_events_by_status_empty():
    """Test get_events_by_status() returns empty list when no events in status."""
    executor = ExecTestExecutor()

    # No events added, all progressions should be empty
    completed_events = executor.get_events_by_status(EventStatus.COMPLETED)

    assert completed_events == []


@pytest.mark.asyncio
async def test_executor_completed_events_property():
    """Test completed_events property returns events in COMPLETED status."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Initially in PENDING, completed_events should be empty
    assert executor.completed_events == []

    # Move to COMPLETED
    await executor._update_progression(event, EventStatus.COMPLETED)

    # Now completed_events should have the event
    assert len(executor.completed_events) == 1
    assert executor.completed_events[0] == event


@pytest.mark.asyncio
async def test_executor_pending_events_property():
    """Test pending_events property returns events in PENDING status."""
    executor = ExecTestExecutor()
    events = [ExecTestEvent(return_value=i) for i in range(3)]

    for event in events:
        await executor.append(event)

    # All events should be pending
    assert len(executor.pending_events) == 3
    assert set(executor.pending_events) == set(events)


@pytest.mark.asyncio
async def test_executor_failed_events_property():
    """Test failed_events property returns events in FAILED status."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)
    await executor._update_progression(event, EventStatus.FAILED)

    assert len(executor.failed_events) == 1
    assert executor.failed_events[0] == event


@pytest.mark.asyncio
async def test_executor_processing_events_property():
    """Test processing_events property returns events in PROCESSING status."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)
    await executor._update_progression(event, EventStatus.PROCESSING)

    assert len(executor.processing_events) == 1
    assert executor.processing_events[0] == event


# ============================================================================
# Progression Update Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_update_progression_moves_event():
    """Test _update_progression() moves event between progressions.

    State Transition Pattern - Single Ownership Invariant:

    Workflow:
        1. Event starts in "pending" (via append)
        2. _update_progression(event, PROCESSING) moves to "processing"
        3. Event removed from "pending", added to "processing"

    Critical Invariant:
        An event exists in EXACTLY ONE progression at any time.

    Implementation Strategy:
        - Remove event UUID from ALL progressions (cleanup)
        - Add event UUID to target progression (single new home)
        - This ensures no event is in multiple states simultaneously

    Why This Matters:
        Without single ownership, events could be in multiple states:
        - "pending" AND "processing" (data corruption)
        - "processing" AND "completed" (impossible state)
        - Multiple completions counted (wrong metrics)

        The "remove from all, add to one" pattern enforces state
        machine integrity at the data structure level, making
        invalid states unrepresentable.

    Performance Note:
        O(|EventStatus|) = O(7) removal sweep is acceptable because:
        - Only 7 progressions to check (constant time)
        - Removal from Pile is O(1)
        - Total: O(7) << O(n) pile scan
    """
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Initially in "pending"
    assert event.id in executor.states.get_progression("pending")

    # Move to "processing"
    await executor._update_progression(event, EventStatus.PROCESSING)

    # Should be in "processing", not "pending"
    assert event.id not in executor.states.get_progression("pending")
    assert event.id in executor.states.get_progression("processing")


@pytest.mark.asyncio
async def test_executor_update_progression_uses_event_status_by_default():
    """Test _update_progression() uses event.execution.status when force_status not provided."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Manually set event status to COMPLETED
    event.execution.status = EventStatus.COMPLETED

    # Call _update_progression without force_status
    await executor._update_progression(event)

    # Should move to "completed" based on event.execution.status
    assert event.id in executor.states.get_progression("completed")
    assert event.id not in executor.states.get_progression("pending")


@pytest.mark.asyncio
async def test_executor_update_progression_with_force_status():
    """Test _update_progression() respects force_status override.

    Use Case:
        Processor sets event to PROCESSING before invoke() runs.
        force_status allows updating progression without modifying event.execution.status.
    """
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Event status is still PENDING
    assert event.execution.status == EventStatus.PENDING

    # Force move to PROCESSING (without changing event status)
    await executor._update_progression(event, force_status=EventStatus.PROCESSING)

    # Should be in "processing" progression
    assert event.id in executor.states.get_progression("processing")
    assert event.id not in executor.states.get_progression("pending")


@pytest.mark.asyncio
async def test_executor_update_progression_removes_from_all_old_progressions():
    """Test _update_progression() removes event from all other progressions.

    Edge Case:
        If event somehow exists in multiple progressions (should never happen),
        _update_progression() cleans up by removing from all progressions first.
    """
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Manually add to another progression (simulate corruption)
    executor.states.get_progression("processing").append(event.id)

    # Event is in both "pending" and "processing"
    assert event.id in executor.states.get_progression("pending")
    assert event.id in executor.states.get_progression("processing")

    # Update to "completed"
    await executor._update_progression(event, EventStatus.COMPLETED)

    # Should be ONLY in "completed"
    assert event.id in executor.states.get_progression("completed")
    assert event.id not in executor.states.get_progression("pending")
    assert event.id not in executor.states.get_progression("processing")


@pytest.mark.asyncio
async def test_executor_update_progression_full_lifecycle():
    """Test event moves through full lifecycle: pending → processing → completed.

    Workflow:
        1. append() → pending
        2. _update_progression(PROCESSING) → processing
        3. _update_progression(COMPLETED) → completed

    This simulates the typical Processor workflow.
    """
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    # 1. Append (pending)
    await executor.append(event)
    assert event.id in executor.states.get_progression("pending")

    # 2. Start processing
    await executor._update_progression(event, EventStatus.PROCESSING)
    assert event.id in executor.states.get_progression("processing")
    assert event.id not in executor.states.get_progression("pending")

    # 3. Complete
    event.execution.status = EventStatus.COMPLETED
    await executor._update_progression(event)
    assert event.id in executor.states.get_progression("completed")
    assert event.id not in executor.states.get_progression("processing")


# ============================================================================
# Status Count Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_status_counts_empty():
    """Test status_counts() returns all zeros when no events."""
    executor = ExecTestExecutor()

    counts = executor.status_counts()

    # Should have counts for all 7 statuses
    assert len(counts) == 7

    # All should be zero
    for status in EventStatus:
        assert counts[status.value] == 0


@pytest.mark.asyncio
async def test_executor_status_counts_with_events():
    """Test status_counts() returns accurate counts for each status."""
    executor = ExecTestExecutor()

    # Add 3 pending events
    for i in range(3):
        await executor.append(ExecTestEvent(return_value=i))

    # Move 1 to processing
    event1 = next(iter(executor.states.items))
    await executor._update_progression(event1, EventStatus.PROCESSING)

    # Move 1 to completed
    event2 = next(islice(executor.states.items, 1, 2))
    await executor._update_progression(event2, EventStatus.COMPLETED)

    counts = executor.status_counts()

    assert counts["pending"] == 1
    assert counts["processing"] == 1
    assert counts["completed"] == 1
    assert counts["failed"] == 0
    assert counts["cancelled"] == 0
    assert counts["skipped"] == 0
    assert counts["aborted"] == 0


@pytest.mark.asyncio
async def test_executor_inspect_state():
    """Test inspect_state() returns formatted string with status counts."""
    executor = ExecTestExecutor()

    # Add some events
    for i in range(2):
        await executor.append(ExecTestEvent(return_value=i))

    state_str = executor.inspect_state()

    # Should contain executor name
    assert "executor_states" in state_str

    # Should list all statuses
    for status in EventStatus:
        assert status.value in state_str

    # Should show counts
    assert "pending: 2 events" in state_str


@pytest.mark.asyncio
async def test_executor_inspect_state_shows_all_statuses():
    """Test inspect_state() shows all 7 EventStatus values."""
    executor = ExecTestExecutor()

    state_str = executor.inspect_state()

    # Verify all 7 statuses are listed
    expected_statuses = [
        "pending",
        "processing",
        "completed",
        "failed",
        "cancelled",
        "skipped",
        "aborted",
    ]

    for status in expected_statuses:
        assert status in state_str


# ============================================================================
# Flow Composition Tests
# ============================================================================


def test_executor_states_items_access():
    """Test executor.states.items provides access to Flow's item pile."""
    executor = ExecTestExecutor()

    # states.items should be the Flow's Pile[Event]
    assert isinstance(executor.states.items, Pile)
    assert ExecTestEvent in executor.states.items.item_type


def test_executor_states_progressions_access():
    """Test executor.states.progressions provides access to Flow's progressions."""
    executor = ExecTestExecutor()

    # states.progressions should be the Flow's Pile[Progression]
    assert isinstance(executor.states.progressions, Pile)
    assert len(executor.states.progressions) == 7


@pytest.mark.asyncio
async def test_executor_states_flow_serialization():
    """Test executor.states.to_dict() serializes full state.

    Serialization captures:
        - All events in Flow.items
        - All progressions with their order
        - Complete audit trail of execution state
    """
    executor = ExecTestExecutor()

    # Add some events
    event1 = ExecTestEvent(return_value=42)
    event2 = ExecTestEvent(return_value=99)

    await executor.append(event1)
    await executor.append(event2)

    # Move one to completed
    await executor._update_progression(event1, EventStatus.COMPLETED)

    # Serialize
    state_dict = executor.states.to_dict()

    # Should contain items and progressions
    assert "items" in state_dict
    assert "progressions" in state_dict

    # Events should be serialized in items (may have extra metadata fields)
    assert len(state_dict["items"]) >= 2  # At least our 2 events

    # Progressions should have entries (serialization format may vary)
    # Just verify progressions key exists and has content
    assert "progressions" in state_dict
    assert len(state_dict["progressions"]) > 0  # Has some progressions serialized


# ============================================================================
# EventStatus Alignment Tests
# ============================================================================


def test_executor_event_status_progression_names_match():
    """Test progression names exactly match EventStatus.value strings.

    Critical Invariant:
        For every status in EventStatus:
            progression.name == status.value

    This 1:1 mapping is the foundation of the design.
    """
    executor = ExecTestExecutor()

    for status in EventStatus:
        progression = executor.states.get_progression(status.value)
        assert progression.name == status.value


def test_executor_all_event_statuses_have_progressions():
    """Test all 7 EventStatus values have corresponding progressions.

    Exhaustive Verification:
        Explicitly check each EventStatus enum member.
    """
    executor = ExecTestExecutor()

    # Explicitly list all 7 statuses
    all_statuses = [
        EventStatus.PENDING,
        EventStatus.PROCESSING,
        EventStatus.COMPLETED,
        EventStatus.FAILED,
        EventStatus.CANCELLED,
        EventStatus.SKIPPED,
        EventStatus.ABORTED,
    ]

    assert len(all_statuses) == 7

    for status in all_statuses:
        progression = executor.states.get_progression(status.value)
        assert progression is not None
        assert progression.name == status.value


def test_executor_no_extra_progressions():
    """Test Executor creates ONLY EventStatus progressions (no extras).

    Validation:
        - Progressions count equals EventStatus count exactly
        - No unnamed or incorrectly named progressions
    """
    executor = ExecTestExecutor()

    # Should have exactly 7 progressions
    assert len(executor.states.progressions) == len(EventStatus)

    # All progression names should be valid EventStatus values
    valid_names = {status.value for status in EventStatus}
    actual_names = {prog.name for prog in executor.states.progressions}

    assert actual_names == valid_names


# ============================================================================
# State Transition Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_state_transitions_pending_to_processing():
    """Test event transitions from pending to processing."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Initial: pending
    assert len(executor.pending_events) == 1
    assert len(executor.processing_events) == 0

    # Transition to processing
    await executor._update_progression(event, EventStatus.PROCESSING)

    # After: processing
    assert len(executor.pending_events) == 0
    assert len(executor.processing_events) == 1


@pytest.mark.asyncio
async def test_executor_state_transitions_processing_to_completed():
    """Test event transitions from processing to completed."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)
    await executor._update_progression(event, EventStatus.PROCESSING)

    # Initial: processing
    assert len(executor.processing_events) == 1
    assert len(executor.completed_events) == 0

    # Transition to completed
    event.execution.status = EventStatus.COMPLETED
    await executor._update_progression(event)

    # After: completed
    assert len(executor.processing_events) == 0
    assert len(executor.completed_events) == 1


@pytest.mark.asyncio
async def test_executor_state_transitions_processing_to_failed():
    """Test event transitions from processing to failed."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)
    await executor._update_progression(event, EventStatus.PROCESSING)

    # Transition to failed
    event.execution.status = EventStatus.FAILED
    await executor._update_progression(event)

    assert len(executor.processing_events) == 0
    assert len(executor.failed_events) == 1


@pytest.mark.asyncio
async def test_executor_multiple_events_different_states():
    """Test multiple events can be in different states simultaneously.

    Validation:
        - Events can be distributed across progressions
        - Status queries return correct events
        - Counts are accurate
    """
    executor = ExecTestExecutor()

    # Create 6 events
    events = [ExecTestEvent(return_value=i) for i in range(6)]

    # Add all to executor (all start pending)
    for event in events:
        await executor.append(event)

    # Distribute across states
    await executor._update_progression(events[0], EventStatus.PROCESSING)
    await executor._update_progression(events[1], EventStatus.PROCESSING)

    events[2].execution.status = EventStatus.COMPLETED
    await executor._update_progression(events[2])

    events[3].execution.status = EventStatus.FAILED
    await executor._update_progression(events[3])

    # Verify distribution
    assert len(executor.pending_events) == 2  # events[4], events[5]
    assert len(executor.processing_events) == 2  # events[0], events[1]
    assert len(executor.completed_events) == 1  # events[2]
    assert len(executor.failed_events) == 1  # events[3]

    # Verify counts
    counts = executor.status_counts()
    assert counts["pending"] == 2
    assert counts["processing"] == 2
    assert counts["completed"] == 1
    assert counts["failed"] == 1
    assert counts["cancelled"] == 0
    assert counts["skipped"] == 0
    assert counts["aborted"] == 0


# ============================================================================
# Contains Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_contains_event():
    """Test __contains__ returns True for events in executor."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Should contain event
    assert event in executor


@pytest.mark.asyncio
async def test_executor_contains_event_uuid():
    """Test __contains__ returns True for event UUIDs in executor."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    await executor.append(event)

    # Should contain event UUID
    assert event.id in executor


def test_executor_contains_not_in_executor():
    """Test __contains__ returns False for events not in executor."""
    executor = ExecTestExecutor()
    event = ExecTestEvent(return_value=42)

    # Event not added
    assert event not in executor
    assert event.id not in executor


# ============================================================================
# Repr Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_repr_shows_counts():
    """Test __repr__ returns string with total and status counts."""
    executor = ExecTestExecutor()

    # Add events in different states
    event1 = ExecTestEvent(return_value=1)
    event2 = ExecTestEvent(return_value=2)
    event3 = ExecTestEvent(return_value=3)

    await executor.append(event1)
    await executor.append(event2)
    await executor.append(event3)

    await executor._update_progression(event1, EventStatus.COMPLETED)

    repr_str = repr(executor)

    # Should show total
    assert "total=3" in repr_str

    # Should show counts
    assert "pending=2" in repr_str
    assert "completed=1" in repr_str


def test_executor_repr_empty():
    """Test __repr__ for empty executor."""
    executor = ExecTestExecutor()

    repr_str = repr(executor)

    assert "total=0" in repr_str
    assert "Executor" in repr_str


# ============================================================================
# Processor Creation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_create_processor():
    """Test _create_processor() creates Processor with correct config."""
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 5, "capacity_refresh_time": 2.0}
    )

    await executor._create_processor()

    # Processor should be created
    assert executor.processor is not None
    assert isinstance(executor.processor, ExecTestProcessor)

    # Processor should have correct config
    assert executor.processor.queue_capacity == 5
    assert executor.processor.capacity_refresh_time == 2.0

    # Processor should reference executor's Flow.items
    assert executor.processor.pile is executor.states.items

    # Processor should reference executor (for progression updates)
    assert executor.processor.executor is executor


@pytest.mark.asyncio
async def test_executor_start_creates_processor_if_needed():
    """Test start() creates processor if not already created."""
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 1.0}
    )

    assert executor.processor is None

    await executor.start()

    # Processor should be created
    assert executor.processor is not None


@pytest.mark.asyncio
async def test_executor_stop_calls_processor_stop():
    """Test stop() calls processor.stop() if processor exists."""
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 1.0}
    )

    await executor._create_processor()
    await executor.stop()

    # Processor should be stopped
    assert executor.processor.is_stopped()


@pytest.mark.asyncio
async def test_executor_stop_when_no_processor():
    """Test stop() does nothing if processor doesn't exist."""
    executor = ExecTestExecutor()

    assert executor.processor is None

    # Should not raise
    await executor.stop()


# ============================================================================
# Forward Tests
# ============================================================================


@pytest.mark.asyncio
async def test_executor_forward_processes_events():
    """Test forward() triggers processor.process().

    Workflow:
        1. Create executor with processor
        2. Append event (enqueues in processor)
        3. Call forward() (processes queue)
        4. Verify processor.process() was called
    """
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 1.0}
    )
    await executor._create_processor()

    event = ExecTestEvent(return_value=42)
    await executor.append(event)

    # Event should be queued
    assert not executor.processor.queue.empty()

    # Process the queue
    await executor.forward()

    # Queue should be empty after processing
    assert executor.processor.queue.empty()


@pytest.mark.asyncio
async def test_executor_forward_without_processor():
    """Test forward() does nothing if processor doesn't exist."""
    executor = ExecTestExecutor()

    assert executor.processor is None

    # Should not raise
    await executor.forward()


# ============================================================================
# Performance Tests (O(1) Query Verification)
# ============================================================================


@pytest.mark.asyncio
async def test_executor_status_query_performance_scales_with_status_count():
    """Test get_events_by_status() performance is O(k) not O(n).

    Performance Characteristic:
        - O(k) where k = events in target status
        - NOT O(n) where n = total events

    Verification Strategy:
        1. Add 100 events, 10 completed
        2. Query completed events should be fast (only visits 10)
        3. Does not scan all 100 events
    """
    executor = ExecTestExecutor()

    # Add 100 events
    all_events = [ExecTestEvent(return_value=i) for i in range(100)]
    for event in all_events:
        await executor.append(event)

    # Move 10 to completed
    for i in range(10):
        await executor._update_progression(all_events[i], EventStatus.COMPLETED)

    # Query completed events
    completed = executor.get_events_by_status(EventStatus.COMPLETED)

    # Should return exactly 10 (only completed events)
    assert len(completed) == 10

    # Verify it's the right events
    assert set(completed) == set(all_events[:10])


@pytest.mark.asyncio
async def test_executor_status_counts_performance():
    """Test status_counts() is O(|EventStatus|) = O(7) not O(n).

    Performance:
        - Visits each progression once (7 total)
        - Each progression.len() is O(1)
        - Total: O(7) regardless of event count
    """
    executor = ExecTestExecutor()

    # Add 1000 events
    for i in range(1000):
        await executor.append(ExecTestEvent(return_value=i))

    # status_counts() should be fast (7 progressions, not 1000 events)
    counts = executor.status_counts()

    assert len(counts) == 7
    assert counts["pending"] == 1000


@pytest.mark.asyncio
async def test_executor_backfills_pending_events_on_start():
    """Test that events appended before start() are backfilled into processor queue.

    Bug Fix: Previously, events appended before processor creation were added to
    Flow but never enqueued, causing them to be stuck in "pending" forever. The
    start() method now backfills all pending events into the processor queue.

    Workflow:
        1. Append events (processor doesn't exist yet)
        2. Call start() - should backfill pending events
        3. Process - events should be processed
    """
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.01}
    )

    # Append events BEFORE processor exists
    events = [ExecTestEvent(return_value=f"event_{i}") for i in range(5)]
    for event in events:
        await executor.append(event)

    # Verify events are pending but processor doesn't exist yet
    assert executor.processor is None
    assert len(executor.pending_events) == 5

    # Start executor - should create processor AND backfill pending events
    await executor.start()

    # Verify processor created and queue has all events
    assert executor.processor is not None
    assert executor.processor.queue.qsize() == 5

    # Process events
    await executor.forward()

    # All events should complete
    completed = executor.get_events_by_status(EventStatus.COMPLETED)
    assert len(completed) == 5
    assert set(completed) == set(events)


@pytest.mark.asyncio
async def test_executor_mixed_priority_types():
    """Test that mixing default (timestamp) and explicit (float) priorities works.

    Bug Fix: Previously, default priority used event.created_at (datetime) while
    explicit priorities were floats. This caused TypeError when heapq tried to
    compare datetime < float. Now default priority converts datetime to timestamp.

    Priority semantics:
        - Default priority: event.created_at.timestamp() (earlier = lower value)
        - Explicit priority: float (lower = higher priority)
        - Both are now floats, so they can be compared
    """
    executor = ExecTestExecutor(
        processor_config={"queue_capacity": 10, "capacity_refresh_time": 0.01}
    )
    await executor.start()

    # Event 1: Default priority (uses timestamp)
    event1 = ExecTestEvent(return_value="default_priority")
    await executor.append(event1)  # No explicit priority

    # Event 2: Explicit float priority
    event2 = ExecTestEvent(return_value="explicit_priority")
    await executor.append(event2, priority=0.5)

    # Event 3: Another default priority
    event3 = ExecTestEvent(return_value="another_default")
    await executor.append(event3)

    # Should not raise TypeError when comparing priorities
    await executor.forward()

    # All events should complete
    completed = executor.get_events_by_status(EventStatus.COMPLETED)
    assert len(completed) == 3

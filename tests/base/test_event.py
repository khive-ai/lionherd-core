# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Event: lifecycle tracking, Sentinel patterns, execution state, retryability.

Event/Execution System - Observability Primitives
==================================================

Events are the fundamental unit of work in lionherd, providing observability
and traceability for all operations. The Event/Execution model separates
concerns:

    Event: The operation to perform (what to do)
    Execution: The runtime state (how it went)

State Machine
-------------
Event lifecycle follows a deterministic finite state automaton:

    ┌─────────┐
    │ PENDING │  (Initial state: created but not started)
    └────┬────┘
         │ invoke()
         ▼
    ┌───────────┐
    │PROCESSING │  (Active execution: _invoke() running)
    └─────┬─────┘
          │
          ├────────┐ Success
          │        ▼
          │   ┌───────────┐
          │   │ COMPLETED │  (Terminal state: response available)
          │   └───────────┘
          │
          ├────────┐ Exception
          │        ▼
          │   ┌────────┐
          │   │ FAILED │  (Terminal state: error stored)
          │   └────────┘
          │
          └────────┐ Cancellation
                   ▼
              ┌───────────┐
              │ CANCELLED │  (Terminal state: timeout/cancel)
              └───────────┘

Transitions are irreversible - terminal states cannot move back to PENDING.
Use as_fresh_event() to create a new event with pristine state.

Observability Properties
------------------------
Every execution captures:
- **Duration**: Microsecond-precision timing (started_at → completed_at)
- **Response**: Output value (Unset if not available, None if legitimate null)
- **Error**: Exception instance (None on success, Unset if never executed)
- **Retryability**: Whether failure is transient (computed from error type)
- **Status**: Current lifecycle state (see state machine above)

Events can publish to EventBus for distributed tracing, enabling:
- Request tracing across services
- Performance profiling (duration analysis)
- Failure analysis (error patterns)
- Retry decision-making (retryability flag)

Sentinel Semantics (Critical for Caching/Memoization)
------------------------------------------------------
The system distinguishes between "no value" and "null value":

    Unset: No value state
        - Event never executed (pending)
        - Event failed (no response available)
        - Event cancelled (interrupted)
        ∴ response is unavailable

    None: Success state
        - Event completed successfully
        - Legitimate null return value
        ∴ response is None (valid result)

This distinction enables correct caching behavior:
    cache[key] = response  # Must distinguish Unset (miss) from None (hit)

Without sentinels, None would be ambiguous: "cache miss" or "cached null"?

ExceptionGroup Support (Python 3.11+)
--------------------------------------
Execution supports ExceptionGroup for parallel failure aggregation:
- Multiple errors captured in single group
- Nested groups for hierarchical errors
- Retryability computed conservatively: retryable=True only if ALL errors retryable

Retryability Logic:
    ALL retryable → group retryable=True
    ANY non-retryable → group retryable=False  (conservative: avoid wasted retries)
    Unknown exceptions → default retryable=True (optimistic: transient assumed)

Invocable Protocol
------------------
Events implement the Invocable protocol:
    async def invoke() -> Any  # Execute and return result
    async def stream() -> AsyncIterator[Any]  # Streaming execution (subclass-specific)
    @property request -> dict  # Request representation (for tracing)
    @property response -> Any  # Response value (read-only)

This enables polymorphism: anything Invocable can be executed uniformly.
"""

from __future__ import annotations

from typing import Any

import pytest

from lionherd_core.base.event import Event, EventStatus, Execution
from lionherd_core.libs.concurrency import create_task_group, fail_after

# ============================================================================
# Test Event Subclasses (Concrete implementations for testing)
# ============================================================================


class SimpleEvent(Event):
    """Simple Event that returns a value."""

    return_value: Any = None

    async def _invoke(self) -> Any:
        """Return the configured value."""
        return self.return_value


class FailingEvent(Event):
    """Event that always raises an exception."""

    error_message: str = "Test error"

    async def _invoke(self) -> Any:
        """Raise configured error."""
        raise ValueError(self.error_message)


class SlowEvent(Event):
    """Event that takes time to complete."""

    delay: float = 0.1
    return_value: Any = "completed"

    async def _invoke(self) -> Any:
        """Wait then return value."""
        import anyio

        await anyio.sleep(self.delay)
        return self.return_value


class ComplexResponseEvent(Event):
    """Event that returns complex unserializable objects."""

    async def _invoke(self) -> Any:
        """Return a complex object with circular references."""

        class CircularObject:
            def __init__(self):
                self.ref = self

        return CircularObject()


class ReturnsNoneEvent(Event):
    """Event that returns None as a legitimate value."""

    async def _invoke(self) -> Any:
        """Return None as legitimate response."""
        return None


# ============================================================================
# Event Lifecycle Tests
# ============================================================================


@pytest.mark.asyncio
async def test_event_lifecycle_success():
    """Test Event transitions through pending→processing→completed on success.

    State Transitions (Happy Path):
        1. PENDING → PROCESSING: invoke() called, _invoke() starts
        2. PROCESSING → COMPLETED: _invoke() returns successfully

    Execution State Changes:
        - response: Unset → actual_value (captured return value)
        - error: Unset → None (explicitly no error)
        - duration: Unset → float (microseconds elapsed)

    Observability: This test validates the complete observability contract -
    every successful execution must capture timing, response, and error status.
    """
    from lionherd_core.types._sentinel import Unset

    event = SimpleEvent(return_value=42)

    # Initial state
    assert event.status == EventStatus.PENDING
    assert event.execution.response is Unset
    assert event.execution.duration is Unset
    assert event.execution.error is Unset

    # Invoke
    result = await event.invoke()

    # Final state
    assert result == 42
    assert event.status == EventStatus.COMPLETED
    assert event.execution.response == 42
    assert event.execution.duration is not None
    assert event.execution.duration >= 0
    assert event.execution.error is None


@pytest.mark.asyncio
async def test_event_lifecycle_failure():
    """Test Event transitions to FAILED on exception.

    State Transitions (Failure Path):
        1. PENDING → PROCESSING: invoke() called, _invoke() starts
        2. PROCESSING → FAILED: _invoke() raises exception

    Execution State Changes:
        - response: Unset → Unset (no response due to failure)
        - error: Unset → Exception (captured exception instance)
        - duration: Unset → float (time until failure)
        - retryable: computed from error type (True for unknown exceptions)

    Error Handling: invoke() catches exceptions and returns None, preventing
    propagation. The exception is stored in execution.error for analysis.
    This enables graceful degradation and retry decision-making.
    """
    from lionherd_core.types._sentinel import Unset

    event = FailingEvent(error_message="Custom error")

    assert event.status == EventStatus.PENDING

    # Invoke catches exception, returns None, sets status to FAILED
    result = await event.invoke()

    # Check failure state
    assert result is None
    assert event.status == EventStatus.FAILED
    assert isinstance(event.execution.error, ValueError)
    assert str(event.execution.error) == "Custom error"
    assert event.execution.response is Unset
    assert event.execution.duration is not None
    assert event.execution.retryable is True  # Unknown exceptions are retryable by default


@pytest.mark.asyncio
async def test_event_lifecycle_cancellation():
    """Test Event transitions to CANCELLED when operation is cancelled via timeout.

    State Transitions (Cancellation Path):
        1. PENDING → PROCESSING: invoke() called, _invoke() starts
        2. PROCESSING → CANCELLED: timeout/cancel signal received

    Execution State Changes:
        - response: Unset → Unset (no response due to cancellation)
        - error: Unset → CancelledError (cancellation exception)
        - duration: Unset → float (time until cancellation)
        - retryable: True (timeouts/cancellations are transient failures)

    Cancellation Semantics: Timeouts are treated as retryable failures
    (network latency, resource contention). The system distinguishes
    cancellation from permanent failures, enabling intelligent retry strategies.
    """
    event = SlowEvent(delay=5.0)

    assert event.status == EventStatus.PENDING

    # Cancel event during execution via timeout
    with pytest.raises(TimeoutError), fail_after(0.1):
        await event.invoke()

    # Check cancelled state
    assert event.status == EventStatus.CANCELLED
    assert event.execution.error is not None  # CancelledError exception stored
    assert event.execution.duration is not None
    assert event.execution.retryable is True  # Timeouts are retryable


@pytest.mark.asyncio
async def test_event_sets_processing_status_before_invoke():
    """Test that status transitions to PROCESSING before _invoke() runs."""
    processing_status_seen = []

    class StatusCheckEvent(Event):
        async def _invoke(self) -> Any:
            # Capture status during execution
            processing_status_seen.append(self.status)
            return "done"

    event = StatusCheckEvent()
    assert event.status == EventStatus.PENDING

    await event.invoke()

    # Status should have been PROCESSING during _invoke()
    assert EventStatus.PROCESSING in processing_status_seen
    assert event.status == EventStatus.COMPLETED


# ============================================================================
# Execution Tracking Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execution_tracks_duration():
    """Test that execution duration is measured and non-zero.

    Observability Requirement: Duration tracking enables:
    - Performance profiling (identify slow operations)
    - SLA monitoring (detect violations)
    - Anomaly detection (unusual latency spikes)
    - Resource optimization (focus on high-impact operations)

    Precision: Microsecond-level timing using high-resolution monotonic clock
    (time.perf_counter), ensuring accurate measurements even for fast operations.
    """
    event = SlowEvent(delay=0.05)  # 50ms delay

    await event.invoke()

    assert event.execution.duration is not None
    assert event.execution.duration >= 0.05  # At least the delay time
    assert event.execution.duration < 1.0  # Reasonable upper bound


@pytest.mark.asyncio
async def test_execution_tracks_response():
    """Test that execution response is captured correctly.

    Response Tracking: Captures the actual return value of _invoke() for:
    - Result verification (correctness validation)
    - Caching/memoization (avoid redundant computation)
    - Debugging (inspect actual outputs)
    - Testing (assert expected results)

    The response property provides read-only access to execution.response,
    encapsulating internal state while exposing observability data.
    """
    test_response = {"key": "value", "number": 123}
    event = SimpleEvent(return_value=test_response)

    result = await event.invoke()

    assert result == test_response
    assert event.execution.response == test_response
    assert event.response == test_response  # Property accessor


@pytest.mark.asyncio
async def test_execution_tracks_error_message():
    """Test that error messages are captured in execution state."""
    error_msg = "Detailed error information"
    event = FailingEvent(error_message=error_msg)

    result = await event.invoke()

    assert result is None
    assert event.status == EventStatus.FAILED
    assert isinstance(event.execution.error, ValueError)
    assert str(event.execution.error) == error_msg
    assert event.execution.retryable is True


@pytest.mark.asyncio
async def test_execution_duration_set_on_cancellation():
    """Test that duration is set even when event is cancelled via timeout."""
    event = SlowEvent(delay=10.0)

    with pytest.raises(TimeoutError), fail_after(0.1):
        await event.invoke()

    # Duration should be set despite cancellation
    assert event.execution.duration is not None
    assert event.execution.duration >= 0
    assert event.execution.retryable is True  # Timeout is retryable


@pytest.mark.asyncio
async def test_execution_duration_set_on_failure():
    """Test that duration is set even when event fails."""
    event = FailingEvent()

    result = await event.invoke()

    # Duration should be set despite failure
    assert result is None
    assert event.status == EventStatus.FAILED
    assert event.execution.duration is not None
    assert event.execution.duration >= 0
    assert event.execution.retryable is True


# ============================================================================
# Status Property Tests
# ============================================================================


def test_status_property_accepts_string():
    """Test status property can be set with string value."""
    event = SimpleEvent()
    event.status = "completed"
    assert event.status == EventStatus.COMPLETED


def test_status_property_accepts_enum():
    """Test status property can be set with EventStatus enum."""
    event = SimpleEvent()
    event.status = EventStatus.FAILED
    assert event.status == EventStatus.FAILED


def test_status_property_rejects_invalid():
    """Test status property rejects invalid values."""
    event = SimpleEvent()
    with pytest.raises(ValueError, match="Invalid status type"):
        event.status = 123


# ============================================================================
# Response Property Tests
# ============================================================================


def test_response_property_getter():
    """Test response property getter (read-only)."""
    from lionherd_core.types._sentinel import Unset

    event = SimpleEvent()
    assert event.response is Unset  # Changed from None - sentinel default

    # Response is set internally by invoke, not externally
    test_value = {"test": "data"}
    event.execution.response = test_value
    assert event.response == test_value
    assert event.execution.response == test_value


# ============================================================================
# Serialization Tests (Execution.to_dict)
# ============================================================================


@pytest.mark.asyncio
async def test_execution_serialization_simple_types():
    """Test Execution.to_dict() handles simple types correctly.

    Serialization Contract: to_dict() produces JSON-serializable output for:
    - Logging/monitoring (send to observability backend)
    - API responses (expose execution state to clients)
    - Persistence (store execution history in database)
    - Testing (assert serialization format)

    All sentinel values (Unset) map to None, with status field providing context.
    """
    event = SimpleEvent(return_value=42)
    await event.invoke()

    serialized = event.execution.to_dict()

    assert serialized["status"] == "completed"
    assert serialized["response"] == 42
    assert serialized["error"] is None
    assert isinstance(serialized["duration"], float)


@pytest.mark.asyncio
async def test_execution_serialization_complex_json_serializable():
    """Test Execution.to_dict() handles complex JSON-serializable objects."""
    response = {"nested": {"data": [1, 2, 3]}, "flag": True}
    event = SimpleEvent(return_value=response)
    await event.invoke()

    serialized = event.execution.to_dict()

    assert serialized["response"] == response


@pytest.mark.asyncio
async def test_execution_serialization_unserializable_response():
    """Test Execution.to_dict() marks unserializable responses appropriately."""
    event = ComplexResponseEvent()
    await event.invoke()

    serialized = event.execution.to_dict()

    # Unserializable objects should be marked
    assert serialized["response"] == "<unserializable>"


@pytest.mark.asyncio
async def test_execution_serialization_with_error():
    """Test Execution.to_dict() includes error message when failed."""
    event = FailingEvent(error_message="Serialization test error")

    result = await event.invoke()

    assert result is None
    assert event.status == EventStatus.FAILED

    serialized = event.execution.to_dict()

    assert serialized["status"] == "failed"
    # Error is serialized as dict with type and message
    assert serialized["error"]["error"] == "ValueError"
    assert serialized["error"]["message"] == "Serialization test error"
    assert serialized["response"] is None
    assert serialized["retryable"] is True


def test_execution_to_dict_without_invocation():
    """Test Execution.to_dict() works for pending events."""
    execution = Execution()

    serialized = execution.to_dict()

    assert serialized["status"] == "pending"
    assert serialized["response"] is None
    assert serialized["error"] is None
    assert serialized["duration"] is None


# ============================================================================
# Event Serialization Tests (Event.to_dict)
# ============================================================================


@pytest.mark.asyncio
async def test_event_to_dict_includes_execution():
    """Test Event.to_dict() includes serialized execution state."""
    event = SimpleEvent(return_value="test")
    await event.invoke()

    serialized = event.to_dict()

    assert "execution" in serialized
    assert serialized["execution"]["status"] == "completed"
    assert serialized["execution"]["response"] == "test"


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_event_captures_exception_in_execution_state():
    """Test that exceptions from _invoke() are caught and stored in execution state."""

    class CustomError(Exception):
        pass

    class CustomFailingEvent(Event):
        async def _invoke(self) -> Any:
            raise CustomError("Custom error")

    event = CustomFailingEvent()

    result = await event.invoke()

    assert result is None
    assert event.status == EventStatus.FAILED
    assert isinstance(event.execution.error, CustomError)
    assert str(event.execution.error) == "Custom error"
    assert event.execution.retryable is True  # Unknown exceptions are retryable


@pytest.mark.asyncio
async def test_event_handles_none_response():
    """Test Event correctly handles None as a valid response."""
    event = SimpleEvent(return_value=None)
    result = await event.invoke()

    assert result is None
    assert event.response is None
    assert event.status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_concurrent_event_invocations():
    """Test multiple Events can execute concurrently without interference."""
    events = [SimpleEvent(return_value=i) for i in range(5)]

    async with create_task_group() as tg:
        for event in events:
            tg.start_soon(event.invoke)

    # All should complete successfully
    for i, event in enumerate(events):
        assert event.status == EventStatus.COMPLETED
        assert event.response == i


# ============================================================================
# Event Retry (as_fresh_event) Tests
# ============================================================================


@pytest.mark.asyncio
async def test_as_fresh_event_preserves_config():
    """Test as_fresh_event creates clone with pristine execution but same config.

    Retry Pattern: as_fresh_event() enables retry logic without mutating original:
        1. Execute event → fails (FAILED state with error)
        2. Check retryable flag → decide whether to retry
        3. Create fresh event → pristine PENDING state, same configuration
        4. Execute fresh event → independent execution state

    State Isolation: Fresh event has new identity (id, created_at), ensuring
    independent tracking. Original event remains unchanged for audit/analysis.

    Configuration Preservation: All config fields (return_value, delay, etc.)
    are copied, ensuring retry executes the same operation.
    """
    # Create and execute event
    original = SimpleEvent(return_value=42)
    await original.invoke()

    assert original.status == EventStatus.COMPLETED
    assert original.response == 42

    # Create fresh event
    fresh = original.as_fresh_event()

    # Fresh event has new identity
    assert fresh.id != original.id
    assert fresh.created_at != original.created_at

    # Fresh event has pristine execution state
    from lionherd_core.types._sentinel import Unset

    assert fresh.status == EventStatus.PENDING
    assert fresh.execution.response is Unset
    assert fresh.execution.error is Unset
    assert fresh.execution.duration is Unset

    # Fresh event preserves configuration
    assert fresh.return_value == original.return_value
    assert fresh.return_value == 42

    # Fresh event is functional
    result = await fresh.invoke()
    assert result == 42
    assert fresh.status == EventStatus.COMPLETED


@pytest.mark.asyncio
async def test_as_fresh_event_with_copy_meta():
    """Test as_fresh_event copies metadata when copy_meta=True."""
    # Create event with metadata
    original = SimpleEvent(return_value="test")
    original.metadata["user_id"] = "user123"
    original.metadata["attempt"] = 1
    await original.invoke()

    # Create fresh event with metadata copy
    fresh = original.as_fresh_event(copy_meta=True)

    # Metadata is copied
    assert fresh.metadata["user_id"] == "user123"
    assert fresh.metadata["attempt"] == 1

    # Original tracking is added
    assert "original" in fresh.metadata
    assert fresh.metadata["original"]["id"] == str(original.id)
    assert fresh.metadata["original"]["created_at"] == original.created_at

    # Modifications to fresh metadata don't affect original
    fresh.metadata["attempt"] = 2
    assert original.metadata["attempt"] == 1
    assert fresh.metadata["attempt"] == 2


def test_as_fresh_event_without_copy_meta():
    """Test as_fresh_event does not copy metadata by default."""
    # Create event with metadata
    original = SimpleEvent(return_value="test")
    original.metadata["user_id"] = "user123"
    original.metadata["attempt"] = 1

    # Create fresh event without copy_meta (default behavior)
    fresh = original.as_fresh_event()

    # Original metadata is not copied
    assert "user_id" not in fresh.metadata
    assert "attempt" not in fresh.metadata

    # But original tracking is still added
    assert "original" in fresh.metadata
    assert fresh.metadata["original"]["id"] == str(original.id)
    assert fresh.metadata["original"]["created_at"] == original.created_at


# ============================================================================
# Abstract Method Tests
# ============================================================================


@pytest.mark.asyncio
async def test_base_event_invoke_not_implemented():
    """Test that base Event._invoke() raises NotImplementedError."""
    event = Event()

    with pytest.raises(NotImplementedError, match="Subclasses must implement _invoke"):
        await event._invoke()


@pytest.mark.asyncio
async def test_event_stream_not_implemented():
    """Test that base Event.stream() raises NotImplementedError."""
    event = Event()

    with pytest.raises(NotImplementedError, match="Subclasses must implement stream"):
        await event.stream()


# ============================================================================
# Request Property Tests
# ============================================================================


def test_event_request_property_default():
    """Test that base Event.request returns empty dict by default."""
    event = SimpleEvent()
    assert event.request == {}


def test_event_request_property_override():
    """Test that subclasses can override request property."""

    class EventWithRequest(Event):
        @property
        def request(self) -> dict:
            return {"method": "GET", "url": "/test"}

        async def _invoke(self) -> Any:
            return "done"

    event = EventWithRequest()
    assert event.request == {"method": "GET", "url": "/test"}


# ============================================================================
# Sentinel Value Coverage Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execution_response_unset_vs_none():
    """Test comprehensive sentinel behavior: Unset vs None.

    Two distinct response states must be differentiated:
    1. Unset: No value state - event pending/failed/cancelled (no response available)
    2. None: Success state - legitimate null return value (successful completion)

    Why This Matters:
    ----------------
    Without sentinels, None is ambiguous in caching/memoization:
        cache.get(key) → None  # Cache miss? Or cached null value?

    With sentinels, we have unambiguous semantics:
        cache.get(key) → Unset  # Cache miss (never computed)
        cache.get(key) → None   # Cache hit (computed result is null)

    This distinction is critical for:
    - Caching: Distinguish "not yet computed" from "computed as null"
    - Optional fields: Know if field was set (even to None) or never touched
    - Database queries: Distinguish "no result" from "result is null"
    - API responses: Omit unset fields vs include null fields

    Serialization Trade-off: Both serialize to None in to_dict(), but status
    field provides context (pending/failed/cancelled vs completed).
    """
    from lionherd_core.types._sentinel import Unset

    # Test 1: Unset - pristine state (never executed)
    pristine_event = SimpleEvent()
    assert pristine_event.execution.response is Unset
    assert pristine_event.status == EventStatus.PENDING

    # Test 2: Unset - failed state (no response due to failure)
    failed_event = FailingEvent()
    await failed_event.invoke()
    assert failed_event.execution.response is Unset
    assert failed_event.status == EventStatus.FAILED

    # Test 3: None - success with legitimate null value
    none_event = ReturnsNoneEvent()
    result = await none_event.invoke()
    assert result is None
    assert none_event.execution.response is None
    assert none_event.status == EventStatus.COMPLETED

    # All serialize to None in to_dict()
    pristine_dict = pristine_event.execution.to_dict()
    failed_dict = failed_event.execution.to_dict()
    none_dict = none_event.execution.to_dict()

    assert pristine_dict["response"] is None
    assert failed_dict["response"] is None
    assert none_dict["response"] is None

    # Status field distinguishes the cases
    assert pristine_dict["status"] == "pending"
    assert failed_dict["status"] == "failed"
    assert none_dict["status"] == "completed"


@pytest.mark.asyncio
async def test_execution_error_field_lifecycle():
    """Test error field transitions: Unset → None → Exception.

    Error field lifecycle:
    1. Initial state: Unset (never executed)
    2. Success: None (explicitly no error)
    3. Failure: Exception instance
    """
    from lionherd_core.types._sentinel import Unset

    # Initial state: error is Unset
    event = SimpleEvent(return_value="test")
    assert event.execution.error is Unset
    assert event.status == EventStatus.PENDING

    # Success: error is None (explicitly no error)
    await event.invoke()
    assert event.execution.error is None
    assert event.status == EventStatus.COMPLETED

    # Failure: error is Exception instance
    failing_event = FailingEvent(error_message="Test error")
    await failing_event.invoke()
    assert isinstance(failing_event.execution.error, ValueError)
    assert str(failing_event.execution.error) == "Test error"
    assert failing_event.status == EventStatus.FAILED


@pytest.mark.asyncio
async def test_execution_serialization_preserves_sentinel_semantics():
    """Test that to_dict() correctly handles all sentinel states.

    Serialization strategy:
    - Unset → None (with status="pending" or "failed" or "cancelled")
    - None → None (with status="completed")
    - status field provides necessary context to interpret None values
    """
    from lionherd_core.types._sentinel import Unset

    # Case 1: Unset state (pristine/pending)
    pristine = SimpleEvent()
    assert pristine.execution.response is Unset
    pristine_serialized = pristine.execution.to_dict()
    assert pristine_serialized["response"] is None
    assert pristine_serialized["status"] == "pending"
    assert pristine_serialized["error"] is None

    # Case 2: Unset state (failed - no response due to failure)
    failed = FailingEvent()
    await failed.invoke()
    assert failed.execution.response is Unset
    failed_serialized = failed.execution.to_dict()
    assert failed_serialized["response"] is None
    assert failed_serialized["status"] == "failed"
    assert failed_serialized["error"] is not None  # Error dict present

    # Case 3: None state (legitimate success with null)
    none_success = ReturnsNoneEvent()
    result = await none_success.invoke()
    assert result is None
    assert none_success.execution.response is None
    none_serialized = none_success.execution.to_dict()
    assert none_serialized["response"] is None
    assert none_serialized["status"] == "completed"
    assert none_serialized["error"] is None

    # Verify all serialize response to None but status distinguishes them
    assert pristine_serialized["response"] is None
    assert failed_serialized["response"] is None
    assert none_serialized["response"] is None
    assert pristine_serialized["status"] != failed_serialized["status"]
    assert failed_serialized["status"] != none_serialized["status"]


# ============================================================================
# ExceptionGroup Support Tests (Python 3.11+)
# ============================================================================


class MultiErrorEvent(Event):
    """Event that raises ExceptionGroup with multiple errors."""

    async def _invoke(self) -> Any:
        """Raise ExceptionGroup with multiple exceptions."""
        errors = [
            ValueError("validation error"),
            TypeError("type mismatch"),
            KeyError("missing key"),
        ]
        raise ExceptionGroup("multiple validation errors", errors)


class NestedExceptionGroupEvent(Event):
    """Event that raises nested ExceptionGroups."""

    async def _invoke(self) -> Any:
        """Raise nested ExceptionGroups."""
        validation_errors = ExceptionGroup(
            "validation errors",
            [ValueError("invalid input"), TypeError("wrong type")],
        )
        processing_errors = ExceptionGroup(
            "processing errors",
            [KeyError("missing key"), RuntimeError("process failed")],
        )
        raise ExceptionGroup("operation failed", [validation_errors, processing_errors])


@pytest.mark.asyncio
async def test_event_captures_exception_group():
    """Test Event correctly captures and stores ExceptionGroup."""
    event = MultiErrorEvent()

    result = await event.invoke()

    # Check failure state
    assert result is None
    assert event.status == EventStatus.FAILED
    assert isinstance(event.execution.error, ExceptionGroup)

    # Verify ExceptionGroup contents
    eg = event.execution.error
    assert len(eg.exceptions) == 3
    assert isinstance(eg.exceptions[0], ValueError)
    assert isinstance(eg.exceptions[1], TypeError)
    assert isinstance(eg.exceptions[2], KeyError)


@pytest.mark.asyncio
async def test_exception_group_serialization():
    """Test ExceptionGroup serializes correctly with nested exceptions."""
    event = MultiErrorEvent()
    await event.invoke()

    serialized = event.execution.to_dict()

    # Check top-level structure
    assert serialized["status"] == "failed"
    assert serialized["error"]["error"] == "ExceptionGroup"
    assert "multiple validation errors" in serialized["error"]["message"]

    # Check nested exceptions array
    exceptions = serialized["error"]["exceptions"]
    assert len(exceptions) == 3
    assert exceptions[0]["error"] == "ValueError"
    assert exceptions[0]["message"] == "validation error"
    assert exceptions[1]["error"] == "TypeError"
    assert exceptions[1]["message"] == "type mismatch"
    assert exceptions[2]["error"] == "KeyError"
    assert exceptions[2]["message"] == "'missing key'"


@pytest.mark.asyncio
async def test_nested_exception_group_serialization():
    """Test nested ExceptionGroups serialize recursively."""
    event = NestedExceptionGroupEvent()
    await event.invoke()

    serialized = event.execution.to_dict()

    # Check top level
    assert serialized["error"]["error"] == "ExceptionGroup"
    assert "operation failed" in serialized["error"]["message"]

    # Check nested groups
    exceptions = serialized["error"]["exceptions"]
    assert len(exceptions) == 2

    # First nested group: validation errors
    validation_group = exceptions[0]
    assert validation_group["error"] == "ExceptionGroup"
    assert "validation errors" in validation_group["message"]
    assert len(validation_group["exceptions"]) == 2
    assert validation_group["exceptions"][0]["error"] == "ValueError"
    assert validation_group["exceptions"][1]["error"] == "TypeError"

    # Second nested group: processing errors
    processing_group = exceptions[1]
    assert processing_group["error"] == "ExceptionGroup"
    assert "processing errors" in processing_group["message"]
    assert len(processing_group["exceptions"]) == 2
    assert processing_group["exceptions"][0]["error"] == "KeyError"
    assert processing_group["exceptions"][1]["error"] == "RuntimeError"


def test_execution_add_error_single():
    """Test add_error() with single error sets error field directly."""
    from lionherd_core.types._sentinel import Unset

    execution = Execution()
    assert execution.error is Unset

    # Add first error
    execution.add_error(ValueError("test error"))

    # Should be stored as single exception, not group
    assert isinstance(execution.error, ValueError)
    assert not isinstance(execution.error, ExceptionGroup)
    assert str(execution.error) == "test error"


def test_execution_add_error_multiple():
    """Test add_error() with multiple errors creates ExceptionGroup."""
    execution = Execution()

    # Add multiple errors
    execution.add_error(ValueError("error 1"))
    execution.add_error(TypeError("error 2"))
    execution.add_error(KeyError("error 3"))

    # Should be stored as ExceptionGroup
    assert isinstance(execution.error, ExceptionGroup)
    assert len(execution.error.exceptions) == 3
    assert isinstance(execution.error.exceptions[0], ValueError)
    assert isinstance(execution.error.exceptions[1], TypeError)
    assert isinstance(execution.error.exceptions[2], KeyError)


def test_execution_add_error_extends_existing_group():
    """Test add_error() extends existing ExceptionGroup."""
    execution = Execution()

    # Create initial group manually
    execution.error = ExceptionGroup(
        "initial errors",
        [ValueError("error 1"), TypeError("error 2")],
    )

    # Add another error
    execution.add_error(KeyError("error 3"))

    # Should extend the group
    assert isinstance(execution.error, ExceptionGroup)
    assert len(execution.error.exceptions) == 3
    assert isinstance(execution.error.exceptions[2], KeyError)


def test_execution_add_error_after_none():
    """Test add_error() after error is None (reset state)."""
    execution = Execution()
    execution.error = None  # Simulate success state

    # Add error to reset state
    execution.add_error(ValueError("new error"))

    # Should set as single error
    assert isinstance(execution.error, ValueError)
    assert str(execution.error) == "new error"


@pytest.mark.asyncio
async def test_exception_group_serialization_in_event_to_dict():
    """Test Event.to_dict() includes serialized ExceptionGroup."""
    event = MultiErrorEvent()
    await event.invoke()

    event_dict = event.to_dict()

    # Check execution is serialized
    assert "execution" in event_dict
    assert event_dict["execution"]["status"] == "failed"

    # Check ExceptionGroup structure
    error = event_dict["execution"]["error"]
    assert error["error"] == "ExceptionGroup"
    assert len(error["exceptions"]) == 3


def test_execution_add_error_preserves_order():
    """Test add_error() preserves error order."""
    execution = Execution()

    # Add errors in specific order
    execution.add_error(ValueError("first"))
    execution.add_error(TypeError("second"))
    execution.add_error(RuntimeError("third"))

    # Check order is preserved
    eg = execution.error
    assert str(eg.exceptions[0]) == "first"
    assert str(eg.exceptions[1]) == "second"
    assert str(eg.exceptions[2]) == "third"


@pytest.mark.asyncio
async def test_exception_group_retryable_all_logic():
    """Test ExceptionGroup retryable logic: retryable=True only if ALL errors are retryable.

    Design Decision: Conservative retryability for ExceptionGroups.
    - If ALL errors are retryable → group is retryable
    - If even ONE error is non-retryable → group is non-retryable
    - Unknown exceptions (non-LionherdError) default to retryable=True

    Why Conservative (ANY non-retryable → False)?
    -----------------------------------------------
    Parallel operations can fail with mixed causes:
        Task A: NetworkError (retryable - transient network issue)
        Task B: ValidationError (non-retryable - bad input data)
        Task C: TimeoutError (retryable - resource contention)

    Optimistic approach (ALL non-retryable → False):
        ❌ Would retry despite bad input in Task B
        ❌ Wastes resources retrying fundamentally broken operation
        ❌ May cause cascading failures (invalid data persists)

    Conservative approach (ANY non-retryable → False):
        ✅ Prevents retry when at least one failure is permanent
        ✅ Requires fixing root cause (ValidationError) before retry
        ✅ Avoids wasted retry attempts on broken operations

    This trades false negatives (missed retry opportunities) for false positive
    prevention (avoiding harmful retries). Better to require manual intervention
    than automate futile retries.
    """
    from lionherd_core.errors import (
        ConnectionError,
        ExecutionError,
        LionherdError,
        ValidationError,
    )

    # Case 1: ALL errors retryable → group retryable=True
    class AllRetryableEvent(Event):
        async def _invoke(self) -> Any:
            errors = [
                ConnectionError("network error"),  # default_retryable=True
                ExecutionError("execution failed"),  # default_retryable=True
                LionherdError("generic error", retryable=True),
            ]
            raise ExceptionGroup("all retryable errors", errors)

    event1 = AllRetryableEvent()
    await event1.invoke()

    assert event1.status == EventStatus.FAILED
    assert isinstance(event1.execution.error, ExceptionGroup)
    assert event1.execution.retryable is True  # ALL retryable → True

    # Case 2: Mixed (some retryable, some not) → group retryable=False
    class MixedRetryableEvent(Event):
        async def _invoke(self) -> Any:
            errors = [
                ConnectionError("network error"),  # retryable=True
                ValidationError("bad input"),  # default_retryable=False (!)
                ExecutionError("execution failed"),  # retryable=True
            ]
            raise ExceptionGroup("mixed retryable errors", errors)

    event2 = MixedRetryableEvent()
    await event2.invoke()

    assert event2.status == EventStatus.FAILED
    assert isinstance(event2.execution.error, ExceptionGroup)
    assert event2.execution.retryable is False  # ONE non-retryable → False

    # Case 3: ALL errors non-retryable → group retryable=False
    class AllNonRetryableEvent(Event):
        async def _invoke(self) -> Any:
            errors = [
                ValidationError("validation failed 1"),  # default_retryable=False
                ValidationError("validation failed 2"),  # default_retryable=False
                LionherdError("explicit non-retryable", retryable=False),
            ]
            raise ExceptionGroup("all non-retryable errors", errors)

    event3 = AllNonRetryableEvent()
    await event3.invoke()

    assert event3.status == EventStatus.FAILED
    assert isinstance(event3.execution.error, ExceptionGroup)
    assert event3.execution.retryable is False  # ALL non-retryable → False

    # Case 4: Unknown exceptions mixed with retryable → retryable=True
    # (unknown exceptions default to retryable=True)
    class UnknownExceptionEvent(Event):
        async def _invoke(self) -> Any:
            errors = [
                ValueError("unknown stdlib exception"),  # Not LionherdError → defaults True
                ConnectionError("network error"),  # retryable=True
                RuntimeError("another unknown"),  # Not LionherdError → defaults True
            ]
            raise ExceptionGroup("mixed with unknown", errors)

    event4 = UnknownExceptionEvent()
    await event4.invoke()

    assert event4.status == EventStatus.FAILED
    assert isinstance(event4.execution.error, ExceptionGroup)
    assert event4.execution.retryable is True  # All unknown/retryable → True


@pytest.mark.asyncio
async def test_exception_group_backward_compatibility():
    """Test single exceptions still work as before (backward compatibility)."""
    # Single exception event (existing behavior)
    single_error_event = FailingEvent(error_message="single error")
    await single_error_event.invoke()

    # Should work exactly as before
    assert single_error_event.status == EventStatus.FAILED
    assert isinstance(single_error_event.execution.error, ValueError)
    assert str(single_error_event.execution.error) == "single error"

    # Serialization should work as before
    serialized = single_error_event.execution.to_dict()
    assert serialized["error"]["error"] == "ValueError"
    assert serialized["error"]["message"] == "single error"
    assert "exceptions" not in serialized["error"]  # No exceptions array for single error

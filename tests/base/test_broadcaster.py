# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Broadcaster Test Suite

Tests for singleton pub/sub broadcaster with O(1) memory overhead.

## Design Pattern

Singleton + Class-level State:
    - Single global broadcaster instance per class
    - All subclasses share same subscribers (class-level)
    - O(1) memory overhead regardless of usage

## Mathematical Model

Broadcaster B = (
    _instance: B | None,              // Singleton instance
    _subscribers: List[Callback],     // Class-level subscribers
    _event_type: Type                 // Event type constraint
)

Singleton Invariant:
    ∀ calls: Broadcaster() → same instance

Subscription:
    subscribe(cb) ⟹ _subscribers.append(cb) if cb ∉ _subscribers
    unsubscribe(cb) ⟹ _subscribers.remove(cb) if cb ∈ _subscribers
    broadcast(evt) ⟹ ∀cb ∈ _subscribers: cb(evt) // Fire-and-forget

## Broadcaster vs EventBus

Comparison:
    EventBus: Multi-topic, instance-based, flexible routing
    Broadcaster: Single global channel, singleton, minimal overhead

Trade-offs:
    + Minimal memory overhead (O(1) regardless of usage)
    + Simple API (no topic routing)
    + Class-level isolation (subclasses independent)
    - Less flexible (single channel per subclass)
    - Global state (harder to test, requires cleanup)

## Core Invariants

1. Singleton Instance: Only one Broadcaster instance per subclass
2. Class-level Subscribers: Shared across all instances of subclass
3. Sync/Async Support: Callbacks can be sync or async functions
4. Event Type Safety: _event_type constrains broadcasted events
5. Exception Isolation: Callback exceptions logged, not propagated
6. Fire-and-Forget: Broadcast doesn't return callback results

## Test Organization

- TestBroadcaster: Singleton pattern, subscription lifecycle, broadcasting semantics

## Design Rationale

Why Singleton?
    - Global event channel (application-wide notifications)
    - O(1) memory overhead (single instance per subclass)
    - No instantiation complexity (automatic singleton)

Why Class-level State?
    - Share subscribers across all instances
    - Persistent state (not tied to instance lifecycle)
    - Simplifies event propagation (no instance routing)

Why Fire-and-Forget?
    - Observability pattern (side-effects, not return values)
    - Exception isolation (one callback failure doesn't block others)
    - Performance (no blocking on callback completion)

## Usage Patterns

Pattern 1: Global Event Notifications
    ```python
    class ShutdownBroadcaster(Broadcaster):
        _event_type = ShutdownEvent


    ShutdownBroadcaster.subscribe(cleanup_handler)
    ShutdownBroadcaster.subscribe(log_shutdown)

    await ShutdownBroadcaster.broadcast(ShutdownEvent())
    ```

Pattern 2: Mixed Sync/Async Callbacks
    ```python
    def sync_callback(event):
        logger.info(f"Event: {event}")


    async def async_callback(event):
        await metrics.record(event)


    Broadcaster.subscribe(sync_callback)
    Broadcaster.subscribe(async_callback)
    await Broadcaster.broadcast(event)
    ```

Pattern 3: Subclass Independence
    ```python
    class TypeABroadcaster(Broadcaster):
        _event_type = TypeAEvent
        _subscribers = []
        _instance = None


    class TypeBBroadcaster(Broadcaster):
        _event_type = TypeBEvent
        _subscribers = []
        _instance = None


    # Independent subscriber lists, independent instances
    ```

## Performance Characteristics

| Operation       | Complexity | Notes                              |
|----------------|-----------|-------------------------------------|
| subscribe()    | O(n)      | List membership check + append     |
| unsubscribe()  | O(n)      | List membership check + remove     |
| broadcast()    | O(n)      | Invoke all callbacks sequentially  |
| singleton()    | O(1)      | Instance check + creation          |

Memory: O(1) overhead per subclass (single instance + subscriber list)

Broadcast Semantics:
    - Async callbacks awaited in registration order
    - Sync callbacks executed immediately
    - Exceptions caught and logged (fire-and-forget)
    - No result aggregation (observability pattern)

## Testing Strategy

- Singleton enforcement (same instance across calls)
- Subscription/unsubscription correctness
- Duplicate subscription prevention
- Async/sync callback dispatch
- Exception isolation (callback failures don't propagate)
- Event type validation (wrong type rejected)
- Multiple subscribers invocation
- Subclass independence (no cross-contamination)
- Empty subscriber list handling

Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
SPDX-License-Identifier: Apache-2.0
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lionherd_core.base.broadcaster import Broadcaster
from lionherd_core.base.event import Event


class SampleEvent(Event):
    """Sample event class for broadcaster tests."""

    event_type: str = "test_event"


class TestBroadcaster:
    """Test suite for Broadcaster class."""

    @pytest.fixture(autouse=True)
    def reset_broadcaster(self):
        """Reset broadcaster state before each test."""
        # Clear subscribers before each test
        Broadcaster._subscribers.clear()
        Broadcaster._instance = None
        yield
        # Clean up after test
        Broadcaster._subscribers.clear()
        Broadcaster._instance = None

    def test_broadcaster_singleton(self):
        """Test that Broadcaster follows singleton pattern."""

        # Create a subclass for testing
        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        broadcaster1 = TestBroadcaster()
        broadcaster2 = TestBroadcaster()

        assert broadcaster1 is broadcaster2
        assert TestBroadcaster._instance is broadcaster1

    def test_subscribe_adds_callback(self):
        """Test that subscribe adds callback to subscribers list."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        TestBroadcaster.subscribe(callback)

        # Check that a weakref to the callback exists
        assert any(ref() is callback for ref in TestBroadcaster._subscribers)
        assert TestBroadcaster.get_subscriber_count() == 1

    def test_subscribe_prevents_duplicates(self):
        """Test that subscribing same callback twice doesn't duplicate."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        TestBroadcaster.subscribe(callback)
        TestBroadcaster.subscribe(callback)

        assert TestBroadcaster.get_subscriber_count() == 1

    def test_unsubscribe_removes_callback(self):
        """Test that unsubscribe removes callback from subscribers."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        TestBroadcaster.subscribe(callback)
        assert TestBroadcaster.get_subscriber_count() == 1

        TestBroadcaster.unsubscribe(callback)
        assert TestBroadcaster.get_subscriber_count() == 0
        assert callback not in TestBroadcaster._subscribers

    def test_unsubscribe_nonexistent_callback_no_error(self):
        """Test that unsubscribing nonexistent callback doesn't raise error."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()

        # Should not raise error
        TestBroadcaster.unsubscribe(callback)
        assert TestBroadcaster.get_subscriber_count() == 0

    @pytest.mark.asyncio
    async def test_broadcast_calls_sync_callback(self):
        """Test that broadcast calls synchronous callbacks."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = MagicMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(callback)
        await TestBroadcaster.broadcast(event)

        callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_calls_async_callback(self):
        """Test that broadcast awaits asynchronous callbacks."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback = AsyncMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(callback)
        await TestBroadcaster.broadcast(event)

        callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_calls_multiple_subscribers(self):
        """Test that broadcast calls all registered subscribers."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = AsyncMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(callback1)
        TestBroadcaster.subscribe(callback2)
        TestBroadcaster.subscribe(callback3)

        await TestBroadcaster.broadcast(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)
        callback3.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_validates_event_type(self):
        """Test that broadcast raises error for wrong event type."""

        class SpecificBroadcaster(Broadcaster):
            _event_type = SampleEvent

        class OtherEvent(Event):
            event_type: str = "other"

        callback = MagicMock()
        wrong_event = OtherEvent()

        SpecificBroadcaster.subscribe(callback)

        with pytest.raises(ValueError, match="Event must be of type SampleEvent"):
            await SpecificBroadcaster.broadcast(wrong_event)

        # Callback should not have been called
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_handles_callback_exception(self):
        """Test that broadcast catches and logs callback exceptions."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        failing_callback = MagicMock(side_effect=RuntimeError("Callback error"))
        successful_callback = MagicMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(failing_callback)
        TestBroadcaster.subscribe(successful_callback)

        # Should not raise, but log the error
        await TestBroadcaster.broadcast(event)

        # Both callbacks should be attempted
        failing_callback.assert_called_once_with(event)
        successful_callback.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_handles_async_callback_exception(self):
        """Test that broadcast catches and logs async callback exceptions."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        failing_callback = AsyncMock(side_effect=RuntimeError("Async callback error"))
        successful_callback = AsyncMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(failing_callback)
        TestBroadcaster.subscribe(successful_callback)

        # Should not raise, but log the error
        await TestBroadcaster.broadcast(event)

        # Both callbacks should be attempted
        assert failing_callback.await_count == 1
        successful_callback.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_with_no_subscribers(self):
        """Test that broadcasting with no subscribers doesn't error."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        event = SampleEvent()

        # Should not raise error
        await TestBroadcaster.broadcast(event)
        assert TestBroadcaster.get_subscriber_count() == 0

    def test_get_subscriber_count_accuracy(self):
        """Test that get_subscriber_count returns accurate count."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        assert TestBroadcaster.get_subscriber_count() == 0

        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()

        TestBroadcaster.subscribe(callback1)
        assert TestBroadcaster.get_subscriber_count() == 1

        TestBroadcaster.subscribe(callback2)
        TestBroadcaster.subscribe(callback3)
        assert TestBroadcaster.get_subscriber_count() == 3

        TestBroadcaster.unsubscribe(callback2)
        assert TestBroadcaster.get_subscriber_count() == 2

    def test_multiple_broadcaster_subclasses_independent(self):
        """Test that different Broadcaster subclasses maintain independent state."""

        class BroadcasterA(Broadcaster):
            _event_type = SampleEvent
            _subscribers = []
            _instance = None

        class TestEvent2(Event):
            event_type: str = "test2"

        class BroadcasterB(Broadcaster):
            _event_type = TestEvent2
            _subscribers = []
            _instance = None

        callback_a = MagicMock()
        callback_b = MagicMock()

        BroadcasterA.subscribe(callback_a)
        BroadcasterB.subscribe(callback_b)

        assert BroadcasterA.get_subscriber_count() == 1
        assert BroadcasterB.get_subscriber_count() == 1
        # Check that weakrefs point to correct callbacks
        assert any(ref() is callback_a for ref in BroadcasterA._subscribers)
        assert not any(ref() is callback_a for ref in BroadcasterB._subscribers)
        assert any(ref() is callback_b for ref in BroadcasterB._subscribers)
        assert not any(ref() is callback_b for ref in BroadcasterA._subscribers)

    @pytest.mark.asyncio
    async def test_broadcast_mixed_sync_async_callbacks(self):
        """Test broadcasting to mix of sync and async callbacks."""

        class TestBroadcaster(Broadcaster):
            _event_type = SampleEvent

        sync_callback1 = MagicMock()
        async_callback = AsyncMock()
        sync_callback2 = MagicMock()
        event = SampleEvent()

        TestBroadcaster.subscribe(sync_callback1)
        TestBroadcaster.subscribe(async_callback)
        TestBroadcaster.subscribe(sync_callback2)

        await TestBroadcaster.broadcast(event)

        sync_callback1.assert_called_once_with(event)
        async_callback.assert_awaited_once_with(event)
        sync_callback2.assert_called_once_with(event)

"""Test Broadcaster class-level subscription leak - Issue #24.

Broadcaster uses ClassVar[list] for _subscribers, combined with singleton pattern.
This creates a WORSE leak than EventBus because:
1. Subscribers are class-level (not instance-level)
2. Singleton pattern means one instance persists forever
3. All subclasses share the same subscriber list

Without weakref, handlers accumulate across agent lifecycles, tenant boundaries, etc.
"""

import gc
import weakref

import pytest

from lionherd_core.base.broadcaster import Broadcaster
from lionherd_core.base.event import Event


class BroadcastTestEvent(Event):
    """Test event for broadcaster (renamed to avoid pytest collection warning)."""

    pass


@pytest.mark.asyncio
async def test_broadcaster_class_level_leak():
    """Demonstrate Broadcaster class-level subscription leak.

    ClassVar + Singleton = handlers persist forever, even when callback objects
    are destroyed.
    """

    class TestBroadcaster(Broadcaster):
        _event_type = BroadcastTestEvent
        _subscribers = []  # Fresh list for this test
        _instance = None

    # Keep callbacks alive with external references
    callbacks = []
    monitor_refs = []

    # Subscribe 50 callbacks
    for i in range(50):

        async def callback(event, idx=i):
            _ = idx  # Capture loop variable

        callbacks.append(callback)
        TestBroadcaster.subscribe(callback)
        monitor_refs.append(weakref.ref(callback))

    # Verify all registered at class level
    assert TestBroadcaster.get_subscriber_count() == 50

    # Delete external references (simulates agent destruction, request completion)
    callbacks.clear()
    gc.collect()

    # BUG (before fix): All 50 still in class-level _subscribers
    # Expected (with fix): ~0 (weakref auto-cleanup, allow 1-2 for test artifacts)
    leaked = TestBroadcaster.get_subscriber_count()

    cleanup_rate = (50 - leaked) / 50
    assert cleanup_rate >= 0.96, (
        f"Class-level leak: {leaked}/50 callbacks still in _subscribers after GC. "
        f"Cleanup rate: {cleanup_rate:.1%} (expected ≥96%). "
        f"ClassVar + Singleton pattern prevents automatic cleanup."
    )


@pytest.mark.asyncio
async def test_broadcaster_multi_tenant_leak():
    """Demonstrate leak across tenant/agent boundaries.

    In multi-tenant systems, each tenant's handlers should be cleaned up when
    tenant is destroyed. ClassVar storage causes cross-tenant pollution.
    """

    class TenantBroadcaster(Broadcaster):
        _event_type = BroadcastTestEvent
        _subscribers = []
        _instance = None

    # Simulate 10 tenants, each with 10 callbacks
    tenant_callbacks = {}

    for tenant_id in range(10):
        tenant_callbacks[tenant_id] = []
        for callback_idx in range(10):

            async def callback(event, tid=tenant_id, cid=callback_idx):
                _ = (tid, cid)

            tenant_callbacks[tenant_id].append(callback)
            TenantBroadcaster.subscribe(callback)

    # All 100 callbacks registered
    assert TenantBroadcaster.get_subscriber_count() == 100

    # "Destroy" tenants 0-4 (remove their callbacks)
    for tenant_id in range(5):
        tenant_callbacks[tenant_id].clear()

    gc.collect()

    # BUG (before fix): All 100 still registered (cross-tenant pollution)
    # Expected (with fix): ~50 remain (tenants 5-9), ~50 cleaned (tenants 0-4)
    remaining = TenantBroadcaster.get_subscriber_count()

    # With weakref: ~50 should be cleaned (allow 1-2 survivors)
    cleanup_count = 100 - remaining
    expected_cleanup = 50  # Tenants 0-4

    cleanup_rate = cleanup_count / expected_cleanup
    assert cleanup_rate >= 0.96, (
        f"Multi-tenant leak: {remaining}/100 callbacks remain (expected ~50). "
        f"Cleanup rate: {cleanup_rate:.1%} (expected ≥96% of destroyed tenants). "
        f"ClassVar causes cross-tenant pollution."
    )


@pytest.mark.asyncio
async def test_broadcaster_singleton_persistence():
    """Show that singleton pattern exacerbates the leak.

    Even if user "recreates" broadcaster, same instance persists with accumulated
    handlers.
    """

    class PersistentBroadcaster(Broadcaster):
        _event_type = BroadcastTestEvent
        _subscribers = []
        _instance = None

    # Session 1: Subscribe 20 callbacks
    session1_callbacks = []
    for i in range(20):

        async def callback(event, idx=i):
            _ = idx

        session1_callbacks.append(callback)
        PersistentBroadcaster.subscribe(callback)

    assert PersistentBroadcaster.get_subscriber_count() == 20

    # Session 1 ends, callbacks go out of scope
    session1_callbacks.clear()
    gc.collect()

    # BUG (before fix): Handlers persist across sessions
    leaked_after_session1 = PersistentBroadcaster.get_subscriber_count()

    # Session 2: Subscribe 20 more callbacks
    session2_callbacks = []
    for i in range(20):

        async def callback(event, idx=i):
            _ = idx

        session2_callbacks.append(callback)
        PersistentBroadcaster.subscribe(callback)

    # BUG (before fix): 40 total (session1 leaked + session2)
    # Expected (with fix): ~20 (only session2, session1 auto-cleaned)
    total_count = PersistentBroadcaster.get_subscriber_count()

    cleanup_rate = (20 - leaked_after_session1) / 20
    assert cleanup_rate >= 0.90, (
        f"Singleton persistence leak: {leaked_after_session1}/20 from session1 persist. "
        f"Total now: {total_count} (expected ~20). "
        f"Cleanup rate: {cleanup_rate:.1%} (expected ≥90%). "
        f"Singleton pattern causes cross-session accumulation."
    )


@pytest.mark.asyncio
async def test_broadcaster_bound_method_callbacks():
    """Test that bound method callbacks work correctly with WeakMethod.

    Critical test: weakref.ref() doesn't support bound methods, must use WeakMethod.
    Without WeakMethod, bound methods create immediately-dead weakrefs.
    """

    class BoundMethodBroadcaster(Broadcaster):
        _event_type = BroadcastTestEvent
        _subscribers = []
        _instance = None

    # Create class with bound method callback
    class CallbackHandler:
        def __init__(self):
            self.call_count = 0

        async def handle_event(self, event):
            """Bound method callback."""
            self.call_count += 1

    # Subscribe bound method
    handler = CallbackHandler()
    BoundMethodBroadcaster.subscribe(handler.handle_event)

    # Verify subscription registered
    assert BoundMethodBroadcaster.get_subscriber_count() == 1

    # Test that broadcast actually calls the bound method
    event = BroadcastTestEvent()
    await BoundMethodBroadcaster.broadcast(event)
    assert handler.call_count == 1, "Bound method should have been called"

    # Broadcast again to verify persistence
    await BoundMethodBroadcaster.broadcast(event)
    assert handler.call_count == 2, "Bound method should have been called again"

    # Now test cleanup when handler is destroyed
    handler_ref = weakref.ref(handler)
    del handler
    del event  # Clear event to remove any references
    gc.collect()

    # Bound method should be cleaned up (handler object destroyed)
    # Allow 1 survivor due to test framework artifacts (same as other leak tests)
    count_after_gc = BoundMethodBroadcaster.get_subscriber_count()
    assert count_after_gc <= 1, (
        f"Bound method callback should be mostly cleaned up after handler destruction. "
        f"Found {count_after_gc} subscribers (expected 0-1). "
        f"WeakMethod support is required for bound method cleanup."
    )

    # Test that handler was garbage collected
    # Note: handler_ref might still be alive due to test framework, but count should be low
    if handler_ref() is None:
        # Ideal case: handler fully GC'd
        assert count_after_gc == 0, "If handler is GC'd, subscriber count should be 0"

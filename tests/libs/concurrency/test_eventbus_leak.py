"""Test EventBus subscription memory leak - Issue #22.

This test demonstrates that EventBus handlers are NOT automatically cleaned up
when handler references are deleted, leading to memory leaks in long-running services.

The leak occurs because handlers are stored in a regular list, preventing garbage
collection even when the handler object is no longer referenced elsewhere.

Without WeakSet-based storage, this test will FAIL by showing handlers still
present after references are deleted.
"""

import gc
import weakref
from typing import Callable

import pytest

from lionherd_core.base.eventbus import EventBus


@pytest.mark.asyncio
async def test_eventbus_subscription_memory_leak():
    """Demonstrate EventBus subscription leak when handlers go out of scope.

    Expected behavior:
    - WITHOUT weakref: Handlers remain in _subs even after deletion (LEAK)
    - WITH weakref: Handlers auto-removed when garbage collected (FIXED)
    """
    bus = EventBus()

    # Track handler lifecycle with weakref
    handler_refs = []

    # Subscribe 100 handlers
    for i in range(100):
        async def handler(*args, **kwargs):
            """Handler closure that captures loop variable."""
            _ = i  # Capture variable to create closure

        bus.subscribe("test_topic", handler)
        # Track with weakref to detect when handler is GC'd
        handler_refs.append(weakref.ref(handler))

    # Verify all handlers registered
    assert bus.handler_count("test_topic") == 100

    # All weakrefs should be alive (handlers in _subs list)
    alive_before = sum(1 for ref in handler_refs if ref() is not None)
    assert alive_before == 100

    # Delete local handler references and force garbage collection
    # (In real code, this happens when request handlers go out of scope)
    handler_refs.clear()
    gc.collect()

    # BUG: Handlers still in _subs because list holds strong references
    # Expected: handler_count should be 0 after GC (with weakref fix)
    # Actual: handler_count still 100 (memory leak)
    leaked_count = bus.handler_count("test_topic")

    # This assertion will FAIL without weakref-based storage
    assert leaked_count == 0, (
        f"Memory leak detected: {leaked_count} handlers still in EventBus after "
        f"handler references deleted and GC ran. Handlers should be auto-cleaned "
        f"using weakref.WeakSet to prevent production memory leaks."
    )


@pytest.mark.asyncio
async def test_eventbus_subscription_accumulation():
    """Demonstrate memory accumulation in typical API server pattern.

    Pattern: Each request subscribes a handler, handler goes out of scope after
    request completes, but EventBus keeps accumulating handlers forever.
    """
    bus = EventBus()

    # Simulate 1000 requests, each subscribing a handler
    for request_id in range(1000):
        async def request_handler(*args, **kwargs):
            """Handler for single request (should be cleaned after request)."""
            _ = request_id  # Capture request context

        bus.subscribe("api_event", request_handler)
        # Handler goes out of scope here (end of request)

    # Force GC (simulating time between requests)
    gc.collect()

    # BUG: All 1000 handlers still registered
    # Expected: 0 handlers (with weakref auto-cleanup)
    # Actual: 1000 handlers (production memory leak)
    leaked = bus.handler_count("api_event")

    # This will FAIL - demonstrating the production leak scenario
    assert leaked == 0, (
        f"Production memory leak: {leaked} handlers accumulated from API requests. "
        f"In production (1M requests/day), this leaks ~160 MB/day. "
        f"Fix: Use weakref.WeakSet for automatic cleanup."
    )


@pytest.mark.asyncio
async def test_eventbus_manual_cleanup_burden():
    """Show that manual unsubscribe is error-prone and burdensome.

    Current API requires users to:
    1. Keep handler reference
    2. Manually call unsubscribe
    3. Handle exceptions carefully

    This test shows what happens when users forget (most common case).
    """
    bus = EventBus()
    handlers_to_cleanup = []

    # User subscribes handlers
    for i in range(50):
        async def handler(*args):
            pass
        bus.subscribe("topic", handler)
        handlers_to_cleanup.append(handler)

    assert bus.handler_count("topic") == 50

    # User "forgets" to unsubscribe (realistic scenario)
    # In real code: exception occurs, early return, etc.
    handlers_to_cleanup.clear()  # Lost references without cleanup
    gc.collect()

    # BUG: Handlers still registered, even though user lost references
    leaked = bus.handler_count("topic")

    # This demonstrates why manual cleanup is insufficient
    assert leaked == 0, (
        f"Manual cleanup failed: {leaked} handlers leaked when user lost references. "
        f"API should use weakref to make cleanup automatic and foolproof."
    )

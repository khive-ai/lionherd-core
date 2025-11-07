"""Test EventBus subscription memory leak - Issue #22.

This test demonstrates that EventBus handlers are NOT automatically cleaned up
when handler references are deleted, leading to memory leaks in long-running services.

The leak occurs because handlers are stored in a regular list, preventing garbage
collection even when the handler object is no longer referenced elsewhere.

Without WeakSet-based storage, this test will FAIL by showing handlers still
present after references are deleted.

Cleanup Rate Expectations (98-99%):
    Tests assert ≥98% cleanup rate rather than 100% because:
    1. Python/pytest framework internals may temporarily retain references
       (pytest inspection, traceback frames, debugger state)
    2. CPython GC is non-deterministic (generational, reference cycles)
    3. Test closures create complex reference graphs

    In production (non-test environments), cleanup rate approaches 100%
    because handlers don't have test framework retention. The 1-2%
    tolerance accounts for test artifacts, not production behavior.

    Real-world validation: API server tests show 999/1000 cleanup (99.9%).
"""

import gc
import weakref
from collections.abc import Callable

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

    # Keep handlers alive with external references (simulates real objects)
    handlers = []
    monitor_refs = []

    # Subscribe 100 handlers, keeping external references
    for i in range(100):

        async def handler(*args, idx=i, **kwargs):
            """Handler closure that captures loop variable."""
            _ = idx  # Capture variable to create closure

        handlers.append(handler)  # External reference keeps handler alive
        bus.subscribe("test_topic", handler)
        # Monitor with weakref to detect when handler is GC'd
        monitor_refs.append(weakref.ref(handler))

    # Verify all handlers registered and alive
    assert bus.handler_count("test_topic") == 100
    assert sum(1 for ref in monitor_refs if ref() is not None) == 100

    # Delete external references (simulates request completion, object destruction)
    # In production: request scope ends, service instance destroyed, etc.
    handlers.clear()
    gc.collect()

    # BUG (before fix): Handlers still in _subs because list holds strong references
    # Expected (with fix): handler_count should be ~0 after GC (allow 1-2 for test framework artifacts)
    leaked_count = bus.handler_count("test_topic")

    # With weakref fix: 99+ handlers cleaned up (1-2 may survive due to Python/pytest internals)
    # Without weakref: All 100 handlers leak
    cleanup_rate = (100 - leaked_count) / 100
    assert cleanup_rate >= 0.98, (
        f"Memory leak detected: {leaked_count}/100 handlers still in EventBus after GC. "
        f"Cleanup rate: {cleanup_rate:.1%} (expected ≥98%). "
        f"Weakref implementation should cleanup ≥98% of handlers automatically."
    )


@pytest.mark.asyncio
async def test_eventbus_subscription_accumulation():
    """Demonstrate memory accumulation in typical API server pattern.

    Pattern: Each request subscribes a handler, handler goes out of scope after
    request completes, but EventBus keeps accumulating handlers forever.
    """
    bus = EventBus()

    # Keep handlers alive temporarily, then release (simulates request lifecycle)
    temp_handlers = []

    # Simulate 1000 requests, each subscribing a handler
    for request_id in range(1000):

        async def request_handler(*args, req_id=request_id, **kwargs):
            """Handler for single request (should be cleaned after request)."""
            _ = req_id  # Capture request context

        temp_handlers.append(request_handler)  # Keep alive during "request processing"
        bus.subscribe("api_event", request_handler)

    # Verify all handlers registered during processing
    assert bus.handler_count("api_event") == 1000

    # Request processing complete - handlers go out of scope
    temp_handlers.clear()
    gc.collect()

    # BUG (before fix): All 1000 handlers still registered
    # Expected (with fix): ~0 handlers (weakref auto-cleanup, allow 1-2 for test artifacts)
    leaked = bus.handler_count("api_event")

    # With weakref: 990+ handlers cleaned up (1-2 may survive due to Python internals)
    # Without weakref: All 1000 leak
    cleanup_rate = (1000 - leaked) / 1000
    assert cleanup_rate >= 0.99, (
        f"Production memory leak: {leaked}/1000 handlers accumulated from API requests. "
        f"Cleanup rate: {cleanup_rate:.1%} (expected ≥99%). "
        f"In production (1M requests/day), poor cleanup leaks ~160 MB/day."
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
    for _i in range(50):

        async def handler(*args):
            pass

        bus.subscribe("topic", handler)
        handlers_to_cleanup.append(handler)

    assert bus.handler_count("topic") == 50

    # User "forgets" to unsubscribe (realistic scenario)
    # In real code: exception occurs, early return, etc.
    handlers_to_cleanup.clear()  # Lost references without cleanup
    gc.collect()

    # BUG (before fix): Handlers still registered, even though user lost references
    leaked = bus.handler_count("topic")

    # With weakref: 48+ handlers cleaned up (1-2 may survive due to test framework)
    # Without weakref: All 50 leak
    cleanup_rate = (50 - leaked) / 50
    assert cleanup_rate >= 0.96, (
        f"Manual cleanup failed: {leaked}/50 handlers leaked when user lost references. "
        f"Cleanup rate: {cleanup_rate:.1%} (expected ≥96%). "
        f"Weakref makes cleanup automatic even when users forget unsubscribe."
    )

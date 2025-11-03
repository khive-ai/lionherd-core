# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for concurrency wrapper functions.

These wrappers abstract anyio primitives to provide a single point
of control for concurrency backend flexibility.
"""

import time

import pytest

from lionherd_core.libs.concurrency import current_time, run_sync, sleep

# ============================================================================
# current_time() Tests
# ============================================================================


@pytest.mark.asyncio
async def test_current_time_returns_float():
    """Test current_time() returns a float timestamp."""
    result = current_time()
    assert isinstance(result, float)
    assert result > 0


@pytest.mark.asyncio
async def test_current_time_monotonic():
    """Test current_time() is monotonically increasing."""
    t1 = current_time()
    await sleep(0.01)
    t2 = current_time()
    assert t2 > t1


@pytest.mark.asyncio
async def test_current_time_precision():
    """Test current_time() has sub-second precision."""
    t1 = current_time()
    await sleep(0.001)  # 1ms
    t2 = current_time()

    diff = t2 - t1
    assert 0.001 <= diff < 0.1  # Should be ~1ms, allow variance


# ============================================================================
# sleep() Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sleep_basic():
    """Test sleep() pauses execution for specified duration."""
    start = current_time()
    await sleep(0.1)
    elapsed = current_time() - start

    # Allow 50ms variance for CI environments
    assert 0.08 <= elapsed <= 0.2


@pytest.mark.asyncio
async def test_sleep_zero():
    """Test sleep(0) yields to event loop without significant delay."""
    start = current_time()
    await sleep(0)
    elapsed = current_time() - start

    # Should be nearly instant (< 10ms)
    assert elapsed < 0.01


@pytest.mark.asyncio
async def test_sleep_cancellable():
    """Test sleep() is cancellable via anyio cancellation."""
    from lionherd_core.libs.concurrency import create_task_group, get_cancelled_exc_class

    cancelled = False

    async def sleeper():
        nonlocal cancelled
        try:
            await sleep(10.0)  # Long sleep
        except get_cancelled_exc_class():
            cancelled = True
            raise

    async with create_task_group() as tg:
        tg.start_soon(sleeper)
        await sleep(0.01)  # Let sleeper start
        tg.cancel_scope.cancel()

    assert cancelled, "sleep() should be cancellable"


# ============================================================================
# run_sync() Tests
# ============================================================================


def blocking_add(a: int, b: int) -> int:
    """Sync function for testing run_sync()."""
    return a + b


def blocking_raises(msg: str) -> None:
    """Sync function that raises for testing error propagation."""
    raise ValueError(msg)


def blocking_with_kwargs(**kwargs) -> dict:
    """Sync function accepting kwargs."""
    return kwargs


@pytest.mark.asyncio
async def test_run_sync_basic():
    """Test run_sync() executes sync function in thread pool."""
    result = await run_sync(blocking_add, 5, 3)
    assert result == 8


@pytest.mark.asyncio
async def test_run_sync_with_kwargs():
    """Test run_sync() passes kwargs correctly via lambda."""
    # anyio.to_thread.run_sync doesn't forward **kwargs,
    # so we use lambda to bind them
    result = await run_sync(lambda: blocking_with_kwargs(foo="bar", baz=123))
    assert result == {"foo": "bar", "baz": 123}


@pytest.mark.asyncio
async def test_run_sync_error_propagation():
    """Test run_sync() propagates exceptions from sync functions."""
    with pytest.raises(ValueError, match="test error"):
        await run_sync(blocking_raises, "test error")


@pytest.mark.asyncio
async def test_run_sync_doesnt_block_event_loop():
    """Test run_sync() doesn't block event loop for blocking operations."""
    from lionherd_core.libs.concurrency import create_task_group

    def slow_blocking():
        time.sleep(0.1)
        return "done"

    results = []

    async def fast_task():
        await sleep(0.01)
        results.append("fast")

    async def slow_task():
        result = await run_sync(slow_blocking)
        results.append(result)

    async with create_task_group() as tg:
        tg.start_soon(slow_task)
        tg.start_soon(fast_task)

    # Fast task should complete before slow task
    assert results[0] == "fast"
    assert results[1] == "done"


@pytest.mark.asyncio
async def test_run_sync_concurrent_execution():
    """Test multiple run_sync() calls execute concurrently in thread pool."""
    from lionherd_core.libs.concurrency import create_task_group

    def blocking_sleep(duration: float, value: int) -> int:
        time.sleep(duration)
        return value

    results = []

    start = current_time()

    async with create_task_group() as tg:
        for i in range(3):

            async def worker(val=i):
                result = await run_sync(blocking_sleep, 0.1, val)
                results.append(result)

            tg.start_soon(worker)

    elapsed = current_time() - start

    # Should complete in ~0.1s (concurrent), not 0.3s (sequential)
    assert 0.08 <= elapsed <= 0.2
    assert len(results) == 3


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_wrapper_integration_timing():
    """Test wrappers work together for timing measurements."""

    def sync_work():
        time.sleep(0.05)
        return "result"

    start = current_time()
    await sleep(0.05)
    result = await run_sync(sync_work)
    elapsed = current_time() - start

    assert result == "result"
    # Total: 50ms sleep + 50ms sync work = ~100ms
    assert 0.08 <= elapsed <= 0.2


@pytest.mark.asyncio
async def test_wrappers_preserve_anyio_semantics():
    """Test wrappers don't change anyio behavior semantics."""
    # Test that wrapped functions behave identically to unwrapped anyio
    import anyio

    # current_time() should match anyio.current_time()
    t1 = current_time()
    t2 = anyio.current_time()
    assert abs(t1 - t2) < 0.001  # Within 1ms

    # sleep() timing should match anyio.sleep()
    start1 = anyio.current_time()
    await sleep(0.05)
    elapsed1 = anyio.current_time() - start1

    start2 = anyio.current_time()
    await anyio.sleep(0.05)
    elapsed2 = anyio.current_time() - start2

    # Both should take similar time (within 20ms variance)
    assert abs(elapsed1 - elapsed2) < 0.02

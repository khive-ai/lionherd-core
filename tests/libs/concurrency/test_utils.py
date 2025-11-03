# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import time
from functools import partial

import anyio
import pytest

from lionherd_core.libs.concurrency import current_time, is_coro_func, run_sync, sleep


class TestIsCoroFunc:
    """Tests for is_coro_func() - Complete Documentation.

    Purpose:
        Check if a function is a coroutine function (async def).

    Behavior:
        Handles functools.partial and other wrappers by unwrapping to the
        underlying function before checking.

    Args:
        func: Callable to check

    Returns:
        True if func is a coroutine function (async def)

    Technical Notes:
        - Uses inspect.iscoroutinefunction, NOT inspect.isawaitable
        - isawaitable checks objects (after calling), not function definitions
        - Unwraps functools.partial wrappers to get underlying function
        - Results are cached for performance

    Edge Cases:
        - functools.partial wrappers: unwraps to check underlying function
        - Async methods vs regular coroutines: both return True
        - Nested partials: recursively unwraps until finding base function
    """

    def test_basic_coroutine_detection(self):
        """Test basic async def detection."""

        async def coro():
            pass

        def func():
            pass

        assert is_coro_func(coro) is True
        assert is_coro_func(func) is False

    def test_caching(self):
        """Test that results are cached for performance."""

        async def coro():
            pass

        # Call twice - should use cache on second call
        assert is_coro_func(coro) is True
        assert is_coro_func(coro) is True

    def test_partial_unwrapping(self):
        """Test handling of functools.partial wrappers."""

        async def async_func(x, y):
            return x + y

        def sync_func(x, y):
            return x + y

        partial_async = partial(async_func, 1)
        partial_sync = partial(sync_func, 1)

        assert is_coro_func(partial_async) is True
        assert is_coro_func(partial_sync) is False

    def test_nested_partial_unwrapping(self):
        """Test deeply nested partial wrappers."""

        async def async_func(a, b, c):
            return a + b + c

        nested = partial(partial(partial(async_func, 1), 2), 3)
        assert is_coro_func(nested) is True


class TestCurrentTime:
    """Tests for current_time() - Complete Documentation.

    Purpose:
        Get current time in seconds using monotonic clock.

    Behavior:
        Wrapper for anyio.current_time() to avoid direct anyio usage throughout codebase.

    Returns:
        float: Current time in seconds since an arbitrary epoch

    Technical Notes:
        - Uses monotonic clock - suitable for measuring elapsed time
        - NOT suitable for wall clock time (use time.time() for that)
        - Monotonic means time never goes backwards (immune to system clock adjustments)
        - Useful for timeouts, performance measurement, deadline calculations

    Use Cases:
        - Calculating elapsed time between operations
        - Setting absolute deadlines for cancellation scopes
        - Performance benchmarking
        - Timeout calculations
    """

    @pytest.mark.anyio
    async def test_returns_float(self):
        """Test that current_time returns a float."""
        t = current_time()
        assert isinstance(t, float)

    @pytest.mark.anyio
    async def test_monotonic_increases(self):
        """Test that time is monotonic (always increases)."""
        t1 = current_time()
        # Small delay
        await sleep(0.001)
        t2 = current_time()
        assert t2 > t1

    @pytest.mark.anyio
    async def test_consistency_with_anyio(self):
        """Test that it matches anyio.current_time()."""
        our_time = current_time()
        anyio_time = anyio.current_time()
        # Should be nearly identical (within 1ms for timing jitter)
        assert abs(our_time - anyio_time) < 0.001


class TestRunSync:
    """Tests for run_sync() - Complete Documentation.

    Purpose:
        Run synchronous function in thread pool without blocking event loop.

    Behavior:
        Wrapper for anyio.to_thread.run_sync() to avoid direct anyio usage throughout codebase.

    Args:
        func: Synchronous callable to run in thread pool
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result from func

    Technical Notes:
        - Allows running blocking I/O or CPU-bound sync functions without blocking
          the async event loop
        - Function executes in a worker thread from anyio's thread pool
        - Uses functools.partial internally to handle kwargs (anyio limitation)
        - Thread pool is managed by anyio - no manual thread management needed

    Use Cases:
        - Blocking I/O operations (file I/O, database calls without async driver)
        - CPU-bound synchronous computations
        - Legacy synchronous APIs that can't be easily converted to async
        - Third-party libraries without async support
    """

    @pytest.mark.anyio
    async def test_basic_sync_function(self):
        """Test running a basic sync function."""

        def sync_add(x, y):
            return x + y

        result = await run_sync(sync_add, 2, 3)
        assert result == 5

    @pytest.mark.anyio
    async def test_with_kwargs(self):
        """Test that kwargs are properly handled via partial."""

        def sync_func(a, b, c=10):
            return a + b + c

        result = await run_sync(sync_func, 1, 2, c=20)
        assert result == 23

    @pytest.mark.anyio
    async def test_blocking_operation_doesnt_block_event_loop(self):
        """Test that blocking operations don't block the event loop."""
        import threading

        main_thread = threading.current_thread()
        worker_thread = None

        def blocking_func():
            nonlocal worker_thread
            worker_thread = threading.current_thread()
            time.sleep(0.01)  # Simulate blocking I/O
            return "done"

        result = await run_sync(blocking_func)
        assert result == "done"
        assert worker_thread is not None
        assert worker_thread != main_thread  # Ran in different thread


class TestSleep:
    """Tests for sleep() - Complete Documentation.

    Purpose:
        Sleep for specified duration without blocking event loop.

    Behavior:
        Wrapper for anyio.sleep() to avoid direct anyio usage throughout codebase.

    Args:
        seconds: Duration to sleep in seconds

    Technical Notes:
        - This is a cancellable sleep
        - If the task is cancelled while sleeping, it will raise CancelledError
          immediately rather than waiting for the full duration
        - Does not block the event loop - other tasks can run concurrently
        - Uses anyio's sleep mechanism under the hood

    Cancellation Behavior:
        - Respects structured concurrency cancellation scopes
        - Will raise exception immediately when cancelled
        - Does not complete the full sleep duration if cancelled
        - Integrates with anyio's cancellation system
    """

    @pytest.mark.anyio
    async def test_basic_sleep(self):
        """Test basic sleep functionality."""
        start = time.perf_counter()
        await sleep(0.01)
        elapsed = time.perf_counter() - start
        # Allow for timing jitter (especially on CI)
        assert 0.005 < elapsed < 0.05

    @pytest.mark.anyio
    async def test_sleep_is_cancellable(self):
        """Test that sleep respects cancellation."""
        from lionherd_core.libs.concurrency import move_on_after

        completed = False
        with move_on_after(0.005):
            await sleep(1.0)  # Would take 1 second
            completed = True

        # Should have been cancelled before completing
        assert completed is False

    @pytest.mark.anyio
    async def test_sleep_doesnt_block_other_tasks(self):
        """Test that sleep allows other tasks to run concurrently."""
        results = []

        async def task1():
            await sleep(0.02)
            results.append("task1")

        async def task2():
            await sleep(0.01)
            results.append("task2")

        async with anyio.create_task_group() as tg:
            tg.start_soon(task1)
            tg.start_soon(task2)

        # task2 should complete first despite being started second
        assert results == ["task2", "task1"]

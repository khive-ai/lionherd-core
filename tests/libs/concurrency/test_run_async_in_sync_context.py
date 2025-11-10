# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for run_async (event loop handling via threading).

This module tests the run_async utility function that was
extracted from duplicated code in EdgeCondition.__call__() and Graph.find_path()
(issue #51).

Implementation Note:
====================
This function was refactored from nest_asyncio-based approach to threading-based
approach. The new implementation creates a new event loop in a separate thread,
avoiding the need to patch asyncio internals.

Test Coverage:
==============
1. Return value correctness (various types)
2. Coroutine with arguments
3. Exception propagation from coroutine
4. Works from sync context (no event loop)
5. Works from async context (event loop exists)
6. Integration patterns matching EdgeCondition and Graph usage
"""

from __future__ import annotations

import asyncio

import pytest

from lionherd_core.libs.concurrency import run_async


# Test coroutines with various return types
async def simple_coro() -> int:
    """Simple coroutine returning int."""
    return 42


async def string_coro() -> str:
    """Coroutine returning string."""
    return "test_value"


async def dict_coro() -> dict:
    """Coroutine returning dict."""
    return {"key": "value", "count": 123}


async def coro_with_args(value: int, multiplier: int = 2) -> int:
    """Coroutine with arguments."""
    return value * multiplier


async def failing_coro() -> None:
    """Coroutine that raises exception."""
    raise ValueError("Test exception")


class TestRunAsyncInSyncContextNoLoop:
    """Tests for run_async when NO event loop is running."""

    def test_simple_return_value(self):
        """Should return correct value when no loop."""
        result = run_async(simple_coro())
        assert result == 42

    def test_string_return_value(self):
        """Should correctly return string values."""
        result = run_async(string_coro())
        assert result == "test_value"

    def test_dict_return_value(self):
        """Should correctly return dict values."""
        result = run_async(dict_coro())
        assert result == {"key": "value", "count": 123}

    def test_with_arguments(self):
        """Should handle coroutines with arguments."""
        result = run_async(coro_with_args(5, multiplier=3))
        assert result == 15

    def test_exception_propagation(self):
        """Should propagate exceptions from coroutine."""
        with pytest.raises(ValueError, match="Test exception"):
            run_async(failing_coro())


class TestRunAsyncInSyncContextWithLoop:
    """Tests for run_async when event loop IS running.

    The threading-based implementation creates a separate event loop in a new
    thread, so it works correctly even when called from within an async context.
    """

    @pytest.mark.asyncio
    async def test_simple_return_value(self):
        """Should work correctly even when event loop exists."""
        result = run_async(simple_coro())
        assert result == 42

    @pytest.mark.asyncio
    async def test_string_return_value(self):
        """Should correctly return string values with event loop."""
        result = run_async(string_coro())
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_dict_return_value(self):
        """Should correctly return dict values with event loop."""
        result = run_async(dict_coro())
        assert result == {"key": "value", "count": 123}

    @pytest.mark.asyncio
    async def test_with_arguments(self):
        """Should handle coroutines with arguments in event loop."""
        result = run_async(coro_with_args(7, multiplier=4))
        assert result == 28

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Should propagate exceptions from coroutine in event loop."""
        with pytest.raises(ValueError, match="Test exception"):
            run_async(failing_coro())


class TestRunAsyncInSyncContextIntegration:
    """Integration tests matching original EdgeCondition and Graph.find_path usage."""

    def test_edge_condition_pattern(self):
        """Should work with EdgeCondition.__call__ pattern (async function wrapper)."""

        async def condition_apply(x: int, y: int) -> bool:
            """Simulates EdgeCondition.apply()."""
            return x > y

        # Pattern from EdgeCondition.__call__
        async def _run():
            return await condition_apply(10, 5)

        result = run_async(_run())
        assert result is True

    def test_graph_find_path_pattern(self):
        """Should work with Graph.find_path pattern (direct coroutine)."""

        async def check_condition() -> bool:
            """Simulates edge.check_condition()."""
            return False

        # Pattern from Graph.find_path
        result = run_async(check_condition())
        assert result is False

    def test_multiple_sequential_calls(self):
        """Should handle multiple sequential calls correctly."""
        results = []
        for i in range(5):

            async def numbered_coro(n=i):
                return n * 10

            results.append(run_async(numbered_coro()))

        assert results == [0, 10, 20, 30, 40]


class TestRunAsyncInSyncContextThreading:
    """Tests specific to threading-based implementation."""

    def test_works_without_patching_asyncio(self):
        """Threading approach doesn't require patching asyncio internals."""
        # This test verifies the function works without any mocking or patching
        result = run_async(simple_coro())
        assert result == 42

    @pytest.mark.asyncio
    async def test_separate_event_loop_in_thread(self):
        """Should create separate event loop in thread, not interfere with existing loop."""
        # Get the current event loop ID
        outer_loop = asyncio.get_running_loop()
        outer_loop_id = id(outer_loop)

        captured_loop_id = None

        async def capture_loop_id():
            """Capture the event loop ID inside run_async."""
            nonlocal captured_loop_id
            loop = asyncio.get_running_loop()
            captured_loop_id = id(loop)
            return 123

        result = run_async(capture_loop_id())

        # Verify the function returned correctly
        assert result == 123

        # Verify a different event loop was used (in separate thread)
        assert captured_loop_id is not None
        assert captured_loop_id != outer_loop_id, "Should use separate event loop in thread"

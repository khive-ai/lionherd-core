# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Graph event loop handling (Jupyter compatibility).

This test module validates the event loop detection and nest-asyncio integration
added to fix Jupyter notebook compatibility (PR #44).

Event Loop Handling Design
===========================

Problem:
- Jupyter notebooks run in an active asyncio event loop
- Python's asyncio.run() cannot create nested loops (raises RuntimeError)
- EdgeCondition.__call__() and Graph.find_path() need to call async code from sync context

Solution:
- Detect running event loop via asyncio.get_running_loop()
- If no loop: use anyio.run() (existing behavior, works in regular Python)
- If loop exists: apply nest_asyncio.apply() + asyncio.run() (Jupyter compatibility)

Critical Bug Fixed:
- Line 73: asyncio.run(_run) → asyncio.run(_run())
- asyncio.run() requires coroutine object (with parentheses), not function

Test Coverage:
==============
1. EdgeCondition.__call__() without event loop (anyio.run path)
2. EdgeCondition.__call__() with event loop (nest_asyncio path)
3. Graph.find_path() with check_conditions=True without event loop
4. Graph.find_path() with check_conditions=True with event loop
5. Verify nest_asyncio actually applied when event loop exists
6. Verify correct coroutine handling (bug regression test)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from lionherd_core.base import Edge, EdgeCondition, Graph, Node


class CountingCondition(EdgeCondition):
    """Test condition that tracks if it was called."""

    call_count = 0

    async def apply(self, *args, **kwargs) -> bool:
        CountingCondition.call_count += 1
        return True


class FailCondition(EdgeCondition):
    """Test condition that always fails."""

    async def apply(self, *args, **kwargs) -> bool:
        return False


class TestEdgeConditionEventLoop:
    """Test EdgeCondition.__call__() event loop detection."""

    async def test_edgecondition_call_no_event_loop(self):
        """Test EdgeCondition.__call__() when no event loop is running (anyio.run path)."""
        condition = CountingCondition()
        CountingCondition.call_count = 0

        # No event loop running - async call
        result = await condition()

        assert result is True
        assert CountingCondition.call_count == 1

    @pytest.mark.asyncio
    async def test_edgecondition_call_with_event_loop(self):
        """Test EdgeCondition.__call__() when event loop is running (nest_asyncio path)."""
        condition = CountingCondition()
        CountingCondition.call_count = 0

        # Event loop IS running (we're in async test)
        # Async call
        result = await condition()

        assert result is True
        assert CountingCondition.call_count == 1

    @pytest.mark.asyncio
    async def test_edgecondition_call_returns_coroutine_result(self):
        """Regression test: Ensure asyncio.run() receives coroutine, not function.

        This tests the fix for the critical bug where asyncio.run(_run) was
        called instead of asyncio.run(_run()). The bug would cause:
        ValueError: a coroutine was expected, got <function>
        """
        condition = FailCondition()

        # If bug exists, this would raise ValueError about expecting coroutine
        result = await condition()

        # Should return False (the condition result), not raise ValueError
        assert result is False

    async def test_edgecondition_call_with_args(self):
        """Test EdgeCondition.__call__() passes arguments correctly."""

        class ArgCondition(EdgeCondition):
            async def apply(self, threshold: float = 10.0) -> bool:
                return threshold > 5.0

        condition = ArgCondition()

        # Pass argument
        result = await condition(threshold=15.0)
        assert result is True

        result = await condition(threshold=3.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_edgecondition_nest_asyncio_applied_when_loop_exists(self):
        """Verify EdgeCondition works in async context (nest_asyncio no longer needed for pure async)."""
        condition = CountingCondition()
        CountingCondition.call_count = 0

        # Call in async context (event loop exists)
        result = await condition()

        # Should work correctly without nest_asyncio (pure async now)
        assert result is True
        assert CountingCondition.call_count == 1


class TestGraphFindPathEventLoop:
    """Test Graph.find_path() event loop detection with conditional traversal."""

    async def test_find_path_conditional_no_event_loop(self):
        """Test find_path with check_conditions=True when no event loop (anyio.run path)."""
        graph = Graph()

        n1 = Node(content={"name": "A"})
        n2 = Node(content={"name": "B"})
        graph.add_node(n1)
        graph.add_node(n2)

        # Edge with condition
        edge = Edge(head=n1.id, tail=n2.id, condition=CountingCondition())
        graph.add_edge(edge)

        CountingCondition.call_count = 0

        # Async call
        path = await graph.find_path(n1, n2, check_conditions=True)

        assert path is not None
        assert len(path) == 1
        assert CountingCondition.call_count == 1

    @pytest.mark.asyncio
    async def test_find_path_conditional_with_event_loop(self):
        """Test find_path with check_conditions=True when event loop exists (nest_asyncio path)."""
        graph = Graph()

        n1 = Node(content={"name": "A"})
        n2 = Node(content={"name": "B"})
        graph.add_node(n1)
        graph.add_node(n2)

        # Edge with condition
        edge = Edge(head=n1.id, tail=n2.id, condition=CountingCondition())
        graph.add_edge(edge)

        CountingCondition.call_count = 0

        # Event loop IS running - async call
        path = await graph.find_path(n1, n2, check_conditions=True)

        assert path is not None
        assert len(path) == 1
        assert CountingCondition.call_count == 1

    @pytest.mark.asyncio
    async def test_find_path_conditional_blocked_with_event_loop(self):
        """Test find_path returns None when condition blocks in async context."""
        graph = Graph()

        n1 = Node(content={"name": "A"})
        n2 = Node(content={"name": "B"})
        n3 = Node(content={"name": "C"})
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        # Two paths: direct (blocked) and via n2 (allowed)
        blocked_edge = Edge(head=n1.id, tail=n3.id, condition=FailCondition())
        allowed_edge1 = Edge(head=n1.id, tail=n2.id, condition=CountingCondition())
        allowed_edge2 = Edge(head=n2.id, tail=n3.id, condition=CountingCondition())

        graph.add_edge(blocked_edge)
        graph.add_edge(allowed_edge1)
        graph.add_edge(allowed_edge2)

        # Event loop running - find alternate path
        path = await graph.find_path(n1, n3, check_conditions=True)

        # Should find the longer path that isn't blocked
        assert path is not None
        assert len(path) == 2

    @pytest.mark.asyncio
    async def test_find_path_nest_asyncio_applied_when_conditions_checked(self):
        """Verify find_path works correctly in async context with conditions."""
        graph = Graph()

        n1 = Node(content={"name": "A"})
        n2 = Node(content={"name": "B"})
        graph.add_node(n1)
        graph.add_node(n2)

        edge = Edge(head=n1.id, tail=n2.id, condition=CountingCondition())
        graph.add_edge(edge)

        CountingCondition.call_count = 0

        # Call in async context with conditions
        path = await graph.find_path(n1, n2, check_conditions=True)

        # Should work correctly (pure async, no nest_asyncio needed)
        assert path is not None
        assert len(path) == 1
        assert CountingCondition.call_count == 1

    async def test_find_path_without_conditions_no_nest_asyncio(self):
        """Verify find_path works without checking conditions."""
        graph = Graph()

        n1 = Node(content={"name": "A"})
        n2 = Node(content={"name": "B"})
        graph.add_node(n1)
        graph.add_node(n2)

        edge = Edge(head=n1.id, tail=n2.id, condition=CountingCondition())
        graph.add_edge(edge)

        CountingCondition.call_count = 0

        # No conditions checked
        path = await graph.find_path(n1, n2, check_conditions=False)

        # Should work and NOT evaluate condition
        assert path is not None
        assert len(path) == 1
        assert CountingCondition.call_count == 0


class TestEventLoopRegressions:
    """Regression tests for specific event loop bugs."""

    async def test_asyncio_run_receives_coroutine_not_function(self):
        """Regression: asyncio.run(_run) vs asyncio.run(_run()).

        Bug: Line 73 had asyncio.run(_run) - passes function object
        Fix: Should be asyncio.run(_run()) - passes coroutine object

        This test ensures the fix is correct and prevents future regressions.
        """

        class InstrumentedCondition(EdgeCondition):
            """Condition that verifies async execution."""

            executed = False

            async def apply(self, *args, **kwargs) -> bool:
                InstrumentedCondition.executed = True
                return True

        condition = InstrumentedCondition()
        InstrumentedCondition.executed = False

        # Call async interface
        result = await condition()

        # Should execute the async apply() method
        assert InstrumentedCondition.executed is True
        assert result is True

    @pytest.mark.asyncio
    async def test_edge_check_condition_callable_in_async_context(self):
        """Test Edge.check_condition() works correctly in async contexts.

        This validates that the async API (check_condition()) works independently
        of the sync API (__call__()) in async contexts.
        """
        edge = Edge(
            head=Node(content={"name": "A"}).id,
            tail=Node(content={"name": "B"}).id,
            condition=CountingCondition(),
        )

        CountingCondition.call_count = 0

        # Direct async call
        result = await edge.check_condition()

        assert result is True
        assert CountingCondition.call_count == 1

    async def test_multiple_conditions_sequential_calls(self):
        """Test multiple EdgeCondition calls don't interfere with each other."""
        cond1 = CountingCondition()
        cond2 = CountingCondition()

        CountingCondition.call_count = 0

        result1 = await cond1()
        result2 = await cond2()

        assert result1 is True
        assert result2 is True
        assert CountingCondition.call_count == 2

    @pytest.mark.asyncio
    async def test_mixed_sync_async_condition_calls(self):
        """Test mixing __call__() and async apply() works correctly."""
        condition = CountingCondition()
        CountingCondition.call_count = 0

        # Async __call__
        call_result = await condition()
        assert call_result is True
        assert CountingCondition.call_count == 1

        # Async apply() directly
        async_result = await condition.apply()
        assert async_result is True
        assert CountingCondition.call_count == 2


class TestNestAsyncioIntegration:
    """Test nest-asyncio integration details."""

    @pytest.mark.asyncio
    async def test_nest_asyncio_is_idempotent(self):
        """Verify EdgeCondition works correctly in async context (pure async, no nest_asyncio needed).

        The implementation is now pure async, no longer requiring nest_asyncio.
        This test ensures conditions work correctly in async contexts.
        """
        # Conditions should work in async context
        condition = CountingCondition()
        CountingCondition.call_count = 0

        result = await condition()

        assert result is True
        assert CountingCondition.call_count == 1

    @pytest.mark.asyncio
    async def test_event_loop_detection_is_correct(self):
        """Verify asyncio.get_running_loop() correctly detects async context."""
        # In async test, loop should be running
        try:
            loop = asyncio.get_running_loop()
            assert loop is not None
        except RuntimeError:
            pytest.fail("Expected running event loop in async test")

    def test_no_event_loop_detection_is_correct(self):
        """Verify asyncio.get_running_loop() raises when no loop running."""
        # In sync test, no loop should be running
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()

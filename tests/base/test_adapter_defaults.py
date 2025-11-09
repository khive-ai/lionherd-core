# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for adapter default parameter behavior in Pile, Node, and Graph.

Validates: Default adapt_meth/adapt_kw setting, setdefault preservation, sync/async handling.
Classes: Pile, Node, Graph | Methods: adapt_to, adapt_from, adapt_to_async, adapt_from_async
"""

from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from pydapter.async_core import AsyncAdapter
from pydapter.core import Adapter, AdapterBase, dispatch_adapt_meth

from lionherd_core.base import Graph, Node, Pile

# ==================== Test Adapters ====================


class CaptureAdapter(AdapterBase, Adapter):
    """Sync test adapter that captures adapt_meth and adapt_kw for verification."""

    adapter_key: ClassVar[str] = "test_capture"
    obj_key: ClassVar[str] = "test_capture"

    # Store captured parameters (class-level for test inspection)
    last_to_params: ClassVar[dict[str, Any]] = {}
    last_from_params: ClassVar[dict[str, Any]] = {}

    @classmethod
    def from_obj(
        cls,
        subj_cls: type,
        obj: Any,
        /,
        *,
        many: bool = False,
        adapt_meth: str | Callable = "model_validate",
        adapt_kw: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Any:
        """Capture from_obj parameters and call adapt_meth on data."""
        # Capture parameters for test verification
        cls.last_from_params = {
            "adapt_meth": adapt_meth,
            "adapt_kw": adapt_kw,
            "many": many,
            **kw,
        }

        # Call the adapt_meth on the parsed data
        return dispatch_adapt_meth(adapt_meth, obj, adapt_kw, subj_cls)

    @classmethod
    def to_obj(
        cls,
        subj: Any,
        /,
        *,
        many: bool = False,
        adapt_meth: str | Callable = "model_dump",
        adapt_kw: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Any:
        """Capture to_obj parameters and call adapt_meth on object."""
        # Capture parameters for test verification
        cls.last_to_params = {
            "adapt_meth": adapt_meth,
            "adapt_kw": adapt_kw,
            "many": many,
            **kw,
        }

        # Call the adapt_meth on the object
        return dispatch_adapt_meth(adapt_meth, subj, adapt_kw, type(subj))


class AsyncCaptureAdapter(AdapterBase, AsyncAdapter):
    """Async test adapter that captures parameters."""

    adapter_key: ClassVar[str] = "test_async_capture"
    obj_key: ClassVar[str] = "test_async_capture"

    last_to_params: ClassVar[dict[str, Any]] = {}
    last_from_params: ClassVar[dict[str, Any]] = {}

    @classmethod
    async def from_obj(
        cls,
        subj_cls: type,
        obj: Any,
        /,
        *,
        many: bool = False,
        adapt_meth: str | Callable = "model_validate",
        adapt_kw: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Any:
        """Async from_obj method (not from_obj_async)."""
        cls.last_from_params = {
            "adapt_meth": adapt_meth,
            "adapt_kw": adapt_kw,
            "many": many,
            **kw,
        }

        # Call the adapt_meth directly on the class
        if isinstance(adapt_meth, str):
            method = getattr(subj_cls, adapt_meth)
            return method(obj, **(adapt_kw or {}))
        else:
            return adapt_meth(obj, **(adapt_kw or {}))

    @classmethod
    async def to_obj(
        cls,
        subj: Any,
        /,
        *,
        many: bool = False,
        adapt_meth: str | Callable = "model_dump",
        adapt_kw: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Any:
        """Async to_obj method (not to_obj_async)."""
        cls.last_to_params = {
            "adapt_meth": adapt_meth,
            "adapt_kw": adapt_kw,
            "many": many,
            **kw,
        }

        # Call the adapt_meth directly on the object
        if isinstance(adapt_meth, str):
            method = getattr(subj, adapt_meth)
            return method(**(adapt_kw or {}))
        else:
            return adapt_meth(subj, **(adapt_kw or {}))


# ==================== Pile Tests ====================


class TestPileAdapterDefaults:
    """Test adapter default parameters for Pile class."""

    def test_pile_adapt_to_uses_defaults(self):
        """Test Pile.adapt_to sets defaults: adapt_meth='to_dict', adapt_kw={'mode': 'db'}."""
        from lionherd_core.base.element import Element

        # Register adapter
        Pile.register_adapter(CaptureAdapter)

        # Create pile with elements
        elem1 = Element()
        elem2 = Element()
        pile = Pile([elem1, elem2])

        # Call adapt_to without custom kwargs (triggers defaults)
        result = pile.adapt_to("test_capture")

        # Verify defaults were set
        assert CaptureAdapter.last_to_params["adapt_meth"] == "to_dict"
        assert CaptureAdapter.last_to_params["adapt_kw"] == {"mode": "db"}

        # Verify result is valid
        assert isinstance(result, dict)

    def test_pile_adapt_from_uses_defaults(self):
        """Test Pile.adapt_from sets default: adapt_meth='from_dict', adapt_kw=None."""
        Pile.register_adapter(CaptureAdapter)

        # Create serialized data (use mode="python" for clean dict)
        from lionherd_core.base.element import Element

        pile = Pile([Element(), Element()])
        data = pile.to_dict(mode="python")

        # Call adapt_from without custom kwargs (triggers default)
        restored = Pile.adapt_from(data, "test_capture")

        # Verify default was set
        assert CaptureAdapter.last_from_params["adapt_meth"] == "from_dict"
        assert CaptureAdapter.last_from_params["adapt_kw"] is None

        # Verify result is valid
        assert isinstance(restored, Pile)

    @pytest.mark.anyio
    async def test_pile_adapt_to_async_uses_defaults(self):
        """Test Pile.adapt_to_async sets same defaults as sync: to_dict + mode='db'."""
        from lionherd_core.base.element import Element

        # Register async adapter
        Pile.register_async_adapter(AsyncCaptureAdapter)

        # Create pile
        pile = Pile([Element(), Element()])

        # Call adapt_to_async without custom kwargs
        result = await pile.adapt_to_async("test_async_capture")

        # Verify defaults were set
        assert AsyncCaptureAdapter.last_to_params["adapt_meth"] == "to_dict"
        assert AsyncCaptureAdapter.last_to_params["adapt_kw"] == {"mode": "db"}

        # Verify result
        assert isinstance(result, dict)

    @pytest.mark.anyio
    async def test_pile_adapt_from_async_uses_defaults(self):
        """Test Pile.adapt_from_async sets default: adapt_meth='from_dict'."""
        Pile.register_async_adapter(AsyncCaptureAdapter)

        # Create serialized data (use mode="python" for clean dict)
        from lionherd_core.base.element import Element

        pile = Pile([Element()])
        data = pile.to_dict(mode="python")

        # Call adapt_from_async without custom kwargs
        restored = await Pile.adapt_from_async(data, "test_async_capture")

        # Verify default was set
        assert AsyncCaptureAdapter.last_from_params["adapt_meth"] == "from_dict"
        assert AsyncCaptureAdapter.last_from_params["adapt_kw"] is None

        # Verify result
        assert isinstance(restored, Pile)


# ==================== Node Tests ====================


class TestNodeAdapterDefaults:
    """Test adapter default parameters for Node class."""

    @pytest.mark.anyio
    async def test_node_adapt_to_async_uses_defaults(self):
        """Test Node.adapt_to_async uses Pile defaults: to_dict + mode='db'."""
        # Register async adapter
        Node.register_async_adapter(AsyncCaptureAdapter)

        # Create node
        node = Node(content={"value": "test node"})

        # Call adapt_to_async without custom kwargs
        result = await node.adapt_to_async("test_async_capture")

        # Verify defaults were set
        assert AsyncCaptureAdapter.last_to_params["adapt_meth"] == "to_dict"
        assert AsyncCaptureAdapter.last_to_params["adapt_kw"] == {"mode": "db"}

        # Verify result
        assert isinstance(result, dict)
        assert "content" in result

    @pytest.mark.anyio
    async def test_node_adapt_from_async_uses_defaults(self):
        """Test Node.adapt_from_async sets default: adapt_meth='from_dict'."""
        Node.register_async_adapter(AsyncCaptureAdapter)

        # Create serialized data (use mode="python" for clean dict)
        node = Node(content={"value": "test"})
        data = node.to_dict(mode="python")

        # Call adapt_from_async without custom kwargs
        restored = await Node.adapt_from_async(data, "test_async_capture")

        # Verify default was set
        assert AsyncCaptureAdapter.last_from_params["adapt_meth"] == "from_dict"
        assert AsyncCaptureAdapter.last_from_params["adapt_kw"] is None

        # Verify result
        assert isinstance(restored, Node)
        assert restored.content == {"value": "test"}


# ==================== Graph Tests ====================


class TestGraphAdapterDefaults:
    """Test adapter default parameters for Graph class."""

    def test_graph_adapt_to_uses_defaults(self):
        """Test Graph.adapt_to uses Pile/Node defaults: to_dict + mode='db'."""
        # Register adapter
        Graph.register_adapter(CaptureAdapter)

        # Create graph with nodes
        graph = Graph()
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        graph.add_node(n1)
        graph.add_node(n2)

        # Call adapt_to without custom kwargs
        result = graph.adapt_to("test_capture")

        # Verify defaults were set
        assert CaptureAdapter.last_to_params["adapt_meth"] == "to_dict"
        assert CaptureAdapter.last_to_params["adapt_kw"] == {"mode": "db"}

        # Verify result
        assert isinstance(result, dict)

    def test_graph_adapt_from_uses_defaults(self):
        """Test Graph.adapt_from sets default: adapt_meth='from_dict'."""
        Graph.register_adapter(CaptureAdapter)

        # Create serialized data (use mode="python" for clean dict)
        graph = Graph()
        graph.add_node(Node(content={"value": "A"}))
        data = graph.to_dict(mode="python")

        # Call adapt_from without custom kwargs
        restored = Graph.adapt_from(data, "test_capture")

        # Verify default was set
        assert CaptureAdapter.last_from_params["adapt_meth"] == "from_dict"
        assert CaptureAdapter.last_from_params["adapt_kw"] is None

        # Verify result
        assert isinstance(restored, Graph)

    @pytest.mark.anyio
    async def test_graph_adapt_to_async_uses_defaults(self):
        """Test Graph.adapt_to_async uses sync defaults: to_dict + mode='db'."""
        # Register async adapter
        Graph.register_async_adapter(AsyncCaptureAdapter)

        # Create graph
        graph = Graph()
        graph.add_node(Node(content={"value": "A"}))

        # Call adapt_to_async without custom kwargs
        result = await graph.adapt_to_async("test_async_capture")

        # Verify defaults were set
        assert AsyncCaptureAdapter.last_to_params["adapt_meth"] == "to_dict"
        assert AsyncCaptureAdapter.last_to_params["adapt_kw"] == {"mode": "db"}

        # Verify result
        assert isinstance(result, dict)

    @pytest.mark.anyio
    async def test_graph_adapt_from_async_uses_defaults(self):
        """Test Graph.adapt_from_async sets default: adapt_meth='from_dict'."""
        Graph.register_async_adapter(AsyncCaptureAdapter)

        # Create serialized data (use mode="python" for clean dict)
        graph = Graph()
        graph.add_node(Node(content={"value": "A"}))
        data = graph.to_dict(mode="python")

        # Call adapt_from_async without custom kwargs
        restored = await Graph.adapt_from_async(data, "test_async_capture")

        # Verify default was set
        assert AsyncCaptureAdapter.last_from_params["adapt_meth"] == "from_dict"
        assert AsyncCaptureAdapter.last_from_params["adapt_kw"] is None

        # Verify result
        assert isinstance(restored, Graph)


# ==================== Override Tests ====================


class TestAdapterOverrideDefaults:
    """Test that setdefault doesn't override existing kwargs."""

    def test_pile_adapt_to_respects_custom_kwargs(self):
        """Test that custom adapt_kw overrides default (setdefault behavior)."""
        from lionherd_core.base.element import Element

        Pile.register_adapter(CaptureAdapter)

        pile = Pile([Element()])

        # Pass custom adapt_kw (should NOT be overridden by setdefault)
        custom_kw = {"mode": "json", "exclude": ["metadata"]}
        result = pile.adapt_to("test_capture", adapt_kw=custom_kw)

        # Verify custom kwargs were used, not defaults
        assert CaptureAdapter.last_to_params["adapt_kw"] == custom_kw
        assert CaptureAdapter.last_to_params["adapt_kw"]["mode"] == "json"

        # Verify result is valid
        assert isinstance(result, dict)

    def test_pile_adapt_from_respects_custom_adapt_meth(self):
        """Test that custom adapt_meth overrides default."""
        from lionherd_core.base.element import Element

        Pile.register_adapter(CaptureAdapter)

        pile = Pile([Element()])
        data = pile.to_dict(mode="python")

        # Pass custom adapt_meth explicitly (setdefault should not override)
        restored = Pile.adapt_from(data, "test_capture", adapt_meth="from_dict")

        # Verify custom adapt_meth was passed through, not overridden
        assert CaptureAdapter.last_from_params["adapt_meth"] == "from_dict"

        # Verify result is valid
        assert isinstance(restored, Pile)

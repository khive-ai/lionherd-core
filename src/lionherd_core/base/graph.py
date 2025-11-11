# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr, field_validator, model_validator
from pydapter import (
    Adaptable as PydapterAdaptable,
    AsyncAdaptable as PydapterAsyncAdaptable,
)
from typing_extensions import override

from ..errors import NotFoundError
from ..protocols import (
    Adaptable,
    AdapterRegisterable,
    AsyncAdaptable,
    AsyncAdapterRegisterable,
    Containable,
    Deserializable,
    Serializable,
    implements,
)
from ._utils import synchronized
from .element import Element
from .node import Node
from .pile import Pile

__all__ = ("Edge", "EdgeCondition", "Graph")


# ==================== EdgeCondition ====================


class EdgeCondition:
    """Runtime predicate for edge traversal (not serialized).

    Override apply() for custom async logic. Callable via __call__() for sync contexts.
    """

    def __init__(self, **kwargs: Any):
        """Initialize condition. Subclasses can store state as needed."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    async def apply(self, *args: Any, **kwargs: Any) -> bool:
        """Evaluate condition. Override for custom logic. Default: always True."""
        return True

    async def __call__(self, *args: Any, **kwargs: Any) -> bool:
        """Async callable interface. Calls apply() directly."""
        return await self.apply(*args, **kwargs)


# ==================== Edge ====================


class Edge(Element):
    """Directed edge with labels, conditions, properties.

    Attributes:
        head: Source node UUID
        tail: Target node UUID
        label: Edge labels (list of strings)
        condition: Runtime traversal predicate (not serialized)
        properties: Custom edge attributes
    """

    head: UUID = Field(description="Source node ID")
    tail: UUID = Field(description="Target node ID")
    label: list[str] = Field(default_factory=list, description="Edge labels")
    condition: EdgeCondition | None = Field(
        default=None, exclude=True, description="Runtime traversal condition (not serialized)"
    )
    properties: dict[str, Any] = Field(default_factory=dict, description="Custom edge properties")

    @field_validator("head", "tail", mode="before")
    @classmethod
    def _validate_uuid(cls, value: Any) -> UUID:
        """Coerce to UUID."""
        return cls._coerce_id(value)

    async def check_condition(self, *args: Any, **kwargs: Any) -> bool:
        """Check if edge is traversable. Returns True if no condition or condition passes."""
        if self.condition is None:
            return True
        return await self.condition.apply(*args, **kwargs)


# ==================== Graph ====================


@implements(
    Serializable,
    Deserializable,
    Containable,
    Adaptable,
    AdapterRegisterable,
    AsyncAdaptable,
    AsyncAdapterRegisterable,
)
class Graph(Element, PydapterAdaptable, PydapterAsyncAdaptable):
    """Directed graph with Pile-backed storage, O(1) operations, graph algorithms.

    Adjacency lists (_out_edges, _in_edges) provide O(1) node/edge queries.
    Supports cycle detection, topological sort, pathfinding.

    Adapter Registration (Rust-like isolated pattern):
        Each Graph subclass has its own independent adapter registry. No auto-registration.
        Must explicitly register adapters on each class that needs them:

        ```python
        from pydapter.adapters import TomlAdapter


        class CustomGraph(Graph):
            pass


        # Must register explicitly (no inheritance from parent)
        CustomGraph.register_adapter(TomlAdapter)
        custom_graph.adapt_to("toml")  # Now works
        ```

        This prevents adapter pollution and ensures explicit control per class.
    """

    nodes: Pile[Node] = Field(
        default_factory=lambda: Pile(item_type=Node),
        description="Node storage via Pile",
    )
    edges: Pile[Edge] = Field(
        default_factory=lambda: Pile(item_type=Edge),
        description="Edge storage via Pile",
    )
    _out_edges: dict[UUID, set[UUID]] = PrivateAttr(default_factory=dict)
    _in_edges: dict[UUID, set[UUID]] = PrivateAttr(default_factory=dict)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @field_validator("nodes", "edges", mode="wrap")
    @classmethod
    def _deserialize_nodes_edges(cls, v: Any, handler) -> Pile:
        """Deserialize nodes/edges from dict format."""
        if isinstance(v, Pile):
            return v
        if isinstance(v, dict):
            return Pile.from_dict(v)
        # Let Pydantic handle other cases (like default_factory)
        return handler(v)

    @model_validator(mode="after")
    def _rebuild_adjacency_after_init(self) -> Graph:
        """Rebuild adjacency lists after model initialization."""
        self._rebuild_adjacency()
        return self

    def _rebuild_adjacency(self) -> None:
        """Rebuild adjacency lists from nodes and edges."""
        self._out_edges = {node_id: set() for node_id in self.nodes.keys()}  # noqa: SIM118
        self._in_edges = {node_id: set() for node_id in self.nodes.keys()}  # noqa: SIM118

        for edge_id in self.edges.keys():  # noqa: SIM118
            edge = self.edges[edge_id]
            if edge.head in self._out_edges:
                self._out_edges[edge.head].add(edge_id)
            if edge.tail in self._in_edges:
                self._in_edges[edge.tail].add(edge_id)

    def _check_node_exists(self, node_id: UUID) -> Node:
        """Verify node exists, re-raising NotFoundError with graph context.

        Args:
            node_id: Node UUID to check

        Returns:
            Node if found

        Raises:
            NotFoundError: With graph context and preserved metadata
        """
        try:
            return self.nodes[node_id]
        except NotFoundError as e:
            raise NotFoundError(
                f"Node {node_id} not found in graph",
                details=e.details,
                retryable=e.retryable,
                cause=e,
            )

    def _check_edge_exists(self, edge_id: UUID) -> Edge:
        """Verify edge exists, re-raising NotFoundError with graph context.

        Args:
            edge_id: Edge UUID to check

        Returns:
            Edge if found

        Raises:
            NotFoundError: With graph context and preserved metadata
        """
        try:
            return self.edges[edge_id]
        except NotFoundError as e:
            raise NotFoundError(
                f"Edge {edge_id} not found in graph",
                details=e.details,
                retryable=e.retryable,
                cause=e,
            )

    # ==================== Node Operations ====================

    @synchronized
    def add_node(self, node: Node) -> None:
        """Add node to graph. Raises ExistsError if already exists.

        Thread-safe: Uses @synchronized to ensure atomic operation across
        nodes.add() and adjacency dict initialization.
        """
        self.nodes.add(node)
        self._out_edges[node.id] = set()
        self._in_edges[node.id] = set()

    @synchronized
    def remove_node(self, node_id: UUID | Node) -> Node:
        """Remove node and all connected edges. Raises NotFoundError if not found.

        Thread-safe: Uses @synchronized with RLock to allow nested calls to
        remove_edge(). Ensures atomic operation across edge removal, dict
        cleanup, and node removal.
        """
        nid = self._coerce_id(node_id)

        # Verify node exists before removing edges
        self._check_node_exists(nid)

        # Remove all connected edges
        for edge_id in list(self._in_edges[nid]):
            self.remove_edge(edge_id)
        for edge_id in list(self._out_edges[nid]):
            self.remove_edge(edge_id)

        # Remove adjacency entries
        del self._in_edges[nid]
        del self._out_edges[nid]

        # Remove and return node
        return self.nodes.remove(nid)

    # ==================== Edge Operations ====================

    @synchronized
    def add_edge(self, edge: Edge) -> None:
        """Add edge to graph. Raises NotFoundError if head/tail missing, ExistsError if already exists.

        Thread-safe: Uses @synchronized to ensure atomic operation across
        edges.add() and adjacency list updates. Critical for Rust port and
        Python 3.13+ nogil where GIL won't protect dict operations.
        """
        if edge.head not in self.nodes:
            raise NotFoundError(f"Head node {edge.head} not in graph")
        if edge.tail not in self.nodes:
            raise NotFoundError(f"Tail node {edge.tail} not in graph")

        self.edges.add(edge)
        self._out_edges[edge.head].add(edge.id)
        self._in_edges[edge.tail].add(edge.id)

    @synchronized
    def remove_edge(self, edge_id: UUID | Edge) -> Edge:
        """Remove edge from graph. Raises NotFoundError if not found.

        Thread-safe: Uses @synchronized to ensure atomic operation across
        adjacency dict updates and edges.remove(). RLock allows nested calls
        from remove_node().
        """
        eid = self._coerce_id(edge_id)
        edge = self._check_edge_exists(eid)

        self._out_edges[edge.head].discard(eid)
        self._in_edges[edge.tail].discard(eid)

        return self.edges.remove(eid)

    # ==================== Graph Queries ====================

    def get_predecessors(self, node_id: UUID | Node) -> list[Node]:
        """Get all nodes with edges pointing to this node."""
        nid = self._coerce_id(node_id)
        predecessors = []
        for edge_id in self._in_edges.get(nid, set()):
            edge = self.edges[edge_id]
            predecessors.append(self.nodes[edge.head])
        return predecessors

    def get_successors(self, node_id: UUID | Node) -> list[Node]:
        """Get all nodes this node points to."""
        nid = self._coerce_id(node_id)
        successors = []
        for edge_id in self._out_edges.get(nid, set()):
            edge = self.edges[edge_id]
            successors.append(self.nodes[edge.tail])
        return successors

    def get_node_edges(
        self,
        node_id: UUID | Node,
        direction: Literal["in", "out", "both"] = "both",
    ) -> list[Edge]:
        """Get edges connected to node.

        Args:
            node_id: Node ID or Node
            direction: in/out/both

        Raises:
            ValueError: If invalid direction
        """
        if direction not in {"in", "out", "both"}:
            raise ValueError(f"Invalid direction: {direction}")

        nid = self._coerce_id(node_id)
        result = []

        if direction in {"in", "both"}:
            for edge_id in self._in_edges.get(nid, set()):
                result.append(self.edges[edge_id])

        if direction in {"out", "both"}:
            for edge_id in self._out_edges.get(nid, set()):
                result.append(self.edges[edge_id])

        return result

    def get_heads(self) -> list[Node]:
        """Get all nodes with no incoming edges (source nodes)."""
        return [self.nodes[nid] for nid, in_edges in self._in_edges.items() if not in_edges]

    def get_tails(self) -> list[Node]:
        """Get all nodes with no outgoing edges (sink nodes)."""
        return [self.nodes[nid] for nid, out_edges in self._out_edges.items() if not out_edges]

    # ==================== Graph Algorithms ====================

    def is_acyclic(self) -> bool:
        """Check if graph is acyclic using three-color DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {nid: WHITE for nid in self.nodes.keys()}  # noqa: SIM118

        def dfs(node_id: UUID) -> bool:
            """DFS visit. Returns True if acyclic, False if cycle found."""
            colors[node_id] = GRAY

            for edge_id in self._out_edges[node_id]:
                neighbor_id = self.edges[edge_id].tail
                if colors[neighbor_id] == GRAY:
                    # Back edge -> cycle detected
                    return False
                if colors[neighbor_id] == WHITE and not dfs(neighbor_id):
                    return False

            colors[node_id] = BLACK
            return True

        # Check all components
        return all(
            not (colors[node_id] == WHITE and not dfs(node_id))
            for node_id in self.nodes.keys()  # noqa: SIM118
        )

    def topological_sort(self) -> list[Node]:
        """Topological sort using Kahn's algorithm. Raises ValueError if cyclic."""
        if not self.is_acyclic():
            raise ValueError("Cannot topologically sort graph with cycles")

        # Calculate in-degrees
        in_degree = {nid: len(edges) for nid, edges in self._in_edges.items()}

        # Queue of nodes with no incoming edges
        queue: deque[UUID] = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result: list[Node] = []

        while queue:
            node_id = queue.popleft()
            result.append(self.nodes[node_id])

            # Reduce in-degree of neighbors
            for edge_id in self._out_edges[node_id]:
                neighbor_id = self.edges[edge_id].tail
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        return result

    async def find_path(
        self,
        start: UUID | Node,
        end: UUID | Node,
        check_conditions: bool = False,
    ) -> list[Edge] | None:
        """Find path from start to end using BFS. Returns edges or None if no path."""
        start_id = self._coerce_id(start)
        end_id = self._coerce_id(end)

        if start_id not in self.nodes or end_id not in self.nodes:
            raise NotFoundError("Start or end node not in graph")

        # BFS with parent tracking
        queue: deque[UUID] = deque([start_id])
        parent: dict[UUID, tuple[UUID, UUID]] = {}  # {node_id: (parent_id, edge_id)}
        visited = {start_id}

        while queue:
            current_id = queue.popleft()

            if current_id == end_id:
                # Reconstruct path
                path = []
                node_id = end_id
                while node_id in parent:
                    parent_id, edge_id = parent[node_id]
                    path.append(self.edges[edge_id])
                    node_id = parent_id
                return list(reversed(path))

            # Explore neighbors
            for edge_id in self._out_edges[current_id]:
                edge: Edge = self.edges[edge_id]
                neighbor_id = edge.tail

                if neighbor_id not in visited:
                    # Check condition if requested
                    if check_conditions and not await edge.check_condition():
                        continue

                    visited.add(neighbor_id)
                    parent[neighbor_id] = (current_id, edge_id)
                    queue.append(neighbor_id)

        return None  # No path found

    def __contains__(self, item: object) -> bool:
        """Check if node or edge is in graph."""
        if isinstance(item, Node):
            return item in self.nodes
        if isinstance(item, Edge):
            return item in self.edges
        if isinstance(item, UUID):
            return item in self.nodes or item in self.edges
        return False

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    # ==================== Serialization ====================

    @override
    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
        meta_key: str | None = None,
        item_meta_key: str | None = None,
        item_created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize graph with manual Pile field handling.

        Args:
            mode: python/json/db
            created_at_format: Timestamp format for Graph
            meta_key: Rename Graph metadata field
            item_meta_key: Pass to Pile.to_dict for node/edge metadata
            item_created_at_format: Pass to Pile.to_dict for node/edge timestamps
            **kwargs: Passed to model_dump()
        """
        # Merge exclude set with any user-provided exclude
        exclude = kwargs.pop("exclude", set())
        if isinstance(exclude, set):
            exclude = exclude | {"nodes", "edges"}
        else:
            exclude = set(exclude) | {"nodes", "edges"}

        # Get base Element serialization, excluding nodes and edges
        data = super().to_dict(
            mode=mode,
            created_at_format=created_at_format,
            meta_key=meta_key,
            exclude=exclude,
            **kwargs,
        )

        # Manually serialize Pile fields with item parameters
        data["nodes"] = self.nodes.to_dict(
            mode=mode,
            item_meta_key=item_meta_key,
            item_created_at_format=item_created_at_format,
        )
        data["edges"] = self.edges.to_dict(
            mode=mode,
            item_meta_key=item_meta_key,
            item_created_at_format=item_created_at_format,
        )

        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        meta_key: str | None = None,
        item_meta_key: str | None = None,
        **kwargs: Any,
    ) -> Graph:
        """Deserialize Graph from dict.

        Args:
            data: Serialized graph data
            meta_key: Restore Graph metadata from this key (db compatibility)
            item_meta_key: Pass to Pile.from_dict for node/edge deserialization
            **kwargs: Additional arguments
        """
        from .pile import Pile

        # Make a copy to avoid mutating input
        data = data.copy()

        # Restore metadata from custom key if specified (db mode deserialization)
        if meta_key and meta_key in data:
            data["metadata"] = data.pop(meta_key)

        # Extract, deserialize, and restore nodes and edges Piles
        nodes_data = data.pop("nodes", None)
        edges_data = data.pop("edges", None)

        # Deserialize Piles and put them back in data for proper construction
        if nodes_data:
            data["nodes"] = Pile.from_dict(
                nodes_data, meta_key=item_meta_key, item_meta_key=item_meta_key
            )
        if edges_data:
            data["edges"] = Pile.from_dict(
                edges_data, meta_key=item_meta_key, item_meta_key=item_meta_key
            )

        # Create graph with all fields properly deserialized
        graph = cls.model_validate(data, **kwargs)

        return graph

    # ==================== Adapter Methods ====================

    @classmethod
    def register_adapter(cls, adapter: Any) -> None:  # pragma: no cover
        """Register adapter for this class."""
        super().register_adapter(adapter)

    @classmethod
    def register_async_adapter(cls, adapter: Any) -> None:  # pragma: no cover
        """Register async adapter for this class."""
        super().register_async_adapter(adapter)

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Convert to external format via pydapter adapter.

        Args:
            obj_key: Adapter key (e.g., "neo4j")
            many: Whether to adapt multiple Graph instances
            **kwargs: Passed to adapter

        Returns:
            Adapted object (format depends on adapter)
        """
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return super().adapt_to(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Graph:
        """Create from external format via pydapter adapter.

        Args:
            obj: Source object
            obj_key: Adapter key (e.g., "neo4j")
            many: Whether to deserialize multiple Graph instances
            **kwargs: Passed to adapter

        Returns:
            Graph instance
        """
        kwargs.setdefault("adapt_meth", "from_dict")
        return super().adapt_from(obj, obj_key=obj_key, many=many, **kwargs)

    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Async convert to external format via pydapter async adapter.

        Args:
            obj_key: Adapter key
            many: Whether to adapt multiple Graph instances
            **kwargs: Passed to adapter

        Returns:
            Adapted object
        """
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return await super().adapt_to_async(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    async def adapt_from_async(
        cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any
    ) -> Graph:
        """Async create from external format via pydapter async adapter.

        Args:
            obj: Source object
            obj_key: Adapter key
            many: Whether to deserialize multiple Graph instances
            **kwargs: Passed to adapter

        Returns:
            Graph instance
        """
        kwargs.setdefault("adapt_meth", "from_dict")
        return await super().adapt_from_async(obj, obj_key=obj_key, many=many, **kwargs)

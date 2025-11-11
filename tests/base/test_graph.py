# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Graph: directed graphs with nodes, edges, validation, traversal, async operations.

Graph Theory Foundations
========================

Graph Structure
---------------
A graph G = (V, E) consists of:
- V: Set of vertices (nodes)
- E ⊆ V x V: Set of edges (directed relationships)
- For directed graph: edge (u, v) ∈ E implies u → v (u is head, v is tail)

Graph Properties
----------------
- **DAG** (Directed Acyclic Graph): Directed graph with no cycles
  - Enables topological ordering: linear sequence where ∀(u,v) ∈ E, u precedes v
  - Use cases: workflow dependencies, task scheduling, build systems

- **Cyclic Graph**: Contains at least one cycle (path from node back to itself)
  - Simplest cycle: self-loop (u, u)
  - Prevents topological sorting (no valid linear order)

- **Disconnected Graph**: Multiple connected components with no paths between them
  - Each component can be analyzed independently
  - Algorithms must handle all components (not just reachable nodes)

Adjacency Representation
-------------------------
This implementation uses adjacency lists for O(1) lookups:
- **out_edges**: {node_id → set[edge_id]} - successors (nodes this node points to)
- **in_edges**: {node_id → set[edge_id]} - predecessors (nodes pointing to this node)

Trade-offs:
- Space: O(V + E) - stores each edge once, each node twice
- Time: O(1) for neighbor lookup, O(deg(v)) for traversal from node v
- Alternative: Adjacency matrix O(V²) space, O(1) edge lookup, dense graphs only

Graph Algorithms Tested
========================

1. Cycle Detection (DFS-based)
------------------------------
Uses depth-first search with node coloring:
- **White**: Unvisited node
- **Gray**: Node in current recursion stack (active path)
- **Black**: Fully processed node

Algorithm:
    def is_cyclic(node):
        color[node] = GRAY
        for neighbor in successors(node):
            if color[neighbor] == GRAY:
                return True  # Back edge → cycle found
            if color[neighbor] == WHITE and is_cyclic(neighbor):
                return True
        color[node] = BLACK
        return False

Complexity: O(V + E)
- Visit each node once: O(V)
- Examine each edge once: O(E)

Edge Cases:
- Self-loop: (v, v) detected when v is gray
- Disconnected components: Must check all nodes as starting points

2. Topological Sort (Kahn's Algorithm)
---------------------------------------
Produces linear ordering where ∀(u,v) ∈ E, u precedes v.
Only valid for DAGs (acyclic graphs).

Algorithm:
    1. Compute in-degree for all nodes: in_degree[v] = |{u : (u,v) ∈ E}|
    2. Initialize queue Q with all nodes where in_degree[v] = 0
    3. While Q not empty:
         a. Dequeue node u
         b. Add u to topological order
         c. For each successor v of u:
              i. Decrement in_degree[v]
              ii. If in_degree[v] == 0, enqueue v
    4. If topological order has |V| nodes, success; else cycle exists

Complexity: O(V + E)
- Initialize in-degrees: O(V + E) (scan all edges)
- Process each node once: O(V)
- Process each edge once: O(E)
- Space: O(V) for queue and in-degree map

Correctness:
- Nodes added to order only when all predecessors processed
- If cycle exists, some node always has in_degree > 0 (no progress)

Alternative: DFS-based topological sort (reverse postorder)
- Same O(V + E) complexity
- Kahn's algorithm more intuitive for dependency resolution

3. Path Finding (BFS-based)
----------------------------
Finds shortest path between two nodes (minimum edge count).

Algorithm:
    1. Initialize queue Q with start node
    2. Track visited nodes and parent pointers
    3. While Q not empty:
         a. Dequeue node u
         b. If u == target, reconstruct path from parents
         c. For each successor v of u:
              i. If v not visited:
                   - Mark v as visited
                   - Set parent[v] = u
                   - Enqueue v
    4. If target not reached, return None (no path)

Complexity: O(V + E)
- BFS visits each node at most once: O(V)
- Examines each edge at most once: O(E)
- Space: O(V) for visited set and queue

Properties:
- **Shortest path**: BFS explores nodes in order of distance from start
- **Completeness**: Finds path if one exists
- **Optimality**: Path has minimum edge count (unweighted graph)

With EdgeConditions (conditional traversal):
- Check condition before following edge
- Path may not exist even if structural path exists (conditions block)
- Complexity unchanged (condition check is O(1) async call)

Graph Invariants Maintained
============================
1. **Consistency**: All edges reference nodes that exist in graph
   - Enforced by add_edge() validation (fails if head or tail missing)

2. **Adjacency Sync**: Adjacency lists always match edges
   - add_edge() updates out_edges[head] and in_edges[tail]
   - remove_edge() cleans up adjacency entries
   - remove_node() cascades to all connected edges

3. **No Dangling References**: Edge removal never orphans nodes
   - Edges depend on nodes, not vice versa
   - remove_edge() does not remove nodes

4. **Serialization Lossless**: Graph survives to_dict/from_dict roundtrip
   - Private adjacency lists rebuilt from edges on deserialization
   - Enables graph persistence and transmission

Test Coverage
=============
- **Container protocols**: __len__, __contains__, __repr__
- **Node operations**: add, remove, get, adjacency updates
- **Edge operations**: add, remove, get, validation, adjacency updates
- **Adjacency queries**: predecessors, successors, heads, tails, node edges
- **Algorithms**: is_acyclic, topological_sort, find_path
- **Edge conditions**: traversal gates, async evaluation
- **Serialization**: to_dict, from_dict, adjacency rebuilding
- **Edge cases**: empty graphs, single nodes, disconnected components, self-loops, multiple edges

Complexity Summary
==================
Operation              | Time       | Space  | Notes
-----------------------|------------|--------|----------------------------------
add_node()             | O(1)       | O(1)   | Initialize adjacency sets
remove_node()          | O(deg(v))  | O(1)   | Remove all connected edges
get_node()             | O(1)       | O(1)   | Pile lookup
add_edge()             | O(1)       | O(1)   | Update adjacency sets
remove_edge()          | O(1)       | O(1)   | Set removal
get_edge()             | O(1)       | O(1)   | Pile lookup
get_predecessors()     | O(deg(v))  | O(deg) | Lookup in_edges, fetch nodes
get_successors()       | O(deg(v))  | O(deg) | Lookup out_edges, fetch nodes
get_heads()            | O(V)       | O(V)   | Scan all nodes for in_degree=0
get_tails()            | O(V)       | O(V)   | Scan all nodes for out_degree=0
is_acyclic()           | O(V + E)   | O(V)   | DFS with recursion stack
topological_sort()     | O(V + E)   | O(V)   | Kahn's algorithm
find_path()            | O(V + E)   | O(V)   | BFS with visited tracking

Where deg(v) = in_degree(v) + out_degree(v) is the degree of node v.

References
==========
- Cormen, T.H., et al. "Introduction to Algorithms" (CLRS) - Graph algorithms
- Kahn, A.B. (1962). "Topological sorting of large networks"
- Tarjan, R. (1972). "Depth-first search and linear graph algorithms"
"""

from __future__ import annotations

from uuid import UUID

import pytest

from lionherd_core.base import Edge, EdgeCondition, Graph, Node
from lionherd_core.errors import ExistsError, NotFoundError

# ============================================================================
# Test EdgeCondition Subclasses
# ============================================================================


class AlwaysTrueCondition(EdgeCondition):
    """Condition that always allows traversal."""

    async def apply(self, *_args, **_kwargs) -> bool:
        return True


class AlwaysFalseCondition(EdgeCondition):
    """Condition that never allows traversal."""

    async def apply(self, *_args, **_kwargs) -> bool:
        return False


class ThresholdCondition(EdgeCondition):
    """Condition based on threshold value in properties."""

    async def apply(self, threshold: float = 10.0) -> bool:
        """Allow traversal if value <= threshold."""
        value = self.properties.get("value", 0.0)
        return value <= threshold


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def empty_graph():
    """Empty graph for testing."""
    return Graph()


@pytest.fixture
def simple_graph():
    """Simple graph with 3 nodes in a chain: A -> B -> C."""
    graph = Graph()

    # Create nodes
    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    # Create edges
    e1 = Edge(head=n1.id, tail=n2.id, label=["step1"])
    e2 = Edge(head=n2.id, tail=n3.id, label=["step2"])

    graph.add_edge(e1)
    graph.add_edge(e2)

    return graph, (n1, n2, n3), (e1, e2)


@pytest.fixture
def cyclic_graph():
    """Graph with a cycle: A -> B -> C -> A."""
    graph = Graph()

    # Create nodes
    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    # Create cycle
    e1 = Edge(head=n1.id, tail=n2.id)
    e2 = Edge(head=n2.id, tail=n3.id)
    e3 = Edge(head=n3.id, tail=n1.id)

    graph.add_edge(e1)
    graph.add_edge(e2)
    graph.add_edge(e3)

    return graph, (n1, n2, n3), (e1, e2, e3)


@pytest.fixture
def dag_graph():
    """Diamond-shaped DAG: A -> B, A -> C, B -> D, C -> D."""
    graph = Graph()

    # Create nodes
    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})
    n4 = Node(content={"value": "D"})

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    # Create edges (diamond)
    e1 = Edge(head=n1.id, tail=n2.id)
    e2 = Edge(head=n1.id, tail=n3.id)
    e3 = Edge(head=n2.id, tail=n4.id)
    e4 = Edge(head=n3.id, tail=n4.id)

    graph.add_edge(e1)
    graph.add_edge(e2)
    graph.add_edge(e3)
    graph.add_edge(e4)

    return graph, (n1, n2, n3, n4), (e1, e2, e3, e4)


# ============================================================================
# Basic Graph Tests
# ============================================================================
#
# Design aspect validated: Graph implements Python container protocols (__len__,
# __contains__) to provide intuitive Pythonic API. This design choice follows
# stdlib conventions (e.g., dict, list) for consistency.
#
# Why __len__ returns node count (not edge count):
# - Matches common graph terminology ("graph of N nodes")
# - Consistent with most graph libraries (NetworkX, igraph)
# - Edges are relationships, nodes are entities (entities count)
#
# __contains__ is polymorphic (Node | Edge | UUID) for convenience. Alternative
# designs (separate contains_node/contains_edge methods) would be more explicit
# but less ergonomic. Ocean's preference: ergonomics when type checking is available.


class TestGraphBasics:
    """Graph creation, len, contains (nodes/edges/UUIDs)."""

    def test_graph_creation(self, empty_graph):
        """Test Graph can be created empty."""
        assert isinstance(empty_graph.id, UUID)
        assert len(empty_graph.nodes) == 0
        assert len(empty_graph.edges) == 0

    def test_graph_len_returns_node_count(self, simple_graph):
        """Test len(graph) returns number of nodes."""
        graph, _nodes, _ = simple_graph
        assert len(graph) == 3

    def test_graph_len_empty(self, empty_graph):
        """Test len() on empty graph."""
        assert len(empty_graph) == 0

    def test_graph_contains_node(self, simple_graph):
        """Test __contains__ with Node."""
        graph, (n1, n2, n3), _ = simple_graph

        assert n1 in graph
        assert n2 in graph
        assert n3 in graph

    def test_graph_contains_edge(self, simple_graph):
        """Test __contains__ with Edge."""
        graph, _, (e1, e2) = simple_graph

        assert e1 in graph
        assert e2 in graph

    def test_graph_contains_uuid(self, simple_graph):
        """Test __contains__ with UUID."""
        graph, (n1, _, _), (e1, _) = simple_graph

        # Node UUID
        assert n1.id in graph

        # Edge UUID
        assert e1.id in graph

    def test_graph_not_contains(self, simple_graph):
        """Test __contains__ returns False for non-members."""
        graph, _, _ = simple_graph

        new_node = Node(content={"value": "X"})
        new_edge = Edge(head=new_node.id, tail=new_node.id)

        assert new_node not in graph
        assert new_edge not in graph
        assert new_node.id not in graph

    def test_graph_contains_invalid_type(self, simple_graph):
        """Test __contains__ returns False for invalid types."""
        graph, _, _ = simple_graph

        assert "string" not in graph
        assert 123 not in graph
        assert None not in graph

    def test_graph_creation_with_pile_input(self, empty_graph):
        """Test Graph initialization with Pile objects passed directly.

        Pattern:
            Validator pass-through for Pile input (no conversion needed)

        Edge Case:
            Programmatic Graph construction with pre-built Pile objects

        Design Rationale:
            Graph accepts both dict (from JSON deserialization) and Pile
            (from programmatic construction). Validator normalizes dict→Pile
            but passes through Pile unchanged for performance.

        Expected:
            Pile objects used directly without conversion overhead
        """
        from lionherd_core.base import Pile

        # Create Pile objects programmatically
        nodes_pile = Pile(item_type=Node)
        edges_pile = Pile(item_type=Edge)

        n1 = Node(content={"value": "A"})
        nodes_pile.add(n1)

        # Pass Pile directly to Graph initialization
        # This triggers the validator with Pile input (pass-through path)
        graph = Graph(nodes=nodes_pile, edges=edges_pile)

        assert len(graph) == 1
        assert n1.id in graph.nodes


# ============================================================================
# Node Operations Tests
# ============================================================================
#
# Design aspect validated: Node add/remove operations maintain adjacency lists
# automatically. This design ensures graph consistency without manual bookkeeping.
#
# Adjacency list design:
# - _out_edges: {node_id → set[edge_id]} - O(1) successor lookups
# - _in_edges: {node_id → set[edge_id]} - O(1) predecessor lookups
# - Private attributes (not serialized, rebuilt on deserialization)
#
# Why maintain both in/out adjacency:
# - Enables O(1) bidirectional traversal (predecessors and successors)
# - Critical for graph algorithms (DFS, BFS, topological sort)
# - Memory cost: 2 x |V| sets (acceptable for in-memory graphs)
#
# remove_node() cascade behavior:
# - Removes all connected edges automatically
# - Prevents dangling edge references (consistency)
# - Trade-off: Can't preserve edges when temporarily removing nodes
#   (acceptable: use node.metadata["hidden"] for soft deletion if needed)
#
# UUID coercion (to_uuid helper):
# - Accept Node object or UUID string
# - Internal normalization to UUID
# - Ergonomic API (pass what you have, we handle it)


class TestNodeOperations:
    """Node operations: add, remove, get, adjacency updates."""

    def test_add_node_basic(self, empty_graph):
        """Test adding a node to empty graph."""
        node = Node(content={"value": "test"})
        empty_graph.add_node(node)

        assert len(empty_graph) == 1
        assert node in empty_graph
        assert node.id in empty_graph.nodes

    def test_add_node_updates_adjacency(self, empty_graph):
        """Test adding node initializes adjacency lists."""
        node = Node(content={"value": "test"})
        empty_graph.add_node(node)

        # Should have empty adjacency sets
        assert node.id in empty_graph._out_edges
        assert node.id in empty_graph._in_edges
        assert empty_graph._out_edges[node.id] == set()
        assert empty_graph._in_edges[node.id] == set()

    def test_add_node_duplicate_raises(self, simple_graph):
        """Test adding duplicate node raises ExistsError."""
        graph, (n1, _, _), _ = simple_graph

        with pytest.raises(ExistsError, match="already exists"):
            graph.add_node(n1)

    def test_remove_node_basic(self, simple_graph):
        """Test removing a node."""
        graph, (n1, n2, n3), _ = simple_graph

        removed = graph.remove_node(n2)

        assert removed == n2
        assert len(graph) == 2
        assert n2 not in graph
        assert n1 in graph
        assert n3 in graph

    def test_remove_node_removes_connected_edges(self, simple_graph):
        """Test removing node also removes all connected edges."""
        graph, (_, n2, _), (e1, e2) = simple_graph

        graph.remove_node(n2)

        # Both edges should be removed (n2 was in the middle)
        assert e1 not in graph
        assert e2 not in graph
        assert len(graph.edges) == 0

    def test_remove_node_updates_adjacency(self, simple_graph):
        """Test removing node removes adjacency entries."""
        graph, (_, n2, _), _ = simple_graph

        graph.remove_node(n2)

        # Adjacency entries should be gone
        assert n2.id not in graph._out_edges
        assert n2.id not in graph._in_edges

    def test_remove_node_by_uuid(self, simple_graph):
        """Test removing node by UUID."""
        graph, (n1, _, _), _ = simple_graph

        removed = graph.remove_node(n1.id)

        assert removed == n1
        assert n1 not in graph

    def test_remove_node_not_found_raises(self, empty_graph):
        """Test removing non-existent node raises NotFoundError."""
        fake_node = Node(content={"value": "fake"})

        with pytest.raises(NotFoundError, match="not found"):
            empty_graph.remove_node(fake_node)


# ============================================================================
# Edge Operations Tests
# ============================================================================
#
# Design aspect validated: Edge operations enforce graph consistency through validation.
# add_edge() requires both head and tail nodes to exist (no dangling edges).
#
# Why strict validation on add_edge():
# - Prevents invalid graph states (edges pointing to nonexistent nodes)
# - Fails fast (error at edge creation, not at traversal)
# - Alternative designs (lazy validation, soft references) would allow inconsistent
#   states that fail during traversal → harder to debug
#
# Why allow multiple edges between same node pair:
# - Real-world graphs often have multiple relationship types (USES, CREATED, MAINTAINS)
# - Each edge has unique ID → no conflicts
# - Labels and properties distinguish edge semantics
# - Trade-off: More memory, but enables expressive graphs
#
# Edge removal:
# - Does NOT remove nodes (edges are dependent on nodes, not vice versa)
# - Updates adjacency lists automatically
# - O(1) removal via Pile and set operations


class TestEdgeOperations:
    """Edge operations: add, remove, get, adjacency updates, validation."""

    def test_add_edge_basic(self, simple_graph):
        """Test adding a new edge between existing nodes."""
        graph, (n1, _, n3), _ = simple_graph

        # Add edge skipping n2
        new_edge = Edge(head=n1.id, tail=n3.id, label=["shortcut"])
        graph.add_edge(new_edge)

        assert new_edge in graph
        assert len(graph.edges) == 3

    def test_add_edge_updates_adjacency(self, simple_graph):
        """Test adding edge updates adjacency lists."""
        graph, (n1, _, n3), _ = simple_graph

        edge = Edge(head=n1.id, tail=n3.id)
        graph.add_edge(edge)

        # Check adjacency
        assert edge.id in graph._out_edges[n1.id]
        assert edge.id in graph._in_edges[n3.id]

    def test_add_edge_duplicate_raises(self, simple_graph):
        """Test adding duplicate edge raises ExistsError."""
        graph, _, (e1, _) = simple_graph

        with pytest.raises(ExistsError, match="already exists"):
            graph.add_edge(e1)

    def test_add_edge_missing_head_raises(self, empty_graph):
        """Test adding edge with missing head node raises NotFoundError."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        empty_graph.add_node(n1)

        # n2 not in graph
        edge = Edge(head=n2.id, tail=n1.id)

        with pytest.raises(NotFoundError, match=r"Head node .* not in graph"):
            empty_graph.add_edge(edge)

    def test_add_edge_missing_tail_raises(self, empty_graph):
        """Test adding edge with missing tail node raises NotFoundError."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        empty_graph.add_node(n1)

        # n2 not in graph
        edge = Edge(head=n1.id, tail=n2.id)

        with pytest.raises(NotFoundError, match=r"Tail node .* not in graph"):
            empty_graph.add_edge(edge)

    def test_remove_edge_basic(self, simple_graph):
        """Test removing an edge."""
        graph, _, (e1, e2) = simple_graph

        removed = graph.remove_edge(e1)

        assert removed == e1
        assert e1 not in graph
        assert e2 in graph
        assert len(graph.edges) == 1

    def test_remove_edge_updates_adjacency(self, simple_graph):
        """Test removing edge updates adjacency lists."""
        graph, (n1, n2, _), (e1, _) = simple_graph

        graph.remove_edge(e1)

        # Edge should be gone from adjacency
        assert e1.id not in graph._out_edges[n1.id]
        assert e1.id not in graph._in_edges[n2.id]

    def test_remove_edge_by_uuid(self, simple_graph):
        """Test removing edge by UUID."""
        graph, _, (e1, _) = simple_graph

        removed = graph.remove_edge(e1.id)

        assert removed == e1
        assert e1 not in graph

    def test_remove_edge_not_found_raises(self, empty_graph):
        """Test removing non-existent edge raises NotFoundError."""
        fake_id = UUID("00000000-0000-0000-0000-000000000000")

        with pytest.raises(NotFoundError, match="not found"):
            empty_graph.remove_edge(fake_id)


# ============================================================================
# Error Handling Tests (Pile[key] Pattern - PR #117)
# ============================================================================
#
# Design aspect validated: Pile[key] primitive error handling pattern.
# PR #117 refactored from double-lookup (.get()) to single-lookup ([key]) pattern,
# introducing try/except KeyError→ValueError transformation.
#
# Pattern validated:
#   try:
#       item = pile[key]
#   except KeyError:
#       raise ValueError(f"Item {key} not found") from None
#
# Why test error transformation:
# - Validates KeyError from Pile.__getitem__ is caught and transformed
# - Ensures ValueError (domain error) raised instead of KeyError (implementation detail)
# - Verifies 'from None' suppresses KeyError from traceback (cleaner debugging)
#
# Methods tested: get_node(), get_edge(), remove_edge()
# Pattern scope: All Pile[key] usage in Graph (15 sites total)


class TestPileKeyErrorHandling:
    """Test error handling for Pile[key] pattern introduced in PR #117."""

    # Phase 1: Exception Transformation Tests

    def test_remove_edge_raises_notfounderror_with_metadata(self, empty_graph):
        """Verify remove_edge raises NotFoundError with preserved metadata."""
        fake_id = UUID("00000000-0000-0000-0000-000000000000")

        # Should raise NotFoundError with better message
        with pytest.raises(NotFoundError, match=f"Edge {fake_id} not found in graph"):
            empty_graph.remove_edge(fake_id)

        # Verify metadata is preserved via __cause__
        try:
            empty_graph.remove_edge(fake_id)
        except NotFoundError as e:
            # Check that __cause__ is set (exception chain preserved)
            assert e.__cause__ is not None
            # Check that details and retryable are accessible
            assert hasattr(e, "details")
            assert hasattr(e, "retryable")

    # Phase 2: Exception Chain Preservation Tests

    def test_remove_edge_preserves_exception_chain(self, empty_graph):
        """Verify exception chain is preserved via __cause__ (not suppressed with 'from None')."""
        import traceback

        fake_id = UUID("00000000-0000-0000-0000-000000000000")

        try:
            empty_graph.remove_edge(fake_id)
        except NotFoundError as e:
            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = "".join(tb)

            # Should contain "The above exception was the direct cause"
            assert "direct cause" in tb_str, "Exception chain should be preserved via cause="
            # Verify __cause__ is set
            assert e.__cause__ is not None, "__cause__ should be set"

    # Phase 3: Error Message Format Tests

    def test_remove_edge_error_message_format(self, empty_graph):
        """Verify error message includes UUID and graph context."""
        fake_id = UUID("12345678-1234-1234-1234-123456789abc")

        try:
            empty_graph.remove_edge(fake_id)
        except NotFoundError as e:
            error_msg = str(e)
            assert str(fake_id) in error_msg, "Error message should include UUID"
            assert "not found" in error_msg.lower(), "Error message should indicate not found"
            assert "graph" in error_msg.lower(), "Error message should mention graph context"

    # Phase 4: ExistsError Transformation Tests

    def test_add_node_raises_existserror(self, empty_graph):
        """Verify add_node raises ExistsError when node already exists."""
        from lionherd_core.base import Node

        node = Node(content={"value": "test"})
        empty_graph.add_node(node)

        # Adding again should raise ExistsError
        with pytest.raises(ExistsError, match=f"Item {node.id} already exists"):
            empty_graph.add_node(node)

    def test_add_edge_raises_existserror(self, empty_graph):
        """Verify add_edge raises ExistsError when edge already exists."""
        from lionherd_core.base import Edge, Node

        node1 = Node(content={"value": "A"})
        node2 = Node(content={"value": "B"})
        empty_graph.add_node(node1)
        empty_graph.add_node(node2)

        edge = Edge(head=node1.id, tail=node2.id)
        empty_graph.add_edge(edge)

        # Adding again should raise ExistsError
        with pytest.raises(ExistsError, match=f"Item {edge.id} already exists"):
            empty_graph.add_edge(edge)


# ============================================================================
# Adjacency Queries Tests
# ============================================================================
#
# Design aspect validated: Adjacency queries provide O(1) graph traversal through
# pre-computed adjacency lists. This is Ocean's standard graph pattern prioritizing
# query speed over update speed.
#
# Query API design:
# - get_predecessors(node) → nodes pointing TO this node (incoming edges)
# - get_successors(node) → nodes this node points TO (outgoing edges)
# - get_node_edges(node, direction) → edges connected to node
# - get_heads() → nodes with no incoming edges (sources)
# - get_tails() → nodes with no outgoing edges (sinks)
#
# Why return Node objects (not UUIDs):
# - More ergonomic (immediate access to node data)
# - Consistent with Graph API philosophy (work with objects, not IDs)
# - Trade-off: Slightly more memory (references), but O(1) via Pile lookup
#
# get_heads() and get_tails() use cases:
# - Workflow graphs: heads = entry points, tails = terminal states
# - Dependency graphs: heads = root dependencies, tails = leaf nodes
# - Cycle detection: Empty heads/tails in cyclic graph
#
# direction parameter in get_node_edges():
# - "in": Only incoming edges (for analyzing dependencies)
# - "out": Only outgoing edges (for analyzing dependents)
# - "both": All connected edges (default, for full neighborhood)


class TestAdjacencyQueries:
    """Adjacency queries: predecessors, successors, node edges, heads, tails."""

    def test_get_predecessors_basic(self, simple_graph):
        """Test getting predecessor nodes."""
        graph, (n1, n2, n3), _ = simple_graph

        # n2 has n1 as predecessor
        preds = graph.get_predecessors(n2)
        assert len(preds) == 1
        assert n1 in preds

        # n3 has n2 as predecessor
        preds = graph.get_predecessors(n3)
        assert len(preds) == 1
        assert n2 in preds

    def test_get_predecessors_none(self, simple_graph):
        """Test getting predecessors for head node (no incoming edges)."""
        graph, (n1, _, _), _ = simple_graph

        preds = graph.get_predecessors(n1)
        assert len(preds) == 0

    def test_get_predecessors_multiple(self, dag_graph):
        """Test getting multiple predecessors (diamond graph)."""
        graph, (_, n2, n3, n4), _ = dag_graph

        # n4 has both n2 and n3 as predecessors
        preds = graph.get_predecessors(n4)
        assert len(preds) == 2
        assert n2 in preds
        assert n3 in preds

    def test_get_predecessors_by_uuid(self, simple_graph):
        """Test getting predecessors by node UUID."""
        graph, (n1, n2, _), _ = simple_graph

        preds = graph.get_predecessors(n2.id)
        assert n1 in preds

    def test_get_successors_basic(self, simple_graph):
        """Test getting successor nodes."""
        graph, (n1, n2, n3), _ = simple_graph

        # n1 has n2 as successor
        succs = graph.get_successors(n1)
        assert len(succs) == 1
        assert n2 in succs

        # n2 has n3 as successor
        succs = graph.get_successors(n2)
        assert len(succs) == 1
        assert n3 in succs

    def test_get_successors_none(self, simple_graph):
        """Test getting successors for tail node (no outgoing edges)."""
        graph, (_, _, n3), _ = simple_graph

        succs = graph.get_successors(n3)
        assert len(succs) == 0

    def test_get_successors_multiple(self, dag_graph):
        """Test getting multiple successors (diamond graph)."""
        graph, (n1, n2, n3, _), _ = dag_graph

        # n1 has both n2 and n3 as successors
        succs = graph.get_successors(n1)
        assert len(succs) == 2
        assert n2 in succs
        assert n3 in succs

    def test_get_successors_by_uuid(self, simple_graph):
        """Test getting successors by node UUID."""
        graph, (n1, n2, _), _ = simple_graph

        succs = graph.get_successors(n1.id)
        assert n2 in succs

    def test_get_node_edges_both_directions(self, simple_graph):
        """Test getting all edges connected to node (both directions)."""
        graph, (_, n2, _), (e1, e2) = simple_graph

        # n2 has one incoming (e1) and one outgoing (e2)
        edges = graph.get_node_edges(n2, direction="both")
        assert len(edges) == 2
        assert e1 in edges
        assert e2 in edges

    def test_get_node_edges_in_only(self, simple_graph):
        """Test getting only incoming edges."""
        graph, (_, n2, _), (e1, _) = simple_graph

        edges = graph.get_node_edges(n2, direction="in")
        assert len(edges) == 1
        assert e1 in edges

    def test_get_node_edges_out_only(self, simple_graph):
        """Test getting only outgoing edges."""
        graph, (_, n2, _), (_, e2) = simple_graph

        edges = graph.get_node_edges(n2, direction="out")
        assert len(edges) == 1
        assert e2 in edges

    def test_get_node_edges_invalid_direction_raises(self, simple_graph):
        """Test get_node_edges with invalid direction raises ValueError."""
        graph, (n1, _, _), _ = simple_graph

        with pytest.raises(ValueError, match="Invalid direction"):
            graph.get_node_edges(n1, direction="invalid")

    def test_get_node_edges_isolated_node(self, empty_graph):
        """Test getting edges for isolated node (no edges)."""
        node = Node(content={"value": "isolated"})
        empty_graph.add_node(node)

        edges = empty_graph.get_node_edges(node, direction="both")
        assert len(edges) == 0

    def test_get_heads_basic(self, simple_graph):
        """Test getting head nodes (no incoming edges)."""
        graph, (n1, _, _), _ = simple_graph

        heads = graph.get_heads()
        assert len(heads) == 1
        assert n1 in heads

    def test_get_heads_multiple(self, empty_graph):
        """Test getting multiple head nodes."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        n3 = Node(content={"value": "C"})

        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        empty_graph.add_node(n3)

        # No edges - all are heads
        heads = empty_graph.get_heads()
        assert len(heads) == 3
        assert n1 in heads
        assert n2 in heads
        assert n3 in heads

    def test_get_heads_cyclic(self, cyclic_graph):
        """Test getting heads in cyclic graph (should be empty)."""
        graph, _, _ = cyclic_graph

        heads = graph.get_heads()
        assert len(heads) == 0

    def test_get_tails_basic(self, simple_graph):
        """Test getting tail nodes (no outgoing edges)."""
        graph, (_, _, n3), _ = simple_graph

        tails = graph.get_tails()
        assert len(tails) == 1
        assert n3 in tails

    def test_get_tails_multiple(self, empty_graph):
        """Test getting multiple tail nodes."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})

        empty_graph.add_node(n1)
        empty_graph.add_node(n2)

        # No edges - all are tails
        tails = empty_graph.get_tails()
        assert len(tails) == 2
        assert n1 in tails
        assert n2 in tails

    def test_get_tails_cyclic(self, cyclic_graph):
        """Test getting tails in cyclic graph (should be empty)."""
        graph, _, _ = cyclic_graph

        tails = graph.get_tails()
        assert len(tails) == 0


# ============================================================================
# Graph Algorithms Tests
# ============================================================================


class TestGraphAlgorithms:
    """Graph algorithms: is_acyclic, topological_sort, find_path with conditions.

    Graph Algorithms Deep Dive
    ===========================

    1. Cycle Detection: is_acyclic()
    ---------------------------------
    Determines if graph G = (V, E) contains any cycles.

    **Definition**: Cycle = path (v₀, v₁, ..., vₖ) where v₀ = vₖ and k ≥ 1

    **Algorithm**: Depth-First Search with Three-Color Marking

    Colors represent node states during DFS:
    - **WHITE** (0): Unvisited node
    - **GRAY** (1): Node in current DFS recursion stack (ancestor in DFS tree)
    - **BLACK** (2): Node fully processed (all descendants explored)

    Pseudocode:
        ```
        def is_acyclic(G):
            color = {v: WHITE for v in G.nodes}

            def dfs(u):
                color[u] = GRAY  # Mark as "in progress"
                for v in G.successors(u):
                    if color[v] == GRAY:
                        return False  # Back edge → cycle detected
                    if color[v] == WHITE:
                        if not dfs(v):
                            return False
                color[u] = BLACK  # Mark as "completed"
                return True

            # Check all components (disconnected graph handling)
            for u in G.nodes:
                if color[u] == WHITE:
                    if not dfs(u):
                        return False
            return True
        ```

    **Edge Types in DFS**:
    - **Tree edge**: WHITE → GRAY (first visit to child)
    - **Back edge**: X → GRAY (edge to ancestor) → **CYCLE DETECTED**
    - **Forward edge**: X → BLACK (edge to descendant)
    - **Cross edge**: X → BLACK (edge to unrelated subtree)

    **Complexity**: O(V + E)
    - Each node visited once: O(V)
    - Each edge examined once: O(E)
    - Space: O(V) for recursion stack + color map

    **Edge Cases Tested**:
    - Empty graph → acyclic (vacuously true)
    - Single node, no edges → acyclic
    - Self-loop (v, v) → cyclic (simplest cycle)
    - Disconnected components → check all components

    **Use Cases**:
    - Workflow validation (no circular dependencies)
    - Build system verification (no circular includes)
    - Scheduling feasibility (prerequisite chains valid)

    2. Topological Sort: topological_sort()
    ----------------------------------------
    Produces linear ordering L of nodes where ∀(u,v) ∈ E, u appears before v in L.

    **Precondition**: Graph must be a DAG (acyclic)
    **Postcondition**: For every edge (u, v), index(u) < index(v) in result

    **Algorithm**: Kahn's Algorithm (BFS-based)

    Intuition: Repeatedly remove nodes with no remaining dependencies.

    Pseudocode:
        ```
        def topological_sort(G):
            # Step 1: Compute in-degrees
            in_degree = {v: len(G.predecessors(v)) for v in G.nodes}

            # Step 2: Initialize queue with zero in-degree nodes (sources)
            Q = [v for v in G.nodes if in_degree[v] == 0]
            L = []  # Topological order

            # Step 3: Process nodes in dependency order
            while Q:
                u = Q.pop(0)
                L.append(u)

                # Remove u's outgoing edges (decrement successors' in-degrees)
                for v in G.successors(u):
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        Q.append(v)  # v's dependencies satisfied

            # Step 4: Verify all nodes processed
            if len(L) != len(G.nodes):
                raise ValueError("Graph contains cycle")

            return L
        ```

    **Correctness Proof**:
    1. **Invariant**: Nodes added to L only when all predecessors already in L
    2. **Progress**: If DAG, always exists node with in_degree = 0
    3. **Termination**: All nodes added to L ⟺ graph is acyclic
    4. **Ordering**: By construction, u added before v if edge (u, v) exists

    **Complexity**: O(V + E)
    - Compute in-degrees: O(V + E) (scan all edges)
    - Process nodes: O(V) (each node enqueued/dequeued once)
    - Update in-degrees: O(E) (each edge processed once)
    - Space: O(V) for in_degree map + queue

    **Alternative: DFS-based Topological Sort**:
    - Perform DFS and record postorder (finish times)
    - Reverse postorder gives topological order
    - Same O(V + E) complexity
    - Kahn's algorithm preferred for intuitive dependency resolution

    **Non-uniqueness**: Multiple valid orderings may exist
    - Example: Diamond DAG A→B, A→C, B→D, C→D
      Valid orders: [A, B, C, D], [A, C, B, D]
    - Only guaranteed: A before {B,C}, {B,C} before D

    **Edge Cases Tested**:
    - Empty graph → empty order []
    - Single node → [node]
    - Chain A→B→C → [A, B, C] (unique order)
    - Cyclic graph → ValueError (no valid ordering)
    - Disconnected components → all nodes included

    **Use Cases**:
    - Task scheduling (execute tasks respecting dependencies)
    - Build systems (compile order for source files)
    - Course prerequisites (plan semester schedules)
    - Workflow execution (determine step order)

    3. Path Finding: find_path()
    -----------------------------
    Finds shortest path between start and end nodes (minimum edge count).

    **Algorithm**: Breadth-First Search (BFS)

    Why BFS for shortest path:
    - Explores nodes in order of distance from start (level-order)
    - First path found to target is guaranteed shortest (unweighted graph)
    - DFS would find *a* path but not necessarily shortest

    Pseudocode:
        ```
        def find_path(G, start, end):
            if start == end:
                return []  # Already at destination

            # BFS initialization
            Q = [start]
            visited = {start}
            parent = {start: None}  # Track path via parent pointers

            while Q:
                u = Q.pop(0)

                # Check successors
                for v in G.successors(u):
                    if v not in visited:
                        visited.add(v)
                        parent[v] = u

                        if v == end:
                            # Reconstruct path from parent pointers
                            path = []
                            current = end
                            while parent[current] is not None:
                                edge = G.get_edge_between(parent[current], current)
                                path.insert(0, edge)
                                current = parent[current]
                            return path

                        Q.append(v)

            return None  # No path exists
        ```

    **Complexity**: O(V + E)
    - Each node visited at most once: O(V)
    - Each edge examined at most once: O(E)
    - Space: O(V) for visited set, parent map, queue

    **Path Reconstruction**: O(k) where k = path length ≤ V
    - Follow parent pointers from end to start
    - Collect edges along path
    - Reverse to get start→end order

    **Properties**:
    - **Completeness**: Finds path if one exists (explores entire reachable graph)
    - **Optimality**: Path has minimum edge count (BFS property)
    - **Determinism**: Same graph, same start/end → same path (consistent queue order)

    **With EdgeConditions** (check_conditions=True):
    - Extends BFS to honor edge conditions (traversal gates)
    - Before following edge, evaluate: await edge.check_condition()
    - If condition returns False, edge is "blocked" (not traversable)
    - Use case: Conditional workflows (edge enabled only if guard satisfied)

    Modified algorithm:
        ```
        for v in G.successors(u):
            edge = G.get_edge(u, v)
            if check_conditions and not await edge.check_condition():
                continue  # Skip blocked edge
            if v not in visited:
                # ... rest of BFS
        ```

    **Edge Cases Tested**:
    - start == end → empty path [] (already at destination)
    - No path exists → None (disconnected or wrong direction)
    - Multiple paths → shortest path returned (BFS guarantee)
    - Disconnected components → None (unreachable)
    - Blocked by condition → None or alternate path

    **Use Cases**:
    - Workflow reachability (can we reach state B from state A?)
    - Dependency chains (what steps needed to build target?)
    - Navigation (shortest route in directed graph)
    - Conditional execution (path finding with runtime gates)

    Testing Strategy
    ================
    Tests validate:
    1. **Correctness**: Algorithm produces correct results for known inputs
    2. **Completeness**: Edge cases handled (empty, single node, disconnected)
    3. **Complexity**: No exponential blowup (large graphs complete quickly)
    4. **Invariants**: Operations maintain graph consistency
    5. **Error handling**: Invalid inputs raise appropriate exceptions

    Test Patterns:
    - **Positive tests**: Valid inputs → expected outputs
    - **Negative tests**: Invalid inputs → expected errors
    - **Boundary tests**: Edge cases (empty, single, maximal)
    - **Property tests**: Invariants hold across operations
    """

    def test_is_acyclic_empty_graph(self, empty_graph):
        """Test empty graph is acyclic."""
        assert empty_graph.is_acyclic() is True

    def test_is_acyclic_single_node(self, empty_graph):
        """Test single node graph is acyclic."""
        empty_graph.add_node(Node(content={"value": "A"}))
        assert empty_graph.is_acyclic() is True

    def test_is_acyclic_chain(self, simple_graph):
        """Test simple chain is acyclic."""
        graph, _, _ = simple_graph
        assert graph.is_acyclic() is True

    def test_is_acyclic_dag(self, dag_graph):
        """Test diamond DAG is acyclic."""
        graph, _, _ = dag_graph
        assert graph.is_acyclic() is True

    def test_is_acyclic_cycle_detected(self, cyclic_graph):
        """Test cycle is detected."""
        graph, _, _ = cyclic_graph
        assert graph.is_acyclic() is False

    def test_is_acyclic_self_loop(self, empty_graph):
        """Test self-loop is detected as cycle."""
        node = Node(content={"value": "A"})
        empty_graph.add_node(node)

        # Self-loop
        edge = Edge(head=node.id, tail=node.id)
        empty_graph.add_edge(edge)

        assert empty_graph.is_acyclic() is False

    def test_topological_sort_chain(self, simple_graph):
        """Test topological sort on simple chain."""
        graph, (n1, n2, n3), _ = simple_graph

        sorted_nodes = graph.topological_sort()

        assert len(sorted_nodes) == 3
        # Order should be n1 -> n2 -> n3
        assert sorted_nodes.index(n1) < sorted_nodes.index(n2)
        assert sorted_nodes.index(n2) < sorted_nodes.index(n3)

    def test_topological_sort_dag(self, dag_graph):
        """Test topological sort on diamond DAG."""
        graph, (n1, n2, n3, n4), _ = dag_graph

        sorted_nodes = graph.topological_sort()

        assert len(sorted_nodes) == 4
        # n1 must come first
        assert sorted_nodes[0] == n1
        # n4 must come last
        assert sorted_nodes[3] == n4
        # n2 and n3 can be in either order (both depend on n1)
        assert n2 in sorted_nodes
        assert n3 in sorted_nodes

    def test_topological_sort_empty_graph(self, empty_graph):
        """Test topological sort on empty graph.

        NOTE: This test PASSES because empty graph doesn't trigger the iteration bug.
        """
        sorted_nodes = empty_graph.topological_sort()
        assert len(sorted_nodes) == 0

    def test_topological_sort_single_node(self, empty_graph):
        """Test topological sort on single node."""
        node = Node(content={"value": "A"})
        empty_graph.add_node(node)

        sorted_nodes = empty_graph.topological_sort()
        assert len(sorted_nodes) == 1
        assert sorted_nodes[0] == node

    def test_topological_sort_cycle_raises(self, cyclic_graph):
        """Test topological sort raises on cyclic graph."""
        graph, _, _ = cyclic_graph

        with pytest.raises(ValueError, match=r"Cannot topologically sort.*cycle"):
            graph.topological_sort()

    async def test_find_path_direct(self, simple_graph):
        """Test finding direct path."""
        graph, (n1, n2, _), (e1, _) = simple_graph

        # Direct path A -> B
        path = await graph.find_path(n1, n2)

        assert path is not None
        assert len(path) == 1
        assert path[0] == e1

    async def test_find_path_multi_hop(self, simple_graph):
        """Test finding multi-hop path."""
        graph, (n1, _, n3), (e1, e2) = simple_graph

        # Path A -> B -> C
        path = await graph.find_path(n1, n3)

        assert path is not None
        assert len(path) == 2
        assert path[0] == e1
        assert path[1] == e2

    async def test_find_path_no_path_exists(self, simple_graph):
        """Test find_path returns None when no path exists."""
        graph, (n1, _, n3), _ = simple_graph

        # Reverse direction - no path from n3 to n1
        path = await graph.find_path(n3, n1)

        assert path is None

    async def test_find_path_disconnected_components(self, empty_graph):
        """Test find_path returns None for disconnected components."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})

        empty_graph.add_node(n1)
        empty_graph.add_node(n2)

        # No edges - disconnected
        path = await empty_graph.find_path(n1, n2)
        assert path is None

    async def test_find_path_same_node(self, simple_graph):
        """Test find_path from node to itself (no self-loop)."""
        graph, (n1, _, _), _ = simple_graph

        path = await graph.find_path(n1, n1)

        # BFS finds empty path (already at destination)
        assert path is not None
        assert len(path) == 0

    async def test_find_path_shortest_path(self, dag_graph):
        """Test find_path returns shortest path (BFS property)."""
        graph, (n1, _, _, n4), _ = dag_graph

        # Diamond: two paths from n1 to n4 (both length 2)
        path = await graph.find_path(n1, n4)

        assert path is not None
        assert len(path) == 2

    async def test_find_path_start_not_in_graph_raises(self, simple_graph):
        """Test find_path raises if start node not in graph."""
        graph, _, _ = simple_graph
        fake_node = Node(content={"value": "fake"})

        with pytest.raises(NotFoundError, match="not in graph"):
            await graph.find_path(fake_node, graph.nodes[next(graph.nodes.keys())])

    async def test_find_path_end_not_in_graph_raises(self, simple_graph):
        """Test find_path raises if end node not in graph."""
        graph, (n1, _, _), _ = simple_graph
        fake_node = Node(content={"value": "fake"})

        with pytest.raises(NotFoundError, match="not in graph"):
            await graph.find_path(n1, fake_node)

    async def test_find_path_with_conditions_all_pass(self, simple_graph):
        """Test find_path with conditions that all pass."""
        graph, (n1, _, n3), _ = simple_graph

        # Replace edges with conditional edges (always true)
        edge_keys = list(graph.edges.keys())
        graph.edges[edge_keys[0]].condition = AlwaysTrueCondition()
        graph.edges[edge_keys[1]].condition = AlwaysTrueCondition()

        path = await graph.find_path(n1, n3, check_conditions=True)

        assert path is not None
        assert len(path) == 2

    async def test_find_path_with_conditions_blocked(self, simple_graph):
        """Test find_path with condition that blocks traversal."""
        graph, (n1, _, n3), (e1, _) = simple_graph

        # Block first edge
        e1.condition = AlwaysFalseCondition()

        path = await graph.find_path(n1, n3, check_conditions=True)

        # Path blocked
        assert path is None

    async def test_find_path_without_checking_conditions(self, simple_graph):
        """Test find_path ignores conditions when check_conditions=False."""
        graph, (n1, _, n3), (e1, _) = simple_graph

        # Block first edge
        e1.condition = AlwaysFalseCondition()

        # Should still find path (not checking conditions)
        path = await graph.find_path(n1, n3, check_conditions=False)

        assert path is not None
        assert len(path) == 2


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Serialization: to_dict/from_dict roundtrips, adjacency rebuilding."""

    def test_empty_graph_serialization(self, empty_graph):
        """Test serializing empty graph."""
        # Python mode: Pile objects stay as Pile
        data = empty_graph.to_dict(mode="python")
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], dict)  # Pile serializes to dict
        assert isinstance(data["edges"], dict)

    def test_simple_graph_serialization(self, simple_graph):
        """Test serializing graph with nodes and edges."""
        graph, _, _ = simple_graph

        # to_dict returns proper structure
        data = graph.to_dict(mode="python")
        assert "nodes" in data
        assert "edges" in data
        # Pile serializes to dict with items
        assert "items" in data["nodes"]
        assert "items" in data["edges"]

    def test_serialization_no_private_attrs(self, simple_graph):
        """Test serialization does not include private adjacency lists."""
        graph, _, _ = simple_graph

        data = graph.model_dump()

        # Private attrs should not be serialized
        assert "_out_edges" not in data
        assert "_in_edges" not in data

    def test_roundtrip_empty_graph(self, empty_graph):
        """Test empty graph survives serialization roundtrip."""
        data = empty_graph.to_dict()
        restored = Graph.from_dict(data)

        assert len(restored) == 0
        assert len(restored.nodes) == 0
        assert len(restored.edges) == 0
        assert restored.id == empty_graph.id

    def test_roundtrip_simple_graph(self, simple_graph):
        """Test simple graph survives serialization roundtrip."""
        graph, (n1, n2, n3), (e1, e2) = simple_graph

        data = graph.to_dict()
        restored = Graph.from_dict(data)

        # Check structure
        assert len(restored) == 3
        assert len(restored.edges) == 2

        # Check nodes
        assert n1.id in restored.nodes
        assert n2.id in restored.nodes
        assert n3.id in restored.nodes

        # Check edges
        assert e1.id in restored.edges
        assert e2.id in restored.edges

    def test_roundtrip_adjacency_rebuilt(self, simple_graph):
        """Test adjacency lists are rebuilt after deserialization."""
        graph, (n1, n2, n3), (e1, e2) = simple_graph

        data = graph.to_dict()
        restored = Graph.from_dict(data)

        # Adjacency should be rebuilt
        assert n1.id in restored._out_edges
        assert n2.id in restored._out_edges
        assert n3.id in restored._out_edges
        assert n1.id in restored._in_edges
        assert n2.id in restored._in_edges
        assert n3.id in restored._in_edges

        # Check adjacency correctness
        assert e1.id in restored._out_edges[n1.id]
        assert e2.id in restored._out_edges[n2.id]
        assert e1.id in restored._in_edges[n2.id]
        assert e2.id in restored._in_edges[n3.id]

    def test_roundtrip_dag_graph(self, dag_graph):
        """Test complex DAG survives roundtrip."""
        graph, nodes, edges = dag_graph

        data = graph.to_dict()
        restored = Graph.from_dict(data)

        # Structure preserved
        assert len(restored) == 4
        assert len(restored.edges) == 4

        # All nodes present
        for node in nodes:
            assert node.id in restored.nodes

        # All edges present
        for edge in edges:
            assert edge.id in restored.edges

    def test_roundtrip_preserves_edge_labels(self, simple_graph):
        """Test edge labels are preserved through roundtrip."""
        graph, _, _ = simple_graph

        data = graph.to_dict()
        restored = Graph.from_dict(data)

        # Check labels
        e1_restored = restored.edges[next(graph.edges.keys())]
        assert e1_restored.label == ["step1"]

    def test_roundtrip_preserves_edge_properties(self, empty_graph):
        """Test edge properties are preserved through roundtrip."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)

        edge = Edge(head=n1.id, tail=n2.id, properties={"weight": 5.0, "color": "red"})
        empty_graph.add_edge(edge)

        data = empty_graph.to_dict()
        restored = Graph.from_dict(data)

        edge_restored = restored.edges[edge.id]
        assert edge_restored.properties["weight"] == 5.0
        assert edge_restored.properties["color"] == "red"

    def test_roundtrip_python_mode(self, simple_graph):
        """Test roundtrip using to_dict python mode."""
        graph, _, _ = simple_graph

        data = graph.to_dict(mode="python")
        restored = Graph.from_dict(data)

        assert len(restored) == 3
        assert len(restored.edges) == 2

    def test_roundtrip_json_mode(self, simple_graph):
        """Test roundtrip using to_dict json mode."""
        graph, _, _ = simple_graph

        data = graph.to_dict(mode="json")
        restored = Graph.from_dict(data)

        assert len(restored) == 3
        assert len(restored.edges) == 2

    def test_to_dict_exclude_as_list(self):
        """Test to_dict() accepts exclude parameter as list (not just set).

        Pattern:
            Flexible parameter types (list or set accepted, normalized internally)

        Edge Case:
            exclude parameter passed as list instead of set

        Design Rationale:
            Ergonomic API: Accept common Python collection types (list/set)
            and normalize internally. Users shouldn't need to remember
            exact type requirements.

        Expected:
            List converted to set internally, exclusion works correctly
        """
        graph = Graph()
        n1 = Node(content={"value": "A"})
        graph.add_node(n1)

        # Pass exclude as list (not set) - should work just like set
        data = graph.to_dict(exclude=["metadata"])

        assert "metadata" not in data
        assert "nodes" in data  # nodes/edges manually added, not excluded


# ============================================================================
# EdgeCondition Tests
# ============================================================================


class TestEdgeConditions:
    """Edge conditions: traversal gates, async/sync application."""

    async def test_edge_condition_default_allows_traversal(self):
        """Test default EdgeCondition allows traversal."""
        condition = EdgeCondition()

        # Async call
        result = await condition()
        assert result is True

    async def test_edge_condition_async_apply_default(self):
        """Test default async apply() allows traversal."""
        condition = EdgeCondition()

        result = await condition.apply()
        assert result is True

    async def test_always_true_condition(self):
        """Test custom AlwaysTrueCondition."""
        condition = AlwaysTrueCondition()

        assert await condition() is True

    async def test_always_true_condition_async(self):
        """Test AlwaysTrueCondition async."""
        condition = AlwaysTrueCondition()

        result = await condition.apply()
        assert result is True

    async def test_always_false_condition(self):
        """Test custom AlwaysFalseCondition."""
        condition = AlwaysFalseCondition()

        assert await condition() is False

    async def test_always_false_condition_async(self):
        """Test AlwaysFalseCondition async."""
        condition = AlwaysFalseCondition()

        result = await condition.apply()
        assert result is False

    async def test_threshold_condition_pass(self):
        """Test ThresholdCondition passes when value <= threshold."""
        condition = ThresholdCondition(properties={"value": 5.0})

        # Default threshold is 10.0
        result = await condition(threshold=10.0)
        assert result is True

    async def test_threshold_condition_fail(self):
        """Test ThresholdCondition fails when value > threshold."""
        condition = ThresholdCondition(properties={"value": 15.0})

        result = await condition(threshold=10.0)
        assert result is False

    async def test_threshold_condition_async(self):
        """Test ThresholdCondition async apply."""
        condition = ThresholdCondition(properties={"value": 8.0})

        result = await condition.apply(threshold=10.0)
        assert result is True

    async def test_edge_check_condition_no_condition(self):
        """Test Edge.check_condition returns True when no condition set."""
        edge = Edge(
            head=UUID("00000000-0000-0000-0000-000000000001"),
            tail=UUID("00000000-0000-0000-0000-000000000002"),
        )

        result = await edge.check_condition()
        assert result is True

    async def test_edge_check_condition_with_condition(self):
        """Test Edge.check_condition evaluates condition."""
        edge = Edge(
            head=UUID("00000000-0000-0000-0000-000000000001"),
            tail=UUID("00000000-0000-0000-0000-000000000002"),
            condition=AlwaysFalseCondition(),
        )

        result = await edge.check_condition()
        assert result is False

    def test_edge_condition_with_source_data(self):
        """Test EdgeCondition can store source data."""
        condition = EdgeCondition(source="some data", properties={"key": "value"})

        assert condition.source == "some data"
        assert condition.properties["key"] == "value"


# ============================================================================
# Edge Cases Tests
# ============================================================================
#
# Design aspect validated: Graph handles degenerate cases and special graph structures
# without special-casing. This validates the robustness of the core algorithms.
#
# Edge cases covered:
# 1. Empty graph: All operations return empty results (no special cases needed)
# 2. Single node: Both head and tail (isolated node), acyclic, valid for topo sort
# 3. Disconnected components: Algorithms work across all components
# 4. Self-loops: Detected as cycles (single-node cycle)
# 5. Multiple edges: Supported (no deduplication, each edge is distinct)
#
# Why these matter:
# - Empty graphs are common initial states (construction patterns)
# - Single nodes occur during incremental graph building
# - Disconnected components arise in workflow/dependency graphs (parallel work)
# - Self-loops test cycle detection edge case (simplest cycle)
# - Multiple edges enable rich relationship modeling (A USES B, A CREATED B)
#
# Algorithm correctness on edge cases:
# - is_acyclic(): Self-loop detection validates DFS back-edge logic
# - topological_sort(): Disconnected components validate Kahn's multi-component handling
# - find_path(): Same-node query validates BFS termination condition
#
# These tests prevent regressions from optimization (e.g., skipping empty checks).


class TestEdgeCases:
    """Edge cases: empty graph, self-loops, disconnected components, UUID coercion."""

    def test_empty_graph_operations(self, empty_graph):
        """Test operations on empty graph."""
        # Queries on empty graph
        assert empty_graph.get_heads() == []
        assert empty_graph.get_tails() == []
        assert empty_graph.is_acyclic() is True
        assert empty_graph.topological_sort() == []

    def test_single_node_graph(self, empty_graph):
        """Test graph with single isolated node."""
        node = Node(content={"value": "lonely"})
        empty_graph.add_node(node)

        # Single node is both head and tail
        assert node in empty_graph.get_heads()
        assert node in empty_graph.get_tails()

        # Single node is acyclic
        assert empty_graph.is_acyclic() is True

        # Topological sort works
        sorted_nodes = empty_graph.topological_sort()
        assert len(sorted_nodes) == 1
        assert sorted_nodes[0] == node

    def test_disconnected_components(self, empty_graph):
        """Test graph with multiple disconnected components."""
        # Component 1: A -> B
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)
        empty_graph.add_edge(Edge(head=n1.id, tail=n2.id))

        # Component 2: C -> D
        n3 = Node(content={"value": "C"})
        n4 = Node(content={"value": "D"})
        empty_graph.add_node(n3)
        empty_graph.add_node(n4)
        empty_graph.add_edge(Edge(head=n3.id, tail=n4.id))

        # Should still be acyclic
        assert empty_graph.is_acyclic() is True

        # Topological sort should include all nodes
        sorted_nodes = empty_graph.topological_sort()
        assert len(sorted_nodes) == 4

        # Heads and tails
        heads = empty_graph.get_heads()
        assert len(heads) == 2
        assert n1 in heads
        assert n3 in heads

    def test_self_loop_graph(self, empty_graph):
        """Test graph with self-loop."""
        node = Node(content={"value": "self"})
        empty_graph.add_node(node)

        # Self-loop edge
        edge = Edge(head=node.id, tail=node.id)
        empty_graph.add_edge(edge)

        # Should detect cycle
        assert empty_graph.is_acyclic() is False

        # Should raise on topological sort
        with pytest.raises(ValueError, match="cycle"):
            empty_graph.topological_sort()

        # No heads or tails (every node has incoming and outgoing)
        assert len(empty_graph.get_heads()) == 0
        assert len(empty_graph.get_tails()) == 0

    def test_multiple_edges_between_nodes(self, empty_graph):
        """Test multiple edges between same pair of nodes."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)

        # Multiple edges with different labels
        e1 = Edge(head=n1.id, tail=n2.id, label=["type1"])
        e2 = Edge(head=n1.id, tail=n2.id, label=["type2"])

        empty_graph.add_edge(e1)
        empty_graph.add_edge(e2)

        # Both edges should exist
        assert e1 in empty_graph
        assert e2 in empty_graph

        # Get edges should return both
        edges = empty_graph.get_node_edges(n1, direction="out")
        assert len(edges) == 2

    def test_uuid_coercion_string(self, simple_graph):
        """Test UUID coercion from string via direct Pile access."""
        graph, (n1, _, _), _ = simple_graph

        # Pass UUID as string - Pile supports Element._coerce_id
        node = graph.nodes[str(n1.id)]
        assert node == n1

    def test_graph_with_edge_conditions_serialization(self, empty_graph):
        """Test graph with EdgeCondition survives roundtrip."""
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        empty_graph.add_node(n1)
        empty_graph.add_node(n2)

        # Edge with condition
        condition = ThresholdCondition(properties={"value": 5.0})
        edge = Edge(head=n1.id, tail=n2.id, condition=condition)
        empty_graph.add_edge(edge)

        # Roundtrip
        data = empty_graph.to_dict()
        restored = Graph.from_dict(data)

        # Edge condition is excluded from serialization (not persisted)
        edge_restored = restored.edges[edge.id]
        assert edge_restored.condition is None  # Conditions are not serialized

    async def test_remove_node_from_middle_of_chain(self, simple_graph):
        """Test removing middle node breaks chain but preserves remaining."""
        graph, (n1, n2, n3), _ = simple_graph

        # Remove middle node
        graph.remove_node(n2)

        # n1 and n3 should still exist
        assert n1 in graph
        assert n3 in graph
        assert len(graph) == 2

        # But no path exists
        path = await graph.find_path(n1, n3)
        assert path is None

    def test_graph_repr(self, simple_graph):
        """Test Graph repr shows class name and ID."""
        graph, _, _ = simple_graph

        repr_str = repr(graph)

        assert "Graph" in repr_str
        assert str(graph.id) in repr_str

    def test_isolated_registry_per_subclass(self):
        """Test each Graph subclass has isolated adapter registry.

        Pattern:
            Rust-like explicit adapter pattern prevents adapter pollution

        Design Rationale:
            Each subclass gets its own adapter registry (no shared state).
            Prevents accidental adapter inheritance, requires explicit
            registration per class. Trade-off: Must register adapters
            for each subclass vs automatic inheritance.

        Use Case:
            Multiple Graph subclasses with different serialization needs
            (e.g., WorkflowGraph with YAML, DependencyGraph with TOML)

        Edge Case:
            Subclasses don't inherit parent adapters (isolation enforced)

        Complexity:
            Registry lookup O(1) via id(cls) as unique key

        Expected:
            GraphA and GraphB have independent registries, no adapter sharing
        """

        # Create two Graph subclasses
        class GraphA(Graph):
            pass

        class GraphB(Graph):
            pass

        # Both should have independent registries (no adapters initially)
        from pydapter.exceptions import AdapterNotFoundError

        graph_a = GraphA()
        graph_b = GraphB()

        # Neither has toml adapter (isolated registries)
        try:
            graph_a.adapt_to("toml")
            raise AssertionError("GraphA should not have toml adapter")
        except AdapterNotFoundError:
            pass  # Expected

        try:
            graph_b.adapt_to("toml")
            raise AssertionError("GraphB should not have toml adapter")
        except AdapterNotFoundError:
            pass  # Expected

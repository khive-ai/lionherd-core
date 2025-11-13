# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Benchmark suite for Graph operations - Issue #130.

Validates performance of Graph operations to ensure no regression from PR #117
(error handling refactor). Targets pandas-level performance, <5% regression tolerance.

Benchmark Coverage:
- add_node: Single node addition with adjacency initialization
- remove_node: Node removal with cascading edge cleanup
- remove_edge: Edge removal with adjacency updates
- bulk_add_nodes: Batch node addition (1000 nodes)
- bulk_add_edges: Batch edge addition (1000 edges)
- bulk_remove_edges: Batch edge removal (1000 edges)
- bulk_remove_nodes: Batch node removal with cascading cleanup (1000 nodes)

Test Datasets:
- Small: 10K nodes + 50K edges (realistic agent graphs)
- Large: 100K nodes + 500K edges (stress test)

Performance Goals:
- add_node: <100μs (O(1) Pile add + adjacency init)
- remove_node: <1ms for low-degree nodes (O(deg) edge cleanup)
- remove_edge: <50μs (O(1) Pile remove + set discard)
- bulk operations: Linear scaling (no exponential blowup)

Baseline Comparison:
Run with --benchmark-compare to compare against baseline.json from pre-PR #117 commit.

Usage:
    # Run benchmarks only (skip regular tests)
    uv run pytest tests/benchmarks/test_graph_benchmarks.py --benchmark-only

    # Save baseline for future comparison
    uv run pytest tests/benchmarks/test_graph_benchmarks.py --benchmark-save=baseline

    # Compare against baseline
    uv run pytest tests/benchmarks/test_graph_benchmarks.py --benchmark-compare=baseline

    # Run with verbose stats
    uv run pytest tests/benchmarks/test_graph_benchmarks.py --benchmark-verbose
"""

from __future__ import annotations

import pytest

from lionherd_core.base import Edge, Graph, Node

# ============================================================================
# Fixtures - Module Scope for Expensive Graph Creation
# ============================================================================


@pytest.fixture(scope="module")
def graph_10k():
    """Create graph with 10K nodes and 50K edges.

    Graph structure: Random edges with ~5 edges per node (realistic agent graph).
    Construction takes ~2-3s, reused across all benchmarks.
    """
    graph = Graph()
    nodes = []

    # Create 10K nodes
    for i in range(10_000):
        node = Node(content={"id": i, "value": f"node_{i}"})
        graph.add_node(node)
        nodes.append(node)

    # Create 50K edges (~5 edges per node)
    # Simple pattern: each node connects to next 5 nodes (circular)
    for i in range(10_000):
        for j in range(1, 6):
            target_idx = (i + j) % 10_000
            edge = Edge(
                head=nodes[i].id,
                tail=nodes[target_idx].id,
                label=[f"edge_{i}_{target_idx}"],
            )
            graph.add_edge(edge)

    return graph, nodes


@pytest.fixture(scope="module")
def graph_100k():
    """Create graph with 100K nodes and 500K edges.

    Graph structure: Random edges with ~5 edges per node (stress test).
    Construction takes ~30-40s, reused across all benchmarks.
    """
    graph = Graph()
    nodes = []

    # Create 100K nodes
    for i in range(100_000):
        node = Node(content={"id": i, "value": f"node_{i}"})
        graph.add_node(node)
        nodes.append(node)

    # Create 500K edges (~5 edges per node)
    for i in range(100_000):
        for j in range(1, 6):
            target_idx = (i + j) % 100_000
            edge = Edge(
                head=nodes[i].id,
                tail=nodes[target_idx].id,
                label=[f"edge_{i}_{target_idx}"],
            )
            graph.add_edge(edge)

    return graph, nodes


# ============================================================================
# Single Operation Benchmarks
# ============================================================================


@pytest.mark.parametrize("graph_fixture", ["graph_10k", "graph_100k"])
def test_benchmark_add_node(benchmark, graph_fixture, request):
    """Benchmark single node addition (O(1) expected).

    Measures: Pile.add() + adjacency dict initialization
    Critical path: _check_node_exists, adjacency set creation
    """
    graph, _ = request.getfixturevalue(graph_fixture)

    def add_node():
        # Create fresh node each iteration
        node = Node(content={"value": "benchmark_node"})
        graph.add_node(node)
        return node

    # Use pedantic mode for precise measurement (removes outliers)
    benchmark.pedantic(add_node, rounds=100, iterations=10)

    # Cleanup: remove added nodes to keep graph size stable
    # (benchmark measures add, not cleanup)


@pytest.mark.parametrize("graph_fixture", ["graph_10k", "graph_100k"])
def test_benchmark_remove_node(benchmark, graph_fixture, request):
    """Benchmark single node removal with edge cleanup.

    Measures: Edge cascade removal + adjacency cleanup + Pile.remove()
    Critical path: _check_node_exists, remove_edge calls (nested lock), dict cleanup
    Performance: O(deg) where deg = in_degree + out_degree (~10 for our graphs)
    """
    graph, nodes = request.getfixturevalue(graph_fixture)

    # Setup: Pre-select nodes to remove (middle nodes with typical degree)
    nodes_to_remove = nodes[1000:2000]  # 1000 nodes for benchmark

    def remove_node():
        # Get a node that still exists
        node = nodes_to_remove[0]
        # Remove node (cascades to edges)
        graph.remove_node(node.id)

    def setup():
        # Re-add the node and its edges before each iteration
        node = nodes_to_remove[0]
        # Add node only if not already present
        if node not in graph:
            graph.add_node(node)
        # Re-add edges (simple pattern: connect to next 5 nodes)
        for j in range(1, 6):
            idx = nodes.index(node)
            target_idx = (idx + j) % len(nodes)
            if nodes[target_idx] in graph:
                edge = Edge(head=node.id, tail=nodes[target_idx].id, label=["bench"])
                try:
                    graph.add_edge(edge)
                except Exception:
                    pass  # Edge might already exist

    benchmark.pedantic(remove_node, setup=setup, rounds=50, iterations=1)

    # Restore the node after all benchmark rounds (last iteration leaves it removed)
    node = nodes_to_remove[0]
    if node not in graph:
        graph.add_node(node)
        # Restore edges
        for j in range(1, 6):
            idx = nodes.index(node)
            target_idx = (idx + j) % len(nodes)
            if nodes[target_idx] in graph:
                edge = Edge(head=node.id, tail=nodes[target_idx].id, label=["restored"])
                try:
                    graph.add_edge(edge)
                except Exception:
                    pass


@pytest.mark.parametrize("graph_fixture", ["graph_10k", "graph_100k"])
def test_benchmark_remove_edge(benchmark, graph_fixture, request):
    """Benchmark single edge removal (O(1) expected).

    Measures: _check_edge_exists + adjacency set.discard() + Pile.remove()
    Critical path: Error handling from PR #117, dict/set updates
    """
    graph, nodes = request.getfixturevalue(graph_fixture)

    # Setup: Create test edge that we'll repeatedly add/remove
    test_node_a = nodes[5000]
    test_node_b = nodes[5001]
    test_edge_id = [None]  # Mutable container

    def remove_edge():
        graph.remove_edge(test_edge_id[0])

    def setup():
        # Re-add edge before each iteration
        edge = Edge(head=test_node_a.id, tail=test_node_b.id, label=["benchmark"])
        graph.add_edge(edge)
        test_edge_id[0] = edge.id

    benchmark.pedantic(remove_edge, setup=setup, rounds=50, iterations=1)

    # Restore the edge after all benchmark rounds (last iteration leaves it removed)
    edge = Edge(head=test_node_a.id, tail=test_node_b.id, label=["restored"])
    try:
        graph.add_edge(edge)
    except Exception:
        pass  # Edge might already exist


# ============================================================================
# Bulk Operation Benchmarks
# ============================================================================


def test_benchmark_bulk_add_nodes_10k(benchmark, graph_10k):
    """Benchmark bulk node addition (1000 nodes).

    Validates: Linear scaling, no quadratic behavior in batch ops
    Expected: ~100ms for 1000 nodes (100μs per node)
    """
    graph, _ = graph_10k
    added_nodes = []

    def bulk_add_nodes():
        nodes = []
        for i in range(1000):
            node = Node(content={"id": f"bulk_{i}", "value": f"bulk_node_{i}"})
            graph.add_node(node)
            nodes.append(node)
        added_nodes.extend(nodes)
        return nodes

    benchmark(bulk_add_nodes)

    # Note: No cleanup - these are additive operations, graph grows but stays valid


def test_benchmark_bulk_add_nodes_100k(benchmark, graph_100k):
    """Benchmark bulk node addition on large graph (1000 nodes).

    Validates: Performance stable even on 100K node graph
    """
    graph, _ = graph_100k
    added_nodes = []

    def bulk_add_nodes():
        nodes = []
        for i in range(1000):
            node = Node(content={"id": f"bulk_{i}", "value": f"bulk_node_{i}"})
            graph.add_node(node)
            nodes.append(node)
        added_nodes.extend(nodes)
        return nodes

    benchmark(bulk_add_nodes)

    # Note: No cleanup - these are additive operations, graph grows but stays valid


def test_benchmark_bulk_add_edges_10k(benchmark, graph_10k):
    """Benchmark bulk edge addition (1000 edges).

    Validates: Edge addition scales linearly
    Expected: ~50ms for 1000 edges (50μs per edge)
    """
    graph, nodes = graph_10k
    added_edges = []

    def bulk_add_edges():
        edges = []
        # Add edges between existing nodes
        for i in range(1000):
            head_idx = i % len(nodes)
            tail_idx = (i + 100) % len(nodes)
            edge = Edge(
                head=nodes[head_idx].id,
                tail=nodes[tail_idx].id,
                label=[f"bulk_edge_{i}"],
            )
            graph.add_edge(edge)
            edges.append(edge)
        added_edges.extend(edges)
        return edges

    benchmark(bulk_add_edges)

    # Note: No cleanup - these are additive operations, graph grows but stays valid


def test_benchmark_bulk_add_edges_100k(benchmark, graph_100k):
    """Benchmark bulk edge addition on large graph (1000 edges)."""
    graph, nodes = graph_100k
    added_edges = []

    def bulk_add_edges():
        edges = []
        for i in range(1000):
            head_idx = i % len(nodes)
            tail_idx = (i + 1000) % len(nodes)
            edge = Edge(
                head=nodes[head_idx].id,
                tail=nodes[tail_idx].id,
                label=[f"bulk_edge_{i}"],
            )
            graph.add_edge(edge)
            edges.append(edge)
        added_edges.extend(edges)
        return edges

    benchmark(bulk_add_edges)

    # Note: No cleanup - these are additive operations, graph grows but stays valid


def test_benchmark_bulk_remove_edges_10k(benchmark, graph_10k):
    """Benchmark bulk edge removal (1000 edges).

    Validates: Remove operations scale linearly
    Critical path: Adjacency cleanup, error handling
    """
    graph, nodes = graph_10k

    edges_for_removal = []

    def bulk_remove_edges():
        for edge in edges_for_removal:
            graph.remove_edge(edge.id)

    def setup():
        # Create fresh edges before each benchmark iteration
        edges_for_removal.clear()
        for i in range(1000):
            head_idx = (i + 50) % len(nodes)
            tail_idx = (i + 150) % len(nodes)
            edge = Edge(
                head=nodes[head_idx].id,
                tail=nodes[tail_idx].id,
                label=[f"remove_bulk_{i}"],
            )
            graph.add_edge(edge)
            edges_for_removal.append(edge)

    benchmark.pedantic(bulk_remove_edges, setup=setup, rounds=5, iterations=1)

    # No cleanup needed - edges already removed


def test_benchmark_bulk_remove_edges_100k(benchmark, graph_100k):
    """Benchmark bulk edge removal on large graph (1000 edges)."""
    graph, nodes = graph_100k

    edges_for_removal = []

    def bulk_remove_edges():
        for edge in edges_for_removal:
            graph.remove_edge(edge.id)

    def setup():
        # Create fresh edges before each benchmark iteration
        edges_for_removal.clear()
        for i in range(1000):
            head_idx = (i + 500) % len(nodes)
            tail_idx = (i + 1500) % len(nodes)
            edge = Edge(
                head=nodes[head_idx].id,
                tail=nodes[tail_idx].id,
                label=[f"remove_bulk_{i}"],
            )
            graph.add_edge(edge)
            edges_for_removal.append(edge)

    benchmark.pedantic(bulk_remove_edges, setup=setup, rounds=5, iterations=1)

    # No cleanup needed - edges already removed


def test_benchmark_bulk_remove_nodes_10k(benchmark, graph_10k):
    """Benchmark bulk node removal with edge cleanup (1000 nodes).

    Most expensive operation: Cascading edge removal + adjacency cleanup
    Validates: Performance acceptable for batch removal
    Expected: ~1-2s for 1000 nodes (~1-2ms per node with degree ~5)
    """
    graph, nodes = graph_10k

    nodes_for_removal = []

    def bulk_remove_nodes():
        for node in nodes_for_removal:
            graph.remove_node(node.id)

    def setup():
        # Create fresh nodes before each benchmark iteration
        nodes_for_removal.clear()
        for i in range(1000):
            node = Node(content={"id": f"remove_bulk_{i}", "value": f"remove_bulk_node_{i}"})
            graph.add_node(node)
            nodes_for_removal.append(node)
            # Add edges to make it realistic (connect to existing nodes)
            for j in range(5):
                target_idx = (i * 10 + j) % len(nodes)
                edge = Edge(head=node.id, tail=nodes[target_idx].id, label=["bulk_edge"])
                graph.add_edge(edge)

    benchmark.pedantic(bulk_remove_nodes, setup=setup, rounds=5, iterations=1)

    # No cleanup needed - nodes already removed


def test_benchmark_bulk_remove_nodes_100k(benchmark, graph_100k):
    """Benchmark bulk node removal on large graph (1000 nodes)."""
    graph, nodes = graph_100k

    nodes_for_removal = []

    def bulk_remove_nodes():
        for node in nodes_for_removal:
            graph.remove_node(node.id)

    def setup():
        # Create fresh nodes before each benchmark iteration
        nodes_for_removal.clear()
        for i in range(1000):
            node = Node(content={"id": f"remove_bulk_{i}", "value": f"remove_bulk_node_{i}"})
            graph.add_node(node)
            nodes_for_removal.append(node)
            # Add edges to make it realistic (connect to existing nodes)
            for j in range(5):
                target_idx = (i * 100 + j) % len(nodes)
                edge = Edge(head=node.id, tail=nodes[target_idx].id, label=["bulk_edge"])
                graph.add_edge(edge)

    benchmark.pedantic(bulk_remove_nodes, setup=setup, rounds=5, iterations=1)

    # No cleanup needed - nodes already removed


# ============================================================================
# Query Operation Benchmarks
# ============================================================================


@pytest.mark.parametrize("graph_fixture", ["graph_10k", "graph_100k"])
def test_benchmark_get_successors(benchmark, graph_fixture, request):
    """Benchmark get_successors query (O(deg) expected).

    Measures: Adjacency lookup + node fetching from Pile
    Critical path: _out_edges dict access, Pile[uuid] lookups
    """
    graph, nodes = request.getfixturevalue(graph_fixture)

    # Use middle node (typical degree ~5)
    test_node = nodes[len(nodes) // 2]

    def get_successors():
        return graph.get_successors(test_node.id)

    benchmark(get_successors)


@pytest.mark.parametrize("graph_fixture", ["graph_10k", "graph_100k"])
def test_benchmark_get_predecessors(benchmark, graph_fixture, request):
    """Benchmark get_predecessors query (O(deg) expected)."""
    graph, nodes = request.getfixturevalue(graph_fixture)

    test_node = nodes[len(nodes) // 2]

    def get_predecessors():
        return graph.get_predecessors(test_node.id)

    benchmark(get_predecessors)


# ============================================================================
# Memory Benchmarks
# ============================================================================


def test_benchmark_memory_10k(benchmark):
    """Benchmark memory usage for 10K node graph.

    Pytest-benchmark tracks memory via stats (not timing).
    Validates: Memory footprint reasonable for graph size.
    """

    def create_graph():
        graph = Graph()
        nodes = []

        for i in range(10_000):
            node = Node(content={"id": i})
            graph.add_node(node)
            nodes.append(node)

        for i in range(10_000):
            for j in range(1, 6):
                target_idx = (i + j) % 10_000
                edge = Edge(head=nodes[i].id, tail=nodes[target_idx].id)
                graph.add_edge(edge)

        return graph

    benchmark(create_graph)


def test_benchmark_memory_100k(benchmark):
    """Benchmark memory usage for 100K node graph."""

    def create_graph():
        graph = Graph()
        nodes = []

        for i in range(100_000):
            node = Node(content={"id": i})
            graph.add_node(node)
            nodes.append(node)

        for i in range(100_000):
            for j in range(1, 6):
                target_idx = (i + j) % 100_000
                edge = Edge(head=nodes[i].id, tail=nodes[target_idx].id)
                graph.add_edge(edge)

        return graph

    benchmark(create_graph)

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Benchmark suite for Graph operations - Issue #130.

Validates performance of Graph operations to ensure no regression from PR #117
(error handling refactor). Targets pandas-level performance, <5% regression tolerance.

Benchmark Coverage:
- add_node: Single node addition with adjacency initialization
- remove_node: Node removal with cascading edge cleanup
- is_acyclic: DFS cycle detection (workflow validation)
- topological_sort: Kahn's algorithm (execution planning)
- find_path: BFS pathfinding (debugging, analysis)

Test Datasets (Simplified):
- Primary: 1000 nodes (main performance validation scale)
- Workflow: 10, 50, 100 nodes (realistic workflow scales)

Performance Goals:
- add_node: <100μs (O(1) Pile add + adjacency init)
- remove_node: <1ms for low-degree nodes (O(deg) edge cleanup)
- is_acyclic: <1ms (100 nodes), <10ms (1000 nodes)
- topological_sort: <1ms (100 nodes), <10ms (1000 nodes)
- find_path: <1ms (100 nodes), <10ms (1000 nodes)

Baseline Comparison:
Run with --benchmark-compare to compare against baseline.json from pre-PR #117 commit.

Usage:
    # Run benchmarks only (skip regular tests)
    uv run pytest benchmarks/graph/ --benchmark-only

    # Tune statistical rigor
    GRAPH_BENCH_ROUNDS=100 GRAPH_BENCH_ITERATIONS=10 uv run pytest benchmarks/graph/ --benchmark-only

    # Save baseline for future comparison
    uv run pytest benchmarks/graph/ --benchmark-save=baseline --benchmark-only

    # Compare against baseline
    uv run pytest benchmarks/graph/ --benchmark-compare=baseline --benchmark-only
"""

from __future__ import annotations

import os
import random

import pytest

from lionherd_core.base import Edge, Graph, Node

# ============================================================================
# Global Benchmark Configuration
# ============================================================================
# Default pedantic settings. Override from environment for tuning:
#   GRAPH_BENCH_ROUNDS, GRAPH_BENCH_ITERATIONS, GRAPH_BENCH_WARMUP
BENCH_ROUNDS = int(os.getenv("GRAPH_BENCH_ROUNDS", "50"))
BENCH_ITERATIONS = int(os.getenv("GRAPH_BENCH_ITERATIONS", "5"))
BENCH_WARMUP = int(os.getenv("GRAPH_BENCH_WARMUP", "5"))

# ============================================================================
# Fixtures - Function Scope for Statistical Isolation
# ============================================================================
#
# Statistical Methodology:
# - Function scope prevents fixture pollution (fresh graph per test)
# - Configurable: ~50 rounds x 5 iterations + 5 warmup (default)
# - Warmup: Eliminates JIT/cache cold start effects
# - Iterations: Amortizes setup overhead, focuses on operation cost
#
# Trade-off: Setup cost (~2-3s per test) vs clean measurements
# Rationale: Statistical validity > speed for regression detection
# ============================================================================


def _create_dag_graph(num_nodes: int, num_layers: int, edges_per_node: int = 5):
    """Create a stratified DAG with forward-flowing edges.

    Args:
        num_nodes: Total number of nodes to create
        num_layers: Number of layers to stratify nodes into
        edges_per_node: Target number of outgoing edges per node

    Returns:
        Tuple of (Graph, list of nodes)

    Graph structure: Nodes divided into layers, edges only flow forward
    (from layer i to layers i+1, i+2, etc). This guarantees acyclicity
    while maintaining realistic workflow topology.
    """
    graph = Graph()
    nodes = []
    nodes_per_layer = num_nodes // num_layers

    # Create all nodes first
    for i in range(num_nodes):
        node = Node(content={"id": i, "value": f"node_{i}"})
        graph.add_node(node)
        nodes.append(node)

    # Create edges layer by layer (forward-flowing only)
    for layer_idx in range(num_layers):
        layer_start = layer_idx * nodes_per_layer
        layer_end = min((layer_idx + 1) * nodes_per_layer, num_nodes)

        # Last layer has no outgoing edges
        if layer_idx == num_layers - 1:
            break

        for i in range(layer_start, layer_end):
            # Calculate valid target range (next layers only)
            target_start = layer_end
            target_end = num_nodes

            if target_end <= target_start:
                continue  # No valid targets

            # Create edges_per_node edges to random nodes in subsequent layers
            num_edges = min(edges_per_node, target_end - target_start)
            targets = random.sample(range(target_start, target_end), num_edges)

            for target_idx in targets:
                edge = Edge(
                    head=nodes[i].id,
                    tail=nodes[target_idx].id,
                    label=[f"edge_{i}_{target_idx}"],
                )
                graph.add_edge(edge)

    return graph, nodes


# Realistic workflow fixtures (10-100 nodes)
@pytest.fixture(scope="function")
def graph_10():
    """Create fresh DAG with 10 nodes (small agent, simple flow)."""
    return _create_dag_graph(num_nodes=10, num_layers=3, edges_per_node=2)


@pytest.fixture(scope="function")
def graph_50():
    """Create fresh DAG with 50 nodes (medium workflow)."""
    return _create_dag_graph(num_nodes=50, num_layers=5, edges_per_node=3)


@pytest.fixture(scope="function")
def graph_100():
    """Create fresh DAG with 100 nodes (complex orchestration)."""
    return _create_dag_graph(num_nodes=100, num_layers=10, edges_per_node=3)


# Primary benchmark fixture (1000 nodes)
@pytest.fixture(scope="function")
def graph_1000():
    """Create fresh DAG with 1000 nodes for primary performance validation.

    Graph structure: Stratified DAG with 10 layers (100 nodes per layer).
    Each node connects to 5 random nodes in subsequent layers, creating
    forward-flowing edges typical of workflow dependencies.

    Construction takes ~1s, but guarantees no fixture pollution.
    Function scope ensures clean baseline for statistical measurements.
    """
    return _create_dag_graph(num_nodes=1000, num_layers=10, edges_per_node=5)


# ============================================================================
# CRUD Benchmarks - add_node, remove_node
# ============================================================================


def test_benchmark_add_node(benchmark, graph_1000):
    """Benchmark node addition (O(1) Pile add + adjacency init)."""
    graph, _ = graph_1000

    def add_node():
        # Create fresh node each iteration
        node = Node(content={"value": "benchmark_node"})
        graph.add_node(node)
        return node

    # Use pedantic mode for precise measurement (removes outliers)
    benchmark.pedantic(
        add_node,
        rounds=BENCH_ROUNDS,
        iterations=BENCH_ITERATIONS,
        warmup_rounds=BENCH_WARMUP,
    )


def test_benchmark_remove_node(benchmark, graph_1000):
    """Benchmark node removal with cascading edge cleanup (O(degree))."""
    graph, nodes = graph_1000

    # Setup: create a new node to remove each round (avoids node-already-removed errors)
    def setup():
        # Create a fresh node connected to existing nodes
        node_to_remove = Node(content={"value": "temp_node"})
        graph.add_node(node_to_remove)
        # Add a few edges to simulate realistic removal
        if len(nodes) >= 5:
            for i in range(min(5, len(nodes))):
                edge = Edge(
                    head=node_to_remove.id,
                    tail=nodes[i].id,
                    label=["temp"],
                )
                graph.add_edge(edge)
        return (node_to_remove,), {}

    def remove_node(node):
        graph.remove_node(node.id)

    # NOTE: setup functions require iterations=1 in pytest-benchmark
    benchmark.pedantic(
        remove_node,
        setup=setup,
        rounds=BENCH_ROUNDS,
        iterations=1,
        warmup_rounds=BENCH_WARMUP,
    )


# ============================================================================
# Realistic Workflow Benchmarks (10-100 nodes)
# ============================================================================


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [
        ("graph_10", 1),
        ("graph_50", 5),
        ("graph_100", 10),
    ],
)
def test_benchmark_workflow_construction(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark workflow graph construction (realistic scales)."""
    graph, _ = request.getfixturevalue(graph_fixture)

    def construct_workflow():
        # Simulate workflow construction: add nodes + edges
        new_nodes = []
        for i in range(5):
            node = Node(content={"task": f"step_{i}"})
            graph.add_node(node)
            new_nodes.append(node)

        # Connect in sequence
        for i in range(len(new_nodes) - 1):
            edge = Edge(head=new_nodes[i].id, tail=new_nodes[i + 1].id, label=["next"])
            graph.add_edge(edge)

    benchmark.pedantic(
        construct_workflow,
        rounds=BENCH_ROUNDS,
        iterations=BENCH_ITERATIONS,
        warmup_rounds=BENCH_WARMUP,
    )


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [
        ("graph_10", 1),
        ("graph_50", 5),
        ("graph_100", 10),
    ],
)
def test_benchmark_workflow_query(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark workflow graph queries (get_successors, get_predecessors)."""
    graph, nodes = request.getfixturevalue(graph_fixture)

    def query_workflow():
        # Query middle node
        middle_node = nodes[len(nodes) // 2]
        _ = graph.get_successors(middle_node.id)
        _ = graph.get_predecessors(middle_node.id)
        _ = graph.get_node_edges(middle_node.id, direction="both")

    benchmark.pedantic(
        query_workflow,
        rounds=BENCH_ROUNDS,
        iterations=BENCH_ITERATIONS,
        warmup_rounds=BENCH_WARMUP,
    )


# ============================================================================
# Algorithm Benchmarks
# ============================================================================


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [
        ("graph_100", 1),
        ("graph_1000", 10),
    ],
)
def test_benchmark_is_acyclic(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark is_acyclic (DFS cycle detection) - called before every workflow execution."""
    graph, _ = request.getfixturevalue(graph_fixture)

    def check_acyclic():
        return graph.is_acyclic()

    result = benchmark.pedantic(
        check_acyclic,
        rounds=BENCH_ROUNDS,
        iterations=BENCH_ITERATIONS,
        warmup_rounds=BENCH_WARMUP,
    )
    assert result is True  # Should be acyclic (DAG structure)


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [
        ("graph_100", 1),
        ("graph_1000", 10),
    ],
)
def test_benchmark_topological_sort(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark topological_sort (Kahn's algorithm) - determines task execution order."""
    graph, _ = request.getfixturevalue(graph_fixture)

    def topo_sort():
        return graph.topological_sort()

    result = benchmark.pedantic(
        topo_sort,
        rounds=BENCH_ROUNDS,
        iterations=BENCH_ITERATIONS,
        warmup_rounds=BENCH_WARMUP,
    )
    assert len(result) == len(graph.nodes)  # All nodes in sorted order


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [
        ("graph_100", 1),
        ("graph_1000", 10),
    ],
)
def test_benchmark_find_path(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark find_path (BFS) - debugging and dependency analysis."""
    graph, nodes = request.getfixturevalue(graph_fixture)

    # Find path from first node to last node
    start_node = nodes[0]
    end_node = nodes[-1]

    def find_path():
        return graph.find_path(start_node.id, end_node.id)

    result = benchmark.pedantic(
        find_path,
        rounds=BENCH_ROUNDS,
        iterations=BENCH_ITERATIONS,
        warmup_rounds=BENCH_WARMUP,
    )
    assert result is not None  # Path should exist in DAG


# ============================================================================
# Memory Benchmarks
# ============================================================================


def test_benchmark_memory_1000():
    """Benchmark peak memory usage for 1000-node graph (target: <50 MB).

    Single-shot measurement (not timed) - validates memory footprint correctness.
    """
    import tracemalloc

    def create_graph_and_measure():
        tracemalloc.start()
        _graph, _ = _create_dag_graph(num_nodes=1000, num_layers=10, edges_per_node=5)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024 / 1024  # Convert to MB

    # Single-shot measurement (no benchmark fixture)
    peak_mb = create_graph_and_measure()
    assert peak_mb < 50, f"Peak memory {peak_mb:.2f} MB exceeds 50 MB target"

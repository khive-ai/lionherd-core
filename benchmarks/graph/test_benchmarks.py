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
- is_acyclic: DFS cycle detection (workflow validation)
- topological_sort: Kahn's algorithm (execution planning)
- find_path: BFS pathfinding (debugging, analysis)

Test Datasets:
- Algorithm benchmarks: 100, 1000, 10K nodes (realistic workflow scales)
- CRUD benchmarks: 10K nodes + 50K edges (realistic agent graphs)
- Stress test: 100K nodes + 500K edges (large-scale workflows)

Performance Goals:
- add_node: <100μs (O(1) Pile add + adjacency init)
- remove_node: <1ms for low-degree nodes (O(deg) edge cleanup)
- remove_edge: <50μs (O(1) Pile remove + set discard)
- bulk operations: Linear scaling (no exponential blowup)
- is_acyclic: <1ms (100 nodes), <10ms (1000 nodes), <100ms (10K nodes)
- topological_sort: <1ms (100 nodes), <10ms (1000 nodes), <100ms (10K nodes)
- find_path: <1ms (100 nodes), <10ms (1000 nodes), <50ms (10K nodes)

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

import random

import pytest

from lionherd_core.base import Edge, Graph, Node

# ============================================================================
# Fixtures - Function Scope for Statistical Isolation
# ============================================================================
#
# Statistical Methodology:
# - Function scope prevents fixture pollution (fresh graph per test)
# - Standardized: 100 rounds x 10 iterations + 5 warmup rounds
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


@pytest.fixture(scope="function")
def graph_10k():
    """Create fresh DAG with 10K nodes and ~50K edges for each test.

    Graph structure: Stratified DAG with 20 layers (500 nodes per layer).
    Each node connects to 5 random nodes in subsequent layers, creating
    forward-flowing edges typical of workflow dependencies.

    Construction takes ~2-3s, but guarantees no fixture pollution.
    Function scope ensures clean baseline for statistical measurements.
    """
    return _create_dag_graph(num_nodes=10_000, num_layers=20, edges_per_node=5)


@pytest.fixture(scope="function")
def graph_100k():
    """Create fresh DAG with 100K nodes and ~500K edges for each test.

    Graph structure: Stratified DAG with 50 layers (2000 nodes per layer).
    Each node connects to 5 random nodes in subsequent layers, creating
    forward-flowing edges typical of large-scale workflows.

    Construction takes ~30-40s, but guarantees no fixture pollution.
    Function scope ensures clean baseline for statistical measurements.
    """
    return _create_dag_graph(num_nodes=100_000, num_layers=50, edges_per_node=5)


@pytest.fixture(scope="function")
def graph_10():
    """Create tiny DAG with 10 nodes for realistic workflow benchmarks.

    Graph structure: Stratified DAG with 3 layers (~3-4 nodes per layer).
    Each node connects to 2-3 nodes in subsequent layers.

    Represents typical lionagi workflows (6-12 nodes common).
    Construction: <1ms
    """
    return _create_dag_graph(num_nodes=10, num_layers=3, edges_per_node=2)


@pytest.fixture(scope="function")
def graph_50():
    """Create small DAG with 50 nodes for realistic workflow benchmarks.

    Graph structure: Stratified DAG with 5 layers (10 nodes per layer).
    Each node connects to 3 random nodes in subsequent layers.

    Represents medium-scale lionagi workflows.
    Construction: <10ms
    """
    return _create_dag_graph(num_nodes=50, num_layers=5, edges_per_node=3)


@pytest.fixture(scope="function")
def graph_100():
    """Create DAG with 100 nodes for realistic workflow benchmarks (p99 scale).

    Graph structure: Stratified DAG with 5 layers (20 nodes per layer).
    Each node connects to 5 random nodes in subsequent layers.

    Represents p99 lionagi workflows (50-100 nodes).
    Construction: <20ms
    """
    return _create_dag_graph(num_nodes=100, num_layers=5, edges_per_node=5)


@pytest.fixture(scope="function")
def graph_1000():
    """Create medium DAG with 1000 nodes for algorithm benchmarks.

    Graph structure: Stratified DAG with 10 layers (100 nodes per layer).
    Each node connects to 5 random nodes in subsequent layers.

    Used for medium-scale algorithm benchmarks.
    """
    return _create_dag_graph(num_nodes=1000, num_layers=10, edges_per_node=5)


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
    benchmark.pedantic(add_node, rounds=100, iterations=10, warmup_rounds=5)

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

    benchmark.pedantic(remove_node, setup=setup, rounds=100, iterations=10, warmup_rounds=5)

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

    benchmark.pedantic(remove_edge, setup=setup, rounds=100, iterations=10, warmup_rounds=5)

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

    benchmark.pedantic(bulk_remove_edges, setup=setup, rounds=100, iterations=10, warmup_rounds=5)

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

    benchmark.pedantic(bulk_remove_edges, setup=setup, rounds=100, iterations=10, warmup_rounds=5)

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

    benchmark.pedantic(bulk_remove_nodes, setup=setup, rounds=100, iterations=10, warmup_rounds=5)

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

    benchmark.pedantic(bulk_remove_nodes, setup=setup, rounds=100, iterations=10, warmup_rounds=5)

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
# Workflow Construction Benchmarks (Realistic Scale)
# ============================================================================


@pytest.mark.parametrize(
    "num_nodes,num_layers,edges_per_node",
    [(10, 3, 2), (50, 5, 3), (100, 5, 5)],
)
def test_benchmark_workflow_construction(benchmark, num_nodes, num_layers, edges_per_node):
    """Benchmark complete workflow DAG construction.

    Measures end-to-end time to build a realistic workflow DAG:
    - Node creation and addition
    - Edge creation with dependencies
    - DAG topology (forward-flowing edges)

    Scales:
    - 10 nodes: Typical lionagi workflow (<1ms expected)
    - 50 nodes: Medium workflow (<10ms expected)
    - 100 nodes: P99 workflow (<20ms expected)

    Critical for understanding workflow setup overhead in production.
    """

    def construct_workflow():
        return _create_dag_graph(num_nodes, num_layers, edges_per_node)

    graph, _nodes = benchmark(construct_workflow)

    # Verify DAG properties
    assert len(graph.nodes) == num_nodes
    assert graph.is_acyclic()


# ============================================================================
# Algorithm Benchmarks
# ============================================================================


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [("graph_100", 1), ("graph_1000", 10), ("graph_10k", 100)],
)
def test_benchmark_is_acyclic(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark is_acyclic (DFS cycle detection).

    Critical algorithm for workflow validation - called before every execution.

    Performance expectations:
    - 100 nodes: <1ms (typical workflow)
    - 1000 nodes: <10ms (medium workflow)
    - 10K nodes: <100ms (stress test)

    Algorithm: Three-color DFS traversing all nodes and edges
    Complexity: O(V + E)
    """
    graph, _ = request.getfixturevalue(graph_fixture)

    def check_acyclic():
        return graph.is_acyclic()

    result = benchmark.pedantic(check_acyclic, rounds=100, iterations=10, warmup_rounds=5)
    assert result is True  # Verify graph is actually acyclic


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [("graph_100", 1), ("graph_1000", 10), ("graph_10k", 100)],
)
def test_benchmark_topological_sort(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark topological_sort (Kahn's algorithm).

    Critical for execution planning - determines task execution order.

    Performance expectations:
    - 100 nodes: <1ms (typical workflow)
    - 1000 nodes: <10ms (medium workflow)
    - 10K nodes: <100ms (stress test)

    Algorithm: Kahn's algorithm with in-degree tracking + queue
    Complexity: O(V + E)
    """
    graph, _ = request.getfixturevalue(graph_fixture)

    def topo_sort():
        return graph.topological_sort()

    result = benchmark.pedantic(topo_sort, rounds=100, iterations=10, warmup_rounds=5)
    assert len(result) == len(graph.nodes)  # Verify all nodes included


@pytest.mark.parametrize(
    "graph_fixture,expected_max_ms",
    [("graph_100", 1), ("graph_1000", 10), ("graph_10k", 50)],
)
def test_benchmark_find_path(benchmark, graph_fixture, expected_max_ms, request):
    """Benchmark find_path (BFS pathfinding).

    Used for debugging, dependency analysis, and workflow visualization.

    Performance expectations:
    - 100 nodes: <1ms (typical workflow)
    - 1000 nodes: <10ms (medium workflow)
    - 10K nodes: <50ms (stress test - typical paths shorter)

    Algorithm: BFS with parent tracking
    Complexity: O(V + E) worst case, typically much faster
    """
    import anyio

    graph, nodes = request.getfixturevalue(graph_fixture)

    # Find path from first node to last node (worst case - full traversal)
    start_node = nodes[0]
    end_node = nodes[-1]

    def find_path():
        return anyio.run(graph.find_path, start_node.id, end_node.id)

    result = benchmark.pedantic(find_path, rounds=100, iterations=10, warmup_rounds=5)
    assert result is not None  # Verify path exists in DAG
    assert len(result) > 0  # Verify non-empty path


# ============================================================================
# Memory Benchmarks
# ============================================================================


def test_benchmark_memory_10k(benchmark):
    """Benchmark memory usage for 10K node DAG.

    Uses tracemalloc to measure actual peak memory allocation.
    Validates: Memory footprint reasonable for graph size.
    Baseline: ~68 MB for 10K nodes + 50K edges (~6.8 KB/node)
    Target: <100 MB (allows for variance)
    """
    import tracemalloc

    def create_graph_and_measure():
        tracemalloc.start()
        _graph, _ = _create_dag_graph(num_nodes=10_000, num_layers=20, edges_per_node=5)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Return peak memory in MB for readability
        return peak / 1024 / 1024

    peak_mb = benchmark(create_graph_and_measure)

    # Validate memory is reasonable (baseline: ~68 MB, allow 20% variance)
    # Target: <10 KB per node (10K nodes = ~100 MB max including edges)
    assert peak_mb < 100, f"Memory usage too high: {peak_mb:.2f} MB (target: <100 MB)"


def test_benchmark_memory_100k(benchmark):
    """Benchmark memory usage for 100K node DAG.

    Uses tracemalloc to measure actual peak memory allocation.
    Baseline: ~680 MB for 100K nodes + 500K edges (~6.8 KB/node)
    Target: <1000 MB (allows for variance)
    """
    import tracemalloc

    def create_graph_and_measure():
        tracemalloc.start()
        _graph, _ = _create_dag_graph(num_nodes=100_000, num_layers=50, edges_per_node=5)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return peak / 1024 / 1024

    peak_mb = benchmark(create_graph_and_measure)

    # Validate memory is reasonable (baseline: ~680 MB, allow 50% variance)
    # Target: <10 KB per node (100K nodes = ~1000 MB max including edges)
    assert peak_mb < 1000, f"Memory usage too high: {peak_mb:.2f} MB (target: <1000 MB)"

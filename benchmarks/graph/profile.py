#!/usr/bin/env python
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Graph Performance Profiling

Profiles Graph operations at scale (10K-100K) to identify hot paths and optimization opportunities.

Usage:
    uv run python scripts/profile_graph.py [--size SIZE] [--memory] [--cpu]

Options:
    --size SIZE     Number of operations (default: 10000)
    --memory        Enable memory profiling (slower, detailed allocations)
    --cpu           Enable CPU profiling (default: always on)
    --help          Show this help

Output:
    - CPU profiling: Top 30 functions by cumulative time
    - Memory profiling: Line-by-line memory usage (if --memory)

Examples:
    # Quick CPU profiling with 10K operations
    uv run python scripts/profile_graph.py

    # Large-scale profiling with memory analysis
    uv run python scripts/profile_graph.py --size 50000 --memory

    # Focus on hot paths with 100K operations
    uv run python scripts/profile_graph.py --size 100000
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lionherd_core import Edge, Graph, Node


def build_graph(size: int) -> tuple[Graph, list[Node], list[Edge]]:
    """Build graph with specified number of nodes and edges.

    Creates a graph with 'size' nodes and approximately 2*size edges
    (forming a directed graph with average degree ~2).

    Args:
        size: Number of nodes to create

    Returns:
        Tuple of (graph, nodes, edges)
    """
    print(f"\n{'='*60}")
    print(f"Building graph with {size:,} nodes...")
    print(f"{'='*60}")

    graph = Graph()
    nodes = []
    edges = []

    # Phase 1: Add nodes
    start = time.perf_counter()
    for i in range(size):
        node = Node(content={"index": i, "data": f"node_{i}"})
        graph.add_node(node)
        nodes.append(node)
        if (i + 1) % (size // 10) == 0:
            print(f"  Added {i+1:,}/{size:,} nodes...")
    elapsed = time.perf_counter() - start
    print(f"✓ Nodes added: {elapsed:.2f}s ({size/elapsed:,.0f} ops/s)")

    # Phase 2: Add edges (create directed edges: i -> (i+1), i -> (i+2))
    start = time.perf_counter()
    for i in range(size):
        # Edge to next node (linear chain)
        if i < size - 1:
            edge = Edge(head=nodes[i].id, tail=nodes[i + 1].id, label=["next"])
            graph.add_edge(edge)
            edges.append(edge)

        # Edge to node+2 (skip connection)
        if i < size - 2:
            edge = Edge(head=nodes[i].id, tail=nodes[i + 2].id, label=["skip"])
            graph.add_edge(edge)
            edges.append(edge)

        if (i + 1) % (size // 10) == 0:
            print(f"  Added {(i+1)*2:,}/{size*2:,} edges...")

    elapsed = time.perf_counter() - start
    edge_count = len(edges)
    print(f"✓ Edges added: {elapsed:.2f}s ({edge_count/elapsed:,.0f} ops/s)")

    print(f"\n📊 Graph Stats:")
    print(f"   Nodes: {len(graph.nodes):,}")
    print(f"   Edges: {len(graph.edges):,}")

    return graph, nodes, edges


def profile_graph_operations(size: int = 10000):
    """Profile Graph add/remove/query operations.

    Args:
        size: Number of operations to perform
    """
    graph, nodes, edges = build_graph(size)

    # Profile graph queries
    print(f"\n{'='*60}")
    print("Profiling graph queries...")
    print(f"{'='*60}")

    start = time.perf_counter()
    for i in range(min(1000, size)):
        _ = graph.get_successors(nodes[i].id)
        _ = graph.get_predecessors(nodes[i].id)
        _ = graph.get_node_edges(nodes[i].id, direction="both")
    elapsed = time.perf_counter() - start
    ops = min(1000, size) * 3
    print(f"✓ Query operations: {elapsed:.2f}s ({ops/elapsed:,.0f} ops/s)")

    # Profile node removal (removes edges automatically)
    print(f"\n{'='*60}")
    print("Profiling node removal (cascading edge removal)...")
    print(f"{'='*60}")

    # Remove 10% of nodes to test cascade
    remove_count = size // 10
    start = time.perf_counter()
    for i in range(remove_count):
        # Remove from middle to test edge cleanup
        idx = size // 2 + i
        if idx < len(nodes):
            graph.remove_node(nodes[idx].id)
        if (i + 1) % (remove_count // 5) == 0:
            print(f"  Removed {i+1:,}/{remove_count:,} nodes...")

    elapsed = time.perf_counter() - start
    print(f"✓ Node removal: {elapsed:.2f}s ({remove_count/elapsed:,.0f} ops/s)")

    print(f"\n📊 Final Graph Stats:")
    print(f"   Nodes: {len(graph.nodes):,}")
    print(f"   Edges: {len(graph.edges):,}")


def profile_graph_algorithms(size: int = 1000):
    """Profile graph algorithms (smaller size for algorithms).

    Args:
        size: Number of nodes (algorithms are O(n²) or worse)
    """
    print(f"\n{'='*60}")
    print(f"Profiling graph algorithms (size={size:,})...")
    print(f"{'='*60}")

    graph, nodes, _ = build_graph(size)

    # Profile is_acyclic
    start = time.perf_counter()
    result = graph.is_acyclic()
    elapsed = time.perf_counter() - start
    print(f"✓ is_acyclic(): {elapsed:.3f}s (result={result})")

    # Profile topological_sort (only if acyclic)
    if result:
        start = time.perf_counter()
        sorted_nodes = graph.topological_sort()
        elapsed = time.perf_counter() - start
        print(f"✓ topological_sort(): {elapsed:.3f}s ({len(sorted_nodes):,} nodes)")


def profile_graph_serialization(size: int = 10000):
    """Profile graph serialization/deserialization.

    Args:
        size: Number of nodes in graph
    """
    print(f"\n{'='*60}")
    print("Profiling serialization...")
    print(f"{'='*60}")

    graph, _, _ = build_graph(size)

    # Profile to_dict
    start = time.perf_counter()
    data = graph.to_dict(mode="json")
    elapsed = time.perf_counter() - start
    print(f"✓ to_dict(mode='json'): {elapsed:.2f}s")

    # Profile from_dict
    start = time.perf_counter()
    restored = Graph.from_dict(data)
    elapsed = time.perf_counter() - start
    print(f"✓ from_dict(): {elapsed:.2f}s")
    print(f"   Restored: {len(restored.nodes):,} nodes, {len(restored.edges):,} edges")


def run_profiling(size: int, enable_memory: bool = False, enable_cpu: bool = True):
    """Run profiling with specified configuration.

    Args:
        size: Number of operations
        enable_memory: Enable memory profiling (slower)
        enable_cpu: Enable CPU profiling
    """
    print(f"\n{'='*60}")
    print(f"GRAPH PROFILING")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Operations: {size:,}")
    print(f"  CPU profiling: {'enabled' if enable_cpu else 'disabled'}")
    print(f"  Memory profiling: {'enabled' if enable_memory else 'disabled'}")

    if enable_cpu:
        # CPU profiling with cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    # Run workloads
    profile_graph_operations(size)
    profile_graph_algorithms(min(size // 10, 2000))  # Algorithms scale poorly
    profile_graph_serialization(min(size, 5000))  # Serialization is expensive

    if enable_cpu:
        profiler.disable()

        # Print CPU profiling results
        print(f"\n{'='*60}")
        print("CPU PROFILING RESULTS (Top 30 by cumulative time)")
        print(f"{'='*60}")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(30)
        print(stream.getvalue())

        # Hot path analysis
        print(f"\n{'='*60}")
        print("HOT PATH ANALYSIS (Top 20 by time)")
        print(f"{'='*60}")
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("time")
        stats.print_stats(20)
        print(stream.getvalue())

    if enable_memory:
        print("\n⚠️  Memory profiling requires @profile decorator on functions.")
        print("To enable detailed memory profiling:")
        print("  1. Add @profile decorator to Graph methods in graph.py")
        print("  2. Run: python -m memory_profiler scripts/profile_graph.py")


def main():
    parser = argparse.ArgumentParser(
        description="Profile Graph operations for performance optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Number of operations (default: 10000)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable memory profiling (slower)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=True,
        help="Enable CPU profiling (default: enabled)",
    )

    args = parser.parse_args()

    run_profiling(size=args.size, enable_memory=args.memory, enable_cpu=args.cpu)

    print(f"\n{'='*60}")
    print("PROFILING COMPLETE")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Review hot paths in CPU profiling results")
    print("  2. Compare against pandas/polars for baseline")
    print("  3. Consider Cython for hot functions (>10% cumulative time)")
    print("  4. Profile memory with: python -m memory_profiler scripts/profile_graph.py")


if __name__ == "__main__":
    main()

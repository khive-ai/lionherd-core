"""Test Graph.add_edge() race condition - Issue #21.

This test FORCES the race condition to trigger by adding artificial delays
that expand the race window between edges.add() and adjacency updates.

Without @synchronized on add_edge(), this test will FAIL.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from lionherd_core.base import Edge, Graph, Node


def test_graph_add_edge_race_condition_forced():
    """Force race condition by expanding the window between operations.

    The bug: Graph.add_edge() releases Pile's lock after edges.add()
    but before updating _out_edges and _in_edges dicts.

    Race window:
        Thread 1: edges.add(edge)       # LOCKED
        <-- LOCK RELEASED -->
        Thread 2: edges.add(edge2)      # LOCKED
        <-- LOCK RELEASED -->
        Thread 1: _out_edges[...] =     # NOT LOCKED - RACE!
        Thread 2: _out_edges[...] =     # NOT LOCKED - RACE!

    This test forces the race by monkey-patching Pile.add() to sleep,
    which expands the race window dramatically.

    Expected behavior:
    - WITHOUT @synchronized: FAILS (missing edges in adjacency)
    - WITH @synchronized: PASSES (atomic operation)
    """
    graph = Graph()

    # Add nodes
    nodes = [Node(content=f"node_{i}") for i in range(5)]
    for node in nodes:
        graph.add_node(node)

    # Patch Pile.add to add artificial delay that expands race window
    original_pile_add = graph.edges.add

    def delayed_add(item):
        result = original_pile_add(item)
        time.sleep(0.001)  # 1ms delay forces context switch
        return result

    graph.edges.add = delayed_add

    # Create edges that will be added concurrently
    edges_to_add = []
    for i in range(20):
        head = nodes[i % 5]
        tail = nodes[(i + 1) % 5]
        edges_to_add.append(Edge(head=head.id, tail=tail.id))

    def add_edge_task(edge):
        """Add single edge - will be called from multiple threads."""
        graph.add_edge(edge)

    # Execute with high concurrency to maximize race probability
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(add_edge_task, edge) for edge in edges_to_add]
        for f in futures:
            f.result()

    # Verify adjacency lists match edges
    # If race occurred, some edge IDs will be missing from adjacency dicts
    missing_out = []
    missing_in = []

    for edge in graph.edges:
        if edge.id not in graph._out_edges.get(edge.head, set()):
            missing_out.append(edge.id)
        if edge.id not in graph._in_edges.get(edge.tail, set()):
            missing_in.append(edge.id)

    # Without @synchronized, this assertion will FAIL
    assert len(missing_out) == 0, (
        f"Race condition detected: {len(missing_out)} edges missing from _out_edges. "
        f"Missing: {missing_out[:5]}"
    )
    assert len(missing_in) == 0, (
        f"Race condition detected: {len(missing_in)} edges missing from _in_edges. "
        f"Missing: {missing_in[:5]}"
    )


class VulnerableGraph(Graph):
    """Graph subclass that AMPLIFIES the race condition for testing.

    This adds an artificial delay between edges.add() and adjacency updates
    to expand the race window and make the bug easy to reproduce.
    """

    def add_edge(self, edge: Edge) -> None:
        """Add edge with expanded race window - DO NOT USE IN PRODUCTION."""
        if edge.id in self.edges:
            raise ValueError(f"Edge {edge.id} already exists in graph")
        if edge.head not in self.nodes:
            raise ValueError(f"Head node {edge.head} not in graph")
        if edge.tail not in self.nodes:
            raise ValueError(f"Tail node {edge.tail} not in graph")

        self.edges.add(edge)
        # CRITICAL: Sleep here expands race window 1000x
        time.sleep(0.001)  # Force context switch between operations
        self._out_edges[edge.head].add(edge.id)
        self._in_edges[edge.tail].add(edge.id)


def test_graph_add_edge_race_amplified():
    """Use VulnerableGraph to prove race condition exists and can corrupt data.

    This test WILL FAIL because we've amplified the race window with time.sleep().

    Once real Graph gets @synchronized, we'll test it also passes.
    """
    graph = VulnerableGraph()

    # Add nodes
    nodes = [Node(content=f"node_{i}") for i in range(3)]
    for node in nodes:
        graph.add_node(node)

    # Create edges
    edges = [Edge(head=nodes[i % 3].id, tail=nodes[(i + 1) % 3].id) for i in range(30)]

    def add_edge_task(edge):
        graph.add_edge(edge)

    # 30 threads adding 30 edges with amplified race window
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(add_edge_task, edge) for edge in edges]
        for f in futures:
            f.result()

    # Check for corruption
    missing_out = []
    missing_in = []

    for edge in graph.edges:
        if edge.id not in graph._out_edges.get(edge.head, set()):
            missing_out.append(edge.id)
        if edge.id not in graph._in_edges.get(edge.tail, set()):
            missing_in.append(edge.id)

    # This WILL FAIL - proving the race condition exists
    assert len(missing_out) == 0, (
        f"RACE CONDITION DETECTED: {len(missing_out)}/{len(graph.edges)} edges "
        f"missing from _out_edges due to concurrent updates"
    )
    assert len(missing_in) == 0, (
        f"RACE CONDITION DETECTED: {len(missing_in)}/{len(graph.edges)} edges "
        f"missing from _in_edges due to concurrent updates"
    )

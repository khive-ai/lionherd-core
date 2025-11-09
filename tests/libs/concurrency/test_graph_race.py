"""Test Graph.add_edge() race condition - Issue #21.

This test verifies that @synchronized on Graph mutation methods prevents
race conditions under concurrent access.

With @synchronized on all mutation methods, these tests PASS.
Without @synchronized, VulnerableGraph demonstrates the corruption.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from lionherd_core.base import Edge, Graph, Node


def test_graph_add_edge_race_condition_forced():
    """Verify @synchronized prevents race conditions under concurrent load.

    The bug (now fixed): Graph.add_edge() had a race window between
    edges.add() and adjacency dict updates (_out_edges, _in_edges).

    Race window (WITHOUT @synchronized):
        Thread 1: edges.add(edge)       # Pile's lock
        Thread 2: edges.add(edge2)      # Pile's lock
        Thread 1: _out_edges[...] =     # NOT LOCKED - RACE!
        Thread 2: _out_edges[...] =     # NOT LOCKED - RACE!

    With @synchronized: Entire operation is atomic via RLock.

    This test stresses the Graph with high concurrency to ensure
    @synchronized correctly prevents adjacency corruption.
    """
    graph = Graph()

    # Add nodes
    nodes = [Node(content={"value": f"node_{i}"}) for i in range(5)]
    for node in nodes:
        graph.add_node(node)

    # Create edges that will be added concurrently
    edges_to_add = []
    for i in range(30):
        head = nodes[i % 5]
        tail = nodes[(i + 1) % 5]
        edges_to_add.append(Edge(head=head.id, tail=tail.id))

    def add_edge_task(edge):
        """Add single edge - will be called from multiple threads."""
        graph.add_edge(edge)
        # Small delay to increase thread interleaving
        time.sleep(0.0001)

    # Execute with high concurrency to stress synchronization
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(add_edge_task, edge) for edge in edges_to_add]
        for f in futures:
            f.result()

    # Verify adjacency lists are consistent with edges
    # With @synchronized, this should ALWAYS pass
    missing_out = []
    missing_in = []

    for edge in graph.edges:
        if edge.id not in graph._out_edges.get(edge.head, set()):
            missing_out.append(edge.id)
        if edge.id not in graph._in_edges.get(edge.tail, set()):
            missing_in.append(edge.id)

    # With @synchronized, adjacency should be consistent
    assert len(missing_out) == 0, (
        f"Adjacency corruption: {len(missing_out)} edges missing from _out_edges. "
        f"@synchronized failed to prevent race. Missing: {missing_out[:5]}"
    )
    assert len(missing_in) == 0, (
        f"Adjacency corruption: {len(missing_in)} edges missing from _in_edges. "
        f"@synchronized failed to prevent race. Missing: {missing_in[:5]}"
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
    """Use VulnerableGraph to demonstrate what happens WITHOUT @synchronized.

    VulnerableGraph removes @synchronized and adds artificial delays to
    amplify the race window, proving the vulnerability pattern exists.

    This test demonstrates the corruption but doesn't fail - it documents
    the race condition behavior for educational purposes.
    """
    graph = VulnerableGraph()

    # Add nodes
    nodes = [Node(content={"value": f"node_{i}"}) for i in range(3)]
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

    # Without @synchronized, we expect corruption
    # This documents the race but doesn't fail the test
    if missing_out or missing_in:
        print(
            f"\n[Expected] VulnerableGraph corruption detected: "
            f"{len(missing_out)} out-edges, {len(missing_in)} in-edges missing. "
            f"This demonstrates why @synchronized is necessary."
        )
    # No assertion - this test documents the vulnerability, doesn't validate fix

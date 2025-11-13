# Graph Benchmark Results - Issue #130

**Date**: 2025-11-13 | **Runtime**: 35.75s (14 tests) | **Status**: ✅ All passing

---

## Summary

### ✅ Meeting Targets

- **Memory**: 3.4 KB/node (target <10KB) - 66% under budget
- **CRUD**: add_node 27μs, remove_node 1.23ms - meets targets
- **Workflows**: All sub-millisecond (10-100 nodes)
- **find_path**: ~1μs (10,000x faster than target)

### ❌ Missing Targets

- **is_acyclic**: 36.4ms (target 10ms) - 3.6x slower
- **topological_sort**: 79.2ms (target 10ms) - 7.9x slower

---

## Performance Results

### Algorithms (1000 nodes)

| Algorithm        | Mean Time | Target  | Ratio  |
|------------------|-----------|---------|--------|
| find_path        | 1.0μs     | <10ms   | 0.0001x|
| is_acyclic       | 36.4ms    | <10ms   | 3.6x   |
| topological_sort | 79.2ms    | <10ms   | 7.9x   |

### Scaling (100→1000 nodes)

Expected: 10x | Actual: 17x (suggests O(n log n) from dict operations)

---

## Root Cause Analysis

**find_path is fast**: Early termination - only traverses ~10 nodes in stratified DAG

**is_acyclic is slow**: Must traverse ALL nodes - O(V+E) = ~6000 operations (expected)

**topological_sort is slowest**: Calls `is_acyclic()` first (~36ms) then runs Kahn's (~43ms)

- **Problem**: Kahn's algorithm naturally detects cycles - redundant check

---

## Quick Win: Remove Redundant Cycle Check

**File**: `src/lionherd_core/base/graph.py:378-401`

**Change**: Remove `if not self.is_acyclic()` check, add cycle detection at end:

```python
def topological_sort(self) -> list[Node]:
    in_degree = {nid: len(edges) for nid, edges in self._in_edges.items()}
    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    result = []

    while queue:
        node_id = queue.popleft()
        result.append(self.nodes[node_id])
        for edge_id in self._out_edges[node_id]:
            neighbor_id = self.edges[edge_id].tail
            in_degree[neighbor_id] -= 1
            if in_degree[neighbor_id] == 0:
                queue.append(neighbor_id)

    if len(result) != len(self.nodes):  # Natural cycle detection
        raise ValueError("Cannot topologically sort graph with cycles")

    return result
```

**Expected**: 79ms → 43ms (45% speedup)

---

## Verdict

Benchmarks are **working correctly**:

- ✅ Identified real inefficiency (double cycle checking)
- ✅ Confirmed expected behavior (early exit vs full traversal)
- ✅ Ready for CI/CD regression detection

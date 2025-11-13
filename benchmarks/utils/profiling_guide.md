# Scripts Directory

This directory contains utility scripts for development, testing, and performance analysis.

## Profiling Scripts

### Overview

Profiling scripts identify performance bottlenecks in Graph and Flow operations at scale (10K-100K operations). Use these to guide optimization decisions and target hot paths for potential Cython implementation.

**Target**: Achieve pandas-level performance through data-driven optimization.

### Quick Start

```bash
# Install profiling dependencies
uv sync --group dev

# Profile Graph operations (10K operations, ~30s)
uv run python scripts/profile_graph.py

# Profile Flow operations (10K operations, ~30s)
uv run python scripts/profile_flow.py

# Large-scale profiling (50K operations, ~2-3min)
uv run python scripts/profile_graph.py --size 50000
uv run python scripts/profile_flow.py --size 50000

# Stress test (100K operations, ~5-10min)
uv run python scripts/profile_graph.py --size 100000
```

### Profile Graph Operations

**Script**: `profile_graph.py`

**Profiles**:
- Node add/remove operations (with cascading edge cleanup)
- Edge add/remove operations
- Graph queries (predecessors, successors, node_edges)
- Graph algorithms (is_acyclic, topological_sort, find_path)
- Serialization (to_dict/from_dict)

**Workload**:
- Nodes: N operations
- Edges: ~2N operations (average degree ~2)
- Queries: min(1000, N) × 3 query types
- Removal: N/10 nodes (tests cascade behavior)
- Algorithms: N/10 nodes (O(n²) scaling)

**Usage**:
```bash
# Basic profiling with 10K nodes
uv run python scripts/profile_graph.py

# Large-scale with 50K nodes
uv run python scripts/profile_graph.py --size 50000

# Show help
uv run python scripts/profile_graph.py --help
```

**Output**:
1. Performance metrics for each operation type
2. CPU profiling: Top 30 functions by cumulative time
3. Hot path analysis: Top 20 functions by self time
4. Optimization recommendations

### Profile Flow Operations

**Script**: `profile_flow.py`

**Profiles**:
- Item add/remove operations (with progression cascade)
- Progression add/remove operations
- Progression queries (by ID/name)
- Referential integrity validation
- Serialization (to_dict/from_dict)

**Workload**:
- Items: N operations
- Progressions: 10 progressions (N/10 items each)
- Item additions: N/10 new items
- Item removal: N/20 items (tests progression cascade)
- Progression management: 50 add + 25 remove

**Usage**:
```bash
# Basic profiling with 10K items
uv run python scripts/profile_flow.py

# Large-scale with 50K items
uv run python scripts/profile_flow.py --size 50000

# Show help
uv run python scripts/profile_flow.py --help
```

**Output**:
1. Performance metrics for each operation type
2. CPU profiling: Top 30 functions by cumulative time
3. Hot path analysis: Top 20 functions by self time
4. Optimization recommendations

### Memory Profiling

For detailed line-by-line memory analysis:

```bash
# Install memory-profiler (already in dev dependencies)
uv sync --group dev

# Add @profile decorator to target functions in source code
# Example: Add to Graph.add_node(), Flow.add_item(), etc.

# Run with memory_profiler
python -m memory_profiler scripts/profile_graph.py
python -m memory_profiler scripts/profile_flow.py
```

**Note**: Memory profiling adds significant overhead (~10-50ms per decorated function). Use selectively on suspected hot paths.

### Interpreting Results

#### CPU Profiling

**Cumulative Time**: Total time spent in function + all callees
- Functions with >10% cumulative time are optimization targets
- Look for functions called many times (ncalls column)

**Self Time**: Time spent in function only (excluding callees)
- High self time = computational bottleneck
- Candidates for Cython/numba optimization

#### Hot Path Identification

Look for:
1. **Function call overhead**: High ncalls, low self time
   - Solution: Inline or Cython compile
2. **Memory allocations**: dict/list operations in hot loops
   - Solution: Preallocate or use numpy arrays
3. **Repeated work**: Same computation in tight loops
   - Solution: Cache or memoize
4. **O(n²) patterns**: nested loops over large collections
   - Solution: Algorithm redesign or C extension

#### Optimization Priority

1. **P0 (Critical)**: >20% cumulative time, called >1M times
2. **P1 (High)**: 10-20% cumulative time, called >100K times
3. **P2 (Medium)**: 5-10% cumulative time, optimization nice-to-have
4. **P3 (Low)**: <5% cumulative time, premature optimization

### Comparison Benchmarks

To validate optimization impact:

```bash
# Before optimization
uv run python scripts/profile_graph.py --size 50000 > before.txt

# After optimization (e.g., Cython implementation)
uv run python scripts/profile_graph.py --size 50000 > after.txt

# Compare results
diff before.txt after.txt
```

**Pandas baseline**: For operations like add/remove at 50K scale, pandas achieves:
- Add operations: ~500K-1M ops/s
- Remove operations: ~200K-500K ops/s
- Queries: ~1M-5M ops/s

Target: Match or exceed pandas performance for equivalent operations.

### Advanced Usage

#### Custom Workload Sizes

Test scaling behavior across sizes:

```bash
# Small (fast feedback, ~5s)
uv run python scripts/profile_graph.py --size 1000

# Medium (default, ~30s)
uv run python scripts/profile_graph.py --size 10000

# Large (stress test, ~2-3min)
uv run python scripts/profile_graph.py --size 50000

# Extra large (production scale, ~10min)
uv run python scripts/profile_graph.py --size 100000
```

#### Profiling Specific Components

Modify scripts to focus on specific operations:

```python
# In profile_graph.py, comment out unwanted sections
# profile_graph_operations(size)
# profile_graph_algorithms(min(size // 10, 2000))  # Skip algorithms
profile_graph_serialization(min(size, 5000))  # Only serialization
```

## Other Scripts

### validate_docs.py

Validates documentation for broken links and outdated references.

```bash
uv run python scripts/validate_docs.py
```

## Contributing

When adding new scripts:

1. Add comprehensive docstrings with usage examples
2. Include CLI help (`--help` flag)
3. Make scripts executable: `chmod +x scripts/new_script.py`
4. Update this README with script description and usage
5. Follow existing patterns for consistency

## Performance Guidelines

### Profiling Best Practices

1. **Profile before optimizing**: Don't guess, measure
2. **Profile representative workloads**: Use realistic data sizes
3. **Profile multiple times**: Average results to reduce noise
4. **Profile incrementally**: One optimization at a time
5. **Compare before/after**: Validate optimization impact

### Optimization Strategy

```
Measure → Identify hotspots → Optimize → Verify → Repeat
```

1. **Measure**: Run profiling scripts, collect baseline
2. **Identify**: Find functions with >10% cumulative time
3. **Optimize**: Apply targeted optimizations (Cython, algorithm, caching)
4. **Verify**: Re-run profiling, compare metrics
5. **Repeat**: Continue until performance targets met

### When to Optimize

- **Do optimize**: Functions with >10% cumulative time, called >100K times
- **Consider**: Functions with 5-10% cumulative time, clear optimization path
- **Don't optimize**: Functions with <5% cumulative time (premature)

### Optimization Techniques

1. **Cython**: For computational hot paths (10-100x speedup)
2. **Caching**: For expensive repeated computations
3. **Preallocate**: Reduce memory allocation overhead
4. **Algorithm**: O(n²) → O(n log n) wins over micro-optimizations
5. **Batching**: Reduce function call overhead

## License

Apache 2.0 - See LICENSE file for details.

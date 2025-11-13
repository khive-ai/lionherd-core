# Graph Benchmarks

Performance benchmarks for Graph operations at realistic and stress test scales.

## Statistical Methodology

**Purpose**: Reliable regression detection with statistically valid measurements.

**Configuration**:

- **Rounds**: 100 (number of times to run the benchmark)
- **Iterations**: 10 (operations per round, amortizes setup overhead)
- **Warmup**: 5 rounds (eliminates JIT/cache cold start effects)
- **Total measurements**: 1000 per benchmark (100 rounds × 10 iterations)

**Fixture Isolation**:

- Function-scoped fixtures prevent pollution
- Each test gets a fresh graph (no cumulative overhead)
- Trade-off: ~2-3s setup per test vs clean baselines

**Why this matters**:

- Warmup rounds eliminate cold start bias (JIT compilation, cache loading)
- Multiple iterations reduce measurement noise
- Function scope prevents fixture accumulation (no growing graphs)
- Consistent statistics enable reliable regression detection (<5% tolerance)

## Scale Distinction

### Realistic Workflows (10-100 nodes)

Typical lionagi workflow patterns based on production usage:

- **Common**: 6-12 nodes (small agents, simple flows)
- **Medium**: 20-50 nodes (multi-step workflows)
- **P99**: 50-100 nodes (complex orchestrations)

**Fixtures**: `graph_10`, `graph_50`, `graph_100`
**Focus**: Workflow construction, validation, execution planning
**Targets**: Sub-millisecond to 10ms for common operations

### Stress Tests (10K-100K nodes)

Large-scale graphs for testing scalability and edge cases:

- **10K nodes**: Large agent systems, complex dependencies
- **100K nodes**: Extreme scale, stress testing

**Fixtures**: `graph_10k`, `graph_100k`
**Focus**: CRUD operations, memory usage, linear scaling
**Targets**: No exponential blowup, reasonable memory footprint

## Quick Start

```bash
# Run all Graph benchmarks (speed + memory)
uv run pytest benchmarks/graph/ --benchmark-only

# Run realistic workflow benchmarks only
uv run pytest benchmarks/graph/ -k "workflow" --benchmark-only

# Run 10K stress benchmarks only
uv run pytest benchmarks/graph/ -k "10k" --benchmark-only

# Run memory benchmarks only
uv run pytest benchmarks/graph/ -k "memory" --benchmark-only

# Save baseline
uv run pytest benchmarks/graph/ --benchmark-save=graph_baseline

# Compare with baseline
uv run pytest benchmarks/graph/ --benchmark-compare=graph_baseline
```

## Profiling

### Speed Profiling

```bash
# Quick CPU profiling (10K operations)
uv run python benchmarks/graph/profiling.py --size 10000

# Large-scale (50K operations)
uv run python benchmarks/graph/profiling.py --size 50000
```

### Memory Profiling

```bash
# Memory profiling with per-operation tracking
uv run python benchmarks/graph/profiling.py --size 10000 --memory

# Large-scale memory analysis
uv run python benchmarks/graph/profiling.py --size 50000 --memory
```

**Memory profiling output includes**:

- Peak memory usage (MB)
- Current memory per operation
- Top 10 memory allocations per phase
- Memory per node (KB/node)

### Comprehensive Profiling

```bash
# Both CPU and memory profiling
uv run python benchmarks/graph/profiling.py --size 10000 --memory --cpu
```

## Algorithm Benchmarks

Critical graph algorithms used in workflow execution and validation:

- `is_acyclic`: DFS cycle detection (called before every workflow execution)
- `topological_sort`: Kahn's algorithm (determines task execution order)
- `find_path`: BFS pathfinding (debugging, dependency analysis)

**Test scales**:

- 100 nodes: Typical workflow size
- 1000 nodes: Medium workflow
- 10K nodes: Stress test

**Performance goals**:

- 100 nodes: <1ms per algorithm
- 1000 nodes: <10ms per algorithm
- 10K nodes: <100ms (is_acyclic, topological_sort), <50ms (find_path)

### Running Algorithm Benchmarks

```bash
# Run all algorithm benchmarks
uv run pytest benchmarks/graph/ -k "algorithm" --benchmark-only

# Run specific algorithm
uv run pytest benchmarks/graph/ -k "is_acyclic" --benchmark-only
uv run pytest benchmarks/graph/ -k "topological_sort" --benchmark-only
uv run pytest benchmarks/graph/ -k "find_path" --benchmark-only

# Run at specific scale
uv run pytest benchmarks/graph/ -k "algorithm and 100" --benchmark-only
uv run pytest benchmarks/graph/ -k "algorithm and 1000" --benchmark-only
```

## Memory Benchmarks

Memory benchmarks use `tracemalloc` to measure actual peak memory allocation (not time):

- `test_benchmark_memory_10k`: 10K nodes + 50K edges (target: <50 MB)
- `test_benchmark_memory_100k`: 100K nodes + 500K edges (target: <500 MB)

**Expected memory per node**: ~5-10 KB/node (including edges, adjacency structures)

## Performance Goals

### Speed

- `add_node`: <100μs (O(1))
- `remove_node`: <1ms for degree ~5
- `remove_edge`: <50μs (O(1))
- Bulk operations: Linear scaling

### Memory

- Peak memory: <10 KB per node (realistic workloads)
- No memory leaks during removal operations
- Stable memory usage across operations

See `analysis.md` for detailed performance breakdown and optimization recommendations.

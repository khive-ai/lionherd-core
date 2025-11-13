# Lionherd Core Benchmarking Guide

Comprehensive guide to performance benchmarking in lionherd-core.

## Available Benchmark Suites

**Currently Implemented**: LNDL Parser (benchmarks/lndl/)
**Planned**: Pile, Flow, Graph (infrastructure ready, implementation pending)

### 1. Pile[T] Benchmarks (PLANNED)

**File**: `test_pile_benchmarks.py`
**Component**: `lionherd_core.base.Pile`
**Comparisons**: dict, OrderedDict, pandas.Index, polars.Series

**Key Findings**:

- Pile is ~100x slower than dict for add operations
- O(1) operations (get/contains) within acceptable range
- Remove is O(n) bottleneck due to Progression scanning
- 4.4x memory overhead vs dict
- Type validation adds minimal overhead (1.6-4.6%)

**See**: PILE_BENCHMARK_ANALYSIS.md (to be created with implementation)

```bash
# Run Pile benchmarks
uv run pytest benchmarks/test_pile_benchmarks.py --benchmark-only
```

### 2. Flow Benchmarks (PLANNED)

**File**: `test_flow_benchmarks.py`
**Component**: `lionherd_core.base.Flow`
**Focus**: Progression operations, referential integrity

**Key Metrics**:

- Pandas-level performance for data structure operations
- <5% regression threshold
- 1K and 10K dataset sizes

**See**: [README.md](./README.md) (Flow-specific)

```bash
# Run Flow benchmarks
uv run pytest benchmarks/test_flow_benchmarks.py --benchmark-only
```

## Quick Start

### Run All Benchmarks

```bash
# All benchmarks in directory
uv run pytest benchmarks/ --benchmark-only

# With better stability
uv run pytest benchmarks/ --benchmark-only --benchmark-disable-gc --benchmark-warmup=on
```

### Run Specific Suite

```bash
# Pile benchmarks only
uv run pytest benchmarks/test_pile_benchmarks.py --benchmark-only

# Flow benchmarks only
uv run pytest benchmarks/test_flow_benchmarks.py --benchmark-only
```

### Save and Compare Results

```bash
# Save baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Run new benchmarks
uv run pytest benchmarks/ --benchmark-only --benchmark-save=new

# Compare
pytest-benchmark compare baseline new
```

## General Benchmarking Principles

### 1. Stable Environment

**Before running benchmarks**:

- Close unnecessary applications
- Disable power management / turbo boost
- Ensure consistent system load
- Run multiple times for validation

**CI Environment**:

- Use dedicated benchmark runners
- Pin CPU frequency
- Isolate from other workloads

### 2. Proper Test Structure

```python
def test_operation(benchmark):
    """Benchmark description."""

    def setup():
        # Expensive setup (not measured)
        data = create_test_data()
        return (data,), {}

    def run(data):
        # Operation to measure
        result = operation(data)

    # Use pedantic for setup isolation
    benchmark.pedantic(run, setup=setup, rounds=10)
```

**Key Points**:

- Use `setup()` to exclude initialization time
- Return data as `(args,), kwargs` tuple
- Use `pedantic()` for explicit control
- Specify `rounds` for consistency

### 3. Parametrization

```python
@pytest.fixture(params=[100, 1_000, 10_000])
def size(request):
    """Parametrize sizes."""
    return request.param

def test_operation(benchmark, size):
    """Test scales with size."""
    data = create_data(size)
    benchmark(operation, data)
```

**Benefits**:

- Single test, multiple sizes
- Easy scaling analysis
- Consistent methodology

### 4. Baseline Comparisons

```python
def test_implementation(benchmark, pile):
    """Your implementation."""
    benchmark(pile.operation)

def test_baseline_dict(benchmark, dict_baseline):
    """Baseline: dict."""
    benchmark(dict_baseline.operation)

def test_baseline_pandas(benchmark, pandas_baseline):
    """Baseline: pandas."""
    benchmark(pandas_baseline.operation)
```

**Pattern**:

- Same test structure for all implementations
- Consistent naming (`test_pile_*`, `test_dict_*`)
- Same size parameters
- Same operation semantics

## Common Options

### Performance

```bash
# Disable garbage collection (more stable)
--benchmark-disable-gc

# Warmup rounds (for JIT)
--benchmark-warmup=on
--benchmark-warmup-iterations=10

# Calibration precision
--benchmark-calibration-precision=10

# Number of rounds
--benchmark-min-rounds=10
```

### Output

```bash
# Specific columns
--benchmark-columns=min,max,mean,median,stddev,ops

# Sort by metric
--benchmark-sort=mean
--benchmark-sort=name

# Verbose
-v

# Save results
--benchmark-save=name
--benchmark-autosave
```

### Filtering

```bash
# Only benchmarks
--benchmark-only

# Skip benchmarks
--benchmark-skip

# Specific tests
-k "test_pile_add"
-k "1k"
```

## Interpreting Results

### Time Units

- **ns (nanoseconds)**: 1 ns = 0.000001 ms
- **μs (microseconds)**: 1 μs = 0.001 ms
- **ms (milliseconds)**: 1 ms = 0.001 s
- **s (seconds)**: 1 s

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Min | Fastest iteration | Best-case performance |
| Max | Slowest iteration | Worst-case performance |
| Mean | Average | Primary metric for comparison |
| Median | Middle value | Robust to outliers |
| StdDev | Variability | Lower = more consistent |
| IQR | Middle 50% | Robust variability measure |
| OPS | Operations/sec | Higher = faster |
| Outliers | Anomalies | May indicate issues |

### Overhead Calculation

```python
# Simple ratio
overhead = implementation_time / baseline_time

# Example:
pile_add = 926.9 μs
dict_add = 12.1 μs
overhead = 926.9 / 12.1 = 76.6x

# Interpretation: Pile is 76x slower than dict for add
```

### Regression Detection

```bash
# Before change
pytest --benchmark-only --benchmark-save=before

# After change
pytest --benchmark-only --benchmark-compare=before

# Fail if >5% slower
pytest --benchmark-only --benchmark-compare=before --benchmark-compare-fail=mean:5%
```

**Regression Thresholds**:

- <2%: Noise (acceptable)
- 2-5%: Minor (investigate)
- 5-10%: Moderate (justify)
- >10%: Major (requires fix)

## Memory Profiling

### Using memory_profiler

```bash
# Install
uv pip install memory-profiler

# Add decorator
@profile
def test_memory():
    pile = Pile(items=[...])
    # ...

# Run
python -m memory_profiler benchmarks/test_pile_benchmarks.py
```

### Using sys.getsizeof

```python
import sys

def test_memory_overhead(benchmark):
    pile = benchmark(create_pile, size=10_000)

    print(f"Pile._items: {sys.getsizeof(pile._items)} bytes")
    print(f"Pile._progression: {sys.getsizeof(pile._progression.order)} bytes")
```

**Limitations**:

- `sys.getsizeof()` doesn't include referenced objects
- Only measures direct container overhead
- Use memory_profiler for comprehensive analysis

## Best Practices

### ✅ DO

1. **Isolate Setup**

   ```python
   def setup():
       return (expensive_setup(),), {}
   benchmark.pedantic(run, setup=setup)
   ```

2. **Fresh Data Per Iteration**

   ```python
   def setup():
       return ([1, 2, 3],), {}  # New list each time
   ```

3. **Equivalent Comparisons**

   ```python
   # Compare same operations
   pile.get(uuid)  vs  dict[uuid]
   ```

4. **Statistical Significance**

   ```python
   # Enough rounds for stable mean
   benchmark.pedantic(run, rounds=10)
   ```

### ❌ DON'T

1. **Measure Setup Time**

   ```python
   # Bad: includes setup
   def run():
       data = expensive_setup()
       operation(data)
   ```

2. **Reuse Mutable Data**

   ```python
   # Bad: first iteration empties list
   data = [1, 2, 3]
   def run():
       data.clear()
   ```

3. **Compare Different Operations**

   ```python
   # Bad: different semantics
   pile[lambda x: x.value > 5]  vs  dict_comprehension
   ```

4. **Ignore Variance**

   ```python
   # Bad: high variance unreliable
   # StdDev > 20% of mean → investigate
   ```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        run: pip install uv

      - name: Install deps
        run: uv sync --all-extras

      - name: Run benchmarks
        run: |
          uv run pytest benchmarks/ \\
            --benchmark-only \\
            --benchmark-json=output.json \\
            --benchmark-compare-fail=mean:10%

      - name: Store results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: output.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

### Regression Gates

```bash
# Fail if mean increases by >10%
--benchmark-compare-fail=mean:10%

# Fail if min increases by >15%
--benchmark-compare-fail=min:15%

# Multiple thresholds
--benchmark-compare-fail=mean:10% --benchmark-compare-fail=stddev:20%
```

## Troubleshooting

### High Variance (StdDev > 10%)

**Symptoms**: Inconsistent results, high StdDev

**Causes**:

- Background processes
- Thermal throttling
- Memory pressure
- GC interference

**Solutions**:

```bash
# Disable GC
--benchmark-disable-gc

# More warmup
--benchmark-warmup=on --benchmark-warmup-iterations=20

# More rounds
--benchmark-min-rounds=20
```

### Outliers

**Symptoms**: Many outliers detected

**Causes**:

- System interrupts
- Disk I/O
- Network activity
- Scheduler preemption

**Solutions**:

- Close background apps
- Isolate benchmark environment
- Use dedicated hardware
- Increase sample size

### Fixture Reuse Issues

**Symptoms**: Benchmarks fail after first iteration

**Cause**: Fixtures are reused but test mutates them

**Solution**:

```python
# Use setup function
def test_operation(benchmark):
    def setup():
        # Fresh data each iteration
        return (create_fresh_data(),), {}

    benchmark.pedantic(run, setup=setup)
```

## Writing New Benchmarks

### 1. Choose Component

Identify what to benchmark:

- Data structure operations (Pile, Flow, Graph)
- Algorithms (pathfinding, validation)
- Serialization (to_dict, from_dict)
- Integration (adapter operations)

### 2. Create Test File

```python
# benchmarks/test_<component>_benchmarks.py

"""Benchmarks for <Component>.

Methodology:
    - Size scales: [100, 1K, 10K]
    - Comparisons: <baseline1>, <baseline2>
    - Operations: <op1>, <op2>, <op3>

See: <COMPONENT>_BENCHMARK_ANALYSIS.md
"""

import pytest
from lionherd_core import <Component>

@pytest.fixture(params=[100, 1_000, 10_000])
def size(request):
    return request.param

class TestCoreOperations:
    def test_operation(self, benchmark, size):
        \"\"\"Benchmark <operation>.\"\"\"
        # Implementation
```

### 3. Add Baseline Comparisons

```python
def test_implementation(benchmark, size):
    \"\"\"Your implementation.\"\"\"
    # ...

def test_baseline_dict(benchmark, size):
    \"\"\"Baseline: dict.\"\"\"
    # ...

def test_baseline_pandas(benchmark, size):
    \"\"\"Baseline: pandas.\"\"\"
    # ...
```

### 4. Document Results

Create `<COMPONENT>_BENCHMARK_ANALYSIS.md`:

- Performance characteristics
- Overhead quantification
- Decision matrix (when to use)
- Optimization recommendations

### 5. Update This Guide

Add section to "Available Benchmark Suites" above.

## Resources

- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [Python performance tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [Profiling Python](https://docs.python.org/3/library/profile.html)
- [memory_profiler](https://pypi.org/project/memory-profiler/)

---

**Last Updated**: 2025-11-13
**Contributors**: Claude (Implementer)
**License**: Apache-2.0

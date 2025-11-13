# Flow Performance Benchmarks

Comprehensive pytest-benchmark suite for Flow operations validation.

## Purpose

Validates that Flow operations maintain pandas-level performance after architectural changes (e.g., PR #117 progression refactor).

**Regression threshold**: <5%
**Target**: Pandas-level performance for data structure operations

## Datasets

- **1K**: 1,000 items + 10 progressions (small workflow)
- **10K**: 10,000 items + 100 progressions (large workflow)

## Benchmark Categories

### Single Operations

- `test_flow_add_item` - Add single item to flow
- `test_flow_remove_item` - Remove single item from flow
- `test_flow_add_progression` - Add single progression
- `test_flow_remove_progression` - Remove single progression
- `test_flow_get_progression_by_name` - Name-based lookup (O(1))
- `test_flow_get_progression_by_uuid` - UUID-based lookup (O(1))

### Bulk Operations

- `test_flow_bulk_add_items` - Add 100 items in loop
- `test_flow_bulk_remove_items` - Remove 100 items in loop

### Progression Operations

- `test_flow_progression_append` - Append item to progression (O(1))
- `test_flow_progression_remove` - Remove item from progression (O(n))
- `test_flow_progression_traversal` - Iterate progression.order
- `test_flow_progression_contains` - Check item membership (O(1))

### Complex Operations

- `test_flow_add_item_to_multiple_progressions` - M:N relationship overhead
- `test_flow_remove_item_cascade` - Cascade delete across progressions

### Serialization

- `test_flow_to_dict` - Flow serialization
- `test_flow_from_dict` - Flow deserialization

### Referential Integrity

- `test_flow_referential_integrity_validation_1k` - Validation during construction

### Name Index

- `test_flow_name_index_lookup_1k` - Dict lookup performance
- `test_flow_name_index_rebuild_after_deserialization` - Index rebuild cost

## Usage

### Run All Benchmarks

```bash
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only
```

### Run Specific Size

```bash
# 1K dataset only
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only -k "1k"

# 10K dataset only
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only -k "10k"
```

### Run Specific Operation

```bash
# All add_item benchmarks
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only -k "add_item"

# Progression operations only
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only -k "progression"
```

### Save Baseline

```bash
# Save current results as baseline
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-save=baseline_pr117
```

### Compare with Baseline

```bash
# Compare current results with baseline
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-compare=0001

# Compare with named baseline
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-compare=baseline_pr117
```

### Advanced Options

```bash
# Warmup + disable GC for more consistent results
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-warmup=on --benchmark-disable-gc

# Verbose output
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only -v

# Generate HTML report
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-json=output.json
```

## Interpreting Results

### Key Metrics

- **Min/Max**: Range of execution times
- **Mean**: Average execution time (primary metric)
- **StdDev**: Consistency (lower is better)
- **Median**: Middle value (robust to outliers)
- **IQR**: Interquartile range (variability)
- **OPS**: Operations per second (higher is better)

### Performance Expectations

| Operation | Expected Complexity | Target Performance |
|-----------|-------------------|--------------------|
| add_item | O(1) | <20μs |
| remove_item | O(P) [P=progressions] | <100μs |
| add_progression | O(1) | <100μs |
| get_progression (name) | O(1) | <10μs |
| get_progression (uuid) | O(1) | <10μs |
| progression.append | O(1) amortized | <100ns |
| progression.remove | O(n) [n=progression length] | Variable |
| progression traversal | O(n) | ~1ms per 100 items |
| bulk operations | O(k) [k=batch size] | Linear scaling |

### Regression Detection

Compare before/after results:

```bash
# Before PR
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-save=before

# After PR
uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-compare=before
```

Look for:

- **>5% increase in Mean**: Performance regression
- **Significant StdDev increase**: Consistency regression
- **Outliers increase**: Edge case issues

## Fixture Design

### Module-Scoped Fixtures

- `items_1k` / `items_10k`: Pre-created items (reusable)
- `flow_1k` / `flow_10k`: Pre-populated flows (read-only)

### Function-Scoped Fixtures

- `fresh_flow_1k` / `fresh_flow_10k`: Modifiable flows (per test)

This design avoids setup time contamination in benchmark measurements.

## Notes

- **10K from_dict**: Skipped (too slow for baseline, >20s per iteration)
- **Warmup**: Recommended for JIT compilation stability
- **GC**: Disabled for more consistent measurements
- **Counters**: Dict-based to avoid closure issues in benchmark loops

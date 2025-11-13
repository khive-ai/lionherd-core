# lionherd-core Benchmarks

Comprehensive performance benchmarks for all lionherd-core components.

## Quick Start

```bash
# Run all benchmarks
uv run pytest benchmarks/ --benchmark-only

# Run specific component
uv run pytest benchmarks/graph/ --benchmark-only

# Save baseline for regression testing
uv run pytest benchmarks/ --benchmark-save=v1.0.0-alpha5

# Compare against baseline
uv run pytest benchmarks/ --benchmark-compare=v1.0.0-alpha5
```

## Components

- **graph/** - Graph operations (10K-100K nodes)
- **flow/** - Flow operations (1K-10K items)  
- **pile/** - Pile vs dict/pandas/polars comparisons
- **lndl/** - LNDL parser vs json/orjson trade-offs
- **utils/** - Shared utilities and profiling tools

## Structure

Each component follows the same structure:

```text
{component}/
├── test_benchmarks.py    # pytest-benchmark tests
├── profile.py            # Profiling script (optional)
├── analysis.md           # Performance analysis
├── README.md             # Component-specific docs
└── baselines/            # Saved benchmark results
```

## CI Integration

Automated benchmark regression detection runs on every PR:

**Regression Thresholds:**
>
- >10% slower: ❌ Fails PR
- 5-10% slower: ⚠️ Warning (doesn't fail)
- >5% faster: ✅ Improvement

**Workflow:**

1. PR triggers benchmark run
2. Compares against baseline (main branch)
3. Comments on PR with results
4. Fails PR if >10% regression detected

**Baseline Management:**

- Baseline updated on merge to main
- Stored as GitHub artifact (90 days retention)
- Accessible via workflow dispatch for manual runs

See `GUIDE.md` for detailed benchmarking best practices.

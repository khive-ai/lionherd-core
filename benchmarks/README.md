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

## Future: CI Integration

Planned benchmark automation for regression detection:

Benchmarks run on every PR to detect performance regressions:

- Baseline: Previous release version
- Threshold: <10% regression allowed
- Alert: >10% regression requires justification

See `GUIDE.md` for detailed benchmarking best practices and CI integration examples.

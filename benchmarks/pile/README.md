# Pile Benchmarks

Performance tracking for Pile[T] data structure.

## Dependencies

Benchmarks require pytest-benchmark:

```bash
uv pip install pytest-benchmark
```

## Quick Start

```bash
# Run all Pile benchmarks
uv run pytest benchmarks/pile/ --benchmark-only

# Run 1K size only (faster)
uv run pytest benchmarks/pile/ -k "1k" --benchmark-only

# Save baseline
uv run pytest benchmarks/pile/ --benchmark-save=v1.0.0-alpha5

# Compare against baseline
uv run pytest benchmarks/pile/ --benchmark-compare=v1.0.0-alpha5
```

See `analysis.md` for current performance measurements.

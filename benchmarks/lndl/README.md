# LNDL Parser Benchmarks

Performance tracking for LNDL parser.

## Quick Start

```bash
# Run benchmarks
uv run pytest benchmarks/lndl/ --benchmark-only

# Save baseline for version
uv run pytest benchmarks/lndl/ --benchmark-save=v1.0.0-alpha5

# Compare against baseline
uv run pytest benchmarks/lndl/ --benchmark-compare=v1.0.0-alpha5
```

See `analysis.md` for current performance measurements.

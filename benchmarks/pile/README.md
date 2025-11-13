# Pile Benchmarks

Comprehensive benchmarks comparing Pile[T] vs dict/pandas/polars.

## Quick Start

```bash
# Run all Pile benchmarks
uv run pytest benchmarks/pile/ --benchmark-only

# Run 1K size only (faster)
uv run pytest benchmarks/pile/ -k "1k" --benchmark-only

# Save baseline
uv run pytest benchmarks/pile/ --benchmark-save=pile_baseline
```

## Key Findings

- Pile provides type safety + Observable protocol at 100x slower than dict
- Memory overhead: 4.4x vs dict
- Recommendation: Use Pile for <10K items when type safety needed

See `analysis.md` for detailed comparison and decision matrix.


# Flow Benchmarks

Performance benchmarks for Flow operations at scale.

## Quick Start

```bash
# Run all Flow benchmarks
uv run pytest benchmarks/flow/ --benchmark-only

# Run 1K benchmarks only (faster)
uv run pytest benchmarks/flow/ -k "1k" --benchmark-only

# Save baseline
uv run pytest benchmarks/flow/ --benchmark-save=flow_baseline
```

## Profiling

```bash
uv run python benchmarks/flow/profile_flow.py --size 10000
```

See `USAGE.md` for performance breakdown and optimization recommendations.

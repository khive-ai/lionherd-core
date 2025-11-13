# Graph Benchmarks

Performance benchmarks for Graph operations at scale.

## Quick Start

```bash
# Run all Graph benchmarks
uv run pytest benchmarks/graph/ --benchmark-only

# Run 10K benchmarks only
uv run pytest benchmarks/graph/ -k "10k" --benchmark-only

# Save baseline
uv run pytest benchmarks/graph/ --benchmark-save=graph_baseline

# Compare with baseline
uv run pytest benchmarks/graph/ --benchmark-compare=graph_baseline
```

## Profiling

```bash
# Quick profiling (5K operations)
uv run python benchmarks/graph/profile.py --size 5000

# Large-scale (50K operations)
uv run python benchmarks/graph/profile.py --size 50000
```

See `analysis.md` for detailed performance breakdown.


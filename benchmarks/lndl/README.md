# LNDL Parser Benchmarks

Benchmarks comparing LNDL fuzzy parser vs json.loads/orjson/pydantic.

## Quick Start

```bash
# Run all LNDL benchmarks
uv run pytest benchmarks/lndl/ --benchmark-only

# Save baseline
uv run pytest benchmarks/lndl/ --benchmark-save=lndl_baseline
```

## Trade-off Analysis

- LNDL: 100% success rate on malformed LLM output
- json.loads: 60% success rate (strict parsing)
- Overhead: +10-50ms (acceptable for I/O-bound LLM workloads)

See `analysis.md` for detailed trade-off analysis and decision matrix.

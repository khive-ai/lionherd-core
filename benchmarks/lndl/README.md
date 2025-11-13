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

- LNDL: 100% success rate on malformed LNDL output (typos, case issues)
- fuzzy_json: 100% success rate on malformed JSON (commas, quotes, brackets)
- pydantic: 80%+ success rate (type coercion)
- Overhead: +~50-90μs for both LNDL and fuzzy_json (acceptable for I/O-bound LLM workloads)

See `analysis.md` for detailed trade-off analysis and decision matrix.

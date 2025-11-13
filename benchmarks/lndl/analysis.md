# LNDL Parser Performance

## Baseline Measurements

**Parsing Speed:**

- LNDL (perfect input): ~43μs for 100B (23K ops/s)
- LNDL (fuzzy mode): ~89μs for 100B (11K ops/s)

**Scaling:**

- 100B: ~43μs
- 1KB: ~47μs
- 10KB: ~182μs

**Comparisons:**

- fuzzy_json: ~6μs (15x faster)
- orjson: ~0.3μs (300x faster)

## Regression Tracking

Use benchmarks to track performance over versions:

```bash
# Save baseline
uv run pytest benchmarks/lndl/ --benchmark-save=v1.0.0-alpha5

# Compare future versions
uv run pytest benchmarks/lndl/ --benchmark-compare=v1.0.0-alpha5
```

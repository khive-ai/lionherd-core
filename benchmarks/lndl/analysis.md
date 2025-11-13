# LNDL Parser Performance Analysis

## Trade-off Validation

**Success Rate:**
- LNDL fuzzy: **100%** (5/5 malformed test cases)
- json.loads: **60%** (3/5 test cases)
- pydantic: **60%** (3/5 test cases)

**Speed:**
- LNDL: ~43μs for 100B input (23K ops/s)
- json.loads: ~1μs (1M ops/s)
- orjson: ~0.3μs (3.5M ops/s)

**Decision Matrix:**
- Use LNDL for LLM output (unreliable format)
- Use orjson for perfect JSON (API responses)
- Use json.loads for stdlib-only (no dependencies)

## Benchmark Results

See test_benchmarks.py for comprehensive comparisons.


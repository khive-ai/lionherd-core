# LNDL Parser Performance Analysis

## Success Rate: Tolerance vs Quality

**Important**: Higher success rate measures **tolerance for LLM output variability**, not quality.
Strict parsers (json.loads, Pydantic) **correctly reject** malformed input - this is their design.

**Test Results on LLM Output Variability (typos, case variations):**

- LNDL fuzzy: **100%** (5/5 test cases with typos/case issues)
- fuzzy_json: **100%** (5/5 - handles malformed JSON: commas, quotes, brackets)
- pydantic: **80%+** (coerces some type mismatches)

**Interpretation:**

- LNDL handles LNDL format typos/case (field names, model names)
- fuzzy_json handles JSON syntax errors (commas, quotes, brackets)
- Both are fuzzy parsers - fair comparison shows LNDL's LNDL-specific tolerance

## Speed: Parsing Overhead Only

**IMPORTANT**: These measurements are parsing overhead only, NOT including fuzzy correction.
Fuzzy matching (typo correction) adds ~50μs additional overhead not shown here.

**Parsing Speed (perfect input):**

- LNDL: ~43μs for 100B input (23K ops/s)
- json.loads: ~1μs (1M ops/s)
- orjson: ~0.3μs (3.5M ops/s)

**Full LNDL Pipeline (parsing + fuzzy correction):**

- ~50-100μs for 100B input (accounting for fuzzy matching overhead)

## Decision Matrix

**When to use LNDL:**

- LLM output where format variability expected (typos, case issues)
- Acceptable to trade speed for tolerance
- Need to handle slightly malformed but recoverable input

**When NOT to use LNDL:**

- Perfect JSON from APIs (use orjson - 100x faster)
- Malformed JSON without LNDL format (use fuzzy_json - handles JSON syntax errors)
- Need strict format validation (use Pydantic)
- Performance critical paths (both LNDL and fuzzy_json have ~50-100μs overhead)

## Benchmark Details

See `test_benchmarks.py` for:

- Comprehensive speed comparisons
- Tolerance testing methodology
- Input size scaling analysis (100B, 1KB, 10KB)

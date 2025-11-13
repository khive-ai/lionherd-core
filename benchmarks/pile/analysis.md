# Pile[T] Performance

## Baseline Measurements

**Core Operations (10K items):**

- Add: ~93ms (88x slower than dict ~1ms)
- Remove: ~70ms (O(n) progression scan)
- Get: ~71ms (100x slower than dict ~0.7ms)
- Contains: ~68ms (100x slower than dict ~0.7ms)
- Iteration: ~9.7ms (177x slower than dict ~55μs)

**Type Validation Overhead:**

- No validation: 90ms baseline
- Permissive mode: 94ms (+4.6%)
- Strict mode: 91ms (+1.6%)

**Memory Overhead:**

- 100 items: 1,145 bytes (4.3x vs dict 264 bytes)
- 1K items: 11,835 bytes (4.3x vs dict 2,777 bytes)
- 10K items: 119,867 bytes (4.4x vs dict 27,198 bytes)

**Comparisons:**

- OrderedDict iteration: 8x faster than Pile
- dict is ~100x faster for lookups (get, contains)
- dict is ~88x faster for add operations

## Scaling

**Add operations:**

- 100 items: ~927μs
- 1K items: ~9.2ms
- 10K items: ~93ms

**Remove operations (50% of items):**

- 100 items: ~600μs
- 1K items: ~6.2ms
- 10K items: ~69ms

## Regression Tracking

Use benchmarks to track performance over versions:

```bash
# Save baseline
uv run pytest benchmarks/pile/ --benchmark-save=v1.0.0-alpha5

# Compare future versions
uv run pytest benchmarks/pile/ --benchmark-compare=v1.0.0-alpha5
```

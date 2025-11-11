# hash_dict

> Stable hashing for nested data structures including dicts, lists, sets, and Pydantic models

## Overview

`hash_dict()` generates stable, deterministic hash values for complex Python data structures that are normally unhashable (dicts, lists, sets). It provides **order-independent hashing** for collections and **automatic Pydantic model support**, making it ideal for caching, deduplication, and comparison operations.

**Key Capabilities:**

- **Order-Independent Hashing**: Dicts and sets hash identically regardless of insertion order
- **Deep Nested Support**: Recursively handles nested structures (dict of lists of sets, etc.)
- **Pydantic Model Support**: Automatically handles BaseModel instances via `model_dump()`
- **Mixed Type Collections**: Stable sorting for sets/frozensets containing mixed unorderable types
- **Fallback Handling**: Graceful handling of unhashable objects via `str()`/`repr()` or type-based IDs

**When to Use hash_dict:**

- Caching structured LLM outputs or API responses
- Deduplicating complex data structures in collections
- Content-based comparison of nested configurations
- Generating stable keys for memoization
- Hashing Pydantic models for cache invalidation

**When NOT to Use hash_dict:**

- Simple hashable primitives (str, int, float, bool, None) - use built-in `hash()`
- Performance-critical tight loops - has overhead from recursive traversal
- Cryptographic hashing - use `hashlib` for security-sensitive hashing
- Identity-based hashing - use object `id()` or UUID-based hashing

## Function Signature

```python
from lionherd_core.ln import hash_dict

def hash_dict(data: Any, strict: bool = False) -> int:
    """Generate stable hash for any data structure including dicts, lists, and Pydantic models.

    Args:
        data: Data to hash (dict, list, BaseModel, or any object)
        strict: If True, deepcopy data before hashing to prevent mutation side effects

    Returns:
        Integer hash value (stable across equivalent structures)

    Raises:
        TypeError: If generated representation is not hashable
    """
```

## Parameters

### hash_dict Parameters

**data** : Any

Data structure to hash. Supports:

- **Primitives**: str, int, float, bool, None (returned as-is)
- **Collections**: dict, list, tuple, set, frozenset (recursively processed)
- **Pydantic Models**: BaseModel instances (converted via `model_dump()`)
- **Custom Objects**: Fallback to `str()`, `repr()`, or type-based identifier

**strict** : bool, default False

Deep-copy protection flag. If True, creates a deep copy of `data` before hashing to prevent potential mutation side effects during traversal.

- **False** (default): Hash directly (faster, assumes data won't mutate during hashing)
- **True**: Deep copy first (safer, adds overhead, prevents mutation issues)

**Use strict=True when:**

- Data contains mutable objects with custom `__hash__` that may mutate during hashing
- Paranoid about side effects in untrusted data structures
- Debugging hash inconsistencies potentially caused by mutation

**Use strict=False (default) when:**

- Data is trusted and won't mutate during hashing
- Performance matters (deep copy overhead avoided)
- Data is already immutable or frozen

## Returns

**int** : Stable hash value

Returns Python's built-in `hash()` of the canonical representation. Hash properties:

- **Deterministic**: Same input always produces same hash (within Python session)
- **Order-Independent**: `{'a': 1, 'b': 2}` hashes identically to `{'b': 2, 'a': 1}`
- **Type-Aware**: `[1, 2, 3]` hashes differently from `(1, 2, 3)` and `{1, 2, 3}`
- **Session-Scoped**: Hashes may differ across Python interpreter restarts (hash randomization)

**Cross-Session Stability**: Python's `hash()` uses process-specific randomization for security. For cross-session/cross-process stable hashing, consider alternatives like `hashlib.md5(repr(data).encode()).hexdigest()`.

## Raises

**TypeError** : If generated representation is not hashable

Raised when the canonical representation generation fails to produce a hashable object. This is rare and typically indicates:

- Circular references creating infinite recursion (Python stack limit)
- Custom objects with broken `__str__`/`__repr__` implementations
- Unexpected edge cases in type handling

Error message includes input type, representation type, and original error for debugging.

## Implementation Details

### Canonical Representation Algorithm

`hash_dict()` converts input data to a **canonical hashable representation** via `_generate_hashable_representation()`:

1. **Primitive Types**: Returned as-is (already hashable)
2. **Pydantic Models**: Converted via `model_dump()`, then recursively processed, marked with `_TYPE_MARKER_PYDANTIC`
3. **Dicts**: Sorted by stringified keys, values recursively processed, marked with `_TYPE_MARKER_DICT`
4. **Lists**: Elements recursively processed, marked with `_TYPE_MARKER_LIST`
5. **Tuples**: Elements recursively processed, marked with `_TYPE_MARKER_TUPLE`
6. **Sets/Frozensets**: Sorted (with fallback for mixed types), recursively processed, marked with `_TYPE_MARKER_SET`/`_TYPE_MARKER_FROZENSET`
7. **Fallback**: `str()` → `repr()` → `<unhashable:TypeName:id>`

**Type Markers** distinguish collection types with identical content (e.g., `[1, 2]` vs `(1, 2)`).

### Mixed Type Sorting

For sets/frozensets with unorderable types (e.g., `{1, 'a', True}`), fallback sorting uses `(str(type(x)), str(x))` for deterministic ordering:

```python
# Handles mixed types that can't be compared directly
{1, 'a', True, None} → sorted by ("<class 'bool'>", 'True'), ("<class 'int'>", '1'), ...
```

This ensures stable hashing even for complex mixed-type collections.

### Lazy Pydantic Import

Pydantic's `BaseModel` is imported on first `hash_dict()` call (not at module import) to:

- Avoid circular import issues
- Reduce import overhead if hash_dict never used
- Support environments without Pydantic (graceful degradation)

Global `_INITIALIZED` flag tracks initialization state.

## Usage Patterns

### Basic Hashing

```python
from lionherd_core.ln import hash_dict

# Hash dict (order-independent)
data1 = {'a': 1, 'b': 2}
data2 = {'b': 2, 'a': 1}
assert hash_dict(data1) == hash_dict(data2)  # True (same content, different order)

# Hash list (order-dependent)
list1 = [1, 2, 3]
list2 = [3, 2, 1]
assert hash_dict(list1) != hash_dict(list2)  # True (different order)

# Hash nested structures
nested = {
    'config': {'timeout': 30, 'retries': 3},
    'handlers': ['log', 'alert'],
    'enabled': True
}
h = hash_dict(nested)
```

### Pydantic Model Hashing

```python
from lionherd_core.ln import hash_dict
from pydantic import BaseModel

class Config(BaseModel):
    timeout: int
    retries: int
    enabled: bool = True

# Hash Pydantic models
config1 = Config(timeout=30, retries=3)
config2 = Config(timeout=30, retries=3)
assert hash_dict(config1) == hash_dict(config2)  # True (same content)

# Different values = different hash
config3 = Config(timeout=60, retries=3)
assert hash_dict(config1) != hash_dict(config3)  # True
```

### Caching and Memoization

```python
from lionherd_core.ln import hash_dict
from functools import lru_cache

# Memoize function with complex inputs
@lru_cache(maxsize=128)
def expensive_computation(data_hash: int):
    # Actual computation...
    return result

# Wrapper to hash complex inputs
def compute(config: dict):
    h = hash_dict(config)
    return expensive_computation(h)

# Cache hits for equivalent configs
result1 = compute({'timeout': 30, 'retries': 3})
result2 = compute({'retries': 3, 'timeout': 30})  # Cache hit (same hash)
```

### Deduplication

```python
from lionherd_core.ln import hash_dict

# Deduplicate list of dicts
configs = [
    {'host': 'a', 'port': 80},
    {'port': 80, 'host': 'a'},  # Duplicate (different order)
    {'host': 'b', 'port': 443},
]

# Using hash as deduplication key
seen_hashes = set()
unique_configs = []

for config in configs:
    h = hash_dict(config)
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique_configs.append(config)

print(len(unique_configs))  # 2 (first two deduplicated)
```

### Strict Mode for Safety

```python
from lionherd_core.ln import hash_dict

# Potentially mutable data
data = {
    'items': [1, 2, 3],
    'config': {'setting': 'value'}
}

# Default: hash directly (fast)
h1 = hash_dict(data)

# Strict: deep copy first (safe from mutations)
h2 = hash_dict(data, strict=True)

assert h1 == h2  # True (same content)

# If data mutates during hashing (rare edge case), strict=True protects
```

## Common Pitfalls

### Pitfall 1: Expecting Cross-Session Stability

**Issue**: Hashes differ across Python interpreter restarts.

```python
# Session 1
data = {'key': 'value'}
h1 = hash_dict(data)
print(h1)  # -3456789012345678901

# Session 2 (new Python process)
data = {'key': 'value'}
h2 = hash_dict(data)
print(h2)  # 1234567890123456789 (different!)
```

**Cause**: Python's `hash()` uses per-process randomization (PEP 456) for security.

**Solution**: For persistent hashing across sessions/processes, use cryptographic hashing:

```python
import hashlib
import json

def stable_hash(data):
    # Deterministic serialization + cryptographic hash
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()
```

### Pitfall 2: Hashing Unhashable Objects

**Issue**: Custom objects without proper `__str__`/`__repr__` produce unstable fallback hashes.

```python
class CustomObj:
    def __init__(self, value):
        self.value = value

obj = CustomObj(42)
h = hash_dict(obj)
# Fallback: hash(f"<unhashable:CustomObj:{id(obj)}>")
# Uses object ID, unstable across runs
```

**Solution**: Implement `__repr__` or convert to dict before hashing:

```python
class CustomObj:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"CustomObj(value={self.value})"

# Or convert to dict
def to_dict(obj):
    return {'value': obj.value}

h = hash_dict(to_dict(obj))  # Stable
```

### Pitfall 3: Type Confusion

**Issue**: Assuming different collection types with same content hash identically.

```python
# Different types = different hashes (even with same content)
assert hash_dict([1, 2, 3]) != hash_dict((1, 2, 3))  # True (list != tuple)
assert hash_dict({1, 2, 3}) != hash_dict([1, 2, 3])  # True (set != list)
```

**Cause**: Type markers distinguish collection types.

**Solution**: Normalize to same type before hashing if type-agnostic comparison needed:

```python
def normalized_hash(data):
    # Convert all to list for type-agnostic hashing
    if isinstance(data, (list, tuple, set, frozenset)):
        return hash_dict(sorted(list(data)))
    return hash_dict(data)
```

### Pitfall 4: Performance on Large Datasets

**Issue**: Recursive traversal on massive nested structures is slow.

```python
# Large nested structure
large_data = {'items': [{'id': i, 'data': list(range(100))} for i in range(10000)]}
h = hash_dict(large_data)  # Slow (deep recursion)
```

**Solution**: Hash summary or subset for large data:

```python
# Hash summary instead of full data
def fast_hash(large_data):
    summary = {
        'count': len(large_data['items']),
        'first_id': large_data['items'][0]['id'],
        'last_id': large_data['items'][-1]['id']
    }
    return hash_dict(summary)
```

## Design Rationale

### Why Order-Independent Hashing?

Dicts and sets are inherently unordered in Python's semantics. Order-independent hashing ensures:

1. **Semantic Equivalence**: `{'a': 1, 'b': 2}` and `{'b': 2, 'a': 1}` represent the same data
2. **Robustness**: Hash stability regardless of insertion order or dict implementation
3. **Deduplication**: Content-based deduplication works correctly for configs/API responses

Lists and tuples preserve order (order-dependent hashing) because order is semantically meaningful.

### Why Type Markers?

Type markers (`_TYPE_MARKER_DICT`, `_TYPE_MARKER_LIST`, etc.) distinguish collection types with identical content:

```python
# Without markers: collisions
[1, 2, 3] → (1, 2, 3) → hash((1, 2, 3))
(1, 2, 3) → (1, 2, 3) → hash((1, 2, 3))  # Collision!

# With markers: distinct
[1, 2, 3] → (_TYPE_MARKER_LIST, (1, 2, 3))
(1, 2, 3) → (_TYPE_MARKER_TUPLE, (1, 2, 3))  # Different!
```

This prevents false positives when comparing different types.

### Why Lazy Pydantic Import?

Importing Pydantic at module level causes:

1. **Circular Import Risk**: lionherd-core uses Pydantic, Pydantic models may use hash_dict
2. **Import Overhead**: Pydantic is heavy; skip if hash_dict unused
3. **Optional Dependency**: Gracefully degrades if Pydantic unavailable

Global `_INITIALIZED` flag ensures one-time import on first use.

### Why Strict Mode?

`strict=True` guards against rare edge cases where:

1. **Custom `__hash__` Mutates**: Object's hash method has side effects
2. **Concurrent Modification**: Data mutates during hashing in multithreaded code
3. **Debugging**: Isolate hash inconsistencies from mutation issues

Default `strict=False` favors performance (deep copy overhead avoided) since mutation during hashing is rare in practice.

## See Also

- **Related Utilities**:
  <!-- TODO: Add link to to_dict() documentation when available (issue #115) -->
  - [HashableModel](../types/model.md): Pydantic model with built-in content-based hashing
- **Related Concepts**:
  - [`Element.__hash__()`](../base/element.md#__hash__): Identity-based hashing (ID-based, not content-based)
  - Python's `hash()` built-in: Primitive hashing for hashable types
  - `hashlib` module: Cryptographic hashing for security-sensitive use cases

## Examples

```python
# Standard imports for ln.hash examples
from lionherd_core.ln import hash_dict
from pydantic import BaseModel
```

### Example 1: LLM Output Deduplication

```python
from lionherd_core.ln import hash_dict

# LLM generates multiple responses (may have duplicates)
responses = [
    {'answer': 'Paris', 'confidence': 0.95},
    {'confidence': 0.95, 'answer': 'Paris'},  # Duplicate (different order)
    {'answer': 'London', 'confidence': 0.80},
    {'answer': 'Paris', 'confidence': 0.95},  # Duplicate (same order)
]

# Deduplicate by content hash
unique_responses = {}
for resp in responses:
    h = hash_dict(resp)
    if h not in unique_responses:
        unique_responses[h] = resp

print(list(unique_responses.values()))
# [
#   {'answer': 'Paris', 'confidence': 0.95},
#   {'answer': 'London', 'confidence': 0.80}
# ]
```

### Example 2: Config Change Detection

```python
from lionherd_core.ln import hash_dict
from pydantic import BaseModel

class AppConfig(BaseModel):
    database_url: str
    timeout: int
    retries: int

# Current config
old_config = AppConfig(database_url='postgres://...', timeout=30, retries=3)
old_hash = hash_dict(old_config)

# New config
new_config = AppConfig(database_url='postgres://...', timeout=60, retries=3)
new_hash = hash_dict(new_config)

if old_hash != new_hash:
    print("Configuration changed! Reloading...")
    # Reload application with new config
```

### Example 3: Nested Structure Comparison

```python
from lionherd_core.ln import hash_dict

# Complex nested structures
config_a = {
    'services': {
        'api': {'port': 8000, 'workers': 4},
        'db': {'host': 'localhost', 'port': 5432}
    },
    'features': ['auth', 'logging', 'metrics']
}

config_b = {
    'features': ['auth', 'logging', 'metrics'],
    'services': {
        'db': {'port': 5432, 'host': 'localhost'},  # Different order
        'api': {'workers': 4, 'port': 8000}
    }
}

# Content-based comparison (order-independent)
assert hash_dict(config_a) == hash_dict(config_b)  # True
```

### Example 4: Memoization with Complex Keys

```python
from lionherd_core.ln import hash_dict
import time

# Cache for expensive operations
cache = {}

def expensive_operation(params: dict):
    # Generate cache key from complex params
    cache_key = hash_dict(params)

    if cache_key in cache:
        print("Cache hit!")
        return cache[cache_key]

    print("Computing...")
    time.sleep(2)  # Simulate expensive work
    result = sum(params.get('values', []))

    cache[cache_key] = result
    return result

# First call (cache miss)
result1 = expensive_operation({'values': [1, 2, 3], 'mode': 'sum'})
# Output: Computing...

# Second call with different order (cache hit)
result2 = expensive_operation({'mode': 'sum', 'values': [1, 2, 3]})
# Output: Cache hit!

assert result1 == result2
```

### Example 5: Mixed Type Set Hashing

```python
from lionherd_core.ln import hash_dict

# Set with mixed unorderable types
mixed_set = {1, 'a', True, None, 3.14}

# hash_dict handles mixed types with stable sorting
h = hash_dict(mixed_set)

# Recreate set with different insertion order
mixed_set2 = {None, 3.14, 1, True, 'a'}

# Same hash (order-independent + stable mixed-type sorting)
assert hash_dict(mixed_set) == hash_dict(mixed_set2)

# Type markers ensure sets hash differently from lists
assert hash_dict(mixed_set) != hash_dict(list(mixed_set))
```

### Example 6: Strict Mode for Untrusted Data

```python
from lionherd_core.ln import hash_dict

# Untrusted data from external API
untrusted_data = {
    'user_input': ['<script>', 'data'],
    'config': {'dangerous': 'value'}
}

# Use strict mode for safety (deep copy before hashing)
h = hash_dict(untrusted_data, strict=True)

# Original data unchanged (deep copy protected it)
print(untrusted_data)
# {'user_input': ['<script>', 'data'], 'config': {'dangerous': 'value'}}
```

# lcall

> Apply functions to iterables with configurable input/output processing

## Overview

`lcall` is a synchronous list processing utility that applies a callable to each element in an iterable with extensive control over input normalization and output transformation. It provides a unified interface for mapping operations with built-in flattening, null filtering, deduplication, and value extraction.

**Key Capabilities:**

- **Synchronous Mapping**: Apply function to each element sequentially
- **Input Processing**: Flatten, dropna, unique, value extraction before function application
- **Output Processing**: Flatten, dropna, unique after function application
- **Flexible Input**: Accepts single items, iterables, or nested structures
- **Type Safety**: Full type hints with generics for input and output types
- **Early Termination**: Graceful handling of `InterruptedError` with partial results

**When to Use lcall:**

- Transforming lists with complex pre/post-processing requirements
- Normalizing heterogeneous input structures before mapping
- Extracting enum values or dict values before processing
- Flattening nested results after function application
- Removing None values from inputs or outputs
- Deduplicating inputs or outputs during transformation

**When NOT to Use lcall:**

- **Async operations**: Use `alcall` for async/await functions (planned in `lionherd_core.ln`)
- **Simple transformations**: Use built-in `map()` or list comprehensions for straightforward cases
- **Parallel execution**: Use `concurrent.futures` or multiprocessing for CPU-bound work
- **Streaming large datasets**: Use generators or iterators to avoid memory overhead

<!-- TODO: Add link to alcall documentation when available (issue #115) -->

## Function Signature

```python
from lionherd_core.ln import lcall

def lcall(
    input_: Iterable[T] | T,
    func: Callable[[T], R] | Iterable[Callable[[T], R]],
    /,
    *args: Any,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_use_values: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    **kwargs: Any,
) -> list[R]:
    """Apply function to each element synchronously with optional input/output processing."""
```

## Parameters

### Positional Parameters

**input_** : Iterable[T] or T (positional-only)

Items to process. Can be a single item (auto-converted to list), an iterable, or nested structures.

- Single items are wrapped in a list: `5` → `[5]`
- Iterables are converted to list: `(1, 2, 3)` → `[1, 2, 3]`
- Nested structures flattened if `input_flatten=True`
- Type: Generic type `T` (input element type)

**func** : Callable[[T], R] or Iterable[Callable[[T], R]] (positional-only)

Callable to apply to each element. If an iterable of callables is provided, it must contain exactly one callable (extracted automatically).

- Signature: Must accept input element as first positional argument
- Additional arguments passed via `*args` and `**kwargs`
- Type: Generic callable `T → R` (input type to return type)
- Validation: Raises `ValueError` if not callable or iterable with ≠1 callable

**\*args** : Any

Positional arguments passed to `func` after the input element.

- Position: Passed as `func(item, *args, **kwargs)`
- Use case: Shared arguments across all function calls

**\*\*kwargs** : Any

Keyword arguments passed to `func`.

- Merged with function call: `func(item, *args, **kwargs)`
- Use case: Configuration parameters, optional settings

### Input Processing Parameters

**input_flatten** : bool, default False

Flatten nested input structures before applying function.

- Recursively flattens lists, tuples (if `input_flatten_tuple_set=True`), sets (if enabled)
- Does NOT flatten strings (treated as atomic values)
- Use case: Normalize nested data structures before processing
- See: `to_list(..., flatten=True)` for flattening behavior

**input_dropna** : bool, default False

Remove `None` and undefined values from input before applying function.

- Filters out: `None`, `undefined` sentinel
- Use case: Skip null values without explicit checks in function
- Combines with `input_flatten` for deep null removal

**input_unique** : bool, default False

Remove duplicate input elements before applying function.

- Deduplication: Uses set-based uniqueness (requires hashable elements)
- Order: Preserves first occurrence order
- Use case: Avoid redundant computation on duplicate inputs
- Requires: `input_flatten=True` or `input_dropna=True` (validated by `to_list`)

**input_use_values** : bool, default False

Extract values from enums and mappings before applying function.

- Enums: Extracts `.value` attribute
- Mappings (dict-like): Extracts values (discards keys)
- Use case: Process enum values or dict values instead of containers
- Combines with flattening for nested value extraction

**input_flatten_tuple_set** : bool, default False

Include tuples and sets when flattening input structures.

- Default behavior: Only flattens lists
- If `True`: Also flattens tuples and sets
- Use case: Normalize all iterable types to flat list
- Requires: `input_flatten=True` to take effect

### Output Processing Parameters

**output_flatten** : bool, default False

Flatten nested output structures after applying function.

- Flattens: Lists, tuples (if `output_flatten_tuple_set=True`), sets (if enabled)
- Use case: Function returns nested lists, need flat result
- Example: `func` returns `[[1, 2], [3, 4]]` → flattened to `[1, 2, 3, 4]`

**output_dropna** : bool, default False

Remove `None` and undefined values from output after applying function.

- Filters: `None`, `undefined` sentinel from results
- Use case: Function may return None for some inputs, filter them out
- Combines with `output_flatten` for deep null removal

**output_unique** : bool, default False

Remove duplicate output elements after applying function.

- Deduplication: Set-based uniqueness (requires hashable results)
- Order: Preserves first occurrence order
- Use case: Function produces duplicate results, need unique set
- Requires: `output_flatten=True` or `output_dropna=True` (raises `ValueError` otherwise)

**output_flatten_tuple_set** : bool, default False

Include tuples and sets when flattening output structures.

- Default: Only flattens lists
- If `True`: Also flattens tuples and sets
- Use case: Normalize all output iterable types
- Requires: `output_flatten=True` to take effect

## Returns

**list[R]**

List of results from applying `func` to each input element, with optional processing applied.

- Type: Generic type `R` (function return type)
- Length: Depends on input size and processing options (may shrink with `dropna`/`unique`, grow with `flatten`)
- Order: Preserves input order (unless flattening changes structure)
- Early termination: Returns partial results if `InterruptedError` raised

## Raises

### ValueError

Raised when:

- `func` is not callable and not an iterable containing exactly one callable
- `output_unique=True` without `output_flatten=True` or `output_dropna=True`

### TypeError

Raised when:

- `func` cannot be processed (not callable, not iterable, or iterable conversion fails)
- Input processing fails (e.g., non-iterable input with processing flags)
- Function execution fails with type errors

### Exception

Propagates any exception raised by `func` during execution (except `InterruptedError`).

**InterruptedError** (caught)

If `func` raises `InterruptedError`, returns partial results collected so far instead of propagating.

## Usage Patterns

### Basic Mapping

```python
from lionherd_core.ln import lcall

# Simple transformation
result = lcall([1, 2, 3], lambda x: x * 2)
# [2, 4, 6]

# With additional arguments
result = lcall([1, 2, 3], pow, 2)  # pow(x, 2)
# [1, 4, 9]

# With keyword arguments
def format_item(x, prefix="", suffix=""):
    return f"{prefix}{x}{suffix}"

result = lcall(["a", "b", "c"], format_item, prefix="[", suffix="]")
# ['[a]', '[b]', '[c]']
```

### Input Processing

```python
from lionherd_core.ln import lcall

# Flatten nested input
nested = [[1, 2], [3, [4, 5]]]
result = lcall(nested, str, input_flatten=True)
# ['1', '2', '3', '4', '5']

# Remove None values
with_nulls = [1, None, 2, None, 3]
result = lcall(with_nulls, lambda x: x * 10, input_dropna=True)
# [10, 20, 30]

# Deduplicate inputs
duplicates = [1, 2, 2, 3, 3, 3]
result = lcall(
    duplicates,
    lambda x: x ** 2,
    input_flatten=True,  # Required for unique
    input_unique=True
)
# [1, 4, 9]  (processed only unique values)

# Extract enum values
from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

statuses = [Status.ACTIVE, Status.INACTIVE]
result = lcall(
    statuses,
    str.upper,
    input_use_values=True
)
# ['ACTIVE', 'INACTIVE']  (extracted .value before processing)
```

### Output Processing

```python
from lionherd_core.ln import lcall

# Flatten nested output
def split_and_reverse(s):
    return [c for c in reversed(s)]

result = lcall(
    ["ab", "cd"],
    split_and_reverse,
    output_flatten=True
)
# ['b', 'a', 'd', 'c']  (flattened [['b', 'a'], ['d', 'c']])

# Remove None from output
def safe_divide(x):
    return 10 / x if x != 0 else None

result = lcall(
    [5, 0, 2, 0],
    safe_divide,
    output_dropna=True
)
# [2.0, 5.0]  (None values removed)

# Deduplicate output
def get_first_char(s):
    return s[0] if s else None

result = lcall(
    ["apple", "apricot", "banana", "berry"],
    get_first_char,
    output_dropna=True,
    output_unique=True
)
# ['a', 'b']  (duplicates removed)
```

### Combined Processing

```python
from lionherd_core.ln import lcall

# Complex pipeline: flatten input, process, flatten output
nested_data = [
    [1, 2, None],
    [3, None, 4],
    [[5, 6], 7]
]

def process(x):
    if x < 3:
        return [x, x * 10]  # Nested output
    return x

result = lcall(
    nested_data,
    process,
    input_flatten=True,      # Flatten nested input
    input_dropna=True,       # Remove None from input
    output_flatten=True,     # Flatten nested output
    output_unique=True       # Deduplicate results
)
# [1, 10, 2, 20, 3, 4, 5, 6, 7]
```

### Callable Extraction

```python
from lionherd_core.ln import lcall

# Function wrapped in iterable (auto-extracted)
def square(x):
    return x ** 2

result = lcall([1, 2, 3], [square])  # Note: func is [square]
# [1, 4, 9]

# Raises ValueError if iterable contains ≠1 callable
try:
    lcall([1, 2], [square, lambda x: x * 2])  # Multiple callables
except ValueError as e:
    print(e)  # "func must contain exactly one callable function."
```

### Early Termination

```python
from lionherd_core.ln import lcall

def process_with_interrupt(x):
    if x > 5:
        raise InterruptedError("Stop processing")
    return x * 2

# Returns partial results on interrupt
result = lcall([1, 2, 3, 6, 7], process_with_interrupt)
# [2, 4, 6]  (stopped at x=6, returned results so far)
```

## Design Rationale

### Why Separate Input/Output Processing?

Different processing requirements exist at different pipeline stages:

1. **Input normalization**: Flatten heterogeneous data structures before processing
2. **Output transformation**: Handle nested results from function execution
3. **Independent control**: Some use cases need input flattening but not output (or vice versa)

Separate `input_*` and `output_*` flags provide fine-grained control without complex flag interactions.

### Why Require `flatten` or `dropna` for `unique`?

The `unique` operation requires `flatten` or `dropna` because:

1. **Efficiency**: Uniqueness checks on raw nested structures are expensive and ambiguous
2. **Semantics**: "Unique nested lists" is ill-defined - should `[[1, 2]]` and `[[2, 1]]` be considered equal?
3. **Consistency**: `to_list` enforces this constraint, `lcall` inherits it for consistency

Users must explicitly choose whether to flatten before deduplication or remove nulls first.

### Why InterruptedError Special Handling?

`InterruptedError` is treated specially (returns partial results instead of propagating) because:

1. **Graceful degradation**: Allows controlled early termination without losing completed work
2. **User signals**: Applications can signal "stop processing" without exception handling overhead
3. **Partial results utility**: In many cases, partial results are valuable even if processing didn't complete

All other exceptions propagate normally for standard error handling.

### Why Support Single Callable in Iterable?

Supporting `func` as `Iterable[Callable]` (with exactly one element) enables:

1. **Higher-order composition**: Functions that wrap callables can pass them through consistently
2. **API uniformity**: Callers don't need special-case logic for "list of one function"
3. **Validation layer**: Explicit error if multiple callables passed (catches mistakes early)

This pattern appears in functional programming contexts where callables are passed through layers.

### Why Use `to_list` for Processing?

Delegating to `to_list` for input/output processing ensures:

1. **Single source of truth**: Flattening/filtering logic defined once, tested thoroughly
2. **Consistency**: Same behavior across `to_list`, `lcall`, `alcall`, and future utilities
3. **Maintainability**: Bug fixes and improvements automatically benefit all consumers
4. **Type safety**: Shared validation logic prevents inconsistent parameter combinations

See [to_list](to_list.md) for detailed processing semantics.

## See Also

- **Related Utilities**:
  - [to_list](to_list.md): Core list conversion/processing utility (underlying implementation)
  <!-- TODO: Add links to alcall and rcall when available (issue #115) -->
- **Related Patterns**:
  - Built-in `map()`: Simple synchronous mapping without processing
  - List comprehensions: Inline transformations with filtering
  - `itertools.chain.from_iterable()`: Output flattening alternative

## Examples

```python
# Standard imports for ln.list_call examples
from lionherd_core.ln import lcall
```

### Example 1: Data Normalization Pipeline

```python
from enum import Enum

class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

# Nested data with enums and nulls
tasks = [
    [Priority.HIGH, Priority.MEDIUM],
    [None, Priority.LOW],
    Priority.HIGH
]

# Normalize to priority values
priorities = lcall(
    tasks,
    lambda p: f"priority_{p}",
    input_flatten=True,
    input_dropna=True,
    input_use_values=True,
    input_unique=True
)
# ['priority_1', 'priority_2', 'priority_3']
```

### Example 2: Text Processing with Flattening

```python
from lionherd_core.ln import lcall

# Split sentences into words, flatten, deduplicate
sentences = [
    "the quick brown fox",
    "the lazy dog",
    "quick brown dog"
]

unique_words = lcall(
    sentences,
    str.split,
    output_flatten=True,
    output_unique=True
)
# ['the', 'quick', 'brown', 'fox', 'lazy', 'dog']
# (order: first occurrence preserved)
```

### Example 3: Conditional Transformation

```python
from lionherd_core.ln import lcall

def transform(x):
    if x % 2 == 0:
        return [x, x // 2]  # Even: return [value, half]
    return None  # Odd: return None

result = lcall(
    range(1, 7),
    transform,
    output_flatten=True,
    output_dropna=True
)
# [2, 1, 4, 2, 6, 3]
# Input: [1, 2, 3, 4, 5, 6]
# After transform: [None, [2, 1], None, [4, 2], None, [6, 3]]
# After dropna: [[2, 1], [4, 2], [6, 3]]
# After flatten: [2, 1, 4, 2, 6, 3]
```

### Example 4: API Response Processing

```python
from lionherd_core.ln import lcall

# Simulate API response data
api_responses = [
    {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]},
    {"users": [{"id": 3, "name": "Charlie"}]},
    {"users": None}  # Failed response
]

def extract_names(response):
    """Extract user names from response."""
    users = response.get("users")
    if users is None:
        return None  # Filtered by output_dropna
    return [u["name"] for u in users]

all_names = lcall(
    api_responses,
    extract_names,
    output_flatten=True,    # Flatten nested [[...], [...], ...]
    output_dropna=True      # Remove None from failed responses
)
# ['Alice', 'Bob', 'Charlie']
```

### Example 5: Type Conversion with Validation

```python
from lionherd_core.ln import lcall

def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

# Convert mixed types to integers
mixed_data = ["123", "456", "invalid", None, 789, "0"]

integers = lcall(
    mixed_data,
    safe_int,
    input_dropna=True,  # Remove None from input
    output_dropna=True  # Remove None from failed conversions
)
# [123, 456, 789, 0]
```

### Example 6: Batch Processing with Shared Config

```python
from lionherd_core.ln import lcall

def process_record(record, db_connection, batch_size):
    # Simulate batch processing with shared config
    return {
        "id": record["id"],
        "processed": True,
        "batch": batch_size
    }

records = [
    {"id": 1, "data": "..."},
    {"id": 2, "data": "..."},
    {"id": 3, "data": "..."}
]

# Simulate db_connection object
db_conn = "connection_string"

results = lcall(
    records,
    process_record,
    db_conn,  # Passed as second argument to process_record
    batch_size=100  # Passed as keyword argument
)
# [
#   {'id': 1, 'processed': True, 'batch': 100},
#   {'id': 2, 'processed': True, 'batch': 100},
#   {'id': 3, 'processed': True, 'batch': 100}
# ]
```

### Example 7: Multi-Stage Pipeline

```python
from lionherd_core.ln import lcall

# Stage 1: Parse and normalize
raw_data = [
    "1,2,3",
    "4,,5",  # Empty value
    "6,7",
    None
]

def parse_csv(line):
    if line is None:
        return None
    return [int(x) if x else None for x in line.split(",")]

# Stage 1: Parse
parsed = lcall(
    raw_data,
    parse_csv,
    input_dropna=True,
    output_flatten=True
)
# [1, 2, 3, 4, None, 5, 6, 7]

# Stage 2: Transform (using parsed as input)
def double(x):
    return x * 2

final = lcall(
    parsed,
    double,
    input_dropna=True,
    input_unique=True
)
# [2, 4, 6, 8, 10, 12, 14]
```

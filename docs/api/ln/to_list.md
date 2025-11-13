# to_list

> Universal list conversion with flattening, deduplication, and value extraction

## Overview

`to_list()` is a utility function that converts any input to a list with optional transformations including recursive flattening, null removal, deduplication, and value extraction from enums and mappings. It handles edge cases like nested iterables, Pydantic models, enums, and unhashable types with intelligent fallback strategies.

**Key Capabilities:**

- **Universal Conversion**: Handles any Python type (primitives, iterables, enums, models, mappings)
- **Recursive Flattening**: Optional nested iterable flattening with configurable tuple/set handling
- **Null Removal**: Filters out `None` and sentinel values (`Undefined`, `Unset`, `PydanticUndefined`)
- **Deduplication**: Removes duplicates with automatic fallback for unhashable types
- **Value Extraction**: Extracts values from enums and mappings instead of objects
- **Smart Type Handling**: Preserves strings/bytes as single items (not character lists)

**When to Use to_list:**

- Normalizing heterogeneous inputs (single values, lists, tuples, sets)
- Flattening nested data structures from APIs or LLM outputs
- Cleaning data by removing nulls and duplicates
- Extracting enum values or mapping values for processing
- Building robust input pipelines that handle multiple input formats

**When NOT to Use to_list:**

- Direct list construction where input format is guaranteed (use `list()` or `[...]`)
- Performance-critical paths with simple inputs (adds overhead for type checking)
- When you need to preserve tuple/set types (flattening converts to list)
- When byte-like objects should be split into characters (to_list preserves them)

## Function Signature

```python
from lionherd_core.ln import to_list

def to_list(
    input_: Any,
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    use_values: bool = False,
    flatten_tuple_set: bool = False,
) -> list: ...
```

## Parameters

**input_** : Any (positional-only)

Value to convert to list. Accepts any Python type.

- **None/sentinel values**: Converted to empty list `[]`
- **Lists**: Returned as-is (no copy)
- **Enum classes**: Converted to list of enum members (or values if `use_values=True`)
- **Strings/bytes/bytearray**: Treated as single item `[input_]` unless `use_values=True`
- **Mappings**: Treated as single item `[input_]` unless `use_values=True` (then `list(input_.values())`)
- **Pydantic BaseModel**: Treated as single item `[input_]`
- **Other Iterables**: Converted via `list(input_)`
- **Non-iterables**: Wrapped as `[input_]`

**flatten** : bool, default False

Recursively flatten nested iterables.

- `False`: Nested iterables remain as nested lists
- `True`: Recursively flattens all nested iterables (except skip types)
- Skip types (not flattened): strings, bytes, bytearray, mappings, Pydantic models, enums
- Use `flatten_tuple_set=True` to also flatten tuples and sets

**dropna** : bool, default False

Remove `None` and undefined sentinel values.

- Filters out: `None`, `UndefinedType`, `UnsetType`, `PydanticUndefinedType`
- Applied during flattening/processing (not just top-level)

**unique** : bool, default False

Remove duplicate items.

- **Requires**: `flatten=True` (raises `ValueError` otherwise)
- **Strategy**: Tries direct hash comparison first, falls back to `hash_dict()` for unhashable mappings
- **Ordering**: Preserves first occurrence order
- **Raises**: `ValueError` if item is unhashable and not a mapping

**use_values** : bool, default False

Extract values from enums and mappings instead of treating them as objects.

- **Enum classes**: Returns `[member.value for member in EnumClass]` instead of `[member, ...]`
- **Strings/bytes**: Returns `list(input_)` (character/byte list) instead of `[input_]`
- **Mappings**: Returns `list(mapping.values())` instead of `[mapping]`

**flatten_tuple_set** : bool, default False

Include tuples and sets in flattening (normally they're preserved as items).

- `False`: Tuples and sets are treated as atomic items (not flattened)
- `True`: Tuples and sets are flattened like lists
- Only relevant when `flatten=True`

## Returns

**list** : Processed list

Transformed list according to specified parameters. The list contains processed items, potentially flattened, deduplicated, and/or filtered.

## Raises

### ValueError

If `unique=True` but `flatten=False`:

```python
to_list([1, 2], unique=True, flatten=False)
# ValueError: unique=True requires flatten=True
```

If unhashable non-mapping type encountered during deduplication:

```python
to_list([[1], [1]], flatten=True, unique=True)
# ValueError: Unhashable type encountered in list unique value processing.
```

## Usage Patterns

### Basic Conversion

```python
from lionherd_core.ln import to_list

# Primitives wrapped in list
to_list(42)           # [42]
to_list("hello")      # ["hello"]
to_list(None)         # []

# Iterables converted
to_list((1, 2, 3))    # [1, 2, 3]
to_list({1, 2, 3})    # [1, 2, 3]
to_list([1, 2, 3])    # [1, 2, 3]

# Nested iterables (no flattening by default)
to_list([[1, 2], [3, 4]])  # [[1, 2], [3, 4]]
```

### Flattening Nested Structures

```python
from lionherd_core.ln import to_list

# Recursive flattening
to_list([[1, 2], [3, [4, 5]]], flatten=True)
# [1, 2, 3, 4, 5]

# Tuples preserved by default (not flattened)
to_list([1, (2, 3), [4, 5]], flatten=True)
# [1, (2, 3), 4, 5]

# Flatten tuples too
to_list([1, (2, 3), [4, 5]], flatten=True, flatten_tuple_set=True)
# [1, 2, 3, 4, 5]

# Strings NOT flattened (treated as atomic)
to_list(["hello", ["world"]], flatten=True)
# ["hello", "world"]  (NOT ["h", "e", "l", "l", "o", "world"])
```

### Null Removal

```python
from lionherd_core.ln import to_list
from lionherd_core.types import Undefined, Unset

# Remove None and sentinel values
to_list([1, None, 2, None, 3], dropna=True)
# [1, 2, 3]

# Works with nested structures (during flattening)
to_list([[1, None], [2, None, 3]], flatten=True, dropna=True)
# [1, 2, 3]

# Filters Undefined/Unset sentinels
to_list([1, Undefined, 2, Unset, 3], dropna=True)
# [1, 2, 3]
```

### Deduplication

```python
from lionherd_core.ln import to_list

# Remove duplicates (requires flatten=True)
to_list([[1, 2], [2, 3]], flatten=True, unique=True)
# [1, 2, 3]

# Preserves first occurrence order
to_list([[3, 1], [1, 2, 3]], flatten=True, unique=True)
# [3, 1, 2]

# Handles unhashable mappings via hash_dict()
to_list([{"a": 1}, {"a": 1}, {"b": 2}], flatten=True, unique=True)
# [{"a": 1}, {"b": 2}]

# Error if unique without flatten
try:
    to_list([1, 2, 2], unique=True, flatten=False)
except ValueError as e:
    print(e)  # "unique=True requires flatten=True"
```

### Value Extraction

```python
from lionherd_core.ln import to_list
from enum import Enum

# Enum class → list of members (default)
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

to_list(Color)
# [<Color.RED: 1>, <Color.GREEN: 2>, <Color.BLUE: 3>]

# Enum class → list of values
to_list(Color, use_values=True)
# [1, 2, 3]

# Mapping → single item (default)
to_list({"a": 1, "b": 2})
# [{"a": 1, "b": 2}]

# Mapping → values list
to_list({"a": 1, "b": 2}, use_values=True)
# [1, 2]

# String → single item (default)
to_list("hello")
# ["hello"]

# String → character list
to_list("hello", use_values=True)
# ["h", "e", "l", "l", "o"]
```

### Combined Operations

```python
from lionherd_core.ln import to_list

# Flatten, remove nulls, deduplicate
data = [[1, None, 2], [2, None, 3], [3, 4]]
result = to_list(data, flatten=True, dropna=True, unique=True)
# [1, 2, 3, 4]

# Normalize heterogeneous inputs
inputs = [
    42,                    # single value
    [1, 2],               # list
    (3, 4),               # tuple
    {5, 6},               # set
    None,                 # null
]
to_list(inputs, flatten=True, dropna=True)
# [42, 1, 2, 3, 4, 5, 6]
```

### Pydantic Model Handling

```python
from lionherd_core.ln import to_list
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=30)

# Model treated as single item (not flattened)
to_list([user, "other"])
# [User(name='Alice', age=30), "other"]

# Even with flatten=True (models are atomic)
to_list([[user], ["other"]], flatten=True)
# [User(name='Alice', age=30), "other"]
```

## ToListParams Class

Frozen dataclass for storing `to_list()` parameters and reusing configurations.

### Class Signature

```python
from lionherd_core.ln import ToListParams

@dataclass(slots=True, frozen=True, init=False)
class ToListParams(Params):
    """Parameter container for to_list() function."""

    flatten: bool
    dropna: bool
    unique: bool
    use_values: bool
    flatten_tuple_set: bool

    def __call__(self, input_: Any, **kw) -> list: ...
```

### Attributes

| Attribute           | Type   | Description                                         |
| ------------------- | ------ | --------------------------------------------------- |
| `flatten`           | `bool` | If True, recursively flatten nested iterables       |
| `dropna`            | `bool` | If True, remove None and undefined values           |
| `unique`            | `bool` | If True, remove duplicates (requires flatten=True)  |
| `use_values`        | `bool` | If True, extract values from enums/mappings         |
| `flatten_tuple_set` | `bool` | If True, include tuples and sets in flattening      |

### Methods

#### `__call__()`

Apply `to_list()` with stored parameters, optionally overriding via kwargs.

**Signature:**

```python
def __call__(self, input_: Any, **kw) -> list: ...
```

**Parameters:**

- `input_` (Any): Value to convert
- `**kw` (Any): Parameter overrides (merged with stored parameters)

**Returns:**

- list: Result of `to_list(input_, **merged_params)`

**Examples:**

```python
from lionherd_core.ln import ToListParams

# Create reusable configuration
params = ToListParams(flatten=True, dropna=True, unique=True)

# Apply to multiple inputs
result1 = params([[1, None, 2], [2, 3]])
# [1, 2, 3]

result2 = params([[4, 5], [5, 6]])
# [4, 5, 6]

# Override parameters for specific call
result3 = params([[1, 1, 2]], unique=False)
# [1, 1, 2]  (unique overridden to False)
```

## Design Rationale

### Why Positional-Only input_?

The `/` marker enforces positional-only for `input_`, preventing ambiguity:

```python
# Clear intent (positional)
to_list(my_data, flatten=True)

# Would be confusing if allowed
# to_list(input_=my_data, flatten=True)  # Verbose and unnatural
```

This follows Python's convention for functions where the first parameter is always the "thing being operated on."

### Why flatten=True Required for unique?

Deduplication without flattening leads to ambiguous semantics:

```python
# What should this return?
to_list([[1, 2], [1, 2]], unique=True, flatten=False)
# [1, 2]?  Or [[1, 2]]?  Or error?
```

Requiring `flatten=True` makes behavior unambiguous: flatten first, then deduplicate the flattened list.

### Why Skip Strings/Bytes in Flattening?

Strings and bytes are technically iterables, but treating them as character/byte sequences is almost never desired:

```python
# Unwanted behavior if strings were flattened
to_list([["hello", "world"]], flatten=True)
# Expected: ["hello", "world"]
# If strings flattened: ["h", "e", "l", "l", "o", "w", "o", "r", "l", "d"]
```

`use_values=True` provides opt-in character splitting for the rare cases where it's needed.

### Why Separate flatten_tuple_set Flag?

Tuples and sets often represent semantic groupings that should remain atomic:

```python
# Coordinates as tuples (should stay grouped)
points = [(0, 0), (1, 1), (2, 2)]
to_list([points], flatten=True)
# Expected: [(0, 0), (1, 1), (2, 2)]  (preserve tuples)

# Explicit flattening when needed
to_list([points], flatten=True, flatten_tuple_set=True)
# [0, 0, 1, 1, 2, 2]  (flatten tuples too)
```

Separate flag gives fine-grained control over flattening behavior.

### Why Hash Fallback for Deduplication?

Python's set-based deduplication fails for unhashable types (dicts, lists). The fallback strategy:

1. **Tries direct hashing first** (fast path for hashable items)
2. **Switches to hash_dict() on first TypeError** (handles mappings)
3. **Raises ValueError for unhashable non-mappings** (lists, sets in unique context)

This provides maximum performance for common cases while gracefully handling mappings.

### Why Lazy Initialization for Global Types?

```python
# noqa:validation
# Initialized on first call, not at import
_INITIALIZED = False

def to_list(...):
    global _INITIALIZED
    if _INITIALIZED is False:
        from pydantic import BaseModel  # Lazy import
        # ... setup globals
        _INITIALIZED = True
```

Avoids circular import issues and reduces startup overhead for modules that import but don't immediately use `to_list()`.

## See Also

- **Related Functions**:
  - [to_dict()](to_dict.md): Dictionary conversion utilities
  - [hash_dict()](hash.md): Dictionary hashing for unhashable deduplication
- **Related Classes**:
  - [Params](../types/base.md): Base class for parameter containers
  - [HashableModel](../types/model.md): Content-based hashing for deduplication

## Examples

```python
# Standard imports for ln.to_list examples
from lionherd_core.ln import to_list, ToListParams
from lionherd_core.types import Undefined, Unset
```

### Example 1: Normalizing LLM Outputs

```python
# LLM might return string, list, or nested structure
def normalize_tags(llm_output):
    """Extract clean tag list from various LLM output formats."""
    return to_list(
        llm_output,
        flatten=True,
        dropna=True,
        unique=True,
    )

# String output
normalize_tags("python")
# ["python"]

# List output
normalize_tags(["python", "ai", "python"])
# ["python", "ai"]

# Nested output
normalize_tags([["python", None], ["ai", "ml"]])
# ["python", "ai", "ml"]

# None/empty output
normalize_tags(None)
# []
```

### Example 2: Enum Value Extraction

```python
from lionherd_core.ln import to_list
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"

# Extract all status values for validation
valid_statuses = to_list(Status, use_values=True)
# ["pending", "active", "completed"]

def validate_status(status: str) -> bool:
    return status in valid_statuses

validate_status("active")    # True
validate_status("invalid")   # False
```

### Example 3: Data Pipeline Cleaning

```python
from lionherd_core.ln import to_list
from lionherd_core.types import Undefined

# Raw data from multiple sources
raw_data = [
    [1, 2, Undefined],      # Source 1 (has undefined)
    3,                      # Source 2 (single value)
    [4, None, 5],          # Source 3 (has null)
    [[6, 7]],              # Source 4 (nested)
    {"values": [8, 9]},    # Source 5 (mapping)
]

# Clean pipeline
clean = to_list(
    [to_list(item, use_values=True) for item in raw_data],
    flatten=True,
    dropna=True,
    unique=True,
)
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Example 4: Reusable Configuration

```python
from lionherd_core.ln import ToListParams

# Define standard cleaning pipeline
clean_flatten = ToListParams(
    flatten=True,
    dropna=True,
    unique=True,
    use_values=False,
    flatten_tuple_set=False,
)

# Apply to multiple datasets
dataset1 = [[1, None, 2], [2, 3]]
dataset2 = [["a", None], ["b", "a"]]

result1 = clean_flatten(dataset1)
# [1, 2, 3]

result2 = clean_flatten(dataset2)
# ["a", "b"]

# Override for specific case
result3 = clean_flatten([[1, 1, 2]], unique=False)
# [1, 1, 2]
```

### Example 5: Handling Mixed Types

```python
from lionherd_core.ln import to_list
from pydantic import BaseModel

class Config(BaseModel):
    enabled: bool

# Mixed type input
mixed = [
    42,                           # int
    "text",                       # str
    [1, 2],                      # list
    (3, 4),                      # tuple
    {5, 6},                      # set
    Config(enabled=True),        # model
    {"key": "value"},            # dict
]

# Flatten (preserves models and dicts as atomic)
result = to_list(mixed, flatten=True)
# [42, "text", 1, 2, (3, 4), 5, 6, Config(enabled=True), {"key": "value"}]

# Flatten with tuple/set flattening
result_full = to_list(mixed, flatten=True, flatten_tuple_set=True)
# [42, "text", 1, 2, 3, 4, 5, 6, Config(enabled=True), {"key": "value"}]
```

### Example 6: Deduplication Edge Cases

```python
from lionherd_core.ln import to_list

# Hashable types (direct deduplication)
to_list([[1, 2], [2, 3]], flatten=True, unique=True)
# [1, 2, 3]

# Unhashable mappings (hash_dict fallback)
data = [
    {"name": "Alice", "age": 30},
    {"name": "Alice", "age": 30},  # Duplicate
    {"name": "Bob", "age": 25},
]
to_list(data, flatten=True, unique=True)
# [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]

# Unhashable non-mappings (error)
try:
    to_list([[[1]], [[1]]], flatten=True, unique=True)
except ValueError as e:
    print(e)  # "Unhashable type encountered in list unique value processing."
```

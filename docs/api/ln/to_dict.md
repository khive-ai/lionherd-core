# to_dict

> Universal dictionary conversion utilities with recursive processing and polymorphic type handling

## Overview

The `to_dict` module provides universal conversion of Python objects to dictionaries with support for JSON parsing, recursive processing, and automatic handling of custom types including Pydantic models, dataclasses, and iterables.

**Key Capabilities:**

- **Universal Conversion**: Handles Pydantic models, dataclasses, enums, mappings, sequences, sets, and custom objects
- **String Parsing**: Automatic JSON parsing of string inputs (orjson-based with fuzzy fallback)
- **Recursive Processing**: Deep conversion of nested structures with configurable depth limits
- **Custom Type Handling**: Automatic detection and conversion via `model_dump()`, `to_dict()`, `dict()`, `to_json()`, or `__dict__`
- **Enum Support**: Flexible enum conversion (members or values)
- **Iterable Enumeration**: Sequences and iterables converted to indexed dictionaries

**When to Use to_dict:**

- Converting API responses to dictionaries for processing
- Normalizing heterogeneous data structures to uniform dict format
- Parsing JSON strings embedded in data structures
- Preparing custom objects for serialization
- Deep conversion of nested Pydantic models and dataclasses

**When NOT to Use:**

- Simple `dict(obj)` suffices for basic mappings
- Object already provides suitable `to_dict()` method (call directly)
- Performance-critical paths with simple data (overhead unnecessary)

## Module Contents

### Public API

#### `to_dict()`

Convert various input types to dictionary with optional recursive processing.

**Function Signature:**

```python
from lionherd_core.ln import to_dict

def to_dict(
    input_: Any,
    /,
    *,
    prioritize_model_dump: bool = False,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    parser: Callable[[str], Any] | None = None,
    recursive: bool = False,
    max_recursive_depth: int | None = None,
    recursive_python_only: bool = True,
    use_enum_values: bool = False,
    **kwargs: Any,
) -> dict[str | int, Any]: ...
```

### Parameters

**input_** : Any (positional-only)

Object to convert to dictionary. Supports:

- **Mappings**: Copied to plain dict
- **Pydantic models**: Converted via `model_dump()` or `to_dict()`
- **Dataclasses**: Converted via `dataclasses.asdict()`
- **Sequences** (list, tuple): Enumerated to `{index: value}`
- **Sets**: Converted to `{value: value}`
- **Enums** (class): Members mapping `{name: member}` or `{name: value}`
- **None/Undefined**: Returns empty dict `{}`
- **Strings**: Parsed as JSON (orjson or custom parser)
- **Custom objects**: Converted via `to_dict()`, `dict()`, `to_json()`, `__dict__`, or `dict(obj)`

**prioritize_model_dump** : bool, default False

If True, prioritize `.model_dump()` for Pydantic v2 models over other conversion methods.

- `False`: Checks methods in order: `to_dict`, `dict`, `to_json`, `json`, `model_dump`
- `True`: Checks `model_dump` first, then falls back to other methods
- Only affects objects with multiple conversion methods

**fuzzy_parse** : bool, default False

Enable fuzzy JSON parsing for malformed string inputs.

- `False`: Strict JSON parsing via `orjson.loads()`
- `True`: Fallback to `fuzzy_json()` for strings with syntax errors, trailing commas, etc.
- Only applies to string inputs

**suppress** : bool, default False

Suppress exceptions and return empty dict on conversion failures.

- `False`: Raises exceptions on conversion errors
- `True`: Returns `{}` on any exception or empty string input
- Use for fault-tolerant pipelines where partial data is acceptable

**parser** : Callable[[str], Any] or None, default None

Custom parser function for string inputs. Overrides default JSON parsing.

- Signature: `parser(s: str, **kwargs) -> Any`
- Receives `**kwargs` from `to_dict()` call
- If provided, `fuzzy_parse` is ignored

**recursive** : bool, default False

Enable recursive processing of nested structures.

- `False`: Single-level conversion only
- `True`: Recursively processes nested dicts, lists, tuples, sets, and custom objects
- Respects `max_recursive_depth` limit

**max_recursive_depth** : int or None, default None

Maximum recursion depth for nested structure processing.

- `None`: Defaults to depth of 5
- Valid range: 0-10 (raises `ValueError` outside this range)
- Prevents infinite recursion on circular references
- Only applies when `recursive=True`

**recursive_python_only** : bool, default True

Restrict recursive processing to Python built-in types only.

- `True`: Recurse into dicts, lists, tuples, sets only (custom objects converted at top level)
- `False`: Also recursively convert nested custom objects via `model_dump()`, `to_dict()`, etc.
- Only applies when `recursive=True`

**use_enum_values** : bool, default False

Use enum values instead of enum members when converting enum classes.

- `False`: `{name: EnumMember}` (preserves enum type)
- `True`: `{name: member.value}` (extracts raw value)
- Applies to enum **classes** (rare in values), not enum instances

**\*\*kwargs** : Any

Additional keyword arguments passed to:

- Custom `parser` function
- Pydantic `model_dump()` calls (e.g., `include`, `exclude`, `by_alias`)
- JSON parsing functions

### Returns

**dict[str | int, Any]**

Dictionary representation of input:

- **String keys**: For mappings, objects with attributes, enums
- **Integer keys**: For sequences, iterables (enumerated by index)
- **Empty dict**: If `suppress=True` and conversion fails, or if `input_` is None/undefined

### Raises

### ValueError

- `max_recursive_depth < 0` or `max_recursive_depth > 10`

### Exception

- Any conversion error if `suppress=False` (type depends on input)
- JSON parsing errors (orjson, custom parser)
- Type coercion failures

**Notes:**

Empty string inputs (`input_=""`) always return `{}` regardless of `suppress` setting.

## Conversion Logic

### Top-Level Conversion Order

Non-recursive mode processes input in this priority:

1. **Set** → `{value: value}` for all members
2. **Enum class** → `{name: member}` or `{name: value}` (if `use_enum_values=True`)
3. **Mapping** → `dict(mapping)` (shallow copy)
4. **None/Undefined** → `{}` (Pydantic `Undefined`, `PydanticUndefined`, etc.)
5. **String** → Parse as JSON via `parser` or `orjson`/`fuzzy_json`
6. **Custom object** (non-Sequence) → Try conversion methods:
   - `.model_dump()` (if `prioritize_model_dump=True`)
   - `.to_dict()`, `.dict()`, `.to_json()`, `.json()`, `.model_dump()` (in order)
   - `dataclasses.asdict(obj)` if dataclass
   - `obj.__dict__` if available
   - `dict(obj)` as last resort
7. **Iterable** (list, tuple, namedtuple, frozenset, etc.) → `{index: value}`
8. **Dataclass** (fallback) → `dataclasses.asdict(obj)`
9. **Last resort** → `dict(obj)` (may raise if not convertible)

### Recursive Processing

When `recursive=True`, nested structures are processed depth-first:

1. **Strings** → Attempt JSON parsing; on success, recurse into parsed result
2. **Mappings** → Recurse into values (keys preserved as-is)
3. **Sequences** (list, tuple, set, frozenset) → Recurse into items, preserve type
4. **Enum classes** → Convert to dict, recurse into result
5. **Custom objects** (if `recursive_python_only=False`) → Convert to mapping, recurse

Container types are preserved during recursion (list stays list, set stays set, etc.). Only at the **top level** does final conversion to dict occur.

### String Parsing

String inputs are parsed as JSON:

```python
# Strict parsing (default)
to_dict('{"key": "value"}')  # {'key': 'value'}

# Fuzzy parsing (tolerates syntax errors)
to_dict('{"key": "value",}', fuzzy_parse=True)  # {'key': 'value'} (trailing comma allowed)

# Custom parser
to_dict('<xml/>', parser=xml_to_dict)  # Custom XML parsing
```

### Custom Object Conversion

For objects with multiple conversion methods:

```python
class MyModel:
    def to_dict(self): return {"method": "to_dict"}
    def dict(self): return {"method": "dict"}
    def model_dump(self): return {"method": "model_dump"}

# Default priority: to_dict → dict → to_json → json → model_dump
to_dict(MyModel())  # {'method': 'to_dict'}

# Prioritize model_dump for Pydantic v2
to_dict(MyModel(), prioritize_model_dump=True)  # {'method': 'model_dump'}
```

## Internal Helpers

These functions are implementation details (not public API):

### `_is_na(obj)`

Check if object is None or Pydantic undefined sentinel.

- Avoids importing Pydantic types by checking `type(obj).__name__`
- Matches: `Undefined`, `UndefinedType`, `PydanticUndefined`, `PydanticUndefinedType`

### `_enum_class_to_dict(enum_cls, use_enum_values)`

Convert enum class to dictionary.

- `use_enum_values=False`: `{name: EnumMember}`
- `use_enum_values=True`: `{name: member.value}`

### `_parse_str(s, fuzzy_parse, parser, **kwargs)`

Parse string to Python object (JSON only).

- Uses custom `parser` if provided
- Falls back to `fuzzy_json()` if `fuzzy_parse=True`
- Default: `orjson.loads(s)`

### `_object_to_mapping_like(obj, prioritize_model_dump, **kwargs)`

Convert custom objects to mapping using available methods.

Priority order:

1. `model_dump()` (if `prioritize_model_dump=True`)
2. `to_dict()`, `dict()`, `to_json()`, `json()`, `model_dump()` (in order)
3. `dataclasses.asdict()` if dataclass
4. `obj.__dict__`
5. `dict(obj)`

### `_enumerate_iterable(it)`

Convert iterable to indexed dictionary: `{0: item0, 1: item1, ...}`

### `_preprocess_recursive(obj, depth, max_depth, recursive_custom_types, str_parse_opts, prioritize_model_dump)`

Recursively process nested structures.

- Depth-first traversal with max depth limit
- Preserves container types (dict → dict, list → list, etc.)
- Optionally recurses into custom objects

### `_convert_top_level_to_dict(obj, fuzzy_parse, parser, prioritize_model_dump, use_enum_values, **kwargs)`

Single-level conversion to dict using priority rules.

## Usage Patterns

### Basic Conversion

```python
from lionherd_core.ln import to_dict

# Mapping
to_dict({'a': 1, 'b': 2})  # {'a': 1, 'b': 2}

# List (enumerated)
to_dict([10, 20, 30])  # {0: 10, 1: 20, 2: 30}

# Set
to_dict({'x', 'y', 'z'})  # {'x': 'x', 'y': 'y', 'z': 'z'}

# None
to_dict(None)  # {}
```

### Pydantic Models

```python
from lionherd_core import BaseModel
from lionherd_core.ln import to_dict

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=30)

# Default conversion
to_dict(user)  # {'name': 'Alice', 'age': 30}

# With Pydantic kwargs
to_dict(user, exclude={'age'})  # {'name': 'Alice'}
to_dict(user, by_alias=True)  # Respects field aliases
```

### JSON String Parsing

```python
from lionherd_core.ln import to_dict

# Valid JSON
to_dict('{"key": "value"}')  # {'key': 'value'}

# Nested JSON string
to_dict('{"nested": "{\\"inner\\": 1}"}')  # {'nested': '{"inner": 1}'} (not recursive by default)

# Fuzzy parsing for malformed JSON
to_dict('{"key": "value",}', fuzzy_parse=True)  # {'key': 'value'} (trailing comma handled)
```

### Recursive Processing

```python
from lionherd_core.ln import to_dict
from lionherd_core import BaseModel

class Address(BaseModel):
    city: str
    zip: str

class Person(BaseModel):
    name: str
    address: Address

person = Person(name="Bob", address=Address(city="NYC", zip="10001"))

# Non-recursive (default)
result = to_dict(person)
# {'name': 'Bob', 'address': Address(city='NYC', zip='10001')}
# (nested Address not converted)

# Recursive conversion
result = to_dict(person, recursive=True, recursive_python_only=False)
# {'name': 'Bob', 'address': {'city': 'NYC', 'zip': '10001'}}
# (nested models fully converted)
```

### Recursive String Parsing

```python
from lionherd_core.ln import to_dict

# Nested JSON strings
data = {
    'config': '{"setting": "value"}',
    'nested': '{"level2": "{\\"level3\\": 1}"}'
}

result = to_dict(data, recursive=True)
# {
#   'config': {'setting': 'value'},
#   'nested': {'level2': {'level3': 1}}
# }
# (all JSON strings recursively parsed)
```

### Dataclass Conversion

```python
from dataclasses import dataclass
from lionherd_core.ln import to_dict

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Line:
    start: Point
    end: Point

line = Line(start=Point(0, 0), end=Point(10, 10))

# Non-recursive
to_dict(line)
# {'start': Point(x=0, y=0), 'end': Point(x=10, y=10)}

# Recursive
to_dict(line, recursive=True, recursive_python_only=False)
# {'start': {'x': 0, 'y': 0}, 'end': {'x': 10, 'y': 10}}
```

### Enum Conversion

```python
from enum import Enum
from lionherd_core.ln import to_dict

class Status(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"

# Enum members (default)
to_dict(Status)
# {'PENDING': <Status.PENDING: 'pending'>, 'ACTIVE': <Status.ACTIVE: 'active'>, 'DONE': <Status.DONE: 'done'>}

# Enum values
to_dict(Status, use_enum_values=True)
# {'PENDING': 'pending', 'ACTIVE': 'active', 'DONE': 'done'}
```

### Error Handling

```python
from lionherd_core.ln import to_dict

# Strict mode (default) - raises on error
try:
    to_dict(object())  # Not convertible
except TypeError:
    print("Conversion failed")

# Suppressed mode - returns empty dict
to_dict(object(), suppress=True)  # {}
to_dict("", suppress=True)  # {} (empty string always returns {})
```

### Custom Parser

```python
from lionherd_core.ln import to_dict
import yaml

# YAML parser
def yaml_parser(s: str, **kwargs) -> dict:
    return yaml.safe_load(s)

yaml_str = """
name: Alice
age: 30
"""

to_dict(yaml_str, parser=yaml_parser)
# {'name': 'Alice', 'age': 30}
```

### Depth Control

```python
from lionherd_core.ln import to_dict

# Deeply nested structure
deep = {
    'l1': {
        'l2': {
            'l3': {
                'l4': {
                    'l5': '{"l6": "too deep"}'
                }
            }
        }
    }
}

# Limited depth (default 5)
result = to_dict(deep, recursive=True)
# Recursion stops at depth 5, 'l6' not parsed

# Custom depth
result = to_dict(deep, recursive=True, max_recursive_depth=10)
# Allows deeper recursion, 'l6' gets parsed

# Depth validation
to_dict(data, max_recursive_depth=-1)  # ValueError: must be non-negative
to_dict(data, max_recursive_depth=11)  # ValueError: must be <= 10
```

## Common Pitfalls

### Pitfall 1: Assuming Recursive by Default

**Issue**: Nested objects not converted automatically.

```python
from lionherd_core import BaseModel
from lionherd_core.ln import to_dict

class Inner(BaseModel):
    value: int

class Outer(BaseModel):
    inner: Inner

obj = Outer(inner=Inner(value=42))

# Non-recursive (default)
result = to_dict(obj)
# {'inner': Inner(value=42)}  # Inner not converted!

# Fix: Enable recursion
result = to_dict(obj, recursive=True, recursive_python_only=False)
# {'inner': {'value': 42}}  # Fully converted
```

**Solution**: Set `recursive=True` and `recursive_python_only=False` for deep custom object conversion.

### Pitfall 2: Recursive Python Only Flag

**Issue**: Custom objects not recursed with default `recursive_python_only=True`.

```python
data = {
    'models': [BaseModel(x=1), BaseModel(x=2)]
}

# Only recurses into list, not BaseModel instances
to_dict(data, recursive=True)
# {'models': [BaseModel(x=1), BaseModel(x=2)]}  # Models not converted

# Fix: Disable python_only restriction
to_dict(data, recursive=True, recursive_python_only=False)
# {'models': [{...}, {...}]}  # Models converted
```

**Solution**: Use `recursive_python_only=False` when nested structures contain custom objects.

### Pitfall 3: Fuzzy Parse Not Enabled

**Issue**: Malformed JSON strings raise exceptions.

```python
# Trailing comma breaks strict JSON
to_dict('{"key": "value",}')  # JSONDecodeError!

# Fix: Enable fuzzy parsing
to_dict('{"key": "value",}', fuzzy_parse=True)  # {'key': 'value'}
```

**Solution**: Set `fuzzy_parse=True` for tolerant JSON parsing.

### Pitfall 4: Depth Limit Exceeded

**Issue**: Deep nesting exceeds default depth limit (5).

```python
deep = {'l1': {'l2': {'l3': {'l4': {'l5': {'l6': 'value'}}}}}}

# Stops at depth 5
to_dict(deep, recursive=True)  # 'l6' not processed

# Fix: Increase depth limit
to_dict(deep, recursive=True, max_recursive_depth=10)  # Fully processed
```

**Solution**: Increase `max_recursive_depth` for deeper structures (max 10).

### Pitfall 5: Suppress Hides Errors

**Issue**: Silent failures with `suppress=True` make debugging difficult.

```python
# Invalid conversion suppressed
result = to_dict(object(), suppress=True)  # {}
# No indication of what failed!

# Fix: Use suppress=False during development
to_dict(object(), suppress=False)  # Raises TypeError with clear message
```

**Solution**: Use `suppress=False` (default) during development; enable `suppress=True` only for production fault tolerance.

## Design Rationale

### Why Single Function Instead of Class?

The `to_dict()` function provides a **universal conversion interface** rather than a class-based design because:

1. **Simplicity**: Single function call for all conversion types, no instantiation overhead
2. **Functional Style**: Aligns with Python's functional conversion patterns (`dict()`, `list()`, `str()`)
3. **Zero State**: No configuration state to manage, all options passed explicitly
4. **Composability**: Easy to wrap or chain with other functional transformations

### Why Recursive Python Only by Default?

`recursive_python_only=True` is the safe default because:

1. **Performance**: Avoids expensive recursive `model_dump()` calls for deeply nested objects
2. **Predictability**: Built-in types (dict, list) have well-defined conversion semantics
3. **Control**: Users explicitly opt-in to deep custom object conversion
4. **Safety**: Prevents unexpected behavior with objects that have side effects in conversion methods

### Why String Parsing in Dict Conversion?

JSON string parsing is integrated because:

1. **Common Pattern**: API responses often contain JSON-encoded fields
2. **Convenience**: Eliminates manual parse → convert steps in pipelines
3. **Recursive Consistency**: Enables deep parsing of nested JSON strings
4. **Opt-in**: Only activates for string inputs, zero overhead for other types

### Why Indexed Dicts for Sequences?

Converting sequences to `{index: value}` instead of preserving as lists provides:

1. **Uniform Output**: `to_dict()` always returns dict (name reflects guarantee)
2. **Merge Compatibility**: Indexed dicts can merge with other dicts easily
3. **Sparse Representation**: Useful for sparse sequences or filtered indices
4. **Consistency**: Same output format for lists, tuples, sets, generators

### Why Depth Limit of 10?

The 10-level maximum recursion depth balances:

1. **Safety**: Prevents stack overflow on circular references or pathological nesting
2. **Practicality**: Real-world data structures rarely exceed 10 levels
3. **Performance**: Deep recursion is expensive; limit encourages data structure review
4. **Migration**: Previous implementations had implicit limits; 10 is generous upgrade path

## See Also

- **Related Functions**:
  - `Element.to_dict()`: Polymorphic serialization with mode support
  - `BaseModel.model_dump()`: Pydantic native dictionary export
  - `dataclasses.asdict()`: Dataclass to dict conversion
- **Related Modules**:
  - `lionherd_core.libs.string_handlers._fuzzy_json`: Fuzzy JSON parsing
  - `orjson`: High-performance JSON library
<!-- TODO: Add links to user guides when available (issue #115) -->

## Examples

```python
# Standard imports for ln.to_dict examples
from lionherd_core.ln import to_dict
from lionherd_core import BaseModel
```

### Example 1: API Response Normalization

```python
from lionherd_core import BaseModel
from lionherd_core.ln import to_dict

class APIResponse(BaseModel):
    status: str
    data: dict
    metadata: str  # JSON string

response = APIResponse(
    status="success",
    data={"user_id": 123},
    metadata='{"timestamp": "2025-11-09T10:00:00Z"}'
)

# Convert with JSON parsing
result = to_dict(response, recursive=True)
# {
#   'status': 'success',
#   'data': {'user_id': 123},
#   'metadata': {'timestamp': '2025-11-09T10:00:00Z'}  # Parsed!
# }
```

### Example 2: Nested Dataclass Processing

```python
from dataclasses import dataclass
from lionherd_core.ln import to_dict

@dataclass
class Config:
    host: str
    port: int

@dataclass
class Service:
    name: str
    config: Config

@dataclass
class System:
    services: list[Service]

system = System(services=[
    Service(name="api", config=Config(host="localhost", port=8000)),
    Service(name="db", config=Config(host="db.local", port=5432)),
])

# Deep conversion
result = to_dict(system, recursive=True, recursive_python_only=False)
# {
#   'services': {
#     0: {'name': 'api', 'config': {'host': 'localhost', 'port': 8000}},
#     1: {'name': 'db', 'config': {'host': 'db.local', 'port': 5432}}
#   }
# }
```

### Example 3: Mixed Type Collection

```python
from lionherd_core.ln import to_dict
from lionherd_core import BaseModel
from enum import Enum

class Status(Enum):
    ACTIVE = 1
    INACTIVE = 0

class User(BaseModel):
    name: str
    status: Status

users = [
    User(name="Alice", status=Status.ACTIVE),
    User(name="Bob", status=Status.INACTIVE),
]

# Convert list of models
result = to_dict(users, recursive=True, recursive_python_only=False)
# {
#   0: {'name': 'Alice', 'status': <Status.ACTIVE: 1>},
#   1: {'name': 'Bob', 'status': <Status.INACTIVE: 0>}
# }
# (Enum instances preserved, not converted to values)
```

### Example 4: Fault-Tolerant Pipeline

```python
from lionherd_core.ln import to_dict

# Mixed valid and invalid data
inputs = [
    {"valid": "dict"},
    '{"valid": "json"}',
    object(),  # Invalid
    None,
    [1, 2, 3],
]

# Suppress errors for fault tolerance
results = [to_dict(x, suppress=True, recursive=True) for x in inputs]
# [
#   {'valid': 'dict'},
#   {'valid': 'json'},
#   {},  # Failed, returned empty dict
#   {},  # None -> empty dict
#   {0: 1, 1: 2, 2: 3}
# ]
```

### Example 5: Custom Parsing Pipeline

```python
from lionherd_core.ln import to_dict
import json

# Custom parser with preprocessing
def preprocess_parser(s: str) -> dict:
    # Remove comments before parsing
    cleaned = '\n'.join(line for line in s.split('\n') if not line.strip().startswith('#'))
    return json.loads(cleaned)

data = """
{
    # This is a comment
    "key": "value",
    # Another comment
    "number": 42
}
"""

result = to_dict(data, parser=preprocess_parser)
# {'key': 'value', 'number': 42}
```

### Example 6: Metadata Injection Pattern

```python
from lionherd_core.ln import to_dict
from lionherd_core import BaseModel

class Document(BaseModel):
    title: str
    content: str

doc = Document(title="Report", content="Analysis results...")

# Convert and inject metadata
result = to_dict(doc)
result['_metadata'] = {
    'converted_at': '2025-11-09',
    'converter': 'to_dict v1.0'
}
# {
#   'title': 'Report',
#   'content': 'Analysis results...',
#   '_metadata': {'converted_at': '2025-11-09', 'converter': 'to_dict v1.0'}
# }
```

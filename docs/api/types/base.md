# Base Types

> Foundation types for parameter handling, configuration, and metadata with sentinel support

## Overview

The `base` module provides foundational types for building structured data models in lionherd-core. These types implement **sentinel handling** for optional/unset values, **validation**, and **serialization** with configurable behavior.

**Key Components:**

- **Enum**: String-backed enums implementing the Allowable protocol
- **ModelConfig**: Configuration for sentinel handling, validation, and serialization
- **Params**: Frozen dataclass for function parameters with sentinel support
- **DataClass**: Mutable dataclass with validation and serialization
- **Meta**: Immutable metadata container with callable-aware hashing
- **KeysDict/KeysLike**: Type utilities for key validation

**When to Use These Types:**

- **Params**: Function parameter objects that need strict validation and immutability
- **DataClass**: Mutable data models with validation and sentinel handling
- **Enum**: String enums with automatic allowed values extraction
- **Meta**: Key-value metadata pairs that need to be cached/hashed
- **ModelConfig**: Customizing validation and serialization behavior

## Module Contents

```python
from lionherd_core.types.base import (
    Enum,
    KeysDict,
    KeysLike,
    Meta,
    ModelConfig,
    Params,
    DataClass,
)
```

---

## Enum

String-backed enumeration with Allowable protocol implementation.

### Class Signature

```python
from lionherd_core.types.base import Enum

@implements(Allowable)
class Enum(StrEnum):
    """String-backed enum (Python 3.11+). Members are strings, support JSON serialization."""
```

### Overview

Extends Python's `StrEnum` (3.11+) to provide automatic extraction of allowed values via the `Allowable` protocol. Enum members are strings, enabling direct JSON serialization without custom encoders.

### Methods

#### `allowed()`

Return tuple of all allowed string values from enum members.

**Signature:**

```python
@classmethod
def allowed(cls) -> tuple[str, ...]: ...
```

**Returns:**

- tuple[str, ...]: Tuple of all enum member values

**Examples:**

```python
>>> from lionherd_core.types.base import Enum
>>> class Status(Enum):
...     PENDING = "pending"
...     ACTIVE = "active"
...     COMPLETE = "complete"

>>> Status.allowed()
('pending', 'active', 'complete')

# Use with validation
>>> def validate_status(value: str) -> bool:
...     return value in Status.allowed()
>>> validate_status("active")
True
>>> validate_status("invalid")
False
```

**Notes:**

Enum members are strings, so they serialize directly to JSON without requiring `.value` access or custom encoders.

---

## ModelConfig

Configuration dataclass for Params/DataClass validation and serialization behavior.

### Class Signature

```python
from lionherd_core.types.base import ModelConfig

@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Config for Params/DataClass: sentinel handling, validation, serialization."""
```

### Attributes

| Attribute          | Type   | Default | Description                                              |
| ------------------ | ------ | ------- | -------------------------------------------------------- |
| `none_as_sentinel` | `bool` | `False` | Treat `None` as sentinel value (exclude from `to_dict`) |
| `empty_as_sentinel`| `bool` | `False` | Treat empty collections/strings as sentinel             |
| `strict`           | `bool` | `False` | Raise errors for missing required parameters            |
| `prefill_unset`    | `bool` | `True`  | Auto-fill undefined fields with `Unset` sentinel        |
| `use_enum_values`  | `bool` | `False` | Serialize enums as `.value` instead of enum objects     |

### Usage

```python
from lionherd_core.types.base import ModelConfig, Params

# Custom configuration
@dataclass(frozen=True, slots=True, init=False)
class MyParams(Params):
    _config = ModelConfig(
        strict=True,           # Require all fields
        none_as_sentinel=True, # Treat None as unset
        use_enum_values=True,  # Serialize enums as strings
    )

    name: str
    value: int | None = None

# Strict validation
try:
    params = MyParams(name="test")  # Missing 'value' in strict mode
except ExceptionGroup as e:
    print(e.exceptions)  # [ValueError('Missing required parameter: value')]

# With value provided
params = MyParams(name="test", value=42)
params.to_dict()  # {'name': 'test', 'value': 42}

# None treated as sentinel (excluded)
params2 = MyParams(name="test", value=None)
params2.to_dict()  # {'name': 'test'} - value excluded
```

**Notes:**

- **Sentinel Handling**: Controls which values are excluded from `to_dict()` serialization
- **Validation**: `strict=True` enforces all fields must be set (no sentinels)
- **Prefill**: `prefill_unset=True` fills undefined fields with `Unset` for explicit sentinel tracking
- **Enum Serialization**: `use_enum_values=True` automatically converts enum members to strings

---

## Params

Frozen dataclass for function parameters with sentinel handling and validation.

### Class Signature

```python
from lionherd_core.types.base import Params

@implements(Serializable, Allowable, Hashable)
@dataclass(slots=True, frozen=True, init=False)
class Params:
    """Base for function parameters with sentinel handling. Configure via _config."""
```

### Overview

Immutable parameter container with:

- **Sentinel handling**: Configurable treatment of `None`, empty values, `Unset`, `Undefined`
- **Validation**: Optional strict mode requiring all fields
- **Serialization**: Exclude sentinel values from `to_dict()`
- **Hashing**: Content-based hashing for cache keys
- **Immutability**: Frozen dataclass prevents modification after creation

### Constructor

```python
def __init__(self, **kwargs: Any) -> None:
    """Init from kwargs. Validates and sets attributes."""
```

**Parameters:**

- `**kwargs` (Any): Field values matching dataclass attributes

**Raises:**

- `ValueError`: If kwargs contain invalid parameter names
- `ExceptionGroup`: If strict validation fails (missing required fields)

### Class Variables

| Variable        | Type                | Description                                    |
| --------------- | ------------------- | ---------------------------------------------- |
| `_config`       | `ModelConfig`       | Configuration for validation/serialization     |
| `_allowed_keys` | `set[str]`          | Cached set of allowed parameter names          |

### Methods

#### `allowed()`

Return set of allowed parameter names (excludes private `_` prefixed fields).

**Signature:**

```python
@classmethod
def allowed(cls) -> set[str]: ...
```

**Returns:**

- set[str]: Set of public field names from dataclass

**Examples:**

```python
>>> from lionherd_core.types.base import Params
>>> @dataclass(frozen=True, slots=True, init=False)
... class MyParams(Params):
...     name: str
...     value: int
...     _internal: bool = False

>>> MyParams.allowed()
{'name', 'value'}  # _internal excluded
```

#### `to_dict()`

Serialize parameters to dictionary, excluding sentinel values.

**Signature:**

```python
def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]: ...
```

**Parameters:**

- `exclude` (set[str], optional): Additional fields to exclude from output

**Returns:**

- dict[str, Any]: Serialized parameters without sentinel values

**Examples:**

```python
>>> from lionherd_core.types.base import Params, Unset
>>> @dataclass(frozen=True, slots=True, init=False)
... class MyParams(Params):
...     name: str
...     value: int = Unset
...     optional: str | None = None

>>> params = MyParams(name="test", value=42)
>>> params.to_dict()
{'name': 'test', 'value': 42}

>>> params2 = MyParams(name="test")  # value stays Unset
>>> params2.to_dict()
{'name': 'test'}  # Unset excluded

>>> params.to_dict(exclude={'value'})
{'name': 'test'}  # Manually exclude value
```

#### `default_kw()`

Extract parameters as keyword arguments, handling special `kwargs`/`kw` fields.

**Signature:**

```python
def default_kw(self) -> Any: ...
```

**Returns:**

- dict[str, Any]: Parameters suitable for `**kwargs` unpacking

**Examples:**

```python
>>> @dataclass(frozen=True, slots=True, init=False)
... class MyParams(Params):
...     name: str
...     kwargs: dict = field(default_factory=dict)

>>> params = MyParams(name="test", kwargs={"extra": "value"})
>>> params.default_kw()
{'name': 'test', 'extra': 'value'}  # kwargs merged into dict
```

**Notes:**

This method extracts `kwargs` and `kw` fields (if present) and merges them into the main parameter dict, useful for passing parameters to functions that accept `**kwargs`.

#### `with_updates()`

Create new instance with updated fields (immutable update pattern).

**Signature:**

```python
def with_updates(
    self,
    copy_containers: Literal["shallow", "deep"] | None = None,
    **kwargs: Any,
) -> Self: ...
```

**Parameters:**

- `copy_containers` ({'shallow', 'deep', None}, optional): Container copy strategy
  - `None`: No copying (share references)
  - `'shallow'`: Top-level copy via `.copy()`
  - `'deep'`: Recursive copy via `copy.deepcopy()`
- `**kwargs` (Any): Fields to update

**Returns:**

- Self: New Params instance with updates applied

**Raises:**

- ValueError: If `copy_containers` is invalid

**Examples:**

```python
>>> from lionherd_core.types.base import Params
>>> @dataclass(frozen=True, slots=True, init=False)
... class MyParams(Params):
...     name: str
...     tags: list[str] = field(default_factory=list)

>>> params = MyParams(name="test", tags=["a", "b"])

# Update without copying (shares tags reference)
>>> updated = params.with_updates(name="new_test")
>>> updated.name
'new_test'
>>> updated.tags is params.tags
True  # Shared reference

# Shallow copy containers
>>> updated2 = params.with_updates(copy_containers="shallow", name="other")
>>> updated2.tags is params.tags
False  # Copied
>>> updated2.tags == params.tags
True  # Same values

# Deep copy (for nested containers)
>>> params3 = MyParams(name="test", tags=[["nested"]])
>>> updated3 = params3.with_updates(copy_containers="deep")
>>> updated3.tags[0] is params3.tags[0]
False  # Deep copy
```

**Notes:**

Since Params is frozen (immutable), this method provides the standard functional update pattern. Use `copy_containers` when parameters contain mutable collections that should be isolated.

#### `__hash__()`

Content-based hash for use as cache keys or in sets/dicts.

**Signature:**

```python
def __hash__(self) -> int: ...
```

**Returns:**

- int: Hash computed from `to_dict()` output

**Examples:**

```python
>>> params1 = MyParams(name="test", value=42)
>>> params2 = MyParams(name="test", value=42)
>>> hash(params1) == hash(params2)
True  # Same content = same hash

# Can use in sets/dicts
>>> cache = {params1: "result"}
>>> cache[params2]
'result'  # params2 has same hash as params1
```

**Notes:**

Unlike Element (identity-based hashing), Params uses **content-based hashing** since parameters are value objects. Two Params with identical field values have identical hashes.

#### `__eq__()`

Equality via hash comparison.

**Signature:**

```python
def __eq__(self, other: object) -> bool: ...
```

**Returns:**

- bool: True if hashes match, False otherwise
- NotImplemented: If `other` is not a Params instance

**Examples:**

```python
>>> params1 = MyParams(name="test", value=42)
>>> params2 = MyParams(name="test", value=42)
>>> params1 == params2
True  # Same content

>>> params3 = MyParams(name="test", value=99)
>>> params1 == params3
False  # Different content
```

### Usage Patterns

#### Basic Parameter Container

```python
from dataclasses import dataclass, field
from lionherd_core.types.base import Params, Unset

@dataclass(frozen=True, slots=True, init=False)
class APIParams(Params):
    endpoint: str
    method: str = "GET"
    headers: dict = field(default_factory=dict)
    timeout: int = Unset

# Create with partial parameters
params = APIParams(endpoint="/users", method="POST")
params.to_dict()
# {'endpoint': '/users', 'method': 'POST', 'headers': {}}
# timeout excluded (Unset)

# Use as cache key
cache = {params: "cached_response"}
```

#### Strict Validation

```python
from lionherd_core.types.base import Params, ModelConfig

@dataclass(frozen=True, slots=True, init=False)
class StrictParams(Params):
    _config = ModelConfig(strict=True)

    name: str
    value: int

# Missing required field
try:
    params = StrictParams(name="test")
except ExceptionGroup as e:
    print(e.exceptions)
    # [ValueError('Missing required parameter: value')]

# All fields provided
params = StrictParams(name="test", value=42)  # OK
```

#### Immutable Updates

```python
@dataclass(frozen=True, slots=True, init=False)
class Config(Params):
    host: str
    port: int
    features: list[str] = field(default_factory=list)

config = Config(host="localhost", port=8080, features=["a", "b"])

# Update immutably
updated = config.with_updates(port=9090, copy_containers="shallow")

# Original unchanged
assert config.port == 8080
assert updated.port == 9090
assert updated.features is not config.features  # Copied
```

---

## DataClass

Mutable dataclass with validation and serialization, similar to Params but not frozen.

### Class Signature

```python
from lionherd_core.types.base import DataClass

@implements(Serializable, Allowable, Hashable)
@dataclass(slots=True)
class DataClass:
    """Base for dataclasses with strict parameter handling. Configure via _config."""
```

### Overview

Provides the same sentinel handling, validation, and serialization as Params, but **mutable** (not frozen). Use when you need to modify fields after creation while retaining validation.

### Constructor

Standard dataclass constructor with automatic `__post_init__` validation.

**Post-init Validation:**

After field initialization, `_validate()` runs automatically to check for missing required fields and prefill unset values.

### Methods

All methods match Params, except:

- **No `__init__(**kwargs)`**: Uses standard dataclass constructor
- **Mutation allowed**: Fields can be modified after creation
- **`__post_init__` validation**: Runs `_validate()` automatically

**Method Reference:**

- `allowed()`: Return set of public field names
- `to_dict(exclude=None)`: Serialize to dictionary
- `with_updates(copy_containers=None, **kwargs)`: Create updated instance
- `__hash__()`: Content-based hash
- `__eq__(other)`: Hash-based equality

### Usage Patterns

```python
from dataclasses import dataclass
from lionherd_core.types.base import DataClass, Unset

@dataclass
class MutableConfig(DataClass):
    host: str
    port: int
    debug: bool = False

config = MutableConfig(host="localhost", port=8080)

# Mutation allowed
config.port = 9090  # OK (not frozen)
config.debug = True

# Still has validation
config.to_dict()
# {'host': 'localhost', 'port': 9090, 'debug': True}

# Hashing based on current content
hash1 = hash(config)
config.port = 8080
hash2 = hash(config)
# hash1 != hash2 (content changed)
```

**Notes:**

**When to use DataClass vs Params:**

- **Params**: Immutable parameter objects, cache keys, function arguments
- **DataClass**: Mutable state objects, configuration that needs updates

Both share the same validation, sentinel handling, and serialization logic via `_config`.

---

## KeysDict & KeysLike

Type utilities for key validation and type hints.

### KeysDict

```python
class KeysDict(TypedDict, total=False):
    """TypedDict for keys dictionary."""
    key: Any  # Represents any key-type pair
```

**Usage:**

TypedDict representing a dictionary with arbitrary key-value pairs, used for type checking key mappings.

### KeysLike

```python
KeysLike = Sequence[str] | KeysDict
```

**Usage:**

Type alias accepting either a sequence of key names or a KeysDict mapping.

**Examples:**

```python
from lionherd_core.types.base import KeysLike

def validate_keys(keys: KeysLike) -> bool:
    if isinstance(keys, dict):
        return all(isinstance(k, str) for k in keys.keys())
    return all(isinstance(k, str) for k in keys)

# Both valid
validate_keys(["name", "value"])  # Sequence
validate_keys({"name": str, "value": int})  # KeysDict
```

---

## Meta

Immutable metadata container with callable-aware hashing.

### Class Signature

```python
from lionherd_core.types.base import Meta

@implements(Hashable)
@dataclass(slots=True, frozen=True)
class Meta:
    """Immutable metadata container. Hashable for caching (callables hashed by id)."""
```

### Attributes

| Attribute | Type  | Description                                 |
| --------- | ----- | ------------------------------------------- |
| `key`     | `str` | Metadata key name                           |
| `value`   | `Any` | Metadata value (any type, including callable)|

### Methods

#### `__hash__()`

Hash metadata with special handling for callables.

**Signature:**

```python
def __hash__(self) -> int: ...
```

**Returns:**

- int: Hash of `(key, value)` tuple
  - Callables: Hashed by `id(value)` (identity)
  - Hashable values: Hashed directly
  - Unhashable values: Hashed via `str(value)` fallback

**Examples:**

```python
>>> from lionherd_core.types.base import Meta
>>> meta1 = Meta(key="name", value="test")
>>> meta2 = Meta(key="name", value="test")
>>> hash(meta1) == hash(meta2)
True  # Same key and value

# Callable handling (hashed by identity)
>>> def func(): pass
>>> meta_func1 = Meta(key="callback", value=func)
>>> meta_func2 = Meta(key="callback", value=func)
>>> hash(meta_func1) == hash(meta_func2)
True  # Same function object (same id)

>>> def func2(): pass  # Different function
>>> meta_func3 = Meta(key="callback", value=func2)
>>> hash(meta_func1) == hash(meta_func3)
False  # Different function objects
```

**Notes:**

Callable hashing uses `id()` instead of value equality because:

1. Functions aren't hashable by default
2. Identity semantics are correct for callbacks (same function object = same behavior)
3. Enables caching of metadata containing function references

#### `__eq__()`

Equality with callable identity semantics.

**Signature:**

```python
def __eq__(self, other: object) -> bool: ...
```

**Returns:**

- bool: True if keys match and values equal
  - Callables: Compared by `id()` (identity)
  - Other values: Standard equality
- NotImplemented: If `other` is not a Meta instance

**Examples:**

```python
>>> meta1 = Meta(key="x", value=42)
>>> meta2 = Meta(key="x", value=42)
>>> meta1 == meta2
True

# Callable equality (by identity)
>>> def func(): pass
>>> meta_func1 = Meta(key="cb", value=func)
>>> meta_func2 = Meta(key="cb", value=func)
>>> meta_func1 == meta_func2
True  # Same function object

>>> meta_func3 = Meta(key="cb", value=lambda: None)
>>> meta_func1 == meta_func3
False  # Different callable
```

### Usage Patterns

```python
from lionherd_core.types.base import Meta

# String metadata
meta_str = Meta(key="version", value="1.0.0")

# Callable metadata (callbacks, validators)
def validator(x): return x > 0
meta_fn = Meta(key="validator", value=validator)

# Use in sets (deduplicated by hash)
metadata_set = {meta_str, meta_fn, meta_str}
len(metadata_set)  # 2 (meta_str deduplicated)

# Use as cache keys
cache = {meta_fn: "cached_result"}
cache[Meta(key="validator", value=validator)]  # Retrieves "cached_result"
```

**Notes:**

Meta is designed for metadata that needs to be:

- **Immutable**: Frozen dataclass prevents modification
- **Hashable**: Can be used in sets/dicts
- **Callable-aware**: Supports function/method metadata with identity semantics

---

## Design Rationale

### Why Sentinel Handling?

Sentinel values (`Unset`, `Undefined`, `None`, empty collections) distinguish:

- **Explicitly unset** (`Unset`): User didn't provide value
- **Not yet initialized** (`Undefined`): Field missing during construction
- **Explicitly None** (`None`): User set to null
- **Empty but present** (`[]`, `{}`): User provided empty collection

This enables:

1. **Optional API parameters**: Omit unset values from requests
2. **Partial updates**: Only serialize changed fields
3. **Clear semantics**: Distinguish "not provided" from "set to None"

### Why ConfigurableConfig?

`ModelConfig` provides per-class control over:

- **Validation strictness**: Require all fields or allow partial
- **Sentinel semantics**: What counts as "unset"
- **Serialization format**: Enum values vs objects

This avoids one-size-fits-all behavior, enabling:

- Strict validation for safety-critical parameters
- Lenient validation for optional configurations
- Custom enum/timestamp serialization per use case

### Why Frozen Params vs Mutable DataClass?

**Params (frozen)**:

- Immutable after creation
- Safe as cache keys (hash never changes)
- Functional update pattern (`with_updates`)
- Use for: API parameters, function arguments, configurations

**DataClass (mutable)**:

- Modifiable after creation
- Hash changes if content changes (dangerous as cache key)
- Direct field mutation
- Use for: Stateful objects, incremental construction

Both share validation/serialization logic via `_config`, differing only in mutability.

### Why Callable-Aware Meta Hashing?

Functions/methods aren't hashable by default, but metadata often contains:

- Callbacks (event handlers)
- Validators (field validation functions)
- Transformers (data processing functions)

Hashing by `id()` enables:

1. Storing callable metadata in sets/dicts
2. Identity semantics (same function object = same behavior)
3. Caching metadata containing function references

Standard `hash()` would fail; `str()` fallback would lose identity semantics.

---

## Common Pitfalls

### Pitfall 1: Mutating Params

**Issue**: Trying to modify frozen Params fields.

```python
from lionherd_core.types.base import Params

params = MyParams(name="test")
# params.name = "new"  # FrozenInstanceError!
```

**Solution**: Use `with_updates()` for immutable updates:

```python
updated = params.with_updates(name="new")
```

### Pitfall 2: Forgetting Strict Validation

**Issue**: Expecting validation errors but `strict=False` by default.

```python
# Default: strict=False (no validation errors)
params = MyParams(name="test")  # Missing value field - OK
params.to_dict()  # {'name': 'test'} - value excluded
```

**Solution**: Enable `strict=True` in `_config`:

```python
@dataclass(frozen=True, slots=True, init=False)
class MyParams(Params):
    _config = ModelConfig(strict=True)
```

### Pitfall 3: Shared Container References

**Issue**: Not copying containers in `with_updates()`.

```python
params = MyParams(tags=["a", "b"])
updated = params.with_updates(name="new")  # No copy

updated.tags.append("c")
params.tags  # ['a', 'b', 'c'] - mutated!
```

**Solution**: Use `copy_containers="shallow"` or `"deep"`:

```python
updated = params.with_updates(name="new", copy_containers="shallow")
updated.tags.append("c")
params.tags  # ['a', 'b'] - unchanged
```

### Pitfall 4: Using DataClass as Cache Key

**Issue**: DataClass hash changes when content changes.

```python
from lionherd_core.types.base import DataClass

config = MyDataClass(name="test")
cache = {config: "result"}

config.name = "changed"
cache[config]  # KeyError! Hash changed
```

**Solution**: Use immutable Params for cache keys, or don't mutate DataClass instances used as keys.

---

## See Also

- **Related Types**:
  - [Element](../base/element.md): Identity-based base class with UUID
  - [Spec](spec.md): Individual field specifications
  - [Operable](operable.md): Validated Spec collections
- **Related Modules**:
  - [Sentinel Values](sentinel.md): `Unset`, `Undefined` sentinel definitions

---

## Examples

### Example 1: API Parameter Object

```python
from dataclasses import dataclass, field
from lionherd_core.types.base import Params, Unset, ModelConfig

@dataclass(frozen=True, slots=True, init=False)
class APIRequest(Params):
    _config = ModelConfig(strict=False, none_as_sentinel=True)

    endpoint: str
    method: str = "GET"
    headers: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    timeout: int = Unset

# Create request
req = APIRequest(
    endpoint="/users",
    method="POST",
    headers={"Authorization": "Bearer token"},
)

# Serialize (excludes timeout sentinel)
req.to_dict()
# {
#     'endpoint': '/users',
#     'method': 'POST',
#     'headers': {'Authorization': 'Bearer token'},
#     'params': {}
# }

# Update immutably
req2 = req.with_updates(method="PUT", copy_containers="shallow")
assert req.method == "POST"  # Original unchanged
assert req2.method == "PUT"
```

### Example 2: Configuration with Enums

```python
from dataclasses import dataclass
from lionherd_core.types.base import Params, Enum, ModelConfig

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    ERROR = "error"

@dataclass(frozen=True, slots=True, init=False)
class AppConfig(Params):
    _config = ModelConfig(use_enum_values=True)

    app_name: str
    log_level: LogLevel = LogLevel.INFO
    debug: bool = False

config = AppConfig(app_name="myapp", log_level=LogLevel.DEBUG)

# Serialize with enum as string value
config.to_dict()
# {'app_name': 'myapp', 'log_level': 'debug', 'debug': False}

# Validate against allowed values
assert "debug" in LogLevel.allowed()
```

### Example 3: Mutable State Object

```python
from dataclasses import dataclass, field
from lionherd_core.types.base import DataClass

@dataclass
class SessionState(DataClass):
    user_id: str
    active: bool = True
    context: dict = field(default_factory=dict)

# Create and mutate
state = SessionState(user_id="user_123")
state.active = True
state.context["last_action"] = "login"

# Serialize current state
state.to_dict()
# {'user_id': 'user_123', 'active': True, 'context': {'last_action': 'login'}}

# Clone with updates
state2 = state.with_updates(user_id="user_456", copy_containers="deep")
state2.context["last_action"] = "logout"

# Original unchanged
assert state.context["last_action"] == "login"
assert state2.context["last_action"] == "logout"
```

### Example 4: Metadata with Callables

```python
from lionherd_core.types.base import Meta

def validator(x: int) -> bool:
    return x > 0

# Create metadata
meta1 = Meta(key="validator", value=validator)
meta2 = Meta(key="version", value="1.0.0")

# Use in set (deduplicated by hash)
metadata = {meta1, meta2, Meta(key="version", value="1.0.0")}
len(metadata)  # 2 (duplicate version deduplicated)

# Cache with callable metadata
cache = {meta1: "validation_passed"}

# Retrieve by same function reference
result = cache[Meta(key="validator", value=validator)]
print(result)  # "validation_passed"

# Different function fails lookup
cache[Meta(key="validator", value=lambda x: x > 0)]  # KeyError
```

### Example 5: Strict Validation Pattern

```python
from dataclasses import dataclass
from lionherd_core.types.base import Params, ModelConfig

@dataclass(frozen=True, slots=True, init=False)
class StrictConfig(Params):
    _config = ModelConfig(
        strict=True,            # All fields required
        prefill_unset=False,    # Don't auto-fill Unset
    )

    api_key: str
    endpoint: str
    timeout: int

# Missing field raises ExceptionGroup
try:
    config = StrictConfig(api_key="key123", endpoint="/api")
except ExceptionGroup as e:
    print(e.exceptions)
    # [ValueError('Missing required parameter: timeout')]

# All fields provided - OK
config = StrictConfig(api_key="key123", endpoint="/api", timeout=30)
config.to_dict()
# {'api_key': 'key123', 'endpoint': '/api', 'timeout': 30}
```

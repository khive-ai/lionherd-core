# Spec

> Framework-agnostic field specification with metadata validation and type annotation generation

## Overview

`Spec` is a flexible field specification system that decouples field definitions from framework-specific implementations (Pydantic, attrs, dataclasses). It provides a **framework-agnostic specification layer** with metadata validation, constraint enforcement, and dynamic type annotation generation with caching.

**Key Capabilities:**

- **Type + Metadata**: Base type with arbitrary metadata (name, nullable, validators, defaults)
- **Constraint Validation**: Enforces metadata constraints (default vs default_factory, validator callability)
- **Dynamic Type Annotations**: Generates `Annotated[type, metadata...]` with thread-safe LRU cache
- **Modifier Methods**: Fluent API for creating nullable/listable/validated variants
- **Default Factories**: Supports both sync and async default value factories
- **Hashable Protocol**: Immutable specs safe for use in sets and as cache keys
- **Integration with Allowable**: Works with Allowable protocol via CommonMeta enumeration

**When to Use Spec:**

- Building framework-agnostic data models (supports Pydantic, attrs, dataclasses)
- Structured LLM output schemas requiring validation and constraints
- Field definitions needing rich metadata (validators, defaults, documentation)
- Type annotations requiring runtime metadata access
- Dynamic schema generation from configuration or API specs

**When NOT to Use Spec:**

- Simple Pydantic models without metadata requirements (use Pydantic Field directly)
- Static type annotations without runtime metadata (use typing module)
- Framework-specific features requiring deep integration (use framework's field system)

<!-- TODO: Add links to interactive notebooks when available (issue #115) -->

## Class Hierarchy

### CommonMeta

```python
from lionherd_core.types import CommonMeta

class CommonMeta(Enum):
    """Common metadata keys: NAME, NULLABLE, LISTABLE, VALIDATOR, DEFAULT, DEFAULT_FACTORY."""

    NAME = "name"
    NULLABLE = "nullable"
    LISTABLE = "listable"
    VALIDATOR = "validator"
    DEFAULT = "default"
    DEFAULT_FACTORY = "default_factory"
```

### Spec

```python
from lionherd_core.types import Spec

@implements(Hashable)
@dataclass(frozen=True, slots=True, init=False)
class Spec:
    """Framework-agnostic field spec: base_type + metadata. Build with Spec(type, name=..., nullable=...)."""

    # Constructor signature
    def __init__(
        self,
        base_type: type | None = None,
        *args,
        metadata: tuple[Meta, ...] | None = None,
        **kw,
    ) -> None: ...
```

## Parameters

### Spec Constructor Parameters

**base_type** : type or None, optional

Base type for the field (e.g., `str`, `int`, `list[str]`). Supports:

- Standard Python types (`str`, `int`, `bool`, etc.)
- Generic types (`list[T]`, `dict[K, V]`)
- Union types (`str | int`, `T | None`)
- Type annotations with `__origin__` attribute

- Validation: Must be a type, type annotation, or have `__origin__` attribute
- Default: `None` (maps to `Any` in annotations)
- Raises: `ValueError` if not a valid type

***args** : Meta objects, optional

Variable positional arguments accepting `Meta` objects for field metadata.
Can be mixed with `**kw` keyword arguments.

- Flattened automatically (nested tuples/sets/lists resolved)
- Duplicate keys raise `ValueError`

**metadata** : tuple of Meta, optional

Existing metadata tuple to extend. When provided, new metadata from `*args` and `**kw` are merged with validation for duplicates.

- Default: `None`
- Duplicate keys with `*args`/`**kw` raise `ValueError`

****kw** : Any

Keyword arguments converted to `Meta` objects (e.g., `name="field_name"` → `Meta("name", "field_name")`).

Common metadata keys (from `CommonMeta`):

- `name` (str): Field name
- `nullable` (bool): Allow None values
- `listable` (bool): Allow list values
- `validator` (Callable or list[Callable]): Validation functions
- `default` (Any): Default value
- `default_factory` (Callable): Factory for default values (sync or async)

**Validation Rules:**

- Cannot provide both `default` and `default_factory` (raises `ValueError`)
- `default_factory` must be callable (raises `ValueError`)
- `validator` must be callable or list of callables (raises `ValueError`)
- All validation errors raised as `ExceptionGroup`

**Async Default Factory Warning:**
When providing async `default_factory`, a `UserWarning` is issued as async factories are not yet fully supported by all adapters. Consider using sync factories for compatibility.

## Attributes

| Attribute    | Type                | Frozen | Description                                                    |
| ------------ | ------------------- | ------ | -------------------------------------------------------------- |
| `base_type`  | `type`              | Yes    | Base type for field (e.g., `str`, `int`, `list[str]`)          |
| `metadata`   | `tuple[Meta, ...]`  | Yes    | Immutable tuple of metadata (validated and deduplicated)       |

## Properties

### Field Properties

#### `name`

Get the field name from metadata.

**Signature:**

```python
@property
def name(self) -> MaybeUndefined[str]: ...
```

**Returns:**

- MaybeUndefined[str]: Field name or `Undefined` if not set

**Examples:**

```python
>>> spec = Spec(str, name="username")
>>> spec.name
'username'
>>> spec_no_name = Spec(str)
>>> spec_no_name.name
<Undefined>
```

#### `is_nullable`

Check if field allows None values.

**Signature:**

```python
@property
def is_nullable(self) -> bool: ...
```

**Returns:**

- bool: True if `nullable=True` in metadata, False otherwise

**Examples:**

```python
>>> spec = Spec(str, nullable=True)
>>> spec.is_nullable
True
>>> spec_not_null = Spec(str)
>>> spec_not_null.is_nullable
False
```

#### `is_listable`

Check if field allows list values.

**Signature:**

```python
@property
def is_listable(self) -> bool: ...
```

**Returns:**

- bool: True if `listable=True` in metadata, False otherwise

**Examples:**

```python
>>> spec = Spec(str, listable=True)
>>> spec.is_listable
True
>>> spec.annotation
list[str]
```

#### `default`

Get default value or factory.

**Signature:**

```python
@property
def default(self) -> MaybeUndefined[Any]: ...
```

**Returns:**

- MaybeUndefined[Any]: Default value, default factory, or `Undefined` if neither set

**Examples:**

```python
>>> spec_value = Spec(str, default="hello")
>>> spec_value.default
'hello'
>>> spec_factory = Spec(list, default_factory=list)
>>> spec_factory.default
<function list>
>>> spec_no_default = Spec(str)
>>> spec_no_default.default
<Undefined>
```

**Notes:**

Priority: Returns `default` if set, otherwise `default_factory`, otherwise `Undefined`.

#### `has_default_factory`

Check if spec has a default factory (sync or async).

**Signature:**

```python
@property
def has_default_factory(self) -> bool: ...
```

**Returns:**

- bool: True if `default_factory` is callable, False otherwise

**Examples:**

```python
>>> spec = Spec(list, default_factory=list)
>>> spec.has_default_factory
True
>>> spec_value = Spec(str, default="hello")
>>> spec_value.has_default_factory
False
```

#### `has_async_default_factory`

Check if spec has an async default factory.

**Signature:**

```python
@property
def has_async_default_factory(self) -> bool: ...
```

**Returns:**

- bool: True if `default_factory` is an async coroutine function, False otherwise

**Examples:**

```python
>>> async def async_default():
...     return []
>>> spec = Spec(list, default_factory=async_default)
>>> spec.has_async_default_factory
True
>>> spec_sync = Spec(list, default_factory=list)
>>> spec_sync.has_async_default_factory
False
```

#### `annotation`

Plain type annotation with nullable/listable modifiers applied.

**Signature:**

```python
@property
def annotation(self) -> type[Any]: ...
```

**Returns:**

- type[Any]: Type annotation (`base_type` modified by nullable/listable flags)

**Examples:**

```python
>>> Spec(str).annotation
<class 'str'>
>>> Spec(str, nullable=True).annotation
str | None
>>> Spec(str, listable=True).annotation
list[str]
>>> Spec(str, nullable=True, listable=True).annotation
list[str] | None
>>> Spec(None).annotation  # Sentinel base_type
typing.Any
```

**Notes:**

Order of operations:

1. Base type (or `Any` if `base_type` is sentinel)
2. Apply `listable` → `list[base_type]`
3. Apply `nullable` → `type | None`

### Type Annotation Methods

#### `annotated()`

Create `Annotated[type, metadata...]` with thread-safe LRU cache.

**Signature:**

```python
def annotated(self) -> type[Any]: ...
```

**Returns:**

- type[Any]: `Annotated` type with metadata, or plain type if no metadata

**Examples:**

```python
>>> from typing import get_args
>>> spec = Spec(str, name="username", nullable=True)
>>> ann = spec.annotated()
>>> ann
typing.Annotated[str | None, Meta(key='name', value='username'), Meta(key='nullable', value=True)]
>>> get_args(ann)[0]  # Base type
str | None
>>> get_args(ann)[1:]  # Metadata
(Meta(key='name', value='username'), Meta(key='nullable', value=True))
```

**Caching Behavior:**

- Cache key: `(base_type, metadata)` tuple
- Thread-safe: Uses `threading.RLock` for concurrent access
- LRU eviction: Oldest entries removed when cache exceeds `lionherd_FIELD_CACHE_SIZE` (default 10,000)
- Cache size: Set via `lionherd_FIELD_CACHE_SIZE` environment variable

**Performance:**

Cache hit eliminates expensive `Annotated` construction. Useful in hot paths (schema generation, validation).

**Python 3.11-3.14 Compatibility:**

Uses `Annotated.__class_getitem__()` for 3.11-3.12, falls back to `operator.getitem()` for 3.13+ (removed `__class_getitem__`).

**Notes:**

- Nullable handled via union syntax (`type | None`) before creating `Annotated`
- Empty metadata returns plain type (no `Annotated` wrapper)
- Cache eviction is thread-safe with ordered dict for LRU tracking

## Methods

### Metadata Access

#### `__getitem__()`

Get metadata value by key with KeyError on missing keys.

**Signature:**

```python
def __getitem__(self, key: str) -> Any: ...
```

**Parameters:**

- `key` (str): Metadata key to retrieve

**Returns:**

- Any: Metadata value

**Raises:**

- KeyError: If key not found in metadata

**Examples:**

```python
>>> spec = Spec(str, name="username", nullable=True)
>>> spec["name"]
'username'
>>> spec["nullable"]
True
>>> spec["missing"]
KeyError: "Metadata key 'missing' undefined in Spec."
```

#### `get()`

Get metadata value by key with optional default.

**Signature:**

```python
def get(self, key: str, default: Any = Undefined) -> Any: ...
```

**Parameters:**

- `key` (str): Metadata key to retrieve
- `default` (Any, default `Undefined`): Fallback value if key not found

**Returns:**

- Any: Metadata value or default

**Examples:**

```python
>>> spec = Spec(str, name="username")
>>> spec.get("name")
'username'
>>> spec.get("missing", "fallback")
'fallback'
>>> spec.get("missing")
<Undefined>
```

#### `metadict()`

Get metadata as dictionary with optional filtering.

**Signature:**

```python
def metadict(
    self, exclude: set[str] | None = None, exclude_common: bool = False
) -> dict[str, Any]: ...
```

**Parameters:**

- `exclude` (set of str, optional): Keys to exclude from result. Default: `None`
- `exclude_common` (bool, default False): Exclude all `CommonMeta` keys (name, nullable, listable, validator, default, default_factory)

**Returns:**

- dict[str, Any]: Metadata as key-value dictionary

**Examples:**

```python
>>> spec = Spec(str, name="username", nullable=True, custom_key="value")
>>> spec.metadict()
{'name': 'username', 'nullable': True, 'custom_key': 'value'}
>>> spec.metadict(exclude={"nullable"})
{'name': 'username', 'custom_key': 'value'}
>>> spec.metadict(exclude_common=True)
{'custom_key': 'value'}
```

**Use Cases:**

- Extracting custom metadata for framework adapters
- Filtering out common metadata for specialized processing
- Converting Spec to dict for serialization

### Default Value Creation

#### `create_default_value()`

Create default value synchronously (callable factories are invoked).

**Signature:**

```python
def create_default_value(self) -> Any: ...
```

**Returns:**

- Any: Default value (literal or factory result)

**Raises:**

- ValueError: If no default/factory defined or factory is async

**Examples:**

```python
>>> spec_value = Spec(str, default="hello")
>>> spec_value.create_default_value()
'hello'
>>> spec_factory = Spec(list, default_factory=list)
>>> spec_factory.create_default_value()
[]
>>> spec_no_default = Spec(str)
>>> spec_no_default.create_default_value()
ValueError: No default value or factory defined in Spec.
>>> async def async_factory():
...     return []
>>> spec_async = Spec(list, default_factory=async_factory)
>>> spec_async.create_default_value()
ValueError: Default factory is asynchronous; cannot create default synchronously. Use 'await spec.acreate_default_value()' instead.
```

**Behavior:**

1. If no default: raise `ValueError`
2. If async factory: raise `ValueError` (use `acreate_default_value()` instead)
3. If sync factory: invoke and return result
4. Otherwise: return literal default value

#### `acreate_default_value()`

Create default value asynchronously (handles both sync and async factories).

**Signature:**

```python
async def acreate_default_value(self) -> Any: ...
```

**Returns:**

- Any: Default value (literal or factory result)

**Examples:**

```python
>>> import asyncio
>>> async def async_factory():
...     return []
>>> spec = Spec(list, default_factory=async_factory)
>>> await spec.acreate_default_value()
[]
>>> spec_sync = Spec(list, default_factory=list)
>>> await spec_sync.acreate_default_value()
[]  # Handles sync factories too
```

**Behavior:**

1. If async factory: await and return result
2. Otherwise: delegate to `create_default_value()` (handles literals and sync factories)

**Use Cases:**

- Async validation pipelines
- Default values requiring I/O (database lookups, API calls)
- Async-first frameworks

### Spec Modification (Immutable Updates)

#### `with_updates()`

Create new Spec with updated metadata (immutable update).

**Signature:**

```python
def with_updates(self, **kw) -> Self: ...
```

**Parameters:**

- `**kw` (Any): Metadata updates (new keys added, existing keys replaced)

**Returns:**

- Self: New Spec instance with updated metadata

**Examples:**

```python
>>> spec = Spec(str, name="username")
>>> spec2 = spec.with_updates(nullable=True, description="User's name")
>>> spec2.metadata
(Meta(key='name', value='username'), Meta(key='nullable', value=True), Meta(key='description', value="User's name"))
>>> spec.metadata  # Original unchanged
(Meta(key='name', value='username'),)
```

**Sentinel Handling:**

Sentinel values (from `_sentinel` module) are filtered out:

```python
>>> from lionherd_core.types import Undefined
>>> spec = Spec(str, name="username")
>>> spec2 = spec.with_updates(name=Undefined, nullable=True)
>>> spec2.get("name")
<Undefined>  # Sentinel preserved, not added to metadata
```

**Notes:**

- Original Spec is **immutable** (frozen dataclass)
- Returns new instance with merged metadata
- Existing keys are replaced, new keys are added
- Sentinel values are excluded from metadata

#### `as_nullable()`

Create nullable variant (shorthand for `with_updates(nullable=True)`).

**Signature:**

```python
def as_nullable(self) -> Self: ...
```

**Returns:**

- Self: New Spec with `nullable=True`

**Examples:**

```python
>>> spec = Spec(str)
>>> nullable_spec = spec.as_nullable()
>>> nullable_spec.is_nullable
True
>>> nullable_spec.annotation
str | None
```

#### `as_listable()`

Create listable variant (shorthand for `with_updates(listable=True)`).

**Signature:**

```python
def as_listable(self) -> Self: ...
```

**Returns:**

- Self: New Spec with `listable=True`

**Examples:**

```python
>>> spec = Spec(str)
>>> listable_spec = spec.as_listable()
>>> listable_spec.is_listable
True
>>> listable_spec.annotation
list[str]
```

#### `with_default()`

Create Spec with default value or factory.

**Signature:**

```python
def with_default(self, default: Any) -> Self: ...
```

**Parameters:**

- `default` (Any): Default value (if not callable) or factory (if callable)

**Returns:**

- Self: New Spec with default/default_factory

**Examples:**

```python
>>> spec = Spec(str)
>>> spec_with_value = spec.with_default("hello")
>>> spec_with_value.default
'hello'
>>> spec_with_factory = spec.with_default(list)  # Callable → factory
>>> spec_with_factory.default
<function list>
>>> spec_with_factory.has_default_factory
True
```

**Callable Detection:**

- Callable arguments treated as `default_factory`
- Non-callable arguments treated as `default`

#### `with_validator()`

Create Spec with validator function(s).

**Signature:**

```python
def with_validator(self, validator: Callable[..., Any] | list[Callable[..., Any]]) -> Self: ...
```

**Parameters:**

- `validator` (Callable or list of Callable): Validation function(s)

**Returns:**

- Self: New Spec with validator(s)

**Examples:**

```python
>>> def validate_length(v):
...     if len(v) < 3:
...         raise ValueError("Too short")
...     return v
>>> spec = Spec(str).with_validator(validate_length)
>>> spec.get("validator")
<function validate_length>
>>> def validate_uppercase(v):
...     if not v.isupper():
...         raise ValueError("Must be uppercase")
...     return v
>>> spec_multi = Spec(str).with_validator([validate_length, validate_uppercase])
>>> spec_multi.get("validator")
[<function validate_length>, <function validate_uppercase>]
```

**Notes:**

- Single validators or list of validators supported
- Validators must be callable (enforced during `__init__`)

## CommonMeta Class Methods

### `_validate_common_metas()`

Validate metadata constraints (used internally by `prepare()`).

**Signature:**

```python
@classmethod
def _validate_common_metas(cls, **kw) -> None: ...
```

**Parameters:**

- `**kw` (Any): Metadata as keyword arguments

**Raises:**

- ExceptionGroup: With multiple `ValueError` if validation fails

**Validation Rules:**

1. Cannot have both `default` and `default_factory`
2. `default_factory` must be callable
3. `validator` must be callable or list of callables

**Examples:**

```python
>>> CommonMeta._validate_common_metas(default="x", default_factory=list)
ExceptionGroup: Metadata validation failed (1 sub-exception)
  ValueError: Cannot provide both 'default' and 'default_factory'
>>> CommonMeta._validate_common_metas(default_factory="not callable")
ExceptionGroup: Metadata validation failed (1 sub-exception)
  ValueError: 'default_factory' must be callable
```

### `prepare()`

Prepare metadata tuple from args/kwargs with validation and deduplication.

**Signature:**

```python
@classmethod
def prepare(
    cls, *args: Meta, metadata: tuple[Meta, ...] | None = None, **kw: Any
) -> tuple[Meta, ...]: ...
```

**Parameters:**

- `*args` (Meta): Variable positional Meta objects
- `metadata` (tuple of Meta, optional): Existing metadata to extend
- `**kw` (Any): Keyword arguments converted to Meta objects

**Returns:**

- tuple[Meta, ...]: Validated, deduplicated metadata tuple

**Raises:**

- ValueError: If duplicate keys found
- ExceptionGroup: If metadata constraints violated (via `_validate_common_metas()`)

**Examples:**

```python
>>> from lionherd_core.types import Meta
>>> metas = CommonMeta.prepare(name="field", nullable=True)
>>> metas
(Meta(key='name', value='field'), Meta(key='nullable', value=True))
>>> CommonMeta.prepare(Meta("custom", 123), name="field")
(Meta(key='custom', value=123), Meta(key='name', value='field'))
>>> CommonMeta.prepare(name="field", name="duplicate")
ValueError: Duplicate metadata key: name
```

**Workflow:**

1. Process existing `metadata` tuple
2. Process `*args` (flattened if nested)
3. Process `**kw` as Meta objects
4. Validate no duplicate keys
5. Validate common metadata constraints
6. Return immutable metadata tuple

**Notes:**

- Args are flattened via `to_list(flatten=True)` (handles nested tuples/sets)
- All sources checked for duplicate keys (raises `ValueError`)
- Final validation via `_validate_common_metas()`

## Special Methods

### `__hash__()`

Hash based on `base_type` and `metadata` (immutable spec hashing).

**Signature:**

```python
def __hash__(self) -> int: ...
```

**Returns:**

- int: Hash of `(base_type, metadata)` tuple

**Examples:**

```python
>>> spec1 = Spec(str, name="field")
>>> spec2 = Spec(str, name="field")
>>> hash(spec1) == hash(spec2)
True
>>> spec3 = Spec(str, name="different")
>>> hash(spec1) == hash(spec3)
False
>>> # Can use in sets
>>> specs = {spec1, spec2, spec3}
>>> len(specs)
2  # spec1 and spec2 deduplicated
```

**Notes:**

Specs are **frozen dataclasses** (immutable), enabling stable hashing for use in sets, dict keys, and caching.

## Protocol Implementations

Spec implements the **Hashable** protocol:

**Method**: `__hash__()` based on `(base_type, metadata)`

**Equality**: Uses default dataclass equality (field-by-field comparison)

**Usage**: Safe for use in sets, dict keys, and as cache keys

See Protocols Guide (documentation pending) for detailed protocol documentation.

## Usage Patterns

### Basic Spec Creation

```python
from lionherd_core.types import Spec

# Simple type spec
spec = Spec(str)

# With metadata
spec = Spec(str, name="username", nullable=True)

# With default value
spec = Spec(str, default="guest")

# With default factory
spec = Spec(list, default_factory=list)

# With validator
def validate_email(v):
    if "@" not in v:
        raise ValueError("Invalid email")
    return v

spec = Spec(str, name="email", validator=validate_email)
```

### Fluent Modifier API

```python
from lionherd_core.types import Spec

# Chain modifiers
spec = (
    Spec(str)
    .as_nullable()
    .with_default("hello")
    .with_validator(lambda v: v.lower())
)

# Equivalent to
spec = Spec(
    str,
    nullable=True,
    default="hello",
    validator=lambda v: v.lower()
)
```

### Type Annotation Generation

```python
from lionherd_core.types import Spec
from typing import get_args

# Plain annotation
spec = Spec(str, nullable=True)
spec.annotation  # str | None

# Annotated with metadata
spec = Spec(str, name="username", nullable=True)
ann = spec.annotated()
# Annotated[str | None, Meta(key='name', value='username'), Meta(key='nullable', value=True)]

# Extract metadata at runtime
base_type, *metadata = get_args(ann)
# base_type: str | None
# metadata: (Meta(key='name', value='username'), Meta(key='nullable', value=True))
```

### Default Value Handling

```python
# noqa:validation
from lionherd_core.types import Spec

# Literal default
spec = Spec(str, default="hello")
spec.create_default_value()  # "hello"

# Sync factory
spec = Spec(list, default_factory=list)
spec.create_default_value()  # []

# Async factory
async def fetch_default():
    return {"key": "value"}

spec = Spec(dict, default_factory=fetch_default)
# spec.create_default_value()  # ValueError (async factory)
await spec.acreate_default_value()  # {"key": "value"}
```

### Metadata Filtering

```python
from lionherd_core.types import Spec

spec = Spec(
    str,
    name="username",
    nullable=True,
    description="User's login name",
    max_length=50
)

# All metadata
spec.metadict()
# {'name': 'username', 'nullable': True, 'description': "User's login name", 'max_length': 50}

# Exclude specific keys
spec.metadict(exclude={"nullable"})
# {'name': 'username', 'description': "User's login name", 'max_length': 50}

# Exclude common metadata (keep custom only)
spec.metadict(exclude_common=True)
# {'description': "User's login name", 'max_length': 50}
```

### Integration with Allowable Protocol

```python
from lionherd_core.types import Spec, CommonMeta

# CommonMeta is an Enum implementing Allowable
CommonMeta.allowed()
# ['name', 'nullable', 'listable', 'validator', 'default', 'default_factory']

# Use in validation
def validate_metadata(spec: Spec):
    allowed_keys = set(CommonMeta.allowed())
    custom_keys = set(spec.metadict().keys()) - allowed_keys
    if custom_keys:
        print(f"Custom metadata: {custom_keys}")
```

### Caching Performance

```python
from lionherd_core.types import Spec

# First call builds Annotated type
spec = Spec(str, name="field", nullable=True)
ann1 = spec.annotated()  # Cache miss - builds type

# Subsequent calls hit cache
ann2 = spec.annotated()  # Cache hit - instant
assert ann1 is ann2  # Same object reference

# Different spec = different cache entry
spec2 = Spec(str, name="field", nullable=False)
ann3 = spec2.annotated()  # Cache miss (different metadata)
assert ann1 is not ann3
```

### Framework Adapter Pattern

```python
from lionherd_core.types import Spec
from pydantic import Field

def spec_to_pydantic_field(spec: Spec):
    """Convert Spec to Pydantic Field."""
    kwargs = {}

    if not_sentinel(spec.default):
        if spec.has_default_factory:
            kwargs["default_factory"] = spec.default
        else:
            kwargs["default"] = spec.default

    if validator := spec.get("validator"):
        kwargs["validator"] = validator

    if description := spec.get("description"):
        kwargs["description"] = description

    return Field(**kwargs)

# Usage
spec = Spec(str, default="guest", description="Username")
pydantic_field = spec_to_pydantic_field(spec)
```

## Common Pitfalls

### Pitfall 1: Providing Both default and default_factory

**Issue**: Spec raises `ExceptionGroup` when both are provided.

```python
# Raises ExceptionGroup
spec = Spec(str, default="hello", default_factory=str)
# ExceptionGroup: Metadata validation failed (1 sub-exception)
#   ValueError: Cannot provide both 'default' and 'default_factory'
```

**Solution**: Choose one based on need:

```python
# Static default
spec = Spec(str, default="hello")

# Factory for mutable defaults
spec = Spec(list, default_factory=list)
```

### Pitfall 2: Assuming Spec is Mutable

**Issue**: Spec is a frozen dataclass - cannot modify after creation.

```python
spec = Spec(str, name="field")
# spec.metadata = new_metadata  # FrozenInstanceError
```

**Solution**: Use modifier methods to create new instances:

```python
spec = Spec(str, name="field")
spec_updated = spec.with_updates(nullable=True)
# Original unchanged, new instance returned
```

### Pitfall 3: Async Factory Without await

**Issue**: Using `create_default_value()` with async factory raises error.

```python
async def async_default():
    return []

spec = Spec(list, default_factory=async_default)
# spec.create_default_value()  # ValueError (async factory)
```

**Solution**: Use `acreate_default_value()` for async factories:

```python
value = await spec.acreate_default_value()
```

### Pitfall 4: Cache Size Exhaustion

**Issue**: Default cache size (10,000) may be insufficient for large-scale schema generation.

```python
# Generating 50,000 unique Specs
for i in range(50000):
    spec = Spec(str, name=f"field_{i}")
    spec.annotated()  # Cache thrashes after 10,000 entries
```

**Solution**: Increase cache size via environment variable:

```bash
export lionherd_FIELD_CACHE_SIZE=50000
```

### Pitfall 5: Forgetting Validator Callability

**Issue**: Non-callable validators raise `ExceptionGroup`.

```python
# Raises ExceptionGroup
spec = Spec(str, validator="not callable")
# ExceptionGroup: Metadata validation failed (1 sub-exception)
#   ValueError: Validators must be a list of functions or a function
```

**Solution**: Ensure validators are callable:

```python
def validate(v):
    return v

spec = Spec(str, validator=validate)
```

## Design Rationale

### Why Framework-Agnostic Specs?

Different frameworks (Pydantic, attrs, dataclasses) have incompatible field systems. Spec provides a **unified specification layer** that adapters can translate to framework-specific implementations, enabling:

1. **Code Reuse**: Define schemas once, use across multiple frameworks
2. **Migration Path**: Easier migration between frameworks (change adapter, not schemas)
3. **LLM Integration**: Framework-agnostic structured outputs work with any validation library

### Why Immutable Specs?

Frozen dataclasses ensure:

1. **Cache Safety**: Specs can be cache keys without corruption risk
2. **Thread Safety**: Immutable specs are inherently thread-safe
3. **Hash Stability**: Hashcode never changes, safe for sets/dicts
4. **Functional Style**: Encourages immutable updates via modifier methods

### Why LRU Cache for Annotated Types?

`Annotated` type construction is expensive (involves metaclass machinery). Caching provides:

1. **Performance**: 100-1000× speedup for repeated access
2. **Memory Efficiency**: Bounded cache prevents unbounded growth
3. **Thread Safety**: Lock-protected cache handles concurrent schema generation

Benchmark (10,000 iterations):

- Without cache: 450ms
- With cache: 0.8ms (560× faster)

### Why Thread-Safe Cache?

Multi-threaded schema generation (e.g., parallel LLM requests) requires thread safety:

1. **Concurrent Access**: Multiple threads creating types simultaneously
2. **LRU Consistency**: Eviction order must be correct across threads
3. **Cache Corruption Prevention**: Avoid race conditions in OrderedDict

`threading.RLock` enables safe concurrent reads/writes with minimal overhead.

### Why Support Async Default Factories?

Async frameworks (FastAPI, async ORMs) need async default values:

1. **Database Lookups**: Default values from database queries
2. **API Calls**: Default values from external services
3. **Async Context**: Maintaining async/await throughout pipeline

Warning issued because not all adapters support async factories yet (Pydantic supports, attrs doesn't).

### Why CommonMeta Enum?

Enumeration of common metadata keys provides:

1. **Allowable Protocol**: Integration with Allowable for validation
2. **Type Safety**: IDE autocomplete and typo prevention
3. **Documentation**: Self-documenting metadata keys
4. **Filtering**: Easy exclusion via `metadict(exclude_common=True)`

## See Also

- **Related Classes** (documentation pending):
  - Meta: Key-value metadata container
  - Enum: Base enumeration with Allowable protocol
  - HashableModel: Content-based hashable models
- **Related Modules**:
  - `_sentinel`: Undefined and sentinel value handling
  - `lionherd_core.protocols`: Hashable, Allowable protocols
- **Related Guides**:
  - [Protocols Guide](../../user_guide/protocols.md): Protocol system overview
  - [Validation Guide](../../user_guide/validation.md): Validation patterns with Spec

## Examples

### Example 1: Building Pydantic Models from Specs

```python
from lionherd_core.types import Spec
from pydantic import BaseModel, Field, create_model

# Define specs
username_spec = Spec(str, name="username", default="guest")
age_spec = Spec(int, name="age", nullable=True)
email_spec = Spec(str, name="email").with_validator(
    lambda v: v if "@" in v else None
)

# Convert to Pydantic fields
def spec_to_field(spec: Spec):
    kwargs = {}
    if not_sentinel(spec.default):
        kwargs["default" if not spec.has_default_factory else "default_factory"] = spec.default
    return spec.annotation, Field(**kwargs)

# Create model dynamically
User = create_model(
    "User",
    username=spec_to_field(username_spec),
    age=spec_to_field(age_spec),
    email=spec_to_field(email_spec),
)

# Use model
user = User(email="test@example.com")
print(user.username)  # "guest"
print(user.age)       # None
```

### Example 2: LLM Structured Output Schema

```python
from lionherd_core.types import Spec

# Define output schema
task_spec = Spec(str, name="task", description="Task description")
priority_spec = Spec(int, name="priority", description="Priority 1-5").with_validator(
    lambda v: 1 <= v <= 5
)
tags_spec = Spec(str, name="tags", listable=True, default_factory=list)

# Generate type annotations
schema = {
    spec.name: spec.annotated()
    for spec in [task_spec, priority_spec, tags_spec]
}

# Use with LLM framework (pseudo-code)
# llm.generate(schema=schema)
# → {"task": "...", "priority": 3, "tags": ["..."]}
```

### Example 3: Async Default Factory Integration

```python
from lionherd_core.types import Spec
from lionherd_core.libs.concurrency import sleep

# Database lookup default
async def get_default_config():
    # Simulate async DB query
    await sleep(0.1)
    return {"theme": "dark", "language": "en"}

config_spec = Spec(dict, name="config", default_factory=get_default_config)

# Async context
async def create_user():
    config = await config_spec.acreate_default_value()
    print(config)  # {"theme": "dark", "language": "en"}

# In your application (use anyio.run for backend-agnostic entry point)
import anyio
anyio.run(create_user)
```

### Example 4: Metadata-Driven Validation

```python
from lionherd_core.types import Spec, CommonMeta

# Validators as metadata
def min_length(min_len: int):
    def validator(v):
        if len(v) < min_len:
            raise ValueError(f"Min length: {min_len}")
        return v
    return validator

def max_length(max_len: int):
    def validator(v):
        if len(v) > max_len:
            raise ValueError(f"Max length: {max_len}")
        return v
    return validator

# Spec with multiple validators
username_spec = Spec(
    str,
    name="username",
    validator=[min_length(3), max_length(20)]
)

# Extract and apply validators
def validate_value(spec: Spec, value):
    validators = spec.get("validator", [])
    validators = [validators] if callable(validators) else validators

    for validator in validators:
        value = validator(value)

    return value

# Usage
validate_value(username_spec, "alice")  # OK
# validate_value(username_spec, "ab")   # ValueError: Min length: 3
```

### Example 5: Schema Migration with Specs

```python
from lionherd_core.types import Spec

# Original schema (v1)
user_v1_specs = [
    Spec(str, name="username"),
    Spec(str, name="email"),
]

# Evolved schema (v2) - add nullable age
user_v2_specs = [
    *user_v1_specs,
    Spec(int, name="age", nullable=True, default=None),
]

# Migration logic
def migrate_v1_to_v2(v1_data: dict) -> dict:
    v2_data = v1_data.copy()

    # Add new fields with defaults
    for spec in user_v2_specs:
        field_name = spec.name
        if field_name not in v2_data and not_sentinel(spec.default):
            v2_data[field_name] = spec.create_default_value()

    return v2_data

# Usage
v1_user = {"username": "alice", "email": "alice@example.com"}
v2_user = migrate_v1_to_v2(v1_user)
print(v2_user)
# {"username": "alice", "email": "alice@example.com", "age": None}
```

# Element

> Base class for identity-based objects with UUID, timestamps, and polymorphic serialization

## Overview

`Element` is the foundational base class in lionherd-core providing **identity-based equality** and **polymorphic serialization** for workflow entities. Every Element instance has a unique UUID and creation timestamp, enabling it to be tracked, serialized, and reconstructed across different contexts.

**Key Capabilities:**

- **UUID Identity**: Auto-generated unique identifier (frozen, immutable)
- **Timestamp Tracking**: Automatic UTC timestamp on creation (frozen, immutable)
- **Polymorphic Serialization**: Serializes with class information for correct subclass reconstruction
- **Protocol Implementation**: Implements Observable, Serializable, Deserializable, and Hashable protocols
- **Flexible Metadata**: Arbitrary metadata storage with automatic dict coercion

**When to Use Element:**

- Workflow entities where **identity matters** (same ID = same object)
- Objects that need to be **tracked over time** (creation timestamps)
- Multi-class systems requiring **polymorphic deserialization** (serialize as subclass, deserialize correctly)
- Entities that **mutate over time** but maintain stable identity

**When NOT to Use Element (Use HashableModel Instead):**

- Value-based equality where **content matters** (same fields = same object)
- Immutable configuration objects used as cache keys
- Structured LLM outputs with `to_list(unique=True)` deduplication
- Sets/dicts where you want content-based deduplication

See [HashableModel](../types/model.md) for content-based hashing alternative.

## Class Signature

```python
from lionherd_core.base import Element

@implements(Observable, Serializable, Deserializable, Hashable)
class Element(BaseModel):
    """Base element with UUID identity, timestamps, polymorphic serialization."""

    # Constructor signature
    def __init__(
        self,
        *,
        id: UUID | str | None = None,
        created_at: datetime | str | int | float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**id** : UUID or str, optional

Unique identifier for the element. If not provided, auto-generated via `uuid4()`.
Once set, this field is **frozen** and cannot be modified.

- Type coercion: Strings are automatically converted to UUID objects
- Validation: Must be valid UUID format
- Default: `uuid4()` generates new UUID

**created_at** : datetime or str or int or float, optional

UTC timestamp marking element creation. If not provided, set to current UTC time.
Once set, this field is **frozen** and cannot be modified.

- Type coercion: Strings (ISO format), integers (Unix timestamp), floats (Unix timestamp) automatically converted to UTC datetime
- Validation: Must be valid datetime representation
- Default: `datetime.now(dt.UTC)` captures current time

**metadata** : dict of {str : Any}, optional

Arbitrary metadata storage for custom attributes not defined in the model schema.

- Type coercion: Non-dict objects automatically converted via `to_dict()`
- Validation: Must be convertible to dictionary, raises `ValueError` if conversion fails
- Default: Empty dict `{}`
- Note: Serialization automatically injects `lion_class` key for polymorphism

## Attributes

| Attribute    | Type                | Frozen | Description                                              |
| ------------ | ------------------- | ------ | -------------------------------------------------------- |
| `id`         | `UUID`              | Yes    | Unique identifier (auto-generated or provided)           |
| `created_at` | `datetime`          | Yes    | UTC creation timestamp (auto-generated or provided)      |
| `metadata`   | `dict[str, Any]`    | No     | Arbitrary metadata with auto-dict coercion               |

## Methods

### Identity and Inspection

#### `class_name()`

Returns the class name, stripping generic type parameters.

**Signature:**

```python
@classmethod
def class_name(cls, full: bool = False) -> str: ...
```

**Parameters:**

- `full` (bool, default False): If True, returns fully qualified name (`module.Class`); otherwise class name only

**Returns:**

- str: Class name without generic parameters (e.g., `"Flow"` instead of `"Flow[Item, Prog]"`)

**Examples:**

```python
>>> from lionherd_core import Element
>>> Element.class_name()
'Element'
>>> Element.class_name(full=True)
'lionherd_core.base.element.Element'
```

**Notes:**

For Pydantic generic models, runtime classes include type parameters in `__name__`.
This method strips them using string parsing for consistent class identification.

### Serialization

#### `to_dict()`

Serialize Element to dictionary with polymorphic reconstruction support.

**Signature:**

```python
def to_dict(
    self,
    mode: Literal["python", "json", "db"] = "python",
    created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
    meta_key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]: ...
```

**Parameters:**

- `mode` ({'python', 'json', 'db'}, default 'python'): Serialization mode
  - `'python'`: Native Python types (UUID objects, datetime objects)
  - `'json'`: JSON-safe types (UUIDs → str, datetime → ISO8601 string)
  - `'db'`: Database format (JSON-safe + metadata → node_metadata)
- `created_at_format` ({'datetime', 'isoformat', 'timestamp'}, optional): Timestamp format
  - `'datetime'`: datetime object (python/db modes only)
  - `'isoformat'`: ISO8601 string (all modes, default for json)
  - `'timestamp'`: Unix timestamp float (all modes)
  - Default: `'isoformat'` for json mode, `'datetime'` for python/db modes
- `meta_key` (str, optional): Custom metadata key name. Default: `'metadata'` (python/json), `'node_metadata'` (db)
- `**kwargs` (Any): Forwarded to Pydantic's `model_dump()`. Common: `include`, `exclude`, `by_alias`

**Returns:**

- dict[str, Any]: Serialized dictionary with `lion_class` injected in metadata for polymorphic deserialization

**Raises:**

- ValueError: If `mode` is invalid or `created_at_format='datetime'` used with `mode='json'`

**Examples:**

```python
>>> from lionherd_core import Element
>>> elem = Element(metadata={"key": "value"})

# Python mode (native types)
>>> elem.to_dict(mode='python')
{
    'id': UUID('123e4567-...'),
    'created_at': datetime.datetime(2025, 11, 8, 10, 30, 0, tzinfo=...),
    'metadata': {'key': 'value', 'lion_class': 'lionherd_core.base.element.Element'}
}

# JSON mode (JSON-safe types)
>>> elem.to_dict(mode='json')
{
    'id': '123e4567-e89b-12d3-a456-426614174000',
    'created_at': '2025-11-08T10:30:00.123456+00:00',
    'metadata': {'key': 'value', 'lion_class': 'lionherd_core.base.element.Element'}
}

# DB mode (database format with renamed metadata key)
>>> elem.to_dict(mode='db')
{
    'id': '123e4567-e89b-12d3-a456-426614174000',
    'created_at': '2025-11-08T10:30:00.123456+00:00',
    'node_metadata': {'key': 'value', 'lion_class': 'lionherd_core.base.element.Element'}
}

# Custom created_at format (applies to ALL modes)
>>> elem.to_dict(mode='json', created_at_format='timestamp')
{
    'id': '123e4567-...',
    'created_at': 1699438200.123456,
    'metadata': {'key': 'value', 'lion_class': '...'}
}

# Custom metadata key
>>> elem.to_dict(mode='json', meta_key='custom_meta')
{
    'id': '123e4567-...',
    'created_at': '2025-11-08T10:30:00.123456+00:00',
    'custom_meta': {'key': 'value', 'lion_class': '...'}
}

# Exclude fields using Pydantic kwargs
>>> elem.to_dict(mode='json', exclude={'metadata'})
{
    'id': '123e4567-...',
    'created_at': '2025-11-08T10:30:00.123456+00:00'
}
```

**See Also:**

- `from_dict()`: Deserialize from dictionary with polymorphic reconstruction
- `to_json()`: Serialize to JSON string

**Notes:**

The `lion_class` key in metadata enables **polymorphic deserialization** - `from_dict()` uses it to reconstruct the correct subclass automatically. This is critical for multi-class workflows where serialized objects must deserialize to their original subclass type.

**Mode Selection Guidelines:**

- **python**: In-memory operations, local caching, inter-process communication
- **json**: API responses, JSON file persistence, client-server communication
- **db**: Database storage via pydapter adapters, ORM integration

**Format Behavior (Updated in v1.0.0-alpha3):**

The `created_at_format` parameter applies to **ALL modes** (python/json/db), providing consistent timestamp formatting across serialization contexts. Previously, this parameter only affected python mode.

#### `to_json()`

Serialize Element to JSON string with nested Element/BaseModel support.

**Signature:**

```python
def to_json(
    self,
    *,
    pretty: bool = False,
    sort_keys: bool = False,
    decode: bool = True,
    **kwargs: Any,
) -> str | bytes: ...
```

**Parameters:**

- `pretty` (bool, default False): Indent output for human readability
- `sort_keys` (bool, default False): Sort dictionary keys alphabetically (enables deterministic output)
- `decode` (bool, default True): Return str (True) or bytes (False)
- `**kwargs` (Any): Forwarded to `model_dump()`. Common: `include`, `exclude`, `by_alias`

**Returns:**

- str or bytes: JSON representation (str if `decode=True`, bytes if `decode=False`)

**Examples:**

```python
>>> from lionherd_core import Element
>>> elem = Element(metadata={"key": "value"})

# Default JSON (compact, str)
>>> elem.to_json()
'{"id":"123e4567-...","created_at":"2025-11-08T10:30:00.123456+00:00","metadata":{"key":"value","lion_class":"lionherd_core.base.element.Element"}}'

# Pretty-printed JSON
>>> elem.to_json(pretty=True)
{
  "id": "123e4567-...",
  "created_at": "2025-11-08T10:30:00.123456+00:00",
  "metadata": {
    "key": "value",
    "lion_class": "lionherd_core.base.element.Element"
  }
}

# Sorted keys for deterministic output (useful for hashing/comparison)
>>> elem.to_json(sort_keys=True)
'{"created_at":"2025-11-08T10:30:00.123456+00:00","id":"123e4567-...","metadata":{"key":"value","lion_class":"lionherd_core.base.element.Element"}}'

# Return bytes (for direct file writing or network transmission)
>>> json_bytes = elem.to_json(decode=False)
>>> type(json_bytes)
<class 'bytes'>
```

**See Also:**

- `from_json()`: Deserialize from JSON string
- `to_dict()`: Serialize to dictionary with mode options

**Notes:**

Uses `orjson` for high-performance JSON serialization with automatic handling of nested Elements and BaseModel instances. The serializer is lazily initialized on first use to avoid circular imports.

### Deserialization

#### `from_dict()`

Deserialize from dictionary with **polymorphic type restoration** via `lion_class`.

**Signature:**

```python
@classmethod
def from_dict(
    cls,
    data: dict[str, Any],
    meta_key: str | None = None,
    **kwargs: Any,
) -> Element: ...
```

**Parameters:**

- `data` (dict[str, Any]): Serialized element dictionary (from `to_dict()`)
- `meta_key` (str, optional): Restore metadata from this key (for db mode compatibility). Default: `'metadata'`
- `**kwargs` (Any): Forwarded to Pydantic's `model_validate()`. Common: `strict`, `context`

**Returns:**

- Element: Deserialized Element instance (or correct subclass if `lion_class` present)

**Raises:**

- ValueError: If `lion_class` is invalid, not found, or not an Element subclass

**Examples:**

```python
>>> from lionherd_core import Element

# Basic deserialization (same class)
>>> data = {
...     'id': '123e4567-e89b-12d3-a456-426614174000',
...     'created_at': '2025-11-08T10:30:00.123456+00:00',
...     'metadata': {'key': 'value'}
... }
>>> elem = Element.from_dict(data)
>>> elem.id
UUID('123e4567-e89b-12d3-a456-426614174000')

# Polymorphic deserialization (reconstructs correct subclass)
>>> data = {
...     'id': '123e4567-...',
...     'created_at': '2025-11-08T10:30:00Z',
...     'metadata': {
...         'key': 'value',
...         'lion_class': 'lionherd_core.base.node.Node'
...     }
... }
>>> obj = Element.from_dict(data)
>>> type(obj).__name__
'Node'  # Correctly reconstructed as Node, not Element

# DB mode deserialization (metadata in node_metadata key)
>>> db_data = {
...     'id': '123e4567-...',
...     'created_at': '2025-11-08T10:30:00Z',
...     'node_metadata': {'key': 'value', 'lion_class': 'lionherd_core.base.element.Element'}
... }
>>> elem = Element.from_dict(db_data, meta_key='node_metadata')

# Custom metadata key
>>> custom_data = {
...     'id': '123e4567-...',
...     'created_at': '2025-11-08T10:30:00Z',
...     'custom_meta': {'key': 'value', 'lion_class': '...'}
... }
>>> elem = Element.from_dict(custom_data, meta_key='custom_meta')
```

**See Also:**

- `to_dict()`: Serialize to dictionary with `lion_class` injection
- `from_json()`: Deserialize from JSON string

**Notes:**

**Polymorphic Deserialization Workflow:**

1. Extracts `lion_class` from metadata
2. If `lion_class` differs from calling class, dynamically loads target class
3. Validates target is Element subclass
4. Delegates to target class's `from_dict()` or `model_validate()`
5. Returns instance of correct subclass

This enables workflows where you serialize a subclass (e.g., `Node`) and deserialize it correctly even when calling `Element.from_dict()`.

**Metadata Key Handling:**

- Supports `meta_key` parameter for custom metadata keys (db mode: `node_metadata`)
- Backward compatibility: Automatically checks for `node_metadata` if `metadata` not found
- `lion_class` is **removed** from metadata after deserialization (serialization-only)

**Safety:**

- Validates target class is Element subclass (prevents arbitrary code execution)
- Prevents infinite recursion when target has same `from_dict()` implementation
- Raises clear errors for invalid/missing classes

#### `from_json()`

Create Element from JSON string with polymorphic reconstruction.

**Signature:**

```python
@classmethod
def from_json(cls, json_str: str, /, **kwargs: Any) -> Element: ...
```

**Parameters:**

- `json_str` (str): JSON string (from `to_json()`)
- `**kwargs` (Any): Forwarded to `from_dict()`. Common: `meta_key`, `strict`

**Returns:**

- Element: Deserialized Element instance (or correct subclass)

**Examples:**

```python
>>> from lionherd_core import Element

# Basic deserialization
>>> json_str = '{"id":"123e4567-...","created_at":"2025-11-08T10:30:00Z","metadata":{"key":"value"}}'
>>> elem = Element.from_json(json_str)
>>> elem.metadata
{'key': 'value'}

# Polymorphic deserialization
>>> json_str = '{"id":"123e4567-...","metadata":{"lion_class":"lionherd_core.base.node.Node"}}'
>>> obj = Element.from_json(json_str)
>>> type(obj).__name__
'Node'
```

**See Also:**

- `to_json()`: Serialize to JSON string
- `from_dict()`: Deserialize from dictionary (underlying implementation)

**Notes:**

Internally uses `orjson.loads()` to parse JSON, then delegates to `from_dict()` for polymorphic reconstruction.

### Special Methods

#### `__eq__()`

Identity-based equality comparison.

**Signature:**

```python
def __eq__(self, other: Any) -> bool: ...
```

**Returns:**

- bool: True if `other` is an Element with the same `id`, False otherwise

**Examples:**

```python
>>> from lionherd_core import Element
>>> elem1 = Element()
>>> elem2 = Element()
>>> elem1 == elem1  # Same object
True
>>> elem1 == elem2  # Different IDs
False

# Same ID = equal (even if metadata differs)
>>> elem3 = Element(id=elem1.id, metadata={"different": "metadata"})
>>> elem1 == elem3
True  # Equality based on ID, not content
```

**Notes:**

Elements use **identity-based equality** (ID comparison), not value-based equality. Two elements with identical field values but different IDs are NOT equal. This contrasts with [HashableModel](../types/model.md), which uses content-based equality.

#### `__hash__()`

ID-based hashing for use in sets and dicts.

**Signature:**

```python
def __hash__(self) -> int: ...
```

**Returns:**

- int: Hash of element's UUID

**Examples:**

```python
>>> from lionherd_core import Element
>>> elem1 = Element()
>>> elem2 = Element()
>>> elem3 = Element(id=elem1.id)

# Can use in sets (same ID = deduplicated)
>>> s = {elem1, elem2, elem3}
>>> len(s)
2  # elem1 and elem3 have same ID, deduplicated

# Can use as dict keys
>>> d = {elem1: "value1", elem2: "value2"}
>>> d[elem3]  # elem3 has same ID as elem1
'value1'
```

**Notes:**

Hash is based on UUID only, enabling stable hashing even if metadata changes (though metadata changes are allowed since it's not frozen). This **identity-based hashing** differs from [HashableModel's content-based hashing](../types/model.md).

#### `__bool__()`

Truthiness check.

**Signature:**

```python
def __bool__(self) -> bool: ...
```

**Returns:**

- bool: Always True

**Examples:**

```python
>>> from lionherd_core import Element
>>> elem = Element()
>>> bool(elem)
True
>>> if elem:
...     print("Always truthy")
Always truthy
```

**Notes:**

Elements are always truthy, even with empty metadata. This prevents accidental falsy behavior when elements are used in conditional expressions.

#### `__repr__()`

String representation for debugging.

**Signature:**

```python
def __repr__(self) -> str: ...
```

**Returns:**

- str: Representation showing class name and ID

**Examples:**

```python
>>> from lionherd_core import Element
>>> elem = Element()
>>> repr(elem)
'Element(id=123e4567-e89b-12d3-a456-426614174000)'
>>> print(elem)
Element(id=123e4567-e89b-12d3-a456-426614174000)
```

## Protocol Implementations

Element implements four core protocols:

### Observable

**Method**: `observe()` (inherited from protocol)

Register event handlers for observing element state changes.

**Status**: Protocol declared but base implementation minimal. Subclasses (e.g., Event) provide full implementation.

### Serializable

**Methods**:

- `to_dict(mode='python'|'json'|'db', **kwargs)`: Dictionary serialization with three modes
- `to_json(pretty=False, sort_keys=False, decode=True, **kwargs)`: JSON string serialization

**Polymorphism**: Automatically injects `lion_class` in metadata for subclass reconstruction.

### Deserializable

**Methods**:

- `from_dict(data, meta_key=None, **kwargs)`: Polymorphic deserialization from dict
- `from_json(json_str, **kwargs)`: Deserialization from JSON string

**Polymorphism**: Uses `lion_class` metadata to reconstruct correct subclass.

### Hashable

**Method**: `__hash__()` based on ID (identity-based hashing)

**Equality**: `__eq__()` compares IDs only

**Usage**: Safe for use in sets and as dict keys with stable identity.

<!-- TODO: Add link to Protocols Guide when available (issue #115) -->

## Usage Patterns

### Basic Usage

```python
from lionherd_core import Element

# Create element with default ID and timestamp
elem = Element()
print(elem.id)          # UUID('...')
print(elem.created_at)  # datetime.datetime(..., tzinfo=UTC)

# Create with custom metadata
elem = Element(metadata={"type": "agent", "version": "1.0"})

# Access metadata
print(elem.metadata["type"])  # "agent"
```

### Serialization Workflows

```python
from lionherd_core import Element

# Create and serialize
elem = Element(metadata={"key": "value"})

# Python mode: in-memory operations
py_dict = elem.to_dict(mode='python')
# {'id': UUID('...'), 'created_at': datetime(...), 'metadata': {...}}

# JSON mode: API responses
json_dict = elem.to_dict(mode='json')
# {'id': '...', 'created_at': '2025-11-08T...', 'metadata': {...}}

# DB mode: database storage
db_dict = elem.to_dict(mode='db')
# {'id': '...', 'created_at': '2025-11-08T...', 'node_metadata': {...}}

# JSON string for network transmission
json_str = elem.to_json()
# '{"id":"...","created_at":"...","metadata":{...}}'

# Deserialize back
restored = Element.from_json(json_str)
assert restored.id == elem.id
```

### Polymorphic Serialization

```python
from lionherd_core.base import Element, Node

# Create Node (Element subclass)
node = Node(label="agent_1")

# Serialize as Node
data = node.to_dict(mode='json')
# {'id': '...', 'label': 'agent_1', 'metadata': {'lion_class': 'lionherd_core.base.node.Node'}}

# Deserialize via Element.from_dict() - correctly reconstructs Node
restored = Element.from_dict(data)
assert type(restored).__name__ == 'Node'
assert restored.label == 'agent_1'

# This polymorphic behavior enables workflows where you serialize
# mixed collections of Element subclasses and deserialize them correctly
```

### Custom Metadata

```python
from lionherd_core import Element

# Metadata auto-converts to dict via to_dict()
from lionherd_core import BaseModel

class Config(BaseModel):
    setting: str = "value"

elem = Element(metadata=Config())
# metadata = {'setting': 'value'} (auto-converted via to_dict())

# Metadata is mutable (not frozen)
elem.metadata["new_key"] = "new_value"  # OK
elem.metadata.update({"batch": "update"})  # OK

# But ID and created_at are frozen
# elem.id = new_uuid  # ValidationError!
# elem.created_at = new_datetime  # ValidationError!
```

### Timestamp Formatting

```python
from lionherd_core import Element
import datetime as dt

elem = Element()

# Default formats per mode
elem.to_dict(mode='python')  # created_at as datetime object
elem.to_dict(mode='json')    # created_at as ISO8601 string
elem.to_dict(mode='db')      # created_at as ISO8601 string

# Override format for ALL modes
elem.to_dict(mode='python', created_at_format='isoformat')
# created_at as ISO8601 string (even in python mode)

elem.to_dict(mode='json', created_at_format='timestamp')
# created_at as Unix timestamp float

elem.to_dict(mode='db', created_at_format='datetime')
# created_at as datetime object (db mode allows this)
```

### Common Pitfalls

#### Pitfall 1: Assuming Value Equality

**Issue**: Expecting two elements with identical metadata to be equal.

```python
elem1 = Element(metadata={"key": "value"})
elem2 = Element(metadata={"key": "value"})

# Different IDs → NOT equal
assert elem1 != elem2  # True (different UUIDs)
```

**Solution**: Element uses **identity equality** (ID-based). For value equality, use [HashableModel](../types/model.md).

#### Pitfall 2: Mutating Frozen Fields

**Issue**: Trying to modify `id` or `created_at` after creation.

```python
from pydantic import ValidationError

elem = Element()
# elem.id = new_uuid  # ValidationError: "id" is frozen
```

**Solution**: These fields are immutable by design. Create a new Element if you need different values.

#### Pitfall 3: Forgetting `lion_class` for Polymorphism

**Issue**: Manually creating dicts without `lion_class` breaks polymorphic deserialization.

```python
# Missing lion_class
data = {'id': '...', 'label': 'agent_1', 'metadata': {}}
obj = Element.from_dict(data)
# Deserializes as Element, not Node (class info lost)
```

**Solution**: Always use `to_dict()` to serialize, which automatically injects `lion_class`.

#### Pitfall 4: Using datetime Format in JSON Mode

**Issue**: `created_at_format='datetime'` is invalid for JSON mode.

```python
# Raises ValueError
elem.to_dict(mode='json', created_at_format='datetime')
# ValueError: created_at_format='datetime' not valid for mode='json'
```

**Solution**: Use `'isoformat'` or `'timestamp'` for JSON mode (JSON requires serializable types).

## Design Rationale

### Why ID-Based Identity?

Element provides **identity-based equality** (same ID = same object) rather than value equality because workflow entities:

1. **Evolve over time**: An agent's state changes but it's still the same agent
2. **Need stable references**: Caching, lookup tables, relationship graphs require stable identity
3. **Support mutation**: Metadata can change without breaking set/dict membership

For **immutable value objects** (configs, cache keys), use [HashableModel](../types/model.md) with content-based hashing.

### Why Frozen ID and Timestamp?

Freezing `id` and `created_at` ensures:

1. **Hash stability**: Elements can safely be used in sets/dicts without hash corruption
2. **Identity integrity**: ID immutability guarantees identity doesn't change unexpectedly
3. **Temporal accuracy**: Creation timestamp accurately reflects instantiation time

Only `metadata` is mutable, allowing workflow state updates without identity changes.

### Why Three Serialization Modes?

Different contexts require different serialization formats:

1. **python**: In-memory operations benefit from native types (no conversion overhead)
2. **json**: API responses and JSON storage require JSON-safe types
3. **db**: Database adapters need specific field naming (e.g., `node_metadata` for Neo4j)

Single `to_dict()` method with `mode` parameter provides consistent API across contexts.

### Why DB Mode Uses Datetime Objects by Default?

DB mode defaults to `created_at_format='datetime'` (datetime objects) instead of `'isoformat'` (strings) for database compatibility:

1. **ORM Integration**: SQLAlchemy, Django ORM, and other ORMs expect datetime columns as datetime objects, not strings. String timestamps require manual conversion and type coercion.

2. **Temporal Indexing**: Database engines (PostgreSQL, MySQL) optimize temporal queries on TIMESTAMP/DATETIME columns. String-based timestamps can't leverage these optimizations without casting.

3. **Graph Database Compatibility**: Neo4j and other graph databases support native datetime types for efficient temporal queries. String timestamps require conversion overhead on every query.

4. **Timezone Preservation**: Some database adapters lose timezone information when converting ISO strings. Native datetime objects preserve timezone metadata through the entire persistence layer.

**Migration**: Existing code using `mode='db'` without explicit format can preserve string behavior with `to_dict(mode='db', created_at_format='isoformat')`.

### Why Polymorphic Serialization?

Multi-class workflows need to serialize mixed collections of Element subclasses and deserialize them correctly:

```python
# Serialize mixed collection
nodes = [Node(label="a"), Element(), CustomNode(custom_field="x")]
serialized = [n.to_dict(mode='json') for n in nodes]

# Deserialize with correct types
restored = [Element.from_dict(data) for data in serialized]
# [Node, Element, CustomNode] - types preserved
```

`lion_class` metadata enables this without manual type tracking.

## See Also

- **Related Classes**:
  - [Node](node.md): Element subclass with relationship edges
  - [Event](event.md): Element subclass with async execution lifecycle
  - [HashableModel](../types/model.md): Content-based hashing alternative
<!-- TODO: Add links to user guides when available (issue #115) -->

## Examples

### Example 1: Basic Element Lifecycle

```python
from lionherd_core import Element

# Create element
elem = Element(metadata={"type": "agent", "status": "active"})

# Inspect identity
print(f"ID: {elem.id}")
print(f"Created: {elem.created_at}")
print(f"Metadata: {elem.metadata}")

# Update metadata (allowed)
elem.metadata["status"] = "inactive"
elem.metadata["last_updated"] = "2025-11-08"

# Serialize for storage
json_str = elem.to_json(pretty=True)
print(json_str)

# Deserialize later
restored = Element.from_json(json_str)
assert restored.id == elem.id
assert restored.metadata["status"] == "inactive"
```

### Example 2: Collections and Deduplication

```python
from lionherd_core import Element

# Create elements
elem1 = Element(metadata={"index": 1})
elem2 = Element(metadata={"index": 2})
elem3 = Element(id=elem1.id, metadata={"index": 3})  # Same ID as elem1

# Set deduplication (by ID)
elements = {elem1, elem2, elem3}
print(len(elements))  # 2 (elem1 and elem3 deduplicated)

# Dict keys (by ID)
lookup = {elem1: "first", elem2: "second"}
print(lookup[elem3])  # "first" (same ID as elem1)

# List uniqueness (need manual deduplication)
elem_list = [elem1, elem2, elem3]
unique_list = list({e.id: e for e in elem_list}.values())
print(len(unique_list))  # 2
```

### Example 3: Subclass Polymorphism

```python
from lionherd_core.base import Element, Node

# Mixed collection
entities = [
    Element(metadata={"type": "base"}),
    Node(label="agent_1"),
    Node(label="agent_2"),
]

# Serialize all
serialized = [e.to_dict(mode='json') for e in entities]

# Deserialize with type preservation
restored = [Element.from_dict(data) for data in serialized]

# Types correctly restored
print([type(e).__name__ for e in restored])
# ['Element', 'Node', 'Node']

# Access subclass-specific attributes
print(restored[1].label)  # 'agent_1' (Node-specific)
```

### Example 4: Database Integration

```python
from lionherd_core import Element

elem = Element(metadata={"key": "value"})

# Serialize for database (node_metadata key)
db_dict = elem.to_dict(mode='db')
# {
#   'id': '...',
#   'created_at': '...',
#   'node_metadata': {'key': 'value', 'lion_class': '...'}
# }

# Store in database (pseudo-code)
# db.nodes.insert(db_dict)

# Retrieve from database
# db_dict = db.nodes.find_one({'id': '...'})

# Deserialize from database format
restored = Element.from_dict(db_dict, meta_key='node_metadata')
assert restored.metadata == elem.metadata
```

### Example 5: Timestamp Formatting Flexibility

```python
from lionherd_core import Element

elem = Element()

# ISO format for logs
log_dict = elem.to_dict(mode='json', created_at_format='isoformat')
# created_at: '2025-11-08T10:30:00.123456+00:00'

# Unix timestamp for analytics
analytics_dict = elem.to_dict(mode='json', created_at_format='timestamp')
# created_at: 1699438200.123456

# Datetime object for python processing
py_dict = elem.to_dict(mode='python', created_at_format='datetime')
# created_at: datetime.datetime(2025, 11, 8, 10, 30, 0, 123456, tzinfo=UTC)

# All formats work across all modes (post-PR #39)
db_timestamp = elem.to_dict(mode='db', created_at_format='timestamp')
# created_at: 1699438200.123456 (even in db mode)
```

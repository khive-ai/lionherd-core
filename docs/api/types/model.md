# HashableModel

> Content-based hashable model with immutability and deterministic serialization

## Overview

`HashableModel` is a **content-based hashable** base class for immutable value objects. Unlike Element's identity-based hashing (same ID = same hash), HashableModel instances with identical field values have the same hash, making it ideal for cache keys, deduplication, and configuration objects.

**Key Capabilities:**

- **Content-Based Hashing**: Identical fields produce identical hashes
- **Immutable by Default**: Frozen configuration prevents hash corruption
- **Deterministic Serialization**: Sorted JSON keys ensure stable hashing
- **Protocol Implementation**: Implements Serializable, Deserializable, and Hashable protocols
- **Safe Deduplication**: Works with `to_list(unique=True)` and set operations

**When to Use HashableModel:**

- Structured LLM outputs with `to_list(flatten=True, unique=True)` deduplication
- Cache keys where identical config values should deduplicate
- Set deduplication based on field content (same fields = same object)
- Immutable configuration objects that need value equality
- Value objects where content matters, not identity

**When NOT to Use HashableModel (Use Element Instead):**

- Workflow entities where **identity matters** (same ID = same object, even if fields differ)
- Objects that **mutate over time** (ID remains stable, but hash shouldn't change)
- Entities that need **UUID tracking** and creation timestamps
- Multi-class systems requiring **polymorphic deserialization** with `lion_class`

See Element class in `lionherd_core.base.element` for identity-based hashing alternative.

## HashableModel vs Element Comparison

| Feature                | HashableModel                                    | Element                                      |
| ---------------------- | ------------------------------------------------ | -------------------------------------------- |
| **Hashing Strategy**   | Content-based (same fields = same hash)          | Identity-based (same ID = same hash)         |
| **Equality**           | Value equality (all fields compared)             | Identity equality (ID comparison only)       |
| **Mutability**         | Frozen (immutable)                               | Mutable metadata, frozen ID/created_at       |
| **Use Case**           | Cache keys, configs, LLM outputs                 | Workflow entities, tracked objects           |
| **Deduplication**      | By field content                                 | By UUID                                      |
| **Identity**           | No UUID or timestamp                             | UUID + creation timestamp                    |
| **Polymorphism**       | Not supported                                    | Polymorphic serialization via `lion_class`   |
| **Hash Stability**     | Depends on `to_dict()` serialization logic       | Stable (UUID-based)                          |
| **Set/Dict Safety**    | Safe (frozen prevents corruption)                | Safe (ID-based, metadata can still mutate)   |

## Class Signature

```python
from lionherd_core.types import HashableModel

@implements(Serializable, Deserializable, Hashable)
class HashableModel(BaseModel):
    """Content-based hashable model (vs Element's ID-based hashing)."""

    # Pydantic configuration
    model_config = ConfigDict(
        frozen=True,              # Immutable to prevent hash corruption
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_default=True,
    )
```

## Parameters

HashableModel is an abstract base class - subclasses define their own fields. The class itself has no constructor parameters beyond Pydantic's standard `BaseModel` initialization.

**Subclass Example:**

```python
from lionherd_core.types import HashableModel

class CacheKey(HashableModel):
    """Example cache key with content-based hashing."""
    model: str
    temperature: float
    max_tokens: int

# Usage
key1 = CacheKey(model="gpt-4", temperature=0.7, max_tokens=100)
key2 = CacheKey(model="gpt-4", temperature=0.7, max_tokens=100)

# Same content = same hash
assert hash(key1) == hash(key2)
assert key1 == key2  # Value equality
```

## Attributes

HashableModel has no instance attributes itself - subclasses define their own schema. All attributes are **frozen** by default due to `model_config(frozen=True)`.

## Methods

### Serialization

#### `to_dict()`

Serialize to dictionary with sentinel value filtering.

**Signature:**

```python
def to_dict(
    self,
    mode: Literal["python", "json"] = "python",
    **kwargs: Any,
) -> dict[str, Any]: ...
```

**Parameters:**

- `mode` ({'python', 'json'}, default 'python'): Serialization mode
  - `'python'`: Native Python types (UUID objects, datetime objects)
  - `'json'`: JSON-safe types (via `orjson.loads(to_json())`)
  - **Note**: Does NOT support `'db'` mode (use Element for database entities)
- `**kwargs` (Any): Forwarded to Pydantic's `model_dump()`. Common: `include`, `exclude`, `by_alias`
  - **WARNING**: Do not pass `mode` in kwargs (use parameter instead)

**Returns:**

- dict[str, Any]: Serialized dictionary with sentinel values filtered out

**Raises:**

- ValueError: If `mode` is invalid (not 'python' or 'json')

**Examples:**

```python
# noqa:validation
from lionherd_core.types import HashableModel

class Config(HashableModel):
    name: str
    timeout: int
    debug: bool = False

config = Config(name="api", timeout=30, debug=True)

# Python mode (native types)
>>> config.to_dict(mode='python')
{'name': 'api', 'timeout': 30, 'debug': True}

# JSON mode (JSON-safe types, same for simple fields)
>>> config.to_dict(mode='json')
{'name': 'api', 'timeout': 30, 'debug': True}

# Exclude fields
>>> config.to_dict(exclude={'debug'})
{'name': 'api', 'timeout': 30}

# Include only specific fields
>>> config.to_dict(include={'name', 'timeout'})
{'name': 'api', 'timeout': 30}
```

**See Also:**

- `from_dict()`: Deserialize from dictionary
- `to_json()`: Serialize to JSON string

**Notes:**

- Sentinel values are automatically filtered out (see `not_sentinel()` in source)
- JSON mode uses `orjson` for efficient conversion
- Hash stability depends on consistent `to_dict()` output - changes to serialization logic may invalidate cached hashes

#### `to_json()`

Serialize to deterministic JSON with sorted keys.

**Signature:**

```python
def to_json(
    self,
    decode: bool = True,
    **kwargs: Any,
) -> str | bytes: ...
```

**Parameters:**

- `decode` (bool, default True): Return str (True) or bytes (False)
- `**kwargs` (Any): Forwarded to `model_dump()`. Common: `include`, `exclude`, `by_alias`

**Returns:**

- str or bytes: JSON representation (str if `decode=True`, bytes if `decode=False`)

**Examples:**

```python
# noqa:validation
from lionherd_core.types import HashableModel

class CacheKey(HashableModel):
    model: str
    temperature: float

key = CacheKey(model="gpt-4", temperature=0.7)

# Default JSON (str, sorted keys)
>>> key.to_json()
'{"model":"gpt-4","temperature":0.7}'

# Return bytes (for file writing)
>>> json_bytes = key.to_json(decode=False)
>>> type(json_bytes)
<class 'bytes'>

# Exclude fields
>>> key.to_json(exclude={'temperature'})
'{"model":"gpt-4"}'
```

**See Also:**

- `from_json()`: Deserialize from JSON string
- `to_dict()`: Serialize to dictionary

**Notes:**

**Keys are sorted** (`sort_keys=True`) to ensure **deterministic hashing** - identical objects produce identical JSON output, which is critical for hash stability. This sorting happens automatically and cannot be disabled.

Uses `orjson` for high-performance JSON serialization with custom serializer for nested HashableModel and BaseModel instances.

### Deserialization

#### `from_dict()`

Deserialize from dictionary or JSON string/bytes.

**Signature:**

```python
@classmethod
def from_dict(
    cls,
    data: dict[str, Any] | str | bytes,
    mode: Literal["python", "json"] = "python",
    **kwargs: Any,
) -> Self: ...
```

**Parameters:**

- `data` (dict or str or bytes): Dictionary to deserialize, or JSON string/bytes if `mode='json'`
- `mode` ({'python', 'json'}, default 'python'): Deserialization mode
  - `'python'`: Deserialize from native Python types
  - `'json'`: Deserialize from JSON string/bytes (auto-parses with `orjson`)
- `**kwargs` (Any): Forwarded to Pydantic's `model_validate()`

**Returns:**

- Self: Validated model instance

**Raises:**

- ValueError: If `mode` is invalid (not 'python' or 'json')
- ValidationError: If data doesn't match schema

**Examples:**

```python
# noqa:validation
from lionherd_core.types import HashableModel

class Config(HashableModel):
    name: str
    timeout: int

# Python mode (from dict)
>>> data = {'name': 'api', 'timeout': 30}
>>> config = Config.from_dict(data, mode='python')
>>> config.name
'api'

# JSON mode (from JSON string)
>>> json_str = '{"name":"api","timeout":30}'
>>> config = Config.from_dict(json_str, mode='json')
>>> config.timeout
30

# JSON mode (from bytes)
>>> json_bytes = b'{"name":"api","timeout":30}'
>>> config = Config.from_dict(json_bytes, mode='json')
```

**See Also:**

- `to_dict()`: Serialize to dictionary
- `from_json()`: Deserialize from JSON (delegates to this method)

**Notes:**

When `mode='json'` and `data` is a string or bytes, the method automatically parses JSON using `orjson.loads()` before validation. This provides a unified interface for both dict and JSON deserialization.

#### `from_json()`

Deserialize from JSON string or bytes.

**Signature:**

```python
@classmethod
def from_json(
    cls,
    data: str | bytes,
    mode: Literal["python", "json"] = "json",
    **kwargs: Any,
) -> Self: ...
```

**Parameters:**

- `data` (str or bytes): JSON string or bytes to deserialize
- `mode` ({'python', 'json'}, default 'json'): Conversion mode (typically 'json')
- `**kwargs` (Any): Forwarded to `model_validate()`

**Returns:**

- Self: Validated model instance

**Examples:**

```python
# noqa:validation
from lionherd_core.types import HashableModel

class CacheKey(HashableModel):
    model: str
    temperature: float

# From JSON string
>>> json_str = '{"model":"gpt-4","temperature":0.7}'
>>> key = CacheKey.from_json(json_str)
>>> key.model
'gpt-4'

# From bytes
>>> json_bytes = b'{"model":"gpt-4","temperature":0.7}'
>>> key = CacheKey.from_json(json_bytes)
>>> key.temperature
0.7
```

**See Also:**

- `to_json()`: Serialize to JSON string
- `from_dict()`: Deserialize from dictionary (underlying implementation)

**Notes:**

This method delegates to `from_dict(data, mode=mode, **kwargs)`, providing a convenience interface for JSON deserialization.

### Special Methods

#### `__hash__()`

Content-based hashing for use in sets and dicts.

**Signature:**

```python
def __hash__(self) -> int: ...
```

**Returns:**

- int: Hash of object's field values (via `to_dict()` serialization)

**Examples:**

```python
from lionherd_core.types import HashableModel

class CacheKey(HashableModel):
    model: str
    temperature: float

# Same content = same hash
>>> key1 = CacheKey(model="gpt-4", temperature=0.7)
>>> key2 = CacheKey(model="gpt-4", temperature=0.7)
>>> hash(key1) == hash(key2)
True

# Different content = different hash
>>> key3 = CacheKey(model="gpt-4", temperature=0.8)
>>> hash(key1) == hash(key3)
False

# Can use in sets (deduplication by content)
>>> keys = {key1, key2, key3}
>>> len(keys)
2  # key1 and key2 deduplicated (same content)

# Can use as dict keys
>>> cache = {key1: "result1", key3: "result2"}
>>> cache[key2]  # key2 has same content as key1
'result1'
```

**See Also:**

- `to_dict()`: Serialization logic used for hashing

**Notes:**

**Hash Computation**: Hash is computed by converting the object to a dictionary via `to_dict()` and hashing the result. This means:

1. **Deterministic**: Identical field values always produce the same hash
2. **Serialization-Dependent**: Changes to `to_dict()` logic may change hash values
3. **Immutability Required**: Objects are frozen to prevent hash corruption

**Hash Stability Warning**: If you modify serialization logic (e.g., change field names, add/remove fields from serialization), previously cached objects with old hashes may become invalid. This is a design trade-off for content-based hashing.

## Protocol Implementations

HashableModel implements three core protocols:

### Serializable

**Methods**:

- `to_dict(mode='python'|'json', **kwargs)`: Dictionary serialization with two modes
- `to_json(decode=True, **kwargs)`: JSON string serialization with sorted keys

**Key Feature**: Deterministic serialization with sorted keys for stable hashing.

### Deserializable

**Methods**:

- `from_dict(data, mode='python'|'json', **kwargs)`: Deserialization from dict or JSON
- `from_json(data, mode='json', **kwargs)`: Deserialization from JSON string/bytes

**Note**: Does NOT support polymorphic deserialization (no `lion_class` metadata).

### Hashable

**Method**: `__hash__()` based on content (field values)

**Equality**: Inherited from Pydantic BaseModel (compares all fields)

**Usage**: Safe for use in sets and as dict keys due to frozen configuration.

<!-- TODO: Add link to Protocols Guide when available (issue #115) -->

## Usage Patterns

### Basic Usage

```python
from lionherd_core.types import HashableModel

class APIConfig(HashableModel):
    endpoint: str
    api_key: str
    timeout: int = 30

# Create immutable config
config = APIConfig(
    endpoint="https://api.example.com",
    api_key="secret",
    timeout=60
)

# Access fields
print(config.endpoint)  # "https://api.example.com"
print(config.timeout)   # 60

# Frozen - cannot modify
# config.timeout = 100  # ValidationError!

# Can use as cache key
cache = {config: "cached_result"}
```

### Cache Keys with Content-Based Deduplication

```python
from lionherd_core.types import HashableModel

class LLMRequest(HashableModel):
    model: str
    prompt: str
    temperature: float
    max_tokens: int

# Create cache
cache: dict[LLMRequest, str] = {}

# First request
req1 = LLMRequest(
    model="gpt-4",
    prompt="Hello",
    temperature=0.7,
    max_tokens=100
)
cache[req1] = "Response 1"

# Identical request (different instance)
req2 = LLMRequest(
    model="gpt-4",
    prompt="Hello",
    temperature=0.7,
    max_tokens=100
)

# Same hash = cache hit
print(cache[req2])  # "Response 1" (same content)
```

### Structured LLM Outputs with Deduplication

```python
from lionherd_core.types import HashableModel
from lionherd_core.ln import to_list

class Entity(HashableModel):
    name: str
    type: str
    confidence: float

# LLM might return duplicates
entities = [
    Entity(name="Apple", type="company", confidence=0.9),
    Entity(name="Google", type="company", confidence=0.85),
    Entity(name="Apple", type="company", confidence=0.9),  # Duplicate
    Entity(name="Microsoft", type="company", confidence=0.92),
]

# Deduplicate by content
unique_entities = to_list(entities, unique=True)
print(len(unique_entities))  # 3 (Apple deduplicated)
```

### Set Operations with Content-Based Equality

```python
from lionherd_core.types import HashableModel

class Tag(HashableModel):
    name: str
    category: str

# Create tags
tags1 = {
    Tag(name="python", category="language"),
    Tag(name="api", category="type"),
    Tag(name="python", category="language"),  # Duplicate
}

tags2 = {
    Tag(name="python", category="language"),
    Tag(name="rest", category="type"),
}

# Set operations work based on content
print(len(tags1))  # 2 (duplicate removed)
print(tags1 & tags2)  # {Tag(name="python", category="language")}
print(tags1 | tags2)  # All unique tags (3 total)
```

### Serialization for Storage

```python
from lionherd_core.types import HashableModel

class UserPreferences(HashableModel):
    theme: str
    language: str
    notifications: bool

prefs = UserPreferences(
    theme="dark",
    language="en",
    notifications=True
)

# Serialize to JSON (deterministic, sorted keys)
json_str = prefs.to_json()
# '{"language":"en","notifications":true,"theme":"dark"}'

# Store to file
# with open("prefs.json", "w") as f:
#     f.write(json_str)

# Load from file
# with open("prefs.json") as f:
#     restored = UserPreferences.from_json(f.read())
```

### Common Pitfalls

#### Pitfall 1: Attempting to Modify Frozen Fields

**Issue**: Trying to modify fields after creation.

```python
from pydantic import ValidationError
from lionherd_core.types import HashableModel

class Config(HashableModel):
    value: int

config = Config(value=10)
# config.value = 20  # ValidationError: "Config" is frozen
```

**Solution**: Create a new instance instead of modifying:

```python
new_config = Config(value=20)
# Or use model_copy with updates (Pydantic method)
new_config = config.model_copy(update={"value": 20})
```

#### Pitfall 2: Hash Invalidation After Serialization Changes

**Issue**: Changing serialization logic invalidates cached hashes.

```python
# Version 1: Initial implementation
class CacheKey(HashableModel):
    model: str
    temperature: float

key = CacheKey(model="gpt-4", temperature=0.7)
cache = {key: "result"}

# Version 2: Added field to serialization
class CacheKey(HashableModel):
    model: str
    temperature: float
    version: int = 1  # New field

# Old cached keys may not match new keys with same model/temperature
new_key = CacheKey(model="gpt-4", temperature=0.7, version=1)
# hash(key) != hash(new_key) - cache miss!
```

**Solution**: Version your cache or clear cache on schema changes.

#### Pitfall 3: Using HashableModel for Workflow Entities

**Issue**: Using content-based hashing for entities that need identity tracking.

```python
# WRONG: Agent should have identity, not content-based equality
class Agent(HashableModel):
    name: str
    status: str

agent1 = Agent(name="Agent-1", status="active")
agent2 = Agent(name="Agent-1", status="active")

# Same content = same hash (probably NOT what you want for agents)
print(agent1 == agent2)  # True (but they're different agents!)
```

**Solution**: Use Element for workflow entities with identity:

```python
from lionherd_core import Element

class Agent(Element):
    name: str
    status: str

agent1 = Agent(name="Agent-1", status="active")
agent2 = Agent(name="Agent-1", status="active")

# Different IDs = different agents (correct)
print(agent1 == agent2)  # False (different UUIDs)
```

#### Pitfall 4: Expecting Polymorphic Deserialization

**Issue**: HashableModel doesn't support polymorphic deserialization.

```python
class BaseConfig(HashableModel):
    type: str

class APIConfig(BaseConfig):
    endpoint: str

api_config = APIConfig(type="api", endpoint="https://api.example.com")
data = api_config.to_dict()

# Deserializes as BaseConfig, not APIConfig
restored = BaseConfig.from_dict(data)
print(type(restored).__name__)  # "BaseConfig" (not "APIConfig")
```

**Solution**: Use [Element](../base/element.md) for polymorphic workflows, or manually track type information.

## Design Rationale

### Why Content-Based Hashing?

HashableModel provides **content-based hashing** (same fields = same hash) rather than identity-based hashing because:

1. **Cache Keys**: Cache lookups need value equality (identical config should hit cache)
2. **Deduplication**: LLM outputs often contain duplicates that should be deduplicated by content
3. **Configuration Objects**: Configs are immutable values where content matters, not identity
4. **Set Operations**: Mathematical set operations (union, intersection) work on values, not identities

For **workflow entities** where identity matters (agents, nodes, sessions), use [Element](../base/element.md) with ID-based hashing.

### Why Frozen by Default?

Freezing fields (`model_config(frozen=True)`) ensures:

1. **Hash Stability**: Once an object is used as a dict key or added to a set, its hash cannot change
2. **Cache Safety**: Cached objects cannot be mutated after caching, preventing corruption
3. **Immutability Semantics**: Configurations and value objects are conceptually immutable
4. **Thread Safety**: Immutable objects are inherently thread-safe for concurrent access

This trade-off sacrifices mutability for correctness in hashing and caching scenarios.

### Why Deterministic JSON Serialization?

HashableModel sorts JSON keys (`sort_keys=True`) to ensure **deterministic hashing**:

1. **Hash Stability**: Identical objects always produce identical JSON, regardless of field insertion order
2. **Cache Consistency**: Same content always generates same hash, improving cache hit rates
3. **Reproducibility**: Serialized output is consistent across runs and environments

The performance cost of sorting is negligible compared to the correctness benefits.

### Why No Polymorphic Deserialization?

HashableModel does NOT inject `lion_class` metadata or support polymorphic deserialization because:

1. **Simplicity**: Value objects rarely need polymorphism (they're data, not entities)
2. **Hash Purity**: Injecting metadata would affect hash computation
3. **Use Case Mismatch**: Polymorphism is for workflow entities (Element), not configs (HashableModel)

If you need polymorphism, use [Element](../base/element.md) instead.

### Why No Database Mode?

HashableModel does NOT support `mode='db'` serialization because:

1. **Database entities need identity**: Use Element with UUID for database storage
2. **No metadata field**: HashableModel lacks the metadata field required for `node_metadata` renaming
3. **Value objects are transient**: Configs and cache keys are typically not persisted to databases

For database persistence, use [Element](../base/element.md) or subclass it.

## See Also

- **Related Classes**:
  - Element: Identity-based hashing alternative for workflow entities
  - Node: Element subclass with relationship edges
  - Event: Element subclass with async execution lifecycle
<!-- TODO: Add links to user guides when available (issue #115) -->

## Examples

### Example 1: LLM Configuration Cache

```python
from lionherd_core.types import HashableModel

class LLMConfig(HashableModel):
    model: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0

# Cache responses by config
response_cache: dict[LLMConfig, str] = {}

# First call
config1 = LLMConfig(model="gpt-4", temperature=0.7, max_tokens=100)
response_cache[config1] = "Generated response..."

# Second call with identical config (different instance)
config2 = LLMConfig(model="gpt-4", temperature=0.7, max_tokens=100)

# Cache hit (same content)
if config2 in response_cache:
    print("Cache hit!")
    print(response_cache[config2])  # "Generated response..."
```

### Example 2: Deduplicating Entity Extractions

```python
from lionherd_core.types import HashableModel
from lionherd_core.ln import to_list

class ExtractedEntity(HashableModel):
    text: str
    label: str
    start: int
    end: int

# LLM might extract duplicates from multiple passes
extractions = [
    ExtractedEntity(text="Apple Inc.", label="ORG", start=0, end=10),
    ExtractedEntity(text="California", label="LOC", start=20, end=30),
    ExtractedEntity(text="Apple Inc.", label="ORG", start=0, end=10),  # Duplicate
    ExtractedEntity(text="Tim Cook", label="PER", start=40, end=48),
    ExtractedEntity(text="California", label="LOC", start=20, end=30),  # Duplicate
]

# Deduplicate by content
unique_extractions = to_list(extractions, unique=True)

print(f"Total extractions: {len(extractions)}")  # 5
print(f"Unique extractions: {len(unique_extractions)}")  # 3
```

### Example 3: Configuration Versioning

```python
from lionherd_core.types import HashableModel
from typing import Literal

class AppConfig(HashableModel):
    environment: Literal["dev", "staging", "prod"]
    debug: bool
    log_level: str
    max_workers: int

# Development config
dev_config = AppConfig(
    environment="dev",
    debug=True,
    log_level="DEBUG",
    max_workers=4
)

# Production config
prod_config = AppConfig(
    environment="prod",
    debug=False,
    log_level="INFO",
    max_workers=16
)

# Store configs by hash
configs = {
    hash(dev_config): dev_config,
    hash(prod_config): prod_config,
}

# Lookup by content
lookup_config = AppConfig(
    environment="dev",
    debug=True,
    log_level="DEBUG",
    max_workers=4
)

current = configs.get(hash(lookup_config))
print(f"Environment: {current.environment}")  # "dev"
print(f"Debug mode: {current.debug}")  # True
```

### Example 4: Set Operations on Tags

```python
from lionherd_core.types import HashableModel

class Tag(HashableModel):
    name: str
    category: str

# Project A tags
project_a = {
    Tag(name="python", category="language"),
    Tag(name="fastapi", category="framework"),
    Tag(name="postgres", category="database"),
}

# Project B tags
project_b = {
    Tag(name="python", category="language"),
    Tag(name="django", category="framework"),
    Tag(name="postgres", category="database"),
}

# Common tags (intersection)
common = project_a & project_b
print("Common tags:")
for tag in common:
    print(f"  {tag.name} ({tag.category})")
# Output:
#   python (language)
#   postgres (database)

# All tags (union)
all_tags = project_a | project_b
print(f"Total unique tags: {len(all_tags)}")  # 4

# Tags unique to project A (difference)
unique_to_a = project_a - project_b
print("Unique to A:")
for tag in unique_to_a:
    print(f"  {tag.name} ({tag.category})")
# Output:
#   fastapi (framework)
```

### Example 5: Deterministic Serialization

```python
from lionherd_core.types import HashableModel

class Person(HashableModel):
    name: str
    age: int
    email: str

# Create instance
person = Person(name="Alice", age=30, email="alice@example.com")

# JSON is always sorted, regardless of field definition order
json1 = person.to_json()
print(json1)
# '{"age":30,"email":"alice@example.com","name":"Alice"}'

# Hash is stable across runs
hash1 = hash(person)

# Recreate from dict
data = {"email": "alice@example.com", "name": "Alice", "age": 30}
person2 = Person.from_dict(data)

json2 = person2.to_json()
hash2 = hash(person2)

# Deterministic serialization = stable hashing
assert json1 == json2
assert hash1 == hash2
print("Hashes match! ✓")
```

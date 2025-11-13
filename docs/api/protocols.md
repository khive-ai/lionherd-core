# Protocols

> Structural typing interfaces for protocol-based composition in lionherd-core

## Overview

The `protocols` module provides **runtime-checkable protocol definitions** that enable structural typing (duck typing with explicit declarations) throughout lionherd-core. This approach follows **Rust traits** and **Go interfaces** philosophy: composition over inheritance, loose coupling, and explicit capability declarations.

**Key Components:**

- **Core Protocols**: Observable, Serializable, Deserializable, Adaptable, AsyncAdaptable
- **Container Protocols**: Containable, Invocable, Hashable, Allowable
- **@implements() Decorator**: Explicit protocol implementation declaration

**Why Protocol-Based Composition?**

Traditional class inheritance creates tight coupling and multiple inheritance complexity. Protocols provide:

1. **Structural Typing**: Objects are compatible if they implement required methods, not if they inherit from specific base classes
2. **Loose Coupling**: No shared base class requirements, components compose freely
3. **Explicit Contracts**: `@implements()` declares capabilities in class definition
4. **Runtime Checking**: `isinstance(obj, Protocol)` validates structural compatibility

**When to Use Protocols:**

- Building composable components without inheritance hierarchies
- Declaring capabilities independent of implementation
- Enabling polymorphism without shared base classes
- Creating adapters and plugins with explicit contracts

## Module Contents

```python
from lionherd_core.protocols import (
    # Core protocols
    Observable,
    Serializable,
    Deserializable,
    Adaptable,
    AsyncAdaptable,
    # Container protocols
    Containable,
    Invocable,
    Hashable,
    Allowable,
    # Decorator
    implements,
)
```

---

## Core Protocols

### Observable

**Purpose**: Objects with unique UUID identifier for tracking and identity-based operations.

#### Protocol Definition

```python
from lionherd_core.protocols import Observable

@runtime_checkable
class Observable(Protocol):
    """Objects with unique UUID identifier. Check via isinstance()."""

    @property
    def id(self) -> UUID:
        """Unique identifier."""
        ...
```

#### Required Interface

- **Property**: `id` → `UUID`
  - Must return a UUID object (not string)
  - Should be unique across all instances
  - Typically immutable (frozen field)

#### Usage Pattern

```python
from uuid import uuid4, UUID
from lionherd_core.protocols import Observable, implements

@implements(Observable)
class Agent:
    def __init__(self):
        self._id = uuid4()

    @property
    def id(self) -> UUID:
        return self._id

# Runtime protocol checking
agent = Agent()
assert isinstance(agent, Observable)
```

#### Design Rationale

UUID-based identity enables:

- **Stable references**: Track objects across serialization/deserialization
- **Graph relationships**: Nodes/edges referenced by UUID, not object pointers
- **Distributed systems**: Globally unique identifiers without coordination
- **Caching**: Dict/set membership based on stable identity

#### See Also

- [Element](base/element.md): Base class implementing Observable with frozen UUID
- [Node](base/node.md): Observable entity with polymorphic content

---

### Serializable

**Purpose**: Objects that can be serialized to dictionary representation.

#### Protocol Definition

```python
from lionherd_core.protocols import Serializable

@runtime_checkable
class Serializable(Protocol):
    """Objects that can be serialized to dict via to_dict()."""

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict. Args: serialization options (mode, format, etc.)."""
        ...
```

#### Required Interface

- **Method**: `to_dict(**kwargs) -> dict[str, Any]`
  - Must return JSON-compatible dictionary
  - `**kwargs` for serialization options (mode, format, exclude, etc.)
  - Should handle nested Serializable objects

#### Usage Pattern

```python
from typing import Any
from lionherd_core.protocols import Serializable, implements

@implements(Serializable)
class Config:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def to_dict(self, **kwargs) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}

# Serialization
config = Config("localhost", 8080)
data = config.to_dict()  # {'host': 'localhost', 'port': 8080}
```

#### Common Kwargs

Implementations typically support:

- `mode`: Serialization mode ('python', 'json', 'db')
- `exclude`: Fields to exclude from output
- `include`: Fields to include (if subset needed)
- Custom format options (timestamps, enums, etc.)

#### Design Rationale

`to_dict()` as standard interface enables:

- **Consistent API**: All serializable objects use same method name
- **Flexible options**: `**kwargs` allows mode-specific customization
- **Composability**: Nested objects serialize recursively
- **Adapter integration**: Uniform interface for format conversion

#### See Also

- [Element.to_dict()](base/element.md#to_dict): Multi-mode serialization implementation
- [Pile.to_dict()](base/pile.md#to_dict): Collection serialization with progression preservation

---

### Deserializable

**Purpose**: Objects that can be deserialized from dictionary representation via classmethod.

#### Protocol Definition

```python
from lionherd_core.protocols import Deserializable

@runtime_checkable
class Deserializable(Protocol):
    """Objects that can be deserialized from dict via from_dict() classmethod."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> Any:
        """Deserialize from dict. Args: data dict, deserialization options."""
        ...
```

#### Required Interface

- **Classmethod**: `from_dict(data, **kwargs) -> Self`
  - Must accept dictionary from `to_dict()`
  - Returns instance of calling class (or correct subclass for polymorphism)
  - `**kwargs` for deserialization options

#### Usage Pattern

```python
from typing import Any
from lionherd_core.protocols import Deserializable, implements

@implements(Deserializable)
class Config:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs) -> "Config":
        return cls(host=data["host"], port=data["port"])

# Deserialization
data = {"host": "localhost", "port": 8080}
config = Config.from_dict(data)
```

#### Polymorphic Deserialization

Advanced pattern: Use metadata to reconstruct correct subclass.

```python
from typing import Any

@implements(Deserializable)
class Base:
    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs) -> "Base":
        # Check for polymorphic class hint
        if "class_type" in data:
            target_class = resolve_class(data["class_type"])
            return target_class.from_dict(data, **kwargs)
        return cls(**data)
```

#### Design Rationale

Classmethod pattern enables:

- **Factory method**: Construct instances from external data
- **Polymorphism**: Return correct subclass based on metadata
- **Validation**: Parse and validate input before construction
- **Roundtrip guarantee**: `from_dict(obj.to_dict()) == obj`

#### See Also

- [Element.from_dict()](base/element.md#from_dict): Polymorphic deserialization via `lion_class`
- [Serializable](#serializable): Companion protocol for roundtrip serialization

---

### Adaptable

**Purpose**: Synchronous adapter protocol for format conversion (TOML, YAML, JSON, SQL).

#### Protocol Definition

```python
from lionherd_core.protocols import Adaptable

@runtime_checkable
class Adaptable(Protocol):
    """Sync adapter protocol for format conversion (TOML/YAML/JSON/SQL). Use AsyncAdaptable for async."""

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Convert to external format. Args: adapter key, many flag, adapter kwargs."""
        ...

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Create from external format. Args: source object, adapter key, many flag."""
        ...

    @classmethod
    def register_adapter(cls, adapter: Any) -> None:
        """Register adapter for this class."""
        ...
```

#### Required Interface

- **Method**: `adapt_to(obj_key, many=False, **kwargs) -> Any`
  - Convert instance to external format
  - `obj_key`: Adapter identifier (e.g., "json", "yaml", "sql")
  - `many`: Single object vs collection flag
- **Classmethod**: `adapt_from(obj, obj_key, many=False, **kwargs) -> Self`
  - Create instance from external format
  - `obj`: Source data in external format
- **Classmethod**: `register_adapter(adapter) -> None`
  - Register adapter for format conversion
  - Per-class adapter registry (isolated state)

#### Usage Pattern

```python
from lionherd_core.protocols import Adaptable, implements
from pydapter import Adapter, to_json, from_json

@implements(Adaptable)
class Node:
    def __init__(self, label: str):
        self.label = label

    @classmethod
    def register_adapter(cls, adapter):
        cls._adapter = adapter

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs):
        return self._adapter.to(self, obj_key, many=many, **kwargs)

    @classmethod
    def adapt_from(cls, obj, obj_key: str, many: bool = False, **kwargs):
        return cls._adapter.from_(obj, obj_key, many=many, **kwargs)

# Register adapter
Node.register_adapter(Adapter(to_json=to_json(), from_json=from_json()))

# Convert to JSON
node = Node(label="agent_1")
json_data = node.adapt_to("json")

# Convert from JSON
restored = Node.adapt_from(json_data, "json")
```

#### Adapter Isolation

Each class maintains its own adapter registry:

```python
# Node adapter doesn't affect Event adapter
Node.register_adapter(node_adapter)
Event.register_adapter(event_adapter)

# Adapters are isolated
node.adapt_to("json")  # Uses node_adapter
event.adapt_to("json")  # Uses event_adapter
```

#### Design Rationale

Adapter protocol enables:

- **Format flexibility**: Single interface for multiple formats
- **Lazy loading**: Register adapters only when needed
- **Isolation**: Per-class registries prevent adapter pollution
- **Extensibility**: Add new formats without modifying core classes

#### Supported Classes

- [Node](base/node.md): Polymorphic content with adapter support
- [Pile](base/pile.md): Type-safe collections with adapter integration
- [Graph](base/graph.md): Directed graphs with format conversion

#### See Also

- [AsyncAdaptable](#asyncadaptable): Async version for I/O-bound conversion
- [pydapter documentation](https://github.com/khive-ai/pydapter): Adapter library

---

### AsyncAdaptable

**Purpose**: Async adapter protocol for I/O-bound format conversion (databases, network, files).

#### Protocol Definition

```python
from lionherd_core.protocols import AsyncAdaptable

@runtime_checkable
class AsyncAdaptable(Protocol):
    """Async adapter protocol for I/O-bound format conversion (DBs, network, files). Use Adaptable for sync."""

    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Async convert to external format. Args: adapter key, many flag, adapter kwargs."""
        ...

    @classmethod
    async def adapt_from_async(
        cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any
    ) -> Any:
        """Async create from external format. Args: source object, adapter key, many flag."""
        ...

    @classmethod
    def register_async_adapter(cls, adapter: Any) -> None:
        """Register async adapter for this class."""
        ...
```

#### Required Interface

- **Async Method**: `adapt_to_async(obj_key, many=False, **kwargs) -> Any`
  - Async convert to external format (non-blocking I/O)
- **Async Classmethod**: `adapt_from_async(obj, obj_key, many=False, **kwargs) -> Self`
  - Async create from external format
- **Classmethod**: `register_async_adapter(adapter) -> None`
  - Register async adapter (separate from sync adapters)

#### Usage Pattern

```python
# noqa:validation
from lionherd_core.protocols import AsyncAdaptable, implements

@implements(AsyncAdaptable)
class Node:
    @classmethod
    def register_async_adapter(cls, adapter):
        cls._async_adapter = adapter

    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs):
        return await self._async_adapter.to_async(self, obj_key, many=many, **kwargs)

    @classmethod
    async def adapt_from_async(cls, obj, obj_key: str, many: bool = False, **kwargs):
        return await cls._async_adapter.from_async(obj, obj_key, many=many, **kwargs)

# Register async adapter
Node.register_async_adapter(async_postgres_adapter)

# Async database operations
node = Node(label="agent_1")
await node.adapt_to_async("postgres")  # Non-blocking DB write
restored = await Node.adapt_from_async(db_row, "postgres")  # Non-blocking DB read
```

#### When to Use Async vs Sync

- **Adaptable (sync)**: In-memory formats (JSON, YAML, TOML), local files
- **AsyncAdaptable (async)**: Databases, network APIs, remote storage

#### Design Rationale

Separate async protocol enables:

- **Non-blocking I/O**: Database/network operations don't block event loop
- **Concurrency**: Batch operations run in parallel
- **Clear interface**: Explicit async/await at call site
- **Adapter isolation**: Sync and async adapters coexist without conflicts

#### See Also

- [Adaptable](#adaptable): Synchronous adapter protocol
- [anyio documentation](https://anyio.readthedocs.io/en/stable/): Async I/O framework

---

## Container Protocols

### Containable

**Purpose**: Objects supporting membership testing via `in` operator (`__contains__`).

#### Protocol Definition

```python
from lionherd_core.protocols import Containable

@runtime_checkable
class Containable(Protocol):
    """Objects that support membership testing via 'in' operator (__contains__)."""

    def __contains__(self, item: Any) -> bool:
        """Check if item is in collection (by UUID or instance)."""
        ...
```

#### Required Interface

- **Method**: `__contains__(item) -> bool`
  - Return True if `item` is in collection
  - Typically supports both UUID lookup and instance lookup

#### Usage Pattern

```python
from uuid import UUID, uuid4
from lionherd_core.protocols import Containable, implements

@implements(Containable)
class Pile:
    def __init__(self):
        self.items = {}

    def __contains__(self, item) -> bool:
        # Support both UUID and instance lookup
        if isinstance(item, UUID):
            return item in self.items
        return item.id in self.items if hasattr(item, 'id') else False

# Membership testing
pile = Pile()
elem = Element()
pile.add(elem)

assert elem in pile          # Instance lookup
assert elem.id in pile       # UUID lookup
assert uuid4() not in pile   # Not found
```

#### Design Rationale

Pythonic `in` operator enables:

- **Ergonomic queries**: `elem in pile` vs `pile.contains(elem)`
- **Type flexibility**: Support UUID, instance, or both
- **Standard protocol**: Consistent with built-in containers

#### See Also

- [Pile.**contains**](base/pile.md#__contains__): Collection membership with UUID/instance support

---

### Invocable

**Purpose**: Objects that can be invoked/executed via async `invoke()` method.

#### Protocol Definition

```python
from lionherd_core.protocols import Invocable

@runtime_checkable
class Invocable(Protocol):
    """Objects that can be invoked/executed via async invoke() method."""

    async def invoke(self) -> Any:
        """Invoke/execute the object. Returns: execution result (any value or None)."""
        ...
```

#### Required Interface

- **Async Method**: `invoke() -> Any`
  - Execute object's primary action
  - Returns execution result (or None)
  - Async to support non-blocking execution

#### Usage Pattern

```python
# noqa:validation
from lionherd_core.protocols import Invocable, implements

@implements(Invocable)
class Task:
    def __init__(self, action):
        self.action = action

    async def invoke(self):
        result = await self.action()
        return result

# Execution
task = Task(action=lambda: fetch_data())
result = await task.invoke()
```

#### Design Rationale

Invocable protocol enables:

- **Uniform execution**: All invocable objects use `invoke()`
- **Async-first**: Non-blocking execution by default
- **Polymorphism**: Execute heterogeneous collections uniformly

#### See Also

- [Event.invoke](base/event.md#invoke): Async event execution

---

### Hashable

**Purpose**: Objects that can be hashed via `__hash__()` for use in sets/dicts.

#### Protocol Definition

```python
from lionherd_core.protocols import Hashable

@runtime_checkable
class Hashable(Protocol):
    """Objects that can be hashed via __hash__() for use in sets/dicts."""

    def __hash__(self) -> int:
        """Return hash value for object (must be immutable or ID-based)."""
        ...
```

#### Required Interface

- **Method**: `__hash__() -> int`
  - Return stable hash value
  - Hash must not change during object lifetime if used in dict/set
  - Implement `__eq__()` consistently (equal objects → equal hashes)

#### Two Hashing Patterns

**Identity-Based Hashing** (Element pattern):

```python
from uuid import uuid4

@implements(Hashable)
class Element:
    def __init__(self):
        self.id = uuid4()
        self.metadata = {}  # Mutable, but not hashed

    def __hash__(self) -> int:
        return hash(self.id)  # Hash by ID only

    def __eq__(self, other):
        return isinstance(other, Element) and self.id == other.id
```

**Content-Based Hashing** (Params pattern):

```python
@implements(Hashable)
class Params:
    def __init__(self, **kwargs):
        self._data = kwargs  # Frozen dataclass in practice

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._data.items())))

    def __eq__(self, other):
        return isinstance(other, Params) and self._data == other._data
```

#### Design Rationale

Hashable protocol enables:

- **Set membership**: Deduplicate by identity or content
- **Dict keys**: Use objects as lookup keys
- **Performance**: O(1) membership testing

#### See Also

- [Element.**hash**](base/element.md#__hash__): Identity-based hashing
- [Params.**hash**](types/base.md#__hash__): Content-based hashing

---

### Allowable

**Purpose**: Objects with defined set of allowed values/keys via `allowed()`.

#### Protocol Definition

```python
from lionherd_core.protocols import Allowable

@runtime_checkable
class Allowable(Protocol):
    """Objects with defined set of allowed values/keys via allowed()."""

    def allowed(self) -> set[str]:
        """Return set of allowed keys/values."""
        ...
```

#### Required Interface

- **Method**: `allowed() -> set[str]` (typically classmethod)
  - Return set of allowed string values
  - Used for validation, enumeration, schema extraction

#### Usage Pattern

**Enum Example**:

```python
from lionherd_core.protocols import Allowable, implements
from enum import StrEnum

@implements(Allowable)
class Status(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"

    @classmethod
    def allowed(cls) -> set[str]:
        return {member.value for member in cls}

# Validation
assert "active" in Status.allowed()
assert "invalid" not in Status.allowed()
```

**Params Example**:

```python
@implements(Allowable)
class APIParams:
    @classmethod
    def allowed(cls) -> set[str]:
        # Return set of allowed parameter names
        return {"endpoint", "method", "headers", "timeout"}

# Validation
def validate_params(params: dict):
    invalid = set(params.keys()) - APIParams.allowed()
    if invalid:
        raise ValueError(f"Invalid params: {invalid}")
```

#### Design Rationale

Allowable protocol enables:

- **Validation**: Check input against allowed set
- **Introspection**: Discover available options at runtime
- **Schema generation**: Extract allowed values for documentation

#### See Also

- [Enum.allowed()](types/base.md#allowed): Enum value extraction
- [Params.allowed()](types/base.md#allowed): Parameter name extraction

---

## @implements() Decorator

**Purpose**: Explicitly declare protocol implementations with strict enforcement (Rust-like trait implementation).

### Decorator Signature

```python
def implements(*protocols: type):
    """Declare protocol implementations (Rust-like: MUST define in class body).

    CRITICAL SEMANTICS (strictest interpretation):
        @implements() means the class **LITERALLY** implements/overrides the method
        or declares the attribute IN ITS OWN CLASS BODY. Inheritance does NOT count.

        This is Rust-like trait implementation: you must provide the implementation
        in the impl block, not rely on inheritance.

    Rules:
        - Method must be defined in class body (even if it calls super())
        - Property must be declared in class body (cannot inherit from parent)
        - Classmethod must be defined in class body
        - NO inheritance: @implements means "I define this, not my parent"

    Args:
        *protocols: Protocol classes that the decorated class **literally** implements

    Returns:
        Class decorator that stores protocols on cls.__protocols__
    """
```

### Critical Rules

#### Rule 1: Literal Implementation Required

**✅ CORRECT**: Method defined in class body

```python
@implements(Serializable)
class MyClass:
    def to_dict(self, **kwargs):  # Defined in this class
        return {"data": "value"}
```

**❌ WRONG**: Relying on inheritance

```python
class Parent:
    def to_dict(self, **kwargs):
        return {"parent": "data"}

@implements(Serializable)  # VIOLATION!
class Child(Parent):
    pass  # No to_dict in Child body → not allowed!
```

#### Rule 2: Explicit Override for Inherited Methods

**✅ CORRECT**: Explicit override even if calling super()

```python
class Parent:
    def to_dict(self, **kwargs):
        return {"parent": "data"}

@implements(Serializable)
class Child(Parent):
    def to_dict(self, **kwargs):  # Explicit in Child body
        return super().to_dict(**kwargs)  # Can call parent
```

#### Rule 3: Properties Must Be Declared

**❌ WRONG**: Inherited property

```python
class Parent:
    @property
    def id(self):
        return self._id

@implements(Observable)  # VIOLATION!
class Child(Parent):
    pass  # id inherited, not declared
```

**✅ CORRECT**: Property declared in class body

```python
@implements(Observable)
class Child(Parent):
    @property
    def id(self):  # Explicit declaration
        return super().id
```

### Design Rationale

The `@implements()` decorator enforces **explicit over implicit** (Rust philosophy):

1. **Clear Ownership**: Each class declares what it implements
2. **No Ambiguity**: Implementation location is always class body
3. **Prevents Accidents**: Can't accidentally claim protocols through inheritance
4. **Documentation**: `@implements()` serves as inline documentation

From test design insights (lines 132-175 in protocols.py):

> "This is Rust-like trait implementation: you must provide the implementation
> in the impl block, not rely on inheritance."

### Multiple Protocols

```python
@implements(Observable, Serializable, Deserializable)
class Element:
    @property
    def id(self) -> UUID:
        return self._id

    def to_dict(self, **kwargs) -> dict:
        return {"id": str(self.id)}

    @classmethod
    def from_dict(cls, data: dict, **kwargs):
        return cls(id=UUID(data["id"]))
```

### Runtime Checking

Protocols are runtime-checkable via `isinstance()`:

```python
@implements(Observable)
class Agent:
    @property
    def id(self):
        return uuid4()

agent = Agent()
assert isinstance(agent, Observable)  # True (structural match)

# Accessing stored protocols
assert Observable in agent.__class__.__protocols__
```

### Common Pitfalls

#### Pitfall 1: Forgetting to Override

```python
# ❌ WRONG
class Parent:
    def to_dict(self):
        return {}

@implements(Serializable)  # Violation!
class Child(Parent):
    pass  # Inherited to_dict doesn't count
```

#### Pitfall 2: Using @implements() Without Implementation

```python
# ❌ WRONG
@implements(Serializable)  # Violation!
class MyClass:
    pass  # No to_dict method at all
```

#### Pitfall 3: Confusing with Inheritance

```python
# ❌ WRONG (conceptual error)
@implements(Serializable)  # This is NOT inheritance!
class MyClass(Serializable):  # Protocols aren't meant for inheritance
    pass
```

**Correct pattern**: Don't inherit from protocols, implement their methods.

---

## Design Patterns

### Pattern 1: Protocol Composition

Combine multiple protocols for rich capabilities:

```python
from uuid import uuid4, UUID
from lionherd_core.protocols import Observable, Serializable, Hashable, implements

@implements(Observable, Serializable, Hashable)
class Agent:
    def __init__(self, name: str):
        self._id = uuid4()
        self.name = name

    @property
    def id(self) -> UUID:
        return self._id

    def to_dict(self, **kwargs) -> dict:
        return {"id": str(self.id), "name": self.name}

    def __hash__(self) -> int:
        return hash(self.id)

# Agent now supports:
# - Identity tracking (Observable)
# - Serialization (Serializable)
# - Set/dict membership (Hashable)
```

### Pattern 2: Adapter Integration

Protocols enable clean adapter pattern:

```python
@implements(Adaptable, Serializable)
class Node:
    @classmethod
    def register_adapter(cls, adapter):
        cls._adapter = adapter

    def adapt_to(self, obj_key: str, **kwargs):
        # Delegate to adapter
        return self._adapter.to(self, obj_key, **kwargs)

    def to_dict(self, **kwargs):
        # Direct serialization (no adapter)
        return {"label": self.label}

# Use adapter OR direct serialization
node = Node(label="agent")
json_via_adapter = node.adapt_to("json")
dict_direct = node.to_dict()
```

### Pattern 3: Polymorphic Collections

Protocols enable type-safe polymorphism:

```python
from lionherd_core.protocols import Serializable

def serialize_all(items: list[Serializable]) -> list[dict]:
    """Serialize heterogeneous collection."""
    return [item.to_dict() for item in items]

# Works with any Serializable objects
nodes = [Node(...), Element(...), Event(...)]
serialized = serialize_all(nodes)
```

### Pattern 4: Runtime Protocol Checking

Validate objects implement required protocols:

```python
def process_observable(obj):
    if not isinstance(obj, Observable):
        raise TypeError(f"{obj} must implement Observable")
    return obj.id

# Type-safe processing
agent = Agent()
process_observable(agent)  # OK

process_observable("string")  # TypeError
```

---

## Common Pitfalls

### Pitfall 1: Don't Inherit from Protocols

**Issue**: Using protocols as base classes.

```python
# ❌ WRONG
class MyClass(Observable, Serializable):  # Protocols aren't base classes!
    pass
```

**Solution**: Implement protocol methods, use @implements():

```python
# ✅ CORRECT
@implements(Observable, Serializable)
class MyClass:
    @property
    def id(self):
        return self._id

    def to_dict(self, **kwargs):
        return {"id": str(self.id)}
```

### Pitfall 2: Missing @implements() for Inherited Methods

**Issue**: Assuming inherited methods count for @implements().

```python
# ❌ WRONG
class Parent:
    def to_dict(self):
        return {}

@implements(Serializable)  # Violation!
class Child(Parent):
    pass  # to_dict inherited, not implemented in body
```

**Solution**: Explicitly override in class body:

```python
# ✅ CORRECT
@implements(Serializable)
class Child(Parent):
    def to_dict(self, **kwargs):  # Explicit override
        return super().to_dict(**kwargs)
```

### Pitfall 3: Forgetting Runtime Checkability

**Issue**: Assuming static type checking is enough.

```python
def process(obj: Serializable):  # Type hint, but no runtime check
    return obj.to_dict()

process("string")  # Passes type hint, fails at runtime
```

**Solution**: Use isinstance() for runtime validation:

```python
def process(obj):
    if not isinstance(obj, Serializable):
        raise TypeError("obj must be Serializable")
    return obj.to_dict()
```

### Pitfall 4: Confusing Protocol and Implementation

**Issue**: Treating protocol as complete interface.

```python
# ❌ WRONG (protocol has no implementation)
obj = Serializable()  # Can't instantiate protocols!
```

**Solution**: Implement protocol in concrete class:

```python
# ✅ CORRECT
@implements(Serializable)
class MyClass:
    def to_dict(self, **kwargs):
        return {"data": "value"}

obj = MyClass()
```

---

## Protocol Checklist

**When implementing protocols:**

- ✅ Use `@implements()` decorator to declare protocols
- ✅ Define ALL protocol methods in class body (not via inheritance)
- ✅ Override inherited methods explicitly if using @implements()
- ✅ Implement `__hash__()` and `__eq__()` consistently for Hashable
- ✅ Use `isinstance(obj, Protocol)` for runtime validation
- ✅ Return correct types matching protocol signatures
- ❌ DON'T inherit from protocol classes
- ❌ DON'T rely on inherited methods for @implements()
- ❌ DON'T instantiate protocols directly

**When using protocols:**

- ✅ Check `isinstance(obj, Protocol)` for runtime validation
- ✅ Accept protocol types in function signatures for polymorphism
- ✅ Combine multiple protocols for rich capabilities
- ❌ DON'T assume static type hints provide runtime safety
- ❌ DON'T use protocols as base classes

---

## Design Rationale

### Why Protocols Over Inheritance?

**Traditional Inheritance Problems:**

- Tight coupling between base and derived classes
- Multiple inheritance diamond problem
- Fragile base class problem (base changes break children)
- Deep hierarchies hard to understand and modify

**Protocol Advantages:**

- Loose coupling: No shared base class required
- Structural typing: Compatible if methods match, not if inheritance matches
- Composition: Mix protocols freely without inheritance conflicts
- Explicit: `@implements()` documents capabilities clearly

### Why @implements() Decorator?

From test design insights:

> "Explicit > Implicit (Rust philosophy). Clear ownership: each class declares
> what it implements. No ambiguity about where implementation lives."

The decorator enforces:

1. **Explicit declaration**: Can't accidentally implement protocols
2. **Documentation**: Serves as inline capability documentation
3. **Runtime access**: `cls.__protocols__` lists all declared protocols
4. **Prevents inheritance confusion**: Can't claim protocols via parent class

### Why Runtime-Checkable?

`@runtime_checkable` decorator on protocols enables `isinstance()` checks:

```python
isinstance(obj, Observable)  # True if obj has .id property
```

This bridges static typing (type hints) and runtime validation (isinstance), enabling:

- **Graceful degradation**: Check if object supports optional protocols
- **Plugin systems**: Validate plugin implements required interfaces
- **Error messages**: Clear runtime errors for missing methods

### Why Separate Sync/Async Adapters?

`Adaptable` and `AsyncAdaptable` are separate protocols because:

1. **Different use cases**: Sync for in-memory, async for I/O
2. **Clear interface**: Explicit async/await at call site
3. **Coexistence**: Object can implement both without method name conflicts
4. **Performance**: Sync adapters avoid async overhead when not needed

---

## See Also

- **Base Classes**:
  - [Element](base/element.md): Implements Observable, Serializable, Deserializable, Hashable
  - [Node](base/node.md): Implements all Element protocols plus Adaptable
  - [Pile](base/pile.md): Implements Containable, Serializable, Adaptable
- **Type System**:
  - [Params](types/base.md#params): Implements Serializable, Allowable, Hashable
  - [Enum](types/base.md#enum): Implements Allowable

---

## Examples

### Example 1: Implementing Observable

```python
from uuid import uuid4, UUID
from lionherd_core.protocols import Observable, implements

@implements(Observable)
class Agent:
    def __init__(self, name: str):
        self._id = uuid4()
        self.name = name

    @property
    def id(self) -> UUID:
        return self._id

# Usage
agent = Agent("agent_1")
print(agent.id)  # UUID('...')
assert isinstance(agent, Observable)
```

### Example 2: Implementing Serializable + Deserializable

```python
from typing import Any
from lionherd_core.protocols import Serializable, Deserializable, implements

@implements(Serializable, Deserializable)
class Config:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "Config":
        return cls(host=data["host"], port=data["port"])

# Roundtrip
config = Config("localhost", 8080)
data = config.to_dict()
restored = Config.from_dict(data)
assert restored.host == config.host
```

### Example 3: Protocol Composition

```python
from typing import Any
from uuid import uuid4, UUID
from lionherd_core.protocols import Observable, Serializable, Hashable, implements

@implements(Observable, Serializable, Hashable)
class Agent:
    def __init__(self, name: str):
        self._id = uuid4()
        self.name = name

    @property
    def id(self) -> UUID:
        return self._id

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        return {"id": str(self.id), "name": self.name}

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Agent) and self.id == other.id

# Agent now supports:
agent = Agent("agent_1")

# Observable
agent_id = agent.id

# Serializable
data = agent.to_dict()

# Hashable
agents = {agent, Agent("agent_2")}  # Set membership
lookup = {agent: "result"}  # Dict key
```

### Example 4: Adapter Pattern

```python
from typing import Any
from lionherd_core.protocols import Adaptable, implements
from pydapter import Adapter, to_json, from_json

@implements(Adaptable)
class Node:
    _adapter = None

    def __init__(self, label: str):
        self.label = label

    @classmethod
    def register_adapter(cls, adapter: Any) -> None:
        cls._adapter = adapter

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        return self._adapter.to(self, obj_key, many=many, **kwargs)

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> "Node":
        instance = cls._adapter.from_(obj, obj_key, many=many, **kwargs)
        return instance

# Register adapter
Node.register_adapter(Adapter(to_json=to_json(), from_json=from_json()))

# Use adapter
node = Node(label="agent_1")
json_data = node.adapt_to("json")
restored = Node.adapt_from(json_data, "json")
```

### Example 5: Explicit Override Pattern

```python
from typing import Any
from lionherd_core.protocols import Serializable, implements

class Parent:
    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        return {"parent": "data"}

# ✅ CORRECT: Explicit override
@implements(Serializable)
class Child(Parent):
    def to_dict(self, **kwargs: Any) -> dict[str, Any]:  # Must be in class body
        data = super().to_dict(**kwargs)
        data["child"] = "additional_data"
        return data

child = Child()
print(child.to_dict())
# {'parent': 'data', 'child': 'additional_data'}
```

### Example 6: Runtime Protocol Validation

```python
from lionherd_core.protocols import Observable, Serializable

def process_entity(obj):
    """Process object that must be Observable and Serializable."""
    if not isinstance(obj, Observable):
        raise TypeError(f"{type(obj).__name__} must implement Observable")
    if not isinstance(obj, Serializable):
        raise TypeError(f"{type(obj).__name__} must implement Serializable")

    # Safe to use protocol methods
    entity_id = obj.id
    data = obj.to_dict()
    return entity_id, data

# Usage
agent = Agent("agent_1")
entity_id, data = process_entity(agent)  # OK

# Type error
process_entity("string")  # TypeError: str must implement Observable
```

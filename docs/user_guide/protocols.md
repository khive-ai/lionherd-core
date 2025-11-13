# Protocols Guide

> Protocol-based composition and structural typing in lionherd-core

## Overview

lionherd-core uses **protocol-based composition** (inspired by Rust traits and Go interfaces) instead of inheritance hierarchies. Protocols define **structural contracts** that classes implement through their public interface, not through inheritance.

**Key Topics:**

- Protocol system overview and philosophy
- Core protocols (Observable, Serializable, Hashable, Adaptable, etc.)
- `@implements` decorator usage and semantics
- Runtime protocol checking with `isinstance()`
- Protocol composition patterns
- Custom protocol definition

**Who Should Read This:**

- Developers building lionherd-core applications
- Library authors integrating with lionherd-core
- Anyone designing protocol-based APIs

---

## Protocol Philosophy

### Why Protocols Over Inheritance?

**Traditional Inheritance Problems:**

```python
# ❌ PROBLEM: Deep inheritance hierarchies
class Base:
    def method_a(self): ...

class MiddleA(Base):
    def method_b(self): ...

class MiddleB(MiddleA):
    def method_c(self): ...

class Concrete(MiddleB):
    def method_d(self): ...
    # Inherits method_a, method_b, method_c implicitly
    # Hard to understand capabilities without reading entire hierarchy
```

**Protocol-Based Solution:**

```python
# ✅ SOLUTION: Explicit protocol implementation
from lionherd_core.protocols import Serializable, Observable, implements

@implements(Observable, Serializable)
class Concrete:
    """Explicitly declares capabilities via protocols."""

    def __init__(self):
        self.id = uuid4()  # Observable requirement

    def to_dict(self, **kwargs):  # Serializable requirement
        return {"id": str(self.id)}

# Clear capabilities from class declaration
# No hidden inherited methods
```

**Benefits:**

1. **Explicitness**: Capabilities visible at class level (no inheritance digging)
2. **Flexibility**: Implement any combination of protocols without diamond problems
3. **Testability**: Easy to mock protocol-implementing objects
4. **Documentation**: Protocols self-document expected behavior

### Structural Typing

Protocols use **structural typing** (duck typing with type checking):

```python
from lionherd_core.protocols import Serializable

class MyClass:
    """Implements Serializable without explicitly declaring it."""

    def to_dict(self, **kwargs):
        return {"key": "value"}

# Runtime check: MyClass has to_dict() → structurally compatible
obj = MyClass()
print(isinstance(obj, Serializable))  # True (structural match)
```

**Key Point**: You don't need to inherit from `Serializable` or use `@implements()` for protocol checking to work. However, `@implements()` provides **explicit documentation and enforcement**.

---

## Core Protocols

### Observable

**Purpose**: Objects with unique UUID identifier

**Contract:**

```python
@runtime_checkable
class Observable(Protocol):
    @property
    def id(self) -> UUID:
        """Unique identifier."""
        ...
```

**Implementation:**

```python
from uuid import uuid4
from lionherd_core.protocols import Observable, implements

@implements(Observable)
class Element:
    def __init__(self):
        self.id = uuid4()  # Required by Observable

# Usage
elem = Element()
print(isinstance(elem, Observable))  # True
print(elem.id)  # UUID('...')
```

**Use Cases:**

- Object identity tracking
- Reference management in graphs
- Deduplication in collections

### Serializable

**Purpose**: Objects that can serialize to dict

**Contract:**

```python
@runtime_checkable
class Serializable(Protocol):
    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict. Args: serialization options (mode, format, etc.)."""
        ...
```

**Implementation:**

```python
from lionherd_core.protocols import Serializable, implements

@implements(Serializable)
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def to_dict(self, **kwargs):
        return {
            "name": self.name,
            "email": self.email,
        }

# Usage
user = User("Alice", "alice@example.com")
print(isinstance(user, Serializable))  # True
print(user.to_dict())  # {"name": "Alice", "email": "alice@example.com"}
```

**Use Cases:**

- JSON serialization
- Database storage
- API responses

### Deserializable

**Purpose**: Objects that can deserialize from dict

**Contract:**

```python
@runtime_checkable
class Deserializable(Protocol):
    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> Any:
        """Deserialize from dict. Args: data dict, deserialization options."""
        ...
```

**Implementation:**

```python
from lionherd_core.protocols import Deserializable, implements

@implements(Deserializable)
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    @classmethod
    def from_dict(cls, data: dict, **kwargs):
        return cls(name=data["name"], email=data["email"])

# Usage
data = {"name": "Alice", "email": "alice@example.com"}
user = User.from_dict(data)
print(isinstance(user, Deserializable))  # True
```

**Use Cases:**

- JSON deserialization
- Database loading
- API request parsing

### Hashable

**Purpose**: Objects that can be hashed for sets/dicts

**Contract:**

```python
@runtime_checkable
class Hashable(Protocol):
    def __hash__(self) -> int:
        """Return hash value for object (must be immutable or ID-based)."""
        ...
```

**Implementation:**

```python
from lionherd_core.protocols import Hashable, Observable, implements
from uuid import uuid4

@implements(Observable, Hashable)
class Element:
    def __init__(self):
        self.id = uuid4()

    def __hash__(self) -> int:
        return hash(self.id)  # ID-based hashing

# Usage
elem1 = Element()
elem2 = Element()

elements = {elem1, elem2}  # Works - hashable
print(len(elements))  # 2
```

**Use Cases:**

- Set membership
- Dict keys
- Fast lookup by identity

### Adaptable

**Purpose**: Sync format conversion (TOML/YAML/JSON/SQL)

**Contract:**

```python
@runtime_checkable
class Adaptable(Protocol):
    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Convert to external format. Args: adapter key, many flag, adapter kwargs."""
        ...

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Create from external format. Args: source object, adapter key, many flag."""
        ...
```

**Implementation:**

```python
from lionherd_core.protocols import Adaptable, implements
from pydapter.adapters import TomlAdapter, YamlAdapter

@implements(Adaptable)
class Config:
    adapters = {
        "toml": TomlAdapter(),
        "yaml": YamlAdapter(),
    }

    def __init__(self, data: dict):
        self.data = data

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs):
        adapter = self.adapters[obj_key]
        return adapter.convert_to(self.data, **kwargs)

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs):
        adapter = cls.adapters[obj_key]
        data = adapter.convert_from(obj, **kwargs)
        return cls(data)

# Usage
config = Config({"host": "localhost", "port": 8080})
toml_str = config.adapt_to("toml")
yaml_str = config.adapt_to("yaml")
```

**Use Cases:**

- Format conversion (TOML ↔ dict ↔ YAML)
- Configuration file I/O
- Data exchange between systems

### AsyncAdaptable

**Purpose**: Async format conversion (DBs, network, files)

**Contract:**

```python
@runtime_checkable
class AsyncAdaptable(Protocol):
    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Async convert to external format. Args: adapter key, many flag, adapter kwargs."""
        ...

    @classmethod
    async def adapt_from_async(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Async create from external format. Args: source object, adapter key, many flag."""
        ...
```

**Implementation:**

```python
from lionherd_core.protocols import AsyncAdaptable, implements

@implements(AsyncAdaptable)
class User:
    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs):
        if obj_key == "db":
            # Async database write
            await db.save(self.to_dict())
            return True
        raise ValueError(f"Unknown adapter: {obj_key}")

    @classmethod
    async def adapt_from_async(cls, obj: Any, obj_key: str, many: bool = False, **kwargs):
        if obj_key == "db":
            # Async database read
            data = await db.load(obj)
            return cls.from_dict(data)
        raise ValueError(f"Unknown adapter: {obj_key}")
```

**Use Cases:**

- Async database I/O
- Network format conversion
- Async file operations

### Invocable

**Purpose**: Objects that can be invoked/executed

**Contract:**

```python
@runtime_checkable
class Invocable(Protocol):
    async def invoke(self) -> Any:
        """Invoke/execute the object. Returns: execution result (any value or None)."""
        ...
```

**Implementation:**

```python
from lionherd_core.libs.concurrency import is_coro_func
from lionherd_core.protocols import Invocable, implements

@implements(Invocable)
class Task:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    async def invoke(self):
        """Execute task function."""
        if is_coro_func(self.fn):
            return await self.fn(*self.args, **self.kwargs)
        return self.fn(*self.args, **self.kwargs)

# Usage
async def work():
    return "done"

# noqa:validation
task = Task(work)
result = await task.invoke()
print(result)  # "done"
```

**Use Cases:**

- Async task execution
- Workflow steps
- Callable objects

### Containable

**Purpose**: Objects supporting membership testing (`in`)

**Contract:**

```python
@runtime_checkable
class Containable(Protocol):
    def __contains__(self, item: Any) -> bool:
        """Check if item is in collection (by UUID or instance)."""
        ...
```

**Implementation:**

```python
from lionherd_core.protocols import Containable, implements

@implements(Containable)
class Collection:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def __contains__(self, item):
        return item in self.items

# Usage
collection = Collection()
collection.add("item1")
print("item1" in collection)  # True
print("item2" in collection)  # False
```

**Use Cases:**

- Custom collections
- Membership testing
- Fast lookups

### Allowable

**Purpose**: Objects with defined allowed values/keys

**Contract:**

```python
@runtime_checkable
class Allowable(Protocol):
    def allowed(self) -> set[str]:
        """Return set of allowed keys/values."""
        ...
```

**Implementation:**

```python
from enum import Enum
from lionherd_core.protocols import Allowable

class Status(Enum):
    """Enum automatically implements Allowable."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

# Enum classes have .allowed() from lionherd-core's Enum base
# (if using lionherd_core.types.Enum)

# Manual implementation
class Config(Allowable):
    ALLOWED_KEYS = {"host", "port", "timeout"}

    def allowed(self) -> set[str]:
        return self.ALLOWED_KEYS

# Usage
config = Config()
print(config.allowed())  # {"host", "port", "timeout"}
```

**Use Cases:**

- Configuration validation
- Enum-like types
- Whitelist enforcement

---

## The `@implements` Decorator

### Semantics: Rust-Like Trait Implementation

The `@implements()` decorator enforces **literal implementation**: methods/properties must be defined **in the class body**, not inherited.

**Rationale:**

- **Explicitness over implicitness** (Zen of Python)
- **Clear ownership**: Each class declares what it implements
- **No ambiguity**: No accidental protocol claims through inheritance

### Rules

#### Rule 1: Methods must be in class body

```python
# ❌ WRONG: Relies on inheritance
class Parent:
    def to_dict(self, **kwargs):
        return {}

@implements(Serializable)  # TypeError!
class Child(Parent):
    pass  # No to_dict() in Child body

# ✅ CORRECT: Explicit override
@implements(Serializable)
class Child(Parent):
    def to_dict(self, **kwargs):  # Defined in Child body
        return super().to_dict(**kwargs)  # Can call parent
```

#### Rule 2: Properties must be declared in class body

```python
# ❌ WRONG: Inherits property
class Parent:
    @property
    def id(self):
        return self._id

@implements(Observable)  # TypeError!
class Child(Parent):
    pass  # No id property in Child body

# ✅ CORRECT: Re-declare property
@implements(Observable)
class Child(Parent):
    @property
    def id(self):  # Declared in Child body
        return super().id
```

#### Rule 3: Pydantic fields in annotations count

```python
from pydantic import BaseModel
from lionherd_core.protocols import Observable, implements
from uuid import UUID

@implements(Observable)
class User(BaseModel):
    id: UUID  # Field in __annotations__ → counts as "in class body"
    name: str
```

### Usage Examples

**Single Protocol:**

```python
from lionherd_core.protocols import Serializable, implements

@implements(Serializable)
class User:
    def to_dict(self, **kwargs):
        return {"name": self.name}
```

**Multiple Protocols:**

```python
from lionherd_core.protocols import Observable, Serializable, Hashable, implements
from uuid import uuid4

@implements(Observable, Serializable, Hashable)
class Element:
    def __init__(self):
        self.id = uuid4()  # Observable

    def to_dict(self, **kwargs):  # Serializable
        return {"id": str(self.id)}

    def __hash__(self):  # Hashable
        return hash(self.id)
```

**Protocol Composition:**

```python
from lionherd_core.protocols import Observable, Serializable, Deserializable, implements
from uuid import UUID, uuid4

@implements(Observable, Serializable, Deserializable)
class User:
    def __init__(self, name: str, user_id: UUID | None = None):
        self.id = user_id or uuid4()  # Observable
        self.name = name

    def to_dict(self, **kwargs):  # Serializable
        return {"id": str(self.id), "name": self.name}

    @classmethod
    def from_dict(cls, data: dict, **kwargs):  # Deserializable
        return cls(name=data["name"], user_id=UUID(data["id"]))
```

---

## Runtime Protocol Checking

### Using `isinstance()`

Protocols marked `@runtime_checkable` support `isinstance()`:

```python
from lionherd_core.protocols import Serializable

class User:
    def to_dict(self, **kwargs):
        return {"name": "Alice"}

user = User()
print(isinstance(user, Serializable))  # True (structural match)
```

**How it works:**

- Python checks if `user` has a `to_dict()` method with compatible signature
- No inheritance or explicit declaration needed
- Pure structural typing

### Type Narrowing with Protocols

Use `isinstance()` for type narrowing:

```python
from lionherd_core.protocols import Serializable

def save_to_json(obj: object) -> None:
    """Save object to JSON if serializable."""
    if isinstance(obj, Serializable):
        # Type checker knows: obj has to_dict()
        data = obj.to_dict(mode="json")
        with open("output.json", "w") as f:
            json.dump(data, f)
    else:
        raise TypeError(f"Object {obj} is not Serializable")
```

### Checking Multiple Protocols

```python
from lionherd_core.protocols import Observable, Serializable

def process(obj: object) -> dict:
    """Process object if it's both observable and serializable."""
    if not isinstance(obj, (Observable, Serializable)):
        raise TypeError("Object must be Observable and Serializable")

    # Type checker knows: obj has .id and .to_dict()
    obj_id = obj.id
    data = obj.to_dict()
    return {"id": str(obj_id), "data": data}
```

---

## Protocol Composition Patterns

### Pattern 1: Base + Extensions

Common pattern: base protocol + optional extensions

```python
from lionherd_core.protocols import Serializable, Adaptable, implements

# Base capability
@implements(Serializable)
class SimpleUser:
    def to_dict(self, **kwargs):
        return {"name": self.name}

# Extended capabilities
@implements(Serializable, Adaptable)
class AdvancedUser:
    def to_dict(self, **kwargs):
        return {"name": self.name, "email": self.email}

    def adapt_to(self, obj_key: str, **kwargs):
        if obj_key == "toml":
            return toml.dumps(self.to_dict())
        raise ValueError(f"Unknown format: {obj_key}")

    @classmethod
    def adapt_from(cls, obj, obj_key: str, **kwargs):
        if obj_key == "toml":
            data = toml.loads(obj)
            return cls.from_dict(data)
        raise ValueError(f"Unknown format: {obj_key}")
```

### Pattern 2: Mixin-Like Protocols

Combine protocols for specific capabilities:

```python
from lionherd_core.protocols import Observable, Hashable, Containable, implements
from uuid import uuid4

@implements(Observable, Hashable, Containable)
class Collection:
    """Collection with identity, hashing, and membership."""

    def __init__(self):
        self.id = uuid4()  # Observable
        self.items = set()

    def __hash__(self):  # Hashable
        return hash(self.id)

    def __contains__(self, item):  # Containable
        return item in self.items

    def add(self, item):
        self.items.add(item)

# Usage
collection = Collection()
collection.add("item1")

# All protocol capabilities available
print(collection.id)  # UUID
print(hash(collection))  # int
print("item1" in collection)  # True
```

### Pattern 3: Conditional Protocol Implementation

Implement protocols conditionally based on configuration:

```python
from lionherd_core.protocols import Serializable, Adaptable, implements

class User:
    """User with optional adapter support."""

    def __init__(self, name: str, enable_adapters: bool = False):
        self.name = name
        self._enable_adapters = enable_adapters

    def to_dict(self, **kwargs):  # Always Serializable
        return {"name": self.name}

    def adapt_to(self, obj_key: str, **kwargs):
        if not self._enable_adapters:
            raise NotImplementedError("Adapters not enabled")
        # Adapter logic...

# Dynamic protocol implementation
def create_user(name: str, with_adapters: bool):
    if with_adapters:
        @implements(Serializable, Adaptable)
        class UserWithAdapters(User):
            def adapt_to(self, obj_key: str, **kwargs):
                return super().adapt_to(obj_key, **kwargs)

            @classmethod
            def adapt_from(cls, obj, obj_key: str, **kwargs):
                # Implementation...
                pass

        return UserWithAdapters(name, enable_adapters=True)
    else:
        @implements(Serializable)
        class SimpleUser(User):
            pass

        return SimpleUser(name, enable_adapters=False)
```

---

## Custom Protocols

### Defining Custom Protocols

Create domain-specific protocols:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Validatable(Protocol):
    """Objects that can validate themselves."""

    def validate(self) -> bool:
        """Validate object state. Returns True if valid."""
        ...

@runtime_checkable
class Cacheable(Protocol):
    """Objects that can be cached."""

    def cache_key(self) -> str:
        """Return unique cache key."""
        ...

    def cache_ttl(self) -> int:
        """Return cache TTL in seconds."""
        ...

# Implementation
from lionherd_core.protocols import implements

@implements(Validatable, Cacheable)
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def validate(self) -> bool:  # Validatable
        return bool(self.name and "@" in self.email)

    def cache_key(self) -> str:  # Cacheable
        return f"user:{self.email}"

    def cache_ttl(self) -> int:  # Cacheable
        return 3600  # 1 hour

# Usage
user = User("Alice", "alice@example.com")
print(isinstance(user, Validatable))  # True
print(isinstance(user, Cacheable))    # True

if user.validate():
    cache.set(user.cache_key(), user.to_dict(), ttl=user.cache_ttl())
```

### Protocol Guidelines

#### 1. Single Responsibility

Each protocol should define one capability:

```python
# ✅ GOOD: Single responsibility
class Serializable(Protocol):
    def to_dict(self, **kwargs): ...

class Deserializable(Protocol):
    @classmethod
    def from_dict(cls, data, **kwargs): ...

# ❌ BAD: Multiple responsibilities
class SerializableDeserializable(Protocol):
    def to_dict(self, **kwargs): ...
    @classmethod
    def from_dict(cls, data, **kwargs): ...
```

#### 2. Clear Naming

Protocol names should describe capability:

```python
# ✅ GOOD: Describes capability
class Invocable(Protocol): ...
class Containable(Protocol): ...
class Adaptable(Protocol): ...

# ❌ BAD: Unclear capability
class Handler(Protocol): ...
class Manager(Protocol): ...
class Helper(Protocol): ...
```

#### 3. Minimal Interface

Keep protocols minimal:

```python
# ✅ GOOD: Minimal interface
class Identifiable(Protocol):
    @property
    def id(self) -> UUID: ...

# ❌ BAD: Too many methods
class ComplexProtocol(Protocol):
    def method1(self): ...
    def method2(self): ...
    def method3(self): ...
    def method4(self): ...
    def method5(self): ...
```

---

## Best Practices

### 1. Use `@implements` for Documentation

Even if not strictly required, `@implements()` documents intent:

```python
# ✅ GOOD: Explicit intent
@implements(Serializable)
class User:
    def to_dict(self, **kwargs):
        return {"name": self.name}

# ❌ OK but less clear
class User:
    def to_dict(self, **kwargs):
        return {"name": self.name}
```

### 2. Prefer Composition Over Inheritance

```python
# ✅ GOOD: Protocol composition
@implements(Observable, Serializable, Hashable)
class Element:
    ...

# ❌ BAD: Inheritance hierarchy
class Base:
    ...
class Middle(Base):
    ...
class Element(Middle):
    ...
```

### 3. Check Protocols at Boundaries

Validate protocol implementation at system boundaries:

```python
def save_to_db(obj: object) -> None:
    """Save object to database (requires Serializable)."""
    if not isinstance(obj, Serializable):
        raise TypeError(f"Object {obj} must be Serializable")

    data = obj.to_dict(mode="db")
    db.save(data)
```

### 4. Document Protocol Requirements

Document which protocols are required/optional:

```python
class DataProcessor:
    """Process data objects.

    Requirements:
        - Input objects must implement Serializable
        - Output objects should implement Deserializable (optional)
    """

    def process(self, obj: Serializable) -> dict:
        """Process serializable object."""
        return obj.to_dict()
```

### 5. Test Protocol Compliance

Test that classes correctly implement protocols:

```python
import pytest
from lionherd_core.protocols import Serializable, Observable

def test_user_implements_serializable():
    user = User("Alice")
    assert isinstance(user, Serializable)
    data = user.to_dict()
    assert isinstance(data, dict)

def test_element_implements_observable():
    elem = Element()
    assert isinstance(elem, Observable)
    assert hasattr(elem, "id")
    from uuid import UUID
    assert isinstance(elem.id, UUID)
```

---

## See Also

- **Type Safety Guide**: TypeGuards and protocol-based type narrowing
- **API Design Guide**: Protocol-based API design patterns
- **Protocol API Reference**: Complete protocol documentation

---

## References

- [PEP 544 – Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [Rust Traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Go Interfaces](https://go.dev/tour/methods/9)
- [Protocol-Oriented Programming](https://developer.apple.com/videos/wwdc2015/)

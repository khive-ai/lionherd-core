# Type Safety Guide

> Type narrowing, TypeGuards, and runtime type checking patterns in lionherd-core

## Overview

lionherd-core leverages Python's type system to provide **compile-time safety with runtime guarantees**. This guide covers type narrowing patterns, TypeGuard usage, and integration with sentinel types for robust type-safe APIs.

**Key Topics:**

- Type narrowing with TypeGuards
- Runtime type checking patterns
- Integration with sentinel types (Unset, Undefined)
- Protocol-based structural typing
- Pydantic validation integration

**Who Should Read This:**

- Developers building type-safe APIs
- Users working with optional parameters and sentinel values
- Anyone integrating lionherd-core with type checkers (mypy, pyright)

---

## Type Narrowing Fundamentals

### What is Type Narrowing?

Type narrowing allows type checkers to **refine types** based on runtime checks. When you test a value's type, the type checker understands the refined type in different code branches.

**Example:**

```python
from lionherd_core.types import MaybeUnset, Unset

def process(value: MaybeUnset[str]) -> str:
    if value is Unset:
        # Type checker knows: value is UnsetType here
        return "default"
    else:
        # Type checker knows: value is str here
        return value.upper()  # OK - str method available
```

**Without narrowing** (type checker error):

```python
def process(value: MaybeUnset[str]) -> str:
    # Type checker error: value might be UnsetType (no .upper())
    return value.upper()
```

### Built-in Type Narrowing

Python's type checkers understand several narrowing patterns:

#### 1. Identity Checks (`is`, `is not`)

```python
from lionherd_core.types import Undefined, MaybeUndefined

def get_field(data: dict, key: str) -> MaybeUndefined[str]:
    return data.get(key, Undefined)

value = get_field({"name": "Alice"}, "name")

if value is Undefined:
    print("Field missing")  # value: UndefinedType
else:
    print(value.upper())    # value: str (narrowed)
```

#### 2. Equality Checks (`==`, `!=`)

```python
def process(value: str | None) -> str:
    if value is None:
        return "empty"      # value: None
    else:
        return value.upper()  # value: str
```

#### 3. Type Checks (`isinstance`)

```python
from typing import Union

def handle(value: Union[str, int, list]) -> str:
    if isinstance(value, str):
        return value.upper()  # value: str
    elif isinstance(value, int):
        return str(value)      # value: int
    else:
        return str(len(value))  # value: list
```

#### 4. Truthiness Checks

```python
def process(value: str | None) -> str:
    if value:
        return value.upper()  # value: str (truthy check excludes None and "")
    return "default"
```

**Warning**: Truthiness checks have limitations:

```python
from lionherd_core.types import Unset

def process(value: MaybeUnset[str]) -> str:
    if value:  # WRONG - Unset is falsy like empty string!
        return value.upper()
    return "default"

# Both return "default":
print(process(""))      # Empty string is falsy
print(process(Unset))   # Unset is falsy
```

**Solution**: Use explicit identity checks for sentinels:

```python
def process(value: MaybeUnset[str]) -> str:
    if value is Unset:
        return "default"
    elif value:  # Now safely checks non-Unset values
        return value.upper()
    else:
        return "empty string"
```

---

## TypeGuards: Custom Type Narrowing

### What are TypeGuards?

`TypeGuard` functions tell type checkers: "If this returns `True`, narrow the type to `T`".

**Signature:**

```python
from typing import TypeGuard

def is_str(value: object) -> TypeGuard[str]:
    """Returns True if value is str, narrows type to str."""
    return isinstance(value, str)
```

**Usage:**

```python
from typing import Union

def process(value: Union[str, int, None]) -> str:
    if is_str(value):
        # Type checker knows: value is str here
        return value.upper()
    return str(value)
```

### lionherd-core's `not_sentinel()`

The `not_sentinel()` function is a TypeGuard for filtering sentinel values:

**Signature:**

```python
from lionherd_core.types import not_sentinel, MaybeSentinel

def not_sentinel(
    value: T | UndefinedType | UnsetType,
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> TypeGuard[T]:
    """Type-narrowing check: NOT a sentinel."""
    ...
```

**Basic Usage:**

```python
from lionherd_core.types import Undefined, Unset, MaybeSentinel, not_sentinel

def process(value: MaybeSentinel[str]) -> str:
    if not_sentinel(value):
        # Type checker knows: value is str (sentinels excluded)
        return value.upper()
    else:
        # Type checker knows: value is UndefinedType | UnsetType
        return "missing"

print(process("hello"))    # "HELLO"
print(process(Undefined))  # "missing"
print(process(Unset))      # "missing"
```

**With `none_as_sentinel`:**

```python
from lionherd_core.types import Unset, not_sentinel

def handle(value: str | None | UnsetType) -> str:
    if not_sentinel(value, none_as_sentinel=True):
        # Type checker knows: value is str (None and Unset excluded)
        return value.upper()
    else:
        # Type checker knows: value is None | UnsetType
        return "empty"

print(handle("test"))  # "TEST"
print(handle(None))    # "empty"
print(handle(Unset))   # "empty"
```

**With `empty_as_sentinel`:**

```python
from lionherd_core.types import not_sentinel, MaybeSentinel

def process_items(items: MaybeSentinel[list[int]]) -> int:
    if not_sentinel(items, empty_as_sentinel=True):
        # Type checker knows: items is list[int] and non-empty
        return sum(items)
    return 0

print(process_items([1, 2, 3]))  # 6
print(process_items([]))          # 0 (empty list treated as sentinel)
```

### Custom TypeGuards

You can create custom TypeGuards for domain-specific narrowing:

#### Example: Email validation

```python
from typing import TypeGuard

EmailStr = str  # Type alias for documentation

def is_email(value: str) -> TypeGuard[EmailStr]:
    """Type-narrowing check: value is a valid email string."""
    return "@" in value and "." in value.split("@")[1]

def send_email(email: str) -> None:
    if not is_email(email):
        raise ValueError(f"Invalid email: {email}")
    # Type checker knows: email is EmailStr here
    # (useful for type-safe email APIs)
    ...
```

#### Example: Non-empty list

```python
from typing import TypeGuard, TypeVar

T = TypeVar("T")

def is_non_empty(items: list[T]) -> TypeGuard[list[T]]:
    """Type-narrowing check: list is non-empty."""
    return len(items) > 0

def process_batch(items: list[int]) -> int:
    if not is_non_empty(items):
        raise ValueError("Cannot process empty batch")
    # Type checker knows: items is guaranteed non-empty
    return items[0]  # Safe - no IndexError
```

---

## Sentinel Type Integration

### Three-State Logic with Type Safety

Sentinels enable type-safe three-state logic: value / None / missing.

#### Pattern: Optional parameter with None as valid value

```python
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

def configure(
    timeout: MaybeUnset[float | None] = Unset
) -> dict:
    """Configure with optional timeout.

    Args:
        timeout: Timeout in seconds, None for infinite, Unset for default
    """
    config = {}

    if not_sentinel(timeout):
        # timeout is float | None (not Unset)
        if timeout is None:
            config["timeout"] = float("inf")
        else:
            # Type checker knows: timeout is float
            config["timeout"] = timeout
    else:
        # timeout is Unset - use default
        config["timeout"] = 30.0

    return config

# Usage
print(configure())                    # {'timeout': 30.0} (default)
print(configure(timeout=60.0))        # {'timeout': 60.0}
print(configure(timeout=None))        # {'timeout': inf}
```

#### Pattern: Dictionary lookup with missing key detection

```python
from lionherd_core.types import Undefined, MaybeUndefined, not_sentinel

def get_config(key: str) -> MaybeUndefined[str]:
    """Get config value, returning Undefined if key missing."""
    config = {"host": "localhost", "port": "8080", "timeout": None}
    return config.get(key, Undefined)

# Type-safe processing
def process_config(key: str) -> str:
    value = get_config(key)

    if not_sentinel(value):
        # value is str | None (key exists)
        if value is None:
            return f"{key}: explicitly disabled"
        else:
            # Type checker knows: value is str
            return f"{key}: {value}"
    else:
        # value is Undefined (key missing)
        return f"{key}: not configured"

print(process_config("host"))     # "host: localhost"
print(process_config("timeout"))  # "timeout: explicitly disabled"
print(process_config("retries"))  # "retries: not configured"
```

### Type-Safe Filtering

Use `not_sentinel()` to filter collections while preserving type information:

#### Example: Filter sentinel values from list

```python
from lionherd_core.types import Undefined, Unset, MaybeSentinel, not_sentinel

data: list[MaybeSentinel[int]] = [1, 2, Undefined, 3, Unset, 4]

# Type-safe filtering
real_values = [x for x in data if not_sentinel(x)]
# Type: list[int] (type checker knows sentinels are excluded)

print(real_values)  # [1, 2, 3, 4]
print(sum(real_values))  # 10 - type-safe sum()
```

#### Example: Filter both sentinels and None

```python
from lionherd_core.types import Unset, not_sentinel

data: list[int | None | UnsetType] = [1, None, 2, Unset, 3]

real_values = [x for x in data if not_sentinel(x, none_as_sentinel=True)]
# Type: list[int] (None and Unset excluded)

print(real_values)  # [1, 2, 3]
```

---

## Runtime Type Validation

### Pydantic Integration

lionherd-core uses **Pydantic V2** for runtime validation. Combine with type narrowing for robust validation:

#### Example: Validated configuration

```python
from pydantic import BaseModel, Field, field_validator
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class ServerConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(default=8080, ge=1, le=65535)
    timeout: float | None = Field(default=30.0, ge=0)

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v > 300:
            raise ValueError("Timeout cannot exceed 300 seconds")
        return v

def update_config(
    host: MaybeUnset[str] = Unset,
    port: MaybeUnset[int] = Unset,
    timeout: MaybeUnset[float | None] = Unset,
) -> ServerConfig:
    """Update config with type-safe partial updates."""
    current = ServerConfig()
    updates = {}

    if not_sentinel(host):
        updates["host"] = host
    if not_sentinel(port):
        updates["port"] = port
    if not_sentinel(timeout):
        updates["timeout"] = timeout

    # Pydantic validates updates
    return current.model_copy(update=updates)

# Usage
config = update_config(port=9000)  # Validates port range
print(config.port)  # 9000

# Invalid update raises ValidationError
try:
    update_config(port=70000)  # Out of range
except Exception as e:
    print(f"Validation error: {e}")
```

### Protocol-Based Type Checking

Use `isinstance()` with runtime-checkable protocols for structural typing:

#### Example: Checking protocol implementation

```python
from lionherd_core.protocols import Serializable, Observable

def save_to_db(obj: object) -> None:
    """Save object to database if serializable."""
    if isinstance(obj, Serializable):
        # Type checker knows: obj has to_dict()
        data = obj.to_dict(mode="db")
        print(f"Saving: {data}")
    else:
        raise TypeError(f"Object {obj} is not Serializable")

# Example
from lionherd_core.base import Element

elem = Element(metadata={"key": "value"})
save_to_db(elem)  # Works - Element implements Serializable
```

#### Example: Multi-protocol check

```python
from lionherd_core.protocols import Observable, Serializable

def sync_object(obj: object) -> None:
    """Sync object to remote if it's observable and serializable."""
    if not isinstance(obj, (Observable, Serializable)):
        raise TypeError("Object must be Observable and Serializable")

    # Type checker knows: obj has .id and .to_dict()
    obj_id = obj.id
    data = obj.to_dict(mode="json")
    print(f"Syncing {obj_id}: {data}")
```

---

## Best Practices

### 1. Use Identity Checks for Sentinels

Always use `is` / `is not` for sentinel comparisons:

```python
# ✅ CORRECT
if value is Unset:
    ...

# ❌ WRONG (works but semantically incorrect)
if value == Unset:
    ...
```

### 2. Prefer TypeGuards for Reusable Checks

Extract common type checks into TypeGuard functions:

```python
# ❌ WRONG: Repeated inline checks
def process1(value: MaybeSentinel[str]) -> str:
    if value is not Undefined and value is not Unset:
        return value.upper()
    return "default"

def process2(value: MaybeSentinel[int]) -> int:
    if value is not Undefined and value is not Unset:
        return value * 2
    return 0

# ✅ CORRECT: Reusable TypeGuard
from lionherd_core.types import not_sentinel

def process1(value: MaybeSentinel[str]) -> str:
    if not_sentinel(value):
        return value.upper()
    return "default"

def process2(value: MaybeSentinel[int]) -> int:
    if not_sentinel(value):
        return value * 2
    return 0
```

### 3. Document Type-Narrowing Assumptions

Explain non-obvious type narrowing in docstrings:

```python
from lionherd_core.types import MaybeUnset, not_sentinel

def update_user(
    name: MaybeUnset[str] = Unset,
    email: MaybeUnset[str | None] = Unset,
) -> dict:
    """Update user fields with partial updates.

    Args:
        name: New name, or Unset to skip update
        email: New email, None to clear, or Unset to skip update

    Returns:
        Updated user dict

    Note:
        Type narrowing via not_sentinel() ensures only provided fields
        are updated. Unset values are filtered out completely.
    """
    updates = {}
    if not_sentinel(name):
        updates["name"] = name
    if not_sentinel(email):
        updates["email"] = email
    return updates
```

### 4. Combine with Pydantic for Validation

Use type narrowing to filter, Pydantic to validate:

```python
from pydantic import BaseModel, field_validator
from lionherd_core.types import MaybeUnset, Unset, not_sentinel

class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v is not None and "@" not in v:
            raise ValueError("Invalid email")
        return v

def apply_update(
    name: MaybeUnset[str] = Unset,
    email: MaybeUnset[str | None] = Unset,
) -> UserUpdate:
    """Apply validated update with type-safe sentinel filtering."""
    # Filter sentinels with type narrowing
    updates = {}
    if not_sentinel(name):
        updates["name"] = name
    if not_sentinel(email):
        updates["email"] = email

    # Validate with Pydantic
    return UserUpdate(**updates)

# Usage
update = apply_update(email="alice@example.com")  # Validates email format
```

### 5. Use Protocols for Flexible Type Checking

Prefer protocols over concrete types for maximum flexibility:

```python
from lionherd_core.protocols import Serializable
from typing import Protocol

# ✅ CORRECT: Protocol-based (works with any serializable object)
def save(obj: Serializable) -> None:
    data = obj.to_dict()
    ...

# ❌ WRONG: Concrete type (only works with Element)
from lionherd_core.base import Element

def save(obj: Element) -> None:
    data = obj.to_dict()
    ...
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Sentinels are Falsy

**Issue**: Using truthiness checks with sentinels

```python
from lionherd_core.types import Unset, MaybeUnset

def process(value: MaybeUnset[str]) -> str:
    if value:  # WRONG - Unset is falsy!
        return value.upper()
    return "default"

# Both return "default":
print(process(""))      # Empty string is falsy
print(process(Unset))   # Unset is falsy too
```

**Solution**: Use explicit sentinel checks

```python
def process(value: MaybeUnset[str]) -> str:
    if value is Unset:
        return "default"
    elif value:
        return value.upper()
    else:
        return "empty string"
```

### Pitfall 2: TypeGuard Order Matters

**Issue**: Checking narrowed type after broader check

```python
from typing import Union

def process(value: Union[str, int, None]) -> str:
    if value:  # Narrows to str | int (excludes None and falsy values)
        if isinstance(value, str):  # Already narrowed, int still possible
            return value.upper()
        return str(value)  # int case
    return "none"

# Works but verbose
```

**Solution**: Check specific types first

```python
def process(value: Union[str, int, None]) -> str:
    if isinstance(value, str):
        return value.upper()
    elif isinstance(value, int):
        return str(value)
    return "none"
```

### Pitfall 3: Overusing TypeGuards

**Issue**: Creating TypeGuards for simple checks

```python
# ❌ WRONG: Unnecessary TypeGuard for simple check
from typing import TypeGuard

def is_not_none(value: str | None) -> TypeGuard[str]:
    return value is not None

def process(value: str | None) -> str:
    if is_not_none(value):
        return value.upper()
    return "none"
```

**Solution**: Use built-in narrowing for simple cases

```python
# ✅ CORRECT: Built-in narrowing suffices
def process(value: str | None) -> str:
    if value is not None:
        return value.upper()
    return "none"
```

### Pitfall 4: Mixing Sentinel Types Without Distinction

**Issue**: Using both `Undefined` and `Unset` without clear semantics

```python
# ❌ UNCLEAR: Why two sentinels?
def update(name=Unset, email=Undefined) -> None:
    ...
```

**Solution**: Use one sentinel consistently, or document the distinction

```python
# ✅ CORRECT: Consistent use
from lionherd_core.types import Unset, MaybeUnset

def update(
    name: MaybeUnset[str] = Unset,
    email: MaybeUnset[str] = Unset,
) -> None:
    """Update user fields.

    Args:
        name: New name, or Unset to skip update
        email: New email, or Unset to skip update
    """
    ...
```

---

## Examples

### Example 1: Type-Safe API Client

```python
from typing import Any
from pydantic import BaseModel
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class APIResponse(BaseModel):
    status: int
    data: dict[str, Any]

class APIClient:
    def update_resource(
        self,
        resource_id: str,
        name: MaybeUnset[str] = Unset,
        tags: MaybeUnset[list[str]] = Unset,
        metadata: MaybeUnset[dict[str, Any]] = Unset,
    ) -> APIResponse:
        """Update resource with type-safe partial updates.

        Uses TypeGuard to ensure only provided fields are sent.
        """
        payload = {"id": resource_id}

        # Type-safe field inclusion
        if not_sentinel(name):
            payload["name"] = name
        if not_sentinel(tags):
            payload["tags"] = tags
        if not_sentinel(metadata):
            payload["metadata"] = metadata

        # Simulate API call
        return APIResponse(status=200, data=payload)

# Usage
client = APIClient()

# Update only name
response = client.update_resource("res-123", name="New Name")
print(response.data)  # {"id": "res-123", "name": "New Name"}

# Update multiple fields
response = client.update_resource(
    "res-123",
    name="Updated",
    tags=["v2", "prod"]
)
print(response.data)
# {"id": "res-123", "name": "Updated", "tags": ["v2", "prod"]}
```

### Example 2: Protocol-Based Serialization

```python
from lionherd_core.protocols import Serializable, Observable

def serialize_if_possible(obj: object) -> dict | None:
    """Serialize object if it implements Serializable protocol."""
    if isinstance(obj, Serializable):
        # Type narrowed to Serializable - .to_dict() available
        return obj.to_dict(mode="json")
    return None

def get_id_if_observable(obj: object) -> str | None:
    """Get UUID string if object is observable."""
    if isinstance(obj, Observable):
        # Type narrowed to Observable - .id available
        return str(obj.id)
    return None

# Usage
from lionherd_core.base import Element

elem = Element(metadata={"key": "value"})

data = serialize_if_possible(elem)
print(data)  # {"id": "...", "created_at": "...", "metadata": {...}}

obj_id = get_id_if_observable(elem)
print(obj_id)  # "123e4567-e89b-12d3-a456-426614174000"

# Non-protocol object
plain_obj = {"key": "value"}
print(serialize_if_possible(plain_obj))  # None
print(get_id_if_observable(plain_obj))   # None
```

### Example 3: Type-Safe Configuration Builder

```python
from pydantic import BaseModel, Field
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    username: str = "admin"
    password: str | None = None
    ssl: bool = True
    timeout: float = Field(default=30.0, gt=0)

class ConfigBuilder:
    """Type-safe configuration builder with partial updates."""

    def __init__(self, base: DatabaseConfig | None = None):
        self.config = base or DatabaseConfig()

    def update(
        self,
        host: MaybeUnset[str] = Unset,
        port: MaybeUnset[int] = Unset,
        username: MaybeUnset[str] = Unset,
        password: MaybeUnset[str | None] = Unset,
        ssl: MaybeUnset[bool] = Unset,
        timeout: MaybeUnset[float] = Unset,
    ) -> "ConfigBuilder":
        """Update config with type-safe field filtering."""
        updates = {}

        # TypeGuard ensures only provided fields are updated
        if not_sentinel(host):
            updates["host"] = host
        if not_sentinel(port):
            updates["port"] = port
        if not_sentinel(username):
            updates["username"] = username
        if not_sentinel(password):
            updates["password"] = password
        if not_sentinel(ssl):
            updates["ssl"] = ssl
        if not_sentinel(timeout):
            updates["timeout"] = timeout

        # Pydantic validates updated config
        self.config = self.config.model_copy(update=updates)
        return self

    def build(self) -> DatabaseConfig:
        """Return final validated config."""
        return self.config

# Usage
config = (
    ConfigBuilder()
    .update(host="db.example.com", port=5433)
    .update(ssl=True, timeout=60.0)
    .build()
)

print(config.model_dump())
# {
#     "host": "db.example.com",
#     "port": 5433,
#     "username": "admin",
#     "password": None,
#     "ssl": True,
#     "timeout": 60.0
# }
```

---

## See Also

- **API Design Guide**: When to use sentinels vs Optional
- **Validation Guide**: Validation patterns with Spec and Pydantic
- **Protocols Guide**: Protocol system overview and implementation
- **Sentinel Values API**: Complete sentinel types reference

---

## References

- [PEP 647 – User-Defined Type Guards](https://peps.python.org/pep-0647/)
- [PEP 544 – Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
- [Mypy Type Narrowing](https://mypy.readthedocs.io/en/stable/type_narrowing.html)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)

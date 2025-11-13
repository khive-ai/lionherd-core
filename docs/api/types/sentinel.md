# Sentinel Values

> Type-safe sentinel values for distinguishing None, missing, and unset states

## Overview

Sentinel values provide a type-safe mechanism to distinguish between `None` (intentional null value), **missing keys/fields** (never existed in namespace), and **unset parameters** (key present but value not provided). This is critical for optional parameters, partial updates, and differentiating explicit `None` from absence.

**Key Capabilities:**

- **Singleton Identity**: Each sentinel type has exactly one instance, enabling safe `is` comparisons
- **Falsy Behavior**: All sentinels evaluate to `False` in boolean context
- **Serialization Safety**: Identity preserved across copy/deepcopy/pickle
- **Type Narrowing**: TypeGuard support for type-safe value extraction
- **Three-State Logic**: Distinguishes `None` vs `Undefined` vs `Unset`

**When to Use Sentinel Values:**

- **Optional Parameters**: Distinguish "user passed None" from "user didn't pass anything"
- **Partial Updates**: PATCH endpoints where missing fields ≠ "set to None"
- **Default Resolution**: Determine if value was explicitly set or should use default
- **Key Existence**: Differentiate "key missing" from "key present with None value"
- **API Design**: Functions where `None` is a valid input distinct from "not provided"

**When NOT to Use Sentinel Values:**

- **Simple None Checks**: If `None` adequately represents absence, use `Optional[T]`
- **Required Fields**: Sentinels are for optional/conditional values only
- **Performance-Critical Paths**: `is` checks are fast but add overhead vs direct `None` checks
- **Public APIs**: Sentinels may confuse external users; reserve for internal logic

## Sentinel Types

### SingletonType (Base Class)

```python
from lionherd_core.types import SingletonType

class SingletonType(metaclass=_SingletonMeta):
    """Base for singleton sentinels. Falsy, identity-preserving across copy/deepcopy."""
```

**Characteristics:**

- **Singleton Pattern**: Only one instance per subclass via metaclass
- **Identity Preservation**: `copy()` and `deepcopy()` return same instance
- **Falsy**: `bool(sentinel)` always returns `False`
- **Abstract**: Subclasses must implement `__bool__()` and `__repr__()`

**Why Singleton?**

Ensures `is` comparisons work reliably:

```python
from copy import copy, deepcopy

# Same instance everywhere
assert Undefined is Undefined
assert copy(Undefined) is Undefined
assert deepcopy(Undefined) is Undefined

# Safe for identity checks
if value is Undefined:
    # Guaranteed to work across modules/scopes
    ...
```

### Undefined

**Sentinel indicating a field/key is entirely missing from a namespace.**

```python
from lionherd_core.types import Undefined, UndefinedType, MaybeUndefined

# Instance (use this)
Undefined: UndefinedType
"""A key or field entirely missing from a namespace"""

# Type alias for annotations
MaybeUndefined[T] = T | UndefinedType
```

**Signature:**

```python
class UndefinedType(SingletonType):
    """Sentinel: field/key entirely missing from namespace. Use for missing keys, never-set fields."""

    def __bool__(self) -> Literal[False]: ...
    def __repr__(self) -> Literal["Undefined"]: ...
    def __str__(self) -> Literal["Undefined"]: ...
    def __reduce__(self) -> tuple[type[UndefinedType], tuple[()]]: ...
```

**Use Cases:**

- **Dictionary Lookups**: `dict.get(key, Undefined)` - distinguish missing key from `None` value
- **Optional Fields**: Fields that may not exist in data structure
- **Schema Validation**: Detect missing required fields vs fields set to `None`

**Examples:**

```python
from lionherd_core.types import Undefined, MaybeUndefined

# Dictionary lookups
data = {"key1": None, "key2": "value"}

result = data.get("key1", Undefined)
print(result)  # None (key exists with None value)

result = data.get("key3", Undefined)
print(result)  # Undefined (key missing)

# Type-safe optional handling
def process(value: MaybeUndefined[str]) -> str:
    if value is Undefined:
        return "Field missing from input"
    elif value is None:
        return "Field explicitly set to None"
    else:
        return f"Value: {value}"

print(process(Undefined))  # "Field missing from input"
print(process(None))       # "Field explicitly set to None"
print(process("data"))     # "Value: data"

# Boolean context (falsy)
if not Undefined:
    print("Sentinels are falsy")  # Prints
```

### Unset

**Sentinel indicating a key is present but value was not provided.**

```python
from lionherd_core.types import Unset, UnsetType, MaybeUnset

# Instance (use this)
Unset: UnsetType
"""A key present but value not yet provided."""

# Type alias for annotations
MaybeUnset[T] = T | UnsetType
```

**Signature:**

```python
class UnsetType(SingletonType):
    """Sentinel: key present but value not provided. Use to distinguish None from 'not provided'."""

    def __bool__(self) -> Literal[False]: ...
    def __repr__(self) -> Literal["Unset"]: ...
    def __str__(self) -> Literal["Unset"]: ...
    def __reduce__(self) -> tuple[type[UnsetType], tuple[()]]: ...
```

**Use Cases:**

- **Function Defaults**: Distinguish "user passed None" from "user didn't pass argument"
- **Partial Updates**: PATCH operations where missing fields should be ignored
- **Optional Parameters**: When `None` is a valid value distinct from "not provided"
- **Lazy Initialization**: Mark fields awaiting explicit assignment

**Examples:**

```python
from lionherd_core.types import Unset, MaybeUnset

# Function with optional parameter where None is valid
def update_config(
    value: MaybeUnset[str | None] = Unset
) -> str:
    if value is Unset:
        return "No update (keep existing value)"
    elif value is None:
        return "Explicitly clear value (set to None)"
    else:
        return f"Update to: {value}"

print(update_config())           # "No update (keep existing value)"
print(update_config(None))       # "Explicitly clear value (set to None)"
print(update_config("new"))      # "Update to: new"

# Partial update pattern (PATCH endpoint)
class PartialUpdate:
    name: MaybeUnset[str] = Unset
    age: MaybeUnset[int | None] = Unset
    email: MaybeUnset[str] = Unset

update = PartialUpdate()
update.name = "Alice"  # Update name
update.age = None      # Explicitly clear age
# email remains Unset (no change)

# Apply updates
for field in ["name", "age", "email"]:
    value = getattr(update, field)
    if value is not Unset:
        print(f"Update {field} to {value}")
# Output:
# Update name to Alice
# Update age to None
```

### MaybeSentinel (Union Type)

**Type alias for values that may be any sentinel type.**

```python
from lionherd_core.types import MaybeSentinel

MaybeSentinel[T] = T | UndefinedType | UnsetType
```

**Use Case:**

Generic functions that accept both `Undefined` and `Unset` sentinels.

**Example:**

```python
from lionherd_core.types import MaybeSentinel, Undefined, Unset

def process(value: MaybeSentinel[str]) -> str:
    if value is Undefined:
        return "Undefined"
    elif value is Unset:
        return "Unset"
    else:
        return f"Value: {value}"

print(process(Undefined))  # "Undefined"
print(process(Unset))      # "Unset"
print(process("data"))     # "Value: data"
```

## Helper Functions

### is_sentinel()

**Check if a value is any sentinel (Undefined or Unset).**

**Signature:**

```python
def is_sentinel(
    value: Any,
    *,
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> bool: ...
```

**Parameters:**

**value** : Any

Value to check.

**none_as_sentinel** : bool, default False

If `True`, treat `None` as a sentinel value.

**empty_as_sentinel** : bool, default False

If `True`, treat empty collections (`()`, `[]`, `{}`, `set()`, `frozenset()`, `""`) as sentinel values.

**Returns:**

- bool: `True` if value is `Undefined`, `Unset`, or matches additional sentinel criteria

**Examples:**

```python
from lionherd_core.types import is_sentinel, Undefined, Unset

# Basic sentinel checks
print(is_sentinel(Undefined))  # True
print(is_sentinel(Unset))      # True
print(is_sentinel(None))       # False
print(is_sentinel("value"))    # False

# Treat None as sentinel
print(is_sentinel(None, none_as_sentinel=True))  # True

# Treat empty collections as sentinels
print(is_sentinel([], empty_as_sentinel=True))   # True
print(is_sentinel({}, empty_as_sentinel=True))   # True
print(is_sentinel("", empty_as_sentinel=True))   # True
print(is_sentinel((), empty_as_sentinel=True))   # True

# Combined flags
print(is_sentinel(None, none_as_sentinel=True, empty_as_sentinel=True))  # True
print(is_sentinel([], none_as_sentinel=True, empty_as_sentinel=True))    # True

# Non-empty values
print(is_sentinel([1], empty_as_sentinel=True))  # False
print(is_sentinel("x", empty_as_sentinel=True))  # False
```

**Use Cases:**

- **Validation**: Check if optional parameter was provided
- **Default Resolution**: Determine if default value should be used
- **Filtering**: Remove sentinel values from collections

### not_sentinel()

**Type-narrowing check: verify value is NOT a sentinel.**

**Signature:**

```python
def not_sentinel(
    value: T | UndefinedType | UnsetType,
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> TypeGuard[T]: ...
```

**Parameters:**

**value** : MaybeSentinel[T]

Value to check (type-narrows from `MaybeSentinel[T]` to `T`).

**none_as_sentinel** : bool, default False

If `True`, treat `None` as a sentinel value.

**empty_as_sentinel** : bool, default False

If `True`, treat empty collections as sentinel values.

**Returns:**

- TypeGuard[T]: `True` if value is NOT a sentinel (type checkers narrow to `T`)

**Examples:**

```python
from lionherd_core.types import not_sentinel, Undefined, Unset, MaybeSentinel

# Type narrowing (mypy/pyright understand this)
def process(value: MaybeSentinel[str]) -> str:
    if not_sentinel(value):
        # Type checker knows value is str here
        return value.upper()  # OK (str method)
    else:
        # value is UndefinedType | UnsetType here
        return "No value"

print(process("hello"))    # "HELLO"
print(process(Undefined))  # "No value"
print(process(Unset))      # "No value"

# Filtering with type safety
values: list[MaybeSentinel[int]] = [1, 2, Undefined, 3, Unset]
real_values = [v for v in values if not_sentinel(v)]
# Type: list[int] (sentinels filtered out)
print(real_values)  # [1, 2, 3]

# With none_as_sentinel
def handle_optional(value: str | None | UnsetType) -> str:
    if not_sentinel(value, none_as_sentinel=True):
        # value is str here (None and Unset excluded)
        return value.upper()
    else:
        # value is None | UnsetType
        return "Missing"

print(handle_optional("test"))  # "TEST"
print(handle_optional(None))    # "Missing"
print(handle_optional(Unset))   # "Missing"
```

**Type Safety:**

This is a `TypeGuard` function - type checkers (mypy, pyright) understand that if it returns `True`, the value's type is narrowed from `MaybeSentinel[T]` to `T`.

## Type Aliases

### Complete Reference

```python
from typing import TypeVar, TypeAlias
from lionherd_core.types import UndefinedType, UnsetType

T = TypeVar("T")

# Maybe undefined (value or Undefined)
MaybeUndefined: TypeAlias = T | UndefinedType

# Maybe unset (value or Unset)
MaybeUnset: TypeAlias = T | UnsetType

# Maybe any sentinel (value or Undefined or Unset)
MaybeSentinel: TypeAlias = T | UndefinedType | UnsetType
```

### Usage in Annotations

```python
from lionherd_core.types import MaybeUndefined, MaybeUnset, MaybeSentinel

# Optional field (may not exist)
def get_field(data: dict) -> MaybeUndefined[str]:
    return data.get("field", Undefined)

# Optional parameter (None is valid, distinct from not provided)
def update(value: MaybeUnset[str | None] = Unset) -> None:
    if value is not Unset:
        print(f"Update to: {value}")

# Generic sentinel handling
def process(value: MaybeSentinel[int]) -> int:
    if value is Undefined or value is Unset:
        return 0  # Default
    return value
```

## Usage Patterns

### Pattern 1: Optional Function Parameters

**Problem**: Distinguish "user passed None" from "user didn't pass anything".

```python
from lionherd_core.types import Unset, MaybeUnset

def configure(
    host: str,
    port: MaybeUnset[int] = Unset,
    timeout: MaybeUnset[float | None] = Unset,
) -> dict:
    config = {"host": host}

    # Port: if provided, use it; otherwise default to 8080
    if port is not Unset:
        config["port"] = port
    else:
        config["port"] = 8080

    # Timeout: None means "no timeout", Unset means "use default 30s"
    if timeout is Unset:
        config["timeout"] = 30.0
    elif timeout is None:
        config["timeout"] = None  # Infinite timeout
    else:
        config["timeout"] = timeout

    return config

# Usage
print(configure("localhost"))
# {'host': 'localhost', 'port': 8080, 'timeout': 30.0}

print(configure("localhost", port=9000))
# {'host': 'localhost', 'port': 9000, 'timeout': 30.0}

print(configure("localhost", timeout=None))
# {'host': 'localhost', 'port': 8080, 'timeout': None}

print(configure("localhost", timeout=60.0))
# {'host': 'localhost', 'port': 8080, 'timeout': 60.0}
```

### Pattern 2: Partial Updates (PATCH Semantics)

**Problem**: PATCH endpoints should only update provided fields.

```python
from lionherd_core.types import Unset, MaybeUnset
from pydantic import BaseModel

class UserUpdate(BaseModel):
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str | None] = Unset
    age: MaybeUnset[int] = Unset

class User(BaseModel):
    name: str
    email: str | None
    age: int

def apply_update(user: User, update: UserUpdate) -> User:
    """Apply partial update, only modifying provided fields."""
    for field in ["name", "email", "age"]:
        value = getattr(update, field)
        if value is not Unset:
            setattr(user, field, value)
    return user

# Original user
user = User(name="Alice", email="alice@example.com", age=30)

# Update only email (set to None = clear it)
update = UserUpdate(email=None)
user = apply_update(user, update)
print(user)
# User(name='Alice', email=None, age=30)

# Update only name (email and age unchanged)
update = UserUpdate(name="Alice Smith")
user = apply_update(user, update)
print(user)
# User(name='Alice Smith', email=None, age=30)

# Update multiple fields
update = UserUpdate(email="alice.smith@example.com", age=31)
user = apply_update(user, update)
print(user)
# User(name='Alice Smith', email='alice.smith@example.com', age=31)
```

### Pattern 3: Dictionary Key Existence

**Problem**: Distinguish "key missing" from "key present with None value".

```python
from lionherd_core.types import Undefined, MaybeUndefined

def safe_get(data: dict, key: str) -> MaybeUndefined[Any]:
    """Get value with explicit missing-key indication."""
    return data.get(key, Undefined)

# Test data
config = {
    "host": "localhost",
    "port": 8080,
    "timeout": None,  # Explicitly set to None
}

# Check each key
host = safe_get(config, "host")
if host is Undefined:
    print("Host not configured")
else:
    print(f"Host: {host}")
# Output: Host: localhost

timeout = safe_get(config, "timeout")
if timeout is Undefined:
    print("Timeout not configured (use default)")
elif timeout is None:
    print("Timeout explicitly disabled")
else:
    print(f"Timeout: {timeout}s")
# Output: Timeout explicitly disabled

retries = safe_get(config, "retries")
if retries is Undefined:
    print("Retries not configured (use default)")
elif retries is None:
    print("Retries explicitly disabled")
else:
    print(f"Retries: {retries}")
# Output: Retries not configured (use default)
```

### Pattern 4: Type-Safe Filtering

**Problem**: Filter sentinel values while maintaining type safety.

```python
from lionherd_core.types import Undefined, Unset, MaybeSentinel, not_sentinel

# Data with sentinels
data: list[MaybeSentinel[int]] = [1, 2, Undefined, 3, Unset, 4, None]

# Filter out sentinels (type-safe)
real_values = [x for x in data if not_sentinel(x)]
# Type: list[int] (mypy knows sentinels are excluded)
print(real_values)  # [1, 2, 3, 4, None]

# Filter sentinels AND None
real_values_no_none = [
    x for x in data
    if not_sentinel(x, none_as_sentinel=True)
]
# Type: list[int] (mypy knows both sentinels and None excluded)
print(real_values_no_none)  # [1, 2, 3, 4]

# Filter empty values too
data2: list[MaybeSentinel[list[int]]] = [
    [1, 2], [], Undefined, [3], Unset
]
non_empty = [
    x for x in data2
    if not_sentinel(x, empty_as_sentinel=True)
]
# Type: list[list[int]]
print(non_empty)  # [[1, 2], [3]]
```

### Pattern 5: Conditional Defaults

**Problem**: Resolve defaults only when value not provided.

```python
from lionherd_core.types import Unset, MaybeUnset

class Config:
    def __init__(
        self,
        debug: MaybeUnset[bool] = Unset,
        log_level: MaybeUnset[str] = Unset,
    ):
        # Debug: default depends on environment
        if debug is Unset:
            import os
            self.debug = os.getenv("DEBUG", "0") == "1"
        else:
            self.debug = debug

        # Log level: default depends on debug flag
        if log_level is Unset:
            self.log_level = "DEBUG" if self.debug else "INFO"
        else:
            self.log_level = log_level

# Usage
config1 = Config()
print(config1.debug, config1.log_level)
# False, INFO (environment defaults)

config2 = Config(debug=True)
print(config2.debug, config2.log_level)
# True, DEBUG (debug=True triggers DEBUG log level)

config3 = Config(debug=True, log_level="WARNING")
print(config3.debug, config3.log_level)
# True, WARNING (explicit log_level overrides debug-based default)
```

## Common Pitfalls

### Pitfall 1: Using `==` Instead of `is`

**Issue**: Sentinels must be compared with `is`, not `==`.

```python
from lionherd_core.types import Undefined, Unset

value = Undefined

# WRONG
if value == Undefined:  # Works but semantically incorrect
    print("Undefined")

# CORRECT
if value is Undefined:  # Proper identity check
    print("Undefined")
```

**Rationale**: Sentinels are singletons. Identity checks (`is`) are faster and more explicit about singleton semantics.

### Pitfall 2: Forgetting Sentinels are Falsy

**Issue**: Sentinel values evaluate to `False` in boolean context.

```python
from lionherd_core.types import Undefined, Unset

def process(value):
    if value:  # WRONG - both Undefined and False trigger this
        return value
    return "default"

print(process(Undefined))  # "default" (unexpected if you wanted to detect sentinel)
print(process(False))      # "default" (same behavior)

# CORRECT - explicit sentinel check
def process(value):
    if value is Undefined:
        return "undefined"
    elif value:
        return value
    return "default"

print(process(Undefined))  # "undefined"
print(process(False))      # "default"
print(process("value"))    # "value"
```

### Pitfall 3: Mixing Undefined and Unset Without Documentation

**Issue**: Using both sentinels without clear semantic distinction confuses users.

```python
# UNCLEAR
def update(name=Unset, email=Undefined):  # Why different sentinels?
    ...

# CLEAR - document the distinction
def update(
    name: MaybeUnset[str] = Unset,    # Unset = no change
    email: MaybeUnset[str] = Unset,   # Unset = no change
):
    """
    Update user fields.

    Args:
        name: New name, or Unset to keep existing
        email: New email, or Unset to keep existing
    """
    ...
```

**Best Practice**: Use `Unset` consistently for "not provided" semantics. Reserve `Undefined` for "missing from namespace" (dictionary lookups, schema validation).

### Pitfall 4: Overusing Sentinels

**Issue**: Using sentinels where `None` or `Optional[T]` suffices.

```python
# OVERCOMPLICATED
def greet(name: MaybeUnset[str] = Unset) -> str:
    if name is Unset:
        return "Hello, stranger"
    return f"Hello, {name}"

# SIMPLER (None works fine here)
def greet(name: str | None = None) -> str:
    if name is None:
        return "Hello, stranger"
    return f"Hello, {name}"
```

**When to use sentinels**: Only when you need to distinguish `None` (valid value) from "not provided".

### Pitfall 5: Forgetting pickle/deepcopy Preservation

**Issue**: Assuming sentinel identity breaks across serialization boundaries.

```python
import pickle
from lionherd_core.types import Undefined

# Sentinels ARE preserved across pickle/deepcopy
data = {"key": Undefined}
pickled = pickle.dumps(data)
restored = pickle.loads(pickled)

assert restored["key"] is Undefined  # True (identity preserved)

# But NOT across process boundaries without shared module
# If you pass Undefined over RPC, receiving side must import from lionherd_core.types
```

**Best Practice**: Sentinel identity is preserved within Python runtime (pickle/deepcopy). For cross-process/cross-language communication, serialize to explicit marker strings (e.g., `"__UNDEFINED__"`).

## Design Rationale

### Why Singleton Pattern?

Sentinels use singleton pattern to enable **identity checks** (`is`) instead of equality checks (`==`). This ensures:

1. **Performance**: `is` checks are faster than `==` (single pointer comparison)
2. **Clarity**: `is Undefined` clearly expresses "is this the sentinel?" vs `== Undefined` (equality)
3. **Safety**: Identity preserved across copy/deepcopy/pickle prevents subtle bugs

### Why Falsy Sentinels?

Sentinels evaluate to `False` because they represent **absence of meaningful value**:

```python
# Allows concise default handling
config = config_value if config_value else default_config
# If config_value is Undefined/Unset, use default

# But explicit checks are safer
if config_value is not Undefined:
    config = config_value
else:
    config = default_config
```

Falsy behavior aligns with `None`, empty collections, and other "absent value" types in Python.

### Why Two Sentinel Types?

`Undefined` and `Unset` serve different semantic roles:

1. **Undefined**: "This key/field doesn't exist in the namespace"
   - Dictionary lookups where key missing
   - Schema validation (required field absent)
   - Deserialization (field not in source data)

2. **Unset**: "This parameter wasn't provided by caller"
   - Function defaults (distinguish from `None`)
   - Partial updates (PATCH semantics)
   - Lazy initialization (value pending)

**Example** (demonstrates why both are needed):

```python
from lionherd_core.types import Undefined, Unset

# API request: {"name": "Alice"}
# email field is UNDEFINED (not in request)

# Function call: update_user(name="Alice")
# email parameter is UNSET (not passed by caller)

def update_user(
    name: str,
    email: str | None | UnsetType = Unset
):
    # Get existing user data
    existing = {"name": "Bob", "email": "bob@example.com"}

    # Update name (always provided)
    existing["name"] = name

    # Update email only if provided
    if email is not Unset:
        existing["email"] = email  # Could be None (clear) or str (update)

    return existing

# Request didn't include email, function call didn't provide email
# Result: email unchanged in database
```

### Why TypeGuard for not_sentinel()?

`not_sentinel()` uses `TypeGuard` to enable **type narrowing** in type checkers:

```python
def process(value: MaybeSentinel[str]) -> str:
    if not_sentinel(value):
        # Type checker knows: value is str (not Undefined | Unset)
        return value.upper()  # OK - str method available
    else:
        # Type checker knows: value is Undefined | Unset
        return str(value)  # Must convert to str
```

Without `TypeGuard`, type checkers wouldn't narrow the type, requiring manual type assertions.

## See Also

- **Related Types**:
  - [Optional](https://docs.python.org/3/library/typing.html#typing.Optional): Standard library optional type (use when `None` suffices)
  - HashableModel (documentation pending): Content-based hashing (sentinels are identity-based)
- **Related Guides**:
  - [Type Safety Guide](../../user_guide/type_safety.md): Type narrowing and TypeGuards
  - [API Design Guide](../../user_guide/api_design.md): When to use sentinels vs Optional
- **Related Patterns**:
  - [HTTP Patterns](../../patterns/http.md): PATCH endpoint semantics with `Unset`

## Examples

### Example 1: Three-State Configuration

```python
from lionherd_core.types import Unset, MaybeUnset

class DatabaseConfig:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        ssl: MaybeUnset[bool] = Unset,
        timeout: MaybeUnset[float | None] = Unset,
    ):
        self.host = host
        self.port = port

        # SSL: Unset → auto-detect, True → force, False → disable
        if ssl is Unset:
            self.ssl = self.port == 5432  # Auto-enable for standard port
        else:
            self.ssl = ssl

        # Timeout: Unset → default, None → infinite, float → explicit
        if timeout is Unset:
            self.timeout = 30.0
        elif timeout is None:
            self.timeout = float("inf")
        else:
            self.timeout = timeout

# Usage
config1 = DatabaseConfig()
print(config1.ssl, config1.timeout)  # True, 30.0 (defaults)

config2 = DatabaseConfig(ssl=False)
print(config2.ssl, config2.timeout)  # False, 30.0 (SSL disabled)

config3 = DatabaseConfig(timeout=None)
print(config3.ssl, config3.timeout)  # True, inf (infinite timeout)
```

### Example 2: Schema Validation with Undefined

```python
from lionherd_core.types import Undefined, MaybeUndefined
from pydantic import BaseModel, field_validator

class UserSchema(BaseModel):
    name: str
    email: str | None
    age: int | None = None

def validate_user_data(data: dict) -> dict:
    """Validate user data, distinguishing missing vs None."""
    errors = {}

    # Name is required
    name = data.get("name", Undefined)
    if name is Undefined:
        errors["name"] = "Field required"
    elif not isinstance(name, str):
        errors["name"] = "Must be string"

    # Email is required (but can be None)
    email = data.get("email", Undefined)
    if email is Undefined:
        errors["email"] = "Field required"
    elif email is not None and not isinstance(email, str):
        errors["email"] = "Must be string or null"

    # Age is optional (missing is OK, null is OK)
    age = data.get("age", Undefined)
    if age is not Undefined and age is not None and not isinstance(age, int):
        errors["age"] = "Must be integer or null"

    return errors

# Test cases
print(validate_user_data({}))
# {'name': 'Field required', 'email': 'Field required'}

print(validate_user_data({"name": "Alice"}))
# {'email': 'Field required'}

print(validate_user_data({"name": "Alice", "email": None}))
# {} (valid - email can be None)

print(validate_user_data({"name": "Alice", "email": None, "age": "invalid"}))
# {'age': 'Must be integer or null'}
```

### Example 3: Batch Operations with Sentinels

```python
from lionherd_core.types import Undefined, Unset, is_sentinel

# Batch update records
updates = [
    {"id": 1, "name": "Alice", "email": Undefined},  # Don't change email
    {"id": 2, "name": Unset, "email": "bob@example.com"},  # Don't change name
    {"id": 3, "name": "Charlie", "email": None},  # Clear email
]

def apply_batch_updates(updates: list[dict]) -> None:
    for update in updates:
        record_id = update["id"]
        changes = {}

        for key, value in update.items():
            if key == "id":
                continue

            if is_sentinel(value):
                print(f"Record {record_id}: Skip {key} (sentinel)")
            elif value is None:
                print(f"Record {record_id}: Clear {key}")
                changes[key] = None
            else:
                print(f"Record {record_id}: Update {key} to {value}")
                changes[key] = value

        print(f"  Applied changes: {changes}\n")

apply_batch_updates(updates)
# Output:
# Record 1: Update name to Alice
# Record 1: Skip email (sentinel)
#   Applied changes: {'name': 'Alice'}
#
# Record 2: Skip name (sentinel)
# Record 2: Update email to bob@example.com
#   Applied changes: {'email': 'bob@example.com'}
#
# Record 3: Update name to Charlie
# Record 3: Clear email
#   Applied changes: {'name': 'Charlie', 'email': None}
```

### Example 4: Type-Safe API Client

```python
from lionherd_core.types import Unset, MaybeUnset, not_sentinel
from typing import Any

class APIClient:
    def update_resource(
        self,
        resource_id: str,
        name: MaybeUnset[str] = Unset,
        description: MaybeUnset[str | None] = Unset,
        tags: MaybeUnset[list[str]] = Unset,
        metadata: MaybeUnset[dict[str, Any]] = Unset,
    ) -> dict:
        """Update resource with partial updates (PATCH semantics)."""
        payload = {"id": resource_id}

        # Only include provided fields in payload
        if not_sentinel(name):
            payload["name"] = name

        if not_sentinel(description):
            payload["description"] = description

        if not_sentinel(tags):
            payload["tags"] = tags

        if not_sentinel(metadata):
            payload["metadata"] = metadata

        print(f"PATCH /resources/{resource_id}")
        print(f"Payload: {payload}")
        return payload

# Usage
client = APIClient()

# Update only name
client.update_resource("res-123", name="New Name")
# PATCH /resources/res-123
# Payload: {'id': 'res-123', 'name': 'New Name'}

# Clear description, update tags
client.update_resource("res-123", description=None, tags=["v2", "prod"])
# PATCH /resources/res-123
# Payload: {'id': 'res-123', 'description': None, 'tags': ['v2', 'prod']}

# Update everything
client.update_resource(
    "res-123",
    name="Final Name",
    description="Updated desc",
    tags=[],
    metadata={"version": 2}
)
# PATCH /resources/res-123
# Payload: {'id': 'res-123', 'name': 'Final Name', 'description': 'Updated desc', 'tags': [], 'metadata': {'version': 2}}
```

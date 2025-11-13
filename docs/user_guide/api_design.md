# API Design Guide

> Best practices for designing type-safe, intuitive APIs with sentinels and optional parameters

## Overview

lionherd-core provides **sentinel values** (Undefined, Unset) for building APIs with precise semantics around optional parameters, missing data, and partial updates. This guide covers when to use sentinels vs `Optional`, three-state logic patterns, and API design best practices.

**Key Topics:**

- When to use sentinels vs `Optional[T]`
- Three-state logic design (value / null / unset)
- Optional parameter patterns
- Partial update semantics (PATCH)
- Default value resolution strategies

**Who Should Read This:**

- API designers building libraries or frameworks
- Developers working with optional parameters and defaults
- Anyone designing functions where `None` is a valid value

---

## Sentinels vs Optional: Decision Framework

### Use `Optional[T]` When

#### 1. Absence is the only special case

```python
# ✅ CORRECT: None means "no value"
def get_user(user_id: str) -> User | None:
    """Get user by ID, returning None if not found."""
    ...

# No need for sentinel - None adequately represents absence
```

#### 2. Two states suffice

```python
# ✅ CORRECT: Two states (value | None)
def format_date(date: datetime | None = None) -> str:
    """Format date, using current date if None."""
    if date is None:
        date = datetime.now()
    return date.strftime("%Y-%m-%d")
```

#### 3. Public API simplicity matters

```python
# ✅ CORRECT: Simple public API
class Config:
    def __init__(self, debug: bool | None = None):
        """
        Args:
            debug: Enable debug mode (None = auto-detect from env)
        """
        if debug is None:
            import os
            self.debug = os.getenv("DEBUG") == "1"
        else:
            self.debug = debug
```

### Use Sentinels When

#### 1. `None` is a valid, distinct value

```python
from lionherd_core.types import Unset, MaybeUnset

# ✅ CORRECT: Three states (value | None | Unset)
def set_timeout(
    timeout: MaybeUnset[float | None] = Unset
) -> None:
    """Set timeout.

    Args:
        timeout: Seconds (None = infinite, Unset = keep current)
    """
    if timeout is Unset:
        pass  # Keep current timeout
    elif timeout is None:
        self.timeout = float("inf")  # None means infinite
    else:
        self.timeout = timeout
```

**Why sentinels?** Because `None` has a specific meaning (infinite timeout), distinct from "don't change current value".

#### 2. Partial updates (PATCH semantics)

```python
from lionherd_core.types import Unset, MaybeUnset

# ✅ CORRECT: Only update provided fields
def update_user(
    name: MaybeUnset[str] = Unset,
    email: MaybeUnset[str | None] = Unset,
    age: MaybeUnset[int] = Unset,
) -> User:
    """Update user fields (only provided fields are changed).

    Args:
        name: New name (Unset = no change)
        email: New email, None to clear (Unset = no change)
        age: New age (Unset = no change)
    """
    updates = {}
    if name is not Unset:
        updates["name"] = name
    if email is not Unset:
        updates["email"] = email
    if age is not Unset:
        updates["age"] = age

    return current_user.copy(update=updates)
```

**Why sentinels?** Distinguishes "field not provided" (Unset) from "clear field" (None).

#### 3. Missing vs present-but-null in data

```python
from lionherd_core.types import Undefined, MaybeUndefined

# ✅ CORRECT: Detect missing keys vs null values
def validate_required(data: dict, required_keys: list[str]) -> list[str]:
    """Validate required fields, distinguishing missing from null.

    Returns:
        List of missing field names
    """
    missing = []
    for key in required_keys:
        value = data.get(key, Undefined)
        if value is Undefined:
            missing.append(key)  # Key missing entirely
        elif value is None:
            pass  # Key present, set to null (might be valid)

    return missing

# Example
data = {"name": "Alice", "email": None}
print(validate_required(data, ["name", "email", "age"]))
# ["age"] - only "age" is missing (email is present but null)
```

**Why sentinels?** Distinguishes "key missing from dict" (Undefined) from "key present with null value" (None).

---

## Three-State Logic Patterns

### Pattern 1: Value / None / Not Provided

**Use Case**: Parameter where `None` is a valid value distinct from "not provided".

```python
from lionherd_core.types import Unset, MaybeUnset

def configure_logging(
    level: MaybeUnset[str | None] = Unset,
    format: MaybeUnset[str] = Unset,
) -> dict:
    """Configure logging.

    Args:
        level: Log level ("DEBUG", "INFO", etc.), None to disable,
               Unset to use default
        format: Log format string (Unset = use default)

    Returns:
        Logging configuration dict
    """
    config = {}

    # Level: None = disabled, Unset = default, str = explicit level
    if level is Unset:
        config["level"] = "INFO"  # Default
    elif level is None:
        config["level"] = None  # Disabled
    else:
        config["level"] = level  # Explicit

    # Format: Unset = default, str = explicit
    if format is not Unset:
        config["format"] = format
    else:
        config["format"] = "%(asctime)s - %(message)s"  # Default

    return config

# Usage examples
print(configure_logging())
# {"level": "INFO", "format": "%(asctime)s - %(message)s"}

print(configure_logging(level="DEBUG"))
# {"level": "DEBUG", "format": "%(asctime)s - %(message)s"}

print(configure_logging(level=None))
# {"level": None, "format": "%(asctime)s - %(message)s"}  # Disabled

print(configure_logging(level="WARNING", format="%(message)s"))
# {"level": "WARNING", "format": "%(message)s"}
```

### Pattern 2: Missing / Present-Null / Present-Value

**Use Case**: Validating API requests where field presence matters.

```python
from lionherd_core.types import Undefined, MaybeUndefined
from pydantic import BaseModel, field_validator

class UserUpdate(BaseModel):
    """User update request with field presence tracking."""

    @staticmethod
    def from_request(data: dict) -> "UserUpdate":
        """Create from request dict, tracking field presence."""
        return UserUpdate(
            name=data.get("name", Undefined),
            email=data.get("email", Undefined),
            bio=data.get("bio", Undefined),
        )

    name: MaybeUndefined[str] = Undefined
    email: MaybeUndefined[str | None] = Undefined
    bio: MaybeUndefined[str | None] = Undefined

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v is Undefined:
            return v  # Not provided - skip validation
        if v is None or not v.strip():
            raise ValueError("Name cannot be empty")
        return v

def apply_update(user: dict, update: UserUpdate) -> dict:
    """Apply update, distinguishing missing/null/value."""
    updated = user.copy()

    # Name: must be provided and non-null
    if update.name is Undefined:
        raise ValueError("Name is required")
    updated["name"] = update.name

    # Email: Undefined = no change, None = clear, str = update
    if update.email is not Undefined:
        updated["email"] = update.email

    # Bio: Undefined = no change, None = clear, str = update
    if update.bio is not Undefined:
        updated["bio"] = update.bio

    return updated

# Example usage
user = {"name": "Alice", "email": "alice@example.com", "bio": "Engineer"}

# Update only email (clear it)
update = UserUpdate.from_request({"name": "Alice", "email": None})
print(apply_update(user, update))
# {"name": "Alice", "email": None, "bio": "Engineer"}

# Update bio, leave email unchanged
update = UserUpdate.from_request({"name": "Alice", "bio": "Senior Engineer"})
print(apply_update(user, update))
# {"name": "Alice", "email": "alice@example.com", "bio": "Senior Engineer"}
```

### Pattern 3: Conditional Defaults

**Use Case**: Default value depends on other parameters or context.

```python
from lionherd_core.types import Unset, MaybeUnset

class Database:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        ssl: MaybeUnset[bool] = Unset,
        timeout: MaybeUnset[float] = Unset,
    ):
        """Initialize database connection.

        Args:
            host: Database host
            port: Database port
            ssl: Enable SSL (Unset = auto-detect based on port)
            timeout: Connection timeout (Unset = auto-detect based on ssl)
        """
        self.host = host
        self.port = port

        # SSL: auto-enable for standard PostgreSQL port
        if ssl is Unset:
            self.ssl = (port == 5432)
        else:
            self.ssl = ssl

        # Timeout: longer for SSL connections
        if timeout is Unset:
            self.timeout = 60.0 if self.ssl else 30.0
        else:
            self.timeout = timeout

# Usage
db1 = Database()
print(db1.ssl, db1.timeout)  # True, 60.0 (auto-detected)

db2 = Database(port=5433)
print(db2.ssl, db2.timeout)  # False, 30.0 (non-standard port)

db3 = Database(ssl=True, timeout=120.0)
print(db3.ssl, db3.timeout)  # True, 120.0 (explicit)
```

---

## Optional Parameter Design

### Guideline 1: Explicit is Better Than Implicit

**❌ Poor: Ambiguous `None` meaning**

```python
def create_user(name: str, email: str | None = None) -> User:
    """Create user.

    Args:
        email: User email (None = ?)  # Unclear what None means
    """
    if email is None:
        email = generate_temp_email()  # Or is None valid?
    ...
```

#### ✅ Better: Use sentinel for "not provided"

```python
from lionherd_core.types import Unset, MaybeUnset

def create_user(
    name: str,
    email: MaybeUnset[str | None] = Unset
) -> User:
    """Create user.

    Args:
        name: User name (required)
        email: User email, None if no email, Unset to generate temp email
    """
    if email is Unset:
        email = generate_temp_email()  # Clear: Unset means generate
    # email can still be None (user explicitly has no email)
    ...
```

### Guideline 2: Document Three States

Always document what each state means:

```python
from lionherd_core.types import Unset, MaybeUnset

def set_cache_ttl(
    key: str,
    ttl: MaybeUnset[int | None] = Unset
) -> None:
    """Set cache TTL for key.

    Args:
        key: Cache key
        ttl: Time-to-live in seconds. Three states:
             - int: Expire after N seconds
             - None: Never expire (permanent)
             - Unset: Use default TTL (3600 seconds)
    """
    if ttl is Unset:
        ttl = 3600  # Default
    # None and int handled by cache implementation
    cache.set(key, value, ttl=ttl)
```

### Guideline 3: Minimize Sentinel Usage in Public APIs

**Internal APIs**: Sentinels are fine for precision

```python
# Internal API - sentinels for precise semantics
from lionherd_core.types import Unset, MaybeUnset

def _internal_update(
    field: MaybeUnset[str | None] = Unset
) -> None:
    """Internal: precise field update semantics."""
    ...
```

**Public APIs**: Consider simpler alternatives

```python
# Public API - simpler interface
def update(field: str | None = None) -> None:
    """Public: simple optional parameter."""
    if field is None:
        field = get_default()
    ...

# Or use separate methods
def update_field(field: str) -> None:
    """Update field to value."""
    ...

def clear_field() -> None:
    """Clear field (set to None)."""
    ...

def reset_field() -> None:
    """Reset field to default."""
    ...
```

---

## Default Value Resolution Strategies

### Strategy 1: Lazy Defaults (Conditional)

Compute defaults only when needed:

```python
from lionherd_core.types import Unset, MaybeUnset

def fetch_data(
    url: str,
    timeout: MaybeUnset[float] = Unset,
    retries: MaybeUnset[int] = Unset,
) -> dict:
    """Fetch data with lazy default resolution.

    Args:
        url: URL to fetch
        timeout: Request timeout (Unset = auto-detect based on URL)
        retries: Retry count (Unset = auto-detect based on timeout)
    """
    # Lazy timeout resolution
    if timeout is Unset:
        timeout = 60.0 if url.startswith("https") else 30.0

    # Lazy retries resolution (depends on timeout)
    if retries is Unset:
        retries = 5 if timeout > 30.0 else 3

    return _fetch(url, timeout=timeout, retries=retries)
```

### Strategy 2: Factory Functions

Use factories for expensive defaults:

```python
from lionherd_core.types import Unset, MaybeUnset

def create_config(
    cache: MaybeUnset[dict] = Unset,
    logger: MaybeUnset[Logger] = Unset,
) -> Config:
    """Create configuration with factory defaults.

    Args:
        cache: Cache dict (Unset = create new dict)
        logger: Logger instance (Unset = create default logger)
    """
    if cache is Unset:
        cache = {}  # Factory: new dict per call

    if logger is Unset:
        # Factory: expensive logger setup only if needed
        logger = create_default_logger()

    return Config(cache=cache, logger=logger)
```

### Strategy 3: Cascading Defaults

Defaults cascade from environment/config:

```python
from lionherd_core.types import Unset, MaybeUnset
import os

class AppConfig:
    def __init__(
        self,
        debug: MaybeUnset[bool] = Unset,
        log_level: MaybeUnset[str] = Unset,
    ):
        """App configuration with cascading defaults.

        Defaults cascade: explicit > env var > config file > hardcoded
        """
        # Debug: explicit > env var > False
        if debug is Unset:
            debug = os.getenv("DEBUG", "0") == "1"
        self.debug = debug

        # Log level: explicit > env var > debug-based > INFO
        if log_level is Unset:
            log_level = os.getenv("LOG_LEVEL")
            if log_level is None:
                log_level = "DEBUG" if self.debug else "INFO"
        self.log_level = log_level

# Usage
config1 = AppConfig()  # All defaults
config2 = AppConfig(debug=True)  # Explicit debug, log level auto-set
config3 = AppConfig(debug=True, log_level="WARNING")  # All explicit
```

---

## API Design Best Practices

### 1. Consistent Sentinel Usage

Use the same sentinel consistently across related APIs:

```python
from lionherd_core.types import Unset, MaybeUnset

class UserManager:
    """Consistent Unset usage across all methods."""

    def create_user(
        self,
        name: str,
        email: MaybeUnset[str | None] = Unset,
        phone: MaybeUnset[str | None] = Unset,
    ) -> User:
        """Create user (Unset = field not provided)."""
        ...

    def update_user(
        self,
        user_id: str,
        name: MaybeUnset[str] = Unset,
        email: MaybeUnset[str | None] = Unset,
        phone: MaybeUnset[str | None] = Unset,
    ) -> User:
        """Update user (Unset = no change)."""
        ...

    def get_user(
        self,
        user_id: str,
        include_email: bool = True,
        include_phone: bool = True,
    ) -> User:
        """Get user (booleans for options, not sentinels)."""
        ...
```

**Rationale**: Consistent sentinel semantics reduce cognitive load.

### 2. Progressive Disclosure

Start simple, add sentinels only when needed:

```python
# Version 1: Simple API
def configure(host: str, port: int = 8080) -> Config:
    """Simple configuration."""
    return Config(host=host, port=port)

# Version 2: Add optional timeout (None means infinite)
def configure(
    host: str,
    port: int = 8080,
    timeout: float | None = 30.0
) -> Config:
    """Configuration with optional timeout."""
    return Config(host=host, port=port, timeout=timeout)

# Version 3: Need to distinguish "keep current timeout" from "set timeout"
from lionherd_core.types import Unset, MaybeUnset

def configure(
    host: str,
    port: int = 8080,
    timeout: MaybeUnset[float | None] = Unset
) -> Config:
    """Configuration with three-state timeout.

    Args:
        timeout: Seconds (None = infinite, Unset = keep current)
    """
    ...
```

### 3. Type Hints Document Behavior

Use type hints to communicate three-state semantics:

```python
from lionherd_core.types import Unset, MaybeUnset

# ✅ CLEAR: Type hint shows three states
def update(
    field: MaybeUnset[str | None] = Unset
) -> None:
    """Update field.

    Args:
        field: New value, None to clear, Unset to skip update
    """
    ...

# ❌ UNCLEAR: Type hint doesn't show sentinel usage
def update(field=Unset) -> None:
    """Update field (missing type hint)."""
    ...
```

### 4. Helper Functions for Common Patterns

Extract sentinel handling into helpers:

```python
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

def apply_updates(current: dict, **updates: MaybeUnset[Any]) -> dict:
    """Apply updates, skipping Unset values.

    Args:
        current: Current values
        **updates: New values (Unset = no change)

    Returns:
        Updated dict
    """
    result = current.copy()
    for key, value in updates.items():
        if not_sentinel(value):
            result[key] = value
    return result

# Usage
user = {"name": "Alice", "email": "alice@example.com", "age": 30}

# Update only email
updated = apply_updates(user, email="alice@newdomain.com")
print(updated)
# {"name": "Alice", "email": "alice@newdomain.com", "age": 30}

# Update name and age, skip email (Unset)
updated = apply_updates(user, name="Alice Smith", age=31, email=Unset)
print(updated)
# {"name": "Alice Smith", "email": "alice@example.com", "age": 31}
```

### 5. Validation Integration

Combine sentinels with Pydantic for validation:

```python
from pydantic import BaseModel, Field, field_validator
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class UserUpdate(BaseModel):
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str | None] = Unset
    age: MaybeUnset[int] = Unset

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not_sentinel(v):  # Only validate if provided
            if not v.strip():
                raise ValueError("Name cannot be empty")
        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if not_sentinel(v):  # Only validate if provided
            if v < 0 or v > 150:
                raise ValueError("Invalid age")
        return v

def update_user(user_id: str, update: UserUpdate) -> User:
    """Update user with validated fields."""
    updates = {}
    if not_sentinel(update.name):
        updates["name"] = update.name
    if not_sentinel(update.email):
        updates["email"] = update.email
    if not_sentinel(update.age):
        updates["age"] = update.age

    return _apply_updates(user_id, updates)
```

---

## Common Anti-Patterns

### Anti-Pattern 1: Overusing Sentinels

#### ❌ Bad: Sentinels where `None` suffices

```python
from lionherd_core.types import Unset, MaybeUnset

def greet(name: MaybeUnset[str] = Unset) -> str:
    """Greet user."""
    if name is Unset:
        return "Hello, stranger"
    return f"Hello, {name}"
```

#### ✅ Good: Use `None` for simple optional

```python
def greet(name: str | None = None) -> str:
    """Greet user."""
    if name is None:
        return "Hello, stranger"
    return f"Hello, {name}"
```

### Anti-Pattern 2: Inconsistent Sentinel Semantics

#### ❌ Bad: Mixed sentinel meanings

```python
from lionherd_core.types import Unset, Undefined

def update(
    name=Unset,    # Unset = no change
    email=Undefined,  # Undefined = no change?
) -> None:
    """Confusing: why two sentinels?"""
    ...
```

#### ✅ Good: Consistent usage

```python
from lionherd_core.types import Unset, MaybeUnset

def update(
    name: MaybeUnset[str] = Unset,
    email: MaybeUnset[str] = Unset,
) -> None:
    """Clear: both use Unset for "no change"."""
    ...
```

### Anti-Pattern 3: Hidden Sentinel Logic

#### ❌ Bad: Sentinel logic buried in implementation

```python
def configure(timeout=Unset) -> Config:
    """Configure timeout (no docs about Unset behavior)."""
    if timeout is Unset:
        timeout = get_default_timeout()
    ...
```

#### ✅ Good: Document sentinel behavior

```python
from lionherd_core.types import Unset, MaybeUnset

def configure(timeout: MaybeUnset[float] = Unset) -> Config:
    """Configure timeout.

    Args:
        timeout: Timeout in seconds (Unset = use default from config)
    """
    if timeout is Unset:
        timeout = get_default_timeout()
    ...
```

---

## Examples

### Example 1: Builder Pattern with Sentinels

```python
from lionherd_core.types import Unset, MaybeUnset, not_sentinel
from pydantic import BaseModel, Field

class RequestConfig(BaseModel):
    url: str
    method: str = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0
    retries: int = 3

class RequestBuilder:
    """Fluent request builder with sentinel-based updates."""

    def __init__(self, url: str):
        self.config = RequestConfig(url=url)

    def method(self, method: str) -> "RequestBuilder":
        """Set HTTP method."""
        self.config.method = method
        return self

    def headers(
        self,
        **headers: MaybeUnset[str]
    ) -> "RequestBuilder":
        """Update headers (Unset = no change to that header)."""
        for key, value in headers.items():
            if not_sentinel(value):
                self.config.headers[key] = value
        return self

    def timeout(self, timeout: float) -> "RequestBuilder":
        """Set timeout."""
        self.config.timeout = timeout
        return self

    def retries(self, retries: int) -> "RequestBuilder":
        """Set retry count."""
        self.config.retries = retries
        return self

    def build(self) -> RequestConfig:
        """Build final config."""
        return self.config

# Usage
config = (
    RequestBuilder("https://api.example.com")
    .method("POST")
    .headers(Authorization="Bearer token", ContentType=Unset)  # Skip ContentType
    .timeout(60.0)
    .build()
)

print(config.model_dump())
# {
#     "url": "https://api.example.com",
#     "method": "POST",
#     "headers": {"Authorization": "Bearer token"},
#     "timeout": 60.0,
#     "retries": 3
# }
```

### Example 2: Partial Update API

```python
from lionherd_core.types import Unset, MaybeUnset, not_sentinel
from pydantic import BaseModel

class User(BaseModel):
    id: str
    name: str
    email: str | None
    bio: str | None
    active: bool

class UserService:
    def __init__(self):
        self.users: dict[str, User] = {}

    def create_user(self, name: str, email: str | None = None) -> User:
        """Create new user."""
        user = User(
            id=str(uuid.uuid4()),
            name=name,
            email=email,
            bio=None,
            active=True,
        )
        self.users[user.id] = user
        return user

    def update_user(
        self,
        user_id: str,
        name: MaybeUnset[str] = Unset,
        email: MaybeUnset[str | None] = Unset,
        bio: MaybeUnset[str | None] = Unset,
        active: MaybeUnset[bool] = Unset,
    ) -> User:
        """Update user fields (Unset = no change).

        Args:
            user_id: User ID
            name: New name (Unset = no change)
            email: New email, None to clear (Unset = no change)
            bio: New bio, None to clear (Unset = no change)
            active: New active status (Unset = no change)
        """
        user = self.users[user_id]
        updates = {}

        if not_sentinel(name):
            updates["name"] = name
        if not_sentinel(email):
            updates["email"] = email
        if not_sentinel(bio):
            updates["bio"] = bio
        if not_sentinel(active):
            updates["active"] = active

        updated_user = user.model_copy(update=updates)
        self.users[user_id] = updated_user
        return updated_user

# Usage
service = UserService()

user = service.create_user("Alice", email="alice@example.com")
print(user)
# User(id='...', name='Alice', email='alice@example.com', bio=None, active=True)

# Update only bio
user = service.update_user(user.id, bio="Software Engineer")
print(user)
# User(id='...', name='Alice', email='alice@example.com', bio='Software Engineer', active=True)

# Clear email, keep everything else
user = service.update_user(user.id, email=None)
print(user)
# User(id='...', name='Alice', email=None, bio='Software Engineer', active=True)
```

### Example 3: Configuration with Smart Defaults

```python
from lionherd_core.types import Unset, MaybeUnset
import os

class ServerConfig:
    """Server configuration with cascading defaults."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        workers: MaybeUnset[int] = Unset,
        debug: MaybeUnset[bool] = Unset,
        log_level: MaybeUnset[str] = Unset,
        ssl_cert: MaybeUnset[str | None] = Unset,
        ssl_key: MaybeUnset[str | None] = Unset,
    ):
        """Initialize server config with smart defaults.

        Args:
            host: Server host
            port: Server port
            workers: Worker count (Unset = CPU count)
            debug: Debug mode (Unset = from DEBUG env var)
            log_level: Log level (Unset = auto-detect from debug)
            ssl_cert: SSL cert path (Unset = auto-detect from env)
            ssl_key: SSL key path (Unset = auto-detect from env)
        """
        self.host = host
        self.port = port

        # Workers: default to CPU count
        if workers is Unset:
            import multiprocessing
            self.workers = multiprocessing.cpu_count()
        else:
            self.workers = workers

        # Debug: from env var or False
        if debug is Unset:
            self.debug = os.getenv("DEBUG", "0") == "1"
        else:
            self.debug = debug

        # Log level: depends on debug
        if log_level is Unset:
            self.log_level = "DEBUG" if self.debug else "INFO"
        else:
            self.log_level = log_level

        # SSL: from env vars or None
        if ssl_cert is Unset:
            self.ssl_cert = os.getenv("SSL_CERT")
        else:
            self.ssl_cert = ssl_cert

        if ssl_key is Unset:
            self.ssl_key = os.getenv("SSL_KEY")
        else:
            self.ssl_key = ssl_key

        # Auto-enable SSL if both cert and key provided
        self.ssl_enabled = bool(self.ssl_cert and self.ssl_key)

    def __repr__(self) -> str:
        return (
            f"ServerConfig(host={self.host!r}, port={self.port}, "
            f"workers={self.workers}, debug={self.debug}, "
            f"log_level={self.log_level!r}, ssl_enabled={self.ssl_enabled})"
        )

# Usage
config1 = ServerConfig()  # All defaults
print(config1)
# ServerConfig(host='localhost', port=8080, workers=8, debug=False, log_level='INFO', ssl_enabled=False)

config2 = ServerConfig(debug=True)  # Debug enabled, log_level auto-set to DEBUG
print(config2)
# ServerConfig(host='localhost', port=8080, workers=8, debug=True, log_level='DEBUG', ssl_enabled=False)

config3 = ServerConfig(workers=4, log_level="WARNING")  # Explicit overrides
print(config3)
# ServerConfig(host='localhost', port=8080, workers=4, debug=False, log_level='WARNING', ssl_enabled=False)
```

---

## See Also

- **Type Safety Guide**: Type narrowing with TypeGuards and sentinels
- **HTTP Patterns**: PATCH endpoint semantics with Unset
- **Validation Guide**: Validation patterns with Spec and Pydantic
- **Sentinel Values API**: Complete sentinel types reference

---

## References

- [REST API Design: PATCH vs PUT](https://www.rfc-editor.org/rfc/rfc5789)
- [Python API Design Best Practices](https://docs.python-guide.org/writing/style/)
- [Pydantic Field Semantics](https://docs.pydantic.dev/latest/concepts/fields/)

# Validation Guide

> Validation patterns with Spec, Pydantic integration, and custom validators

## Overview

lionherd-core provides **framework-agnostic field specifications** via `Spec`, enabling rich validation patterns across Pydantic, dataclasses, and attrs. This guide covers validation strategies, Pydantic integration, custom validators, and error handling patterns.

**Key Topics:**

- Validation patterns with Spec
- Pydantic V2 integration
- Custom validator functions
- Constraint enforcement (metadata-driven)
- Error handling and validation pipelines
- Async validation patterns

**Who Should Read This:**

- Developers building validated data models
- API designers requiring input validation
- Anyone working with structured LLM outputs or schemas

---

## Validation with Spec

### Basic Field Validation

`Spec` supports **validator functions** via the `validator` metadata key:

**Single Validator:**

```python
from lionherd_core.types import Spec

def validate_positive(value: int) -> int:
    """Validator: ensure value is positive."""
    if value <= 0:
        raise ValueError("Value must be positive")
    return value

spec = Spec(int, name="age", validator=validate_positive)

# Access validator
print(spec.get("validator"))  # <function validate_positive>

# Validation would be applied by framework adapter (Pydantic, etc.)
```

**Multiple Validators (Chain):**

```python
from lionherd_core.types import Spec

def min_length(value: str) -> str:
    """Validator: minimum length 3."""
    if len(value) < 3:
        raise ValueError("Minimum length: 3")
    return value

def max_length(value: str) -> str:
    """Validator: maximum length 20."""
    if len(value) > 20:
        raise ValueError("Maximum length: 20")
    return value

spec = Spec(
    str,
    name="username",
    validator=[min_length, max_length]
)

# Multiple validators are chained
validators = spec.get("validator")
print(validators)  # [<function min_length>, <function max_length>]
```

### Validator Metadata Constraints

`Spec` enforces validator metadata constraints at construction:

#### Constraint 1: Validators must be callable

```python
# ❌ WRONG: Non-callable validator
try:
    spec = Spec(str, validator="not callable")
except ExceptionGroup as e:
    print(e)
    # ExceptionGroup: Metadata validation failed (1 sub-exception)
    #   ValueError: Validators must be a list of functions or a function
```

#### Constraint 2: Validator list must contain callables

```python
# ❌ WRONG: List with non-callable
try:
    spec = Spec(str, validator=[lambda x: x, "not callable"])
except ExceptionGroup as e:
    print(e)
    # ExceptionGroup: Metadata validation failed (1 sub-exception)
    #   ValueError: Validators must be a list of functions or a function
```

### Fluent Validator API

Use `.with_validator()` for fluent validator addition:

```python
from lionherd_core.types import Spec

spec = (
    Spec(str, name="email")
    .with_validator(lambda v: v.lower())  # Normalize to lowercase
    .with_validator(lambda v: v if "@" in v else ValueError("Invalid email"))
)

print(spec.get("validator"))
# [<function <lambda>>, <function <lambda>>]
```

---

## Pydantic V2 Integration

### Basic Pydantic Model with Spec

Convert Spec to Pydantic `Field`:

```python
from pydantic import BaseModel, Field
from lionherd_core.types import Spec, not_sentinel

def spec_to_field(spec: Spec):
    """Convert Spec to Pydantic Field."""
    kwargs = {}

    # Handle default/default_factory
    if not_sentinel(spec.default):
        if spec.has_default_factory:
            kwargs["default_factory"] = spec.default
        else:
            kwargs["default"] = spec.default

    # Handle description
    if description := spec.get("description"):
        kwargs["description"] = description

    return Field(**kwargs)

# Define specs
username_spec = Spec(str, name="username", default="guest")
email_spec = Spec(str, name="email", description="User email address")

# Create Pydantic model
class User(BaseModel):
    username: str = spec_to_field(username_spec)
    email: str = spec_to_field(email_spec)

# Usage
user = User(email="alice@example.com")
print(user.username)  # "guest"
print(user.email)     # "alice@example.com"
```

### Pydantic field_validator with Spec

Integrate Spec validators with Pydantic `field_validator`:

```python
from pydantic import BaseModel, field_validator
from lionherd_core.types import Spec

# Define specs with validators
def validate_email(v: str) -> str:
    if "@" not in v:
        raise ValueError("Invalid email format")
    return v.lower()

def validate_age(v: int) -> int:
    if v < 0 or v > 150:
        raise ValueError("Age must be between 0 and 150")
    return v

email_spec = Spec(str, name="email", validator=validate_email)
age_spec = Spec(int, name="age", validator=validate_age)

class User(BaseModel):
    email: str
    age: int

    @field_validator("email")
    @classmethod
    def validate_email_field(cls, v):
        # Apply Spec validator
        validator = email_spec.get("validator")
        return validator(v) if validator else v

    @field_validator("age")
    @classmethod
    def validate_age_field(cls, v):
        # Apply Spec validator
        validator = age_spec.get("validator")
        return validator(v) if validator else v

# Usage
user = User(email="ALICE@EXAMPLE.COM", age=30)
print(user.email)  # "alice@example.com" (normalized by validator)

try:
    User(email="invalid", age=30)
except Exception as e:
    print(f"Validation error: {e}")
    # Validation error: Invalid email format
```

### Dynamic Pydantic Models from Specs

Generate Pydantic models dynamically from Spec definitions:

```python
from pydantic import BaseModel, Field, create_model, field_validator
from lionherd_core.types import Spec, not_sentinel

def create_model_from_specs(
    model_name: str,
    specs: list[Spec]
) -> type[BaseModel]:
    """Create Pydantic model dynamically from Spec list."""
    fields = {}

    for spec in specs:
        field_name = spec.name
        if field_name is None:
            raise ValueError(f"Spec must have 'name' metadata: {spec}")

        # Build Field kwargs
        field_kwargs = {}
        if not_sentinel(spec.default):
            if spec.has_default_factory:
                field_kwargs["default_factory"] = spec.default
            else:
                field_kwargs["default"] = spec.default

        if description := spec.get("description"):
            field_kwargs["description"] = description

        # Add field to model
        fields[field_name] = (spec.annotation, Field(**field_kwargs))

    # Create model
    return create_model(model_name, **fields)

# Define specs
specs = [
    Spec(str, name="username", default="guest"),
    Spec(str, name="email"),
    Spec(int, name="age", nullable=True, default=None),
]

# Generate model
User = create_model_from_specs("User", specs)

# Usage
user = User(email="alice@example.com")
print(user.model_dump())
# {"username": "guest", "email": "alice@example.com", "age": None}
```

### Sentinel Validation with Pydantic

Combine sentinels with Pydantic for partial update validation:

```python
from pydantic import BaseModel, field_validator
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class UserUpdate(BaseModel):
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str | None] = Unset
    age: MaybeUnset[int] = Unset

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not_sentinel(v):  # Only validate if provided
            if not v or not v.strip():
                raise ValueError("Name cannot be empty")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if not_sentinel(v):  # Only validate if provided
            if v is not None and "@" not in v:
                raise ValueError("Invalid email format")
        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if not_sentinel(v):  # Only validate if provided
            if v < 0 or v > 150:
                raise ValueError("Age must be between 0 and 150")
        return v

# Usage
update = UserUpdate(name="Alice", age=30)  # Only validates provided fields
print(update.model_dump())
# {"name": "Alice", "email": <Unset>, "age": 30}

try:
    UserUpdate(name="", email="invalid")  # Validation errors
except Exception as e:
    print(f"Validation failed: {e}")
```

---

## Custom Validators

### Validator Function Patterns

#### Pattern 1: Value Transformer

Validators that normalize/transform values:

```python
def normalize_email(value: str) -> str:
    """Normalize email to lowercase."""
    return value.lower().strip()

def slugify(value: str) -> str:
    """Convert string to URL-safe slug."""
    import re
    value = value.lower().strip()
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[-\s]+', '-', value)
    return value

# Usage in Spec
email_spec = Spec(str, name="email", validator=normalize_email)
slug_spec = Spec(str, name="slug", validator=slugify)
```

#### Pattern 2: Constraint Validator

Validators that enforce constraints:

```python
def min_value(min_val: float):
    """Factory: validator for minimum value constraint."""
    def validator(value: float) -> float:
        if value < min_val:
            raise ValueError(f"Value must be >= {min_val}")
        return value
    return validator

def max_value(max_val: float):
    """Factory: validator for maximum value constraint."""
    def validator(value: float) -> float:
        if value > max_val:
            raise ValueError(f"Value must be <= {max_val}")
        return value
    return validator

def in_range(min_val: float, max_val: float):
    """Factory: validator for value range."""
    def validator(value: float) -> float:
        if not (min_val <= value <= max_val):
            raise ValueError(f"Value must be in range [{min_val}, {max_val}]")
        return value
    return validator

# Usage
age_spec = Spec(int, name="age", validator=[min_value(0), max_value(150)])
score_spec = Spec(float, name="score", validator=in_range(0.0, 100.0))
```

#### Pattern 3: Conditional Validator

Validators with conditional logic:

```python
from lionherd_core.types import Spec

def validate_if_present(validator_fn):
    """Wrapper: only validate if value is not None."""
    def wrapper(value):
        if value is None:
            return value
        return validator_fn(value)
    return wrapper

def validate_email(value: str) -> str:
    if "@" not in value:
        raise ValueError("Invalid email")
    return value

# Email can be None, but if provided must be valid
email_spec = Spec(
    str,
    name="email",
    nullable=True,
    validator=validate_if_present(validate_email)
)
```

#### Pattern 4: Multi-Field Validator

Validators that depend on other fields (use Pydantic `model_validator`):

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode="after")
    def validate_date_range(self):
        """Validate end_date is after start_date."""
        from datetime import datetime

        start = datetime.fromisoformat(self.start_date)
        end = datetime.fromisoformat(self.end_date)

        if end < start:
            raise ValueError("end_date must be after start_date")

        return self

# Usage
range1 = DateRange(start_date="2025-01-01", end_date="2025-01-31")  # OK

try:
    range2 = DateRange(start_date="2025-01-31", end_date="2025-01-01")  # Error
except Exception as e:
    print(f"Validation error: {e}")
```

### Reusable Validator Library

Build a library of reusable validators:

```python
"""Common validators for lionherd-core projects."""

from typing import Any, Callable

def not_empty(value: str) -> str:
    """Validate string is not empty."""
    if not value.strip():
        raise ValueError("Value cannot be empty")
    return value

def email(value: str) -> str:
    """Validate email format."""
    if "@" not in value or "." not in value.split("@")[1]:
        raise ValueError("Invalid email format")
    return value.lower()

def url(value: str) -> str:
    """Validate URL format."""
    if not value.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return value

def one_of(*allowed_values: Any) -> Callable[[Any], Any]:
    """Factory: validate value is one of allowed values."""
    def validator(value: Any) -> Any:
        if value not in allowed_values:
            raise ValueError(f"Value must be one of {allowed_values}")
        return value
    return validator

def regex_match(pattern: str) -> Callable[[str], str]:
    """Factory: validate string matches regex pattern."""
    import re
    compiled = re.compile(pattern)

    def validator(value: str) -> str:
        if not compiled.match(value):
            raise ValueError(f"Value must match pattern: {pattern}")
        return value
    return validator

# Usage
from lionherd_core.types import Spec

email_spec = Spec(str, name="email", validator=email)
status_spec = Spec(str, name="status", validator=one_of("active", "inactive", "pending"))
phone_spec = Spec(str, name="phone", validator=regex_match(r"^\d{3}-\d{3}-\d{4}$"))
```

---

## Constraint Enforcement

### Metadata-Driven Constraints

Use Spec metadata for declarative constraints:

```python
from lionherd_core.types import Spec

# Constraint metadata
age_spec = Spec(
    int,
    name="age",
    min_value=0,
    max_value=150,
    description="User age"
)

email_spec = Spec(
    str,
    name="email",
    pattern=r"^[^@]+@[^@]+\.[^@]+$",
    description="Email address"
)

# Extract constraints for validation
def build_validator_from_spec(spec: Spec) -> Callable[[Any], Any]:
    """Build validator function from Spec constraints."""
    validators = []

    # Extract constraint metadata
    if min_val := spec.get("min_value"):
        validators.append(lambda v: v if v >= min_val else ValueError(f"Min: {min_val}"))

    if max_val := spec.get("max_value"):
        validators.append(lambda v: v if v <= max_val else ValueError(f"Max: {max_val}"))

    if pattern := spec.get("pattern"):
        import re
        regex = re.compile(pattern)
        validators.append(lambda v: v if regex.match(v) else ValueError(f"Pattern: {pattern}"))

    # Chain validators
    def validator(value):
        for v in validators:
            result = v(value)
            if isinstance(result, ValueError):
                raise result
        return value

    return validator

# Use metadata-driven validator
age_validator = build_validator_from_spec(age_spec)
print(age_validator(30))  # 30 (OK)

try:
    age_validator(200)  # ValueError
except ValueError as e:
    print(e)
```

### Pydantic Field Constraints

Map Spec constraints to Pydantic Field constraints:

```python
from pydantic import BaseModel, Field
from lionherd_core.types import Spec, not_sentinel

def spec_to_pydantic_field(spec: Spec):
    """Convert Spec to Pydantic Field with constraints."""
    kwargs = {}

    # Default handling
    if not_sentinel(spec.default):
        if spec.has_default_factory:
            kwargs["default_factory"] = spec.default
        else:
            kwargs["default"] = spec.default

    # Constraints (Pydantic V2 syntax)
    if min_val := spec.get("min_value"):
        kwargs["ge"] = min_val
    if max_val := spec.get("max_value"):
        kwargs["le"] = max_val
    if min_len := spec.get("min_length"):
        kwargs["min_length"] = min_len
    if max_len := spec.get("max_length"):
        kwargs["max_length"] = max_len
    if pattern := spec.get("pattern"):
        kwargs["pattern"] = pattern

    # Description
    if description := spec.get("description"):
        kwargs["description"] = description

    return Field(**kwargs)

# Define specs with constraints
username_spec = Spec(
    str,
    name="username",
    min_length=3,
    max_length=20,
    pattern=r"^[a-zA-Z0-9_]+$",
)

age_spec = Spec(
    int,
    name="age",
    min_value=0,
    max_value=150,
)

# Create Pydantic model
class User(BaseModel):
    username: str = spec_to_pydantic_field(username_spec)
    age: int = spec_to_pydantic_field(age_spec)

# Usage
user = User(username="alice_123", age=30)  # OK

try:
    User(username="a", age=30)  # Min length violation
except Exception as e:
    print(f"Validation error: {e}")

try:
    User(username="alice_123", age=200)  # Max value violation
except Exception as e:
    print(f"Validation error: {e}")
```

---

## Error Handling Patterns

### Validation Error Collection

Collect all validation errors before raising:

```python
from typing import Any

def validate_all(data: dict, specs: list[Spec]) -> dict[str, list[str]]:
    """Validate data against specs, collecting all errors.

    Returns:
        Dict mapping field names to error messages
    """
    errors = {}

    for spec in specs:
        field_name = spec.name
        if field_name is None:
            continue

        value = data.get(field_name)
        field_errors = []

        # Get validators
        validators = spec.get("validator", [])
        if callable(validators):
            validators = [validators]

        # Run each validator
        for validator in validators:
            try:
                value = validator(value)
            except Exception as e:
                field_errors.append(str(e))

        if field_errors:
            errors[field_name] = field_errors

    return errors

# Usage
from lionherd_core.types import Spec

def min_length(v):
    if len(v) < 3:
        raise ValueError("Min length: 3")
    return v

def max_length(v):
    if len(v) > 20:
        raise ValueError("Max length: 20")
    return v

username_spec = Spec(str, name="username", validator=[min_length, max_length])
email_spec = Spec(str, name="email", validator=lambda v: v if "@" in v else ValueError("Invalid email"))

specs = [username_spec, email_spec]

data = {"username": "ab", "email": "invalid"}
errors = validate_all(data, specs)
print(errors)
# {
#     "username": ["Min length: 3"],
#     "email": ["Invalid email"]
# }
```

### Graceful Validation Failures

Handle validation failures gracefully with defaults:

```python
from lionherd_core.types import Spec

def validate_with_fallback(
    value: Any,
    spec: Spec,
    fallback: Any = None
) -> tuple[Any, list[str]]:
    """Validate value, returning fallback and errors on failure.

    Returns:
        Tuple of (validated_value, errors_list)
    """
    validators = spec.get("validator", [])
    if callable(validators):
        validators = [validators]

    errors = []
    for validator in validators:
        try:
            value = validator(value)
        except Exception as e:
            errors.append(str(e))
            return fallback, errors

    return value, errors

# Usage
age_spec = Spec(
    int,
    name="age",
    validator=lambda v: v if 0 <= v <= 150 else ValueError("Age out of range")
)

value, errors = validate_with_fallback(30, age_spec)
print(value, errors)  # 30, []

value, errors = validate_with_fallback(200, age_spec, fallback=None)
print(value, errors)  # None, ["Age out of range"]
```

---

## Async Validation

### Async Validators

Validators can be async for I/O-bound validation:

```python
from lionherd_core.libs.concurrency import is_coro_func, sleep
from lionherd_core.types import Spec

async def validate_unique_email(email: str) -> str:
    """Async validator: check email uniqueness in database."""
    # Simulate database query
    await sleep(0.1)
    existing_emails = ["alice@example.com", "bob@example.com"]

    if email in existing_emails:
        raise ValueError("Email already exists")
    return email

# Store async validator in Spec
email_spec = Spec(str, name="email", validator=validate_unique_email)

# Apply async validation
async def validate_user_email(email: str) -> str:
    validator = email_spec.get("validator")
    if is_coro_func(validator):
        return await validator(email)
    return validator(email)

# Usage
async def main():
    try:
        email = await validate_user_email("alice@example.com")
    except ValueError as e:
        print(f"Validation error: {e}")
        # Validation error: Email already exists

    email = await validate_user_email("charlie@example.com")
    print(f"Valid email: {email}")
    # Valid email: charlie@example.com

# Run
import anyio
anyio.run(main)
```

### Async Validation Pipelines

Chain async validators for complex validation:

```python
from lionherd_core.libs.concurrency import sleep

async def validate_email_format(email: str) -> str:
    """Validate email format."""
    if "@" not in email:
        raise ValueError("Invalid email format")
    return email

async def validate_email_domain(email: str) -> str:
    """Validate email domain exists (simulated DNS lookup)."""
    domain = email.split("@")[1]
    await sleep(0.1)  # Simulate DNS lookup

    valid_domains = ["example.com", "test.com"]
    if domain not in valid_domains:
        raise ValueError(f"Invalid domain: {domain}")
    return email

async def validate_email_unique(email: str) -> str:
    """Validate email is unique (simulated DB query)."""
    await sleep(0.1)  # Simulate DB query

    existing = ["alice@example.com"]
    if email in existing:
        raise ValueError("Email already exists")
    return email

async def validate_email_pipeline(email: str) -> str:
    """Run async validation pipeline."""
    validators = [
        validate_email_format,
        validate_email_domain,
        validate_email_unique,
    ]

    for validator in validators:
        email = await validator(email)

    return email

# Usage
async def main():
    try:
        email = await validate_email_pipeline("alice@example.com")
    except ValueError as e:
        print(f"Validation failed: {e}")
        # Validation failed: Email already exists

    try:
        email = await validate_email_pipeline("bob@invalid.com")
    except ValueError as e:
        print(f"Validation failed: {e}")
        # Validation failed: Invalid domain: invalid.com

    email = await validate_email_pipeline("bob@example.com")
    print(f"Valid email: {email}")
    # Valid email: bob@example.com

import anyio
anyio.run(main)
```

---

## Best Practices

### 1. Prefer Pydantic for Complex Validation

Use Pydantic's built-in validators for complex logic:

```python
from pydantic import BaseModel, field_validator, model_validator

class User(BaseModel):
    username: str
    email: str
    age: int

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError("Username too short")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()

    @model_validator(mode="after")
    def validate_model(self):
        # Cross-field validation
        if self.age < 13 and "@" not in self.email:
            raise ValueError("Users under 13 must have valid email")
        return self
```

### 2. Extract Validators for Reusability

Create validator libraries:

```python
# noqa:validation
# validators.py - Example reusable validator module
"""Reusable validator functions."""

def not_empty(v: str) -> str:
    if not v.strip():
        raise ValueError("Cannot be empty")
    return v

def email(v: str) -> str:
    if "@" not in v:
        raise ValueError("Invalid email")
    return v.lower()

# Use across projects
from lionherd_core.types import Spec
import validators

email_spec = Spec(str, name="email", validator=validators.email)
```

### 3. Use Type Hints in Validators

Type hint validators for better tooling support:

```python
from typing import TypeVar

T = TypeVar("T")

def min_value(min_val: T) -> Callable[[T], T]:
    """Validator factory with type hints."""
    def validator(value: T) -> T:
        if value < min_val:
            raise ValueError(f"Min value: {min_val}")
        return value
    return validator

# Type checker understands validator signature
int_validator = min_value(0)  # Callable[[int], int]
float_validator = min_value(0.0)  # Callable[[float], float]
```

### 4. Document Validation Rules

Document validation rules in docstrings:

```python
from lionherd_core.types import Spec

username_spec = Spec(
    str,
    name="username",
    description="Username (3-20 chars, alphanumeric + underscore)",
    min_length=3,
    max_length=20,
    pattern=r"^[a-zA-Z0-9_]+$",
)

# Or in validator docstrings
def validate_username(v: str) -> str:
    """Validate username.

    Rules:
        - Length: 3-20 characters
        - Characters: alphanumeric and underscore only
        - Case-insensitive

    Raises:
        ValueError: If validation fails
    """
    if len(v) < 3 or len(v) > 20:
        raise ValueError("Username must be 3-20 characters")
    if not v.replace("_", "").isalnum():
        raise ValueError("Username must be alphanumeric + underscore")
    return v
```

### 5. Fail Fast for Required Fields

Validate required fields first:

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    email: str
    age: int | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        # Required field - validate immediately
        if not v or not v.strip():
            raise ValueError("Name is required")
        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        # Optional field - skip validation if None
        if v is not None and (v < 0 or v > 150):
            raise ValueError("Invalid age")
        return v
```

---

## See Also

- **Type Safety Guide**: TypeGuards and type narrowing
- **API Design Guide**: Optional parameters and sentinels
- **Protocols Guide**: Protocol-based validation
- **Spec API Reference**: Complete Spec documentation

---

## References

- [Pydantic V2 Validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [Pydantic Field Constraints](https://docs.pydantic.dev/latest/concepts/fields/)

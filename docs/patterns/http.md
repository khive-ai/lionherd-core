# HTTP Patterns

> PATCH semantics, partial updates, and three-state field handling with lionherd-core

## Overview

lionherd-core's sentinel values enable **precise PATCH semantics** for HTTP APIs. This guide covers partial update patterns, three-state field handling (value/null/unset), and FastAPI/Pydantic integration for type-safe REST APIs.

**Key Topics:**

- PATCH vs PUT semantics
- Three-state field handling with Unset
- Partial update patterns
- FastAPI integration
- Request/response models with sentinels
- Error handling for partial updates

**Who Should Read This:**

- API developers building REST/HTTP services
- FastAPI users working with partial updates
- Anyone implementing PATCH endpoints

---

## PATCH vs PUT Semantics

### PUT: Full Replacement

**PUT** replaces the **entire resource** with the request body:

```python
# PUT /users/123
# Request body: {"name": "Alice", "email": "alice@example.com"}
# → Replaces entire user resource

from pydantic import BaseModel

class UserPut(BaseModel):
    name: str
    email: str

def update_user_put(user_id: str, update: UserPut) -> User:
    """PUT: replace entire user."""
    user = get_user(user_id)
    user.name = update.name
    user.email = update.email
    # All fields must be provided
    return user
```

**Problem**: Client must send **all fields**, even unchanged ones:

```python
# Want to update only email, but must send name too:
PUT /users/123
{
    "name": "Alice",        # Unchanged (but required)
    "email": "alice@newdomain.com"  # Changed
}
```

### PATCH: Partial Update

**PATCH** updates **only specified fields**, leaving others unchanged:

```python
# PATCH /users/123
# Request body: {"email": "alice@newdomain.com"}
# → Updates only email, name unchanged

from lionherd_core.types import Unset, MaybeUnset
from pydantic import BaseModel

class UserPatch(BaseModel):
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str] = Unset

def update_user_patch(user_id: str, update: UserPatch) -> User:
    """PATCH: update only provided fields."""
    user = get_user(user_id)

    if update.name is not Unset:
        user.name = update.name
    if update.email is not Unset:
        user.email = update.email

    return user
```

**Benefit**: Client sends **only changed fields**:

```python
# Update only email:
PATCH /users/123
{
    "email": "alice@newdomain.com"
}
# name is Unset → not updated
```

---

## Three-State Field Handling

### The Three States

PATCH endpoints must distinguish three field states:

1. **Not provided** (Unset): Don't change the field
2. **Null** (None): Clear/delete the field
3. **Value** (str/int/etc.): Update to new value

**Example:**

```python
from lionherd_core.types import Unset, MaybeUnset

class UserPatch(BaseModel):
    bio: MaybeUnset[str | None] = Unset

# Three possible states:
# 1. {"bio": <not in request>} → Unset → Don't change
# 2. {"bio": null}            → None → Clear bio
# 3. {"bio": "New bio text"}  → str → Update bio
```

### Implementation Pattern

```python
from lionherd_core.types import Unset, MaybeUnset, not_sentinel
from pydantic import BaseModel

class User(BaseModel):
    id: str
    name: str
    email: str
    bio: str | None = None
    phone: str | None = None

class UserPatch(BaseModel):
    """Partial update model with three-state fields."""
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str | None] = Unset
    bio: MaybeUnset[str | None] = Unset
    phone: MaybeUnset[str | None] = Unset

def apply_patch(user: User, patch: UserPatch) -> User:
    """Apply partial update to user."""
    updates = {}

    # Only update provided fields (not Unset)
    if not_sentinel(patch.name):
        updates["name"] = patch.name

    if not_sentinel(patch.email):
        updates["email"] = patch.email

    if not_sentinel(patch.bio):
        updates["bio"] = patch.bio  # Can be None (clear) or str (update)

    if not_sentinel(patch.phone):
        updates["phone"] = patch.phone  # Can be None (clear) or str (update)

    # Pydantic model update
    return user.model_copy(update=updates)

# Usage examples:

# Example 1: Update only name
user = User(id="123", name="Alice", email="alice@example.com", bio="Engineer")
patch = UserPatch(name="Alice Smith")
updated = apply_patch(user, patch)
print(updated)
# User(id="123", name="Alice Smith", email="alice@example.com", bio="Engineer")

# Example 2: Clear bio (set to None)
patch = UserPatch(bio=None)
updated = apply_patch(user, patch)
print(updated)
# User(id="123", name="Alice", email="alice@example.com", bio=None)

# Example 3: Update email, clear phone
patch = UserPatch(email="alice@newdomain.com", phone=None)
updated = apply_patch(user, patch)
print(updated)
# User(id="123", name="Alice", email="alice@newdomain.com", bio="Engineer", phone=None)
```

---

## FastAPI Integration

### Basic PATCH Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

app = FastAPI()

# Models
class User(BaseModel):
    id: str
    name: str
    email: str
    bio: str | None = None

class UserPatch(BaseModel):
    """Partial update model."""
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str | None] = Unset
    bio: MaybeUnset[str | None] = Unset

# In-memory storage (replace with database)
users_db: dict[str, User] = {}

@app.patch("/users/{user_id}", response_model=User)
async def update_user(user_id: str, patch: UserPatch):
    """Update user with partial update (PATCH semantics)."""
    # Get existing user
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Apply updates
    updates = {}
    if not_sentinel(patch.name):
        updates["name"] = patch.name
    if not_sentinel(patch.email):
        updates["email"] = patch.email
    if not_sentinel(patch.bio):
        updates["bio"] = patch.bio

    # Update user
    updated_user = user.model_copy(update=updates)
    users_db[user_id] = updated_user

    return updated_user

# Example requests:

# 1. Update only name:
# PATCH /users/123
# {"name": "Alice Smith"}

# 2. Clear bio:
# PATCH /users/123
# {"bio": null}

# 3. Update email and bio:
# PATCH /users/123
# {"email": "alice@newdomain.com", "bio": "Senior Engineer"}
```

### Validation with PATCH Models

Add field validators for partial updates:

```python
from pydantic import BaseModel, field_validator
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class UserPatch(BaseModel):
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

# FastAPI endpoint with validation
@app.patch("/users/{user_id}")
async def update_user(user_id: str, patch: UserPatch):
    """Update user with validated partial updates."""
    # Pydantic automatically validates provided fields
    # Unset fields are skipped
    ...
```

### Response Models

Return partial update results:

```python
from pydantic import BaseModel

class UserPatchResponse(BaseModel):
    """Response for PATCH endpoint."""
    user: User
    updated_fields: list[str]

@app.patch("/users/{user_id}", response_model=UserPatchResponse)
async def update_user(user_id: str, patch: UserPatch):
    """Update user and return updated fields."""
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Track updated fields
    updated_fields = []
    updates = {}

    if not_sentinel(patch.name):
        updates["name"] = patch.name
        updated_fields.append("name")

    if not_sentinel(patch.email):
        updates["email"] = patch.email
        updated_fields.append("email")

    if not_sentinel(patch.bio):
        updates["bio"] = patch.bio
        updated_fields.append("bio")

    # Apply updates
    updated_user = user.model_copy(update=updates)
    users_db[user_id] = updated_user

    return UserPatchResponse(user=updated_user, updated_fields=updated_fields)

# Example response:
# {
#     "user": {
#         "id": "123",
#         "name": "Alice Smith",
#         "email": "alice@example.com",
#         "bio": "Engineer"
#     },
#     "updated_fields": ["name"]
# }
```

---

## Advanced Patterns

### Nested Resource Updates

Handle nested resource updates with sentinels:

```python
from pydantic import BaseModel
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class AddressPatch(BaseModel):
    """Partial address update."""
    street: MaybeUnset[str] = Unset
    city: MaybeUnset[str] = Unset
    zip_code: MaybeUnset[str] = Unset

class UserPatch(BaseModel):
    """User with nested address update."""
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str] = Unset
    address: MaybeUnset[AddressPatch] = Unset

def apply_nested_patch(user: User, patch: UserPatch) -> User:
    """Apply patch with nested updates."""
    updates = {}

    if not_sentinel(patch.name):
        updates["name"] = patch.name

    if not_sentinel(patch.email):
        updates["email"] = patch.email

    # Nested address update
    if not_sentinel(patch.address):
        address_updates = {}
        if not_sentinel(patch.address.street):
            address_updates["street"] = patch.address.street
        if not_sentinel(patch.address.city):
            address_updates["city"] = patch.address.city
        if not_sentinel(patch.address.zip_code):
            address_updates["zip_code"] = patch.address.zip_code

        updated_address = user.address.model_copy(update=address_updates)
        updates["address"] = updated_address

    return user.model_copy(update=updates)

# Example: Update only address city
patch = UserPatch(address=AddressPatch(city="New York"))
updated = apply_nested_patch(user, patch)
# Only address.city updated, other fields unchanged
```

### Array Operations

Handle array updates with special semantics:

```python
from enum import Enum
from pydantic import BaseModel
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class ArrayOperation(str, Enum):
    """Array update operations."""
    REPLACE = "replace"  # Replace entire array
    APPEND = "append"    # Append items
    REMOVE = "remove"    # Remove items

class TagsUpdate(BaseModel):
    """Array update with operation."""
    operation: ArrayOperation
    values: list[str]

class UserPatch(BaseModel):
    name: MaybeUnset[str] = Unset
    tags: MaybeUnset[TagsUpdate] = Unset

def apply_array_patch(user: User, patch: UserPatch) -> User:
    """Apply patch with array operations."""
    updates = {}

    if not_sentinel(patch.name):
        updates["name"] = patch.name

    if not_sentinel(patch.tags):
        current_tags = set(user.tags)

        if patch.tags.operation == ArrayOperation.REPLACE:
            updates["tags"] = patch.tags.values
        elif patch.tags.operation == ArrayOperation.APPEND:
            updates["tags"] = list(current_tags | set(patch.tags.values))
        elif patch.tags.operation == ArrayOperation.REMOVE:
            updates["tags"] = list(current_tags - set(patch.tags.values))

    return user.model_copy(update=updates)

# Examples:

# Replace tags:
patch = UserPatch(tags=TagsUpdate(operation=ArrayOperation.REPLACE, values=["python", "rust"]))

# Append tags:
patch = UserPatch(tags=TagsUpdate(operation=ArrayOperation.APPEND, values=["go"]))

# Remove tags:
patch = UserPatch(tags=TagsUpdate(operation=ArrayOperation.REMOVE, values=["python"]))
```

### Conditional Updates

Apply updates based on conditions:

```python
from pydantic import BaseModel, model_validator
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class UserPatch(BaseModel):
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str | None] = Unset
    role: MaybeUnset[str] = Unset

    @model_validator(mode="after")
    def validate_conditional_updates(self):
        """Validate cross-field constraints."""
        # If role is being updated to admin, email is required
        if not_sentinel(self.role) and self.role == "admin":
            if self.email is Unset or self.email is None:
                raise ValueError("Admin users must have an email")

        return self

# Usage
try:
    patch = UserPatch(role="admin", email=None)  # Validation error
except ValueError as e:
    print(e)  # "Admin users must have an email"

patch = UserPatch(role="admin", email="admin@example.com")  # OK
```

### Optimistic Locking

Implement optimistic locking for safe partial updates:

```python
from pydantic import BaseModel
from lionherd_core.types import Unset, MaybeUnset, not_sentinel

class User(BaseModel):
    id: str
    name: str
    email: str
    version: int  # Optimistic lock version

class UserPatch(BaseModel):
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str] = Unset
    expected_version: int  # Required for optimistic locking

@app.patch("/users/{user_id}")
async def update_user(user_id: str, patch: UserPatch):
    """Update user with optimistic locking."""
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check version
    if user.version != patch.expected_version:
        raise HTTPException(
            status_code=409,
            detail=f"Version conflict: expected {patch.expected_version}, got {user.version}"
        )

    # Apply updates
    updates = {}
    if not_sentinel(patch.name):
        updates["name"] = patch.name
    if not_sentinel(patch.email):
        updates["email"] = patch.email

    # Increment version
    updates["version"] = user.version + 1

    updated_user = user.model_copy(update=updates)
    users_db[user_id] = updated_user

    return updated_user

# Example request:
# PATCH /users/123
# {
#     "name": "Alice Smith",
#     "expected_version": 5
# }
```

---

## Error Handling

### Validation Errors

Handle validation errors for partial updates:

```python
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from lionherd_core.types import Unset, MaybeUnset

@app.patch("/users/{user_id}")
async def update_user(user_id: str, patch: UserPatch):
    """Update user with error handling."""
    try:
        # Pydantic validation happens automatically
        user = users_db.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Apply updates...
        return updated_user

    except ValidationError as e:
        # Return detailed validation errors
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Validation error",
                "errors": e.errors(),
            }
        )

# Example error response:
# {
#     "message": "Validation error",
#     "errors": [
#         {
#             "loc": ["email"],
#             "msg": "Invalid email format",
#             "type": "value_error"
#         }
#     ]
# }
```

### Conflict Handling

Handle update conflicts:

```python
@app.patch("/users/{user_id}")
async def update_user(user_id: str, patch: UserPatch):
    """Update user with conflict handling."""
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check for conflicts (e.g., unique email)
    if not_sentinel(patch.email) and patch.email is not None:
        existing = find_user_by_email(patch.email)
        if existing and existing.id != user_id:
            raise HTTPException(
                status_code=409,
                detail=f"Email {patch.email} already in use"
            )

    # Apply updates...
    return updated_user
```

### Partial Success Reporting

Report which fields were successfully updated:

```python
from pydantic import BaseModel

class FieldError(BaseModel):
    field: str
    error: str

class PatchResult(BaseModel):
    user: User
    updated_fields: list[str]
    failed_fields: list[FieldError]

@app.patch("/users/{user_id}", response_model=PatchResult)
async def update_user(user_id: str, patch: UserPatch):
    """Update user with partial success reporting."""
    user = users_db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    updated_fields = []
    failed_fields = []
    updates = {}

    # Try to update each field
    if not_sentinel(patch.name):
        try:
            validate_name(patch.name)
            updates["name"] = patch.name
            updated_fields.append("name")
        except ValueError as e:
            failed_fields.append(FieldError(field="name", error=str(e)))

    if not_sentinel(patch.email):
        try:
            validate_email(patch.email)
            updates["email"] = patch.email
            updated_fields.append("email")
        except ValueError as e:
            failed_fields.append(FieldError(field="email", error=str(e)))

    # Apply successful updates
    updated_user = user.model_copy(update=updates)
    users_db[user_id] = updated_user

    return PatchResult(
        user=updated_user,
        updated_fields=updated_fields,
        failed_fields=failed_fields,
    )
```

---

## Testing PATCH Endpoints

### Test Three-State Handling

```python
import pytest
from fastapi.testclient import TestClient

client = TestClient(app)

def test_patch_update_single_field():
    """Test updating single field leaves others unchanged."""
    # Create user
    user = create_test_user(name="Alice", email="alice@example.com", bio="Engineer")

    # Update only name
    response = client.patch(
        f"/users/{user.id}",
        json={"name": "Alice Smith"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice Smith"
    assert data["email"] == "alice@example.com"  # Unchanged
    assert data["bio"] == "Engineer"  # Unchanged

def test_patch_clear_field():
    """Test clearing field with null."""
    user = create_test_user(name="Alice", bio="Engineer")

    # Clear bio
    response = client.patch(
        f"/users/{user.id}",
        json={"bio": None}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["bio"] is None  # Cleared

def test_patch_omitted_fields_unchanged():
    """Test omitted fields remain unchanged."""
    user = create_test_user(name="Alice", email="alice@example.com")

    # Update email only (name omitted)
    response = client.patch(
        f"/users/{user.id}",
        json={"email": "alice@newdomain.com"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice"  # Unchanged (omitted)
    assert data["email"] == "alice@newdomain.com"  # Updated

def test_patch_validation_only_provided_fields():
    """Test validation only applies to provided fields."""
    user = create_test_user(name="Alice", email="alice@example.com")

    # Invalid email, but name not validated
    response = client.patch(
        f"/users/{user.id}",
        json={"email": "invalid"}
    )

    assert response.status_code == 422  # Validation error
    assert "email" in response.json()["errors"][0]["loc"]
```

---

## Best Practices

### 1. Use `MaybeUnset` for All PATCH Fields

Every field in a PATCH model should use `MaybeUnset`:

```python
# ✅ CORRECT
class UserPatch(BaseModel):
    name: MaybeUnset[str] = Unset
    email: MaybeUnset[str | None] = Unset

# ❌ WRONG
class UserPatch(BaseModel):
    name: str | None = None  # Can't distinguish omitted vs null
    email: str | None = None
```

### 2. Document Three-State Semantics

Document what null means for each field:

```python
class UserPatch(BaseModel):
    """Partial user update.

    Three-state semantics for each field:
    - Omitted: Field not updated (remains unchanged)
    - null: Clear field (set to None)
    - value: Update to new value
    """
    bio: MaybeUnset[str | None] = Unset
    phone: MaybeUnset[str | None] = Unset
```

### 3. Validate Only Provided Fields

Use `not_sentinel()` in validators:

```python
@field_validator("email")
@classmethod
def validate_email(cls, v):
    if not_sentinel(v):  # Only validate if provided
        if v is not None and "@" not in v:
            raise ValueError("Invalid email")
    return v
```

### 4. Return Updated Fields in Response

Help clients understand what changed:

```python
@app.patch("/users/{user_id}")
async def update_user(user_id: str, patch: UserPatch):
    """Update user and return changed fields."""
    # ...apply updates...
    return {
        "user": updated_user,
        "updated_fields": ["name", "email"],  # What changed
    }
```

### 5. Test Edge Cases

Test all three states for each field:

```python
# Test omitted (Unset)
client.patch("/users/123", json={})

# Test null (None)
client.patch("/users/123", json={"bio": None})

# Test value
client.patch("/users/123", json={"bio": "New bio"})
```

---

## See Also

- **API Design Guide**: When to use sentinels vs Optional
- **Type Safety Guide**: Type narrowing with sentinels
- **Validation Guide**: Validation patterns with Pydantic
- **Sentinel Values API**: Complete sentinel types reference

---

## References

- [RFC 5789 – PATCH Method for HTTP](https://www.rfc-editor.org/rfc/rfc5789)
- [RFC 7396 – JSON Merge Patch](https://www.rfc-editor.org/rfc/rfc7396)
- [FastAPI Partial Updates](https://fastapi.tiangolo.com/tutorial/body-updates/)
- [REST API Design: PATCH Best Practices](https://williamdurand.fr/2014/02/14/please-do-not-patch-like-an-idiot/)

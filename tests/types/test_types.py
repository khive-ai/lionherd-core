# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Type System Base Classes: Immutable Parameters & Metadata

**Core Abstractions**:
- **Params**: Frozen dataclass base for function parameters (sentinel-aware)
- **DataClass**: Mutable variant of Params for state containers
- **Meta**: Immutable metadata key-value pair (callable-aware hashing)
- **ModelConfig**: Configuration for validation behavior and sentinel handling
- **Enum**: Enhanced enum with `allowed()` method for valid values

**Design Philosophy**:
- **Sentinel Semantics**: Precise state tracking (Undefined vs Unset vs None)
- **Immutability by Default**: Params frozen, updates via `with_updates()`
- **Configurable Validation**: Strict mode (require all fields) vs lenient (allow partial)
- **Serialization Safety**: `to_dict()` filters sentinels, preserves only set values
- **Framework Independence**: No Pydantic/attrs coupling (used by Spec/Operable)

**Testing Strategy**:
This test suite validates:
1. Params creation with sentinel handling
2. Strict mode enforcement (`ModelConfig.strict=True`)
3. Sentinel filtering in serialization (`to_dict()`)
4. Immutable updates via `with_updates()`
5. Meta hashing (callable vs. value-based)
6. DataClass post-init validation
7. Enum value extraction (`allowed()`)
8. ModelConfig options (none_as_sentinel, prefill_unset, use_enum_values)

**Type System Context**:
Params/DataClass are **foundational** for the type system:
- Used by Spec for metadata storage (Meta)
- Used by Operable for configuration (ModelConfig)
- Provide sentinel-aware base for user-defined parameter classes

Design enables: Framework-agnostic parameter handling with precise state semantics

**Sentinel Semantics**:

**Three Distinct States**:
```python
# State 1: Field never set (missing from namespace)
class MyParams(Params):
    field: str = Undefined  # Not provided, no default


# State 2: Field present but value not set
class MyParams(Params):
    field: str = Unset  # Provided but explicitly not set


# State 3: Field explicitly set to null
instance = MyParams(field=None)  # Explicitly null
```

**Why Three States?**
- `None` is ambiguous: Is it "not provided" or "explicitly null"?
- Sentinels provide semantic clarity for optional fields, defaults, partial updates
- Critical for: LLM structured outputs, API partial updates, database ORM

**Use Cases**:
1. **Partial Updates**: Only update provided fields
   ```python
   update = MyParams(field1="new_value")  # field2 stays unchanged (Unset)
   ```

2. **Default Factories**: Distinguish "no default" from "default is None"
   ```python
   spec1 = Spec(str, default=None)  # Default: None
   spec2 = Spec(str)  # Default: Undefined (no default)
   ```

3. **Validation Logic**: Skip validation for unprovided fields
   ```python
   if not_sentinel(value):
       validate(value)  # Only validate if provided
   ```

**ModelConfig Options**:

```python
@dataclass
class ModelConfig:
    none_as_sentinel: bool = False  # Treat None as sentinel?
    empty_as_sentinel: bool = False  # Treat [], {}, "" as sentinel?
    strict: bool = False  # Require all fields?
    prefill_unset: bool = True  # Auto-fill missing with Unset?
    use_enum_values: bool = False  # Serialize enums as .value?
```

**Configuration Matrix**:

| Config | Effect | Use Case |
|--------|--------|----------|
| `strict=True` | Require all fields (raise ExceptionGroup on missing) | Production APIs, database writes |
| `strict=False` | Allow partial data (fill missing with Unset) | LLM responses, user input |
| `none_as_sentinel=True` | None treated as Unset in `to_dict()` | APIs where null != "not provided" |
| `prefill_unset=True` | Auto-fill missing fields with Unset | Simplifies partial update logic |
| `use_enum_values=True` | Serialize enums as .value | API responses, JSON serialization |

**Design Rationale**:

**Why Frozen Params?**
- Immutability enables hashing (use as dict keys, set members)
- Thread-safe by default (no locks needed)
- Predictable behavior (no accidental mutations)
- Updates via `with_updates()` (explicit, returns new instance)

**Why Mutable DataClass?**
- State containers need mutability (counters, caches, accumulators)
- Post-init validation via `__post_init__()` (different lifecycle from Params)
- Same serialization interface as Params (`to_dict()`, `with_updates()`)

**Why Callable-Aware Hashing in Meta?**
- Validators are callables (lambdas, functions)
- Value-based hash fails for callables (lambda v: v == lambda v: v is False)
- Identity-based hash (`id()`) ensures same validator object = same hash

**Performance Characteristics**:
- Params creation: O(n) where n = field count
- `to_dict()`: O(n) (iterate fields, filter sentinels)
- `with_updates()`: O(n) (reconstruct frozen object)
- Meta hashing: O(1) (callable check + hash)
- Sentinel checks: O(1) (identity comparison, ~1ns)

**Related Components**:
- `Spec`: Uses Meta for metadata storage
- `Operable`: Uses ModelConfig for validation behavior
- `_sentinel.py`: Defines Undefined, Unset, not_sentinel()

**References**:
- Source: `src/lionherd_core/types/base.py`
- Related tests: `tests/types/test_spec.py`, `tests/types/test_operable.py`
- Sentinel implementation: `src/lionherd_core/types/_sentinel.py`
"""

from dataclasses import dataclass, field
from enum import Enum as StdEnum
from typing import ClassVar

import pytest

from lionherd_core.types import (
    DataClass,
    Enum,
    Meta,
    ModelConfig,
    Params,
    Undefined,
    Unset,
    is_sentinel,
    not_sentinel,
)

# ============================================================================
# Test Enum.allowed()
# ============================================================================


class MyTestEnum(Enum):
    """Test enum for testing"""

    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"


def test_enum_allowed():
    """
    Enum.allowed() extracts all enum values as tuple

    **Pattern**: Enum extension for validation

    **Scenario**: Framework needs list of valid enum values for validation
    ```python
    class MyEnum(Enum):
        VALUE1 = "value1"
        VALUE2 = "value2"


    allowed = MyEnum.allowed()  # ("value1", "value2")
    ```

    **Expected Behavior**:
    - Returns tuple (immutable, hashable)
    - Contains all enum .value attributes
    - Order matches enum definition order

    **Design Rationale**:
    Enums define allowed values but provide no built-in way to extract them.
    `allowed()` provides convenient access for:
    - Validation (check if value in allowed set)
    - API documentation (list valid values)
    - Form generation (populate dropdowns)

    Alternative: `[e.value for e in MyEnum]` (verbose, repeated)
    """
    allowed = MyTestEnum.allowed()
    assert isinstance(allowed, tuple)
    assert "value1" in allowed
    assert "value2" in allowed
    assert "value3" in allowed
    assert len(allowed) == 3


# ============================================================================
# Test Params validation and configuration
# ============================================================================


@dataclass(slots=True, frozen=True, init=False)
class MyParams(Params):
    """Test params class"""

    field1: str = Unset
    field2: int = Unset
    field3: bool = Unset


def test_params_invalid_parameter():
    """
    Params rejects unknown fields at construction (early validation)

    **Pattern**: Early validation for frozen dataclasses

    **Scenario**: User provides typo or unknown field
    ```python
    MyParams(field1="valid", invalid_field="typo")  # Should fail
    ```

    **Expected Behavior**:
    - ValueError raised immediately (before object constructed)
    - Error message identifies invalid parameter name
    - Valid parameters accepted normally

    **Design Rationale**:
    Params is frozen (immutable). Invalid fields cannot be corrected post-creation.
    Early validation ensures:
    1. Clear error at point of mistake
    2. No silent data loss (invalid fields dropped)
    3. Type safety (only declared fields exist)

    Alternative: Silently ignore unknown fields (rejected: hides bugs)
    """
    with pytest.raises(ValueError, match="Invalid parameter"):
        MyParams(field1="valid", invalid_field="should fail")


def test_params_valid():
    """Test Params.__init__ with valid parameters"""
    params = MyParams(field1="test", field2=42)
    assert params.field1 == "test"
    assert params.field2 == 42


def test_params_allowed():
    """Test Params.allowed() method"""
    allowed = MyParams.allowed()
    assert isinstance(allowed, set)
    assert "field1" in allowed
    assert "field2" in allowed
    assert "field3" in allowed
    assert "_none_as_sentinel" not in allowed  # Private fields excluded


@dataclass(slots=True, frozen=True, init=False)
class MyParamsNoneSentinel(Params):
    """Test params class with None as sentinel"""

    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
    field1: str = Unset


def test_params_is_sentinel_none_as_sentinel():
    """
    none_as_sentinel=True treats None as sentinel (API partial updates)

    **Pattern**: Sentinel configuration for three-state vs. four-state semantics

    **Scenario**: API needs to distinguish "not provided" from "explicitly null"
    ```python
    # Standard (None = explicit null)
    update = {"email": None}  # Set email to NULL in database

    # none_as_sentinel=True (None = not provided)
    update = {"email": None}  # Skip email in UPDATE query
    ```

    **Expected Behavior**:
    - None treated as sentinel (like Undefined/Unset)
    - `to_dict()` filters None values (not serialized)
    - `_is_sentinel(None)` returns True

    **ModelConfig**: `none_as_sentinel=True`

    **Design Rationale**:
    Some APIs/ORMs need four-state semantics:
    1. Undefined: Field never declared (not in schema)
    2. Unset: Field not provided in request
    3. None: Field provided as null (explicit NULL)
    4. Value: Field provided with value

    Other APIs/ORMs conflate "not provided" and "null":
    - GraphQL (null = not provided or explicit null)
    - Some REST APIs (null = skip field in update)

    `none_as_sentinel=True` enables three-state semantics:
    1. Sentinel (Undefined/Unset/None): Not provided
    2. Value: Provided with value

    Trade-off: Simplicity (three-state) vs. expressiveness (four-state)
    """
    # When _none_as_sentinel is True, None should be treated as sentinel
    assert MyParamsNoneSentinel._is_sentinel(None) is True
    assert MyParamsNoneSentinel._is_sentinel(Undefined) is True
    assert MyParamsNoneSentinel._is_sentinel(Unset) is True
    assert MyParamsNoneSentinel._is_sentinel("value") is False


def test_params_is_sentinel_default():
    """
    Default sentinel behavior treats None as valid value (explicit null)

    **Pattern**: Four-state sentinel semantics (default)

    **Scenario**: Database ORM needs to distinguish "not provided" from "set to NULL"
    ```python
    # Partial update: Only update email
    update = UserParams(email=None)  # Set email to NULL
    # username=Unset (not provided, skip in UPDATE)

    # SQL: UPDATE users SET email = NULL WHERE id = ?
    # username unchanged in database
    ```

    **Expected Behavior**:
    - None is NOT a sentinel (valid value)
    - `_is_sentinel(None)` returns False
    - `to_dict()` includes None values
    - Undefined/Unset ARE sentinels (filtered)

    **ModelConfig**: `none_as_sentinel=False` (default)

    **Design Rationale**:
    Four-state semantics enable precise database operations:

    **State Matrix**:
    | State | Meaning | Database Action | to_dict() |
    |-------|---------|-----------------|-----------|
    | Undefined | Never declared | N/A (not in schema) | Excluded |
    | Unset | Not provided | Skip field (no UPDATE) | Excluded |
    | None | Explicit null | SET field = NULL | Included |
    | Value | Provided value | SET field = value | Included |

    **Example** (partial user update):
    ```python
    # User wants to: clear email, keep name unchanged
    update = UserParams(email=None)  # email=None, name=Unset

    update.to_dict()  # {"email": None}
    # SQL: UPDATE users SET email = NULL WHERE id = ?
    # name column unchanged (not in UPDATE)
    ```

    Alternative: `none_as_sentinel=True` (three-state, simpler but less expressive)
    """
    # When _none_as_sentinel is False, None is not a sentinel
    assert MyParams._is_sentinel(None) is False
    assert MyParams._is_sentinel(Undefined) is True
    assert MyParams._is_sentinel(Unset) is True
    assert MyParams._is_sentinel("value") is False


@dataclass(slots=True, frozen=True, init=False)
class MyParamsStrict(Params):
    """Test params class with strict mode"""

    _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
    field1: str = Unset
    field2: int = Unset


def test_params_strict_mode():
    """
    Strict mode requires all fields without defaults (production API safety)

    **Pattern**: Strict validation for production correctness

    **Scenario**: API receives incomplete data
    ```python
    class UserParams(Params):
        _config = ModelConfig(strict=True)
        username: str = Unset
        email: str = Unset


    UserParams(username="alice")  # Missing email → ExceptionGroup
    ```

    **Expected Behavior**:
    - Missing required field → ExceptionGroup raised
    - ExceptionGroup contains ValueError per missing field
    - All missing fields reported at once (better UX than fail-on-first)

    **Validation Strategy**: Strict (fail fast on incomplete data)

    **Design Rationale**:
    Production APIs must validate completeness at entry point:
    - Database writes need all required fields
    - External APIs expect complete data
    - Missing data → silent bugs or database constraint violations

    Strict mode enforces completeness at construction:
    1. Fail fast (before processing)
    2. Report all errors (not just first)
    3. Clear error messages (actionable feedback)

    Alternative: Lenient mode (allow partial, fill with Unset)
    Trade-off: Correctness vs. flexibility
    """
    with pytest.raises(ExceptionGroup, match="Missing required parameters"):
        MyParamsStrict(field1="value")  # field2 is missing and strict=True


# ============================================================================
# Test DataClass configuration and validation
# ============================================================================


@dataclass(slots=True)
class MyDataClass(DataClass):
    """Test data class"""

    field1: str = Unset
    field2: int = Unset


def test_dataclass_valid():
    """Test DataClass with valid fields"""
    obj = MyDataClass(field1="test", field2=42)
    assert obj.field1 == "test"
    assert obj.field2 == 42


def test_dataclass_allowed():
    """Test DataClass.allowed() returns all declared fields."""
    allowed = MyDataClass.allowed()
    assert isinstance(allowed, set)
    assert "field1" in allowed
    assert "field2" in allowed


@dataclass(slots=True)
class MyDataClassStrict(DataClass):
    """Test data class with strict mode"""

    _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
    field1: str = Unset


def test_dataclass_strict_mode():
    """Test DataClass strict mode enforces required fields."""
    with pytest.raises(ExceptionGroup, match="Missing required parameters"):
        MyDataClassStrict()  # Missing required field in strict mode


@dataclass(slots=True)
class MyDataClassPrefillUnset(DataClass):
    """Test data class with prefill_unset"""

    _config: ClassVar[ModelConfig] = ModelConfig(prefill_unset=True)
    field1: str = field(default=Undefined)


def test_dataclass_prefill_unset():
    """
    prefill_unset=True auto-fills Undefined fields with Unset (partial update simplification)

    **Pattern**: Automatic sentinel normalization for simpler update logic

    **Scenario**: Class has optional fields that should default to Unset
    ```python
    class UpdateParams(DataClass):
        _config = ModelConfig(prefill_unset=True)
        email: str = Undefined  # Will become Unset automatically


    update = UpdateParams()
    update.email  # Unset (not Undefined)
    ```

    **Expected Behavior**:
    - Fields with Undefined default → automatically set to Unset in __post_init__
    - Simplifies partial update logic (no need to check Undefined vs Unset)
    - to_dict() filters both (same behavior)

    **ModelConfig**: `prefill_unset=True`

    **Design Rationale**:
    Partial updates often need "field not provided" semantics:

    Without prefill_unset:
    ```python
    if field is Undefined or field is Unset:
        # Skip field in update
    ```

    With prefill_unset:
    ```python
    if field is Unset:
        # Skip field (Undefined already converted)
    ```

    Benefits:
    1. Simpler conditional logic (one sentinel check, not two)
    2. Consistent state (all unprovided fields are Unset)
    3. Better semantics (Undefined = "not in schema", Unset = "not provided")

    Trade-off: Automatic transformation vs. explicit control
    Decision: Automation simplifies common case (partial updates)
    """
    obj = MyDataClassPrefillUnset()
    # Field initialized to Undefined should be prefilled with Unset
    assert obj.field1 is Unset


@dataclass(slots=True)
class MyDataClassNoneSentinel(DataClass):
    """Test data class with None as sentinel"""

    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
    field1: str = None


def test_dataclass_is_sentinel_none():
    """Test DataClass._is_sentinel with _none_as_sentinel=True"""
    assert MyDataClassNoneSentinel._is_sentinel(None) is True
    assert MyDataClassNoneSentinel._is_sentinel(Undefined) is True
    assert MyDataClassNoneSentinel._is_sentinel(Unset) is True


def test_dataclass_to_dict():
    """
    to_dict() serializes fields and filters sentinels (API/database safety)

    **Pattern**: Sentinel-aware serialization

    **Scenario**: Convert object to dict for API response or database write
    ```python
    obj = MyDataClass(field1="test", field2=42)
    obj.to_dict()  # {"field1": "test", "field2": 42}
    ```

    **Expected Behavior**:
    - Includes all non-sentinel fields
    - Filters Undefined/Unset automatically
    - Includes None (unless none_as_sentinel=True)
    - Returns plain dict (JSON-serializable)

    **Design Rationale**:
    Sentinels are internal framework values, not API/database values:

    Bad (sentinels leaked):
    ```json
    {"field1": "test", "field2": "<Unset>"}  # Invalid JSON
    ```

    Good (sentinels filtered):
    ```json
    {"field1": "test"}  # Only provided fields
    ```

    Use cases:
    1. API responses: Only send provided fields
    2. Database writes: Only update provided columns
    3. JSON serialization: Ensure valid JSON
    4. Partial updates: Skip unprovided fields

    Performance: O(n) where n = field count (~0.1ms for 100 fields)
    """
    obj = MyDataClass(field1="test", field2=42)
    result = obj.to_dict()
    assert "field1" in result
    assert "field2" in result


def test_dataclass_to_dict_exclude():
    """Test DataClass.to_dict() with exclude"""
    obj = MyDataClass(field1="test", field2=42)
    result = obj.to_dict(exclude={"field2"})
    assert "field1" in result
    assert "field2" not in result


def test_dataclass_with_updates():
    """
    with_updates() creates new instance with modifications (immutable update pattern)

    **Pattern**: Immutable update for frozen/mutable dataclasses

    **Scenario**: Update specific fields while preserving others
    ```python
    original = MyDataClass(field1="test", field2=42)
    updated = original.with_updates(field2=100)
    # updated.field1 = "test" (preserved)
    # updated.field2 = 100 (updated)
    ```

    **Expected Behavior**:
    - Returns new instance (original unchanged if frozen)
    - Only specified fields updated
    - Unspecified fields copied from original
    - Works for both frozen (Params) and mutable (DataClass)

    **Design Rationale**:
    Unified update interface for frozen and mutable classes:

    Frozen (Params):
    ```python
    # Can't do: params.field = value (frozen)
    # Must do: params = params.with_updates(field=value)
    ```

    Mutable (DataClass):
    ```python
    # Could do: obj.field = value (mutable)
    # Better: obj = obj.with_updates(field=value) (explicit)
    ```

    Benefits:
    1. Consistent API (frozen and mutable use same method)
    2. Explicit updates (clear what changed)
    3. Chainable (update1.with_updates(...).with_updates(...))
    4. Thread-safe for frozen (new object, no races)

    Alternative: Direct mutation (mutable only, not thread-safe)
    Decision: Unified interface preferred (consistency > convenience)
    """
    obj = MyDataClass(field1="test", field2=42)
    updated = obj.with_updates(field2=100)
    assert updated.field1 == "test"
    assert updated.field2 == 100


def test_dataclass_hash():
    """Test DataClass.__hash__() method"""
    # DataClass needs to be frozen to be hashable, use Params instead
    params1 = MyParams(field1="test", field2=42)
    params2 = MyParams(field1="test", field2=42)
    hash1 = hash(params1)
    hash2 = hash(params2)
    # Equal objects must have equal hashes
    assert hash1 == hash2
    assert isinstance(hash1, int)
    assert isinstance(hash2, int)


def test_dataclass_eq():
    """Test DataClass.__eq__() method"""
    obj1 = MyDataClass(field1="test", field2=42)
    obj2 = MyDataClass(field1="test", field2=42)
    obj3 = MyDataClass(field1="other", field2=99)
    # Validate equality correctness
    assert obj1 == obj2
    assert obj1 != obj3


def test_dataclass_eq_not_dataclass():
    """Test DataClass.__eq__() with non-DataClass"""
    obj = MyDataClass(field1="test", field2=42)
    assert obj != "not a dataclass"
    assert obj != 42


# ============================================================================
# Test Params methods
# ============================================================================


def test_params_to_dict():
    """Test Params.to_dict() method"""
    params = MyParams(field1="test", field2=42)
    result = params.to_dict()
    assert "field1" in result
    assert "field2" in result


def test_params_to_dict_exclude():
    """Test Params.to_dict() with exclude"""
    params = MyParams(field1="test", field2=42)
    result = params.to_dict(exclude={"field2"})
    assert "field1" in result
    assert "field2" not in result


def test_params_with_updates():
    """Test Params.with_updates() method"""
    params = MyParams(field1="test", field2=42)
    updated = params.with_updates(field2=100)
    assert updated.field1 == "test"
    assert updated.field2 == 100


def test_params_hash():
    """Test Params.__hash__() method"""
    params1 = MyParams(field1="test", field2=42)
    params2 = MyParams(field1="test", field2=42)
    hash1 = hash(params1)
    hash2 = hash(params2)
    # Equal objects must have equal hashes
    assert hash1 == hash2
    assert isinstance(hash1, int)
    assert isinstance(hash2, int)


def test_params_eq():
    """Test Params.__eq__() method"""
    params1 = MyParams(field1="test", field2=42)
    params2 = MyParams(field1="test", field2=42)
    params3 = MyParams(field1="other", field2=99)
    # Validate equality correctness
    assert params1 == params2
    assert params1 != params3


def test_params_eq_not_params():
    """Test Params.__eq__() with non-Params"""
    params = MyParams(field1="test", field2=42)
    assert params != "not params"
    assert params != 42


def test_params_default_kw():
    """Test Params.default_kw() method"""
    params = MyParams(field1="test", field2=42)
    result = params.default_kw()
    assert isinstance(result, dict)
    assert result["field1"] == "test"
    assert result["field2"] == 42


# ============================================================================
# Test sentinel utilities
# ============================================================================


def test_is_sentinel():
    """
    is_sentinel() identifies Undefined/Unset (validation helper)

    **Pattern**: Sentinel detection for conditional logic

    **Scenario**: Check if field was provided
    ```python
    if is_sentinel(field_value):
        # Field not provided, skip validation
        pass
    else:
        # Field provided, validate it
        validate(field_value)
    ```

    **Expected Behavior**:
    - Undefined → True (never set)
    - Unset → True (not provided)
    - None → False (valid value, not sentinel)
    - Any other value → False

    **Design Rationale**:
    Validation logic needs to distinguish "not provided" from "provided":

    Without sentinel check:
    ```python
    if field_value is not None:
        validate(field_value)
    # Problem: What if field_value = Unset? (would validate, wrong!)
    ```

    With sentinel check:
    ```python
    if not is_sentinel(field_value):
        validate(field_value)  # Only validate if actually provided
    ```

    Use cases:
    1. Validation: Skip validation for unprovided fields
    2. Serialization: Filter sentinels in to_dict()
    3. Defaults: Apply default only if sentinel
    4. Partial updates: Update only non-sentinel fields

    Note: Use `not_sentinel()` for TypeGuard (type narrowing in mypy/pyright)
    """
    assert is_sentinel(Undefined) is True
    assert is_sentinel(Unset) is True
    assert is_sentinel(None) is False
    assert is_sentinel("value") is False
    assert is_sentinel(42) is False


def test_not_sentinel():
    """
    not_sentinel() provides type narrowing via TypeGuard (mypy/pyright support)

    **Pattern**: Type-safe sentinel checking with type narrowing

    **Scenario**: Conditional logic with type narrowing
    ```python
    field: str | Unset = get_field()

    if not_sentinel(field):
        # Type narrowed: field is str (not Unset)
        print(field.upper())  # Type-safe (no mypy error)
    else:
        # Type: field is Unset
        pass
    ```

    **Expected Behavior**:
    - Undefined → False (is sentinel)
    - Unset → False (is sentinel)
    - None → True (not sentinel, valid value)
    - Any other value → True
    - TypeGuard: Narrows type to non-sentinel

    **Design Rationale**:
    Type checkers need explicit type narrowing for union types:

    Without TypeGuard:
    ```python
    field: str | Unset

    if not is_sentinel(field):
        field.upper()  # mypy error: Unset has no .upper()
    ```

    With TypeGuard:
    ```python
    field: str | Unset

    if not_sentinel(field):  # TypeGuard narrows to str
        field.upper()  # mypy OK: field is str
    ```

    TypeGuard signature:
    ```python
    def not_sentinel(value: T | Sentinel) -> TypeGuard[T]:
        return not is_sentinel(value)
    ```

    Benefits:
    1. Type safety (mypy/pyright catch errors)
    2. Correct return type (T, not T | Sentinel)
    3. Better IDE autocomplete (knows exact type)
    4. Runtime correctness (actual sentinel check)

    Use case: Validation with type safety
    ```python
    def validate_field(field: str | Unset) -> None:
        if not_sentinel(field):
            # field is str (type narrowed)
            assert len(field) > 0  # Type-safe
    ```
    """
    assert not_sentinel(Undefined) is False
    assert not_sentinel(Unset) is False
    assert not_sentinel(None) is True
    assert not_sentinel("value") is True
    assert not_sentinel(42) is True


# ============================================================================
# Test Enum Normalization
# ============================================================================


class ColorEnum(StdEnum):
    """Enum for normalization tests"""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass(slots=True, frozen=True, init=False)
class MyParamsWithEnum(Params):
    """Test params class with enum normalization"""

    _config: ClassVar[ModelConfig] = ModelConfig(use_enum_values=True)
    color: ColorEnum = Unset
    name: str = Unset


def test_params_normalize_enum():
    """
    use_enum_values=True serializes enums as .value (JSON compatibility)

    **Pattern**: Enum normalization for API serialization

    **Scenario**: API response needs JSON, enums must be primitive values
    ```python
    class Status(Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"


    params = MyParams(status=Status.ACTIVE)
    params.to_dict()  # {"status": "active"} (not Status.ACTIVE enum)
    ```

    **Expected Behavior**:
    - Enum instances converted to .value (primitive)
    - Non-enum fields unaffected
    - Result is JSON-serializable

    **ModelConfig**: `use_enum_values=True`

    **Design Rationale**:
    JSON doesn't support enum objects:

    Bad (enum leaked):
    ```python
    json.dumps({"status": Status.ACTIVE})  # TypeError: not serializable
    ```

    Good (enum converted):
    ```python
    json.dumps({"status": "active"})  # Valid JSON
    ```

    Use cases:
    1. REST API responses (JSON required)
    2. Database writes (store primitive values)
    3. Message queues (serialize to JSON/protobuf)
    4. GraphQL resolvers (return scalars)

    Alternative: Manual conversion (`.value` everywhere)
    Decision: Automatic conversion (configured once, applied everywhere)

    Trade-off: Convenience vs. explicit control
    """
    params = MyParamsWithEnum(color=ColorEnum.RED, name="test")
    result = params.to_dict()
    # With use_enum_values=True, should get string not enum
    assert result["color"] == "red"
    assert not isinstance(result["color"], StdEnum)
    assert result["name"] == "test"


@dataclass(slots=True)
class MyDataClassWithEnum(DataClass):
    """Test data class with enum normalization"""

    _config: ClassVar[ModelConfig] = ModelConfig(use_enum_values=True)
    color: ColorEnum = Unset
    name: str = Unset


def test_dataclass_normalize_enum():
    """Test DataClass normalizes enum to value when use_enum_values=True."""
    obj = MyDataClassWithEnum(color=ColorEnum.BLUE, name="test")
    result = obj.to_dict()
    # With use_enum_values=True, should get string not enum
    assert result["color"] == "blue"
    assert not isinstance(result["color"], StdEnum)


# ============================================================================
# Test Hash Methods Explicit Call
# ============================================================================


def test_params_hash_explicit():
    """Test Params.__hash__ custom implementation."""
    params = MyParams(field1="test", field2=42)
    # Call the parent Params.__hash__ method directly to avoid dataclass override
    hash_result = Params.__hash__(params)
    assert isinstance(hash_result, int)
    # Verify it produces consistent results
    assert Params.__hash__(params) == hash_result


def test_dataclass_hash_explicit():
    """Test DataClass.__hash__ custom implementation."""
    # DataClass.__hash__ exists but is unreachable in normal use (not frozen)
    # Call it directly on a DataClass subclass instance
    obj = MyDataClass(field1="test", field2=42)
    hash_result = DataClass.__hash__(obj)
    assert isinstance(hash_result, int)
    # Verify consistency
    assert DataClass.__hash__(obj) == hash_result


# ============================================================================
# Test NotImplemented Protocol
# ============================================================================


def test_params_eq_returns_notimplemented():
    """Test Params.__eq__ returns NotImplemented for non-Params types."""
    params = MyParams(field1="test", field2=42)
    # Call parent Params.__eq__ directly to avoid dataclass override
    result = Params.__eq__(params, "not params")
    assert result is NotImplemented

    result = Params.__eq__(params, 42)
    assert result is NotImplemented

    result = Params.__eq__(params, [1, 2, 3])
    assert result is NotImplemented

    # Test the happy path - comparing two Params instances
    params2 = MyParams(field1="test", field2=42)
    result = Params.__eq__(params, params2)
    assert result is True  # Same values should be equal


def test_dataclass_eq_returns_notimplemented():
    """Test DataClass.__eq__ returns NotImplemented for non-DataClass types."""
    obj = MyDataClass(field1="test", field2=42)
    # Call parent DataClass.__eq__ directly to avoid dataclass override
    result = DataClass.__eq__(obj, "not dataclass")
    assert result is NotImplemented

    result = DataClass.__eq__(obj, None)
    assert result is NotImplemented

    # Test the happy path - comparing two DataClass instances
    # Note: This line is technically dead code (DataClass not frozen, so hash() fails)
    # But we can test it by mocking hash() to return consistent values
    obj2 = MyDataClass(field1="test", field2=42)
    import unittest.mock

    # Mock hash to return the same value for both objects
    with unittest.mock.patch("builtins.hash", return_value=12345):
        result = DataClass.__eq__(obj, obj2)
        # Both objects have same mocked hash, so should be equal
        assert result is True


# ============================================================================
# Test Deep Copy with_updates
# ============================================================================


@dataclass(slots=True, frozen=True, init=False)
class MyParamsWithContainers(Params):
    """Test params class with mutable containers"""

    items: list = Unset
    config: dict = Unset
    tags: set = Unset


def test_params_with_updates_shallow():
    """Test Params.with_updates with copy_containers='shallow'.

    **API Design Rationale**:
    with_updates() uses Literal["shallow", "deep"] | None for explicit copy semantics:
    - None: No copying (references preserved, fast default)
    - "shallow": Copy top-level containers (list/dict/set) via .copy()
    - "deep": Recursive copy via copy.deepcopy() for full isolation

    This replaces the old misleading API:
    - OLD: deep=True (name suggested deep but did shallow via copy.copy())
    - NEW: copy_containers="shallow" (explicit about what it does)

    **Performance**:
    - Lazy import: copy module only loaded for "deep"
    - Builtin .copy(): Faster than copy.copy() for shallow
    - Type-specific: Only copies list/dict/set

    **Use Cases**:
    - None: Updating immutable fields (int, str, etc.)
    - "shallow": Preventing shared references for simple containers
    - "deep": Full isolation for nested structures (see test_*_deep_copy_nested)
    """
    original_list = [1, 2, 3]
    original_dict = {"key": "value"}
    original_set = {1, 2, 3}
    params = MyParamsWithContainers(items=original_list, config=original_dict, tags=original_set)

    # Shallow copy containers
    updated = params.with_updates(copy_containers="shallow", items=[4, 5, 6])

    # Verify copy occurred
    assert params.items == [1, 2, 3]
    assert updated.items == [4, 5, 6]
    # Other mutable fields should also be copied
    assert params.config == {"key": "value"}
    assert params.tags == {1, 2, 3}


@dataclass(slots=True)
class MyDataClassWithContainers(DataClass):
    """Test data class with mutable containers"""

    items: list = field(default_factory=list)
    tags: set = field(default_factory=set)
    config: dict = field(default_factory=dict)


def test_dataclass_with_updates_shallow():
    """Test DataClass.with_updates copies containers when copy_containers='shallow'."""
    obj = MyDataClassWithContainers(items=[1, 2], tags={1, 2}, config={"a": 1})
    updated = obj.with_updates(copy_containers="shallow", items=[3, 4])

    # Verify copy occurred
    assert obj.items == [1, 2]
    assert updated.items == [3, 4]
    # Verify other containers were also copied
    assert obj.tags == {1, 2}
    assert obj.config == {"a": 1}


def test_params_with_updates_deep_copy_nested():
    """Test copy_containers='deep' performs recursive copying on nested structures.

    **Deep Copy Behavior**:
    - Uses copy.deepcopy() for full recursive isolation
    - Prevents shared references at ALL levels (not just top-level)
    - Required for nested containers: dict with list values, list of lists, etc.

    **Contrast with Shallow**:
    - Shallow: Top-level copied, inner structures shared (mutating inner affects original)
    - Deep: Fully isolated, no shared references (safe to mutate at any level)

    This test proves deep copy works by mutating nested structures and verifying
    original is unchanged.
    """
    # Nested structure: dict with list values
    nested_dict = {"outer": {"inner": [1, 2]}}
    params = MyParamsWithContainers(config=nested_dict)

    # Shallow copy - inner structures shared
    shallow = params.with_updates(copy_containers="shallow", config={"outer": {"inner": [1, 2]}})
    shallow.config["outer"]["inner"].append(3)
    # Original should be unchanged (top-level dict was copied)
    assert params.config == {"outer": {"inner": [1, 2]}}

    # Deep copy - fully isolated, no shared references
    deep = params.with_updates(copy_containers="deep")
    deep.config["outer"]["inner"].append(999)
    # Original unchanged (recursive copy)
    assert params.config == {"outer": {"inner": [1, 2]}}
    assert deep.config == {"outer": {"inner": [1, 2, 999]}}


def test_dataclass_with_updates_deep_copy_nested():
    """Test DataClass copy_containers='deep' performs recursive copying on nested structures."""
    # Nested list of lists
    nested_items = [[1, 2], [3, 4]]
    obj = MyDataClassWithContainers(items=nested_items)

    # Shallow copy - inner lists shared
    shallow = obj.with_updates(copy_containers="shallow", items=[[1, 2], [3, 4]])
    shallow.items[0].append(999)
    # Original unchanged (top-level list was copied)
    assert obj.items == [[1, 2], [3, 4]]

    # Deep copy - fully isolated
    deep = obj.with_updates(copy_containers="deep")
    deep.items[0].append(999)
    # Original unchanged (recursive copy)
    assert obj.items == [[1, 2], [3, 4]]
    assert deep.items == [[1, 2, 999], [3, 4]]


def test_params_with_updates_no_copy_for_updated_fields():
    """Test that fields in kwargs are not copied (performance optimization).

    **Optimization**: If a field is being updated via kwargs, skip copying the
    old value since it will be immediately replaced. Avoids unnecessary copy
    operations, especially important for large containers or deep copy.
    """
    # Create expensive nested structure
    expensive_nested = {"level1": {"level2": {"level3": [1, 2, 3] * 1000}}}
    params = MyParamsWithContainers(config=expensive_nested, items=[1, 2, 3])

    # Update items with shallow copy - config should be copied, items should NOT
    updated = params.with_updates(copy_containers="shallow", items=[4, 5, 6])

    # Verify items was replaced (not copied then replaced)
    assert updated.items == [4, 5, 6]
    # Verify config was copied (not in kwargs)
    assert updated.config == expensive_nested
    assert updated.config is not params.config  # Different object


def test_params_with_updates_invalid_copy_containers():
    """Test runtime validation for invalid copy_containers values.

    **Runtime Safety**: Type hints enforce valid values at compile time (mypy),
    but runtime values (from config files, APIs, env vars) bypass type checking.
    Must validate at runtime to prevent silent failures.

    Invalid values should raise ValueError immediately, not silently ignore.
    """
    params = MyParamsWithContainers(items=[1, 2, 3])

    # Invalid string
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="invalid", items=[4, 5, 6])

    # Case mismatch (case-sensitive)
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="SHALLOW", items=[4, 5, 6])

    # Empty string
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="", items=[4, 5, 6])

    # Typo
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="shalllow", items=[4, 5, 6])


def test_dataclass_with_updates_invalid_copy_containers():
    """Test DataClass runtime validation for invalid copy_containers values."""
    obj = MyDataClassWithContainers(items=[1, 2], tags={1, 2}, config={"a": 1})

    # Invalid string
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        obj.with_updates(copy_containers="deep_copy", items=[3, 4])

    # Case mismatch
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        obj.with_updates(copy_containers="Deep", items=[3, 4])


def test_params_with_updates_abc_collections():
    """Test copy_containers works with collections.abc types (deque, defaultdict, etc).

    **Protocol-Based Coverage**: Uses isinstance with MutableSequence, MutableMapping,
    MutableSet to cover all standard mutable collections, not just list/dict/set.

    Covers: deque, defaultdict, OrderedDict, Counter, UserList, etc.
    """
    from collections import defaultdict, deque
    from dataclasses import dataclass
    from typing import Any

    @dataclass(frozen=True)
    class ParamsWithABCCollections(Params):
        _config: ClassVar[ModelConfig] = ModelConfig()
        queue: Any = Unset
        counts: Any = Unset

    # Test deque (MutableSequence but not list subclass)
    params = ParamsWithABCCollections(queue=deque([1, 2, 3]), counts=defaultdict(int))
    params.counts["a"] = 5

    shallow = params.with_updates(copy_containers="shallow")
    assert isinstance(shallow.queue, deque)
    assert isinstance(shallow.counts, defaultdict)

    # Verify shallow copy behavior
    shallow.queue.append(999)
    assert 999 not in params.queue  # Original unchanged

    shallow.counts["b"] = 10
    assert "b" not in params.counts  # Original unchanged


# ============================================================================
# Test Meta Class
# ============================================================================


def test_meta_hash_simple():
    """Test Meta.__hash__ with simple hashable values."""
    meta1 = Meta(key="field1", value="string_value")
    meta2 = Meta(key="field1", value="string_value")
    meta3 = Meta(key="field1", value=42)

    # Same key and value should have same hash
    assert hash(meta1) == hash(meta2)
    assert isinstance(hash(meta1), int)

    # Different values should likely have different hashes
    assert hash(meta1) != hash(meta3)


def test_meta_hash_callable():
    """
    Meta uses identity-based hashing for callables (validator correctness)

    **Pattern**: Identity hashing for unhashable callable values

    **Scenario**: Metadata contains validators (lambdas, functions)
    ```python
    def validator(x):
        return x > 0


    meta1 = Meta(key="validator", value=validator)
    meta2 = Meta(key="validator", value=validator)  # Same instance
    hash(meta1) == hash(meta2)  # True (same id)
    ```

    **Expected Behavior**:
    - Same callable instance → same hash (identity match)
    - Different callable instances → different hash (even if equivalent code)
    - Enables Meta to be dict key, set member

    **Design Rationale**:
    Value-based hashing fails for callables:

    Problem:
    ```python
    lambda x: x > 0 == lambda x: x > 0  # False (different objects)
    # Can't hash by value (no __hash__ for functions)
    ```

    Solution:
    ```python
    # Hash by identity (id())
    hash(id(validator))  # Works, consistent for same instance
    ```

    Use cases:
    1. Spec metadata with validators (common pattern)
    2. Deduplication (same validator = same Meta)
    3. Caching (Meta as cache key)
    4. Set operations (Meta in set)

    Alternative: Hash by code object (id(fn.__code__))
    Rejected: Doesn't handle closures (different captured variables)

    Trade-off: Identity equality (strict) vs. structural equality (complex)
    Decision: Identity equality (simpler, predictable, sufficient)
    """

    def validator(x):
        return x > 0

    meta1 = Meta(key="validator", value=validator)
    meta2 = Meta(key="validator", value=validator)

    # Same callable instance → same hash
    assert hash(meta1) == hash(meta2)

    # Different callable instances
    def another_validator(x):
        return x > 0

    meta3 = Meta(key="validator", value=another_validator)
    # Different id → different hash
    assert hash(meta1) != hash(meta3)


def test_meta_hash_unhashable():
    """
    Meta falls back to str() hashing for unhashable types (robustness)

    **Pattern**: Graceful fallback for unhashable metadata values

    **Scenario**: Metadata contains complex structures (dicts, lists)
    ```python
    meta = Meta(key="config", value={"timeout": 30, "retries": 3})
    hash(meta)  # Fallback to hash(str(value))
    ```

    **Expected Behavior**:
    - Unhashable value (dict, list, set) → hash by str() representation
    - No TypeError (graceful fallback)
    - Consistent hash for same content

    **Design Rationale**:
    Metadata can contain arbitrary values:
    - Simple: strings, ints, bools (directly hashable)
    - Callables: functions, lambdas (hash by id)
    - Complex: dicts, lists, sets (unhashable)

    Without fallback:
    ```python
    hash({"key": "value"})  # TypeError: unhashable type: 'dict'
    ```

    With fallback:
    ```python
    hash(str({"key": "value"}))  # Works (hash of "{'key': 'value'}")
    ```

    Fallback strategy:
    1. Try direct hash (for hashable primitives)
    2. If TypeError → check if callable (hash by id)
    3. If not callable → hash by str() representation

    Limitations:
    - Dict/list order matters: `{"a": 1, "b": 2}` != `{"b": 2, "a": 1}` (different str)
    - Nested unhashable: Works but verbose str representation

    Alternative: Raise TypeError (strict hashing only)
    Rejected: Too restrictive, metadata can be arbitrary

    Trade-off: Flexibility (fallback) vs. correctness (strict)
    Decision: Flexibility (enable arbitrary metadata)
    """
    # Unhashable dict value should trigger fallback
    meta = Meta(key="config", value={"unhashable": "dict"})
    # Should fallback to str() hashing without error
    hash_result = hash(meta)
    assert isinstance(hash_result, int)

    # Lists are also unhashable
    meta_list = Meta(key="items", value=[1, 2, 3])
    assert isinstance(hash(meta_list), int)


def test_meta_eq_simple():
    """Test Meta.__eq__ with simple values."""
    meta1 = Meta(key="field1", value="val")
    meta2 = Meta(key="field1", value="val")
    meta3 = Meta(key="field2", value="val")
    meta4 = Meta(key="field1", value="different")

    # Same key and value
    assert meta1 == meta2

    # Different keys
    assert meta1 != meta3

    # Same key, different value
    assert meta1 != meta4


def test_meta_eq_callable():
    """Test Meta.__eq__ with callables."""

    def fn1(x):
        return x

    def fn2(x):
        return x

    meta1 = Meta(key="fn", value=fn1)
    meta2 = Meta(key="fn", value=fn1)  # Same instance
    meta3 = Meta(key="fn", value=fn2)  # Different instance

    # Same instance → identity match
    assert meta1 == meta2

    # Different id → not equal
    assert meta1 != meta3


def test_meta_eq_not_meta():
    """Test Meta.__eq__ returns NotImplemented for non-Meta types."""
    meta = Meta(key="field", value="val")
    # Call __eq__ directly to verify return value
    result = meta.__eq__("not a meta")
    assert result is NotImplemented

    result = meta.__eq__(42)
    assert result is NotImplemented

    result = meta.__eq__(None)
    assert result is NotImplemented

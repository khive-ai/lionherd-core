# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Operable Tests: Validated Spec Collections for Model Generation

**Core Abstractions**:
- **Operable**: Ordered, validated collection of Specs with name uniqueness enforcement
- **Field Filtering**: Include/exclude subsets for partial model generation
- **Name Uniqueness**: Enforced at construction (duplicate detection)
- **Model Generation**: Delegates to framework-specific adapters (Pydantic, attrs, etc.)

**Design Philosophy**:
- **Single Responsibility**: Operable manages collection, Adapter handles generation
- **Early Validation**: Name uniqueness and type checking at construction time
- **Composability**: Generate multiple models from same Operable (different includes/excludes)
- **Framework Agnostic**: No direct Pydantic/attrs dependencies (pure specification layer)

**Testing Strategy**:
This test suite validates:
1. Operable creation with duplicate name detection and type validation
2. Field filtering (include/exclude combinations, overlap detection)
3. Field access (`get()`, `allowed()`, `check_allowed()` with various modes)
4. Model generation delegation to adapters (error handling, naming conventions)
5. Immutability enforcement (frozen dataclass semantics)
6. Field ordering preservation (insertion order maintained)

**Type System Context**:
Operable is **Layer 2** of the 3-layer architecture:
- **Spec** (Layer 1): Individual field schema (framework-agnostic)
- **Operable** (Layer 2 - this): Validated Spec collection
- **Adapter** (Layer 3): Framework-specific model generation

Design enables:
1. Define schema once (Operable) → Reuse across frameworks
2. Generate multiple models (include/exclude subsets) → Public/internal views
3. Support multiple frameworks (swap adapters) → Pydantic, attrs, dataclasses

**Design Rationale**:

**Why Validate Names at Construction?**
- Specs are immutable (can't change names after creation)
- Operable is immutable (can't add/remove Specs after creation)
- Early validation = clear errors at point of mistake (fail fast principle)
- Alternative (lazy validation) rejected: Would allow invalid Operables to exist

**Why Tuple[Spec, ...]?**
- **Ordered**: Preserve field definition order (matters for serialization, UI forms)
- **Immutable**: Safe caching, thread-safe sharing across workers
- **Hashable**: Can hash entire Operable (future: cache generated models)
- Alternative (list) rejected: Mutability breaks caching, not hashable

**Why Include/Exclude?**
- **Common pattern**: Public vs. internal fields (e.g., password field exclusion)
  ```python
  PublicUser = op.create_model(exclude={"password", "internal_id"})
  FullUser = op.create_model()  # All fields for admin
  ```
- **Avoid duplication**: One schema, multiple views (DRY principle)
- **Type safety**: Filtered fields still type-checked at compile time
- Alternative (multiple Operables) rejected: Duplication, harder to maintain

**Why Delegate to Adapters?**
- **Framework isolation**: Operable has zero framework knowledge
- **Plugin architecture**: Add new frameworks without modifying Operable
- **Single Responsibility**: Operable = collection, Adapter = generation
- Alternative (direct Pydantic) rejected: Tight coupling, can't support attrs/msgspec

**Performance Characteristics**:
- **Creation**: O(n) where n = spec count (uniqueness check via set)
- **allowed()**: O(1) (cached set, computed once)
- **get()**: O(n) (linear search by name, specs typically <50)
- **get_specs()**: O(n) (filter specs by include/exclude)
- **create_model()**: O(adapter_overhead) (delegates entirely to adapter)

**Include/Exclude Validation**:

| Scenario | Validation | Error | Rationale |
|----------|-----------|-------|-----------|
| `include={"a", "b"}` | Check all names in Operable | ValueError if unknown | Typo protection, explicit contract |
| `exclude={"a", "b"}` | Check all names in Operable | ValueError if unknown | Typo protection, explicit contract |
| `include={"a"}, exclude={"a"}` | Check no overlap | ValueError on overlap | Ambiguous semantics, explicit error |
| No include/exclude | No validation | All specs included | Default: include everything |

**Collection Semantics**:

**Immutability Contract**:
- Operable is frozen dataclass (Python `@dataclass(frozen=True)`)
- Attempting to modify `__op_fields__` or `name` raises `FrozenInstanceError`
- Updates require creating new Operable (functional update pattern)

**Iteration Protocol**:
- Operable is not directly iterable (no `__iter__`)
- Access specs via `get_specs()` (explicit filtering support)
- Design choice: Explicit over implicit (clarity in code)

**Equality Semantics**:
- Operables equal if same specs in same order + same name
- Order matters: `Operable([a, b])` ≠ `Operable([b, a])`
- Rationale: Field order affects serialization, UI generation

**Related Components**:
- `Spec`: Individual field schema (building block, Layer 1)
- `PydanticSpecAdapter`: Pydantic model generation (Layer 3)
- `ModelConfig`: Validation behavior configuration (used by adapters)
- `Meta`: Metadata key-value pairs (stored in Spec, not Operable)

**References**:
- Source: `src/lionherd_core/types/operable.py`
- Related tests: `tests/types/test_spec.py`, `tests/types/spec_adapters/test_adapters_py311.py`
- Architecture: `.khive/workspaces/test_docs_002_types_analysis/types_architecture_analysis.md`
"""

import builtins
from unittest.mock import patch

import pytest

from lionherd_core.types import Operable, Spec, Unset


class TestOperable:
    """
    Operable Core Functionality: Creation, Access, Filtering, and Model Generation

    **Tested Behaviors**:
    - **Creation**: Basic instantiation, empty Operable, list-to-tuple conversion
    - **Validation**: Duplicate name detection, type validation, immutability enforcement
    - **Field Access**: `get()`, `allowed()`, `check_allowed()` with various modes
    - **Filtering**: `get_specs()` with include/exclude (validation and overlap detection)
    - **Model Generation**: Adapter delegation (Pydantic), error handling, naming conventions
    - **Ordering**: Field insertion order preservation

    **Edge Cases**:
    - Empty Operable (zero specs) → Valid, `allowed()` returns empty set
    - Single spec → Minimal valid collection
    - Missing field access → `get()` returns `Unset` (not None, not error)
    - Invalid adapter name → Clear error message with supported adapters
    - Pydantic unavailable → Helpful ImportError with install instructions

    **Pattern Context**:
    These tests validate the collection semantics of Operable. Unlike Spec (individual field),
    Operable manages relationships between fields (uniqueness, ordering, filtering).

    The 3-layer architecture is validated here:
    1. Spec creation (Layer 1) → Tested in test_spec.py
    2. Operable collection (Layer 2) → **Tested here**
    3. Adapter generation (Layer 3) → Integration tested here, full tests in test_adapters_py311.py

    **Design Validation**:
    - Early validation principle: Duplicate names caught at construction (not later)
    - Immutability enforcement: Frozen dataclass prevents accidental mutation
    - Explicit over implicit: `get_specs()` instead of `__iter__` (clarity in filtering)
    """

    def test_basic_creation(self):
        """
        Basic Operable creation with named Specs

        **Pattern**: Simple collection instantiation

        **Scenario**: Create Operable with two Specs of different types
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2), name="TestModel")
        ```

        **Expected Behavior**:
        - Operable stores both specs in `__op_fields__`
        - Length matches spec count (2)
        - Name attribute preserved ("TestModel")
        - No validation errors (unique names, valid Spec types)

        **Design Rationale**:
        Operable name is optional but recommended for generated model naming.
        When `create_model()` is called without `model_name`, Operable name is used.
        """
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2), name="TestModel")
        assert len(operable.__op_fields__) == 2
        assert operable.name == "TestModel"

    def test_empty_operable(self):
        """
        Empty Operable is valid (zero specs)

        **Pattern**: Minimal valid collection (edge case)

        **Scenario**: Create Operable without any specs
        ```python
        operable = Operable()  # No specs provided
        ```

        **Expected Behavior**:
        - `__op_fields__` is empty tuple (length 0)
        - `allowed()` returns empty set (no field names)
        - No validation errors (empty is valid)

        **Design Rationale**:
        Empty Operable is useful for:
        1. Base class models (no fields, just behavior)
        2. Incremental construction patterns (start empty, build up)
        3. Testing edge cases in adapters

        Alternative (require at least one spec) rejected: Reduces flexibility,
        breaks composition patterns where specs added dynamically.
        """
        operable = Operable()
        assert len(operable.__op_fields__) == 0
        assert operable.allowed() == set()

    def test_allowed(self):
        """
        allowed() returns set of all field names

        **Pattern**: Query collection membership

        **Scenario**: Extract all field names from Operable for validation
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))
        allowed = operable.allowed()  # {"field1", "field2"}
        ```

        **Expected Behavior**:
        - Returns set (not list/tuple) for O(1) membership checks
        - Contains all field names from specs
        - No duplicates (set semantics)

        **Design Rationale**:
        Set return type enables efficient validation:
        ```python
        if field_name in operable.allowed():  # O(1) check
            process(field_name)
        ```
        Alternative (list) rejected: O(n) membership check, duplicates possible.
        """
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))
        allowed = operable.allowed()
        assert allowed == {"field1", "field2"}

    def test_check_allowed_valid(self):
        """
        check_allowed() returns True for valid field names (default mode)

        **Pattern**: Field validation with exception on error

        **Scenario**: Validate that a field name exists in Operable
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        operable.check_allowed("field1")  # True
        ```

        **Expected Behavior**:
        - Returns `True` if field name is in `allowed()`
        - Does not raise exception for valid field

        **Design Rationale**:
        Default mode (raise on error) is safer for production:
        - Forces explicit error handling
        - Fails fast on typos
        - Clear error messages with field name
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        assert operable.check_allowed("field1") is True

    def test_check_allowed_invalid_raises(self):
        """
        check_allowed() raises ValueError for invalid field names (default mode)

        **Pattern**: Fail-fast validation

        **Scenario**: Attempt to validate non-existent field
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        operable.check_allowed("field2")  # ValueError: "field2" not allowed
        ```

        **Expected Behavior**:
        - Raises `ValueError` with descriptive message
        - Error message includes invalid field name
        - Error message includes allowed fields (for debugging)

        **Design Rationale**:
        Explicit errors prevent silent failures:
        - Typos caught immediately (field2 vs field1)
        - Clear feedback for developers
        - Production-safe (strict by default)

        Alternative (return False) considered: Too easy to ignore,
        silent failures in validation logic.
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        with pytest.raises(ValueError, match="not allowed"):
            operable.check_allowed("field2")

    def test_check_allowed_as_boolean(self):
        """
        check_allowed() with as_boolean=True returns bool (no exception)

        **Pattern**: Boolean validation (lenient mode)

        **Scenario**: Check field existence without exception handling
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        exists = operable.check_allowed("field1", as_boolean=True)  # True
        missing = operable.check_allowed("field2", as_boolean=True)  # False
        ```

        **Expected Behavior**:
        - Returns `True` if field exists
        - Returns `False` if field missing (no exception)
        - Useful for conditional logic without try/except

        **Design Rationale**:
        Boolean mode enables cleaner code for optional fields:
        ```python
        # With boolean mode (clean)
        if operable.check_allowed("optional_field", as_boolean=True):
            process(field)

        # Without boolean mode (verbose)
        try:
            operable.check_allowed("optional_field")
            process(field)
        except ValueError:
            pass
        ```

        Trade-off: Boolean mode less safe (easy to ignore False),
        but more ergonomic for optional field handling.
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        assert operable.check_allowed("field1", as_boolean=True) is True
        assert operable.check_allowed("field2", as_boolean=True) is False

    def test_get_existing(self):
        """
        get() returns Spec for existing field name

        **Pattern**: Dictionary-like field retrieval

        **Scenario**: Retrieve Spec by field name
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        result = operable.get("field1")  # Returns spec1
        ```

        **Expected Behavior**:
        - Returns the Spec object with matching name
        - Identity preserved (`result is spec1`)
        - No copy/clone (returns original reference)

        **Design Rationale**:
        `get()` provides dict-like access pattern:
        ```python
        spec = operable.get("field_name")
        if spec is not Unset:
            use_spec(spec)
        ```
        Alternative (raise KeyError) rejected: Too strict, forces try/except.
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        result = operable.get("field1")
        assert result is spec1

    def test_get_missing_returns_unset(self):
        """
        get() returns Unset for missing field (not None, not error)

        **Pattern**: Sentinel-based optional retrieval

        **Scenario**: Attempt to retrieve non-existent field
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        result = operable.get("field2")  # Returns Unset
        ```

        **Expected Behavior**:
        - Returns `Unset` sentinel (not `None`, not exception)
        - Can check with `result is Unset`
        - Enables conditional logic without exception handling

        **Design Rationale**:
        Why `Unset` instead of `None`?
        - `None` is ambiguous: Could be "missing" or "explicitly None"
        - `Unset` is unambiguous: Definitely means "not found"
        - Consistent with Spec sentinel semantics (Undefined, Unset, None)

        Pattern enables clean optional field access:
        ```python
        spec = operable.get("optional_field")
        if spec is not Unset:  # Field exists
            process(spec)
        ```

        Alternative (raise KeyError) rejected: Forces try/except for optional fields.
        Alternative (return None) rejected: Ambiguous, can't distinguish "missing" from "null".
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        result = operable.get("field2")
        assert result is Unset

    def test_get_with_default(self):
        """
        get() accepts custom default value for missing fields

        **Pattern**: Dictionary-like get with default

        **Scenario**: Provide fallback value when field missing
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        result = operable.get("field2", default="custom_default")  # "custom_default"
        ```

        **Expected Behavior**:
        - Returns provided default if field not found
        - Overrides `Unset` sentinel behavior
        - Default can be any value (not just sentinel types)

        **Design Rationale**:
        Custom defaults enable graceful degradation:
        ```python
        # With custom default (clean)
        spec = operable.get("optional_field", default=fallback_spec)
        process(spec)

        # Without custom default (verbose)
        spec = operable.get("optional_field")
        if spec is Unset:
            spec = fallback_spec
        process(spec)
        ```

        Matches Python dict API: `dict.get(key, default)` → Familiar pattern.
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        result = operable.get("field2", default="custom_default")
        assert result == "custom_default"

    def test_get_specs_no_filter(self):
        """
        get_specs() without filters returns all specs (in order)

        **Pattern**: Retrieve entire collection

        **Scenario**: Get all specs from Operable for iteration/processing
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))
        specs = operable.get_specs()  # (spec1, spec2)
        ```

        **Expected Behavior**:
        - Returns tuple containing all specs
        - Order preserved (same order as construction)
        - Identity preserved (same objects, not copies)
        - Returns empty tuple for empty Operable

        **Design Rationale**:
        Default behavior (no filtering) provides direct access to collection.
        Tuple return type (not list) signals immutability:
        - Can't accidentally modify collection via returned value
        - Matches `__op_fields__` internal storage type
        - Hashable (useful for caching)

        Use case: Iterate over all fields for model generation, serialization.
        """
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))
        specs = operable.get_specs()
        assert specs == (spec1, spec2)

    def test_get_specs_include(self):
        """
        get_specs() with include filters to specified fields

        **Pattern**: Partial collection retrieval (whitelist)

        **Scenario**: Generate partial model with subset of fields
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))
        specs = operable.get_specs(include={"field1", "field3"})  # (spec1, spec3)
        ```

        **Expected Behavior**:
        - Returns only specs with names in `include` set
        - Order preserved (relative to original order)
        - Validates all names in `include` exist (raises on unknown)
        - Returns tuple (immutable)

        **Design Rationale**:
        Include filtering enables **public vs. internal field separation**:
        ```python
        # Full model (all fields)
        FullUser = operable.create_model()

        # Public API model (exclude sensitive fields)
        specs = operable.get_specs(include={"username", "email", "created_at"})
        PublicUser = adapter.create_model(Operable(specs))
        ```

        Alternative (lazy filtering in adapter) rejected:
        - Violates single responsibility (filtering is collection concern)
        - Reduces testability (can't test filtering independently)
        - Duplicates logic across adapters

        **Use Cases**:
        1. API response models (public fields only)
        2. Form models (user-editable fields only)
        3. Database queries (SELECT specific columns)
        """
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))
        specs = operable.get_specs(include={"field1", "field3"})
        assert len(specs) == 2
        assert spec1 in specs
        assert spec3 in specs
        assert spec2 not in specs

    def test_get_specs_exclude(self):
        """
        get_specs() with exclude filters out specified fields

        **Pattern**: Partial collection retrieval (blacklist)

        **Scenario**: Generate model excluding specific fields
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))
        specs = operable.get_specs(exclude={"field2"})  # (spec1, spec3)
        ```

        **Expected Behavior**:
        - Returns specs **not** in `exclude` set
        - Order preserved (relative to original order)
        - Validates all names in `exclude` exist (raises on unknown)
        - Returns tuple (immutable)

        **Design Rationale**:
        Exclude filtering enables **sensitive field removal**:
        ```python
        # Internal model (all fields including password)
        InternalUser = operable.create_model()

        # Public model (exclude sensitive fields)
        specs = operable.get_specs(exclude={"password", "api_key", "internal_id"})
        PublicUser = adapter.create_model(Operable(specs))
        ```

        **Include vs. Exclude**:
        - **Use include when**: Subset is small, explicit whitelist desired
        - **Use exclude when**: Most fields needed, few fields sensitive

        Alternative (include inverse set) considered:
        ```python
        # Verbose: include = all_fields - exclude_fields
        include = operable.allowed() - {"password", "api_key"}
        specs = operable.get_specs(include=include)
        ```
        Rejected: Verbose, error-prone (easy to forget to update when fields added).

        **Use Cases**:
        1. Public API responses (exclude internal/sensitive fields)
        2. Logging (exclude PII fields)
        3. Caching (exclude computed/volatile fields)
        """
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))
        specs = operable.get_specs(exclude={"field2"})
        assert len(specs) == 2
        assert spec1 in specs
        assert spec3 in specs
        assert spec2 not in specs

    def test_get_specs_both_include_exclude_raises(self):
        """
        get_specs() raises ValueError when both include and exclude specified

        **Pattern**: Mutual exclusivity validation

        **Scenario**: Attempt to use both include and exclude (ambiguous)
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        operable.get_specs(include={"field1"}, exclude={"field2"})  # ValueError
        ```

        **Expected Behavior**:
        - Raises `ValueError` with clear message
        - Error message: "Cannot specify both include and exclude"
        - Prevents ambiguous filtering logic

        **Design Rationale**:
        Combining include + exclude creates ambiguous semantics:
        ```python
        # Ambiguous: What does this mean?
        specs = operable.get_specs(include={"a", "b"}, exclude={"b", "c"})
        # Is "b" included or excluded? Unclear!
        ```

        Enforcing mutual exclusivity ensures clarity:
        - **Include**: Explicit whitelist (only these fields)
        - **Exclude**: Explicit blacklist (all except these)
        - **Both**: Ambiguous (forbidden)

        Alternative (precedence rule) rejected:
        - "include takes precedence" or "exclude takes precedence" is arbitrary
        - Hides user mistakes (wanted one, typed both)
        - Better to fail explicitly than guess intent

        **Implementation Note**:
        Check performed at argument validation (before any filtering).
        Fail fast principle: Error before any computation.
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        with pytest.raises(ValueError, match="Cannot specify both"):
            operable.get_specs(include={"field1"}, exclude={"field2"})

    def test_get_specs_include_invalid_raises(self):
        """
        get_specs() raises ValueError when include contains invalid field names

        **Pattern**: Typo protection via early validation

        **Scenario**: Attempt to include non-existent field
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        operable.get_specs(include={"field1", "invalid_field"})  # ValueError
        ```

        **Expected Behavior**:
        - Raises `ValueError` before filtering
        - Error message identifies invalid field name
        - Error message lists allowed fields (for debugging)

        **Design Rationale**:
        Strict validation prevents silent failures:
        ```python
        # Typo in include set
        specs = operable.get_specs(include={"usernme", "email"})  # "usernme" typo
        # Without validation: Silently returns {"email"} (missing "username")
        # With validation: ValueError("usernme not allowed. Did you mean 'username'?")
        ```

        Early validation provides:
        1. **Clear errors**: Typo identified immediately
        2. **Fast feedback**: Fail before model generation
        3. **Correct behavior**: No silent data loss

        Alternative (silent ignore) rejected:
        - Hides bugs (typos go unnoticed)
        - Data loss (expected fields missing from model)
        - Debugging nightmare (why is field missing?)

        **Production Impact**:
        In production, this prevents:
        - Accidentally exposing fields (typo in exclude)
        - Missing required fields (typo in include)
        - Silent API contract changes
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        with pytest.raises(ValueError, match="not allowed"):
            operable.get_specs(include={"field1", "invalid_field"})

    def test_field_ordering_preserved(self):
        """
        Field ordering is preserved from construction (insertion order)

        **Pattern**: Deterministic field order (critical for serialization)

        **Scenario**: Create Operable with specific field order
        ```python
        specs = [
            Spec(str, name="field_a"),
            Spec(int, name="field_b"),
            Spec(bool, name="field_c"),
        ]
        operable = Operable(tuple(specs))
        # Order: field_a, field_b, field_c
        ```

        **Expected Behavior**:
        - Field order matches construction order
        - `__op_fields__` is tuple (preserves insertion order)
        - Order maintained through `get_specs()` and model generation
        - No alphabetic sorting, no arbitrary reordering

        **Design Rationale**:
        Field order matters for:

        1. **Serialization**: JSON key order, CSV column order
           ```python
           # Serialized as: {"field_a": ..., "field_b": ..., "field_c": ...}
           # Order matches schema definition
           ```

        2. **UI Generation**: Form field order, table column order
           ```python
           # Forms render: field_a input, field_b input, field_c input
           # Order matches schema definition (not alphabetic)
           ```

        3. **API Stability**: Field order in responses is part of contract
           ```python
           # API v1: {"name": ..., "age": ...}
           # API v2: {"age": ..., "name": ...}  # Breaking change!
           ```

        4. **Human Readability**: Schema definition order is intentional
           ```python
           # Logical grouping: identity fields, then metadata, then computed
           # Preserving order maintains design intent
           ```

        Alternative (unordered dict) rejected:
        - Python 3.7+ dicts preserve insertion order (de facto standard)
        - Tuple enforces immutability + order preservation
        - No alphabetic sorting (breaks logical grouping)

        **Historical Note**:
        This addresses Issue #1: Early implementation used dict (pre-3.7 style),
        causing non-deterministic field order. Tuple ensures consistency.
        """
        specs = [
            Spec(str, name="field_a"),
            Spec(int, name="field_b"),
            Spec(bool, name="field_c"),
        ]
        operable = Operable(tuple(specs))

        # Verify order preserved
        field_names = [s.name for s in operable.__op_fields__]
        assert field_names == ["field_a", "field_b", "field_c"]

        # Verify it's a tuple
        assert isinstance(operable.__op_fields__, tuple)

    def test_create_model_pydantic_import_error(self):
        """
        create_model() raises helpful ImportError when Pydantic unavailable

        **Pattern**: Graceful dependency failure with actionable error

        **Scenario**: Request Pydantic adapter without Pydantic installed
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        operable.create_model(adapter="pydantic")  # ImportError
        ```

        **Expected Behavior**:
        - Raises `ImportError` (not generic exception)
        - Error message: "PydanticSpecAdapter requires Pydantic"
        - Error message includes install instructions (actionable)
        - Original ImportError preserved in chain (debugging)

        **Design Rationale**:
        Better error messages improve developer experience:
        ```python
        # Without enhancement (cryptic)
        ImportError: No module named 'pydantic'
        # Where did this come from? What do I need to install?

        # With enhancement (clear)
        ImportError: PydanticSpecAdapter requires Pydantic. Install with: pip install pydantic
        # Clear: Install pydantic, retry
        ```

        **Implementation Pattern**:
        Catch ImportError during adapter import, re-raise with context:
        ```python
        try:
            from .spec_adapters.pydantic_field import PydanticSpecAdapter
        except ImportError as e:
            raise ImportError(
                "PydanticSpecAdapter requires Pydantic. Install with: pip install pydantic"
            ) from e
        ```

        Alternative (silent ignore) rejected: User has no idea what went wrong.
        Alternative (assert at module import) rejected: Breaks import even if adapter not used.
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))

        # Mock the import to fail
        import sys

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "pydantic_field" in name:
                raise ImportError("Mocked: pydantic not available")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="PydanticSpecAdapter requires Pydantic"),
        ):
            operable.create_model(adapter="pydantic")

    def test_create_model_unsupported_adapter(self):
        """
        create_model() raises ValueError for unsupported adapter names

        **Pattern**: Explicit adapter validation (fail fast)

        **Scenario**: Request adapter that doesn't exist
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        operable.create_model(adapter="unsupported")  # ValueError
        ```

        **Expected Behavior**:
        - Raises `ValueError` (not ImportError, not AttributeError)
        - Error message: "Unsupported adapter: unsupported"
        - Error message lists supported adapters (e.g., "pydantic")
        - Fail before attempting import/instantiation

        **Design Rationale**:
        Validate adapter name before importing:
        1. **Faster errors**: No import overhead for invalid names
        2. **Clearer errors**: "Unsupported" vs "Import failed" (different causes)
        3. **Future-proof**: Easy to add new adapters to validation list

        **Supported Adapters**:
        - `"pydantic"`: Pydantic v2 (PydanticSpecAdapter)
        - Future: `"attrs"`, `"dataclass"`, `"msgspec"`

        Alternative (dynamic import + catch error) rejected:
        ```python
        # Dynamic import (slow, unclear errors)
        try:
            adapter_module = importlib.import_module(f"adapters.{name}")
        except ImportError:
            # Is it: typo in name, or missing dependency?
        ```

        Current approach (explicit validation) advantages:
        - Clear error messages (typo vs. dependency)
        - Faster failures (no import attempts)
        - Explicit supported adapter list (self-documenting)
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        with pytest.raises(ValueError, match="Unsupported adapter"):
            operable.create_model(adapter="unsupported")  # type: ignore

    def test_immutability(self):
        """
        Operable is immutable (frozen dataclass semantics)

        **Pattern**: Immutability enforcement via frozen dataclass

        **Scenario**: Attempt to modify Operable after construction
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        operable.name = "new_name"  # FrozenInstanceError
        ```

        **Expected Behavior**:
        - Raises `FrozenInstanceError` (Python 3.11+) or `AttributeError` (earlier)
        - Cannot modify `name`, `__op_fields__`, or any attribute
        - Tuple storage prevents field mutation

        **Design Rationale**:
        Immutability provides:

        1. **Thread Safety**: Share Operable across threads without locks
           ```python
           # Safe: Multiple threads read same Operable
           def worker():
               model = shared_operable.create_model()
           ```

        2. **Caching**: Hash Operable for cache keys (future optimization)
           ```python
           @lru_cache
           def get_model(operable: Operable):
               return operable.create_model()
           ```

        3. **Predictability**: No action-at-a-distance bugs
           ```python
           # Can't accidentally modify shared operable
           func_a(operable)  # Reads
           func_b(operable)  # Reads
           # Both see same data (no mutation)
           ```

        4. **Functional Updates**: Explicit copy-on-write pattern
           ```python
           # Want changes? Create new Operable
           new_operable = Operable(operable.__op_fields__ + (new_spec,), name=operable.name)
           ```

        Alternative (mutable) rejected:
        - Requires defensive copies (performance cost)
        - Thread-safety requires locks (complexity, performance)
        - Action-at-a-distance bugs (hard to debug)

        **Implementation**: `@dataclass(frozen=True)` enforces immutability.
        """
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))
        with pytest.raises((AttributeError, TypeError)):  # Frozen dataclass errors
            operable.name = "new_name"

    def test_type_validation(self):
        """
        Operable rejects non-Spec objects (type safety)

        **Pattern**: Early type validation (fail fast)

        **Scenario**: Attempt to create Operable with invalid types
        ```python
        Operable(("not_a_spec",))  # TypeError
        Operable(([1, 2, 3],))  # TypeError
        Operable((None,))  # TypeError
        ```

        **Expected Behavior**:
        - Raises `TypeError` at construction
        - Error message: "All specs must be Spec objects"
        - Validation happens before storing in `__op_fields__`

        **Design Rationale**:
        Type validation prevents runtime errors later:
        ```python
        # Without validation
        operable = Operable(("not_a_spec",))  # Succeeds (bad!)
        specs = operable.get_specs()
        for spec in specs:
            spec.name  # AttributeError: str has no 'name'

        # With validation
        operable = Operable(("not_a_spec",))  # TypeError (good!)
        # Error at construction, clear message
        ```

        Early validation advantages:
        1. **Clear errors**: "Not a Spec" vs "No attribute 'name'" (unclear)
        2. **Fail fast**: Error at construction, not later during use
        3. **Type safety**: Guarantees all items are Spec instances

        Alternative (duck typing) rejected:
        ```python
        # Duck typing: "If it has .name, it's fine"
        for spec in specs:
            if hasattr(spec, "name"):
                use(spec)
        ```
        Rejected because:
        - Silent failures (missing attributes discovered late)
        - No type hints (IDE can't help)
        - Weak contract (any object with .name accepted)

        **Implementation**: `isinstance(spec, Spec)` check for all items.
        """
        with pytest.raises(TypeError, match="All specs must be Spec objects"):
            Operable(("not_a_spec",))

    def test_duplicate_name_detection(self):
        """
        Operable rejects duplicate field names (uniqueness enforcement)

        **Pattern**: Early validation (fail fast on duplicates)

        **Scenario**: Attempt to create Operable with duplicate field names
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field1")  # Duplicate name
        Operable((spec1, spec2))  # ValueError
        ```

        **Expected Behavior**:
        - Raises `ValueError` at construction
        - Error message: "Duplicate field names found: field1"
        - Lists all duplicate names (if multiple)
        - Validation before storing in `__op_fields__`

        **Design Rationale**:
        Name uniqueness is critical for:

        1. **Model Generation**: Field names become model attributes
           ```python
           class User:
               field1: str  # First occurrence
               field1: int  # Syntax error! Can't redefine
           ```

        2. **Field Access**: `get()` must be deterministic
           ```python
           spec = operable.get("field1")
           # Which spec? First or second? Ambiguous!
           ```

        3. **Serialization**: Dict keys must be unique
           ```python
           {"field1": "value1", "field1": "value2"}
           # Which value? Last-write-wins, but which is "last"?
           ```

        4. **API Contracts**: Field names are public API
           ```python
           # API schema: {"field1": "string", "field1": "integer"}
           # Invalid JSON Schema! Tools reject it.
           ```

        Alternative (last-write-wins) rejected:
        ```python
        # Silently use last spec with duplicate name
        operable = Operable((spec1, spec2))  # Accepts, uses spec2
        ```
        Rejected because:
        - Hides bugs (typos go unnoticed)
        - Unclear semantics (which one is used?)
        - Silent data loss (first spec ignored)

        Alternative (namespace fields) rejected:
        ```python
        # Add suffix: field1, field1_2, field1_3
        ```
        Rejected because:
        - Breaks user expectations (wanted "field1", got "field1_2")
        - Naming conflict (what if user has field1_2?)
        - Hides the real problem (duplicate definition)

        **Validation Algorithm**: O(n) using set to track seen names.
        """
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field1")  # Duplicate name
        with pytest.raises(ValueError, match="Duplicate field names found"):
            Operable((spec1, spec2))

    def test_creation_with_list(self):
        """
        Operable accepts list and converts to tuple (convenience + immutability)

        **Pattern**: Flexible API with immutable storage

        **Scenario**: Create Operable using list (not tuple)
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable([spec1, spec2], name="TestModel")  # List, not tuple
        ```

        **Expected Behavior**:
        - Accepts both `list` and `tuple` input
        - Internally stores as tuple (`__op_fields__` is tuple)
        - No mutation possible (even if original list mutated)
        - Name attribute preserved

        **Design Rationale**:
        Flexible input + immutable storage provides best of both worlds:

        **API Flexibility**:
        ```python
        # Users can pass list (natural for building specs dynamically)
        specs = []
        specs.append(Spec(str, name="field1"))
        specs.append(Spec(int, name="field2"))
        operable = Operable(specs)  # Accepts list

        # Or tuple (explicit immutability)
        operable = Operable((spec1, spec2))  # Accepts tuple
        ```

        **Internal Immutability**:
        ```python
        # Original list mutation doesn't affect Operable
        specs = [spec1, spec2]
        operable = Operable(specs)
        specs.append(spec3)  # Mutate original list
        # operable.__op_fields__ still has only 2 specs (copied to tuple)
        ```

        Alternative (require tuple) rejected:
        ```python
        operable = Operable(tuple([spec1, spec2]))  # Verbose!
        ```
        Rejected because:
        - Verbose API (forces users to convert)
        - Less Pythonic (most Python APIs accept list or tuple)
        - No benefit (conversion is cheap, O(n) one-time)

        Alternative (store as list) rejected:
        - Breaks immutability (internal mutation possible)
        - Can't hash (future caching impossible)
        - Inconsistent with frozen dataclass semantics

        **Pattern**: Accept flexible input, normalize to canonical form.
        Similar to: `dict(...)` accepts many input formats but stores as dict.
        """
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        # Pass list instead of tuple
        operable = Operable([spec1, spec2], name="TestModel")
        assert len(operable.__op_fields__) == 2
        assert isinstance(operable.__op_fields__, tuple)  # Should be converted to tuple
        assert operable.name == "TestModel"

    def test_create_model_pydantic_success(self):
        """
        create_model() generates Pydantic model from Operable (integration test)

        **Pattern**: 3-layer architecture integration (Operable → Adapter → Model)

        **Scenario**: Generate Pydantic BaseModel from Operable specs
        ```python
        spec1 = Spec(str, name="field1", default="default_value")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2), name="TestModel")
        model = operable.create_model(adapter="pydantic", model_name="GeneratedModel")
        ```

        **Expected Behavior**:
        - Returns Pydantic model class (subclass of `BaseModel`)
        - Model has correct name ("GeneratedModel")
        - Model fields match Operable specs (field1: str, field2: int)
        - Can instantiate model with validation

        **Design Rationale**:
        This test validates the **3-layer architecture integration**:

        **Layer 1 (Spec)**: Define field schema (framework-agnostic)
        ```python
        spec1 = Spec(str, name="field1", default="default_value")
        # Pure specification, no Pydantic knowledge
        ```

        **Layer 2 (Operable)**: Collect specs, validate uniqueness
        ```python
        operable = Operable((spec1, spec2))
        # Collection semantics, no Pydantic knowledge
        ```

        **Layer 3 (Adapter)**: Generate framework-specific model
        ```python
        model = operable.create_model(adapter="pydantic")
        # Pydantic-specific generation logic
        # Converts Spec metadata → Pydantic FieldInfo
        # Creates model using create_model() from Pydantic
        ```

        **Integration Points**:
        1. Operable delegates to PydanticSpecAdapter
        2. Adapter converts each Spec → Pydantic FieldInfo
        3. Adapter calls Pydantic's `create_model()`
        4. Returns BaseModel subclass

        **Why Delegate?**
        - Operable has **zero Pydantic knowledge** (framework-agnostic)
        - Adapter encapsulates **all Pydantic logic** (single responsibility)
        - Can add new adapters (attrs, msgspec) **without modifying Operable**

        Alternative (direct Pydantic in Operable) rejected:
        ```python
        # Bad: Tight coupling
        class Operable:
            def create_pydantic_model(self):
                from pydantic import create_model  # Tight coupling!
        ```
        Rejected because:
        - Breaks framework independence
        - Can't support multiple frameworks
        - Requires Pydantic even if not used

        **Test Scope**: Integration only (validates delegation works).
        Full Pydantic adapter tests in `test_adapters_py311.py`.
        """

        spec1 = Spec(str, name="field1", default="default_value")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2), name="TestModel")

        # Create model with pydantic adapter
        model = operable.create_model(adapter="pydantic", model_name="GeneratedModel")

        # Verify model was created
        assert model is not None
        assert model.__name__ == "GeneratedModel"

    def test_create_model_pydantic_with_filters(self):
        """
        create_model() respects include/exclude filters (partial model generation)

        **Pattern**: Multiple views from single schema (DRY principle)

        **Scenario**: Generate different models from same Operable using filters
        ```python
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3), name="TestModel")

        # Public model (include subset)
        PublicModel = operable.create_model(
            adapter="pydantic", model_name="PublicModel", include={"field1", "field3"}
        )

        # Internal model (exclude sensitive)
        InternalModel = operable.create_model(
            adapter="pydantic", model_name="InternalModel", exclude={"field2"}
        )
        ```

        **Expected Behavior**:
        - `PublicModel` has only field1, field3 (not field2)
        - `InternalModel` has field1, field3 (not field2)
        - Both models validate correctly
        - Filters validated by `get_specs()` (raises on invalid names)

        **Design Rationale**:
        Filtering enables **multiple views without duplication**:

        **Use Case 1: Public vs. Internal APIs**
        ```python
        # Schema: All fields
        user_operable = Operable(
            [
                Spec(str, name="username"),
                Spec(str, name="email"),
                Spec(str, name="password"),  # Sensitive
                Spec(str, name="internal_id"),  # Internal
            ]
        )

        # Public API: Exclude sensitive/internal
        PublicUser = user_operable.create_model(
            adapter="pydantic", exclude={"password", "internal_id"}
        )

        # Admin API: All fields
        AdminUser = user_operable.create_model(adapter="pydantic")
        ```

        **Use Case 2: Request vs. Response Models**
        ```python
        # Schema: All fields
        post_operable = Operable(
            [
                Spec(str, name="id"),  # Generated server-side
                Spec(str, name="title"),
                Spec(str, name="content"),
                Spec(datetime, name="created_at"),  # Generated server-side
            ]
        )

        # POST request model: User provides title, content only
        CreatePost = post_operable.create_model(adapter="pydantic", include={"title", "content"})

        # GET response model: All fields
        Post = post_operable.create_model(adapter="pydantic")
        ```

        **Benefits**:
        1. **Single Source of Truth**: One Operable, multiple models
        2. **Type Safety**: Filters validated (typos caught early)
        3. **Maintainability**: Add field once, appears in all relevant models
        4. **No Duplication**: Don't repeat field definitions

        Alternative (separate Operables) rejected:
        ```python
        # Duplication (bad)
        public_operable = Operable([username_spec, email_spec])
        admin_operable = Operable([username_spec, email_spec, password_spec, id_spec])
        ```
        Rejected because:
        - Duplication (DRY violation)
        - Easy to get out of sync (add field to one, forget other)
        - More code to maintain

        **Integration**: Operable.create_model() calls `get_specs(include/exclude)`
        before delegating to adapter. Validation happens at Operable layer.
        """

        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3), name="TestModel")

        # Test with include
        model_include = operable.create_model(
            adapter="pydantic", model_name="IncludeModel", include={"field1", "field3"}
        )
        assert model_include is not None

        # Test with exclude
        model_exclude = operable.create_model(
            adapter="pydantic", model_name="ExcludeModel", exclude={"field2"}
        )
        assert model_exclude is not None

    def test_create_model_pydantic_default_name(self):
        """
        create_model() uses Operable name as default model name (convenience)

        **Pattern**: Sensible defaults with override capability

        **Scenario**: Generate model without explicit model_name
        ```python
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,), name="OperableName")

        # No model_name provided
        model = operable.create_model(adapter="pydantic")
        # Uses operable.name: model.__name__ == "OperableName"
        ```

        **Expected Behavior**:
        - If `model_name` not provided → Use `operable.name`
        - If `operable.name` is None → Use "DynamicModel" (fallback)
        - If `model_name` provided → Use `model_name` (explicit override)

        **Naming Hierarchy**:
        1. **Explicit**: `model_name` parameter (highest priority)
        2. **Implicit**: `operable.name` attribute (middle priority)
        3. **Fallback**: "DynamicModel" literal (lowest priority)

        **Design Rationale**:
        Default naming reduces boilerplate for common case:

        **Common Case (simple naming)**:
        ```python
        # Without default (verbose)
        user_operable = Operable([...], name="User")
        UserModel = user_operable.create_model(
            adapter="pydantic",
            model_name="User",  # Duplicate name!
        )

        # With default (clean)
        user_operable = Operable([...], name="User")
        UserModel = user_operable.create_model(adapter="pydantic")
        # model.__name__ == "User" (from operable.name)
        ```

        **Explicit Override (when needed)**:
        ```python
        # Same Operable, different model names
        base_operable = Operable([...], name="Base")

        PublicModel = base_operable.create_model(
            adapter="pydantic",
            model_name="Public",  # Override
        )

        InternalModel = base_operable.create_model(
            adapter="pydantic",
            model_name="Internal",  # Override
        )
        ```

        **Fallback (no name set)**:
        ```python
        # Anonymous Operable (no name)
        operable = Operable([spec1, spec2])  # name=None
        model = operable.create_model(adapter="pydantic")
        # model.__name__ == "DynamicModel" (fallback)
        ```

        **Why "DynamicModel" fallback?**
        - Better than error (allows anonymous Operables)
        - Better than empty string (valid Python identifier required)
        - Better than UUID (predictable, debuggable)
        - Signals: "This model was dynamically generated without name"

        Alternative (require model_name) rejected:
        ```python
        operable.create_model(adapter="pydantic")  # ValueError: model_name required
        ```
        Rejected because:
        - Verbose (forces specifying name even when operable.name exists)
        - Breaks DRY (name specified twice: Operable + create_model)
        - Less Pythonic (most APIs have sensible defaults)

        **Implementation**: Check `model_name` param → `operable.name` → "DynamicModel".
        """

        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,), name="OperableName")

        # Don't provide model_name, should use operable.name
        model = operable.create_model(adapter="pydantic")
        assert model is not None
        assert model.__name__ == "OperableName"

        # Test without operable.name - should use "DynamicModel"
        operable_no_name = Operable((spec1,))
        model_dynamic = operable_no_name.create_model(adapter="pydantic")
        assert model_dynamic is not None
        assert model_dynamic.__name__ == "DynamicModel"

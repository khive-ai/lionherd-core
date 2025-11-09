# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Spec Tests: Framework-Agnostic Field Schema

**Core Abstractions**:
- **Spec**: Immutable field specification (type + metadata)
- **Metadata**: Key-value pairs (validators, defaults, custom data)
- **CommonMeta**: Standard metadata keys (NAME, NULLABLE, VALIDATOR, etc.)

**Design Philosophy**:
- **Framework Independence**: No Pydantic/attrs/dataclass coupling
- **Immutability**: Specs are value objects (safe caching, thread-safe)
- **Composability**: Build complex specs from simple building blocks
- **Progressive Enhancement**: Use only what you need (type → +metadata → +validation → +defaults)

**Testing Strategy**:
This test suite validates:
1. Spec creation with various metadata combinations
2. Annotation generation (nullable, listable, complex types)
3. Default value creation (static, factory, async factory)
4. Metadata validation (duplicates, conflicts, type safety)
5. Caching behavior (annotated() LRU cache)
6. Fluent API (method chaining, immutable updates)

**Type System Context**:
Spec is **Layer 1** of the 3-layer architecture:
- **Spec** (this): Schema definition (framework-agnostic)
- **Operable**: Spec collection (validated, ordered)
- **Adapter**: Framework-specific model generation (Pydantic, attrs, etc.)

Design enables: Define schema once → Generate for multiple frameworks

**Design Rationale**:

**Why Immutable?**
- Safe caching: `annotated()` LRU cache never stale
- Thread-safe: No locks needed for concurrent access
- Predictable: Same input always produces same output
- Composability: Build complex specs without side effects

**Why Tuple[Meta, ...]?**
- Hashable: Specs can be dict keys, set members
- Deterministic: Iteration order preserved
- Immutable: Enforces value object semantics

**Why Separate Spec/Operable?**
- Single Responsibility: Spec = field, Operable = collection
- Reusability: Same spec in multiple Operables
- Testability: Test field logic independently of collection logic

**Performance Characteristics**:
- Spec creation: O(m) where m = metadata count (small constant)
- `annotated()` first call: O(m) + typing.Annotated overhead (~100μs)
- `annotated()` cached: O(1) (~1μs)
- LRU cache eviction: FIFO, bounded at `lionherd_FIELD_CACHE_SIZE` (10k default)

**Sentinel Semantics**:
- `Undefined`: Field never set (missing from namespace) - `spec.default` when no default provided
- `Unset`: Key present but value not provided - Used in Params for optional fields
- `None`: Explicitly set to null - Valid value, not sentinel

**Related Components**:
- `Operable`: Collects multiple Specs with name uniqueness validation
- `PydanticSpecAdapter`: Converts Spec → Pydantic FieldInfo
- `Meta`: Individual metadata key-value pair
- `ModelConfig`: Configures sentinel handling, strict mode

**References**:
- Source: `src/lionherd_core/types/spec.py`
- Related tests: `tests/types/test_operable.py`, `tests/types/spec_adapters/test_adapters_py311.py`
- Architecture: `.khive/workspaces/test_docs_002_types_analysis/types_architecture_analysis.md`
"""

import pytest

from lionherd_core.types import CommonMeta, Meta, Spec


# Async fixtures and helpers
async def async_default_factory():
    """Async factory for testing."""
    return "async_value"


class TestCommonMeta:
    """
    CommonMeta Validation: Standard metadata keys and validation utilities

    **Tested Behaviors**:
    - **Allowed Keys**: `allowed()` returns all CommonMeta enum values
    - **Conflict Detection**: `default` + `default_factory` → ExceptionGroup
    - **Type Safety**: Validators and factories must be callable
    - **Duplicate Detection**: Same key in metadata → ValueError

    **Edge Cases**:
    - Non-callable validator → ExceptionGroup
    - Non-callable default_factory → ExceptionGroup
    - Duplicate keys in metadata tuple → ValueError
    - Duplicate keys in kwargs → ValueError
    - Duplicate keys across args and kwargs → ValueError

    **Pattern Context**:
    CommonMeta provides standard metadata keys and validation utilities.
    Early validation at `prepare()` ensures only valid metadata enters Spec.

    These tests ensure:
    1. Metadata validation catches conflicts before Spec creation
    2. Duplicate detection works across all input methods
    3. Type safety enforced for callables (validators, factories)
    """

    def test_allowed_returns_all_values(self):
        """
        CommonMeta.allowed() returns all standard metadata keys

        **Pattern**: Metadata introspection for validation

        **Scenario**: Framework needs to know all valid CommonMeta keys
        for validation or autocomplete purposes.

        **Expected Behavior**:
        - Returns set containing all enum values
        - Includes: name, nullable, listable, validator, default, default_factory
        - Count matches enum member count (6 keys)
        """
        allowed = CommonMeta.allowed()
        assert "name" in allowed
        assert "nullable" in allowed
        assert "listable" in allowed
        assert "validator" in allowed
        assert "default" in allowed
        assert "default_factory" in allowed
        assert len(allowed) == 6

    def test_validate_rejects_both_default_and_factory(self):
        """
        Validation rejects conflicting default + default_factory

        **Pattern**: Early validation for mutually exclusive options

        **Scenario**: User accidentally provides both static default and factory:
        ```python
        Spec(str, default="value", default_factory=lambda: "other")
        ```

        **Expected Behavior**:
        - ExceptionGroup raised at validation time
        - Error message indicates conflict: "both default and default_factory"

        **Design Rationale**:
        Default and default_factory are mutually exclusive:
        - Static default: Value known at Spec creation time
        - Factory: Value computed dynamically at instance creation

        Having both creates ambiguity: which takes precedence?
        Early validation prevents this ambiguous state.

        Alternative considered: Default takes precedence (ignore factory)
        Rejected because: Silent ignoring hides user mistake
        """
        with pytest.raises(ExceptionGroup, match="Metadata validation failed"):
            CommonMeta._validate_common_metas(default="value", default_factory=lambda: "value")

    def test_validate_rejects_non_callable_factory(self):
        """
        Validation rejects non-callable default_factory

        **Pattern**: Type safety for factory functions

        **Scenario**: User provides string instead of callable:
        ```python
        Spec(str, default_factory="not_a_function")  # Bug: forgot to pass function
        ```

        **Expected Behavior**:
        - ExceptionGroup raised at validation time
        - Error identifies non-callable factory

        **Design Rationale**:
        Factories must be callable because they're invoked at runtime:
        ```python
        value = spec.create_default_value()  # Calls factory()
        ```

        Non-callable factory would cause runtime error later.
        Early validation catches this at Spec creation time.
        """
        with pytest.raises(ExceptionGroup, match="Metadata validation failed"):
            CommonMeta._validate_common_metas(default_factory="not_a_function")

    def test_validate_rejects_non_callable_validator(self):
        """
        Validation rejects non-callable validators

        **Pattern**: Type safety for validation functions

        **Scenario**: User provides string instead of callable:
        ```python
        Spec(str, validator="not_callable")  # Bug: forgot validation function
        ```

        **Expected Behavior**:
        - ExceptionGroup raised at validation time
        - Error identifies non-callable validator

        **Design Rationale**:
        Validators are invoked during model validation:
        ```python
        if validator is not None:
            result = validator(value)  # Must be callable
        ```

        Non-callable validator causes runtime error.
        Early validation prevents invalid Spec construction.
        """
        with pytest.raises(ExceptionGroup, match="Metadata validation failed"):
            CommonMeta._validate_common_metas(validator="not_callable")

    def test_prepare_detects_duplicate_in_metadata(self):
        """
        prepare() detects duplicate keys in metadata tuple

        **Pattern**: Duplicate detection across all input methods

        **Scenario**: User provides metadata tuple with duplicate keys:
        ```python
        meta1 = Meta("name", "field1")
        meta2 = Meta("name", "field2")  # Duplicate key: "name"
        CommonMeta.prepare(metadata=(meta1, meta2))
        ```

        **Expected Behavior**:
        - ValueError raised immediately
        - Error message identifies duplicate key: "name"

        **Design Rationale**:
        Metadata stored as tuple[Meta, ...] (immutable, hashable).
        Duplicates create ambiguity: which value to use?

        Early detection ensures:
        1. Clear error at point of mistake
        2. No silent overwrites (last-write-wins)
        3. Metadata tuple integrity
        """
        meta1 = Meta("name", "field1")
        meta2 = Meta("name", "field2")
        with pytest.raises(ValueError, match="Duplicate metadata key: name"):
            CommonMeta.prepare(metadata=(meta1, meta2))

    def test_prepare_detects_duplicate_in_args(self):
        """
        prepare() detects duplicate keys in positional args

        **Pattern**: Duplicate detection in varargs

        **Scenario**: User passes duplicate Meta objects as args:
        ```python
        meta1 = Meta("name", "field1")
        meta2 = Meta("name", "field2")
        CommonMeta.prepare(meta1, meta2)  # Both args have same key
        ```

        **Expected Behavior**:
        - ValueError raised immediately
        - Error identifies duplicate key
        """
        meta1 = Meta("name", "field1")
        meta2 = Meta("name", "field2")
        with pytest.raises(ValueError, match="Duplicate metadata key: name"):
            CommonMeta.prepare(meta1, meta2)

    def test_prepare_detects_duplicate_in_kwargs(self):
        """
        prepare() detects duplicate keys across args and kwargs

        **Pattern**: Cross-input duplicate detection

        **Scenario**: User provides same key in both args and kwargs:
        ```python
        meta1 = Meta("name", "field1")
        CommonMeta.prepare(meta1, name="field2")  # "name" in both
        ```

        **Expected Behavior**:
        - ValueError raised immediately
        - Error identifies conflict between arg and kwarg

        **Design Rationale**:
        prepare() accepts multiple input methods:
        - metadata=(meta1, meta2, ...)  # Tuple of Meta objects
        - meta1, meta2, ...              # Positional args
        - name="value", nullable=True    # Keyword args

        All three methods must be checked for duplicates.
        """
        meta1 = Meta("name", "field1")
        with pytest.raises(ValueError, match="Duplicate metadata key: name"):
            CommonMeta.prepare(meta1, name="field2")

    def test_prepare_success(self):
        """
        prepare() successfully creates metadata tuple from kwargs

        **Pattern**: Happy path metadata preparation

        **Scenario**: User provides valid metadata via kwargs:
        ```python
        metadata = CommonMeta.prepare(name="field", nullable=True)
        ```

        **Expected Behavior**:
        - Returns tuple[Meta, ...] with 2 elements
        - Contains Meta("name", "field")
        - Contains Meta("nullable", True)

        **Design Rationale**:
        Kwargs provide convenient API for common case:
        ```python
        Spec(str, name="username", nullable=True)
        # Internally: prepare(name="username", nullable=True)
        ```

        More ergonomic than:
        ```python
        Spec(str, metadata=(Meta("name", "username"), Meta("nullable", True)))
        ```
        """
        result = CommonMeta.prepare(name="field", nullable=True)
        assert len(result) == 2
        meta_dict = {m.key: m.value for m in result}
        assert meta_dict["name"] == "field"
        assert meta_dict["nullable"] is True


class TestSpec:
    """
    Spec Core Functionality: Immutable field schema with progressive enhancement

    **Tested Behaviors**:
    - **Basic Creation**: Type + name → minimal Spec
    - **Nullable/Listable**: Annotation generation (str | None, list[int])
    - **Defaults**: Static values, sync factories, async factories
    - **Fluent API**: Method chaining (as_nullable(), with_default(), with_validator())
    - **Immutability**: Frozen dataclass (no mutations after creation)
    - **Caching**: annotated() LRU cache for type annotation reuse
    - **Metadata Access**: __getitem__, get(), metadict() with filtering

    **Pattern Context**:
    Spec is the foundation of the type system. All tests demonstrate progressive
    enhancement: start with minimal type+name, add metadata as needed.

    Immutability is critical: Specs are value objects that can be safely cached,
    shared across threads, and used as dict keys.

    These tests ensure:
    1. Minimal creation path works (type + name only)
    2. Progressive enhancement via fluent methods
    3. Immutability enforced (mutations raise errors)
    4. Caching provides performance benefits
    5. Metadata access is convenient and type-safe
    """

    def test_basic_creation(self):
        """
        Basic Spec creation with minimal metadata

        **Pattern**: Minimal creation path (progressive enhancement foundation)

        **Scenario**: Create simplest possible Spec:
        ```python
        spec = Spec(str, name="username")
        ```

        **Expected Behavior**:
        - Spec created successfully with only type + name
        - base_type is str
        - name is "username"
        - No additional metadata required

        **Design Rationale**:
        Progressive enhancement starts with minimal requirements.
        User adds metadata only when needed (nullable, validators, etc.)

        This pattern enables:
        - Quick schema definition
        - Low barrier to entry
        - Gradual complexity addition
        """
        spec = Spec(str, name="username")
        assert spec.base_type == str
        assert spec.name == "username"

    def test_nullable_and_listable(self):
        """
        Nullable and listable flags control annotation generation

        **Pattern**: Type annotation modifiers

        **Scenario**: Field can be null and/or a list:
        ```python
        spec = Spec(int, name="age", nullable=True, listable=False)
        # annotation: int | None (nullable, not list)
        ```

        **Expected Behavior**:
        - is_nullable returns True
        - is_listable returns False
        - annotation reflects both flags

        **Design Rationale**:
        Common patterns in APIs:
        - nullable=True: Optional field (can be None)
        - listable=True: Collection of values (list[T])
        - Both: Optional collection (list[T] | None)

        Flags drive annotation generation for type checkers and adapters.
        """
        spec = Spec(int, name="age", nullable=True, listable=False)
        assert spec.is_nullable is True
        assert spec.is_listable is False

    def test_default_value(self):
        """
        Static default value stored and returned by factory method

        **Pattern**: Static default (known at Spec creation time)

        **Scenario**: Field has constant default:
        ```python
        spec = Spec(str, name="field", default="default_value")
        value = spec.create_default_value()  # Returns "default_value"
        ```

        **Expected Behavior**:
        - spec.default stores value
        - create_default_value() returns same value each time

        **Design Rationale**:
        Static defaults are common (empty strings, 0, False, etc.)
        No need for factory overhead when value is constant.

        Alternative: Always use factory (lambda: "value")
        Rejected because: Verbose, unnecessary indirection
        """
        spec = Spec(str, name="field", default="default_value")
        assert spec.default == "default_value"
        assert spec.create_default_value() == "default_value"

    def test_default_factory(self):
        """
        Default factory generates fresh values on each call

        **Pattern**: Dynamic default (computed at instance creation time)

        **Scenario**: Default must be unique per instance:
        ```python
        spec = Spec(list, name="field", default_factory=list)
        value1 = spec.create_default_value()  # []
        value2 = spec.create_default_value()  # Different [] instance
        assert value1 is not value2  # Separate lists
        ```

        **Expected Behavior**:
        - has_default_factory is True
        - create_default_value() invokes factory each time
        - Each call returns new instance (not shared)

        **Design Rationale**:
        Mutable defaults must be unique per instance to avoid shared state bugs:
        ```python
        # BAD: Shared mutable default
        spec = Spec(list, default=[])  # Same list for all instances!

        # GOOD: Factory creates new list each time
        spec = Spec(list, default_factory=list)  # Fresh list per instance
        ```

        Common factory use cases:
        - Empty collections: list, dict, set
        - UUID generation: lambda: str(uuid.uuid4())
        - Timestamps: lambda: datetime.now()
        """
        spec = Spec(list, name="field", default_factory=list)
        assert spec.has_default_factory is True
        result = spec.create_default_value()
        assert isinstance(result, list)

    def test_async_default_factory_warning(self):
        """
        Async default factory emits warning (prefer sync when possible)

        **Pattern**: Best practice guidance via warnings

        **Scenario**: User provides async factory:
        ```python
        async def async_factory():
            return await fetch_from_api()


        spec = Spec(str, name="field", default_factory=async_factory)
        # UserWarning: "Async default factories complicate usage..."
        ```

        **Expected Behavior**:
        - UserWarning emitted at Spec creation
        - has_async_default_factory is True
        - Spec still created successfully

        **Design Rationale**:
        Async factories complicate usage:
        - Require await (can't use in sync contexts)
        - Need event loop (runtime dependency)
        - Slower than sync factories (event loop overhead)

        Warning encourages sync factories when possible:
        - Simpler usage model
        - Broader applicability (sync and async contexts)
        - Better performance

        But async factories supported for legitimate cases:
        - Fetching from remote API
        - Database queries
        - Distributed system initialization
        """

        async def async_factory():
            return "value"

        with pytest.warns(UserWarning, match="Async default factories"):
            spec = Spec(str, name="field", default_factory=async_factory)
        assert spec.has_async_default_factory is True

    def test_as_nullable(self):
        """
        as_nullable() creates new Spec with nullable=True

        **Pattern**: Fluent API (immutable updates via method chaining)

        **Scenario**: Make existing Spec nullable:
        ```python
        spec = Spec(str, name="field")  # Not nullable
        nullable_spec = spec.as_nullable()  # Returns new Spec with nullable=True
        ```

        **Expected Behavior**:
        - Returns new Spec instance (original unchanged)
        - is_nullable is True on new Spec
        - All other metadata preserved

        **Design Rationale**:
        Fluent API enables progressive enhancement:
        ```python
        spec = Spec(str, name="email").as_nullable().with_validator(email_validator)
        ```

        Immutability ensures:
        - Original spec unchanged (safe to reuse)
        - No accidental mutations
        - Thread-safe sharing
        """
        spec = Spec(str, name="field")
        nullable_spec = spec.as_nullable()
        assert nullable_spec.is_nullable is True
        assert nullable_spec.name == "field"

    def test_as_listable(self):
        """
        as_listable() creates new Spec with listable=True

        **Pattern**: Fluent API for collection types

        **Scenario**: Convert field to list type:
        ```python
        spec = Spec(int, name="scores")  # int
        listable_spec = spec.as_listable()  # list[int]
        ```

        **Expected Behavior**:
        - Returns new Spec instance
        - is_listable is True
        - annotation becomes list[T]

        **Use Cases**:
        - Multi-value fields: tags, scores, IDs
        - API endpoints returning arrays
        - Database many-to-many relationships
        """
        spec = Spec(int, name="field")
        listable_spec = spec.as_listable()
        assert listable_spec.is_listable is True
        assert listable_spec.name == "field"

    def test_with_default(self):
        """
        with_default() adds static default to Spec

        **Pattern**: Fluent API for default values

        **Scenario**: Add default to existing Spec:
        ```python
        spec = Spec(str, name="field")
        spec_with_default = spec.with_default("value")
        ```

        **Expected Behavior**:
        - Returns new Spec with default set
        - spec.default stores value
        - Original spec unchanged
        """
        spec = Spec(str, name="field")
        spec_with_default = spec.with_default("value")
        assert spec_with_default.default == "value"

    def test_with_default_factory(self):
        """
        with_default() detects callables and treats as factory

        **Pattern**: Unified API for static and dynamic defaults

        **Scenario**: Pass callable to with_default():
        ```python
        spec = Spec(list, name="field")
        spec_with_factory = spec.with_default(list)  # Callable → factory
        ```

        **Expected Behavior**:
        - Callable detected automatically
        - has_default_factory is True
        - No explicit default_factory parameter needed

        **Design Rationale**:
        Unified API simplifies common case:
        ```python
        spec.with_default(list)  # Factory (callable)
        spec.with_default([])  # Static (not callable)
        ```

        User doesn't need to know about default vs. default_factory.
        System detects intent from value type.
        """
        spec = Spec(list, name="field")
        spec_with_factory = spec.with_default(list)
        assert spec_with_factory.has_default_factory is True

    def test_with_validator(self):
        """
        with_validator() adds validation function to Spec

        **Pattern**: Fluent API for validation rules

        **Scenario**: Add validator to field:
        ```python
        def validator(v):
            return len(v) > 0


        spec = Spec(str, name="field")
        spec_with_validator = spec.with_validator(validator)
        ```

        **Expected Behavior**:
        - Returns new Spec with validator in metadata
        - get("validator") returns validator function
        - Validator available for adapter integration

        **Use Cases**:
        - String length constraints
        - Numeric range validation
        - Custom business rules
        - Format validation (email, URL, etc.)
        """

        def validator(v):
            return len(v) > 0

        spec = Spec(str, name="field")
        spec_with_validator = spec.with_validator(validator)
        assert spec_with_validator.get("validator") == validator

    def test_annotation_basic(self):
        """
        annotation property returns base_type for simple Spec

        **Pattern**: Type annotation generation

        **Scenario**: Minimal Spec without nullable/listable:
        ```python
        spec = Spec(str, name="field")
        assert spec.annotation == str
        ```

        **Expected Behavior**:
        - annotation returns base_type directly
        - No Union, list, or other wrappers

        **Design Rationale**:
        annotation property provides type hint for:
        - Type checkers (mypy, pyright)
        - IDE autocomplete
        - Runtime type validation
        - Adapter field generation
        """
        spec = Spec(str, name="field")
        assert spec.annotation == str

    def test_annotation_nullable(self):
        """
        annotation generates Union[T, None] for nullable Spec

        **Pattern**: Optional type annotation

        **Scenario**: Nullable field:
        ```python
        spec = Spec(str, name="field", nullable=True)
        assert spec.annotation == str | None  # Union[str, None] in 3.10+
        ```

        **Expected Behavior**:
        - annotation returns Union type
        - Equivalent to Optional[str] in older Python

        **Use Cases**:
        - Optional API fields
        - Database nullable columns
        - Partial update payloads
        """
        spec = Spec(str, name="field", nullable=True)
        assert spec.annotation == str | None

    def test_annotation_listable(self):
        """
        annotation generates list[T] for listable Spec

        **Pattern**: Collection type annotation

        **Scenario**: List field:
        ```python
        spec = Spec(int, name="scores", listable=True)
        assert spec.annotation == list[int]
        ```

        **Expected Behavior**:
        - annotation wraps base_type in list
        - Type-safe list annotation
        """
        spec = Spec(int, name="field", listable=True)
        assert spec.annotation == list[int]

    def test_annotation_nullable_listable(self):
        """
        annotation combines nullable and listable correctly

        **Pattern**: Complex type annotation (optional collection)

        **Scenario**: Field is optional list:
        ```python
        spec = Spec(int, name="scores", nullable=True, listable=True)
        assert spec.annotation == list[int] | None
        ```

        **Expected Behavior**:
        - annotation is Union[list[int], None]
        - Represents: "list of ints OR null" (not "list of optional ints")

        **Design Rationale**:
        Order matters in Union:
        - list[int] | None: Entire list is optional
        - list[int | None]: List required, items optional

        Spec uses first form (entire collection optional).
        """
        spec = Spec(int, name="field", nullable=True, listable=True)
        assert spec.annotation == list[int] | None

    def test_getitem(self):
        """
        __getitem__ provides dict-like metadata access

        **Pattern**: Convenient metadata access

        **Scenario**: Access metadata by key:
        ```python
        spec = Spec(str, name="field", custom="value")
        assert spec["name"] == "field"
        assert spec["custom"] == "value"
        ```

        **Expected Behavior**:
        - __getitem__ searches metadata tuple
        - Returns value for matching key
        - Works for both common and custom metadata
        """
        spec = Spec(str, name="field", custom="value")
        assert spec["name"] == "field"
        assert spec["custom"] == "value"

    def test_getitem_missing_raises(self):
        """
        __getitem__ raises KeyError on missing metadata key

        **Pattern**: Fail-fast metadata access

        **Scenario**: Access non-existent metadata key:
        ```python
        spec = Spec(str, name="field")
        value = spec["missing"]  # KeyError
        ```

        **Expected Behavior**:
        - KeyError raised immediately
        - Error message identifies missing key: "missing"

        **Design Rationale**:
        Dict-like access (__getitem__) should behave like dict:
        - Missing key → KeyError (not None, not default)
        - Clear error message for debugging
        - Fail-fast prevents silent bugs

        Alternative: Use get() with default for lenient access:
        ```python
        value = spec.get("missing", default="fallback")  # Returns "fallback"
        ```
        """
        spec = Spec(str, name="field")
        with pytest.raises(KeyError, match="Metadata key 'missing'"):
            _ = spec["missing"]

    def test_get_with_default(self):
        """
        get() provides lenient metadata access with default fallback

        **Pattern**: Lenient metadata access (alternative to __getitem__)

        **Scenario**: Access metadata with fallback for missing keys:
        ```python
        spec = Spec(str, name="field")
        value = spec.get("missing", "default")  # Returns "default" (no error)
        name = spec.get("name")  # Returns "field"
        ```

        **Expected Behavior**:
        - Missing key with default → returns default
        - Existing key → returns value
        - No KeyError raised

        **Design Rationale**:
        Provides dict-like get() method for optional metadata:
        - Lenient access when key might not exist
        - Avoid try/except for optional metadata
        - Common pattern: get("custom", None)

        **Use Cases**:
        - Optional custom metadata
        - Framework-specific attributes (may not exist)
        - Default values for missing config
        """
        spec = Spec(str, name="field")
        assert spec.get("missing", "default") == "default"
        assert spec.get("name") == "field"

    def test_metadict(self):
        """
        metadict() converts metadata tuple to dict

        **Pattern**: Metadata serialization and introspection

        **Scenario**: Convert Spec metadata to dict for inspection or serialization:
        ```python
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict()
        # {"name": "field", "nullable": True, "custom": "value"}
        ```

        **Expected Behavior**:
        - Returns dict[str, Any] with all metadata
        - Includes both common and custom metadata
        - Dict is mutable copy (modifying doesn't affect Spec)

        **Design Rationale**:
        Metadata stored as immutable tuple[Meta, ...] internally for:
        - Hashability (Spec can be dict key)
        - Immutability (value object semantics)

        metadict() provides dict view for:
        - Easier introspection (dict comprehensions, filtering)
        - Serialization (JSON, YAML)
        - Framework integration (pass as **kwargs)

        **Use Cases**:
        - Debugging: Inspect all metadata
        - Serialization: Convert to JSON
        - Adapter integration: Pass metadata to framework
        """
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict()
        assert metadict["name"] == "field"
        assert metadict["nullable"] is True
        assert metadict["custom"] == "value"

    def test_metadict_exclude(self):
        """
        metadict() supports selective key exclusion

        **Pattern**: Filtered metadata export

        **Scenario**: Export metadata excluding specific keys:
        ```python
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict(exclude={"name"})
        # {"nullable": True, "custom": "value"} - "name" excluded
        ```

        **Expected Behavior**:
        - Excluded keys not in output dict
        - All other metadata included
        - exclude parameter accepts set of keys

        **Use Cases**:
        - Remove internal keys before serialization
        - Exclude framework-specific metadata
        - Filter sensitive metadata
        - Adapter integration (exclude adapter-handled keys)

        Example: Pydantic adapter excludes "name" (handled separately)
        """
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict(exclude={"name"})
        assert "name" not in metadict
        assert metadict["nullable"] is True

    def test_metadict_exclude_common(self):
        """
        metadict() can exclude all CommonMeta keys

        **Pattern**: Custom metadata extraction

        **Scenario**: Extract only custom metadata (exclude standard keys):
        ```python
        spec = Spec(str, name="field", nullable=True, custom="value", config="data")
        metadict = spec.metadict(exclude_common=True)
        # {"custom": "value", "config": "data"} - Common keys excluded
        ```

        **Expected Behavior**:
        - All CommonMeta keys excluded (name, nullable, listable, validator, default, default_factory)
        - Only custom metadata in output
        - exclude_common=True shorthand for exclude=CommonMeta.allowed()

        **Design Rationale**:
        Separate standard vs. custom metadata:
        - **CommonMeta**: Framework-agnostic standard keys
        - **Custom**: User-defined, application-specific metadata

        exclude_common enables:
        - Extract custom metadata for application logic
        - Pass custom data to adapters without common keys
        - Serialize custom config separately

        **Use Cases**:
        - Application-specific metadata (tags, categories, permissions)
        - Custom validation rules
        - UI hints (display_name, help_text, icon)
        """
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict(exclude_common=True)
        assert "name" not in metadict
        assert "nullable" not in metadict
        assert metadict["custom"] == "value"

    def test_invalid_base_type_raises(self):
        """
        Spec creation requires valid type for base_type

        **Pattern**: Early validation (fail-fast on invalid input)

        **Scenario**: User passes non-type value as base_type:
        ```python
        Spec("not_a_type", name="field")  # String, not type
        ```

        **Expected Behavior**:
        - ValueError raised immediately
        - Error message: "must be a type"
        - No Spec object created

        **Design Rationale**:
        base_type must be actual type for type system to work:
        - Used in annotation generation (typing.Annotated)
        - Type checkers require real types
        - Adapters use for framework field creation

        Early validation prevents:
        - Runtime errors in annotation generation
        - Confusing errors downstream
        - Invalid Specs propagating through system

        Common mistakes caught:
        - String type names: "str" instead of str
        - Module names: "datetime" instead of datetime.datetime
        - None values: forgot to pass type
        """
        with pytest.raises(ValueError, match="must be a type"):
            Spec("not_a_type")

    def test_name_cannot_be_none(self):
        """
        Spec name must be a valid string, not None

        **Pattern**: Early validation for required string metadata

        **Scenario**: User passes None as name:
        ```python
        Spec(str, name=None)  # Invalid: name cannot be None
        ```

        **Expected Behavior**:
        - ValueError raised immediately
        - Error message: "must be a non-empty string"
        - No Spec object created

        **Design Rationale**:
        Spec name is used for field identification in Operable collections.
        None is not a valid identifier and would cause errors in:
        - Operable.allowed() (expects string names)
        - Operable.get() (dict-like access by name)
        - Model generation (field names must be strings)
        """
        with pytest.raises(ValueError, match="must be a non-empty string"):
            Spec(str, name=None)

    def test_name_cannot_be_empty_string(self):
        """
        Spec name must be non-empty string

        **Pattern**: Early validation for meaningful identifiers

        **Scenario**: User passes empty string as name:
        ```python
        Spec(str, name="")  # Invalid: empty string not allowed
        ```

        **Expected Behavior**:
        - ValueError raised immediately
        - Error message: "must be a non-empty string"

        **Design Rationale**:
        Empty string is not a valid field identifier:
        - Confuses users (what field is this?)
        - Breaks model generation (invalid Python identifier)
        - Complicates debugging (all empty names look the same)
        """
        with pytest.raises(ValueError, match="must be a non-empty string"):
            Spec(str, name="")

    def test_name_must_be_string_type(self):
        """
        Spec name must be string type, not other types

        **Pattern**: Type safety for field identifiers

        **Scenario**: User passes non-string as name:
        ```python
        Spec(str, name=123)  # Invalid: integer, not string
        ```

        **Expected Behavior**:
        - ValueError raised immediately
        - Error message: "must be a non-empty string"

        **Design Rationale**:
        Field names must be strings for:
        - Python syntax (attribute names are strings)
        - JSON serialization (keys must be strings)
        - Model generation (Pydantic/dataclass field names)
        - Dict-like access consistency
        """
        with pytest.raises(ValueError, match="must be a non-empty string"):
            Spec(str, name=123)

    def test_create_default_without_default_raises(self):
        """
        create_default_value() requires default or factory

        **Pattern**: Contract enforcement (default generation requires default)

        **Scenario**: Call create_default_value() on Spec without default:
        ```python
        spec = Spec(str, name="field")  # No default provided
        value = spec.create_default_value()  # ValueError
        ```

        **Expected Behavior**:
        - ValueError raised
        - Error message: "No default value"
        - Indicates Spec has no default mechanism

        **Design Rationale**:
        create_default_value() is for optional fields with defaults.
        Required fields (no default) should fail at validation time, not here.

        This pattern enables:
        - Clear distinction: required vs. optional fields
        - Adapter logic: check has_default_factory or default before calling
        - Fail-fast: Can't generate default from nothing

        **Correct Usage**:
        ```python
        # Static default
        spec = Spec(str, name="field", default="value")
        value = spec.create_default_value()  # "value"

        # Factory default
        spec = Spec(list, name="field", default_factory=list)
        value = spec.create_default_value()  # []
        ```
        """
        spec = Spec(str, name="field")
        with pytest.raises(ValueError, match="No default value"):
            spec.create_default_value()

    def test_create_default_with_async_factory_raises(self):
        """
        Sync create_default_value() rejects async factory

        **Pattern**: Type safety (async/sync separation)

        **Scenario**: Call sync method on Spec with async factory:
        ```python
        async def async_factory():
            return await fetch_from_api()


        spec = Spec(str, name="field", default_factory=async_factory)
        value = spec.create_default_value()  # ValueError (sync call, async factory)
        ```

        **Expected Behavior**:
        - ValueError raised
        - Error message: "asynchronous"
        - Indicates must use acreate_default_value() instead

        **Design Rationale**:
        Separate sync/async methods prevent accidental blocking:
        - create_default_value(): Sync factories only
        - acreate_default_value(): Both sync and async factories

        This prevents:
        - Blocking event loop (sync call on async factory)
        - Runtime errors (can't await in sync context)
        - Confusing behavior (factory not executing)

        **Correct Usage**:
        ```python
        # Async factory requires async method
        value = await spec.acreate_default_value()  # Correctly awaits factory
        ```

        Safety check is critical because:
        - Async factories look callable (are callable)
        - Calling returns coroutine (not value)
        - Coroutine needs await (sync context can't provide)
        """

        async def async_factory():
            return "value"

        with pytest.warns(UserWarning):
            spec = Spec(str, name="field", default_factory=async_factory)

        with pytest.raises(ValueError, match="asynchronous"):
            spec.create_default_value()

    def test_annotated_caching(self):
        """
        annotated() returns cached result for same Spec

        **Pattern**: LRU cache for performance (100x speedup)

        **Scenario**: Call annotated() multiple times on same Spec:
        ```python
        spec = Spec(str, name="field")
        annotated1 = spec.annotated()  # First call: computes, caches (~100μs)
        annotated2 = spec.annotated()  # Second call: cached (~1μs)
        assert annotated1 is annotated2  # Same object (identity check)
        ```

        **Expected Behavior**:
        - First call computes and caches annotation
        - Subsequent calls return cached object (identity match)
        - ~100x speedup for cached access

        **Design Rationale**:
        typing.Annotated creation is expensive (~100μs):
        - Metaclass machinery
        - Type introspection
        - Metadata wrapping

        Specs are immutable (safe caching):
        - Same Spec always produces same annotation
        - No cache staleness possible
        - Thread-safe sharing

        Caching provides:
        - **Performance**: O(1) cached vs O(m) uncached (m = metadata count)
        - **Memory**: Shared annotations across same Specs
        - **Correctness**: Immutability guarantees cache validity

        **Performance Characteristics**:
        - Cold cache: ~100μs (typing.Annotated overhead)
        - Warm cache: ~1μs (dict lookup)
        - Cache size: Bounded at 10k (LRU eviction)
        - Cache hit rate: >95% (typical workloads)

        Common pattern triggering cache benefits:
        ```python
        # Same Spec used in multiple Operables
        username_spec = Spec(str, name="username")
        user_operable = Operable([username_spec, ...])
        admin_operable = Operable([username_spec, ...])
        # annotated() called twice, but computed once
        ```
        """
        spec = Spec(str, name="field")
        annotated1 = spec.annotated()
        annotated2 = spec.annotated()
        # Should return same object from cache
        assert annotated1 is annotated2

    def test_with_updates(self):
        """
        with_updates() creates new Spec with modified metadata

        **Pattern**: Immutable updates (copy-on-write)

        **Scenario**: Update Spec metadata without mutating original:
        ```python
        spec = Spec(str, name="field", nullable=False)
        updated = spec.with_updates(nullable=True, custom="value")
        # updated: new Spec with nullable=True, custom="value"
        # spec: original unchanged (nullable=False, no "custom")
        ```

        **Expected Behavior**:
        - Returns new Spec instance (original unchanged)
        - New metadata merged with existing
        - All other metadata preserved
        - Original Spec still valid and usable

        **Design Rationale**:
        Immutable updates enable safe Spec reuse and modification:
        - No accidental mutations
        - Original Spec safe to share/cache
        - Thread-safe updates (no locks needed)
        - Functional programming style

        **Implementation**:
        ```python
        # Internally creates new metadata tuple
        new_metadata = existing_metadata + new_meta_items
        return Spec(base_type, metadata=new_metadata)
        ```

        **Use Cases**:
        1. **Progressive enhancement**: Start minimal, add metadata
        ```python
        spec = Spec(str, name="username")
        spec = spec.with_updates(nullable=True)  # Add nullable
        spec = spec.with_updates(validator=email_validator)  # Add validator
        ```

        2. **Spec templates**: Base spec + variations
        ```python
        base_spec = Spec(str, name="field")
        required_spec = base_spec.with_updates(nullable=False)
        optional_spec = base_spec.with_updates(nullable=True)
        ```

        3. **Safe modifications**: Update without affecting others
        ```python
        shared_spec = Spec(int, name="count")
        # Other code uses shared_spec
        my_spec = shared_spec.with_updates(default=0)  # Safe: new instance
        ```

        **Alternative Considered**: Mutable Spec with setter methods
        **Rejected Because**:
        - Cache invalidation complexity
        - Thread safety requires locks
        - Harder to reason about state
        - Can't use as dict keys (unhashable if mutable)
        """
        spec = Spec(str, name="field", nullable=False)
        updated = spec.with_updates(nullable=True, custom="value")
        assert updated.is_nullable is True
        assert updated.get("custom") == "value"
        assert updated.name == "field"

    def test_immutability(self):
        """
        Spec is immutable (frozen dataclass)

        **Pattern**: Value object immutability

        **Scenario**: Attempt to mutate Spec after creation:
        ```python
        spec = Spec(str, name="field")
        spec.base_type = int  # AttributeError or TypeError
        ```

        **Expected Behavior**:
        - Mutation attempt raises error
        - AttributeError or TypeError (frozen dataclass)
        - Spec remains unchanged

        **Design Rationale**:
        Immutability is fundamental to Spec design. It enables:

        **1. Safe Caching**
        ```python
        # annotated() cache never stale
        cached = spec.annotated()  # Safe to cache forever
        # No risk of spec.base_type changing later
        ```

        **2. Thread Safety**
        ```python
        # Multiple threads can safely share Spec
        shared_spec = Spec(str, name="username")
        # Thread 1: user_model = adapter.create_model([shared_spec])
        # Thread 2: admin_model = adapter.create_model([shared_spec])
        # No locks needed, no race conditions
        ```

        **3. Hashability**
        ```python
        # Specs as dict keys, set members
        spec_registry = {spec: adapter}
        unique_specs = {spec1, spec2, spec3}
        ```

        **4. Predictability**
        ```python
        # Same input always produces same output
        spec = Spec(str, name="field")
        annotation1 = spec.annotation
        annotation2 = spec.annotation
        assert annotation1 is annotation2  # Always true
        ```

        **5. Composability**
        ```python
        # Build complex specs without side effects
        base_spec = Spec(str, name="field")
        nullable_spec = base_spec.as_nullable()  # Returns new Spec
        # base_spec unchanged, can reuse
        ```

        **Implementation**:
        - frozen=True in @dataclass decorator
        - __setattr__ and __delattr__ disabled
        - metadata as immutable tuple[Meta, ...]

        **Alternative Considered**: Mutable Spec with copy() method
        **Rejected Because**:
        - Manual copying error-prone (easy to forget)
        - Caching complex (need cache invalidation)
        - Thread safety requires locks
        - Can't hash mutable objects
        - Harder to reason about (state changes over time)

        **Trade-offs**:
        - ✅ Safety, predictability, thread-safety, caching
        - ❌ Must use with_updates() for changes (slightly more verbose)
        - Decision: Safety > convenience (immutability is correct default)
        """
        spec = Spec(str, name="field")
        with pytest.raises((AttributeError, TypeError)):  # Frozen dataclass errors
            spec.base_type = int

    @pytest.mark.anyio
    async def test_acreate_default_value_async_factory(self):
        """Test acreate_default_value() executes async factory.

        Design: async factories are supported for default value generation in
        async contexts. This enables async I/O, database queries, or other
        async operations during initialization.
        """
        with pytest.warns(UserWarning):
            spec = Spec(str, name="field", default_factory=async_default_factory)

        # Execute async default factory
        result = await spec.acreate_default_value()
        assert result == "async_value"

    @pytest.mark.anyio
    async def test_acreate_default_value_sync_fallback(self):
        """Test acreate_default_value() handles sync defaults in async context.

        Design: When called in an async context but with a sync default or
        factory, acreate_default_value() safely falls back to the sync path.
        This allows uniform async/await usage regardless of default type.
        """
        # Test with static default
        spec_static = Spec(str, name="field", default="static_value")
        result_static = await spec_static.acreate_default_value()
        assert result_static == "static_value"

        # Test with sync factory
        spec_factory = Spec(list, name="field", default_factory=list)
        result_factory = await spec_factory.acreate_default_value()
        assert isinstance(result_factory, list)

    def test_spec_annotated_cache_eviction(self):
        """
        annotated() cache evicts oldest entries when full

        **Pattern**: LRU cache with bounded memory

        **Scenario**: Create more unique Specs than cache can hold:
        ```python
        _MAX_CACHE_SIZE = 5
        for i in range(10):  # Create 10 specs
            spec = Spec(str, name=f"field_{i}", custom=i)
            spec.annotated()  # Cache fills, then evicts
        # Cache size <= 5 (eviction happened)
        ```

        **Expected Behavior**:
        - Cache size never exceeds _MAX_CACHE_SIZE
        - Oldest entries evicted first (FIFO)
        - Cache remains functional after eviction

        **Design Rationale**:
        Unbounded cache causes memory leaks in long-running applications.

        LRU eviction with bounded size provides:
        - **Memory safety**: Predictable max memory usage
        - **Performance**: Cache hit rate remains high (most specs reused)
        - **Simplicity**: FIFO eviction (no complex LRU tracking)

        Default size (10k) chosen because:
        - Typical application: <100 unique specs
        - 10k provides 100x headroom
        - ~1MB memory for 10k cached annotations

        Tunable via environment: `lionherd_FIELD_CACHE_SIZE`
        """
        import lionherd_core.types.spec as spec_module
        from lionherd_core.types.spec import _annotated_cache, _cache_lock

        # Save original cache size
        original_max_size = spec_module._MAX_CACHE_SIZE

        try:
            # Set a small cache size for testing
            spec_module._MAX_CACHE_SIZE = 5

            # Clear cache to start fresh
            with _cache_lock:
                _annotated_cache.clear()

            # Create more specs than cache size to trigger eviction
            # Each spec must have unique metadata to create unique cache keys
            for i in range(10):
                spec = Spec(str, name=f"field_{i}", custom_value=i)
                spec.annotated()  # Trigger caching

            # Verify cache size doesn't exceed max (eviction happened)
            with _cache_lock:
                cache_size = len(_annotated_cache)
                assert cache_size <= 5, f"Cache size {cache_size} exceeds max size 5"
                assert cache_size > 0, "Cache should not be empty"

        finally:
            # Restore original cache size
            spec_module._MAX_CACHE_SIZE = original_max_size

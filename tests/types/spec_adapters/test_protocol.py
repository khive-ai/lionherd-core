# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
SpecAdapter Protocol Tests: Framework-Agnostic Adapter Interface

**Core Abstractions**:
- **SpecAdapter[M]**: Generic protocol for framework-specific model generation
- **Abstract Methods**: create_field, create_model, validate_model, dump_model, fuzzy_match_fields
- **Shared Utilities**: parse_json, validate_response, update_model (concrete implementations)
- **Plugin Architecture**: Add new frameworks without modifying Spec/Operable code

**Design Philosophy**:
- **Framework Isolation**: Framework-specific imports only in concrete adapters
- **Generic Type Safety**: `SpecAdapter[M]` where M = model instance type (BaseModel, AttrsClass, etc.)
- **Shared Logic**: Common utilities (JSON parsing, fuzzy matching) in base protocol
- **Hook Pattern**: `create_validator` as optional hook for framework-specific validator creation

**Testing Strategy**:
This test suite validates:
1. Protocol contract (abstract methods must be implemented by concrete adapters)
2. Base class behavior (default implementations, hooks)
3. Hook pattern (`create_validator` returns None by default)
4. Minimal adapter pattern (testing base class without framework dependencies)

**Type System Context**:
SpecAdapter is **Layer 3** of the 3-layer architecture:
- **Spec**: Schema definition (framework-agnostic)
- **Operable**: Spec collection (validated, name-unique)
- **Adapter** (this): Framework-specific model generation

Design enables plugin architecture:
1. Define schema once (Spec/Operable)
2. Generate for Pydantic (PydanticSpecAdapter)
3. Generate for attrs (future: AttrsSpecAdapter)
4. Generate for dataclasses (future: DataclassAdapter)
5. All without changing Spec/Operable code

**Adapter Protocol Contract**:

```python
class SpecAdapter(ABC, Generic[M]):
    # M = Model instance type (BaseModel, AttrsModel, DataClass, etc.)

    @abstractmethod
    def create_field(cls, spec: Spec) -> Any:
        '''Convert Spec → framework field (FieldInfo, Attribute, Field, etc.)'''

    @abstractmethod
    def create_model(cls, operable: Operable, model_name: str, **kwargs) -> type[M]:
        '''Generate model class from Operable'''

    @abstractmethod
    def validate_model(cls, model_cls: type[M], data: dict) -> M:
        '''Validate dict → model instance'''

    @abstractmethod
    def dump_model(cls, instance: M) -> dict[str, Any]:
        '''Dump model → dict'''

    @abstractmethod
    def fuzzy_match_fields(cls, data: dict, model_cls: type[M], strict: bool) -> dict:
        '''Match data keys to model fields (fuzzy matching logic)'''

    # Hook method (optional override)
    def create_validator(cls, spec: Spec) -> Any | None:
        '''Hook for framework-specific validator creation. Default: None'''
```

**Hook Pattern: create_validator**:

**Purpose**: Allow concrete adapters to create framework-specific validators from Spec metadata

**Default Behavior**: Returns None (no validator created)

**Override Examples**:
- **PydanticSpecAdapter**: Returns `field_validator` decorator for Spec.validator
- **AttrsSpecAdapter**: Returns `attr.validators.instance_of()` wrapper
- **DataclassAdapter**: May return None (dataclasses have no built-in validators)

**Why Hook Pattern?**
- Not all frameworks have validators (dataclasses don't)
- Different frameworks have incompatible validator APIs (Pydantic vs attrs)
- Base protocol can't know framework-specific details
- Hook enables optional feature without breaking contract

**Design Rationale**:

**Why Protocol + Generic?**
- Type safety: `PydanticSpecAdapter[BaseModel]` vs `AttrsSpecAdapter[AttrsClass]`
- Contract enforcement: All abstract methods must be implemented
- IDE support: Autocomplete, type checking
- Framework isolation: No Pydantic/attrs imports in base protocol

**Why Minimal Adapter?**
- Test base class behavior without framework dependencies
- Verify hook pattern works (create_validator returns None)
- Establish pattern for testing adapter contract
- No pydantic/attrs imports needed (pure Python)

**Performance Characteristics**:
- Protocol overhead: Zero (compile-time only, no runtime cost)
- Hook invocation: O(1) (single method call)
- Minimal adapter: Instant (no-op implementations)

**Related Components**:
- `PydanticSpecAdapter`: Concrete Pydantic v2 implementation
- `Spec`: Field schema (input to create_field)
- `Operable`: Spec collection (input to create_model)

**References**:
- Source: `src/lionherd_core/types/spec_adapters/_protocol.py`
- Related tests: `tests/types/spec_adapters/test_adapters_py311.py`
- Pydantic adapter: `src/lionherd_core/types/spec_adapters/pydantic_field.py`
"""

from typing import Any

import pytest

pytest.importorskip("pydantic")

from lionherd_core.types import Spec
from lionherd_core.types.spec_adapters._protocol import SpecAdapter


class MinimalAdapter(SpecAdapter):
    """
    Minimal Concrete Adapter: Test fixture for SpecAdapter base class behavior

    **Purpose**: Test base protocol functionality without framework dependencies

    **Implementation**:
    - All abstract methods implemented as no-ops (minimal logic)
    - No framework imports (pure Python)
    - Returns sensible defaults (None, empty dict, pass-through)
    - Used to test hook pattern and base class methods

    **Use Cases**:
    - Test `create_validator` hook returns None by default
    - Test base protocol can be instantiated (all abstract methods implemented)
    - Verify protocol contract without Pydantic/attrs dependencies
    - Establish pattern for adapter testing

    **Pattern Context**:
    Real adapters (PydanticSpecAdapter) have complex implementations.
    MinimalAdapter strips away all complexity to test base class behavior in isolation.
    Analogous to mock objects, but implements full protocol contract.
    """

    @classmethod
    def create_field(cls, spec: Spec) -> Any:
        return None

    @classmethod
    def create_model(cls, operable, model_name: str, **kwargs):
        return type(model_name, (), {})

    @classmethod
    def validate_model(cls, model_cls, data: dict):
        return model_cls()

    @classmethod
    def dump_model(cls, instance):
        return {}

    @classmethod
    def fuzzy_match_fields(cls, data: dict, model_cls, strict: bool = False):
        return data


def test_create_validator_returns_none():
    """
    Base create_validator hook returns None (no validator created by default)

    **Pattern**: Hook pattern for optional framework-specific features

    **Scenario**: Adapter that doesn't support validators calls base implementation
    ```python
    # Minimal adapter with no validator support
    validator = MinimalAdapter.create_validator(spec)
    # Returns None (no validator created)
    ```

    **Expected Behavior**:
    - Base implementation returns None (not an error, just "no validator")
    - Concrete adapters override to return framework-specific validators
    - Enables optional feature without breaking protocol contract

    **Design Rationale**:

    **Why Hook Pattern?**
    - Not all frameworks have validators (e.g., dataclasses have no built-in validation)
    - Validator APIs vary widely across frameworks:
      - Pydantic: `field_validator` decorator
      - attrs: `attr.validators.instance_of()`
      - dataclasses: No standard validator mechanism
    - Base protocol can't provide universal validator logic
    - None signals "no validator" (not an error)

    **Why Return None (not raise NotImplementedError)?**
    - Validators are optional feature (not required by all adapters)
    - Raising forces all adapters to implement (even if no validator support)
    - None is graceful: "this adapter doesn't support validators"
    - Concrete adapters can check `if validator is not None:` before using

    **Pattern Context**:
    Hook pattern enables extensibility without tight coupling:
    1. Base protocol provides hook (default: no-op or None)
    2. Concrete adapters override if feature supported
    3. Callers check return value (None vs. actual validator)
    4. No exceptions, no forced implementation

    **Alternative Considered**: Make `create_validator` abstract (force all adapters to implement)
    **Rejected Because**: Not all frameworks support validators; forcing implementation would lead to dummy implementations or exceptions

    **Performance**: O(1) (single return None)
    """
    # Create a simple Spec
    spec = Spec(int, name="test_field", description="Test field")

    # Call create_validator on the minimal adapter (uses base implementation)
    result = MinimalAdapter.create_validator(spec)

    # Should return None (base implementation)
    assert result is None


def test_create_validator_with_metadata():
    """
    Base create_validator ignores Spec validator metadata (concrete adapters extract it)

    **Pattern**: Hook pattern with metadata extraction left to concrete adapters

    **Scenario**: Spec contains validator metadata, but base implementation doesn't use it
    ```python
    spec = Spec(int, name="age", validator=nonneg_validator)
    # Base implementation ignores validator metadata
    result = MinimalAdapter.create_validator(spec)
    # Returns None (metadata not extracted)
    ```

    **Expected Behavior**:
    - Base implementation returns None (even with validator metadata present)
    - Validator metadata exists in Spec but is not extracted by base class
    - Concrete adapters extract and use validator metadata appropriately
    - Base class establishes contract, not implementation

    **Design Rationale**:

    **Why Base Implementation Ignores Metadata?**
    - Validator metadata format is framework-agnostic (callable functions)
    - Each framework needs different transformation:
      - Pydantic: Wrap in `field_validator` decorator
      - attrs: Wrap in `attr.validators.instance_of()` or custom validator
      - dataclasses: No standard validator mechanism (metadata unused)
    - Base protocol can't know how to transform for all frameworks
    - Metadata extraction is framework-specific concern

    **Separation of Concerns**:
    - **Spec**: Stores validator as generic callable (framework-agnostic)
    - **Base Protocol**: Defines hook contract (returns None by default)
    - **Concrete Adapter**: Extracts metadata and transforms for framework

    **Example (PydanticSpecAdapter)**:
    ```python
    @classmethod
    def create_validator(cls, spec: Spec) -> Any | None:
        # Extract validator from Spec metadata
        validator_func = spec.get_meta_value("validator")
        if validator_func:
            # Transform to Pydantic field_validator
            return field_validator(spec.name, check_fields=False)(validator_func)
        return None
    ```

    **Pattern Context**:
    Base protocol establishes "what" (method signature, return type).
    Concrete adapters implement "how" (metadata extraction, transformation).
    This enables each framework to handle validators in their own way.

    **Alternative Considered**: Base class extracts metadata and passes to abstract method
    **Rejected Because**: Would require base class to know metadata keys (tight coupling); adapters may need different metadata beyond validator

    **Performance**: O(1) (single return None, no metadata traversal)
    """

    def validator_func(value):
        return value

    # Create a Spec with validator
    spec = Spec(
        int,
        name="age",
        description="User age",
        validator=validator_func,
    )

    # Call create_validator on minimal adapter (base implementation)
    result = MinimalAdapter.create_validator(spec)

    # Base implementation returns None
    # Even with validator metadata present
    assert result is None

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

__all__ = (
    "Adaptable",
    "AdapterRegisterable",
    "Allowable",
    "AsyncAdaptable",
    "AsyncAdapterRegisterable",
    "Containable",
    "Deserializable",
    "Hashable",
    "Invocable",
    "Observable",
    "Serializable",
    "implements",
)


@runtime_checkable
class ObservableProto(Protocol):
    """Objects with unique UUID identifier. Check via isinstance()."""

    @property
    def id(self) -> UUID:
        """Unique identifier."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Objects that can be serialized to dict via to_dict()."""

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict. Args: serialization options (mode, format, etc.)."""
        ...


@runtime_checkable
class Deserializable(Protocol):
    """Objects that can be deserialized from dict via from_dict() classmethod."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> Any:
        """Deserialize from dict. Args: data dict, deserialization options."""
        ...


@runtime_checkable
class Adaptable(Protocol):
    """Sync adapter protocol for format conversion (TOML/YAML/JSON/SQL). Use AsyncAdaptable for async."""

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Convert to external format. Args: adapter key, many flag, adapter kwargs."""
        ...

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Create from external format. Args: source object, adapter key, many flag."""
        ...


@runtime_checkable
class AdapterRegisterable(Protocol):
    """Mutable adapter registry. Compose with Adaptable for configurable adapters."""

    @classmethod
    def register_adapter(cls, adapter: Any) -> None:
        """Register adapter for this class."""
        ...


@runtime_checkable
class AsyncAdaptable(Protocol):
    """Async adapter protocol for I/O-bound format conversion (DBs, network, files). Use Adaptable for sync."""

    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Async convert to external format. Args: adapter key, many flag, adapter kwargs."""
        ...

    @classmethod
    async def adapt_from_async(
        cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any
    ) -> Any:
        """Async create from external format. Args: source object, adapter key, many flag."""
        ...


@runtime_checkable
class AsyncAdapterRegisterable(Protocol):
    """Mutable async adapter registry. Compose with AsyncAdaptable for configurable async adapters."""

    @classmethod
    def register_async_adapter(cls, adapter: Any) -> None:
        """Register async adapter for this class."""
        ...


@runtime_checkable
class Containable(Protocol):
    """Objects that support membership testing via 'in' operator (__contains__)."""

    def __contains__(self, item: Any) -> bool:
        """Check if item is in collection (by UUID or instance)."""
        ...


@runtime_checkable
class Invocable(Protocol):
    """Objects that can be invoked/executed via async invoke() method."""

    async def invoke(self) -> Any:
        """Invoke/execute the object. Returns: execution result (any value or None)."""
        ...


@runtime_checkable
class Hashable(Protocol):
    """Objects that can be hashed via __hash__() for use in sets/dicts."""

    def __hash__(self) -> int:
        """Return hash value for object (must be immutable or ID-based)."""
        ...


@runtime_checkable
class Allowable(Protocol):
    """Objects with defined set of allowed values/keys via allowed()."""

    def allowed(self) -> set[str]:
        """Return set of allowed keys/values."""
        ...


Observable = ObservableProto


def implements(*protocols: type):
    """Declare protocol implementations (Rust-like: MUST define in class body).

    CRITICAL SEMANTICS (strictest interpretation):
        @implements() means the class **LITERALLY** implements/overrides the method
        or declares the attribute IN ITS OWN CLASS BODY. Inheritance does NOT count.

        This is Rust-like trait implementation: you must provide the implementation
        in the impl block, not rely on inheritance.

    Rules:
        - Method must be defined in class body (even if it calls super())
        - Property must be declared in class body (cannot inherit from parent)
        - Classmethod must be defined in class body
        - NO inheritance: @implements means "I define this, not my parent"

    Args:
        *protocols: Protocol classes that the decorated class **literally** implements

    Returns:
        Class decorator that stores protocols on cls.__protocols__

    Raises:
        TypeError: If class does not define required protocol members in its class body

    Usage:
        ✓ CORRECT: Literal implementation
        @implements(Serializable, Deserializable)
        class MyClass:
            def to_dict(self, **kwargs): ...      # Defined in this class
            @classmethod
            def from_dict(cls, data, **kwargs): ...  # Defined in this class

        ✗ WRONG: Relying on inheritance
        @implements(Serializable)  # VIOLATION!
        class Child(Parent):  # Parent has to_dict()
            pass  # No to_dict in Child body → not allowed!

        ✓ CORRECT: Explicit override
        @implements(Serializable)
        class Child(Parent):
            def to_dict(self, **kwargs):  # Explicit in Child body
                return super().to_dict(**kwargs)  # Can call parent

    Rationale:
        - Explicit > Implicit (Rust philosophy)
        - Clear ownership: each class declares what it implements
        - No ambiguity about where implementation lives
        - Prevents accidental protocol claims through inheritance
    """

    def decorator(cls):
        # Validate that all protocol members are defined in class body
        for protocol in protocols:
            # Get protocol members from protocol class annotations
            import inspect

            protocol_members = {}
            for name, obj in inspect.getmembers(protocol):
                if name.startswith("_"):
                    continue
                # Include methods, properties, classmethods
                if callable(obj) or isinstance(obj, (property, classmethod)):
                    protocol_members[name] = obj

            # Check each required member is in cls.__dict__ (not inherited)
            for member_name in protocol_members:
                # Check if member is in class body
                # For Pydantic models, also check __annotations__ for fields
                in_class_body = member_name in cls.__dict__

                # For Pydantic models, check if it's a field annotation
                if not in_class_body and hasattr(cls, "__annotations__"):
                    in_class_body = member_name in cls.__annotations__

                if not in_class_body:
                    protocol_name = protocol.__name__
                    raise TypeError(
                        f"{cls.__name__} declares @implements({protocol_name}) but does not "
                        f"define '{member_name}' in its class body (inheritance not allowed)"
                    )

        cls.__protocols__ = protocols
        return cls

    return decorator

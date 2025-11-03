# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import field_serializer, field_validator
from pydapter import (
    Adaptable as PydapterAdaptable,
    AsyncAdaptable as PydapterAsyncAdaptable,
)

from ..protocols import Adaptable, AsyncAdaptable, Deserializable, implements
from .element import Element

NODE_REGISTRY: dict[str, type[Node]] = {}


@implements(Deserializable, Adaptable, AsyncAdaptable)
class Node(Element, PydapterAdaptable, PydapterAsyncAdaptable):
    """Polymorphic node with arbitrary content, embeddings, pydapter integration.

    Attributes:
        content: Arbitrary data (auto-serializes nested Elements)
        embedding: Optional float vector

    Auto-registers subclasses in NODE_REGISTRY for polymorphic deserialization.

    Adapter Registration (Rust-like isolated pattern):
        Base Node has toml/yaml built-in. Subclasses get ISOLATED registries (no inheritance):

        ```python
        from pydapter.adapters import TomlAdapter, YamlAdapter

        # Base Node has toml/yaml
        Node(content="test").adapt_to("toml")  # ✓ Works


        # Subclasses do NOT inherit adapters
        class MyNode(Node):
            pass


        MyNode(content="test").adapt_to("toml")  # ✗ Fails (isolated registry)

        # Must explicitly register on subclass
        MyNode.register_adapter(TomlAdapter)
        MyNode(content="test").adapt_to("toml")  # ✓ Now works
        ```

        This prevents adapter pollution while keeping base Node convenient.
    """

    content: Any = None
    embedding: list[float] | None = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclasses with isolated adapter registries."""
        super().__pydantic_init_subclass__(**kwargs)
        NODE_REGISTRY[cls.__name__] = cls
        NODE_REGISTRY[f"{cls.__module__}.{cls.__name__}"] = cls

        # Force creation of isolated registry for subclass (prevents parent inheritance)
        # This ensures each subclass has its own registry, not inheriting from Node
        if cls is not Node:
            # Access _registry() to trigger creation of isolated registry
            _ = cls._registry()

    @field_serializer("content")
    def _serialize_content(self, value: Any) -> Any:
        return value.to_dict() if isinstance(value, Element) else value

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: Any) -> Any:
        if isinstance(value, dict) and "metadata" in value:
            metadata = value.get("metadata", {})
            lion_class = metadata.get("lion_class")
            if lion_class:
                if lion_class in NODE_REGISTRY or lion_class.split(".")[-1] in NODE_REGISTRY:
                    return Node.from_dict(value)
                return Element.from_dict(value)
        return value

    @field_validator("embedding", mode="before")
    @classmethod
    def _validate_embedding(cls, value: Any) -> list[float] | None:
        """Validate embedding. Accepts list, JSON string, or None. Coerces ints to floats."""
        if value is None:
            return None

        # Coerce JSON string to list (common from DB queries)
        if isinstance(value, str):
            import orjson

            try:
                value = orjson.loads(value)
            except Exception as e:
                raise ValueError(f"Failed to parse embedding JSON string: {e}")

        if not isinstance(value, list):
            raise ValueError("embedding must be a list, JSON string, or None")
        if not value:
            raise ValueError("embedding list cannot be empty")
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("embedding must contain only numeric values")
        return [float(x) for x in value]

    @classmethod
    def from_dict(cls, data: dict[str, Any], meta_key: str | None = None, **kwargs: Any) -> Node:
        """Deserialize with polymorphic type restoration via NODE_REGISTRY.

        Args:
            data: Serialized dict
            meta_key: Restore metadata from this key (db compatibility)
            **kwargs: Passed to model_validate
        """
        # Make a copy to avoid mutating input
        data = data.copy()

        # Restore metadata from custom key if specified
        if meta_key and meta_key in data:
            data["metadata"] = data.pop(meta_key)
        # Backward compatibility: handle legacy "node_metadata" key
        elif "node_metadata" in data and "metadata" not in data:
            data["metadata"] = data.pop("node_metadata")

        # Clean up any remaining node_metadata key to avoid validation errors
        data.pop("node_metadata", None)

        # Extract and remove lion_class from metadata (serialization-only metadata)
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = metadata.copy()
            data["metadata"] = metadata
            lion_class = metadata.pop("lion_class", None)
        else:
            lion_class = None

        if lion_class and lion_class != cls.class_name(full=True):
            target_cls = NODE_REGISTRY.get(lion_class) or NODE_REGISTRY.get(
                lion_class.split(".")[-1]
            )
            if target_cls is not None and target_cls is not cls:
                return target_cls.from_dict(data, **kwargs)

        return cls.model_validate(data, **kwargs)

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Convert to external format via pydapter.

        Args:
            obj_key: Adapter key (e.g., "toml", "yaml"). Must register adapter first!
            many: Adapt multiple instances
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return super().adapt_to(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Node:
        """Create from external format via pydapter (polymorphic).

        Args:
            obj: Source object
            obj_key: Adapter key
            many: Deserialize multiple instances
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "from_dict")
        return super().adapt_from(obj, obj_key=obj_key, many=many, **kwargs)

    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Async convert to external format via pydapter.

        Args:
            obj_key: Adapter key
            many: Adapt multiple instances
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return await super().adapt_to_async(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    async def adapt_from_async(
        cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any
    ) -> Node:
        """Async create from external format via pydapter (polymorphic).

        Args:
            obj: Source object
            obj_key: Adapter key
            many: Deserialize multiple instances
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "from_dict")
        return await super().adapt_from_async(obj, obj_key=obj_key, many=many, **kwargs)


NODE_REGISTRY[Node.__name__] = Node
NODE_REGISTRY[Node.class_name(full=True)] = Node

from pydapter.adapters import TomlAdapter, YamlAdapter

Node.register_adapter(TomlAdapter)  # type: ignore[type-abstract]
Node.register_adapter(YamlAdapter)  # type: ignore[type-abstract]

__all__ = ("NODE_REGISTRY", "Node")

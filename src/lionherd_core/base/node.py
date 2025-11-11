# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, field_serializer, field_validator
from pydantic_core.core_schema import SerializationInfo
from pydapter import (
    Adaptable as PydapterAdaptable,
    AsyncAdaptable as PydapterAsyncAdaptable,
)

from ..protocols import (
    Adaptable,
    AdapterRegisterable,
    AsyncAdaptable,
    AsyncAdapterRegisterable,
    Deserializable,
    Serializable,
    implements,
)
from .element import Element

NODE_REGISTRY: dict[str, type[Node]] = {}


@implements(
    Adaptable,
    AdapterRegisterable,
    AsyncAdaptable,
    AsyncAdapterRegisterable,
    Deserializable,
    Serializable,
)
class Node(Element, PydapterAdaptable, PydapterAsyncAdaptable):
    """Polymorphic container for structured, composable data with embeddings.

    Enables graph-of-graphs, JSONB query-ability, and type-safe composition.
    Auto-registers subclasses for polymorphic deserialization via NODE_REGISTRY.
    Base Node includes toml/yaml adapters; subclasses use isolated registries.
    """

    content: dict[str, Any] | Serializable | BaseModel | None = None
    embedding: list[float] | None = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclass in NODE_REGISTRY with isolated adapter registry."""
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
        """Serialize content using recursive to_dict.

        Uses ln.to_dict which handles:
        - Element instances (to_dict)
        - Pydantic models (model_dump)
        - Dataclasses (asdict)
        - Nested structures (recursive)
        - Base types (pass through)
        """
        from lionherd_core.ln import to_dict

        return to_dict(value, recursive=True, recursive_python_only=False)

    @field_serializer("embedding")
    def _serialize_embedding(self, value: list[float] | None, info: SerializationInfo) -> Any:
        """Serialize embedding to pgvector/jsonb/list format based on context."""
        if value is None:
            return None

        # Get format from context (default to "list" for backward compatibility)
        embedding_format = None
        if info.context:
            embedding_format = info.context.get("embedding_format")

        if embedding_format == "pgvector":
            # PostgreSQL pgvector extension: compact JSON string (no spaces)
            import orjson

            return orjson.dumps(value).decode()
        elif embedding_format == "jsonb":
            # PostgreSQL JSONB storage: standard JSON string with spaces
            import json

            return json.dumps(value)
        else:
            # Default: Python list (backward compatible)
            return value

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: Any) -> Any:
        """Validate structured content and handle polymorphic deserialization."""
        # Strict type enforcement
        if value is not None and not isinstance(value, (Serializable, BaseModel, dict)):
            raise TypeError(
                f"content must be Serializable, BaseModel, dict, or None. "
                f"Got {type(value).__name__}. "
                f"Use dict for unstructured data: content={{'value': {value!r}}} "
                f"or Element.metadata for simple key-value pairs."
            )

        # Polymorphic deserialization for dicts with lion_class metadata
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
        """Validate embedding, coerce JSON strings and ints to float list."""
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

    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
        meta_key: str | None = None,
        embedding_format: Literal["pgvector", "jsonb", "list"] | None = None,
        content_serializer: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize with optional embedding format and content serializer.

        Args:
            mode: Serialization mode (python/json/db)
            created_at_format: Format for created_at field (datetime/isoformat/timestamp)
            meta_key: Rename metadata field (overrides db default)
            embedding_format: Format for embedding serialization (pgvector/jsonb/list)
            content_serializer: Optional callable to serialize content field.
                If provided, content is excluded from model_dump and replaced
                with content_serializer(self.content) result.
            **kwargs: Additional arguments passed to model_dump()

        Returns:
            Serialized dict with optional custom content serialization
        """
        # Inject embedding_format into context for field_serializer
        # Only apply formatting in json/db modes (python mode preserves Python types)
        if embedding_format is not None and mode != "python":
            context = kwargs.get("context", {})
            context["embedding_format"] = embedding_format
            kwargs["context"] = context

        # Handle content_serializer
        if content_serializer is not None:
            # Fail-fast validation
            if not callable(content_serializer):
                raise TypeError(
                    f"content_serializer must be callable, got {type(content_serializer).__name__}"
                )

            # Test call to fail fast if serializer is broken
            try:
                _ = content_serializer(self.content)
            except Exception as e:
                raise ValueError(f"content_serializer failed on test call: {e}") from e

            # Exclude content from model_dump
            exclude = kwargs.get("exclude", set())
            if isinstance(exclude, set):
                exclude = exclude | {"content"}
            elif isinstance(exclude, dict):
                # dict exclude format: {"content": True}
                exclude = exclude.copy()
                exclude["content"] = True
            else:
                exclude = {"content"}
            kwargs["exclude"] = exclude

            # Get dict without content
            result = super().to_dict(
                mode=mode,
                created_at_format=created_at_format,
                meta_key=meta_key,
                **kwargs,
            )

            # Add serialized content
            result["content"] = content_serializer(self.content)
            return result

        # Delegate to Element.to_dict with context
        return super().to_dict(
            mode=mode,
            created_at_format=created_at_format,
            meta_key=meta_key,
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        meta_key: str | None = None,
        content_deserializer: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> Node:
        """Deserialize with polymorphic type restoration and optional content deserialization.

        Args:
            data: Dictionary to deserialize
            meta_key: Custom metadata field name (default: "metadata" or "node_metadata")
            content_deserializer: Optional callable to deserialize content field.
                Applied to content field before model_validate. Enables round-trip
                serialization with custom transformations (compression, encryption, etc.).
            **kwargs: Additional arguments passed to model_validate()

        Returns:
            Deserialized Node instance

        Examples:
            # Round-trip with compression
            compressed_data = node.to_dict(content_serializer=compress)
            restored = Node.from_dict(compressed_data, content_deserializer=decompress)

            # Round-trip with encryption
            encrypted_data = node.to_dict(content_serializer=encrypt)
            restored = Node.from_dict(encrypted_data, content_deserializer=decrypt)
        """
        # Make a copy to avoid mutating input
        data = data.copy()

        # Apply content_deserializer if provided
        if content_deserializer is not None:
            # Fail-fast validation
            if not callable(content_deserializer):
                raise TypeError(
                    f"content_deserializer must be callable, got {type(content_deserializer).__name__}"
                )

            # Apply deserializer to content field
            if "content" in data:
                try:
                    data["content"] = content_deserializer(data["content"])
                except Exception as e:
                    raise ValueError(f"content_deserializer failed: {e}") from e

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
                return target_cls.from_dict(
                    data, content_deserializer=content_deserializer, **kwargs
                )

        return cls.model_validate(data, **kwargs)

    @classmethod
    def register_adapter(cls, adapter: Any) -> None:  # pragma: no cover
        """Register adapter for this class."""
        super().register_adapter(adapter)

    @classmethod
    def register_async_adapter(cls, adapter: Any) -> None:  # pragma: no cover
        """Register async adapter for this class."""
        super().register_async_adapter(adapter)

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Convert to external format via pydapter (defaults to mode='db')."""
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return super().adapt_to(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Node:
        """Create from external format via pydapter (polymorphic)."""
        kwargs.setdefault("adapt_meth", "from_dict")
        return super().adapt_from(obj, obj_key=obj_key, many=many, **kwargs)

    async def adapt_to_async(
        self, obj_key: str, many: bool = False, **kwargs: Any
    ) -> Any:  # pragma: no cover
        """Async convert to external format via pydapter (defaults to mode='db')."""
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return await super().adapt_to_async(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    async def adapt_from_async(  # pragma: no cover
        cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any
    ) -> Node:
        """Async create from external format via pydapter (polymorphic)."""
        kwargs.setdefault("adapt_meth", "from_dict")
        return await super().adapt_from_async(obj, obj_key=obj_key, many=many, **kwargs)


NODE_REGISTRY[Node.__name__] = Node
NODE_REGISTRY[Node.class_name(full=True)] = Node

from pydapter.adapters import TomlAdapter, YamlAdapter

Node.register_adapter(TomlAdapter)  # type: ignore[type-abstract]
Node.register_adapter(YamlAdapter)  # type: ignore[type-abstract]

__all__ = ("NODE_REGISTRY", "Node")

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from ..protocols import (
    Deserializable,
    Hashable,
    Observable,
    Serializable,
    implements,
)
from ._utils import coerce_created_at, to_uuid

__all__ = ("DEFAULT_ELEMENT_SERIALIZER", "LN_ELEMENT_FIELDS", "Element")


@implements(Observable, Serializable, Deserializable, Hashable)
class Element(BaseModel):
    """Base element with UUID identity, timestamps, polymorphic serialization.

    Attributes:
        id: UUID identifier (frozen, auto-generated)
        created_at: UTC datetime (frozen, auto-generated)
        metadata: Arbitrary metadata dict

    Serialization injects lion_class for polymorphic deserialization.
    """

    id: UUID = Field(default_factory=uuid4, frozen=True)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC), frozen=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "use_enum_values": True,
        "extra": "forbid",
    }

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v) -> UUID:
        """Coerce to UUID4."""
        return to_uuid(v)

    @field_validator("created_at", mode="before")
    @classmethod
    def _coerce_created_at(cls, v) -> dt.datetime:
        """Coerce to UTC datetime."""
        return coerce_created_at(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_meta_integrity(cls, val: dict[str, Any] | None) -> dict[str, Any]:
        """Validate and coerce metadata to dict. Raises ValueError if conversion fails."""
        if not val:
            return {}

        if not isinstance(val, dict):
            from lionherd_core.ln import to_dict

            val = to_dict(val, recursive=True, suppress=True)

        if not isinstance(val, dict):
            raise ValueError("Invalid metadata: must be a dictionary")

        return val

    @classmethod
    def class_name(cls, full: bool = False) -> str:
        """Returns this class's name, stripping generic type parameters.

        For generic classes (e.g., Flow[Item, Prog]), returns the origin class name
        without type parameters (e.g., Flow).

        Args:
            full: If True, returns fully qualified name (module.Class); otherwise class name only

        Returns:
            Class name string without generic parameters

        Note:
            For Pydantic generic models, runtime classes have type parameters in __name__.
            We strip these using string parsing since typing.get_origin() doesn't work
            on Pydantic runtime instances.
        """
        # For Pydantic generic models, __name__ and __qualname__ include type params at runtime
        # e.g., "Flow[Item, Prog]" instead of "Flow"
        name = cls.__qualname__ if full else cls.__name__

        # Strip generic type parameters (Flow[E, P] -> Flow)
        if "[" in name:
            name = name.split("[")[0]

        if full:
            return f"{cls.__module__}.{name}"
        return name

    def _to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict with lion_class injected in metadata."""
        data = self.model_dump(**kwargs)

        # Inject lion_class for polymorphic deserialization, if not explicitly excluded
        if "metadata" in data:
            data["metadata"]["lion_class"] = self.__class__.class_name(full=True)

        return data

    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
        meta_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize to dict with lion_class metadata.

        Args:
            mode: python/json/db (db auto-renames metadata to node_metadata)
            created_at_format: datetime/isoformat/timestamp (auto-selected by mode)
            meta_key: Rename metadata field (overrides db default)
            **kwargs: Passed to model_dump()
        """
        if created_at_format is None:
            created_at_format = "isoformat" if mode == "json" else "datetime"

        if meta_key is None and mode == "db":
            meta_key = "node_metadata"

        if mode == "python":
            data = self._to_dict(**kwargs)
        elif mode in ("json", "db"):
            import orjson

            kwargs.pop("mode", None)  # Avoid recursion
            json_bytes = self.to_json(decode=False, **kwargs)
            data = orjson.loads(json_bytes)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'python', 'json', or 'db'")

        if "created_at" in data:
            if created_at_format == "isoformat":
                # json/db modes already converted via orjson, python mode needs conversion
                if mode == "python":
                    data["created_at"] = self.created_at.isoformat()
            elif created_at_format == "timestamp":
                data["created_at"] = self.created_at.timestamp()
            elif created_at_format == "datetime":
                # Only valid for python/db modes - json mode requires JSON-serializable
                if mode == "json":
                    raise ValueError(
                        "created_at_format='datetime' not valid for mode='json'. "
                        "Use 'isoformat' or 'timestamp' for JSON serialization."
                    )
                # db mode: convert isoformat string back to datetime
                if mode == "db" and isinstance(data["created_at"], str):
                    data["created_at"] = self.created_at

        # Rename metadata key if specified (works with any mode)
        if meta_key and "metadata" in data:
            data[meta_key] = data.pop("metadata")

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], meta_key: str | None = None, **kwargs: Any) -> Element:
        """Deserialize from dict with polymorphic type restoration via lion_class.

        Args:
            data: Serialized element dict
            meta_key: Restore metadata from this key (db mode compatibility)
            **kwargs: Passed to model_validate()

        Raises:
            ValueError: If lion_class invalid or not Element subclass
        """
        data = data.copy()  # avoid mutating input

        # Restore metadata from custom key if specified (db mode deserialization)
        if meta_key and meta_key in data:
            data["metadata"] = data.pop(meta_key)
        elif "node_metadata" in data and "metadata" not in data:  # backward compatibility
            data["metadata"] = data.pop("node_metadata")

        data.pop("node_metadata", None)  # remove legacy key if present

        # Extract and remove lion_class from metadata (serialization-only metadata)
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = metadata.copy()
            data["metadata"] = metadata
            lion_class = metadata.pop("lion_class", None)
        else:
            lion_class = None

        if lion_class and lion_class != cls.class_name(full=True):
            from ._utils import load_type_from_string  # Dynamic import to load the target class

            try:
                target_cls = load_type_from_string(lion_class)
            except ValueError as e:
                raise ValueError(f"Failed to deserialize class '{lion_class}': {e}") from e

            if not issubclass(target_cls, Element):
                raise ValueError(
                    f"'{lion_class}' is not an Element subclass. "
                    f"Cannot deserialize into {cls.__name__}"
                )

            # Polymorphic deserialization requires from_dict
            if not hasattr(target_cls, "from_dict") or not callable(
                getattr(target_cls, "from_dict", None)
            ):
                raise ValueError(
                    f"'{lion_class}' does not implement from_dict(). "
                    f"Cannot perform polymorphic deserialization"
                )

            # Prevent infinite recursion: check if target has different from_dict implementation
            # Use getattr to safely access __func__ (classmethods have it, but type system doesn't guarantee)
            target_func = getattr(target_cls.from_dict, "__func__", target_cls.from_dict)
            cls_func = getattr(cls.from_dict, "__func__", cls.from_dict)
            if target_func is cls_func:
                return target_cls.model_validate(data, **kwargs)

            # Delegate to target class's from_dict (different implementation)
            return target_cls.from_dict(data, **kwargs)

        return cls.model_validate(data, **kwargs)

    @classmethod
    def from_json(cls, json_str: str, /, **kwargs: Any) -> Element:
        """Create from JSON string."""
        import orjson

        return cls.from_dict(orjson.loads(json_str), **kwargs)

    def to_json(
        self,
        *,
        pretty: bool = False,
        sort_keys: bool = False,
        decode: bool = True,
        **kwargs: Any,
    ) -> str | bytes:
        """Serialize to JSON with nested Element/BaseModel support.

        Args:
            pretty: Indent output
            sort_keys: Sort dict keys
            decode: Return str (True) or bytes (False)
            **kwargs: Passed to model_dump()
        """
        from lionherd_core.ln import json_dumps

        # Get dict with lion_class metadata (python mode for nested object handling)
        data = self._to_dict(**kwargs)

        return json_dumps(
            data,
            default=_get_default_serializer(),
            pretty=pretty,
            sort_keys=sort_keys,
            decode=decode,
        )

    def __eq__(self, other: Any) -> bool:
        """Elements are equal if they have the same ID."""
        if not isinstance(other, Element):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash by ID for use in sets/dicts."""
        return hash(self.id)

    def __bool__(self) -> bool:
        """Elements are always truthy."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"


LN_ELEMENT_FIELDS = frozenset(("id", "created_at", "metadata"))
DEFAULT_ELEMENT_SERIALIZER = None


def _get_default_serializer():
    """Get or create default orjson serializer (lazy init to avoid circular imports)."""
    global DEFAULT_ELEMENT_SERIALIZER

    if DEFAULT_ELEMENT_SERIALIZER is None:
        from lionherd_core.ln import get_orjson_default

        from ._utils import get_element_serializer_config

        order, additional = get_element_serializer_config()
        DEFAULT_ELEMENT_SERIALIZER = get_orjson_default(order=order, additional=additional)

    return DEFAULT_ELEMENT_SERIALIZER

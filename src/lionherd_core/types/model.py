# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal, Self

import orjson
from pydantic import BaseModel, ConfigDict

from ..ln._hash import hash_dict
from ..ln._json_dump import get_orjson_default, json_dumps
from ..protocols import Deserializable, Hashable, Serializable, implements
from ._sentinel import not_sentinel

__all__ = ("ConversionMode", "HashableModel")

ConversionMode = Literal["python", "json"]


@implements(Serializable, Deserializable, Hashable)
class HashableModel(BaseModel):
    """Content-based hashable model (vs Element's ID-based hashing).

    Two instances with identical fields have same hash. Use for cache keys,
    deduplication, configs where value equality matters.

    Frozen by default (immutable) to prevent hash corruption when used in
    sets/dicts. This ensures safe use with to_list(unique=True) and as
    cache keys.

    Use Cases:
        - Structured LLM outputs with to_list(flatten=True, unique=True)
        - Cache keys where identical config values should deduplicate
        - Set deduplication based on field content
        - Value equality (same fields = same hash)

    When to Use Element Instead:
        - Workflow entities where identity matters (ID-based hashing)
        - Entities that mutate over time (ID remains stable)

    Hash Stability:
        Hash computation delegates to to_dict() serialization. Changes to
        serialization logic may affect hash values, potentially invalidating
        cached objects.
    """

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_default=True,
    )

    def _to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Convert to dict filtering sentinel values."""
        dict_ = self.model_dump(**kwargs)
        return {k: v for k, v in dict_.items() if not_sentinel(v)}

    def to_dict(self, mode: ConversionMode = "python", **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict.

        Args:
            mode: python (native types) or json (JSON-safe strings)
            **kwargs: Passed to model_dump(). Common: include, exclude, by_alias.
                WARNING: Do not pass 'mode' in kwargs (use parameter instead).
        """
        if mode == "python":
            return self._to_dict(**kwargs)
        elif mode == "json":
            return orjson.loads(self.to_json(decode=False))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'python' or 'json'")

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | str | bytes, mode: ConversionMode = "python", **kwargs: Any
    ) -> Self:
        """Deserialize from dict.

        Args:
            data: Dictionary to deserialize (or JSON string/bytes if mode='json')
            mode: 'python' for native types, 'json' for JSON-safe deserialization
            **kwargs: Passed to model_validate()

        Returns:
            Validated model instance

        Raises:
            ValueError: If mode is invalid
            ValidationError: If data doesn't match schema
        """
        if mode == "python":
            return cls.model_validate(data, **kwargs)
        elif mode == "json":
            if isinstance(data, (str, bytes)):
                data = orjson.loads(data)
            return cls.model_validate(data, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'python' or 'json'")

    def to_json(self, decode: bool = True, **kwargs: Any) -> str | bytes:
        """Serialize to deterministic JSON.

        Keys are sorted to ensure deterministic hashing - identical objects
        produce identical JSON output, critical for hash stability.

        Args:
            decode: If True, return str; if False, return bytes
            **kwargs: Passed to model_dump()

        Returns:
            JSON string (decode=True) or bytes (decode=False)
        """
        dict_ = self._to_dict(**kwargs)
        json_bytes = json_dumps(
            dict_,
            default=_get_default_hashable_serializer(),
            decode=False,
            sort_keys=True,
        )
        if decode:
            return json_bytes.decode("utf-8")
        return json_bytes

    @classmethod
    def from_json(cls, data: str | bytes, mode: ConversionMode = "json", **kwargs: Any) -> Self:
        """Deserialize from JSON string or bytes.

        Args:
            data: JSON string or bytes to deserialize
            mode: Conversion mode (typically 'json')
            **kwargs: Passed to model_validate()

        Returns:
            Validated model instance
        """
        return cls.from_dict(data, mode=mode, **kwargs)

    def __hash__(self) -> int:
        """Content-based hash (identical fields → same hash)."""
        return hash_dict(self.to_dict())


@lru_cache(maxsize=1)
def _get_default_hashable_serializer() -> Callable[[Any], Any]:
    """Lazy-init orjson serializer for nested models (thread-safe)."""
    return get_orjson_default(
        order=[Serializable, BaseModel],
        additional={
            HashableModel: lambda o: o.to_dict(),
            BaseModel: lambda o: o.model_dump(mode="json"),
        },
    )

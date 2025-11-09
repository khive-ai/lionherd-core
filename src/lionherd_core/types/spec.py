# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Self

from lionherd_core.libs.concurrency import is_coro_func
from lionherd_core.protocols import Hashable, implements

from ._sentinel import MaybeUndefined, Undefined, is_sentinel, not_sentinel
from .base import Enum, Meta

# Global cache for annotated types with bounded size
_MAX_CACHE_SIZE = int(os.environ.get("lionherd_FIELD_CACHE_SIZE", "10000"))
_annotated_cache: OrderedDict[tuple[type, tuple[Meta, ...]], type] = OrderedDict()
_cache_lock = threading.RLock()  # Thread-safe access to cache


__all__ = ("CommonMeta", "Spec")


class CommonMeta(Enum):
    """Common metadata keys: NAME, NULLABLE, LISTABLE, VALIDATOR, DEFAULT, DEFAULT_FACTORY."""

    NAME = "name"
    NULLABLE = "nullable"
    LISTABLE = "listable"
    VALIDATOR = "validator"
    DEFAULT = "default"
    DEFAULT_FACTORY = "default_factory"

    @classmethod
    def _validate_common_metas(cls, **kw):
        """Validate metadata constraints. Uses ExceptionGroup for multiple errors."""
        errors: list[Exception] = []

        if kw.get("default") and kw.get("default_factory"):
            errors.append(ValueError("Cannot provide both 'default' and 'default_factory'"))
        if (_df := kw.get("default_factory")) and not callable(_df):
            errors.append(ValueError("'default_factory' must be callable"))
        if _val := kw.get("validator"):
            _val = [_val] if not isinstance(_val, list) else _val
            if not all(callable(v) for v in _val):
                errors.append(ValueError("Validators must be a list of functions or a function"))

        if errors:
            raise ExceptionGroup("Metadata validation failed", errors)

    @classmethod
    def prepare(
        cls, *args: Meta, metadata: tuple[Meta, ...] | None = None, **kw: Any
    ) -> tuple[Meta, ...]:
        """Prepare metadata tuple from args/kw. Validates no duplicates, constraints."""
        # Lazy import to avoid circular dependency
        from ..ln._to_list import to_list

        seen_keys = set()
        metas = []

        # Process existing metadata
        if metadata:
            for meta in metadata:
                if meta.key in seen_keys:
                    raise ValueError(f"Duplicate metadata key: {meta.key}")
                seen_keys.add(meta.key)
                metas.append(meta)

        # Process args
        if args:
            _args = to_list(args, flatten=True, flatten_tuple_set=True, dropna=True)
            for meta in _args:
                if meta.key in seen_keys:
                    raise ValueError(f"Duplicate metadata key: {meta.key}")
                seen_keys.add(meta.key)
                metas.append(meta)

        # Process kwargs
        for k, v in kw.items():
            if k in seen_keys:
                raise ValueError(f"Duplicate metadata key: {k}")
            seen_keys.add(k)
            metas.append(Meta(k, v))

        # Validate common metadata constraints
        meta_dict = {m.key: m.value for m in metas}
        cls._validate_common_metas(**meta_dict)

        return tuple(metas)


@implements(Hashable)
@dataclass(frozen=True, slots=True, init=False)
class Spec:
    """Framework-agnostic field spec: base_type + metadata. Build with Spec(type, name=..., nullable=...)."""

    base_type: type
    metadata: tuple[Meta, ...]

    def __init__(
        self,
        base_type: type | None = None,
        *args,
        metadata: tuple[Meta, ...] | None = None,
        **kw,
    ) -> None:
        """Init with type and metadata. Args: base_type, Meta objects, kw as Meta."""
        metas = CommonMeta.prepare(*args, metadata=metadata, **kw)

        # Validate name is a valid string if provided
        meta_dict = {m.key: m.value for m in metas}
        if "name" in meta_dict:
            name_value = meta_dict["name"]
            if not isinstance(name_value, str) or not name_value:
                raise ValueError("Spec name must be a non-empty string")

        if not_sentinel(base_type, True):
            import types

            is_valid_type = (
                isinstance(base_type, type)
                or hasattr(base_type, "__origin__")
                or isinstance(base_type, types.UnionType)
            )
            if not is_valid_type:
                raise ValueError(f"base_type must be a type or type annotation, got {base_type}")

        # Check for async default factory and warn
        if kw.get("default_factory") and is_coro_func(kw["default_factory"]):
            import warnings

            warnings.warn(
                "Async default factories are not yet fully supported by all adapters. "
                "Consider using sync factories for compatibility.",
                UserWarning,
                stacklevel=2,
            )

        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "metadata", metas)

    def __getitem__(self, key: str) -> Any:
        """Get metadata by key. Raises KeyError if not found."""
        for meta in self.metadata:
            if meta.key == key:
                return meta.value
        raise KeyError(f"Metadata key '{key}' undefined in Spec.")

    def get(self, key: str, default: Any = Undefined) -> Any:
        """Get metadata by key with default."""
        with contextlib.suppress(KeyError):
            return self[key]
        return default

    @property
    def name(self) -> MaybeUndefined[str]:
        """Get the field name from metadata."""
        return self.get(CommonMeta.NAME.value)

    @property
    def is_nullable(self) -> bool:
        """Check if field is nullable."""
        return self.get(CommonMeta.NULLABLE.value) is True

    @property
    def is_listable(self) -> bool:
        """Check if field is listable."""
        return self.get(CommonMeta.LISTABLE.value) is True

    @property
    def default(self) -> MaybeUndefined[Any]:
        """Get default value or factory."""
        return self.get(
            CommonMeta.DEFAULT.value,
            self.get(CommonMeta.DEFAULT_FACTORY.value),
        )

    @property
    def has_default_factory(self) -> bool:
        """Check if this spec has a default factory."""
        return _is_factory(self.get(CommonMeta.DEFAULT_FACTORY.value))[0]

    @property
    def has_async_default_factory(self) -> bool:
        """Check if this spec has an async default factory."""
        return _is_factory(self.get(CommonMeta.DEFAULT_FACTORY.value))[1]

    def create_default_value(self) -> Any:
        """Create default value (sync). Raises ValueError if no default or async factory."""
        if self.default is Undefined:
            raise ValueError("No default value or factory defined in Spec.")
        if self.has_async_default_factory:
            raise ValueError(
                "Default factory is asynchronous; cannot create default synchronously. "
                "Use 'await spec.acreate_default_value()' instead."
            )
        if self.has_default_factory:
            return self.default()  # type: ignore[operator]
        return self.default

    async def acreate_default_value(self) -> Any:
        """Create default value (async). Handles both sync/async factories."""
        if self.has_async_default_factory:
            return await self.default()  # type: ignore[operator]
        return self.create_default_value()

    def with_updates(self, **kw) -> Self:
        """Create new Spec with updated metadata."""
        _filtered = [meta for meta in self.metadata if meta.key not in kw]
        for k, v in kw.items():
            if not_sentinel(v):
                _filtered.append(Meta(k, v))
        _metas = tuple(_filtered)
        return type(self)(self.base_type, metadata=_metas)

    def as_nullable(self) -> Self:
        """Create nullable version."""
        return self.with_updates(nullable=True)

    def as_listable(self) -> Self:
        """Create listable version."""
        return self.with_updates(listable=True)

    def with_default(self, default: Any) -> Self:
        """Create spec with default value/factory. Callables treated as factories."""
        if callable(default):
            return self.with_updates(default_factory=default)
        return self.with_updates(default=default)

    def with_validator(self, validator: Callable[..., Any] | list[Callable[..., Any]]) -> Self:
        """Create spec with validator(s)."""
        return self.with_updates(validator=validator)

    @property
    def annotation(self) -> type[Any]:
        """Plain type annotation: base_type + nullable/listable modifiers."""
        if is_sentinel(self.base_type, none_as_sentinel=True):
            return Any
        t_ = self.base_type  # type: ignore[valid-type]
        if self.is_listable:
            t_ = list[t_]  # type: ignore[valid-type]
        if self.is_nullable:
            t_ = t_ | None  # type: ignore[assignment]
        return t_  # type: ignore[return-value]

    def annotated(self) -> type[Any]:
        """Create Annotated[type, metadata...] with LRU cache."""
        # Check cache first with thread safety
        cache_key = (self.base_type, self.metadata)

        with _cache_lock:
            if cache_key in _annotated_cache:
                # Move to end to mark as recently used
                _annotated_cache.move_to_end(cache_key)
                return _annotated_cache[cache_key]

            # Handle nullable case - wrap in Optional-like union
            actual_type = (
                Any if is_sentinel(self.base_type, none_as_sentinel=True) else self.base_type
            )
            current_metadata = self.metadata

            if any(m.key == "nullable" and m.value for m in current_metadata):
                # Use union syntax for nullable
                actual_type = actual_type | None  # type: ignore

            if current_metadata:
                args = [actual_type, *list(current_metadata)]
                # Python 3.11-3.14 compatibility: try __class_getitem__ first (3.11-3.12),
                # fall back to direct subscripting approach for 3.13+
                try:
                    result = Annotated.__class_getitem__(tuple(args))  # type: ignore
                except AttributeError:
                    # Python 3.13+ removed __class_getitem__, use operator approach
                    import operator

                    result = operator.getitem(Annotated, tuple(args))  # type: ignore
            else:
                result = actual_type  # type: ignore[misc]

            # Cache the result with LRU eviction
            _annotated_cache[cache_key] = result  # type: ignore[assignment]

            # Evict oldest if cache is too large
            while len(_annotated_cache) > _MAX_CACHE_SIZE:
                _annotated_cache.popitem(last=False)  # Remove oldest

        return result  # type: ignore[return-value]

    def metadict(
        self, exclude: set[str] | None = None, exclude_common: bool = False
    ) -> dict[str, Any]:
        """Get metadata as dict. Args: exclude keys, exclude_common flag."""
        if exclude is None:
            exclude = set()
        if exclude_common:
            exclude = exclude | set(CommonMeta.allowed())
        return {meta.key: meta.value for meta in self.metadata if meta.key not in exclude}


def _is_factory(obj: Any) -> tuple[bool, bool]:
    """Check if object is a factory function.

    Args:
        obj: Object to check

    Returns:
        Tuple of (is_factory, is_async)
    """
    if not callable(obj):
        return (False, False)
    if is_coro_func(obj):
        return (True, True)
    return (True, False)

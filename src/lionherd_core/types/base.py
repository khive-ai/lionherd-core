# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence, MutableSet, Sequence
from dataclasses import dataclass
from enum import (
    Enum as _Enum,
    StrEnum,
)
from typing import Any, ClassVar, Literal, Self, TypedDict

from typing_extensions import override

from ..protocols import Allowable, Hashable, Serializable, implements
from ._sentinel import Undefined, Unset, is_sentinel

__all__ = (
    "DataClass",
    "Enum",
    "KeysDict",
    "KeysLike",
    "Meta",
    "ModelConfig",
    "Params",
)


@implements(Allowable)
class Enum(StrEnum):
    """String-backed enum (Python 3.11+). Members are strings, support JSON serialization."""

    @classmethod
    def allowed(cls) -> tuple[str, ...]:
        """Return tuple of all allowed string values."""
        return tuple(e.value for e in cls)


class KeysDict(TypedDict, total=False):
    """TypedDict for keys dictionary."""

    key: Any  # Represents any key-type pair


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Config for Params/DataClass: sentinel handling, validation, serialization."""

    # Sentinel handling (controls what gets excluded from to_dict)
    none_as_sentinel: bool = False
    empty_as_sentinel: bool = False

    # Validation
    strict: bool = False
    prefill_unset: bool = True

    # Serialization
    use_enum_values: bool = False


@implements(Serializable, Allowable, Hashable)
@dataclass(slots=True, frozen=True, init=False)
class Params:
    """Base for function parameters with sentinel handling. Configure via _config."""

    _config: ClassVar[ModelConfig] = ModelConfig()
    _allowed_keys: ClassVar[set[str]] = set()

    def __init__(self, **kwargs: Any):
        """Init from kwargs. Validates and sets attributes."""
        # Set all attributes from kwargs, allowing for sentinel values
        for k, v in kwargs.items():
            if k in self.allowed():
                object.__setattr__(self, k, v)
            else:
                raise ValueError(f"Invalid parameter: {k}")

        # Validate after setting all attributes
        self._validate()

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if value is sentinel (respects config)."""
        return is_sentinel(
            value,
            none_as_sentinel=cls._config.none_as_sentinel,
            empty_as_sentinel=cls._config.empty_as_sentinel,
        )

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Normalize value for serialization (enum handling, etc.)."""
        if cls._config.use_enum_values and isinstance(value, _Enum):
            return value.value
        return value

    @classmethod
    def allowed(cls) -> set[str]:
        """Return the keys of the parameters."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {i for i in cls.__dataclass_fields__ if not i.startswith("_")}
        return cls._allowed_keys

    def _validate(self) -> None:
        """Validate params. Collects errors in ExceptionGroup. Prefills unset if configured."""
        missing: list[Exception] = []
        for k in self.allowed():
            if self._config.strict and self._is_sentinel(getattr(self, k, Unset)):
                missing.append(ValueError(f"Missing required parameter: {k}"))
            if self._config.prefill_unset and getattr(self, k, Undefined) is Undefined:
                object.__setattr__(self, k, Unset)
        if missing:
            raise ExceptionGroup("Missing required parameters", missing)

    def default_kw(self) -> Any:
        # create a partial function with the current parameters
        dict_ = self.to_dict()

        # handle kwargs if present, handle both 'kwargs' and 'kw'
        kw_ = {}
        kw_.update(dict_.pop("kwargs", {}))
        kw_.update(dict_.pop("kw", {}))
        dict_.update(kw_)
        return dict_

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        data = {}
        exclude = exclude or set()
        for k in self.allowed():
            if k not in exclude:
                v = getattr(self, k, Undefined)
                if not self._is_sentinel(v):
                    data[k] = self._normalize_value(v)
        return data

    def __hash__(self) -> int:
        from ..ln._hash import hash_dict

        return hash_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        """Equality via hash comparison. Returns NotImplemented for non-Params types."""
        if not isinstance(other, Params):
            return NotImplemented
        return hash(self) == hash(other)

    def with_updates(
        self, copy_containers: Literal["shallow", "deep"] | None = None, **kwargs: Any
    ) -> Self:
        """Return new instance with updated fields.

        Args:
            copy_containers: None (no copy), "shallow" (top-level), or "deep" (recursive).
            **kwargs: Field updates.
        """
        dict_ = self.to_dict()

        def _out(d: dict):
            d.update(kwargs)
            return type(self)(**d)

        if copy_containers is None:
            return _out(dict_)

        match copy_containers:
            case "shallow":
                for k, v in dict_.items():
                    if k not in kwargs and isinstance(
                        v, (MutableSequence, MutableMapping, MutableSet)
                    ):
                        dict_[k] = v.copy()
                return _out(dict_)

            case "deep":
                import copy

                for k, v in dict_.items():
                    if k not in kwargs and isinstance(
                        v, (MutableSequence, MutableMapping, MutableSet)
                    ):
                        dict_[k] = copy.deepcopy(v)
                return _out(dict_)

        raise ValueError(
            f"Invalid copy_containers: {copy_containers!r}. Must be 'shallow', 'deep', or None."
        )


@implements(Serializable, Allowable, Hashable)
@dataclass(slots=True)
class DataClass:
    """Base for dataclasses with strict parameter handling. Configure via _config."""

    _config: ClassVar[ModelConfig] = ModelConfig()
    _allowed_keys: ClassVar[set[str]] = set()

    def __post_init__(self):
        """Post-init: validates all fields."""
        self._validate()

    @classmethod
    def allowed(cls) -> set[str]:
        """Return the keys of the parameters."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {i for i in cls.__dataclass_fields__ if not i.startswith("_")}
        return cls._allowed_keys

    def _validate(self) -> None:
        """Validate params. Collects errors in ExceptionGroup. Prefills unset if configured."""
        missing: list[Exception] = []
        for k in self.allowed():
            if self._config.strict and self._is_sentinel(getattr(self, k, Unset)):
                missing.append(ValueError(f"Missing required parameter: {k}"))
            if self._config.prefill_unset and getattr(self, k, Undefined) is Undefined:
                self.__setattr__(k, Unset)
        if missing:
            raise ExceptionGroup("Missing required parameters", missing)

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        data = {}
        exclude = exclude or set()
        for k in type(self).allowed():
            if k not in exclude:
                v = getattr(self, k)
                if not self._is_sentinel(v):
                    data[k] = self._normalize_value(v)
        return data

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if value is sentinel (respects config)."""
        return is_sentinel(
            value,
            none_as_sentinel=cls._config.none_as_sentinel,
            empty_as_sentinel=cls._config.empty_as_sentinel,
        )

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Normalize value for serialization (enum handling, etc.)."""
        from enum import Enum as _Enum

        if cls._config.use_enum_values and isinstance(value, _Enum):
            return value.value
        return value

    def with_updates(
        self, copy_containers: Literal["shallow", "deep"] | None = None, **kwargs: Any
    ) -> Self:
        """Return new instance with updated fields.

        Args:
            copy_containers: None (no copy), "shallow" (top-level), or "deep" (recursive).
            **kwargs: Field updates.
        """
        dict_ = self.to_dict()

        def _out(d: dict):
            d.update(kwargs)
            return type(self)(**d)

        if copy_containers is None:
            return _out(dict_)

        match copy_containers:
            case "shallow":
                for k, v in dict_.items():
                    if k not in kwargs and isinstance(
                        v, (MutableSequence, MutableMapping, MutableSet)
                    ):
                        dict_[k] = v.copy()
                return _out(dict_)

            case "deep":
                import copy

                for k, v in dict_.items():
                    if k not in kwargs and isinstance(
                        v, (MutableSequence, MutableMapping, MutableSet)
                    ):
                        dict_[k] = copy.deepcopy(v)
                return _out(dict_)

        raise ValueError(
            f"Invalid copy_containers: {copy_containers!r}. Must be 'shallow', 'deep', or None."
        )

    def __hash__(self) -> int:
        from ..ln._hash import hash_dict

        return hash_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        """Equality via hash comparison. Returns NotImplemented for non-DataClass types."""
        if not isinstance(other, DataClass):
            return NotImplemented
        return hash(self) == hash(other)


KeysLike = Sequence[str] | KeysDict


@implements(Hashable)
@dataclass(slots=True, frozen=True)
class Meta:
    """Immutable metadata container. Hashable for caching (callables hashed by id)."""

    key: str
    value: Any

    @override
    def __hash__(self) -> int:
        """Hash metadata. Callables use id() for identity semantics."""
        # For callables, use their id
        if callable(self.value):
            return hash((self.key, id(self.value)))
        # For other values, try to hash directly
        try:
            return hash((self.key, self.value))
        except TypeError:
            # Fallback for unhashable types
            return hash((self.key, str(self.value)))

    @override
    def __eq__(self, other: object) -> bool:
        """Equality: callables compared by id, others by standard equality."""
        if not isinstance(other, Meta):
            return NotImplemented

        if self.key != other.key:
            return False

        # For callables, compare by identity
        if callable(self.value) and callable(other.value):
            return id(self.value) == id(other.value)

        # For other values, use standard equality
        return bool(self.value == other.value)

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar, Final, Literal, Self, TypeAlias, TypeGuard, TypeVar, Union

__all__ = (
    "MaybeSentinel",
    "MaybeUndefined",
    "MaybeUnset",
    "SingletonType",
    "T",
    "Undefined",
    "UndefinedType",
    "Unset",
    "UnsetType",
    "is_sentinel",
    "not_sentinel",
)

T = TypeVar("T")


class _SingletonMeta(type):
    """Metaclass: ensures one instance per subclass for safe 'is' checks."""

    _cache: ClassVar[dict[type, SingletonType]] = {}

    def __call__(cls, *a, **kw):
        if cls not in cls._cache:
            cls._cache[cls] = super().__call__(*a, **kw)
        return cls._cache[cls]


class SingletonType(metaclass=_SingletonMeta):
    """Base for singleton sentinels. Falsy, identity-preserving across copy/deepcopy."""

    __slots__: tuple[str, ...] = ()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Preserve singleton identity across deepcopy."""
        return self

    def __copy__(self) -> Self:
        """Preserve singleton identity across copy."""
        return self

    # concrete classes *must* override the two methods below
    def __bool__(self) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class UndefinedType(SingletonType):
    """Sentinel: field/key entirely missing from namespace. Use for missing keys, never-set fields."""

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __str__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __reduce__(self) -> tuple[type[UndefinedType], tuple[()]]:
        """Preserve singleton identity across pickle/unpickle."""
        return (UndefinedType, ())

    def __or__(self, other: type) -> Any:
        """Enable union syntax: str | Undefined or Undefined | Unset"""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[type(self), other_type]

    def __ror__(self, other: type) -> Any:
        """Enable reverse union: Undefined | str or Unset | Undefined"""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[other_type, type(self)]


class UnsetType(SingletonType):
    """Sentinel: key present but value not provided. Use to distinguish None from 'not provided'."""

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Unset"]:
        return "Unset"

    def __str__(self) -> Literal["Unset"]:
        return "Unset"

    def __reduce__(self) -> tuple[type[UnsetType], tuple[()]]:
        """Preserve singleton identity across pickle/unpickle."""
        return (UnsetType, ())

    def __or__(self, other: type) -> Any:
        """Enable union syntax: str | Unset or Unset | Undefined"""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[type(self), other_type]

    def __ror__(self, other: type) -> Any:
        """Enable reverse union: Unset | str or Undefined | Unset"""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[other_type, type(self)]


Undefined: Final[UndefinedType] = UndefinedType()
"""A key or field entirely missing from a namespace"""
Unset: Final[UnsetType] = UnsetType()
"""A key present but value not yet provided."""

MaybeUndefined: TypeAlias = T | UndefinedType
MaybeUnset: TypeAlias = T | UnsetType
MaybeSentinel: TypeAlias = T | UndefinedType | UnsetType

_EMPTY_TUPLE: tuple[Any, ...] = (tuple(), set(), frozenset(), dict(), list(), "")


def is_sentinel(
    value: Any,
    *,
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> bool:
    """Check if a value is any sentinel (Undefined or Unset)."""
    if none_as_sentinel and value is None:
        return True
    if empty_as_sentinel and value in _EMPTY_TUPLE:
        return True
    return value is Undefined or value is Unset


def not_sentinel(
    value: T | UndefinedType | UnsetType,
    none_as_sentinel: bool = False,
    empty_as_sentinel: bool = False,
) -> TypeGuard[T]:
    """Type-narrowing check: NOT a sentinel. Narrows MaybeSentinel[T] to T for type checkers."""
    return not is_sentinel(
        value,
        none_as_sentinel=none_as_sentinel,
        empty_as_sentinel=empty_as_sentinel,
    )

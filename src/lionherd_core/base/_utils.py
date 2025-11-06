# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import contextlib
import datetime as dt
import types
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, Union, get_args, get_origin
from uuid import UUID

from ..protocols import Observable, Serializable

__all__ = (
    "async_synchronized",
    "coerce_created_at",
    "extract_types",
    "get_element_serializer_config",
    "get_json_serializable",
    "load_type_from_string",
    "synchronized",
    "to_uuid",
)


def synchronized(func: Callable) -> Callable:
    """Decorator for thread-safe method execution.

    Executes method within self._lock context. The class must have a _lock
    attribute (typically threading.RLock()).

    Example:
        @synchronized
        def append(self, item):
            self.items.append(item)
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


def async_synchronized(func: Callable) -> Callable:
    """Decorator for async-safe method execution.

    Executes async method within self._async_lock context. The class must have a
    _async_lock attribute (typically asyncio.Lock()).

    Example:
        @async_synchronized
        async def append_async(self, item):
            self.items.append(item)
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with self._async_lock:
            return await func(self, *args, **kwargs)

    return wrapper


_TYPE_CACHE: dict[str, type] = {}


def load_type_from_string(type_str: str) -> type:
    """Load type from fully qualified module path string.

    Args:
        type_str: Fully qualified type path (e.g., 'lionherd.base.Node')

    Returns:
        Type class

    Raises:
        ValueError: If type string is invalid or type cannot be loaded

    Example:
        >>> load_type_from_string("lionherd.base.Node")
        <class 'lionherd.base.node.Node'>
    """
    # Check cache first
    if type_str in _TYPE_CACHE:
        return _TYPE_CACHE[type_str]

    if not isinstance(type_str, str):
        raise ValueError(f"Expected string, got {type(type_str)}")

    if "." not in type_str:
        raise ValueError(f"Invalid type path (no module): {type_str}")

    try:
        module_path, class_name = type_str.rsplit(".", 1)

        # Import module using importlib for correct behavior
        import importlib

        module = importlib.import_module(module_path)
        if module is None:
            raise ImportError(f"Module '{module_path}' not found")

        type_class = getattr(module, class_name)

        # Validate it's actually a type
        if not isinstance(type_class, type):
            raise ValueError(f"'{type_str}' is not a type")

        # Cache result
        _TYPE_CACHE[type_str] = type_class

        return type_class

    except (ValueError, ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load type '{type_str}': {e}") from e


def extract_types(item_type: Any) -> set[type]:
    """Extract types from union types or convert single types to set.

    Handles Python 3.10+ pipe syntax (int | str), typing.Union, lists, and sets.

    Args:
        item_type: Single type, list of types, set of types, or Union type

    Returns:
        Set of extracted types

    Examples:
        extract_types(int) -> {int}
        extract_types([int, str]) -> {int, str}
        extract_types(int | str) -> {int, str}
        extract_types(Union[int, str]) -> {int, str}
    """

    # Helper to check if type is a union
    def is_union(t):
        origin = get_origin(t)
        # Python 3.10+ pipe syntax (Element | Node) creates types.UnionType
        # typing.Union creates Union origin
        return origin is Union or isinstance(t, types.UnionType)

    extracted: set[type] = set()

    # Already a set
    if isinstance(item_type, set):
        # Check if any items in set are unions and extract them
        for t in item_type:
            if is_union(t):
                extracted.update(get_args(t))
            else:
                extracted.add(t)
        return extracted

    # List of types
    if isinstance(item_type, list):
        for t in item_type:
            if is_union(t):
                extracted.update(get_args(t))
            else:
                extracted.add(t)
        return extracted

    # Union type (int | str or Union[int, str])
    if is_union(item_type):
        return set(get_args(item_type))

    # Single type
    return {item_type}


def to_uuid(value: Any) -> UUID:
    """Convert various ID representations to UUID.

    Args:
        value: UUID, UUID string, or Observable object with id property

    Returns:
        UUID instance

    Raises:
        ValueError: If value cannot be converted to UUID or Observable.id is callable

    Examples:
        >>> to_uuid(UUID("12345678-1234-5678-1234-567812345678"))
        UUID('12345678-1234-5678-1234-567812345678')
        >>> to_uuid("12345678-1234-5678-1234-567812345678")
        UUID('12345678-1234-5678-1234-567812345678')
    """
    if isinstance(value, Observable):
        id_value = value.id
        if callable(id_value):
            raise ValueError(
                f"Observable.id must be a property, not a method. Got callable: {type(value).__name__}.id()"
            )
        return id_value
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise ValueError("Cannot get ID from item.")


def coerce_created_at(v) -> dt.datetime:
    """Coerce datetime, timestamp, or ISO string to UTC-aware datetime.

    Args:
        v: datetime, float timestamp, or ISO 8601 string

    Returns:
        UTC-aware datetime

    Raises:
        ValueError: If cannot coerce to datetime
    """
    # datetime object
    if isinstance(v, dt.datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=dt.UTC)
        return v

    # Float timestamp (seconds since epoch)
    if isinstance(v, (int, float)):
        return dt.datetime.fromtimestamp(v, tz=dt.UTC)

    if isinstance(v, str):
        # Try parsing as float timestamp
        with contextlib.suppress(ValueError):
            timestamp = float(v)
            return dt.datetime.fromtimestamp(timestamp, tz=dt.UTC)
        # Try parsing as ISO format
        with contextlib.suppress(ValueError):
            return dt.datetime.fromisoformat(v)

        raise ValueError(f"String '{v}' is neither a valid timestamp nor ISO format datetime")

    raise ValueError(
        f"created_at must be datetime, timestamp (int/float), or string, got {type(v).__name__}"
    )


_SIMPLE_TYPE = (str, bytes, bytearray, int, float, type(None), Enum)


def get_json_serializable(data) -> dict[str, Any] | Any:
    """Serialize to dict."""
    from ..ln import json_dumpb, to_dict
    from ..types._sentinel import Unset

    if data is Unset:
        return Unset

    if isinstance(data, _SIMPLE_TYPE):
        return data

    with contextlib.suppress(Exception):
        # Check if response is JSON serializable
        json_dumpb(data)
        return data

    with contextlib.suppress(Exception):
        # Attempt to force convert to dict recursively
        d_ = to_dict(
            data,
            recursive=True,
            recursive_python_only=False,
            use_enum_values=True,
        )
        json_dumpb(d_)
        return d_

    return Unset


def get_element_serializer_config() -> tuple[list, dict]:
    """Get serializer config for Element.to_json().

    Returns:
        Tuple of (order, additional) for get_orjson_default():
        - order: [Serializable, BaseModel]
        - additional: {type: serializer_func}
    """
    from pydantic import BaseModel

    order = [Serializable, BaseModel]

    additional = {
        Serializable: lambda o: o.to_dict(),
        BaseModel: lambda o: o.model_dump(mode="json"),
    }

    return order, additional

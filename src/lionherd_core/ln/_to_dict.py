# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum as _Enum
from typing import Any, cast

import orjson

from ..libs.string_handlers._fuzzy_json import fuzzy_json


def _is_na(obj: Any) -> bool:
    """None / Pydantic undefined sentinels -> treat as NA."""
    if obj is None:
        return True
    # Avoid importing pydantic types; match by typename to stay lightweight
    tname = type(obj).__name__
    return tname in {
        "Undefined",
        "UndefinedType",
        "PydanticUndefined",
        "PydanticUndefinedType",
    }


def _enum_class_to_dict(enum_cls: type[_Enum], use_enum_values: bool) -> dict[str, Any]:
    members = dict(enum_cls.__members__)  # cheap, stable
    if use_enum_values:
        return {k: v.value for k, v in members.items()}
    return {k: v for k, v in members.items()}


def _parse_str(
    s: str,
    *,
    fuzzy_parse: bool,
    parser: Callable[[str], Any] | None,
    **kwargs: Any,
) -> Any:
    """Parse str -> Python object (JSON only).

    Keep imports local to avoid cold start overhead.
    """
    if parser is not None:
        return parser(s, **kwargs)

    # JSON path
    if fuzzy_parse:
        # fuzzy_json() accepts only a string parameter, no kwargs
        # Passing kwargs causes TypeError, which was silently suppressed
        with contextlib.suppress(NameError):
            return fuzzy_json(s)  # type: ignore[name-defined]
    # orjson.loads() doesn't accept kwargs like parse_float, object_hook, etc.
    return orjson.loads(s)


def _object_to_mapping_like(
    obj: Any,
    *,
    prioritize_model_dump: bool = False,
    **kwargs: Any,
) -> Mapping | dict | Any:
    """
    Convert 'custom' objects to mapping-like, if possible.
    Order:
      1) Pydantic v2 'model_dump' (duck-typed)
      2) Common methods: to_dict, dict, to_json/json (parsed if string)
      3) Dataclass
      4) __dict__
      5) dict(obj)
    """
    # 1) Pydantic v2
    if prioritize_model_dump and hasattr(obj, "model_dump"):
        return obj.model_dump(**kwargs)

    # 2) Common methods
    for name in ("to_dict", "dict", "to_json", "json", "model_dump"):
        if hasattr(obj, name):
            res = getattr(obj, name)(**kwargs)
            return orjson.loads(res) if isinstance(res, str) else res

    # 3) Dataclass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # asdict is already recursive; keep it (fast enough & simple)
        return dataclasses.asdict(obj)  # type: ignore[arg-type]

    # 4) __dict__
    if hasattr(obj, "__dict__"):
        return obj.__dict__

    # 5) Try dict() fallback
    return dict(obj)  # may raise -> handled by caller


def _enumerate_iterable(it: Iterable) -> dict[int, Any]:
    return {i: v for i, v in enumerate(it)}


# ---------------------------------------
# Recursive pre-processing (single pass)
# ---------------------------------------


def _preprocess_recursive(
    obj: Any,
    *,
    depth: int,
    max_depth: int,
    recursive_custom_types: bool,
    str_parse_opts: dict[str, Any],
    prioritize_model_dump: bool,
) -> Any:
    """
    Recursively process nested structures:
      - Parse strings (JSON only, or custom parser)
      - Recurse into dict/list/tuple/set/etc.
      - If recursive_custom_types=True, convert custom objects to mapping-like then continue
    Containers retain their original types (dict stays dict, list stays list, set stays set, etc.)
    """
    if depth >= max_depth:
        return obj

    # Fast paths by exact type where possible
    t = type(obj)

    # Strings: try to parse; on failure, keep as-is
    if t is str:
        with contextlib.suppress(Exception):
            return _preprocess_recursive(
                _parse_str(obj, **str_parse_opts),
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
        return obj

    # Dict-like
    if isinstance(obj, Mapping):
        # Recurse only into values (keys kept as-is)
        return {
            k: _preprocess_recursive(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
            for k, v in obj.items()
        }

    # Sequence/Set-like (but not str)
    if isinstance(obj, list | tuple | set | frozenset):
        items = [
            _preprocess_recursive(
                v,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
            for v in obj
        ]
        if t is list:
            return items
        if t is tuple:
            return tuple(items)
        if t is set:
            return set(items)
        if t is frozenset:
            return frozenset(items)

    # Enum *class* (rare in values, but preserve your original attempt)
    if isinstance(obj, type) and issubclass(obj, _Enum):
        with contextlib.suppress(Exception):
            enum_map = _enum_class_to_dict(
                obj,
                use_enum_values=str_parse_opts.get("use_enum_values", True),
            )
            return _preprocess_recursive(
                enum_map,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )
        return obj

    # Custom objects
    if recursive_custom_types:
        with contextlib.suppress(Exception):
            mapped = _object_to_mapping_like(obj, prioritize_model_dump=prioritize_model_dump)
            return _preprocess_recursive(
                mapped,
                depth=depth + 1,
                max_depth=max_depth,
                recursive_custom_types=recursive_custom_types,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )

    return obj


# ---------------------------------------
# Top-level conversion (non-recursive)
# ---------------------------------------


def _convert_top_level_to_dict(
    obj: Any,
    *,
    fuzzy_parse: bool,
    parser: Callable[[str], Any] | None,
    prioritize_model_dump: bool,
    use_enum_values: bool,
    **kwargs: Any,
) -> dict[str | int, Any]:
    """
    Convert a *single* object to dict using the 'brute force' rules.
    Mirrors your original order, with fixes & optimizations.
    """
    # Set -> {v: v}
    if isinstance(obj, set):
        return cast(dict[str | int, Any], {v: v for v in obj})

    # Enum class -> members mapping
    if isinstance(obj, type) and issubclass(obj, _Enum):
        return cast(dict[str | int, Any], _enum_class_to_dict(obj, use_enum_values))

    # Mapping -> copy to plain dict (preserve your copy semantics)
    if isinstance(obj, Mapping):
        return cast(dict[str | int, Any], dict(obj))

    # None / pydantic undefined -> {}
    if _is_na(obj):
        return cast(dict[str | int, Any], {})

    # str -> parse (and return *as parsed*, which may be list, dict, etc.)
    if isinstance(obj, str):
        return _parse_str(
            obj,
            fuzzy_parse=fuzzy_parse,
            parser=parser,
            **kwargs,
        )

    # Try "custom" object conversions
    # (Covers BaseModel via model_dump, dataclasses, __dict__, json-strings, etc.)
    with contextlib.suppress(Exception):
        # If it's *not* a Sequence (e.g., numbers, objects) we try object conversion first,
        # faithfully following your previous "non-Sequence -> model path" behavior.
        if not isinstance(obj, Sequence):
            converted = _object_to_mapping_like(
                obj, prioritize_model_dump=prioritize_model_dump, **kwargs
            )
            # If conversion returned a string, try to parse JSON to mapping; else pass-through
            if isinstance(converted, str):
                return _parse_str(
                    converted,
                    fuzzy_parse=fuzzy_parse,
                    parser=None,
                )
            if isinstance(converted, Mapping):
                return dict(converted)
            # If it's a list/tuple/etc., enumerate (your original did that after the fact)
            if isinstance(converted, Iterable) and not isinstance(
                converted, str | bytes | bytearray
            ):
                return cast(dict[str | int, Any], _enumerate_iterable(converted))
            # Best effort final cast
            return dict(converted)

    # Iterable (list/tuple/namedtuple/frozenset/…): enumerate
    if isinstance(obj, Iterable) and not isinstance(obj, str | bytes | bytearray):
        return cast(dict[str | int, Any], _enumerate_iterable(obj))

    # Dataclass fallback (reachable only if it wasn't caught above)
    with contextlib.suppress(Exception):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return cast(dict[str | int, Any], dataclasses.asdict(obj))  # type: ignore[arg-type]

    # Last-ditch attempt
    return dict(obj)  # may raise, handled by top-level try/except


def to_dict(
    input_: Any,
    /,
    *,
    prioritize_model_dump: bool = False,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    parser: Callable[[str], Any] | None = None,
    recursive: bool = False,
    max_recursive_depth: int | None = None,
    recursive_python_only: bool = True,
    use_enum_values: bool = False,
    **kwargs: Any,
) -> dict[str | int, Any]:
    """
    Convert various input types to a dictionary, with optional recursive processing.

    Supports JSON string parsing via orjson (or custom parser).

    Args:
        input_: Object to convert to dict
        prioritize_model_dump: If True, prefer .model_dump() for Pydantic models
        fuzzy_parse: If True, use fuzzy JSON parsing for strings
        suppress: If True, return {} on errors instead of raising
        parser: Custom parser callable for string inputs
        recursive: If True, recursively process nested structures
        max_recursive_depth: Maximum recursion depth (default 5, max 10)
        recursive_python_only: If True, only recurse into Python builtins
        use_enum_values: If True, use .value for Enum members
        **kwargs: Additional kwargs passed to parser

    Returns:
        Dictionary representation of input
    """
    try:
        # Clamp recursion depth (match your constraints)
        if not isinstance(max_recursive_depth, int):
            max_depth = 5
        else:
            if max_recursive_depth < 0:
                raise ValueError("max_recursive_depth must be a non-negative integer")
            if max_recursive_depth > 10:
                raise ValueError("max_recursive_depth must be less than or equal to 10")
            max_depth = max_recursive_depth

        # Prepare one small dict to avoid repeated arg passing and lookups
        str_parse_opts = {
            "fuzzy_parse": fuzzy_parse,
            "parser": parser,
            "use_enum_values": use_enum_values,  # threaded for enum class in recursion
            **kwargs,
        }

        obj = input_
        if recursive:
            obj = _preprocess_recursive(
                obj,
                depth=0,
                max_depth=max_depth,
                recursive_custom_types=not recursive_python_only,
                str_parse_opts=str_parse_opts,
                prioritize_model_dump=prioritize_model_dump,
            )

        # Final top-level conversion
        return _convert_top_level_to_dict(
            obj,
            fuzzy_parse=fuzzy_parse,
            parser=parser,
            prioritize_model_dump=prioritize_model_dump,
            use_enum_values=use_enum_values,
            **kwargs,
        )

    except Exception as e:
        if suppress or input_ == "":
            return {}
        raise e

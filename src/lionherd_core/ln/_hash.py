# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import copy
from typing import Any

__all__ = ("hash_dict",)

# Global initialization state
_INITIALIZED = False
PydanticBaseModel = None

# --- Canonical Representation Generator ---
_PRIMITIVE_TYPES = (str, int, float, bool, type(None))
_TYPE_MARKER_DICT = 0
_TYPE_MARKER_LIST = 1
_TYPE_MARKER_TUPLE = 2
_TYPE_MARKER_SET = 3
_TYPE_MARKER_FROZENSET = 4
_TYPE_MARKER_PYDANTIC = 5  # Distinguishes dumped Pydantic models


def _generate_hashable_representation(item: Any) -> Any:
    """Convert object to stable hashable representation recursively.

    Ensures order-independent hashing for dicts/sets and consistent handling of collections.
    """
    if isinstance(item, _PRIMITIVE_TYPES):
        return item

    if PydanticBaseModel and isinstance(item, PydanticBaseModel):
        # Process the Pydantic model by first dumping it to a dict, then processing that dict.
        # The type marker distinguishes this from a regular dictionary.
        return (
            _TYPE_MARKER_PYDANTIC,
            _generate_hashable_representation(item.model_dump()),
        )

    if isinstance(item, dict):
        # Sort dictionary items by key (stringified) for order-insensitivity.
        return (
            _TYPE_MARKER_DICT,
            tuple(
                (str(k), _generate_hashable_representation(v))
                for k, v in sorted(item.items(), key=lambda x: str(x[0]))
            ),
        )

    if isinstance(item, list):
        return (
            _TYPE_MARKER_LIST,
            tuple(_generate_hashable_representation(elem) for elem in item),
        )

    if isinstance(item, tuple):
        return (
            _TYPE_MARKER_TUPLE,
            tuple(_generate_hashable_representation(elem) for elem in item),
        )

    # frozenset must be checked before set
    if isinstance(item, frozenset):
        try:  # Attempt direct sort for comparable elements
            sorted_elements = sorted(list(item))
        except TypeError:  # Fallback for unorderable mixed types

            def sort_key(x):
                # Deterministic ordering across mixed, unorderable types
                # Sort strictly by textual type then textual value.
                # This also naturally places bool before int because
                # "<class 'bool'>" < "<class 'int'>" lexicographically.
                return (str(type(x)), str(x))

            sorted_elements = sorted(list(item), key=sort_key)
        return (
            _TYPE_MARKER_FROZENSET,
            tuple(_generate_hashable_representation(elem) for elem in sorted_elements),
        )

    if isinstance(item, set):
        try:
            sorted_elements = sorted(list(item))
        except TypeError:
            # For mixed types, use a deterministic, portable sort key
            def sort_key(x):
                # Sort by textual type then textual value for stability.
                return (str(type(x)), str(x))

            sorted_elements = sorted(list(item), key=sort_key)
        return (
            _TYPE_MARKER_SET,
            tuple(_generate_hashable_representation(elem) for elem in sorted_elements),
        )

    # Fallback for other types (e.g., custom objects not derived from the above)
    with contextlib.suppress(Exception):
        return str(item)
    with contextlib.suppress(Exception):
        return repr(item)

    # If both str() and repr() fail, return a stable fallback based on type and id
    return f"<unhashable:{type(item).__name__}:{id(item)}>"


def hash_dict(data: Any, strict: bool = False) -> int:
    """Generate stable hash for any data structure including dicts, lists, and Pydantic models.

    Args:
        data: Data to hash (dict, list, BaseModel, or any object)
        strict: If True, deepcopy data before hashing to prevent mutation side effects

    Returns:
        Integer hash value (stable across equivalent structures)

    Raises:
        TypeError: If generated representation is not hashable
    """
    global _INITIALIZED, PydanticBaseModel
    if _INITIALIZED is False:
        from pydantic import BaseModel

        PydanticBaseModel = BaseModel
        _INITIALIZED = True

    data_to_process = data
    if strict:
        data_to_process = copy.deepcopy(data)

    hashable_repr = _generate_hashable_representation(data_to_process)

    try:
        return hash(hashable_repr)
    except TypeError as e:
        raise TypeError(
            f"The generated representation for the input data was not hashable. "
            f"Input type: {type(data).__name__}, Representation type: {type(hashable_repr).__name__}. "
            f"Original error: {e}"
        )

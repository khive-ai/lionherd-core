# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Python 3.11+ type system features in ln.types module.

This module tests:
- typing.Self return types (PEP 673)
- enum.StrEnum behavior (Python 3.11)
- typing.TypeGuard for type narrowing (PEP 647)
- typing.assert_never for exhaustiveness (PEP 484)
- Pickle identity preservation with tuple-form __reduce__
- ClassVar, Final, TypeAlias annotations
"""

from __future__ import annotations

import copy
import json
import pickle
from typing import assert_never

from lionherd_core.types import (
    Enum,
    Undefined,
    UndefinedType,
    Unset,
    UnsetType,
    not_sentinel,
)

# ---------------------------------------------------------------------------
# StrEnum behavior + allowed()
# ---------------------------------------------------------------------------


class SampleStrEnum(Enum):
    """Sample enum using StrEnum base for testing."""

    FOO = "foo"
    BAR = "bar"
    BAZ = "baz"


def test_strenum_string_behavior():
    """StrEnum members should behave like strings."""
    assert str(SampleStrEnum.FOO) == "foo"
    assert SampleStrEnum.FOO == "foo"  # Direct equality with string
    assert SampleStrEnum.FOO == "foo"


def test_strenum_constructor_lookup():
    """StrEnum should support constructor lookup by value."""
    assert SampleStrEnum("foo") is SampleStrEnum.FOO
    assert SampleStrEnum("bar") is SampleStrEnum.BAR


def test_strenum_json_serialization():
    """StrEnum should serialize cleanly to JSON."""
    data = {"key": SampleStrEnum.BAR, "items": [SampleStrEnum.FOO, SampleStrEnum.BAZ]}
    json_str = json.dumps(data)
    assert json_str == '{"key": "bar", "items": ["foo", "baz"]}'


def test_strenum_allowed_method():
    """Custom allowed() method should work with StrEnum."""
    assert SampleStrEnum.allowed() == ("foo", "bar", "baz")
    assert set(SampleStrEnum.allowed()) == {"foo", "bar", "baz"}


def test_strenum_exhaustiveness_check():
    """Demonstrate exhaustiveness checking with assert_never."""

    def evaluate_enum(e: SampleStrEnum) -> int:
        match e:
            case SampleStrEnum.FOO:
                return 1
            case SampleStrEnum.BAR:
                return 2
            case SampleStrEnum.BAZ:
                return 3
            case _:
                # If new member added and not handled, type checker will flag this
                assert_never(e)  # Raises AssertionError if reached at runtime

    assert evaluate_enum(SampleStrEnum.FOO) == 1
    assert evaluate_enum(SampleStrEnum.BAR) == 2
    assert evaluate_enum(SampleStrEnum.BAZ) == 3


# ---------------------------------------------------------------------------
# Sentinels: Self-typed copy/deepcopy + pickle identity
# ---------------------------------------------------------------------------


def test_sentinel_copy_returns_self():
    """copy.copy() should return the same singleton instance."""
    assert copy.copy(Undefined) is Undefined
    assert copy.copy(Unset) is Unset


def test_sentinel_deepcopy_returns_self():
    """copy.deepcopy() should return the same singleton instance."""
    assert copy.deepcopy(Undefined) is Undefined
    assert copy.deepcopy(Unset) is Unset


def test_sentinel_deepcopy_with_memo():
    """Test deepcopy with memo dict preserves identity."""
    memo: dict[int, object] = {}
    result = copy.deepcopy(Undefined, memo)
    # Identity preservation is the key requirement
    assert result is Undefined

    # Deepcopy in nested structure should also preserve identity
    nested = {"data": Undefined}
    nested_copy = copy.deepcopy(nested, memo)
    assert nested_copy["data"] is Undefined


def test_sentinel_pickle_identity():
    """Pickled sentinels should unpickle to the same singleton."""
    undefined_bytes = pickle.dumps(Undefined)
    unset_bytes = pickle.dumps(Unset)

    undefined_restored = pickle.loads(undefined_bytes)
    unset_restored = pickle.loads(unset_bytes)

    # Identity must be preserved
    assert undefined_restored is Undefined
    assert unset_restored is Unset


def test_sentinel_pickle_protocol_versions():
    """Test pickle identity across different protocol versions."""
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        undefined_bytes = pickle.dumps(Undefined, protocol=protocol)
        unset_bytes = pickle.dumps(Unset, protocol=protocol)

        assert pickle.loads(undefined_bytes) is Undefined
        assert pickle.loads(unset_bytes) is Unset


def test_sentinel_nested_pickle():
    """Sentinels in nested structures should preserve identity."""
    data = {
        "values": [1, 2, Undefined, 3],
        "optional": Unset,
        "nested": {"deep": Undefined},
    }

    restored = pickle.loads(pickle.dumps(data))

    assert restored["values"][2] is Undefined
    assert restored["optional"] is Unset
    assert restored["nested"]["deep"] is Undefined


# ---------------------------------------------------------------------------
# TypeGuard: not_sentinel narrows types
# ---------------------------------------------------------------------------


def test_not_sentinel_basic_filtering():
    """not_sentinel should filter out sentinel values."""
    data: list[int | UndefinedType | UnsetType] = [1, 2, Undefined, Unset, 3, 4]
    filtered = [x for x in data if not_sentinel(x)]

    # Runtime behavior
    assert filtered == [1, 2, 3, 4]
    assert Undefined not in filtered
    assert Unset not in filtered


def test_not_sentinel_with_none():
    """not_sentinel should optionally treat None as sentinel."""
    data = [1, None, 2, Undefined, 3]

    # None not treated as sentinel
    filtered_default = [x for x in data if not_sentinel(x)]
    assert None in filtered_default
    assert filtered_default == [1, None, 2, 3]

    # None treated as sentinel
    filtered_none = [x for x in data if not_sentinel(x, none_as_sentinel=True)]
    assert None not in filtered_none
    assert filtered_none == [1, 2, 3]


def test_not_sentinel_with_empty():
    """not_sentinel should optionally treat empty collections as sentinels."""
    data = [1, [], {}, "", 2, Undefined]

    # Empty not treated as sentinel
    filtered_default = [x for x in data if not_sentinel(x)]
    assert [] in filtered_default
    assert filtered_default == [1, [], {}, "", 2]

    # Empty treated as sentinel
    filtered_empty = [x for x in data if not_sentinel(x, empty_as_sentinel=True)]
    assert [] not in filtered_empty
    assert {} not in filtered_empty
    assert "" not in filtered_empty
    assert filtered_empty == [1, 2]


def test_not_sentinel_preserves_types():
    """TypeGuard should allow type checkers to narrow types after filtering."""
    # This test documents the expected type-narrowing behavior
    # Type checkers should infer filtered list is list[int], not list[int | UndefinedType]
    data: list[int | UndefinedType] = [1, 2, Undefined, 3]
    filtered = [x for x in data if not_sentinel(x)]

    # Runtime verification
    assert all(isinstance(x, int) for x in filtered)
    assert filtered == [1, 2, 3]


def test_not_sentinel_dict_filtering():
    """TypeGuard should work with dict comprehensions."""
    data: dict[str, int | UnsetType] = {
        "a": 1,
        "b": Unset,
        "c": 2,
        "d": Unset,
    }

    filtered = {k: v for k, v in data.items() if not_sentinel(v)}

    assert filtered == {"a": 1, "c": 2}
    assert "b" not in filtered
    assert "d" not in filtered


# ---------------------------------------------------------------------------
# Edge cases and behavior validation
# ---------------------------------------------------------------------------


def test_sentinel_boolean_falsy():
    """Sentinels should evaluate to False in boolean context."""
    assert not Undefined
    assert not Unset
    assert bool(Undefined) is False
    assert bool(Unset) is False


def test_sentinel_repr_str():
    """Sentinels should have clean repr/str."""
    assert repr(Undefined) == "Undefined"
    assert str(Undefined) == "Undefined"
    assert repr(Unset) == "Unset"
    assert str(Unset) == "Unset"


def test_sentinel_identity_checks():
    """Sentinels should use identity checks (is), not equality."""
    # These are singletons, identity is guaranteed
    assert Undefined is UndefinedType()
    assert Unset is UnsetType()

    # Multiple instantiation attempts return same singleton
    sentinel1 = UndefinedType()
    sentinel2 = UndefinedType()
    assert sentinel1 is sentinel2
    assert sentinel1 is Undefined


def test_enum_comparison_with_strings():
    """StrEnum should support natural comparisons with strings."""
    assert SampleStrEnum.FOO == "foo"
    assert SampleStrEnum.BAR == "bar"
    assert SampleStrEnum.FOO != "bar"
    assert SampleStrEnum.FOO in ["foo", "baz"]
    assert "foo" in [SampleStrEnum.FOO, SampleStrEnum.BAR]


def test_sentinel_union_syntax():
    """Sentinels should support Python 3.10+ union syntax with | operator."""
    from typing import get_args

    # Test str | Unset (uses __ror__ on Unset)
    result1 = str | Unset
    assert get_args(result1) == (str, UnsetType)

    # Test Unset | str (uses __or__ on Unset)
    result2 = Unset | str
    assert get_args(result2) == (UnsetType, str)

    # Test int | Undefined (uses __ror__ on Undefined)
    result3 = int | Undefined
    assert get_args(result3) == (int, UndefinedType)

    # Test Undefined | int (uses __or__ on Undefined)
    result4 = Undefined | int
    assert get_args(result4) == (UndefinedType, int)

    # Test complex union: str | None | Unset
    result5 = str | None | Unset
    args5 = get_args(result5)
    assert str in args5
    assert type(None) in args5
    assert UnsetType in args5

    # Test in function annotations
    def func(value: str | Unset = Unset) -> str:  # type: ignore[misc]
        if value is Unset:
            return "unset"
        return value

    assert func() == "unset"
    assert func("hello") == "hello"

    # Test singleton identity preserved after union operations
    assert Unset is Unset
    assert Undefined is Undefined

    # Test sentinel-to-sentinel unions (edge cases)
    # Undefined | Unset should produce Union[UndefinedType, UnsetType] (both types)
    result6 = Undefined | Unset
    args6 = get_args(result6)
    assert args6 == (UndefinedType, UnsetType), f"Expected (UndefinedType, UnsetType), got {args6}"
    assert UndefinedType in args6
    assert UnsetType in args6
    # Ensure no sentinel instances leaked into union args
    assert Undefined not in args6
    assert Unset not in args6

    # Unset | Undefined (reverse) should also produce both types
    result7 = Unset | Undefined
    args7 = get_args(result7)
    assert args7 == (UnsetType, UndefinedType), f"Expected (UnsetType, UndefinedType), got {args7}"
    assert UnsetType in args7
    assert UndefinedType in args7
    assert Unset not in args7
    assert Undefined not in args7

    # Test edge case: invalid operands (non-type values) should still work
    # Python's Union handles these gracefully by accepting the value as-is
    result8 = Unset | 123
    args8 = get_args(result8)
    assert UnsetType in args8
    assert 123 in args8  # Union accepts literal integers

    # Test edge case: self-union (type | same type)
    # Python's Union deduplicates these automatically
    result9 = str | str
    args9 = get_args(result9)
    # Union[str, str] collapses to just str, so get_args returns ()
    assert args9 == () or args9 == (str,)  # Python typing behavior varies by version

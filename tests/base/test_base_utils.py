# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionherd.base._utils module.

Tests: to_uuid, coerce_created_at, get_json_serializable, ObservableProto, decorators, type loading.
Focus: Type coercion, error handling, protocol structural typing, sentinel awareness.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from lionherd_core.base._utils import (
    coerce_created_at,
    get_json_serializable,
    to_uuid,
)
from lionherd_core.ln import json_dict
from lionherd_core.protocols import Observable, ObservableProto

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def valid_uuid_str() -> str:
    """Valid UUID string (lowercase with hyphens)."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def valid_uuid_obj() -> UUID:
    """Valid UUID object."""
    return uuid4()


@pytest.fixture
def observable_mock(valid_uuid_obj: UUID):
    """Mock object implementing Observable protocol."""

    class MockObservable:
        @property
        def id(self) -> UUID:
            return valid_uuid_obj

    return MockObservable()


@pytest.fixture
def aware_datetime() -> dt.datetime:
    """Timezone-aware datetime (UTC)."""
    return dt.datetime(2025, 10, 31, 12, 30, 45, tzinfo=dt.UTC)


@pytest.fixture
def naive_datetime() -> dt.datetime:
    """Timezone-naive datetime."""
    return dt.datetime(2025, 10, 31, 12, 30, 45)


@pytest.fixture
def iso_string_with_tz() -> str:
    """ISO 8601 string with timezone."""
    return "2025-10-31T12:30:45+00:00"


@pytest.fixture
def iso_string_without_tz() -> str:
    """ISO 8601 string without timezone."""
    return "2025-10-31T12:30:45"


# ============================================================================
# Tests: synchronized() and async_synchronized() decorators
# ============================================================================


class TestSynchronizedDecorators:
    """Test suite for synchronized() and async_synchronized() decorators."""

    def test_synchronized_when_multiple_calls_then_thread_safe(self):
        """Test synchronized() decorator ensures thread-safe execution."""
        import threading

        from lionherd_core.base._utils import synchronized

        class Counter:
            def __init__(self):
                self._lock = threading.RLock()
                self.count = 0

            @synchronized
            def increment(self):
                # Simulate non-atomic operation
                current = self.count
                current += 1
                self.count = current

        counter = Counter()
        threads = []

        # Create 10 threads that each increment 100 times
        for _ in range(10):
            thread = threading.Thread(target=lambda: [counter.increment() for _ in range(100)])
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # If synchronized works, count should be exactly 1000
        assert counter.count == 1000

    def test_synchronized_when_exception_then_lock_released(self):
        """Test synchronized() releases lock even when exception raised."""
        import threading

        from lionherd_core.base._utils import synchronized

        class FailingOperation:
            def __init__(self):
                self._lock = threading.RLock()
                self.call_count = 0

            @synchronized
            def fail_first_time(self):
                self.call_count += 1
                if self.call_count == 1:
                    raise ValueError("First call fails")
                return "success"

        obj = FailingOperation()

        # First call should fail
        with pytest.raises(ValueError):
            obj.fail_first_time()

        # Second call should succeed (lock was released)
        result = obj.fail_first_time()
        assert result == "success"
        assert obj.call_count == 2

    @pytest.mark.asyncio
    async def test_async_synchronized_when_multiple_calls_then_async_safe(self):
        """Test async_synchronized() decorator ensures async-safe execution."""
        import asyncio

        from lionherd_core.base._utils import async_synchronized

        class AsyncCounter:
            def __init__(self):
                self._async_lock = asyncio.Lock()
                self.count = 0

            @async_synchronized
            async def increment(self):
                # Simulate non-atomic async operation
                current = self.count
                await asyncio.sleep(0.001)  # Yield control
                self.count = current + 1

        counter = AsyncCounter()

        # Create 10 coroutines that each increment 10 times
        tasks = []
        for _ in range(10):
            for _ in range(10):
                tasks.append(counter.increment())

        await asyncio.gather(*tasks)

        # If async_synchronized works, count should be exactly 100
        assert counter.count == 100

    @pytest.mark.asyncio
    async def test_async_synchronized_when_exception_then_lock_released(self):
        """Test async_synchronized() releases lock even when exception raised."""
        import asyncio

        from lionherd_core.base._utils import async_synchronized

        class AsyncFailingOperation:
            def __init__(self):
                self._async_lock = asyncio.Lock()
                self.call_count = 0

            @async_synchronized
            async def fail_first_time(self):
                self.call_count += 1
                if self.call_count == 1:
                    raise ValueError("First call fails")
                return "success"

        obj = AsyncFailingOperation()

        # First call should fail
        with pytest.raises(ValueError):
            await obj.fail_first_time()

        # Second call should succeed (lock was released)
        result = await obj.fail_first_time()
        assert result == "success"
        assert obj.call_count == 2


# ============================================================================
# Tests: load_type_from_string()
# ============================================================================


class TestLoadTypeFromString:
    """Test suite for load_type_from_string() function."""

    def test_load_type_from_string_when_valid_type_then_returns_type(self):
        """Test load_type_from_string() loads valid type from string path."""
        from lionherd_core.base._utils import load_type_from_string

        result = load_type_from_string("lionherd_core.base.element.Element")

        assert result is not None
        assert isinstance(result, type)
        assert result.__name__ == "Element"

    def test_load_type_from_string_when_builtin_type_then_returns_type(self):
        """Test load_type_from_string() loads builtin types."""
        from lionherd_core.base._utils import load_type_from_string

        # Test with dict type
        result = load_type_from_string("builtins.dict")

        assert result is dict
        assert isinstance(result, type)

    def test_load_type_from_string_when_cached_then_returns_from_cache(self):
        """Test load_type_from_string() uses cache for repeated calls."""
        from lionherd_core.base._utils import _TYPE_CACHE, load_type_from_string

        type_str = "lionherd_core.base.node.Node"

        # Clear cache for this type
        _TYPE_CACHE.pop(type_str, None)

        # First call - loads and caches
        result1 = load_type_from_string(type_str)

        # Verify it's in cache
        assert type_str in _TYPE_CACHE

        # Second call - returns from cache
        result2 = load_type_from_string(type_str)

        # Should be exact same object (not just equal)
        assert result1 is result2

    def test_load_type_from_string_when_not_string_then_raises_valueerror(self):
        """Test load_type_from_string() raises ValueError for non-string input."""
        from lionherd_core.base._utils import load_type_from_string

        with pytest.raises(ValueError, match="Expected string, got"):
            load_type_from_string(123)

        with pytest.raises(ValueError, match="Expected string, got"):
            load_type_from_string(None)

    def test_load_type_from_string_when_no_module_path_then_raises_valueerror(self):
        """Test load_type_from_string() raises ValueError when no dot in path."""
        from lionherd_core.base._utils import load_type_from_string

        with pytest.raises(ValueError, match="Invalid type path \\(no module\\)"):
            load_type_from_string("NoModule")

    def test_load_type_from_string_when_invalid_module_then_raises_valueerror(self):
        """Test load_type_from_string() raises ValueError for invalid module."""
        from lionherd_core.base._utils import load_type_from_string

        with pytest.raises(ValueError, match="Failed to load type"):
            load_type_from_string("nonexistent.module.Type")

    def test_load_type_from_string_when_invalid_class_then_raises_valueerror(self):
        """Test load_type_from_string() raises ValueError for invalid class name."""
        from lionherd_core.base._utils import load_type_from_string

        with pytest.raises(ValueError, match="Failed to load type"):
            load_type_from_string("lionherd_core.base.element.NonExistentClass")

    def test_load_type_from_string_when_not_a_type_then_raises_valueerror(self):
        """Test load_type_from_string() raises ValueError when attribute is not a type."""
        from lionherd_core.base._utils import load_type_from_string

        # Try loading a module-level constant (not a type)
        with pytest.raises(ValueError, match="is not a type"):
            load_type_from_string("lionherd_core.base._utils.__all__")

    def test_load_type_from_string_when_import_returns_none_then_raises(self):
        """Test load_type_from_string raises ValueError when importlib returns None (defensive case)."""
        from unittest.mock import patch

        from lionherd_core.base._utils import _TYPE_CACHE, load_type_from_string

        type_str = "fake.module.Type"
        # Clear from cache
        _TYPE_CACHE.pop(type_str, None)

        # Mock importlib.import_module to return None (defensive case)
        # Raises ImportError which gets caught and re-raised as ValueError
        with (
            patch("importlib.import_module", return_value=None),
            pytest.raises(ValueError, match=r"Failed to load type.*Module.*not found"),
        ):
            load_type_from_string(type_str)


# ============================================================================
# Tests: extract_types()
# ============================================================================


class TestExtractTypes:
    """Test suite for extract_types() function."""

    def test_extract_types_when_single_type_then_returns_set(self):
        """Test extract_types() converts single type to set."""
        from lionherd_core.base._utils import extract_types

        result = extract_types(int)

        assert result == {int}
        assert isinstance(result, set)

    def test_extract_types_when_list_of_types_then_returns_set(self):
        """Test extract_types() converts list of types to set."""
        from lionherd_core.base._utils import extract_types

        result = extract_types([int, str, float])

        assert result == {int, str, float}
        assert isinstance(result, set)

    def test_extract_types_when_set_of_types_then_returns_same_set(self):
        """Test extract_types() returns set of types unchanged."""
        from lionherd_core.base._utils import extract_types

        input_set = {int, str, float}
        result = extract_types(input_set)

        assert result == input_set
        assert isinstance(result, set)

    def test_extract_types_when_typing_union_then_extracts_types(self):
        """Test extract_types() extracts types from typing.Union."""
        from typing import Union

        from lionherd_core.base._utils import extract_types

        result = extract_types(Union[int, str])

        assert result == {int, str}
        assert isinstance(result, set)

    def test_extract_types_when_pipe_union_then_extracts_types(self):
        """Test extract_types() extracts types from Python 3.10+ pipe union (int | str)."""
        from lionherd_core.base._utils import extract_types

        # Python 3.10+ pipe syntax
        result = extract_types(int | str)

        assert result == {int, str}
        assert isinstance(result, set)

    def test_extract_types_when_complex_union_then_extracts_all_types(self):
        """Test extract_types() handles complex union with multiple types."""
        from lionherd_core.base._utils import extract_types

        result = extract_types(int | str | float | bool)

        assert result == {int, str, float, bool}
        assert isinstance(result, set)

    def test_extract_types_when_list_with_unions_then_extracts_all(self):
        """Test extract_types() extracts types from list containing unions."""
        from lionherd_core.base._utils import extract_types

        result = extract_types([int | str, float])

        assert result == {int, str, float}
        assert isinstance(result, set)

    def test_extract_types_when_set_with_unions_then_extracts_all(self):
        """Test extract_types() extracts types from set containing unions."""
        from lionherd_core.base._utils import extract_types

        result = extract_types({int | str, float | bool})

        assert result == {int, str, float, bool}
        assert isinstance(result, set)

    def test_extract_types_when_nested_typing_union_in_list_then_extracts_all(self):
        """Test extract_types() handles typing.Union in list."""
        from typing import Union

        from lionherd_core.base._utils import extract_types

        result = extract_types([Union[int, str], float])

        assert result == {int, str, float}
        assert isinstance(result, set)

    def test_extract_types_when_mixed_unions_in_set_then_extracts_all(self):
        """Test extract_types() handles mixed typing.Union and pipe union in set."""
        from typing import Union

        from lionherd_core.base._utils import extract_types

        result = extract_types({Union[int, str], float | bool})

        assert result == {int, str, float, bool}
        assert isinstance(result, set)


# ============================================================================
# Tests: get_element_serializer_config()
# ============================================================================


class TestGetElementSerializerConfig:
    """Test suite for get_element_serializer_config() function."""

    def test_get_element_serializer_config_returns_tuple(self):
        """Test get_element_serializer_config() returns tuple of (order, additional)."""
        from lionherd_core.base._utils import get_element_serializer_config

        result = get_element_serializer_config()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_element_serializer_config_order_contains_serializable_and_basemodel(self):
        """Test get_element_serializer_config() order list contains Serializable and BaseModel."""
        from pydantic import BaseModel

        from lionherd_core.base._utils import get_element_serializer_config
        from lionherd_core.protocols import Serializable

        order, _ = get_element_serializer_config()

        assert isinstance(order, list)
        assert len(order) == 2
        assert order[0] is Serializable
        assert order[1] is BaseModel

    def test_get_element_serializer_config_additional_has_serializers(self):
        """Test get_element_serializer_config() additional dict has serializer functions."""
        from pydantic import BaseModel

        from lionherd_core.base._utils import get_element_serializer_config
        from lionherd_core.protocols import Serializable

        _, additional = get_element_serializer_config()

        assert isinstance(additional, dict)
        assert Serializable in additional
        assert BaseModel in additional
        assert callable(additional[Serializable])
        assert callable(additional[BaseModel])

    def test_get_element_serializer_config_serializable_calls_to_dict(self):
        """Test get_element_serializer_config() Serializable serializer calls to_dict()."""
        from lionherd_core.base._utils import get_element_serializer_config

        _, additional = get_element_serializer_config()
        serializer = additional[next(iter(additional.keys()))]  # Get Serializable serializer

        # Create mock object with to_dict method
        class MockSerializable:
            def to_dict(self):
                return {"test": "data"}

        obj = MockSerializable()
        result = serializer(obj)

        assert result == {"test": "data"}

    def test_get_element_serializer_config_basemodel_calls_model_dump(self):
        """Test get_element_serializer_config() BaseModel serializer calls model_dump()."""
        from pydantic import BaseModel

        from lionherd_core.base._utils import get_element_serializer_config

        _, additional = get_element_serializer_config()
        serializer = additional[BaseModel]

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = serializer(model)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42


# ============================================================================
# Tests: to_uuid()
# ============================================================================


class TestToUuid:
    """Test suite for to_uuid() function."""

    def test_to_uuid_when_uuid_instance_then_returns_same(self, valid_uuid_obj: UUID):
        """Test to_uuid() returns same UUID instance when given UUID object."""
        result = to_uuid(valid_uuid_obj)

        assert result == valid_uuid_obj
        assert isinstance(result, UUID)

    def test_to_uuid_when_valid_string_lowercase_then_returns_uuid(self, valid_uuid_str: str):
        """Test to_uuid() parses lowercase UUID string correctly."""
        result = to_uuid(valid_uuid_str)

        assert isinstance(result, UUID)
        assert str(result) == valid_uuid_str

    def test_to_uuid_when_valid_string_uppercase_then_returns_uuid(self):
        """Test to_uuid() handles uppercase UUID strings."""
        uppercase_uuid = "550E8400-E29B-41D4-A716-446655440000"
        result = to_uuid(uppercase_uuid)

        assert isinstance(result, UUID)
        assert str(result).lower() == uppercase_uuid.lower()

    def test_to_uuid_when_string_without_hyphens_then_returns_uuid(self):
        """Test to_uuid() parses UUID strings without hyphens."""
        no_hyphens = "550e8400e29b41d4a716446655440000"
        result = to_uuid(no_hyphens)

        assert isinstance(result, UUID)
        assert str(result) == "550e8400-e29b-41d4-a716-446655440000"

    def test_to_uuid_when_observable_with_id_then_returns_id(
        self, observable_mock, valid_uuid_obj: UUID
    ):
        """Test to_uuid() extracts .id property from Observable objects."""
        result = to_uuid(observable_mock)

        assert result == valid_uuid_obj
        assert isinstance(result, UUID)

    def test_to_uuid_when_invalid_string_then_raises_valueerror(self):
        """Test to_uuid() raises ValueError for non-UUID strings."""
        with pytest.raises(ValueError):
            to_uuid("not-a-uuid")

    def test_to_uuid_when_empty_string_then_raises_valueerror(self):
        """Test to_uuid() raises ValueError for empty strings."""
        with pytest.raises(ValueError):
            to_uuid("")

    def test_to_uuid_when_int_then_raises_valueerror(self):
        """Test to_uuid() raises ValueError for integers."""
        with pytest.raises(ValueError, match="Cannot get ID from item"):
            to_uuid(12345)

    def test_to_uuid_when_none_then_raises_valueerror(self):
        """Test to_uuid() raises ValueError for None."""
        with pytest.raises(ValueError, match="Cannot get ID from item"):
            to_uuid(None)

    def test_to_uuid_when_dict_then_raises_valueerror(self):
        """Test to_uuid() raises ValueError for dict objects."""
        with pytest.raises(ValueError, match="Cannot get ID from item"):
            to_uuid({"id": "550e8400-e29b-41d4-a716-446655440000"})

    def test_to_uuid_when_object_has_id_method_then_raises_valueerror(self):
        """Test to_uuid() rejects objects where .id is a method, not a property."""

        class ObjectWithIdMethod:
            def id(self) -> UUID:
                return uuid4()

        obj = ObjectWithIdMethod()

        # Object with id() method matches Observable protocol
        assert isinstance(obj, Observable)

        # But to_uuid() should reject it since .id is a method, not a property
        with pytest.raises(ValueError, match=r"Observable\.id must be a property, not a method"):
            to_uuid(obj)


# ============================================================================
# Tests: coerce_created_at()
# ============================================================================


class TestCoerceCreatedAt:
    """Test suite for coerce_created_at() function."""

    def test_coerce_created_at_when_aware_datetime_then_passthrough(
        self, aware_datetime: dt.datetime
    ):
        """Test coerce_created_at() returns aware datetime unchanged."""
        result = coerce_created_at(aware_datetime)

        assert result == aware_datetime
        assert result.tzinfo is not None
        assert result.tzinfo == dt.UTC

    def test_coerce_created_at_when_naive_datetime_then_adds_utc(self, naive_datetime: dt.datetime):
        """Test coerce_created_at() adds UTC timezone to naive datetime."""
        result = coerce_created_at(naive_datetime)

        assert result.tzinfo is not None
        assert result.tzinfo == dt.UTC
        # Same wall time, just with UTC attached
        assert result.replace(tzinfo=None) == naive_datetime

    def test_coerce_created_at_when_iso_string_with_tz_then_parses(self, iso_string_with_tz: str):
        """Test coerce_created_at() parses ISO 8601 string with timezone."""
        result = coerce_created_at(iso_string_with_tz)

        assert isinstance(result, dt.datetime)
        assert result.tzinfo is not None

    def test_coerce_created_at_when_iso_string_without_tz_then_parses(
        self, iso_string_without_tz: str
    ):
        """Test coerce_created_at() parses ISO 8601 string without timezone."""
        result = coerce_created_at(iso_string_without_tz)

        assert isinstance(result, dt.datetime)
        # fromisoformat() preserves naive state, but user can convert with UTC after

    def test_coerce_created_at_when_different_timezone_then_preserves(self):
        """Test coerce_created_at() preserves non-UTC timezones."""
        # Eastern time (UTC-5)
        eastern = dt.datetime(2025, 10, 31, 12, 30, 45, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
        result = coerce_created_at(eastern)

        assert result == eastern
        assert result.tzinfo == dt.timezone(dt.timedelta(hours=-5))

    def test_coerce_created_at_when_invalid_string_then_raises_valueerror(self):
        """Test coerce_created_at() raises ValueError for invalid date strings."""
        with pytest.raises(ValueError):
            coerce_created_at("not-a-date")

    def test_coerce_created_at_accepts_int_timestamp(self):
        """Test coerce_created_at() accepts integer Unix timestamps."""
        timestamp = 1609459200  # 2021-01-01 00:00:00 UTC
        result = coerce_created_at(timestamp)
        assert isinstance(result, dt.datetime)
        assert result.tzinfo == dt.UTC
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1

    def test_coerce_created_at_accepts_float_timestamp(self):
        """Test coerce_created_at() accepts float Unix timestamps."""
        timestamp = 1609459200.123  # 2021-01-01 00:00:00.123 UTC
        result = coerce_created_at(timestamp)
        assert isinstance(result, dt.datetime)
        assert result.tzinfo == dt.UTC

    def test_coerce_created_at_accepts_string_timestamp(self):
        """Test coerce_created_at() accepts string Unix timestamps."""
        result = coerce_created_at("1609459200")
        assert isinstance(result, dt.datetime)
        assert result.tzinfo == dt.UTC
        assert result.year == 2021

    def test_coerce_created_at_when_none_then_raises_valueerror(self):
        """Test coerce_created_at() raises ValueError for None."""
        with pytest.raises(ValueError, match="created_at must be datetime, timestamp"):
            coerce_created_at(None)

    @pytest.mark.parametrize(
        "iso_string,expected_tz",
        [
            ("2025-10-31T12:30:45Z", dt.UTC),
            ("2025-10-31T12:30:45+00:00", dt.UTC),
            ("2025-10-31T12:30:45-05:00", dt.timezone(dt.timedelta(hours=-5))),
            ("2025-10-31T12:30:45+08:00", dt.timezone(dt.timedelta(hours=8))),
        ],
    )
    def test_coerce_created_at_when_various_timezones_then_parses_correctly(
        self, iso_string: str, expected_tz: dt.tzinfo
    ):
        """Test coerce_created_at() handles various timezone formats."""
        result = coerce_created_at(iso_string)

        assert isinstance(result, dt.datetime)
        assert result.tzinfo == expected_tz


# ============================================================================
# Tests: get_json_serializable()
# ============================================================================


class TestGetJsonSerializable:
    """Test suite for get_json_serializable() function."""

    def test_get_json_serializable_when_unset_then_returns_unset(self):
        """Test get_json_serializable() preserves Unset sentinel."""
        from lionherd_core.types._sentinel import Unset

        result = get_json_serializable(Unset)

        assert result is Unset

    @pytest.mark.parametrize(
        "simple_value",
        [
            "string",
            b"bytes",
            bytearray(b"bytearray"),
            42,
            3.14,
            None,
        ],
    )
    def test_get_json_serializable_when_simple_types_then_passthrough(self, simple_value: Any):
        """Test get_json_serializable() returns simple types unchanged."""
        result = get_json_serializable(simple_value)

        assert result == simple_value

    def test_get_json_serializable_when_enum_then_passthrough(self):
        """Test get_json_serializable() handles Enum values."""

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = get_json_serializable(Color.RED)

        assert result == Color.RED

    def test_get_json_serializable_when_json_serializable_dict_then_passthrough(self):
        """Test get_json_serializable() passes through JSON-serializable dicts."""
        data = {"key": "value", "nested": {"number": 42}}
        result = get_json_serializable(data)

        assert result == data

    def test_get_json_serializable_when_json_serializable_list_then_passthrough(self):
        """Test get_json_serializable() passes through JSON-serializable lists."""
        data = [1, 2, 3, "four", {"five": 5}]
        result = get_json_serializable(data)

        assert result == data

    def test_get_json_serializable_when_pydantic_model_then_passthrough(self):
        """Test get_json_serializable() passes through Pydantic models (JSON serializable)."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = get_json_serializable(model)

        # Pydantic models pass through as-is (they're JSON serializable)
        assert result == model
        assert isinstance(result, TestModel)

    def test_get_json_serializable_when_nested_pydantic_models_then_passthrough(self):
        """Test get_json_serializable() passes through nested Pydantic models."""

        class InnerModel(BaseModel):
            inner_value: str

        class OuterModel(BaseModel):
            outer_value: int
            inner: InnerModel

        model = OuterModel(outer_value=42, inner=InnerModel(inner_value="nested"))
        result = get_json_serializable(model)

        # Nested Pydantic models are JSON serializable, pass through as-is
        assert result == model
        assert isinstance(result, OuterModel)

    def test_get_json_serializable_when_pydantic_model_with_enum_then_passthrough(self):
        """Test get_json_serializable() passes through Pydantic models with enums."""

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class ModelWithEnum(BaseModel):
            status: Status

        model = ModelWithEnum(status=Status.ACTIVE)
        result = get_json_serializable(model)

        # Pydantic models with enums are JSON serializable, pass through
        assert result == model
        assert isinstance(result, ModelWithEnum)

    def test_get_json_serializable_when_unserializable_object_then_attempts_conversion(self):
        """Test get_json_serializable() converts complex objects via to_dict() (lambdas → empty dicts)."""
        from lionherd_core.types._sentinel import Unset

        class ComplexObject:
            def __init__(self):
                self.ref = lambda: None  # Functions are not JSON serializable

        obj = ComplexObject()
        result = get_json_serializable(obj)

        # to_dict() successfully converts lambdas to empty dicts
        # Result should be a dict (converted), not the original object
        assert isinstance(result, dict)
        assert result == {"ref": {}}  # Lambda converted to empty dict

    def test_get_json_serializable_when_circular_reference_then_returns_unset(self):
        """Test get_json_serializable() returns Unset for circular references."""
        from lionherd_core.types._sentinel import Unset

        class CircularObject:
            def __init__(self):
                self.ref = self

        obj = CircularObject()
        result = get_json_serializable(obj)

        assert result is Unset

    def test_get_json_serializable_when_bytes_in_simple_types_then_passthrough(self):
        """Test get_json_serializable() handles bytes as simple type."""
        data = b"binary data"
        result = get_json_serializable(data)

        assert result == data
        assert isinstance(result, bytes)


# ============================================================================
# Tests: json_dict()
# ============================================================================


class TestJsonDict:
    """Test suite for json_dict() function."""

    def test_json_dict_when_simple_dict_then_returns_dict(self):
        """Test json_dict() performs round-trip serialization on simple dict."""
        data = {"name": "test", "value": 42, "active": True}
        result = json_dict(data)

        assert result == data
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["active"] is True

    def test_json_dict_when_pydantic_model_then_returns_dict(self):
        """Test json_dict() converts Pydantic model to dict via round-trip serialization."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = json_dict(model)

        # Round-trip should produce a dict, not a Pydantic model
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_json_dict_when_nested_structures_then_preserves_structure(self):
        """Test json_dict() handles nested dicts and lists correctly."""
        data = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            "metadata": {"count": 2, "active": True},
        }
        result = json_dict(data)

        assert result == data
        assert isinstance(result, dict)
        assert len(result["users"]) == 2
        assert result["users"][0]["name"] == "Alice"

    def test_json_dict_when_uuid_then_converts_to_string(self):
        """Test json_dict() converts UUID to string during serialization."""
        test_uuid = uuid4()
        data = {"id": test_uuid, "name": "test"}
        result = json_dict(data)

        # UUID should be serialized as string
        assert isinstance(result["id"], str)
        assert result["id"] == str(test_uuid)
        assert result["name"] == "test"

    def test_json_dict_when_datetime_then_converts_to_string(self):
        """Test json_dict() converts datetime to ISO format string."""
        test_dt = dt.datetime(2025, 10, 31, 12, 30, 45, tzinfo=dt.UTC)
        data = {"timestamp": test_dt, "event": "test"}
        result = json_dict(data)

        # Datetime should be serialized as ISO string
        assert isinstance(result["timestamp"], str)
        assert result["event"] == "test"


# ============================================================================
# Tests: ObservableProto Protocol
# ============================================================================


class TestObservableProto:
    """Test suite for ObservableProto structural protocol."""

    def test_observable_proto_when_object_has_id_property_then_isinstance_true(self):
        """Test ObservableProto matches objects with .id property and serialization methods."""

        class HasIdProperty:
            @property
            def id(self) -> UUID:
                return uuid4()

            def to_dict(self, **kwargs):
                return {"id": str(self.id)}

            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        obj = HasIdProperty()

        assert isinstance(obj, ObservableProto)
        assert isinstance(obj, Observable)  # Alias

    def test_observable_proto_when_object_without_id_then_isinstance_false(self):
        """Test ObservableProto rejects objects without .id property."""

        class NoId:
            pass

        obj = NoId()

        assert not isinstance(obj, ObservableProto)

    def test_observable_proto_when_object_has_id_method_then_isinstance_true(self):
        """Test ObservableProto accepts objects with id() method (runtime_checkable doesn't distinguish methods/properties)."""

        class HasIdMethod:
            def id(self) -> UUID:
                return uuid4()

            def to_dict(self, **kwargs):
                return {"id": str(self.id())}

            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        obj = HasIdMethod()

        # Methods also match the protocol
        assert isinstance(obj, ObservableProto)

    def test_observable_proto_when_object_has_id_attribute_then_isinstance_true(self):
        """Test ObservableProto matches objects with .id attribute and serialization methods."""

        class HasIdAttribute:
            def __init__(self):
                self.id = uuid4()

            def to_dict(self, **kwargs):
                return {"id": str(self.id)}

            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        obj = HasIdAttribute()

        # Python's runtime_checkable protocol checks for attribute existence
        assert isinstance(obj, ObservableProto)

    def test_observable_proto_when_pydantic_model_with_id_then_isinstance_true(self):
        """Test ObservableProto matches Pydantic models with id field and serialization."""

        class ModelWithId(BaseModel):
            id: UUID

            def to_dict(self, **kwargs):
                return {"id": str(self.id)}

            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls(**data)

        model = ModelWithId(id=uuid4())

        assert isinstance(model, ObservableProto)

    def test_observable_alias_is_same_as_protocol(self):
        """Test Observable is an alias for ObservableProto."""
        assert Observable is ObservableProto


# ============================================================================
# Integration Tests
# ============================================================================


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_to_uuid_with_pydantic_model_implementing_observable(self):
        """Test to_uuid() works with Pydantic models that implement Observable."""

        class Event(BaseModel):
            id: UUID
            name: str

            def to_dict(self, **kwargs):
                return {"id": str(self.id), "name": self.name}

            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls(**data)

        event = Event(id=uuid4(), name="test_event")

        # Pydantic models with .id field and serialization methods match Observable protocol
        assert isinstance(event, Observable)

        # to_uuid() should extract the id
        result = to_uuid(event)
        assert result == event.id

    def test_coerce_created_at_with_get_json_serializable(self, aware_datetime: dt.datetime):
        """Test datetime coercion followed by JSON serialization."""
        # Coerce to datetime
        dt_result = coerce_created_at(aware_datetime)

        # Then serialize (datetime is JSON serializable via json_dumpb)
        json_result = get_json_serializable(dt_result)

        # Should preserve datetime (json_dumpb handles it)
        assert json_result == dt_result

    def test_full_workflow_observable_with_created_at(self):
        """Test complete workflow: Observable with UUID and timestamp."""

        class TrackedEvent(BaseModel):
            id: UUID
            created_at: dt.datetime

        # Create event with naive datetime
        naive_dt = dt.datetime(2025, 10, 31, 12, 0, 0)
        event = TrackedEvent(id=uuid4(), created_at=naive_dt)

        # Extract UUID
        event_id = to_uuid(event)
        assert event_id == event.id

        # Coerce timestamp to UTC
        utc_timestamp = coerce_created_at(event.created_at)
        assert utc_timestamp.tzinfo == dt.UTC

        # Serialize entire event (Pydantic models are JSON serializable)
        serialized = get_json_serializable(event)
        # Pydantic model passes through as-is (it's JSON serializable)
        assert serialized == event
        assert isinstance(serialized, TrackedEvent)

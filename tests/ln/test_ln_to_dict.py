# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.ln._to_dict import to_dict


class TestToDictBasic:
    """Test basic to_dict functionality."""

    def test_dict_input(self):
        """Test basic dict input passes through."""
        d = {"a": 1, "b": 2}
        result = to_dict(d)
        assert result == {"a": 1, "b": 2}

    def test_none_input(self):
        """Test None returns empty dict."""
        result = to_dict(None)
        assert result == {}

    def test_empty_string_suppressed(self):
        """Test empty string returns empty dict (always suppressed)."""
        result = to_dict("")
        assert result == {}


class TestToDictFuzzyParsing:
    """Test fuzzy JSON parsing functionality."""

    def test_fuzzy_parse_with_kwargs(self):
        """Regression test for issue #107: to_dict with fuzzy_parse=True and kwargs should not crash.

        Bug: to_dict() was calling fuzzy_json(s, **kwargs) but fuzzy_json has
        positional-only signature: def fuzzy_json(str_to_parse: str, /)
        This caused TypeError when additional kwargs were passed.

        Fix: Changed to fuzzy_json(s) without passing kwargs.
        """
        # Malformed JSON string with trailing commas and unquoted keys
        malformed_json = '{malformed: "value1", trailing: "value2",}'

        # Call to_dict with fuzzy_parse=True AND additional kwargs
        # This should NOT crash with TypeError
        result = to_dict(
            malformed_json,
            fuzzy_parse=True,
            recursive=True,
            max_recursive_depth=5,
        )

        # Verify fuzzy parsing worked correctly
        assert isinstance(result, dict)
        assert "malformed" in result
        assert result["malformed"] == "value1"
        assert "trailing" in result
        assert result["trailing"] == "value2"

    def test_fuzzy_parse_without_kwargs(self):
        """Test fuzzy parsing works without additional kwargs."""
        malformed_json = "{key: 'value'}"
        result = to_dict(malformed_json, fuzzy_parse=True)

        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_fuzzy_parse_with_single_quotes(self):
        """Test fuzzy parsing handles single quotes."""
        json_str = "{'name': 'test', 'value': 42}"
        result = to_dict(json_str, fuzzy_parse=True)

        assert result["name"] == "test"
        assert result["value"] == 42

    def test_fuzzy_parse_disabled(self):
        """Test that malformed JSON fails when fuzzy_parse=False."""
        malformed_json = "{malformed: 'value'}"

        with pytest.raises(Exception):  # orjson.JSONDecodeError or similar
            to_dict(malformed_json, fuzzy_parse=False)


class TestToDictRecursive:
    """Test recursive processing functionality."""

    def test_recursive_with_nested_json_strings(self):
        """Test recursive parsing of nested JSON strings."""
        nested = {"outer": '{"inner": "value"}'}
        result = to_dict(nested, recursive=True, fuzzy_parse=True)

        assert isinstance(result["outer"], dict)
        assert result["outer"]["inner"] == "value"

    def test_max_recursive_depth(self):
        """Test max_recursive_depth parameter."""
        # Nested structure
        nested = {"level1": {"level2": {"level3": "value"}}}
        result = to_dict(nested, recursive=True, max_recursive_depth=2)

        assert isinstance(result, dict)
        assert "level1" in result

    def test_recursive_depth_validation(self):
        """Test max_recursive_depth validation."""
        # Test negative depth raises ValueError
        with pytest.raises(ValueError, match="non-negative integer"):
            to_dict({"a": 1}, recursive=True, max_recursive_depth=-1)

        # Test depth > 10 raises ValueError
        with pytest.raises(ValueError, match="less than or equal to 10"):
            to_dict({"a": 1}, recursive=True, max_recursive_depth=11)


class TestToDictSuppress:
    """Test error suppression functionality."""

    def test_suppress_true(self):
        """Test suppress=True returns empty dict on errors."""
        # Invalid input that would normally raise
        result = to_dict(object(), suppress=True)
        assert result == {}

    def test_suppress_false(self):
        """Test suppress=False raises on errors."""
        with pytest.raises(Exception):
            to_dict(object(), suppress=False)


class TestToDictEdgeCases:
    """Test edge cases and special scenarios."""

    def test_set_input(self):
        """Test set input converts to dict."""
        s = {1, 2, 3}
        result = to_dict(s)
        assert isinstance(result, dict)
        # Set converts to {v: v for v in set}
        assert result == {1: 1, 2: 2, 3: 3}

    def test_list_input(self):
        """Test list input converts to enumerated dict."""
        lst = ["a", "b", "c"]
        result = to_dict(lst)
        assert result == {0: "a", 1: "b", 2: "c"}

    def test_tuple_input(self):
        """Test tuple input converts to enumerated dict."""
        tpl = ("x", "y", "z")
        result = to_dict(tpl)
        assert result == {0: "x", 1: "y", 2: "z"}

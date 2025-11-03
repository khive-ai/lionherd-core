# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from lionherd_core.libs.schema_handlers._function_call_parser import (
    map_positional_args,
    nest_arguments_by_schema,
    parse_function_call,
)


class TestParseFunctionCall:
    """Test parse_function_call function."""

    def test_simple_positional_arg(self):
        """Test parsing simple positional argument."""
        result = parse_function_call('search("AI news")')

        assert result["tool"] == "search"
        assert result["arguments"]["_pos_0"] == "AI news"

    def test_multiple_positional_args(self):
        """Test parsing multiple positional arguments."""
        result = parse_function_call('create_user("john@example.com", "password123")')

        assert result["tool"] == "create_user"
        assert result["arguments"]["_pos_0"] == "john@example.com"
        assert result["arguments"]["_pos_1"] == "password123"

    def test_keyword_arguments(self):
        """Test parsing keyword arguments."""
        result = parse_function_call('search(query="AI news", limit=10)')

        assert result["tool"] == "search"
        assert result["arguments"]["query"] == "AI news"
        assert result["arguments"]["limit"] == 10

    def test_mixed_positional_and_keyword(self):
        """Test parsing mixed positional and keyword arguments."""
        result = parse_function_call('search("AI news", limit=10, enabled=True)')

        assert result["tool"] == "search"
        assert result["arguments"]["_pos_0"] == "AI news"
        assert result["arguments"]["limit"] == 10
        assert result["arguments"]["enabled"] is True

    def test_list_arguments(self):
        """Test parsing list/array arguments."""
        result = parse_function_call('search(query="AI", categories=["company", "news"])')

        assert result["tool"] == "search"
        assert result["arguments"]["query"] == "AI"
        assert result["arguments"]["categories"] == ["company", "news"]

    def test_dict_arguments(self):
        """Test parsing dictionary arguments."""
        result = parse_function_call(
            'search(query="AI", filters={"date": "2024-01-01", "domain": "tech"})'
        )

        assert result["tool"] == "search"
        assert result["arguments"]["query"] == "AI"
        assert result["arguments"]["filters"] == {
            "date": "2024-01-01",
            "domain": "tech",
        }

    def test_integer_arguments(self):
        """Test parsing integer arguments."""
        result = parse_function_call("search(limit=100, offset=50)")

        assert result["arguments"]["limit"] == 100
        assert result["arguments"]["offset"] == 50

    def test_float_arguments(self):
        """Test parsing float arguments."""
        result = parse_function_call("search(threshold=0.75, score=9.5)")

        assert result["arguments"]["threshold"] == 0.75
        assert result["arguments"]["score"] == 9.5

    def test_boolean_arguments(self):
        """Test parsing boolean arguments."""
        result = parse_function_call("search(enabled=True, verbose=False)")

        assert result["arguments"]["enabled"] is True
        assert result["arguments"]["verbose"] is False

    def test_none_arguments(self):
        """Test parsing None/null arguments."""
        result = parse_function_call("search(filter=None, tag=None)")

        assert result["arguments"]["filter"] is None
        assert result["arguments"]["tag"] is None

    def test_empty_function_call(self):
        """Test parsing function call with no arguments."""
        result = parse_function_call("get_status()")

        assert result["tool"] == "get_status"
        assert result["arguments"] == {}

    def test_nested_lists(self):
        """Test parsing nested list structures."""
        result = parse_function_call("search(data=[[1, 2], [3, 4]])")

        assert result["arguments"]["data"] == [[1, 2], [3, 4]]

    def test_nested_dicts(self):
        """Test parsing nested dictionary structures."""
        result = parse_function_call('search(config={"api": {"key": "abc", "url": "example.com"}})')

        assert result["arguments"]["config"] == {"api": {"key": "abc", "url": "example.com"}}

    def test_string_with_quotes(self):
        """Test parsing strings containing quotes."""
        result = parse_function_call('search("He said \\"hello\\"")')

        assert result["arguments"]["_pos_0"] == 'He said "hello"'

    def test_multiline_string(self):
        """Test parsing multiline strings."""
        call_str = """search(query="Line 1\\nLine 2")"""
        result = parse_function_call(call_str)

        assert "Line 1\nLine 2" in result["arguments"]["query"]

    def test_attribute_function_call(self):
        """Test parsing method-style calls (client.search)."""
        result = parse_function_call('client.search("query")')

        assert result["tool"] == "search"  # Should extract method name
        assert result["arguments"]["_pos_0"] == "query"

    def test_unicode_strings(self):
        """Test parsing Unicode strings."""
        result = parse_function_call('search("日本語検索")')

        assert result["arguments"]["_pos_0"] == "日本語検索"

    def test_special_characters_in_strings(self):
        """Test parsing strings with special characters."""
        result = parse_function_call('search("query with @#$% chars")')

        assert result["arguments"]["_pos_0"] == "query with @#$% chars"


class TestParseFunctionCallErrors:
    """Test error handling in parse_function_call."""

    def test_invalid_syntax_raises_error(self):
        """Test that invalid syntax raises ValueError."""
        with pytest.raises(ValueError, match="Invalid function call syntax"):
            parse_function_call("not a function call")

    def test_missing_closing_paren_raises_error(self):
        """Test that missing closing parenthesis raises error."""
        with pytest.raises(ValueError):
            parse_function_call('search("query"')

    def test_missing_opening_paren_raises_error(self):
        """Test that missing opening parenthesis raises error."""
        with pytest.raises(ValueError):
            parse_function_call('search"query")')

    def test_kwargs_syntax_raises_error(self):
        """Test that **kwargs syntax raises error."""
        with pytest.raises(ValueError, match="not supported"):
            parse_function_call("search(**options)")

    def test_empty_string_raises_error(self):
        """Test that empty string raises error."""
        with pytest.raises(ValueError):
            parse_function_call("")

    def test_just_function_name_raises_error(self):
        """Test that function name without parens raises error."""
        with pytest.raises(ValueError):
            parse_function_call("search")

    def test_invalid_expression_raises_error(self):
        """Test that invalid expressions raise error."""
        with pytest.raises(ValueError):
            parse_function_call("1 + 2")  # Not a function call


class TestMapPositionalArgs:
    """Test map_positional_args function."""

    def test_map_single_positional(self):
        """Test mapping single positional argument."""
        arguments = {"_pos_0": "value"}
        param_names = ["query"]

        result = map_positional_args(arguments, param_names)

        assert result == {"query": "value"}

    def test_map_multiple_positional(self):
        """Test mapping multiple positional arguments."""
        arguments = {"_pos_0": "john@example.com", "_pos_1": "password123"}
        param_names = ["email", "password"]

        result = map_positional_args(arguments, param_names)

        assert result == {"email": "john@example.com", "password": "password123"}

    def test_map_preserves_keyword_args(self):
        """Test that keyword arguments are preserved."""
        arguments = {"_pos_0": "query", "limit": 10, "enabled": True}
        param_names = ["query"]

        result = map_positional_args(arguments, param_names)

        assert result == {"query": "query", "limit": 10, "enabled": True}

    def test_map_mixed_positional_and_keyword(self):
        """Test mapping mixed positional and keyword arguments."""
        arguments = {"_pos_0": "value1", "_pos_1": "value2", "extra": "value3"}
        param_names = ["param1", "param2"]

        result = map_positional_args(arguments, param_names)

        assert result == {
            "param1": "value1",
            "param2": "value2",
            "extra": "value3",
        }

    def test_map_no_positional_args(self):
        """Test mapping when there are no positional arguments."""
        arguments = {"query": "value", "limit": 10}
        param_names = ["query", "limit"]

        result = map_positional_args(arguments, param_names)

        assert result == {"query": "value", "limit": 10}

    def test_map_too_many_positional_raises_error(self):
        """Test that too many positional args raises error."""
        arguments = {"_pos_0": "val1", "_pos_1": "val2", "_pos_2": "val3"}
        param_names = ["param1", "param2"]  # Only 2 params

        with pytest.raises(ValueError, match="Too many positional arguments"):
            map_positional_args(arguments, param_names)

    def test_map_empty_arguments(self):
        """Test mapping empty arguments dict."""
        arguments = {}
        param_names = ["query"]

        result = map_positional_args(arguments, param_names)

        assert result == {}

    def test_map_preserves_argument_order(self):
        """Test that argument order is preserved."""
        arguments = {
            "_pos_0": "first",
            "_pos_1": "second",
            "_pos_2": "third",
        }
        param_names = ["param1", "param2", "param3"]

        result = map_positional_args(arguments, param_names)

        assert list(result.keys()) == ["param1", "param2", "param3"]
        assert list(result.values()) == ["first", "second", "third"]


class TestIntegrationParsing:
    """Integration tests combining parsing and mapping."""

    def test_full_workflow_positional(self):
        """Test full workflow from parsing to mapping positional args."""
        call_str = 'search("AI news", 10)'
        parsed = parse_function_call(call_str)
        param_names = ["query", "limit"]

        mapped = map_positional_args(parsed["arguments"], param_names)

        assert mapped == {"query": "AI news", "limit": 10}

    def test_full_workflow_mixed(self):
        """Test full workflow with mixed argument styles."""
        call_str = 'search("AI news", limit=10, enabled=True)'
        parsed = parse_function_call(call_str)
        param_names = ["query"]

        mapped = map_positional_args(parsed["arguments"], param_names)

        assert mapped == {"query": "AI news", "limit": 10, "enabled": True}

    def test_full_workflow_keyword_only(self):
        """Test full workflow with keyword-only arguments."""
        call_str = 'search(query="AI news", limit=10)'
        parsed = parse_function_call(call_str)
        param_names = ["query", "limit"]

        mapped = map_positional_args(parsed["arguments"], param_names)

        assert mapped == {"query": "AI news", "limit": 10}

    def test_complex_nested_structures(self):
        """Test parsing and mapping complex nested structures."""
        call_str = 'search("query", filters={"tags": ["a", "b"], "date": "2024-01-01"})'
        parsed = parse_function_call(call_str)
        param_names = ["query"]

        mapped = map_positional_args(parsed["arguments"], param_names)

        assert mapped["query"] == "query"
        assert mapped["filters"] == {"tags": ["a", "b"], "date": "2024-01-01"}

    def test_unsupported_function_type(self):
        """Test that unsupported function types raise error."""
        # Create a call with unsupported function type (not Name or Attribute)
        call_str = "(lambda x: x)(42)"  # Lambda call

        with pytest.raises(ValueError, match="Unsupported function type"):
            parse_function_call(call_str)


class TestNestArgumentsBySchema:
    """Test nest_arguments_by_schema function."""

    def test_no_schema_returns_unchanged(self):
        """Test that no schema returns arguments unchanged."""
        arguments = {"query": "test", "limit": 10}

        result = nest_arguments_by_schema(arguments, None)

        assert result == arguments

    def test_schema_without_model_fields_returns_unchanged(self):
        """Test that schema without model_fields returns unchanged."""
        arguments = {"query": "test", "limit": 10}

        class NoFieldsClass:
            pass

        result = nest_arguments_by_schema(arguments, NoFieldsClass)

        assert result == arguments

    def test_flat_arguments_no_nesting_needed(self):
        """Test flat arguments when all are top-level fields."""

        class SearchSchema(BaseModel):
            query: str
            limit: int
            enabled: bool

        arguments = {"query": "test", "limit": 10, "enabled": True}

        result = nest_arguments_by_schema(arguments, SearchSchema)

        # Should remain unchanged since all are top-level
        assert result == arguments

    def test_nested_pydantic_model(self):
        """Test nesting with Pydantic model field."""

        class FilterOptions(BaseModel):
            date: str
            domain: str

        class SearchSchema(BaseModel):
            query: str
            filters: FilterOptions

        # Flat arguments
        arguments = {"query": "test", "date": "2024-01-01", "domain": "example.com"}

        result = nest_arguments_by_schema(arguments, SearchSchema)

        # Should nest date and domain under filters
        assert result == {
            "query": "test",
            "filters": {"date": "2024-01-01", "domain": "example.com"},
        }

    def test_union_type_field(self):
        """Test nesting with union type field."""

        class OptionA(BaseModel):
            field_a: str

        class OptionB(BaseModel):
            field_b: int

        class SearchSchema(BaseModel):
            query: str
            options: OptionA | OptionB

        # Arguments that match OptionA
        arguments = {"query": "test", "field_a": "value"}

        result = nest_arguments_by_schema(arguments, SearchSchema)

        # Should nest field_a under options
        assert result == {"query": "test", "options": {"field_a": "value"}}

    def test_multiple_nested_fields(self):
        """Test multiple nested field structures."""

        class Auth(BaseModel):
            token: str
            expires: int

        class Pagination(BaseModel):
            limit: int
            offset: int

        class SearchSchema(BaseModel):
            query: str
            auth: Auth
            pagination: Pagination

        arguments = {"query": "test", "token": "abc123", "expires": 3600, "limit": 10, "offset": 0}

        result = nest_arguments_by_schema(arguments, SearchSchema)

        # Should nest auth and pagination separately
        assert result == {
            "query": "test",
            "auth": {"token": "abc123", "expires": 3600},
            "pagination": {"limit": 10, "offset": 0},
        }

    def test_unknown_field_kept_at_top_level(self):
        """Test that unknown fields are kept at top level."""

        class FilterOptions(BaseModel):
            date: str

        class SearchSchema(BaseModel):
            query: str
            filters: FilterOptions

        arguments = {"query": "test", "date": "2024-01-01", "unknown_field": "value"}

        result = nest_arguments_by_schema(arguments, SearchSchema)

        # Should nest date under filters, keep unknown at top
        assert result == {
            "query": "test",
            "filters": {"date": "2024-01-01"},
            "unknown_field": "value",
        }

    def test_mixed_top_level_and_nested(self):
        """Test mix of top-level and nested arguments."""

        class Metadata(BaseModel):
            version: str
            timestamp: str

        class SearchSchema(BaseModel):
            query: str
            enabled: bool
            metadata: Metadata

        arguments = {"query": "test", "enabled": True, "version": "1.0", "timestamp": "2024-01-01"}

        result = nest_arguments_by_schema(arguments, SearchSchema)

        # Should keep query and enabled at top, nest version/timestamp
        assert result == {
            "query": "test",
            "enabled": True,
            "metadata": {"version": "1.0", "timestamp": "2024-01-01"},
        }

    def test_empty_arguments(self):
        """Test with empty arguments dict."""

        class SearchSchema(BaseModel):
            query: str

        arguments = {}

        result = nest_arguments_by_schema(arguments, SearchSchema)

        assert result == {}

    def test_no_nested_fields_in_schema(self):
        """Test schema with only simple types (no nesting)."""

        class SimpleSchema(BaseModel):
            query: str
            limit: int
            enabled: bool

        arguments = {"query": "test", "limit": 10, "enabled": True}

        result = nest_arguments_by_schema(arguments, SimpleSchema)

        # Should return unchanged
        assert result == arguments

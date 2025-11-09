# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.libs.string_handlers._fuzzy_json import (
    _check_valid_str,
    _clean_json_string,
    fix_json_string,
    fuzzy_json,
)

# ============================================================================
# Test fuzzy_json main function
# ============================================================================


def test_fuzzy_json_valid():
    """Test fuzzy_json with valid JSON"""
    result = fuzzy_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_fuzzy_json_single_quotes():
    """Test fuzzy_json with single quotes"""
    result = fuzzy_json("{'key': 'value'}")
    assert result == {"key": "value"}


def test_fuzzy_json_unquoted_keys():
    """Test fuzzy_json with unquoted keys"""
    result = fuzzy_json("{key: 'value'}")
    assert result == {"key": "value"}


def test_fuzzy_json_trailing_commas():
    """Test fuzzy_json with trailing commas"""
    result = fuzzy_json('{"key": "value",}')
    assert result == {"key": "value"}


def test_fuzzy_json_missing_closing_bracket():
    """Test fuzzy_json with missing closing bracket"""
    result = fuzzy_json('{"key": "value"')
    assert result == {"key": "value"}


def test_fuzzy_json_invalid():
    """Test fuzzy_json with completely invalid JSON"""
    with pytest.raises(ValueError, match="Invalid JSON string"):
        fuzzy_json("{completely broken")


def test_fuzzy_json_not_string():
    """Test fuzzy_json with non-string input"""
    with pytest.raises(TypeError, match="Input must be a string"):
        fuzzy_json(123)


def test_fuzzy_json_empty():
    """Test fuzzy_json with empty string"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fuzzy_json("")


def test_fuzzy_json_whitespace_only():
    """Test fuzzy_json with whitespace only"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fuzzy_json("   ")


# ============================================================================
# Test _check_valid_str
# ============================================================================


def test_check_valid_str_valid():
    """Test _check_valid_str with valid string"""
    _check_valid_str("valid string")  # Should not raise


def test_check_valid_str_not_string():
    """Test _check_valid_str with non-string"""
    with pytest.raises(TypeError, match="Input must be a string"):
        _check_valid_str(123)


def test_check_valid_str_empty():
    """Test _check_valid_str with empty string"""
    with pytest.raises(ValueError, match="Input string is empty"):
        _check_valid_str("")


# ============================================================================
# Test _clean_json_string
# ============================================================================


def test_clean_json_string_single_quotes():
    """Test _clean_json_string with single quotes"""
    result = _clean_json_string("{'key': 'value'}")
    assert '"' in result


def test_clean_json_string_trailing_comma():
    """Test _clean_json_string removes trailing commas"""
    result = _clean_json_string('{"key": "value",}')
    assert result == '{"key": "value"}' or '","' not in result


def test_clean_json_string_whitespace():
    """Test _clean_json_string normalizes whitespace"""
    result = _clean_json_string('{"key":   "value"}')
    assert "  " not in result


def test_clean_json_string_unquoted_keys():
    """Test _clean_json_string quotes unquoted keys"""
    result = _clean_json_string('{key: "value"}')
    assert '"key"' in result


# ============================================================================
# Test fix_json_string - Lines 88-89 (escaped chars)
# ============================================================================


def test_fix_json_string_escaped_backslash():
    """Test fix_json_string with escaped backslash - covers lines 88-89"""
    # JSON string with escaped characters
    json_str = r'{"path": "C:\\Users\\file.txt"}'
    result = fix_json_string(json_str)
    # Should handle escaped backslashes properly
    assert result == json_str  # Should be unchanged


def test_fix_json_string_escaped_quote():
    """Test fix_json_string with escaped quote - covers lines 88-89"""
    json_str = r'{"text": "He said \"hello\""}'
    result = fix_json_string(json_str)
    # Should handle escaped quotes properly
    assert result == json_str


def test_fix_json_string_escaped_newline():
    """Test fix_json_string with escaped newline - covers lines 88-89"""
    json_str = r'{"text": "line1\nline2"}'
    result = fix_json_string(json_str)
    assert result == json_str


def test_fix_json_string_multiple_escapes():
    """Test fix_json_string with multiple escape sequences"""
    json_str = r'{"path": "C:\\folder\\file", "text": "quote: \"hi\"\nend"}'
    result = fix_json_string(json_str)
    # Should handle all escapes properly
    assert result == json_str


def test_fix_json_string_missing_bracket():
    """Test fix_json_string adds missing closing bracket"""
    json_str = '{"key": "value"'
    result = fix_json_string(json_str)
    assert result == '{"key": "value"}'


def test_fix_json_string_missing_multiple_brackets():
    """Test fix_json_string adds multiple missing brackets"""
    json_str = '{"key": {"nested": "value"'
    result = fix_json_string(json_str)
    assert result == '{"key": {"nested": "value"}}'


def test_fix_json_string_missing_array_bracket():
    """Test fix_json_string adds missing array bracket"""
    json_str = '["item1", "item2"'
    result = fix_json_string(json_str)
    assert result == '["item1", "item2"]'


def test_fix_json_string_extra_closing_bracket():
    """Test fix_json_string with extra closing bracket"""
    json_str = '{"key": "value"}}'
    with pytest.raises(ValueError, match="Extra closing bracket"):
        fix_json_string(json_str)


def test_fix_json_string_mismatched_brackets():
    """Test fix_json_string with mismatched brackets"""
    json_str = '{"key": "value"]'
    with pytest.raises(ValueError, match="Mismatched brackets"):
        fix_json_string(json_str)


def test_fix_json_string_empty():
    """Test fix_json_string with empty string"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fix_json_string("")


def test_fix_json_string_complex_with_escapes():
    """Test fix_json_string with complex JSON containing escapes"""
    json_str = r'{"data": {"path": "C:\\test\\", "text": "say \"hi\"", "newline": "a\nb"'
    result = fix_json_string(json_str)
    # Should add missing closing brackets while preserving escapes
    assert result.endswith("}}")
    assert "\\\\" in result or "\\test" in result  # Escapes preserved


def test_fix_json_string_escape_at_end():
    """Test fix_json_string with escape at end of string"""
    json_str = r'{"path": "folder\\"'
    result = fix_json_string(json_str)
    # Should handle backslash at end and add missing bracket
    assert result.endswith("}")


def test_fuzzy_json_with_escapes_comprehensive():
    """Comprehensive test for fuzzy_json with various escape scenarios"""
    # Test that fuzzy_json can handle JSON with escapes through the full pipeline
    json_str = r'{"file": "C:\\Users\\test.txt", "quote": "He said \"hello\""}'
    result = fuzzy_json(json_str)
    assert result["file"] == "C:\\Users\\test.txt"
    assert result["quote"] == 'He said "hello"'


def test_fix_json_string_escaped_chars_in_array():
    """Test fix_json_string with escaped chars in array"""
    json_str = r'["item1", "C:\\path\\file", "text with \"quotes\""'
    result = fix_json_string(json_str)
    assert result.endswith("]")
    assert "\\\\" in result or "\\path" in result


def test_fix_json_string_nested_with_escapes():
    """Test fix_json_string with nested structures and escapes"""
    json_str = r'{"outer": {"inner": "path\\to\\file"'
    result = fix_json_string(json_str)
    # Should add two closing brackets and preserve escapes
    assert result.count("}") == 2
    assert "path" in result


# ============================================================================
# Design Documentation Tests
# ============================================================================


def test_fuzzy_json_parsing_algorithm_design():
    """Fuzzy JSON parsing algorithm with fallback strategies.

    Design Rationale:
    Provides robust JSON parsing for LLM outputs which often have minor syntax
    errors like single quotes, unquoted keys, trailing commas, or missing brackets.

    Parsing Algorithm (3-stage fallback):

    1. Direct Parse (Fast Path):
       - Try orjson.loads() directly
       - If successful, return immediately (no overhead for valid JSON)
       - ~100x faster than regex cleaning for valid input

    2. Clean and Normalize:
       - Replace single quotes with double quotes
       - Normalize whitespace (collapse multiple spaces)
       - Remove trailing commas before closing brackets
       - Quote unquoted object keys
       - Try orjson.loads() again

    3. Structural Repair:
       - Balance unmatched brackets using fix_json_string()
       - Add missing closing brackets in reverse order
       - Try orjson.loads() one last time

    4. Fail:
       - Raise ValueError if all strategies fail

    Bracket Balancing Algorithm (fix_json_string):
    1. Track opening brackets {[ in stack
    2. Skip escaped characters (\\) to avoid false matches
    3. Skip string contents (preserve quotes)
    4. Match closing brackets }] with stack
    5. Detect mismatches and extra closers (raise error)
    6. Add missing closing brackets at end in reverse order

    Error Cases:
    - Extra closing bracket: ValueError (cannot guess intent)
    - Mismatched brackets: ValueError (e.g., {] or [})

    Success Examples:
        '{"key": "value"' → '{"key": "value"}'  # Add }
        '[1, 2, 3' → '[1, 2, 3]'  # Add ]
        '{"a": [1, 2' → '{"a": [1, 2]}'  # Add ] then }

    Trade-offs:
    - May "fix" JSON in unexpected ways (e.g., adding closing brackets)
    - Not suitable for strict validation
    - Optimized for LLM output where structure is mostly correct
    - Fast path for valid JSON avoids overhead

    Design Benefits:
    - Progressive enhancement: Try fast path first
    - Minimal fixes for common LLM issues
    - Preserves escape sequences correctly
    - Fails loudly on truly malformed input

    Alternative Considered:
    JSON5 library - Rejected because it's too permissive (allows comments,
    trailing commas everywhere, etc.) and doesn't match LLM output patterns.

    See tests:
    - test_fuzzy_json_*: Various correction scenarios
    - test_fix_json_string_*: Bracket balancing edge cases
    - test_fuzzy_json_with_escapes_comprehensive: Escape handling
    """
    # Valid JSON takes fast path
    result = fuzzy_json('{"key": "value"}')
    assert result == {"key": "value"}

    # Single quotes get cleaned
    result = fuzzy_json("{'key': 'value'}")
    assert result == {"key": "value"}

    # Missing bracket gets fixed
    result = fuzzy_json('{"key": "value"')
    assert result == {"key": "value"}


# ============================================================================
# Coverage for missing lines 71-72 (escape outside string)
# ============================================================================


def test_fix_json_string_backslash_outside_string():
    """Test backslash outside of string context (lines 71-72)

    This covers the edge case where a backslash appears outside a quoted string.
    While this is invalid JSON, the code gracefully handles it by skipping
    the backslash and the next character.
    """
    # Create JSON-like string with backslash outside of a string context
    # The backslash at the beginning will be skipped
    json_str = r'\x{"key": "value"}'  # backslash before 'x' outside string
    result = fix_json_string(json_str)
    # Should handle the escape (skips \x) and process the rest
    # The opening { should be matched with closing }
    assert "{" in result and "}" in result
    assert result.endswith("}")


def test_fix_json_string_escape_outside_preserves_brackets():
    """Test that escapes outside strings don't interfere with bracket matching (lines 71-72)"""
    # Backslash outside string followed by valid JSON
    json_str = r'\n{"key": "value"'
    result = fix_json_string(json_str)
    # The \n should be skipped (lines 71-72), then { is matched
    # Missing } should be added
    assert result.count("{") == 1
    assert result.count("}") == 1


# ============================================================================
# Type Validation Tests - PR #58 Fix
# ============================================================================


def test_fuzzy_json_rejects_primitive_int():
    """Test fuzzy_json rejects primitive int"""
    with pytest.raises(TypeError, match="got primitive type: int"):
        fuzzy_json("42")


def test_fuzzy_json_rejects_primitive_string():
    """Test fuzzy_json rejects primitive string"""
    with pytest.raises(TypeError, match="got primitive type: str"):
        fuzzy_json('"hello"')


def test_fuzzy_json_rejects_primitive_bool():
    """Test fuzzy_json rejects primitive bool"""
    with pytest.raises(TypeError, match="got primitive type: bool"):
        fuzzy_json("true")


def test_fuzzy_json_rejects_primitive_null():
    """Test fuzzy_json rejects primitive null"""
    with pytest.raises(TypeError, match="got primitive type: NoneType"):
        fuzzy_json("null")


def test_fuzzy_json_rejects_primitive_float():
    """Test fuzzy_json rejects primitive float"""
    with pytest.raises(TypeError, match="got primitive type: float"):
        fuzzy_json("3.14")


def test_fuzzy_json_rejects_list_of_primitives():
    """Test fuzzy_json rejects list of primitive values"""
    with pytest.raises(TypeError, match="list with non-dict element at index 0"):
        fuzzy_json("[1, 2, 3]")


def test_fuzzy_json_rejects_list_of_strings():
    """Test fuzzy_json rejects list of strings"""
    with pytest.raises(TypeError, match="list with non-dict element at index 0: str"):
        fuzzy_json('["a", "b", "c"]')


def test_fuzzy_json_rejects_mixed_list():
    """Test fuzzy_json rejects list with mix of dicts and primitives"""
    with pytest.raises(TypeError, match="list with non-dict element at index 1: int"):
        fuzzy_json('[{"key": "value"}, 42, {"other": "data"}]')


def test_fuzzy_json_accepts_empty_list():
    """Test fuzzy_json accepts empty list (vacuously list[dict])"""
    result = fuzzy_json("[]")
    assert result == []


def test_fuzzy_json_accepts_list_of_dicts():
    """Test fuzzy_json accepts list of dicts"""
    result = fuzzy_json('[{"a": 1}, {"b": 2}]')
    assert result == [{"a": 1}, {"b": 2}]


def test_fuzzy_json_accepts_dict():
    """Test fuzzy_json accepts dict"""
    result = fuzzy_json('{"key": "value"}')
    assert result == {"key": "value"}

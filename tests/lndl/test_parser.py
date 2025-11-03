# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.lndl import (
    MissingOutBlockError,
    extract_lvars,
    extract_out_block,
)


class TestExtractLvars:
    """Test lvar extraction."""

    def test_single_lvar(self):
        """Test extracting single lvar."""
        text = "<lvar x>42</lvar>"
        result = extract_lvars(text)
        assert result == {"x": "42"}

    def test_multiple_lvars(self):
        """Test extracting multiple lvars."""
        text = "<lvar x>42</lvar> <lvar y>hello</lvar>"
        result = extract_lvars(text)
        assert result == {"x": "42", "y": "hello"}

    def test_multiline_content(self):
        """Test lvar with multiline content."""
        text = """
        <lvar reason>
        This is a long
        multiline explanation
        </lvar>
        """
        result = extract_lvars(text)
        assert "reason" in result
        assert "multiline" in result["reason"]

    def test_lvar_with_whitespace(self):
        """Test lvar content whitespace handling."""
        text = "<lvar x>  42  </lvar>"
        result = extract_lvars(text)
        assert result == {"x": "42"}  # Trimmed

    def test_lvar_with_special_chars(self):
        """Test lvar content with special characters."""
        text = '<lvar query>search(query="AI", limit=10)</lvar>'
        result = extract_lvars(text)
        assert "query" in result
        assert "AI" in result["query"]

    def test_empty_lvars(self):
        """Test when no lvars present."""
        text = "no lvars here"
        result = extract_lvars(text)
        assert result == {}


class TestExtractOutBlock:
    """Test OUT{} block extraction."""

    def test_simple_out_block(self):
        """Test extracting simple OUT{} block."""
        text = "OUT{reason: Reason(confidence=x)}"
        result = extract_out_block(text)
        assert "reason" in result

    def test_multiline_out_block(self):
        """Test extracting multiline OUT{} block."""
        text = """
        OUT{
            reason: Reason(confidence=x),
            report: Report(title=t)
        }
        """
        result = extract_out_block(text)
        assert "reason" in result
        assert "report" in result

    def test_out_block_case_insensitive(self):
        """Test OUT{} case insensitivity."""
        text = "out{reason: Reason(confidence=x)}"
        result = extract_out_block(text)
        assert "reason" in result

    def test_missing_out_block(self):
        """Test error when OUT{} block not found."""
        text = "no OUT block here"
        with pytest.raises(MissingOutBlockError):
            extract_out_block(text)

    def test_extract_out_block_missing(self):
        """Test MissingOutBlockError when no OUT{} found."""
        text = "<lvar Report.title t>Title</lvar>"

        with pytest.raises(MissingOutBlockError, match="No OUT"):
            extract_out_block(text)


class TestParseOutBlockArray:
    """Tests for parse_out_block_array edge cases."""

    def test_parse_out_block_array_empty_content(self):
        """Test parsing empty OUT block content."""
        from lionherd_core.lndl.parser import parse_out_block_array

        result = parse_out_block_array("")
        assert result == {}

    def test_parse_out_block_array_field_no_colon(self):
        """Test parsing field without colon (graceful failure)."""
        from lionherd_core.lndl.parser import parse_out_block_array

        # Field name followed by non-colon character (not end of string)
        result = parse_out_block_array("fieldname something_else")  # Missing colon
        assert result == {}  # Should handle gracefully

        # Also test when end of string is reached
        result2 = parse_out_block_array("fieldname")
        assert result2 == {}  #

    def test_parse_out_block_array_escaped_quote(self):
        """Test parsing with escaped quotes in string."""
        from lionherd_core.lndl.parser import parse_out_block_array

        result = parse_out_block_array('field:"value with \\" quote"')
        assert result == {"field": '"value with \\" quote"'}


# Legacy test classes removed (parse_out_block, parse_constructor)
# These functions are no longer part of the namespace-prefixed LNDL syntax


class TestPhase3ParserCoverage:
    """Tests for parser edge cases and special syntax."""

    def test_parse_value_null(self):
        """Test parsing null literal (case-insensitive)."""
        from lionherd_core.lndl.parser import parse_value

        result = parse_value("null")
        assert result is None

    def test_parse_value_null_uppercase(self):
        """Test parsing NULL literal case insensitive."""
        from lionherd_core.lndl.parser import parse_value

        result = parse_value("NULL")
        assert result is None

    def test_extract_balanced_with_escaped_chars(self):
        """Test balanced scanner handles escaped characters."""
        # Test escaped quote
        text = r'OUT{field:"value with \" escaped quote"}'
        result = extract_out_block(text)
        assert "escaped quote" in result

        # Test escaped brace in string
        text2 = r'OUT{field:"value with \} brace"}'
        result2 = extract_out_block(text2)
        assert result2  # Should successfully extract

    def test_parse_out_block_array_empty_with_whitespace(self):
        """Test empty content after whitespace skip."""
        from lionherd_core.lndl.parser import parse_out_block_array

        # Only whitespace
        result = parse_out_block_array("   \n  \t  ")
        assert result == {}

    def test_parse_out_block_array_field_without_colon(self):
        """Test field extraction when colon missing (partial parsing)."""
        from lionherd_core.lndl.parser import parse_out_block_array

        # Field name at end without colon
        result = parse_out_block_array("field1:value1, field2")
        assert "field1" in result
        assert "field2" not in result  # Breaks when no colon

    def test_parse_nested_brackets(self):
        """Test parsing nested bracket syntax."""
        from lionherd_core.lndl.parser import parse_out_block_array

        # Nested array syntax
        result = parse_out_block_array("field:[[var1, var2], [var3]]")
        assert "field" in result
        # The nested structure should be captured as a single string
        assert isinstance(result["field"], (str, list))

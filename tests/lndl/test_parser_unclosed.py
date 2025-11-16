"""Tests for unclosed tag error handling in parser.py.

These tests target specific uncovered lines:
- Line 356: Unclosed lvar tag (EOF before </lvar>)
- Line 445: Unclosed lact tag (EOF before </lact>)
- Line 479: break after newlines in OUT{} block
- Line 511: break after newlines in array
"""

import pytest

from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import ParseError, Parser


class TestUnclosedTags:
    """Test error handling for unclosed tags.

    Note: parse_lvar() and parse_lact() are not called by parse() in current architecture
    (uses regex extraction instead), but these tests ensure they work correctly for:
    - Future refactoring
    - Debugging/validation
    - Alternative parsing modes
    """

    def test_unclosed_lvar_tag_missing_close(self):
        """Test ParseError when lvar tag is not closed - missing </lvar> (line 356)."""
        # Source has opening tag but file ends before closing tag
        source = "<lvar Report.title t>Content here\n\nMore content but no closing tag"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        # Manually call parse_lvar() to test unclosed tag handling
        with pytest.raises(ParseError) as exc_info:
            parser.parse_lvar()

        assert "Unclosed lvar tag" in str(exc_info.value)
        assert "missing </lvar>" in str(exc_info.value)

    def test_unclosed_lvar_tag_eof_in_content(self):
        """Test ParseError when EOF occurs while scanning lvar content (line 356)."""
        source = "<lvar Report.summary s>Summary content\nwith multiple lines"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError) as exc_info:
            parser.parse_lvar()

        assert "Unclosed lvar tag" in str(exc_info.value)

    def test_unclosed_lact_tag_missing_close(self):
        """Test ParseError when lact tag is not closed - missing </lact> (line 445)."""
        # Source has opening tag but file ends before closing tag
        source = "<lact Report.process p>process(arg1, arg2)\n\nMore content"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError) as exc_info:
            parser.parse_lact()

        assert "Unclosed lact tag" in str(exc_info.value)
        assert "missing </lact>" in str(exc_info.value)

    def test_unclosed_lact_tag_eof_in_call(self):
        """Test ParseError when EOF occurs while scanning lact call (line 445)."""
        source = "<lact Analysis.analyze a>analyze(data, threshold=0.95)"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError) as exc_info:
            parser.parse_lact()

        assert "Unclosed lact tag" in str(exc_info.value)


class TestNewlineBreaks:
    """Test break statements after newlines in parsing loops."""

    def test_out_block_with_trailing_newlines(self):
        """Test OUT{} block with newlines before closing brace (line 479)."""
        source = """OUT{
            field1: value1,
            field2: value2

        }"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should successfully parse despite trailing newlines
        assert "field1" in out_block.fields
        assert "field2" in out_block.fields

    def test_out_block_empty_with_newlines(self):
        """Test empty OUT{} block with only newlines (line 479)."""
        source = """OUT{

        }"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should parse as empty OUT{} block
        assert len(out_block.fields) == 0

    def test_array_with_trailing_newlines(self):
        """Test array with newlines before closing bracket (line 511)."""
        source = """OUT{field: [
            val1,
            val2

        ]}"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should successfully parse array despite trailing newlines
        assert "field" in out_block.fields
        assert len(out_block.fields["field"]) == 2

    def test_array_empty_with_newlines(self):
        """Test empty array with only newlines (line 511)."""
        source = """OUT{field: [

        ]}"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should parse as empty array
        assert "field" in out_block.fields
        assert out_block.fields["field"] == []

    def test_array_with_newlines_between_elements(self):
        """Test array with multiple newlines between elements (line 511)."""
        source = """OUT{field: [
            elem1

            ,

            elem2

        ]}"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should successfully parse
        assert "field" in out_block.fields
        assert len(out_block.fields["field"]) == 2

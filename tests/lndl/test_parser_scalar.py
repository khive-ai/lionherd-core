# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for parser.py edge cases and error conditions to increase coverage."""

import pytest

from lionherd_core.lndl.lexer import Lexer, Token, TokenType
from lionherd_core.lndl.parser import ParseError, Parser, parse_value


class TestParserEdgeCases:
    """Test parser edge cases for boundary conditions."""

    def test_current_token_when_pos_beyond_length(self):
        """Test current_token returns EOF when pos >= len(tokens)."""
        tokens = [
            Token(TokenType.LVAR_OPEN, "<lvar", 1, 1),
            Token(TokenType.ID, "x", 1, 7),
            Token(TokenType.GT, ">", 1, 8),
            Token(TokenType.LVAR_CLOSE, "</lvar>", 1, 15),
            Token(TokenType.EOF, "", 1, 22),
        ]
        parser = Parser(tokens, source_text="<lvar x>test</lvar>")

        # Advance beyond tokens
        parser.pos = len(tokens)
        current = parser.current_token()

        # Should return EOF token (last token)
        assert current.type == TokenType.EOF

        # Even further beyond
        parser.pos = len(tokens) + 10
        current = parser.current_token()
        assert current.type == TokenType.EOF

    def test_peek_token_beyond_bounds(self):
        """Test peek_token returns EOF when offset goes beyond tokens."""
        tokens = [
            Token(TokenType.ID, "x", 1, 1),
            Token(TokenType.EOF, "", 1, 2),
        ]
        parser = Parser(tokens, source_text="x")

        # Peek far beyond
        peeked = parser.peek_token(offset=10)
        assert peeked.type == TokenType.EOF

        # Peek from last position
        parser.pos = len(tokens) - 1
        peeked = parser.peek_token(offset=5)
        assert peeked.type == TokenType.EOF

    def test_peek_token_within_bounds(self):
        """Test peek_token returns correct token when within bounds (line 166)."""
        tokens = [
            Token(TokenType.ID, "x", 1, 1),
            Token(TokenType.ID, "y", 1, 3),
            Token(TokenType.ID, "z", 1, 5),
            Token(TokenType.EOF, "", 1, 6),
        ]
        parser = Parser(tokens, source_text="x y z")

        # Peek ahead by 1 from position 0
        parser.pos = 0
        peeked = parser.peek_token(offset=1)
        assert peeked.type == TokenType.ID
        assert peeked.value == "y"

        # Peek ahead by 2 from position 0
        peeked = parser.peek_token(offset=2)
        assert peeked.type == TokenType.ID
        assert peeked.value == "z"

    def test_parse_without_source_text(self):
        """Test parse() raises ParseError when source_text is None."""
        lexer = Lexer("<lvar x>test</lvar>")
        tokens = lexer.tokenize()

        # Create parser without source_text
        parser = Parser(tokens, source_text=None)

        with pytest.raises(ParseError, match="requires source_text"):
            parser.parse()

    def test_parse_with_eof_after_newlines(self):
        """Test parse() break statement when EOF after newlines (line 261)."""
        source = "\n\n\n"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        program = parser.parse()

        # Should parse successfully with no content
        assert len(program.lvars) == 0
        assert len(program.lacts) == 0
        assert program.out_block is None


class TestParserLvarErrors:
    """Test lvar parsing error conditions."""

    def test_parse_lvar_without_source_text(self):
        """Test parse_lvar raises ParseError when source_text is None (line 331)."""
        # Namespaced format: <lvar Model.field>
        tokens = [
            Token(TokenType.LVAR_OPEN, "<lvar", 1, 1),
            Token(TokenType.ID, "Model", 1, 7),
            Token(TokenType.DOT, ".", 1, 12),
            Token(TokenType.ID, "field", 1, 13),
            Token(TokenType.GT, ">", 1, 18),
            Token(TokenType.ID, "test", 1, 19),
            Token(TokenType.LVAR_CLOSE, "</lvar>", 1, 23),
            Token(TokenType.EOF, "", 1, 30),
        ]

        # Create parser without source_text
        parser = Parser(tokens, source_text=None)

        with pytest.raises(ParseError, match="requires source_text"):
            parser.parse_lvar()

    def test_parse_lvar_content_extraction_failure(self):
        """Test parse_lvar raises ParseError when pattern doesn't match (line 347)."""
        # Create tokens that parse successfully but source doesn't match regex pattern
        # Use special characters that will cause regex pattern to fail
        tokens = [
            Token(TokenType.LVAR_OPEN, "<lvar", 1, 1),
            Token(TokenType.ID, "Model", 1, 7),
            Token(TokenType.DOT, ".", 1, 12),
            Token(TokenType.ID, "field", 1, 13),
            Token(TokenType.GT, ">", 1, 18),
            Token(TokenType.ID, "content", 1, 19),
            Token(TokenType.LVAR_CLOSE, "</lvar>", 1, 26),
            Token(TokenType.EOF, "", 1, 33),
        ]

        # Source has closing tag but contains malformed syntax that breaks regex
        # Token stream says "Model.field" but source has special chars
        parser = Parser(tokens, source_text="<lvar Model(.field>content</lvar>")

        with pytest.raises(ParseError, match="Could not extract lvar content"):
            parser.parse_lvar()

    def test_parse_lvar_unclosed_tag_eof(self):
        """Test parse_lvar raises ParseError for unclosed tag (line 356)."""
        # Source with no closing tag (namespaced format)
        source = "<lvar Model.field>content without closing tag"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError, match="Unclosed lvar tag"):
            parser.parse_lvar()

    def test_parse_lvar_eof_while_skipping_tokens(self):
        """Test parse_lvar hits EOF while skipping tokens to closing tag (line 356)."""
        # Manually create token stream that has content but EOF before LVAR_CLOSE
        tokens = [
            Token(TokenType.LVAR_OPEN, "<lvar", 1, 1),
            Token(TokenType.ID, "Model", 1, 7),
            Token(TokenType.DOT, ".", 1, 12),
            Token(TokenType.ID, "field", 1, 13),
            Token(TokenType.GT, ">", 1, 18),
            Token(TokenType.ID, "content", 1, 19),
            Token(TokenType.ID, "more", 1, 27),
            Token(TokenType.ID, "stuff", 1, 32),
            Token(TokenType.EOF, "", 1, 38),  # EOF instead of LVAR_CLOSE
        ]
        parser = Parser(tokens, source_text="<lvar Model.field>content more stuff")

        with pytest.raises(ParseError, match="Unclosed lvar tag"):
            parser.parse_lvar()


class TestParserLactErrors:
    """Test lact parsing error conditions."""

    def test_parse_lact_without_source_text(self):
        """Test parse_lact raises ParseError when source_text is None (line 415)."""
        tokens = [
            Token(TokenType.LACT_OPEN, "<lact", 1, 1),
            Token(TokenType.ID, "x", 1, 7),
            Token(TokenType.GT, ">", 1, 8),
            Token(TokenType.ID, "func", 1, 9),
            Token(TokenType.LACT_CLOSE, "</lact>", 1, 15),
            Token(TokenType.EOF, "", 1, 22),
        ]

        # Create parser without source_text
        parser = Parser(tokens, source_text=None)

        with pytest.raises(ParseError, match="requires source_text"):
            parser.parse_lact()

    def test_parse_lact_content_extraction_failure(self):
        """Test parse_lact raises ParseError when pattern doesn't match (line 436)."""
        tokens = [
            Token(TokenType.LACT_OPEN, "<lact", 1, 1),
            Token(TokenType.ID, "x", 1, 7),
            Token(TokenType.GT, ">", 1, 8),
            Token(TokenType.ID, "func", 1, 9),
            Token(TokenType.LACT_CLOSE, "</lact>", 1, 14),
            Token(TokenType.EOF, "", 1, 21),
        ]

        # Source has closing tag but name contains special regex chars
        parser = Parser(tokens, source_text="<lact x(>func()</lact>")

        with pytest.raises(ParseError, match="Could not extract lact call"):
            parser.parse_lact()

    def test_parse_lact_unclosed_tag_eof(self):
        """Test parse_lact raises ParseError for unclosed tag (line 445)."""
        source = "<lact x>func() without closing tag"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError, match="Unclosed lact tag"):
            parser.parse_lact()

    def test_parse_lact_eof_while_skipping_tokens(self):
        """Test parse_lact hits EOF while skipping tokens to closing tag (line 445)."""
        # Manually create token stream that has content but EOF before LACT_CLOSE
        tokens = [
            Token(TokenType.LACT_OPEN, "<lact", 1, 1),
            Token(TokenType.ID, "x", 1, 7),
            Token(TokenType.GT, ">", 1, 8),
            Token(TokenType.ID, "func", 1, 9),
            Token(TokenType.LPAREN, "(", 1, 13),
            Token(TokenType.RPAREN, ")", 1, 14),
            Token(TokenType.EOF, "", 1, 15),  # EOF instead of LACT_CLOSE
        ]
        parser = Parser(tokens, source_text="<lact x>func()")

        with pytest.raises(ParseError, match="Unclosed lact tag"):
            parser.parse_lact()


class TestParseOutBlockEdgeCases:
    """Test OUT{} block parsing edge cases."""

    def test_out_block_eof_after_newlines(self):
        """Test OUT{} break when OUT_CLOSE/EOF after newlines (line 479)."""
        source = "OUT{\n\n\n}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should parse empty block successfully
        assert out_block.fields == {}

    def test_out_block_break_on_out_close_after_field(self):
        """Test OUT{} break statement after field parsing (line 479)."""
        # Test that we hit the break after skip_newlines when OUT_CLOSE is found
        source = "OUT{field: 1\n\n}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        assert out_block.fields["field"] == 1

    def test_out_block_unexpected_token_instead_of_id(self):
        """Test OUT{} skips unexpected tokens (lines 484-485)."""
        # Manually construct tokens with unexpected token type
        tokens = [
            Token(TokenType.OUT_OPEN, "OUT{", 1, 1),
            Token(TokenType.COMMA, ",", 1, 5),  # Unexpected comma instead of ID
            Token(TokenType.ID, "field", 1, 7),
            Token(TokenType.COLON, ":", 1, 12),
            Token(TokenType.NUM, "1", 1, 13),
            Token(TokenType.OUT_CLOSE, "}", 1, 14),
            Token(TokenType.EOF, "", 1, 15),
        ]
        parser = Parser(tokens, source_text="OUT{,field:1}")

        out_block = parser.parse_out_block()

        # Should skip comma and parse field normally
        assert "field" in out_block.fields
        assert out_block.fields["field"] == 1

    def test_out_block_field_without_colon(self):
        """Test OUT{} skips field without colon (line 495)."""
        # Field name without colon
        tokens = [
            Token(TokenType.OUT_OPEN, "OUT{", 1, 1),
            Token(TokenType.ID, "malformed", 1, 5),
            Token(TokenType.ID, "field", 1, 15),  # Next ID instead of colon
            Token(TokenType.COLON, ":", 1, 20),
            Token(TokenType.NUM, "1", 1, 21),
            Token(TokenType.OUT_CLOSE, "}", 1, 22),
            Token(TokenType.EOF, "", 1, 23),
        ]
        parser = Parser(tokens, source_text="OUT{malformed field:1}")

        out_block = parser.parse_out_block()

        # Should skip malformed and parse field
        assert "malformed" not in out_block.fields
        assert "field" in out_block.fields
        assert out_block.fields["field"] == 1


class TestOutBlockArrayParsing:
    """Test array parsing edge cases in OUT{} blocks."""

    def test_array_break_with_rbracket_after_newlines(self):
        """Test array parsing break when RBRACKET after newlines (line 511)."""
        source = "OUT{field:[\n\n\n]}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should parse empty array successfully
        assert out_block.fields["field"] == []

    def test_array_break_after_element_with_newlines(self):
        """Test array parsing break after element with newlines (line 511)."""
        # Test that we hit the break after skip_newlines when RBRACKET is found
        source = "OUT{field:[ref1\n\n]}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        assert out_block.fields["field"] == ["ref1"]

    def test_array_with_string_literals(self):
        """Test array parsing with string literals (lines 520-523)."""
        source = 'OUT{field:["str1", "str2", "str3"]}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        # Should parse string array (STR tokens have quotes stripped by lexer)
        assert "field" in out_block.fields
        assert out_block.fields["field"] == ["str1", "str2", "str3"]

    def test_array_with_mixed_types(self):
        """Test array with IDs, numbers, and strings (lines 513-526)."""
        source = 'OUT{field:[ref1, 630, 4.02, "text"]}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        assert "field" in out_block.fields
        # All converted to strings in refs list
        assert len(out_block.fields["field"]) == 4
        assert "ref1" in out_block.fields["field"]
        assert "630" in out_block.fields["field"]

    def test_array_with_unknown_token_type(self):
        """Test array with unknown token type gets skipped (lines 524-526)."""
        # Manually create tokens with unknown type
        tokens = [
            Token(TokenType.OUT_OPEN, "OUT{", 1, 1),
            Token(TokenType.ID, "field", 1, 5),
            Token(TokenType.COLON, ":", 1, 10),
            Token(TokenType.LBRACKET, "[", 1, 11),
            Token(TokenType.ID, "ref1", 1, 12),
            Token(TokenType.COMMA, ",", 1, 16),
            # Simulate unknown token by using OUT_OPEN inside array (invalid)
            Token(TokenType.OUT_OPEN, "OUT{", 1, 17),
            Token(TokenType.COMMA, ",", 1, 21),
            Token(TokenType.ID, "ref2", 1, 22),
            Token(TokenType.RBRACKET, "]", 1, 26),
            Token(TokenType.OUT_CLOSE, "}", 1, 27),
            Token(TokenType.EOF, "", 1, 28),
        ]
        parser = Parser(tokens, source_text="OUT{field:[ref1,OUT{,ref2]}")

        out_block = parser.parse_out_block()

        # Should skip unknown token and parse valid ones
        assert "field" in out_block.fields
        assert "ref1" in out_block.fields["field"]
        assert "ref2" in out_block.fields["field"]


class TestOutBlockValueTypes:
    """Test OUT{} value type handling."""

    def test_field_with_unknown_value_type(self):
        """Test field with unknown value type gets skipped (line 570)."""
        # Create tokens where value is unexpected type
        tokens = [
            Token(TokenType.OUT_OPEN, "OUT{", 1, 1),
            Token(TokenType.ID, "field1", 1, 5),
            Token(TokenType.COLON, ":", 1, 11),
            # Unexpected token type (LVAR_OPEN as value)
            Token(TokenType.LVAR_OPEN, "<lvar", 1, 12),
            Token(TokenType.COMMA, ",", 1, 17),
            Token(TokenType.ID, "field2", 1, 18),
            Token(TokenType.COLON, ":", 1, 24),
            Token(TokenType.NUM, "630", 1, 25),
            Token(TokenType.OUT_CLOSE, "}", 1, 27),
            Token(TokenType.EOF, "", 1, 28),
        ]
        parser = Parser(tokens, source_text="OUT{field1:<lvar,field2:630}")

        out_block = parser.parse_out_block()

        # field1 should be skipped, field2 should parse
        assert "field1" not in out_block.fields
        assert "field2" in out_block.fields
        assert out_block.fields["field2"] == 630


class TestParseValueFunction:
    """Test parse_value function for all branches."""

    def test_parse_value_non_string_passthrough(self):
        """Test parse_value returns non-string values as-is (line 625)."""
        # Test with various non-string types
        assert parse_value(630) == 630
        assert parse_value(4.02) == 4.02
        assert parse_value(True) is True
        assert parse_value(None) is None
        assert parse_value([1, 2, 3]) == [1, 2, 3]
        assert parse_value({"key": "value"}) == {"key": "value"}

    def test_parse_value_true_string(self):
        """Test parse_value converts 'true' string to True (line 628)."""
        assert parse_value("true") is True
        assert parse_value("TRUE") is True
        assert parse_value("True") is True
        assert parse_value("  true  ") is True

    def test_parse_value_false_string(self):
        """Test parse_value converts 'false' string to False (line 630)."""
        assert parse_value("false") is False
        assert parse_value("FALSE") is False
        assert parse_value("False") is False
        assert parse_value("  false  ") is False

    def test_parse_value_null_string(self):
        """Test parse_value converts 'null' string to None (line 632)."""
        assert parse_value("null") is None
        assert parse_value("NULL") is None
        assert parse_value("Null") is None
        assert parse_value("  null  ") is None

    def test_parse_value_literal_eval(self):
        """Test parse_value uses ast.literal_eval for other values."""
        # Numbers
        assert parse_value("630") == 630
        assert parse_value("4.02") == 4.02
        assert parse_value("-5") == -5

        # Strings with quotes
        assert parse_value('"hello"') == "hello"
        assert parse_value("'world'") == "world"

        # Lists
        assert parse_value("[1, 2, 3]") == [1, 2, 3]

        # Dicts
        assert parse_value("{'key': 'value'}") == {"key": "value"}

    def test_parse_value_invalid_literal(self):
        """Test parse_value returns string for unparseable values."""
        # Invalid Python literals should return as string
        assert parse_value("not-a-literal") == "not-a-literal"
        assert parse_value("hello world") == "hello world"
        assert parse_value("undefined_var") == "undefined_var"


class TestComplexParsingScenarios:
    """Test complex parsing scenarios combining multiple edge cases."""

    def test_out_block_with_all_value_types(self):
        """Test OUT{} with IDs, strings, numbers, booleans, and arrays."""
        source = """OUT{
            single_ref: ref1,
            str_val: "text",
            int_val: 630,
            float_val: 4.02,
            bool_true: true,
            bool_false: false,
            array_refs: [r1, r2, r3],
            array_nums: [1, 2, 3],
            array_strs: ["a", "b"]
        }"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        assert out_block.fields["single_ref"] == ["ref1"]
        assert out_block.fields["str_val"] == "text"  # STR tokens have quotes stripped
        assert out_block.fields["int_val"] == 630
        assert out_block.fields["float_val"] == 4.02
        assert out_block.fields["bool_true"] is True
        assert out_block.fields["bool_false"] is False
        assert out_block.fields["array_refs"] == ["r1", "r2", "r3"]
        assert "1" in out_block.fields["array_nums"]

    def test_malformed_out_block_recovery(self):
        """Test parser recovers from malformed fields in OUT{} block."""
        # Multiple malformed patterns in one block
        tokens = [
            Token(TokenType.OUT_OPEN, "OUT{", 1, 1),
            # Missing colon
            Token(TokenType.ID, "bad1", 1, 5),
            Token(TokenType.ID, "bad2", 1, 10),
            Token(TokenType.COMMA, ",", 1, 14),
            # Unexpected token instead of ID
            Token(TokenType.COMMA, ",", 1, 15),
            # Valid field
            Token(TokenType.ID, "good", 1, 16),
            Token(TokenType.COLON, ":", 1, 20),
            Token(TokenType.NUM, "1", 1, 21),
            Token(TokenType.OUT_CLOSE, "}", 1, 22),
            Token(TokenType.EOF, "", 1, 23),
        ]
        parser = Parser(tokens, source_text="OUT{bad1 bad2,,good:1}")

        out_block = parser.parse_out_block()

        # Should skip malformed and parse valid field
        assert "bad1" not in out_block.fields
        assert "bad2" not in out_block.fields
        assert "good" in out_block.fields
        assert out_block.fields["good"] == 1

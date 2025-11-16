"""Unit tests for lexer.py - targeting uncovered code paths.

Focuses on:
- String tokenization edge cases (lines 185-203)
- Error handling for unterminated strings
- Context-aware string tokenization (in_out_block flag)
- Position tracking edge cases
"""

import pytest

from lionherd_core.lndl.lexer import Lexer, Token, TokenType


class TestStringTokenization:
    """Test string tokenization including edge cases."""

    def test_string_inside_out_block(self):
        """Test string tokenization works inside OUT{} blocks"""
        source = 'OUT{name: "John Doe"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Find the string token
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "John Doe"

    def test_string_outside_out_block_not_tokenized(self):
        """Test that strings outside OUT{} are not tokenized as STR"""
        source = "I'll generate the response. OUT{}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should not have any STR tokens (apostrophe not treated as string)
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 0

    def test_unterminated_string_in_out_block(self):
        """Test unterminated string inside OUT{} block"""
        source = 'OUT{name: "unterminated'
        lexer = Lexer(source)

        # Should handle gracefully - either raise error or treat as ID
        tokens = lexer.tokenize()
        # At minimum, should not crash and should have EOF token
        assert tokens[-1].type == TokenType.EOF

    def test_empty_string(self):
        """Test tokenization of empty string"""
        source = 'OUT{name: ""}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == ""

    def test_string_with_escape_sequences(self):
        """Test string containing escaped quotes"""
        source = r'OUT{msg: "He said \"hello\""}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        # Should handle escaped quotes
        assert len(str_tokens) >= 1
        assert str_tokens[0].value == 'He said "hello"'

    def test_string_with_newline_escape(self):
        """Test string with \\n escape sequence"""
        source = r'OUT{msg: "Line1\nLine2"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "Line1\nLine2"

    def test_string_with_tab_escape(self):
        """Test string with \\t escape sequence"""
        source = r'OUT{msg: "Col1\tCol2"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "Col1\tCol2"

    def test_string_with_carriage_return_escape(self):
        """Test string with \\r escape sequence"""
        source = r'OUT{msg: "Text\rOverwrite"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "Text\rOverwrite"

    def test_string_with_backslash_escape(self):
        """Test string with \\\\ escape sequence"""
        source = r'OUT{path: "C:\\Users\\file"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "C:\\Users\\file"

    def test_string_with_single_quote_escape(self):
        """Test string with \\' escape sequence"""
        source = r'OUT{msg: "It\'s working"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "It's working"

    def test_string_with_unknown_escape(self):
        """Test string with unknown escape sequence (keeps literal)"""
        source = r'OUT{msg: "Test\xUnknown"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        # Unknown escape \x should keep literal 'x'
        assert str_tokens[0].value == "TestxUnknown"

    def test_string_with_single_quotes(self):
        """Test that single quotes work same as double quotes in OUT{}"""
        source = "OUT{name: 'John'}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "John"

    def test_multiple_strings_in_out_block(self):
        """Test multiple strings in same OUT{} block"""
        source = 'OUT{first: "John", last: "Doe"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 2
        assert str_tokens[0].value == "John"
        assert str_tokens[1].value == "Doe"


class TestContextAwareTokenization:
    """Test that tokenization is context-aware (in_out_block flag)."""

    def test_apostrophe_in_narrative_not_string(self):
        """Test apostrophes in narrative text not treated as strings"""
        source = "I'll generate a response with data."
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should have ID tokens but no STR tokens
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 0

    def test_out_block_enables_string_tokenization(self):
        """Test that entering OUT{} enables string tokenization"""
        source = 'Here is text. OUT{field: "value"}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should only have STR token inside OUT{}, not before
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1

    def test_out_close_disables_string_tokenization(self):
        """Test that exiting OUT{} disables string tokenization"""
        source = 'OUT{field: "value"} More text with apostrophe\'s here'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should only have one STR token (inside OUT{})
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "value"

    def test_nested_out_blocks_maintain_context(self):
        """Test nested OUT{} blocks maintain string tokenization context"""
        source = 'OUT{outer: "val1", inner: OUT{nested: "val2"}}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        # Should tokenize strings in both outer and nested OUT{}
        assert len(str_tokens) >= 2


class TestPositionTracking:
    """Test position tracking for tokens (line and column)."""

    def test_single_line_positions(self):
        """Test column positions advance correctly on single line"""
        source = "OUT{a: 1}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # All tokens should be on line 1
        for token in tokens[:-1]:  # Exclude EOF
            assert token.line == 1

        # Column should advance (1-indexed)
        assert tokens[0].column == 1  # OUT_OPEN at column 1

    def test_multiline_positions(self):
        """Test line numbers advance correctly across newlines"""
        source = """OUT{
    field: "value"
}"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should have tokens on multiple lines
        lines = {t.line for t in tokens}
        assert len(lines) > 1
        assert 1 in lines  # First line
        assert 2 in lines or 3 in lines  # Later lines

    def test_newline_token_position(self):
        """Test NEWLINE tokens have correct position"""
        source = "OUT{\nfield: 1\n}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        newline_tokens = [t for t in tokens if t.type == TokenType.NEWLINE]
        assert len(newline_tokens) >= 1
        # Newlines should track their line number
        for nl in newline_tokens:
            assert nl.line >= 1


class TestSpecialTokens:
    """Test special tokens and edge cases."""

    def test_eof_token_always_last(self):
        """Test EOF token is always the last token"""
        source = "OUT{field: 1}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        assert tokens[-1].type == TokenType.EOF

    def test_empty_source_produces_eof(self):
        """Test empty source produces only EOF token"""
        source = ""
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only_source(self):
        """Test source with only whitespace"""
        source = "   \n  \n   "
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should have newlines and EOF
        newlines = [t for t in tokens if t.type == TokenType.NEWLINE]
        assert len(newlines) >= 2
        assert tokens[-1].type == TokenType.EOF

    def test_lvar_lact_tags_tokenized(self):
        """Test <lvar> and <lact> tags are tokenized correctly"""
        source = "<lvar test>content</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should have LVAR_OPEN and LVAR_CLOSE tokens
        assert any(t.type == TokenType.LVAR_OPEN for t in tokens)
        assert any(t.type == TokenType.LVAR_CLOSE for t in tokens)

    def test_all_punctuation_tokens(self):
        """Test all punctuation types are tokenized"""
        source = "OUT{field: func(a, b.c[0])}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        token_types = {t.type for t in tokens}

        assert TokenType.COLON in token_types
        assert TokenType.COMMA in token_types
        assert TokenType.DOT in token_types
        assert TokenType.LPAREN in token_types
        assert TokenType.RPAREN in token_types
        assert TokenType.LBRACKET in token_types
        assert TokenType.RBRACKET in token_types


class TestNumberTokenization:
    """Test number tokenization edge cases."""

    def test_integer_tokenization(self):
        """Test integers are tokenized as NUM"""
        source = "OUT{count: 42}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "42"

    def test_float_tokenization(self):
        """Test floats are tokenized as NUM"""
        source = "OUT{score: 3.14}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "3.14"

    def test_negative_number(self):
        """Test negative numbers are tokenized"""
        source = "OUT{temp: -10}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should have number token with negative value
        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "-10"

    def test_negative_float(self):
        """Test negative float numbers are tokenized"""
        source = "OUT{temp: -5.5}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "-5.5"

    def test_negative_number_outside_out_block(self):
        """Test negative numbers outside OUT{} are not tokenized as single NUM"""
        source = "The temperature is -10 degrees"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Outside OUT{}, minus is skipped and 10 is tokenized separately
        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        # Should have "10" not "-10" (minus not part of number outside OUT{})
        if num_tokens:
            assert num_tokens[0].value == "10"


class TestIdentifierTokenization:
    """Test identifier tokenization."""

    def test_simple_identifier(self):
        """Test simple identifier"""
        source = "OUT{name: value}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert len(id_tokens) == 2  # "name" and "value"
        assert "name" in [t.value for t in id_tokens]
        assert "value" in [t.value for t in id_tokens]

    def test_identifier_with_underscores(self):
        """Test identifier with underscores"""
        source = "OUT{field_name: some_value}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert any("field_name" in t.value for t in id_tokens)
        assert any("some_value" in t.value for t in id_tokens)

    def test_identifier_with_numbers(self):
        """Test identifier with numbers (not starting with number)"""
        source = "OUT{field1: value2}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert any("field1" in t.value for t in id_tokens)
        assert any("value2" in t.value for t in id_tokens)


class TestPeekCharEdgeCases:
    """Test peek_char method edge cases."""

    def test_peek_past_end_of_text(self):
        """Test peek_char returns None when peeking past end"""
        source = "a"
        lexer = Lexer(source)

        # Position at 'a' (pos=0)
        assert lexer.current_char() == "a"

        # Peek one ahead (past end at pos=1)
        assert lexer.peek_char(1) is None

        # Peek multiple positions ahead (way past end)
        assert lexer.peek_char(10) is None


class TestUnrecognizedTags:
    """Test handling of unrecognized tags starting with <."""

    def test_unrecognized_tag_skipped(self):
        """Test that unrecognized tags starting with < are skipped"""
        source = "<unknown>content</unknown> OUT{a: 1}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should tokenize the OUT block but skip unknown tags
        assert any(t.type == TokenType.OUT_OPEN for t in tokens)
        # Unknown tag content should not create special tokens
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert any(t.value == "content" for t in id_tokens)

    def test_html_like_tag_skipped(self):
        """Test HTML-like tags are skipped"""
        source = "<div>text</div> OUT{field: value}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should still tokenize OUT block correctly
        assert any(t.type == TokenType.OUT_OPEN for t in tokens)
        assert any(t.type == TokenType.ID and t.value == "field" for t in tokens)

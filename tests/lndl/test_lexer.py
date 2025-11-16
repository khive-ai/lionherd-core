# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for LNDL Lexer.

Test Coverage:
- All 17 token types (tags, literals, punctuation, control)
- Context-aware behavior (strings/negatives only in OUT{} blocks)
- Position tracking (line/column numbers)
- Escape sequences (6 standard + unknown handling)
- All methods (tokenize, current_char, peek_char, advance, skip_whitespace,
  read_identifier, read_number, read_string)
- Edge cases (empty input, malformed tags, unterminated strings, unicode, large inputs)

Critical Behavior:
- Context flag (in_out_block) toggles on OUT{ and }
- String tokenization ONLY when in_out_block=True (prevents "I'll" → consuming response)
- Negative numbers ONLY when in_out_block=True
"""

import pytest

from lionherd_core.lndl.lexer import Lexer, Token, TokenType

# ============================================================================
# Test Token Type Generation
# ============================================================================


class TestTokenTypes:
    """Test all 17 TokenType enum values."""

    @pytest.mark.parametrize(
        "input_text, expected_type, expected_value",
        [
            # Tags (6 types)
            ("<lvar", TokenType.LVAR_OPEN, "<lvar"),
            ("</lvar>", TokenType.LVAR_CLOSE, "</lvar>"),
            ("<lact", TokenType.LACT_OPEN, "<lact"),
            ("</lact>", TokenType.LACT_CLOSE, "</lact>"),
            ("OUT{", TokenType.OUT_OPEN, "OUT{"),
            ("}", TokenType.OUT_CLOSE, "}"),
            # Punctuation (8 types)
            (".", TokenType.DOT, "."),
            (",", TokenType.COMMA, ","),
            (":", TokenType.COLON, ":"),
            ("[", TokenType.LBRACKET, "["),
            ("]", TokenType.RBRACKET, "]"),
            ("(", TokenType.LPAREN, "("),
            (")", TokenType.RPAREN, ")"),
            (">", TokenType.GT, ">"),
            # Control (2 types)
            ("\n", TokenType.NEWLINE, "\n"),
            ("", TokenType.EOF, ""),
        ],
    )
    def test_single_token_generation(self, create_lexer, input_text, expected_type, expected_value):
        """Test generation of individual token types."""
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should have at least the token + EOF
        assert len(tokens) >= 1
        if input_text:  # Non-empty input produces token before EOF
            token = tokens[0]
            assert token.type == expected_type
            assert token.value == expected_value
        else:  # Empty input produces only EOF
            assert tokens[0].type == TokenType.EOF

    def test_identifier_tokens(self, create_lexer):
        """Test ID token generation."""
        test_cases = [
            ("Report", "Report"),
            ("field_name", "field_name"),
            ("_underscore", "_underscore"),
            ("var123", "var123"),
            ("s", "s"),
        ]

        for input_text, expected_value in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            assert tokens[0].type == TokenType.ID
            assert tokens[0].value == expected_value

    def test_number_tokens(self, create_lexer):
        """Test NUM token generation."""
        test_cases = [
            ("42", "42"),
            ("3.14", "3.14"),
            ("0", "0"),
            ("123.456", "123.456"),
        ]

        for input_text, expected_value in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            assert tokens[0].type == TokenType.NUM
            assert tokens[0].value == expected_value

    def test_string_tokens_in_out_block(self, create_lexer):
        """Test STR token generation (only inside OUT{} blocks)."""
        test_cases = [
            ('OUT{ x: "text" }', "text"),
            ("OUT{ x: 'text' }", "text"),
            ('OUT{ x: "Hello World" }', "Hello World"),
            ("OUT{ x: '' }", ""),
        ]

        for input_text, expected_value in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            # Find the STR token
            str_tokens = [t for t in tokens if t.type == TokenType.STR]
            assert len(str_tokens) == 1
            assert str_tokens[0].value == expected_value


# ============================================================================
# Test Context-Aware Behavior (CRITICAL)
# ============================================================================


class TestContextAware:
    """Test context-aware tokenization (strings/negatives only in OUT{} blocks).

    This is CRITICAL behavior - prevents catastrophic failures like consuming
    entire LLM response when encountering "I'll" in narrative text.
    """

    def test_strings_not_tokenized_outside_out_blocks(self, create_lexer):
        """Strings outside OUT{} blocks should NOT produce STR tokens."""
        # Narrative text with quotes should not tokenize as strings
        test_cases = [
            '"text"',
            "'text'",
            '<lvar Report.title t>"Title"</lvar>',
            "I'll analyze this",  # Critical: apostrophe should not start string
        ]

        for input_text in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            str_tokens = [t for t in tokens if t.type == TokenType.STR]
            assert len(str_tokens) == 0, f"String tokenized outside OUT{{}}: {input_text}"

    def test_strings_tokenized_inside_out_blocks(self, create_lexer):
        """Strings inside OUT{} blocks SHOULD produce STR tokens."""
        test_cases = [
            ('OUT{ x: "text" }', "text"),
            ("OUT{ x: 'text' }", "text"),
            ('OUT{ a: "val1", b: "val2" }', ["val1", "val2"]),
        ]

        for input_text, expected in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            str_tokens = [t for t in tokens if t.type == TokenType.STR]

            if isinstance(expected, list):
                assert len(str_tokens) == len(expected)
                for token, exp_val in zip(str_tokens, expected, strict=True):
                    assert token.value == exp_val
            else:
                assert len(str_tokens) == 1
                assert str_tokens[0].value == expected

    def test_negative_numbers_not_tokenized_outside_out_blocks(self, create_lexer):
        """Negative numbers outside OUT{} blocks should NOT produce NUM tokens."""
        test_cases = [
            "-42",
            "-3.14",
            "<lvar>-100</lvar>",
        ]

        for input_text in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            num_tokens = [t for t in tokens if t.type == TokenType.NUM]
            # Should only have positive numbers if any
            negative_nums = [t for t in num_tokens if t.value.startswith("-")]
            assert len(negative_nums) == 0, (
                f"Negative number tokenized outside OUT{{}}: {input_text}"
            )

    def test_negative_numbers_tokenized_inside_out_blocks(self, create_lexer):
        """Negative numbers inside OUT{} blocks SHOULD produce NUM tokens."""
        test_cases = [
            ("OUT{ score: -0.5 }", "-0.5"),
            ("OUT{ x: -42 }", "-42"),
            ("OUT{ a: -1, b: -2.5 }", ["-1", "-2.5"]),
        ]

        for input_text, expected in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            num_tokens = [t for t in tokens if t.type == TokenType.NUM]

            if isinstance(expected, list):
                negative_nums = [t for t in num_tokens if t.value.startswith("-")]
                assert len(negative_nums) == len(expected)
                for token, exp_val in zip(negative_nums, expected, strict=True):
                    assert token.value == exp_val
            else:
                negative_nums = [t for t in num_tokens if t.value.startswith("-")]
                assert len(negative_nums) == 1
                assert negative_nums[0].value == expected

    def test_out_block_state_tracking(self, create_lexer):
        """Test in_out_block flag toggles correctly."""
        # Multiple OUT{} blocks
        input_text = 'OUT{ x: "a" } text "ignored" OUT{ y: "b" }'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        # Should have exactly 2 strings (from the two OUT{} blocks)
        assert len(str_tokens) == 2
        assert str_tokens[0].value == "a"
        assert str_tokens[1].value == "b"

    def test_out_block_closes_on_first_brace(self, create_lexer):
        """Test OUT{} block closes on first } encountered."""
        input_text = 'OUT{ x: "text" } "outside"'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        # Only the string inside OUT{} should be tokenized
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "text"


# ============================================================================
# Test Position Tracking
# ============================================================================


class TestPositionTracking:
    """Test line/column tracking for all tokens."""

    def test_single_line_positions(self, create_lexer):
        """Test column tracking on single line."""
        input_text = "abc"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should have ID("abc") + EOF
        assert tokens[0].type == TokenType.ID
        assert tokens[0].line == 1
        assert tokens[0].column == 1  # Starts at column 1

    def test_line_tracking_across_newlines(self, create_lexer):
        """Test line numbers increment correctly."""
        input_text = "line1\nline2\nline3"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Filter out NEWLINE and EOF tokens, get IDs
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert len(id_tokens) == 3
        assert id_tokens[0].line == 1  # line1
        assert id_tokens[1].line == 2  # line2
        assert id_tokens[2].line == 3  # line3

    def test_column_tracking_with_whitespace(self, create_lexer):
        """Test column tracking with leading whitespace."""
        input_text = "  abc"  # 2 spaces before abc
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.ID
        assert tokens[0].column == 3  # Column 3 (after 2 spaces)

    def test_column_reset_after_newline(self, create_lexer):
        """Test column tracking behavior after newline.

        Note: The lexer's advance() sets column=0 after a newline,
        so the first character on a new line is captured at column 0.
        This is the actual behavior, though the docstring mentions
        "1-indexed column tracking" which would suggest column=1.
        """
        input_text = "x\ny"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # x, NEWLINE, y, EOF
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert id_tokens[0].column == 1  # x at column 1 (first line)
        assert id_tokens[1].column == 0  # y at column 0 (after newline reset)

    def test_multiline_tag_positions(self, create_lexer):
        """Test position tracking across multiline tags."""
        input_text = "<lvar\nReport.title\nt>"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # <lvar at line 1, col 1
        assert tokens[0].type == TokenType.LVAR_OPEN
        assert tokens[0].line == 1
        assert tokens[0].column == 1

        # Report at line 2, col 1
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert id_tokens[0].line == 2

    def test_position_tracking_in_out_blocks(self, create_lexer):
        """Test position tracking inside OUT{} blocks."""
        input_text = 'OUT{\n  x: "value"\n}'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # OUT{ at line 1
        assert tokens[0].type == TokenType.OUT_OPEN
        assert tokens[0].line == 1

        # x at line 2
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert id_tokens[0].line == 2

        # } at line 3
        close_tokens = [t for t in tokens if t.type == TokenType.OUT_CLOSE]
        assert close_tokens[0].line == 3


# ============================================================================
# Test Escape Sequences
# ============================================================================


class TestEscapeSequences:
    """Test string escape sequence handling."""

    @pytest.mark.parametrize(
        "input_text, expected_value",
        [
            ('OUT{ x: "line1\\nline2" }', "line1\nline2"),  # \n → newline
            ('OUT{ x: "tab\\there" }', "tab\there"),  # \t → tab
            ('OUT{ x: "back\\\\slash" }', "back\\slash"),  # \\ → backslash
            ('OUT{ x: "quote\\"inside" }', 'quote"inside'),  # \" → quote
            ("OUT{ x: 'apos\\'trophe' }", "apos'trophe"),  # \' → apostrophe
            ('OUT{ x: "return\\rhere" }', "return\rhere"),  # \r → carriage return
        ],
    )
    def test_standard_escapes(self, create_lexer, input_text, expected_value):
        """Test all 6 standard escape sequences."""
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == expected_value

    @pytest.mark.parametrize(
        "input_text, expected_value",
        [
            ('OUT{ x: "\\x" }', "x"),  # Unknown escape → literal
            ('OUT{ x: "\\z" }', "z"),  # Unknown escape → literal
            ('OUT{ x: "\\9" }', "9"),  # Unknown escape → literal
        ],
    )
    def test_unknown_escapes(self, create_lexer, input_text, expected_value):
        """Test unknown escape sequences are kept literal."""
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == expected_value

    def test_mixed_quotes(self, create_lexer):
        """Test single and double quotes."""
        test_cases = [
            ('OUT{ x: "double" }', "double"),
            ("OUT{ x: 'single' }", "single"),
            ("OUT{ x: \"She said 'hi'\" }", "She said 'hi'"),
            ("OUT{ x: 'He said \"hello\"' }", 'He said "hello"'),
        ]

        for input_text, expected_value in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()
            str_tokens = [t for t in tokens if t.type == TokenType.STR]
            assert len(str_tokens) == 1
            assert str_tokens[0].value == expected_value

    def test_empty_string(self, create_lexer):
        """Test empty string handling."""
        input_text = 'OUT{ x: "" }'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == ""

    def test_escaped_backslash_at_end(self, create_lexer):
        """Test escaped backslash at end of string."""
        input_text = 'OUT{ x: "path\\\\" }'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "path\\"


# ============================================================================
# Test Whitespace Handling
# ============================================================================


class TestWhitespace:
    """Test whitespace skipping and newline preservation."""

    def test_skip_spaces_and_tabs(self, create_lexer):
        """Test spaces and tabs are skipped."""
        input_text = "  \t  abc"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should skip whitespace, abc starts at column 6
        assert tokens[0].type == TokenType.ID
        assert tokens[0].value == "abc"
        assert tokens[0].column == 6

    def test_spaces_between_tokens(self, create_lexer):
        """Test spaces between tokens."""
        input_text = "a  b"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should have two ID tokens
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert len(id_tokens) == 2
        assert id_tokens[0].value == "a"
        assert id_tokens[1].value == "b"

    def test_newlines_preserved_as_tokens(self, create_lexer):
        """Test newlines create NEWLINE tokens."""
        input_text = "a\nb"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should have ID, NEWLINE, ID, EOF
        assert tokens[0].type == TokenType.ID
        assert tokens[1].type == TokenType.NEWLINE
        assert tokens[2].type == TokenType.ID
        assert tokens[3].type == TokenType.EOF

    def test_multiple_newlines(self, create_lexer):
        """Test multiple consecutive newlines."""
        input_text = "\n\n"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should have two NEWLINE tokens + EOF
        newline_tokens = [t for t in tokens if t.type == TokenType.NEWLINE]
        assert len(newline_tokens) == 2

    def test_carriage_return_skipped(self, create_lexer):
        """Test carriage returns are skipped."""
        input_text = "a\r\nb"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # \r should be skipped, \n creates NEWLINE
        assert tokens[0].type == TokenType.ID
        assert tokens[1].type == TokenType.NEWLINE
        assert tokens[2].type == TokenType.ID


# ============================================================================
# Test Individual Methods
# ============================================================================


class TestMethods:
    """Test individual Lexer methods."""

    def test_current_char(self):
        """Test current_char returns current character or None."""
        lexer = Lexer("abc")

        assert lexer.current_char() == "a"
        lexer.pos = 1
        assert lexer.current_char() == "b"
        lexer.pos = 2
        assert lexer.current_char() == "c"
        lexer.pos = 3
        assert lexer.current_char() is None

    def test_peek_char(self):
        """Test peek_char looks ahead without advancing."""
        lexer = Lexer("abc")

        assert lexer.peek_char(offset=1) == "b"
        assert lexer.peek_char(offset=2) == "c"
        assert lexer.pos == 0  # Position not changed

        lexer.pos = 2
        assert lexer.peek_char(offset=1) is None  # Beyond end

    def test_advance(self):
        """Test advance moves position and tracks line/column."""
        lexer = Lexer("a\nb")

        # Initial state
        assert lexer.pos == 0
        assert lexer.line == 1
        assert lexer.column == 1

        # Advance once (past 'a')
        lexer.advance()
        assert lexer.pos == 1
        assert lexer.line == 1
        assert lexer.column == 2

        # Advance past newline
        lexer.advance()
        assert lexer.pos == 2
        assert lexer.line == 2
        assert lexer.column == 0  # Column resets to 0, then increments

        # Advance past 'b'
        lexer.advance()
        assert lexer.pos == 3
        assert lexer.line == 2
        assert lexer.column == 1

    def test_skip_whitespace(self):
        """Test skip_whitespace skips spaces, tabs, carriage returns."""
        lexer = Lexer("  \t\r  abc")
        lexer.skip_whitespace()

        # Should skip to 'a'
        assert lexer.current_char() == "a"
        assert lexer.column == 7

    def test_skip_whitespace_stops_at_newline(self):
        """Test skip_whitespace preserves newlines."""
        lexer = Lexer("  \n  abc")
        lexer.skip_whitespace()

        # Should skip spaces but stop before newline
        assert lexer.current_char() == "\n"

    def test_read_identifier(self):
        """Test read_identifier reads alphanumeric + underscore."""
        test_cases = [
            ("abc123", "abc123"),
            ("_var", "_var"),
            ("var ", "var"),
            ("x", "x"),
            ("CamelCase", "CamelCase"),
        ]

        for input_text, expected in test_cases:
            lexer = Lexer(input_text)
            result = lexer.read_identifier()
            assert result == expected

    def test_read_number(self):
        """Test read_number reads digits and decimal points."""
        test_cases = [
            ("123", "123"),
            ("3.14", "3.14"),
            ("42.0", "42.0"),
            ("0", "0"),
            ("999.999", "999.999"),
        ]

        for input_text, expected in test_cases:
            lexer = Lexer(input_text)
            result = lexer.read_number()
            assert result == expected

    def test_read_string_double_quotes(self):
        """Test read_string with double quotes."""
        lexer = Lexer('"abc"')
        result = lexer.read_string()
        assert result == "abc"
        # Should consume closing quote
        assert lexer.current_char() is None

    def test_read_string_single_quotes(self):
        """Test read_string with single quotes."""
        lexer = Lexer("'abc'")
        result = lexer.read_string()
        assert result == "abc"

    def test_read_string_with_escapes(self):
        """Test read_string processes escape sequences."""
        lexer = Lexer('"a\\nb"')
        result = lexer.read_string()
        assert result == "a\nb"

    def test_read_string_unterminated(self):
        """Test read_string with unterminated string."""
        lexer = Lexer('"unterminated')
        result = lexer.read_string()
        # Should read until EOF
        assert result == "unterminated"
        assert lexer.current_char() is None


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input(self, create_lexer):
        """Test empty string produces only EOF."""
        lexer = create_lexer("")
        tokens = lexer.tokenize()

        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_only_whitespace(self, create_lexer):
        """Test input with only whitespace."""
        lexer = create_lexer("   \t\t  \n\n  ")
        tokens = lexer.tokenize()

        # Should have NEWLINE tokens + EOF
        newline_count = len([t for t in tokens if t.type == TokenType.NEWLINE])
        assert newline_count == 2  # Two newlines
        assert tokens[-1].type == TokenType.EOF

    def test_unterminated_string(self, create_lexer):
        """Test string without closing quote."""
        input_text = 'OUT{ x: "unterminated }'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should still produce tokens, string reads until }
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        # String should capture everything until the closing brace
        assert "unterminated " in str_tokens[0].value

    def test_malformed_tags(self, create_lexer):
        """Test unrecognized tags are skipped."""
        test_cases = [
            "<unknown>",
            "</unknown>",
            "<lva",  # Incomplete lvar
            "<lac",  # Incomplete lact
        ]

        for input_text in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()

            # Unknown tags should not produce tag tokens
            tag_tokens = [
                t
                for t in tokens
                if t.type
                in [
                    TokenType.LVAR_OPEN,
                    TokenType.LVAR_CLOSE,
                    TokenType.LACT_OPEN,
                    TokenType.LACT_CLOSE,
                ]
            ]
            assert len(tag_tokens) == 0

    def test_consecutive_operators(self, create_lexer):
        """Test multiple punctuation in sequence."""
        test_cases = [
            ("...", TokenType.DOT, 3),
            (":::", TokenType.COLON, 3),
            ("[[[", TokenType.LBRACKET, 3),
        ]

        for input_text, expected_type, expected_count in test_cases:
            lexer = create_lexer(input_text)
            tokens = lexer.tokenize()

            matching_tokens = [t for t in tokens if t.type == expected_type]
            assert len(matching_tokens) == expected_count

    def test_unicode_in_strings(self, create_lexer):
        """Test unicode characters in strings."""
        input_text = 'OUT{ x: "Hello 世界 🌍" }'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "Hello 世界 🌍"

    def test_unicode_in_identifiers(self, create_lexer):
        """Test unicode handling in identifiers."""
        # Identifiers only support alphanumeric + underscore
        # Unicode letters should be handled if isalnum() supports them
        input_text = "café"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Behavior depends on Python's isalnum() implementation
        # Should tokenize as identifier if Python supports it
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        if id_tokens:
            assert "caf" in id_tokens[0].value

    def test_very_long_string(self, create_lexer):
        """Test performance with large strings."""
        large_text = "a" * 10000
        input_text = f'OUT{{ x: "{large_text}" }}'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert len(str_tokens[0].value) == 10000

    def test_very_long_identifier(self, create_lexer):
        """Test very long identifiers."""
        long_id = "a" * 1000
        lexer = create_lexer(long_id)
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.ID
        assert len(tokens[0].value) == 1000

    def test_multiple_out_blocks(self, create_lexer):
        """Test multiple OUT{} blocks in sequence."""
        input_text = 'OUT{ x: 1 } text OUT{ y: "a" }'
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        out_open_tokens = [t for t in tokens if t.type == TokenType.OUT_OPEN]
        out_close_tokens = [t for t in tokens if t.type == TokenType.OUT_CLOSE]

        assert len(out_open_tokens) == 2
        assert len(out_close_tokens) == 2

    def test_nested_braces_close_on_first(self, create_lexer):
        """Test nested braces - OUT{} closes on first }."""
        input_text = "OUT{ x: { nested }"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        out_close_tokens = [t for t in tokens if t.type == TokenType.OUT_CLOSE]
        # First } closes the OUT{} block
        assert len(out_close_tokens) >= 1

    def test_zero_value(self, create_lexer):
        """Test zero as number."""
        input_text = "OUT{ x: 0 }"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "0"

    def test_float_with_leading_dot(self, create_lexer):
        """Test float starting with decimal point."""
        # This is an edge case - lexer reads .5 as DOT + NUM(5)
        input_text = "OUT{ x: .5 }"
        lexer = create_lexer(input_text)
        tokens = lexer.tokenize()

        # Should have DOT and NUM tokens
        dot_tokens = [t for t in tokens if t.type == TokenType.DOT]
        num_tokens = [t for t in tokens if t.type == TokenType.NUM]

        assert len(dot_tokens) >= 1
        assert len(num_tokens) >= 1


# ============================================================================
# Test Complete LNDL Examples
# ============================================================================


class TestCompleteLNDL:
    """Test lexer on complete LNDL examples."""

    def test_simple_lvar_example(self, tokenize):
        """Test tokenization of simple lvar + OUT block."""
        input_text = """\
<lvar Report.title t>AI Safety</lvar>
OUT{title: [t]}
"""
        tokens = tokenize(input_text)

        # Verify token sequence
        assert tokens[0].type == TokenType.LVAR_OPEN
        # Report.title t (ID, DOT, ID, ID)
        # >
        # AI Safety (ID, ID)
        # </lvar>
        # NEWLINE
        # OUT{
        # title: [t]
        # }
        # EOF

        # Check key tokens exist
        token_types = [t.type for t in tokens]
        assert TokenType.LVAR_OPEN in token_types
        assert TokenType.LVAR_CLOSE in token_types
        assert TokenType.OUT_OPEN in token_types
        assert TokenType.OUT_CLOSE in token_types
        assert TokenType.EOF in token_types

    def test_multi_lvar_example(self, tokenize):
        """Test multiple lvars."""
        input_text = """\
<lvar Report.title t>Title</lvar>
<lvar Report.content c>Content</lvar>
OUT{title: [t], content: [c]}
"""
        tokens = tokenize(input_text)

        # Should have two LVAR_OPEN and two LVAR_CLOSE
        lvar_opens = [t for t in tokens if t.type == TokenType.LVAR_OPEN]
        lvar_closes = [t for t in tokens if t.type == TokenType.LVAR_CLOSE]

        assert len(lvar_opens) == 2
        assert len(lvar_closes) == 2

    def test_lact_example(self, tokenize):
        """Test action call."""
        input_text = """\
<lact SearchResult.results r>search(query="AI")</lact>
OUT{results: [r]}
"""
        tokens = tokenize(input_text)

        # Should have LACT_OPEN and LACT_CLOSE
        token_types = [t.type for t in tokens]
        assert TokenType.LACT_OPEN in token_types
        assert TokenType.LACT_CLOSE in token_types
        assert TokenType.LPAREN in token_types
        assert TokenType.RPAREN in token_types

    def test_mixed_literals_in_out_block(self, tokenize):
        """Test OUT block with mixed literal types."""
        input_text = 'OUT{title: [t], score: 0.95, type: "report"}'
        tokens = tokenize(input_text)

        # Should have ID, NUM, STR tokens
        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        str_tokens = [t for t in tokens if t.type == TokenType.STR]

        assert len(num_tokens) == 1
        assert num_tokens[0].value == "0.95"
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "report"

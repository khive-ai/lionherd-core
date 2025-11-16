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


class TestTokenStreamUnclosedTags:
    """Test unclosed tag errors when token stream ends before closing tag.

    These tests target lines 347 and 442 which are only hit when:
    - Source text HAS the closing tag (so regex succeeds at line 333/428)
    - But token stream hits EOF before LVAR_CLOSE/LACT_CLOSE token (lines 345-347/440-442)

    This requires manually constructing a token stream without closing tag tokens.
    """

    def test_lvar_eof_in_token_stream_line_347(self):
        """Test line 347: Token stream EOF before LVAR_CLOSE (regex succeeds)."""
        from lionherd_core.lndl.lexer import Token, TokenType

        # Source has closing tag - regex will succeed at line 333
        source = "<lvar Report.title t>Content here</lvar>"

        # Manually construct token stream WITHOUT LVAR_CLOSE
        tokens = [
            Token(TokenType.LVAR_OPEN, "<lvar", 1, 1),
            Token(TokenType.ID, "Report", 1, 7),
            Token(TokenType.DOT, ".", 1, 13),
            Token(TokenType.ID, "title", 1, 14),
            Token(TokenType.ID, "t", 1, 20),
            Token(TokenType.GT, ">", 1, 21),
            Token(TokenType.ID, "Content", 1, 22),
            Token(TokenType.ID, "here", 1, 30),
            # EOF without LVAR_CLOSE - this triggers line 347
            Token(TokenType.EOF, "", 1, 48),
        ]

        parser = Parser(tokens, source_text=source)

        # parse_lvar() will:
        # 1. Extract content via regex (succeeds - source has </lvar>)
        # 2. Loop through tokens (line 345: while not self.match(LVAR_CLOSE))
        # 3. Hit EOF before finding LVAR_CLOSE -> line 347
        with pytest.raises(ParseError) as exc_info:
            parser.parse_lvar()

        assert "Unclosed lvar tag" in str(exc_info.value)
        assert "missing </lvar>" in str(exc_info.value)

    def test_lact_eof_in_token_stream_line_442(self):
        """Test line 442: Token stream EOF before LACT_CLOSE (regex succeeds)."""
        from lionherd_core.lndl.lexer import Token, TokenType

        # Source has closing tag - regex will succeed at line 428
        source = "<lact Report.process p>process(arg1, arg2)</lact>"

        # Manually construct token stream WITHOUT LACT_CLOSE
        tokens = [
            Token(TokenType.LACT_OPEN, "<lact", 1, 1),
            Token(TokenType.ID, "Report", 1, 7),
            Token(TokenType.DOT, ".", 1, 13),
            Token(TokenType.ID, "process", 1, 14),
            Token(TokenType.ID, "p", 1, 22),
            Token(TokenType.GT, ">", 1, 23),
            Token(TokenType.ID, "process", 1, 24),
            Token(TokenType.LPAREN, "(", 1, 31),
            Token(TokenType.ID, "arg1", 1, 32),
            Token(TokenType.COMMA, ",", 1, 36),
            Token(TokenType.ID, "arg2", 1, 38),
            Token(TokenType.RPAREN, ")", 1, 42),
            # EOF without LACT_CLOSE - this triggers line 442
            Token(TokenType.EOF, "", 1, 50),
        ]

        parser = Parser(tokens, source_text=source)

        # parse_lact() will:
        # 1. Extract call via regex (succeeds - source has </lact>)
        # 2. Loop through tokens (line 440: while not self.match(LACT_CLOSE))
        # 3. Hit EOF before finding LACT_CLOSE -> line 442
        with pytest.raises(ParseError) as exc_info:
            parser.parse_lact()

        assert "Unclosed lact tag" in str(exc_info.value)
        assert "missing </lact>" in str(exc_info.value)


class TestSkipNewlinesBreaks:
    """Test break statements after skip_newlines() in parsing loops.

    Lines 485 and 517 are break statements executed when:
    - Loop enters (condition false - not matching OUT_CLOSE/RBRACKET)
    - skip_newlines() consumes NEWLINE tokens
    - Then match succeeds (closing token or EOF found) -> break

    These require manually constructed token streams with specific NEWLINE placement.
    """

    def test_out_block_break_after_newlines_line_485(self):
        """Test line 485: break after skip_newlines() in OUT{} block.

        To hit line 485, we need:
        1. Enter the while loop (current token is NOT OUT_CLOSE initially)
        2. skip_newlines() at line 482 finds and consumes newlines
        3. After skip, current token IS OUT_CLOSE -> break executes

        Strategy: Use unexpected token (COMMA) that gets skipped via line 490,
        leaving parser at NEWLINE. Next iteration hits the break.
        """
        from lionherd_core.lndl.lexer import Token, TokenType

        # OUT{ with unexpected COMMA token, then newlines, then }
        source = "OUT{,\n}"

        tokens = [
            Token(TokenType.OUT_OPEN, "OUT{", 1, 1),
            Token(TokenType.COMMA, ",", 1, 5),  # Unexpected token - gets skipped
            Token(TokenType.NEWLINE, "\n", 1, 6),
            Token(TokenType.OUT_CLOSE, "}", 2, 1),
            Token(TokenType.EOF, "", 2, 2),
        ]

        parser = Parser(tokens, source_text=source)
        out_block = parser.parse_out_block()

        # Execution flow:
        # 1. Line 475: expect(OUT_OPEN) -> current = COMMA (position 1)
        # 2. Line 476: skip_newlines() -> no newlines, stays at COMMA
        # 3. Loop iteration 1:
        #    - Line 481: while not match(OUT_CLOSE) -> COMMA != OUT_CLOSE, enter loop
        #    - Line 482: skip_newlines() -> no newlines yet
        #    - Line 484: if match(OUT_CLOSE) -> COMMA != OUT_CLOSE, false
        #    - Line 488: if not match(ID) -> COMMA is not ID, true
        #    - Line 490: advance() -> current = NEWLINE (position 2)
        #    - Line 491: continue
        # 4. Loop iteration 2:
        #    - Line 481: while not match(OUT_CLOSE) -> NEWLINE != OUT_CLOSE, enter loop
        #    - Line 482: skip_newlines() -> consumes NEWLINE, current = OUT_CLOSE (position 3)
        #    - Line 484: if match(OUT_CLOSE) -> OUT_CLOSE == OUT_CLOSE, TRUE
        #    - Line 485: break -> EXECUTES THIS LINE!

        assert len(out_block.fields) == 0

    def test_array_break_after_newlines_line_517(self):
        """Test line 517: break after skip_newlines() in array.

        To hit line 517, we need:
        1. Enter the array parsing while loop (current token is NOT RBRACKET initially)
        2. skip_newlines() at line 514 finds and consumes newlines
        3. After skip, current token IS RBRACKET -> break executes

        Strategy: Similar to line 485 - use unexpected token in array that gets
        handled, leaving parser at NEWLINE. Next iteration hits the break.
        """
        from lionherd_core.lndl.lexer import Token, TokenType

        # Array with comma followed by newlines (no element after comma)
        source = "OUT{field: [val1,\n]}"

        tokens = [
            Token(TokenType.OUT_OPEN, "OUT{", 1, 1),
            Token(TokenType.ID, "field", 1, 5),
            Token(TokenType.COLON, ":", 1, 10),
            Token(TokenType.LBRACKET, "[", 1, 12),
            Token(TokenType.ID, "val1", 1, 13),
            Token(TokenType.COMMA, ",", 1, 17),
            Token(TokenType.NEWLINE, "\n", 1, 18),
            Token(TokenType.RBRACKET, "]", 2, 1),
            Token(TokenType.OUT_CLOSE, "}", 2, 2),
            Token(TokenType.EOF, "", 2, 3),
        ]

        parser = Parser(tokens, source_text=source)
        out_block = parser.parse_out_block()

        # Execution flow for array parsing (starts at line 507):
        # 1. Line 508: advance() past LBRACKET -> current = ID "val1"
        # 2. Line 509: skip_newlines() -> no newlines yet
        # 3. Loop iteration 1 (parsing val1):
        #    - Line 513: while not match(RBRACKET) -> ID != RBRACKET, enter loop
        #    - Line 514: skip_newlines() -> no newlines
        #    - Line 516: if match(RBRACKET) -> ID != RBRACKET, false
        #    - Line 519: if match(ID) -> true
        #    - Line 520: append "val1"
        #    - Line 521: advance() -> current = COMMA
        #    - Line 528: if match(COMMA) -> true
        #    - Line 529: advance() -> current = NEWLINE
        #    - (loop continues)
        # 4. Loop iteration 2:
        #    - Line 513: while not match(RBRACKET) -> NEWLINE != RBRACKET, enter loop
        #    - Line 514: skip_newlines() -> consumes NEWLINE, current = RBRACKET
        #    - Line 516: if match(RBRACKET) -> RBRACKET == RBRACKET, TRUE
        #    - Line 517: break -> EXECUTES THIS LINE!

        assert "field" in out_block.fields
        assert len(out_block.fields["field"]) == 1
        assert out_block.fields["field"][0] == "val1"

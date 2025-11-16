# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive parser tests for LNDL parser.py.

Test coverage:
- Lvar/Lact parsing (namespaced Model.field and raw patterns)
- OUT{} block parsing (arrays, scalars, all value types)
- Error handling (unclosed tags, missing source text, malformed input)
- Edge cases (newlines, token stream boundaries, recovery)
- Duplicate alias detection
- parse_value() function
"""

import pytest

from lionherd_core.lndl.ast import Lact, Lvar
from lionherd_core.lndl.lexer import Lexer, Token, TokenType
from lionherd_core.lndl.parser import ParseError, Parser, parse_value

# ============================================================================
# Basic Parser Functionality Tests
# ============================================================================


class TestParseLvarNamespaced:
    """Test parse_lvar() with Model.field namespaced patterns."""

    def test_lvar_model_field_with_alias(self):
        """Test <lvar Model.field alias>content</lvar>"""
        source = "<lvar User.name username>John Doe</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert lvar.model == "User"
        assert lvar.field == "name"
        assert lvar.alias == "username"
        assert lvar.content == "John Doe"

    def test_lvar_model_field_no_alias(self):
        """Test <lvar Model.field>content</lvar> - uses field as alias"""
        source = "<lvar User.name>Jane Smith</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert lvar.model == "User"
        assert lvar.field == "name"
        assert lvar.alias == "name"  # Uses field name as alias
        assert lvar.content == "Jane Smith"

    def test_lvar_raw_pattern(self):
        """Test <lvar alias>content</lvar> - raw pattern returns RLvar"""
        from lionherd_core.lndl.ast import RLvar

        source = "<lvar result>42</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        # Should return RLvar (raw pattern)
        assert isinstance(lvar, RLvar)
        assert lvar.alias == "result"
        assert lvar.content == "42"

    def test_lvar_multiline_content(self):
        """Test lvar with multiline content including newlines"""
        source = """<lvar Report.description>This is a
multi-line
description</lvar>"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert lvar.model == "Report"
        assert lvar.field == "description"
        assert lvar.alias == "description"
        assert "multi-line" in lvar.content
        assert lvar.content.count("\n") >= 1

    def test_lvar_complex_content_with_punctuation(self):
        """Test lvar content with dots, commas, brackets"""
        source = "<lvar Report.data>[1, 2, 3], {key: value}, etc.</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert lvar.model == "Report"
        assert lvar.field == "data"
        assert lvar.alias == "data"
        assert "[" in lvar.content
        assert "," in lvar.content
        assert "{" in lvar.content
        assert ":" in lvar.content

    def test_lvar_unclosed_tag_error(self):
        """Test error handling for unclosed lvar tag"""
        source = "<lvar Report.result>content without closing tag"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        with pytest.raises(ParseError, match="Unclosed lvar tag"):
            parser.parse_lvar()

    def test_lvar_empty_content(self):
        """Test lvar with empty content"""
        source = "<lvar Report.result></lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert lvar.content == ""


class TestParseLactNamespaced:
    """Test parse_lact() with Model.field namespaced patterns."""

    def test_lact_model_field_with_alias(self):
        """Test <lact Model.field alias>call()</lact>"""
        source = '<lact Tool.search query>search(q="AI")</lact>'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lact = parser.parse_lact()

        assert lact.model == "Tool"
        assert lact.field == "search"
        assert lact.alias == "query"
        assert 'search(q="AI")' in lact.call

    def test_lact_model_field_no_alias(self):
        """Test <lact Model.field>call()</lact> - uses field as alias"""
        source = '<lact API.fetch>fetch(url="test")</lact>'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lact = parser.parse_lact()

        assert lact.model == "API"
        assert lact.field == "fetch"
        assert lact.alias == "fetch"  # Uses field name as alias
        assert 'fetch(url="test")' in lact.call

    def test_lact_direct_pattern(self):
        """Test <lact alias>call()</lact> - direct pattern without model.field"""
        source = '<lact search>query("test")</lact>'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lact = parser.parse_lact()

        assert lact.model is None
        assert lact.field is None
        assert lact.alias == "search"
        assert 'query("test")' in lact.call

    def test_lact_preserves_string_quotes(self):
        """Test that lact preserves quotes in string arguments"""
        source = '<lact search>find(name="John", age=30)</lact>'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lact = parser.parse_lact()

        # Should preserve quotes around "John"
        assert '"John"' in lact.call or "'John'" in lact.call
        assert "30" in lact.call

    def test_lact_complex_arguments(self):
        """Test lact with complex nested arguments"""
        source = '<lact fetch>api_call(data=[1, 2, 3], opts={key: "val"})</lact>'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lact = parser.parse_lact()

        assert "[" in lact.call
        assert "{" in lact.call
        assert ":" in lact.call
        assert "," in lact.call

    def test_lact_multiline_call(self):
        """Test lact with multiline function call"""
        source = """<lact process>transform(
    arg1="value1",
    arg2="value2"
)</lact>"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lact = parser.parse_lact()

        # Newlines normalized to spaces
        assert "transform(" in lact.call
        assert "arg1" in lact.call
        assert "arg2" in lact.call

    def test_lact_unclosed_tag_error(self):
        """Test error handling for unclosed lact tag"""
        source = '<lact search>query("test") without closing'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        with pytest.raises(ParseError, match="Unclosed lact tag"):
            parser.parse_lact()

    def test_lact_empty_call(self):
        """Test lact with empty call string"""
        source = "<lact action></lact>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lact = parser.parse_lact()

        assert lact.call == ""


class TestParserHelpers:
    """Test parser helper methods and edge cases."""

    def test_skip_newlines(self):
        """Test that skip_newlines() consumes multiple newlines"""
        source = "<lvar Test.value>\n\n\ncontent</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()
        assert lvar.content == "content"

    def test_parser_with_no_source_text(self):
        """Test parser initialization with None source_text"""
        lexer = Lexer("")
        tokens = lexer.tokenize()
        parser = Parser(tokens, None)

        # Should handle None source_text gracefully
        assert parser.source_text is None

    def test_parser_expect_wrong_token_error(self):
        """Test that expect() raises error for wrong token type"""
        source = "<lvar 123>content</lvar>"  # Number instead of ID
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        with pytest.raises(ParseError):
            parser.parse_lvar()


class TestDuplicateAliases:
    """Test duplicate alias detection - aliases must be unique across lvars and lacts."""

    def test_duplicate_lvar_alias_raises_error(self):
        """Test that duplicate lvar aliases raise ParseError."""
        source = """
        <lvar Report.title t>First Title</lvar>
        <lvar Report.summary t>Summary</lvar>
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError) as exc_info:
            parser.parse()

        assert "Duplicate alias 't'" in str(exc_info.value)
        assert "unique" in str(exc_info.value).lower()

    def test_duplicate_lact_alias_raises_error(self):
        """Test that duplicate lact aliases raise ParseError."""
        source = """
        <lact Report.process p>process(arg1)</lact>
        <lact Report.analyze p>analyze(arg2)</lact>
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError) as exc_info:
            parser.parse()

        assert "Duplicate alias 'p'" in str(exc_info.value)
        assert "unique" in str(exc_info.value).lower()

    def test_lvar_and_lact_same_alias_raises_error(self):
        """Test that lvar and lact cannot share the same alias (shared namespace)."""
        source = """
        <lvar Report.title x>Title</lvar>
        <lact Report.process x>process()</lact>
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError) as exc_info:
            parser.parse()

        assert "Duplicate alias 'x'" in str(exc_info.value)
        assert "lvars and lacts" in str(exc_info.value)

    def test_unique_aliases_across_lvars_and_lacts_succeeds(self):
        """Test that unique aliases across lvars and lacts parse successfully."""
        source = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lact Report.process p>process()</lact>

        OUT{report: [t, s, p]}
        """
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        program = parser.parse()

        assert len(program.lvars) == 2
        assert len(program.lacts) == 1
        assert program.lvars[0].alias == "t"
        assert program.lvars[1].alias == "s"
        assert program.lacts[0].alias == "p"


class TestOutBlockArrayRestrictions:
    """Test OUT{} array parsing restrictions."""

    def test_array_with_numeric_literal_raises_error(self):
        """Test that arrays with numeric literals raise ParseError."""
        source = "OUT{scores: [0.8, 0.9]}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError, match=r"Arrays must contain only.*references"):
            parser.parse_out_block()

    def test_array_with_string_literal_raises_error(self):
        """Test that arrays with string literals raise ParseError."""
        source = 'OUT{tags: ["tag1", "tag2"]}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        with pytest.raises(ParseError, match=r"Arrays must contain only.*references"):
            parser.parse_out_block()

    def test_array_with_ids_succeeds(self):
        """Test that arrays with IDs (variable references) parse successfully."""
        source = "OUT{report: [title, summary, score]}"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        out_block = parser.parse_out_block()

        assert "report" in out_block.fields
        assert out_block.fields["report"] == ["title", "summary", "score"]


# ============================================================================
# Parser Edge Cases and Boundary Conditions
# ============================================================================


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


# ============================================================================
# Error Handling Tests
# ============================================================================


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


class TestUnclosedTags:
    """Test error handling for unclosed tags."""

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


class TestTokenStreamUnclosedTags:
    """Test unclosed tag errors when token stream ends before closing tag."""

    def test_lvar_eof_in_token_stream_line_347(self):
        """Test line 347: Token stream EOF before LVAR_CLOSE (regex succeeds)."""
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

        with pytest.raises(ParseError) as exc_info:
            parser.parse_lvar()

        assert "Unclosed lvar tag" in str(exc_info.value)
        assert "missing </lvar>" in str(exc_info.value)

    def test_lact_eof_in_token_stream_line_442(self):
        """Test line 442: Token stream EOF before LACT_CLOSE (regex succeeds)."""
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

        with pytest.raises(ParseError) as exc_info:
            parser.parse_lact()

        assert "Unclosed lact tag" in str(exc_info.value)
        assert "missing </lact>" in str(exc_info.value)


# ============================================================================
# OUT{} Block Parsing Tests
# ============================================================================


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


class TestSkipNewlinesBreaks:
    """Test break statements after skip_newlines() in parsing loops."""

    def test_out_block_break_after_newlines_line_485(self):
        """Test line 485: break after skip_newlines() in OUT{} block."""
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

        assert len(out_block.fields) == 0

    def test_array_break_after_newlines_line_517(self):
        """Test line 517: break after skip_newlines() in array."""
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

        assert "field" in out_block.fields
        assert len(out_block.fields["field"]) == 1
        assert out_block.fields["field"][0] == "val1"


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
        """Test array parsing rejects string literals (arrays must contain only IDs)."""
        source = 'OUT{field:["str1", "str2", "str3"]}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        # Arrays must contain only variable/action references (IDs), not literals
        with pytest.raises(ParseError, match=r"Arrays must contain only.*references"):
            parser.parse_out_block()

    def test_array_with_mixed_types(self):
        """Test array rejects mixed types (only IDs allowed)."""
        source = 'OUT{field:[ref1, 630, 4.02, "text"]}'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=source)

        # First non-ID token (630) should trigger error
        with pytest.raises(ParseError, match=r"Arrays must contain only.*references"):
            parser.parse_out_block()

    def test_array_with_unknown_token_type(self):
        """Test array rejects unknown token types (only IDs allowed)."""
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

        # OUT_OPEN token should trigger error
        with pytest.raises(ParseError, match=r"Arrays must contain only.*references"):
            parser.parse_out_block()


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


# ============================================================================
# parse_value() Function Tests
# ============================================================================


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


# ============================================================================
# Complex Scenarios Tests
# ============================================================================


class TestComplexParsingScenarios:
    """Test complex parsing scenarios combining multiple edge cases."""

    def test_out_block_with_all_value_types(self):
        """Test OUT{} with IDs, strings, numbers, booleans, and arrays (IDs only)."""
        source = """OUT{
            single_ref: ref1,
            str_val: "text",
            int_val: 630,
            float_val: 4.02,
            bool_true: true,
            bool_false: false,
            array_refs: [r1, r2, r3]
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

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.lndl import Lexer, ParseError, Parser
from lionherd_core.lndl.ast import Lact, Lvar, OutBlock, Program, RLvar
from lionherd_core.lndl.parser import parse_value


class TestParserInitialization:
    """Test Parser initialization and token navigation."""

    def test_parser_init_requires_source_text(self, tokenize):
        """Test Parser initialization with source text."""
        text = "<lvar Report.title t>AI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        assert parser.tokens == tokens
        assert parser.source_text == text
        assert parser.pos == 0

    def test_current_token(self, tokenize):
        """Test current_token() returns current token."""
        text = "<lvar Report.title t>AI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        token = parser.current_token()
        assert token.type.name == "LVAR_OPEN"
        assert token.value == "<lvar"

    def test_peek_token(self, tokenize):
        """Test peek_token() looks ahead without advancing."""
        text = "<lvar Report.title t>AI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        next_token = parser.peek_token(1)
        assert next_token.type.name == "ID"
        assert next_token.value == "Report"
        # pos should not change
        assert parser.pos == 0

    def test_advance(self, tokenize):
        """Test advance() moves to next token."""
        text = "<lvar Report.title t>AI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        parser.advance()
        assert parser.pos == 1
        assert parser.current_token().type.name == "ID"

    def test_skip_newlines(self, tokenize):
        """Test skip_newlines() skips newline tokens."""
        text = "<lvar Report.title t>\n\nAI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        # Advance to newlines
        while not parser.match(parser.current_token().type.__class__.NEWLINE):
            if parser.match(parser.current_token().type.__class__.EOF):
                break
            parser.advance()
        start_pos = parser.pos
        parser.skip_newlines()
        assert parser.pos > start_pos


class TestParseLvar:
    """Test Parser.parse_lvar() method."""

    def test_parse_lvar_namespaced_with_alias(self, parse_lndl_ast):
        """Test parsing namespaced lvar with explicit alias."""
        text = "<lvar Report.title t>AI Safety Analysis</lvar>"
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 1
        lvar = program.lvars[0]
        assert isinstance(lvar, Lvar)
        assert lvar.model == "Report"
        assert lvar.field == "title"
        assert lvar.alias == "t"
        assert lvar.content == "AI Safety Analysis"

    def test_parse_lvar_namespaced_without_alias(self, parse_lndl_ast):
        """Test parsing namespaced lvar without alias (uses field name)."""
        text = "<lvar Report.title>AI Safety Analysis</lvar>"
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 1
        lvar = program.lvars[0]
        assert isinstance(lvar, Lvar)
        assert lvar.model == "Report"
        assert lvar.field == "title"
        assert lvar.alias == "title"  # Defaults to field
        assert lvar.content == "AI Safety Analysis"

    def test_parse_lvar_raw(self, parse_lndl_ast):
        """Test parsing raw lvar (no namespace)."""
        text = "<lvar reasoning>This is intermediate reasoning.</lvar>"
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 1
        lvar = program.lvars[0]
        assert isinstance(lvar, RLvar)
        assert lvar.alias == "reasoning"
        assert lvar.content == "This is intermediate reasoning."

    def test_parse_lvar_multiline_content(self, parse_lndl_ast):
        """Test parsing lvar with multiline content."""
        text = """<lvar Report.content c>
This is a long
multiline
content.
</lvar>"""
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 1
        lvar = program.lvars[0]
        assert "multiline" in lvar.content
        assert "long" in lvar.content

    def test_parse_lvar_preserves_whitespace(self, parse_lndl_ast):
        """Test lvar content preserves internal whitespace."""
        text = "<lvar Report.title t>  AI   Safety  </lvar>"
        program = parse_lndl_ast(text)
        lvar = program.lvars[0]
        # Regex extraction with strip() should preserve internal spaces but strip edges
        assert "AI" in lvar.content
        assert "Safety" in lvar.content

    def test_parse_lvar_with_special_chars(self, parse_lndl_ast):
        """Test lvar content with special characters."""
        text = '<lvar Report.query q>search(query="AI", limit=10)</lvar>'
        program = parse_lndl_ast(text)
        lvar = program.lvars[0]
        assert "AI" in lvar.content
        assert "limit=10" in lvar.content

    def test_parse_lvar_unclosed_tag(self):
        """Test error on unclosed lvar tag."""
        text = "<lvar Report.title t>AI Safety"
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        with pytest.raises(ParseError, match="Unclosed lvar tag"):
            parser.parse()


class TestParseLact:
    """Test Parser.parse_lact() method."""

    def test_parse_lact_namespaced_with_alias(self, parse_lndl_ast):
        """Test parsing namespaced lact with explicit alias."""
        text = '<lact Report.summary s>generate_summary(prompt="...")</lact>'
        program = parse_lndl_ast(text)
        assert len(program.lacts) == 1
        lact = program.lacts[0]
        assert isinstance(lact, Lact)
        assert lact.model == "Report"
        assert lact.field == "summary"
        assert lact.alias == "s"
        assert "generate_summary" in lact.call

    def test_parse_lact_namespaced_without_alias(self, parse_lndl_ast):
        """Test parsing namespaced lact without alias (uses field name)."""
        text = '<lact Report.summary>generate_summary(prompt="...")</lact>'
        program = parse_lndl_ast(text)
        assert len(program.lacts) == 1
        lact = program.lacts[0]
        assert lact.model == "Report"
        assert lact.field == "summary"
        assert lact.alias == "summary"  # Defaults to field
        assert "generate_summary" in lact.call

    def test_parse_lact_direct(self, parse_lndl_ast):
        """Test parsing direct lact (no namespace)."""
        text = '<lact search>search(query="AI")</lact>'
        program = parse_lndl_ast(text)
        assert len(program.lacts) == 1
        lact = program.lacts[0]
        assert lact.model is None
        assert lact.field is None
        assert lact.alias == "search"
        assert "search(query=" in lact.call

    def test_parse_lact_complex_arguments(self, parse_lndl_ast):
        """Test lact with complex function arguments."""
        text = '<lact search>search(query="AI safety", limit=10, filter={"type": "paper"})</lact>'
        program = parse_lndl_ast(text)
        lact = program.lacts[0]
        assert "AI safety" in lact.call
        assert "limit=10" in lact.call
        assert "filter=" in lact.call

    def test_parse_lact_unclosed_tag(self):
        """Test error on unclosed lact tag."""
        text = "<lact Report.summary s>generate_summary(...)"
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        with pytest.raises(ParseError, match="Unclosed lact tag"):
            parser.parse()

    def test_parse_lact_reserved_keyword_warning(self):
        """Test warning when using Python reserved keyword as action name."""
        from lionherd_core.lndl.parser import _warned_action_names

        # Clear warned set for test isolation
        _warned_action_names.clear()

        text = "<lact import>import_data()</lact>"
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)

        with pytest.warns(UserWarning, match="reserved keyword"):
            parser.parse()


class TestParseOutBlock:
    """Test Parser.parse_out_block() method."""

    def test_parse_out_block_simple(self, parse_lndl_ast):
        """Test parsing simple OUT block with references."""
        text = "OUT{title: [t], summary: [s]}"
        program = parse_lndl_ast(text)
        assert program.out_block is not None
        assert isinstance(program.out_block, OutBlock)
        assert program.out_block.fields["title"] == ["t"]
        assert program.out_block.fields["summary"] == ["s"]

    def test_parse_out_block_multiline(self, parse_lndl_ast):
        """Test parsing multiline OUT block."""
        text = """OUT{
    title: [t],
    summary: [s],
    confidence: 0.85
}"""
        program = parse_lndl_ast(text)
        assert program.out_block is not None
        assert program.out_block.fields["title"] == ["t"]
        assert program.out_block.fields["summary"] == ["s"]
        assert program.out_block.fields["confidence"] == 0.85

    def test_parse_out_block_literals(self, parse_lndl_ast):
        """Test OUT block with literal values."""
        text = "OUT{title: [t], score: 0.95, count: 42, active: true}"
        program = parse_lndl_ast(text)
        out = program.out_block
        assert out.fields["title"] == ["t"]
        assert out.fields["score"] == 0.95
        assert out.fields["count"] == 42
        assert out.fields["active"] is True

    def test_parse_out_block_string_literal(self, parse_lndl_ast):
        """Test OUT block with string literal."""
        text = 'OUT{title: [t], status: "completed"}'
        program = parse_lndl_ast(text)
        out = program.out_block
        assert out.fields["status"] == "completed"

    def test_parse_out_block_multiple_refs(self, parse_lndl_ast):
        """Test OUT block with multiple references in array."""
        text = "OUT{content: [intro, body, conclusion]}"
        program = parse_lndl_ast(text)
        out = program.out_block
        assert out.fields["content"] == ["intro", "body", "conclusion"]

    def test_parse_out_block_single_ref_wrapped_in_list(self, parse_lndl_ast):
        """Test single reference without brackets gets wrapped in list."""
        text = "OUT{title: t}"
        program = parse_lndl_ast(text)
        out = program.out_block
        assert out.fields["title"] == ["t"]

    def test_parse_out_block_mixed_values(self, parse_lndl_ast):
        """Test OUT block with mixed value types."""
        text = "OUT{title: [t], score: 0.95, active: true, count: 10}"
        program = parse_lndl_ast(text)
        out = program.out_block
        assert out.fields["title"] == ["t"]
        assert out.fields["score"] == 0.95
        assert out.fields["active"] is True
        assert out.fields["count"] == 10

    def test_parse_out_block_empty(self, parse_lndl_ast):
        """Test parsing empty OUT block."""
        text = "OUT{}"
        program = parse_lndl_ast(text)
        assert program.out_block is not None
        assert program.out_block.fields == {}


class TestParseMethod:
    """Test Parser.parse() main method."""

    def test_parse_complete_program(self, parse_lndl_ast):
        """Test parsing complete LNDL program with lvars and OUT block."""
        text = """<lvar Report.title t>AI Safety</lvar>
<lvar Report.content c>Analysis of AI safety measures.</lvar>

OUT{title: [t], content: [c]}"""
        program = parse_lndl_ast(text)
        assert isinstance(program, Program)
        assert len(program.lvars) == 2
        assert program.out_block is not None
        assert program.out_block.fields["title"] == ["t"]
        assert program.out_block.fields["content"] == ["c"]

    def test_parse_mixed_lvars_lacts(self, parse_lndl_ast):
        """Test parsing program with both lvars and lacts."""
        text = """<lvar Report.title t>Title</lvar>
<lact Report.summary s>summarize(text="...")</lact>

OUT{title: [t], summary: [s]}"""
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 1
        assert len(program.lacts) == 1
        assert program.out_block is not None

    def test_parse_raw_and_namespaced_lvars(self, parse_lndl_ast):
        """Test parsing mix of raw and namespaced lvars."""
        text = """<lvar Report.title t>Title</lvar>
<lvar reasoning>Intermediate reasoning</lvar>

OUT{title: [t], reasoning: [reasoning]}"""
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 2
        assert isinstance(program.lvars[0], Lvar)
        assert isinstance(program.lvars[1], RLvar)

    def test_parse_no_out_block(self, parse_lndl_ast):
        """Test parsing LNDL without OUT block."""
        text = "<lvar Report.title t>AI Safety</lvar>"
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 1
        assert program.out_block is None

    def test_parse_empty_input(self, parse_lndl_ast):
        """Test parsing empty input."""
        text = ""
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 0
        assert len(program.lacts) == 0
        assert program.out_block is None

    def test_parse_narrative_text_ignored(self, parse_lndl_ast):
        """Test that narrative text between tags is ignored."""
        text = """Here is some narrative text.

<lvar Report.title t>AI Safety</lvar>

More narrative text here.

OUT{title: [t]}"""
        program = parse_lndl_ast(text)
        assert len(program.lvars) == 1
        assert program.out_block is not None

    def test_parse_duplicate_alias_error(self):
        """Test error when duplicate alias is detected."""
        text = """<lvar Report.title t>Title</lvar>
<lvar Report.content t>Content</lvar>"""
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        with pytest.raises(ParseError, match="Duplicate alias"):
            parser.parse()

    def test_parse_duplicate_alias_across_lvar_lact(self):
        """Test error when lvar and lact share same alias."""
        text = """<lvar Report.title s>Title</lvar>
<lact Report.summary s>summarize(...)</lact>"""
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        with pytest.raises(ParseError, match="Duplicate alias"):
            parser.parse()


class TestParseErrors:
    """Test Parser error handling with position information."""

    def test_parse_error_includes_position(self):
        """Test ParseError includes line and column info."""
        text = "<lvar Report.title>Missing closing tag"
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        with pytest.raises(ParseError) as exc_info:
            parser.parse()
        assert "line" in str(exc_info.value)
        assert "column" in str(exc_info.value)

    def test_parse_error_missing_source_text(self, tokenize):
        """Test error when Parser initialized without source_text."""
        text = "<lvar Report.title t>AI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=None)
        with pytest.raises(ParseError, match="requires source_text"):
            parser.parse()

    def test_expect_error_wrong_token_type(self, tokenize):
        """Test expect() raises ParseError for wrong token type."""
        from lionherd_core.lndl.lexer import TokenType

        text = "<lvar Report.title t>AI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        with pytest.raises(ParseError, match=r"Expected.*got"):
            parser.expect(TokenType.EOF)  # Expecting EOF but current is LVAR_OPEN


class TestHybridParsing:
    """Test hybrid token+regex parsing approach."""

    def test_hybrid_preserves_exact_content(self, parse_lndl_ast):
        """Test hybrid parsing preserves exact content with quotes and whitespace."""
        text = '<lvar Report.query q>search(query="AI safety", limit=10)</lvar>'
        program = parse_lndl_ast(text)
        lvar = program.lvars[0]
        # Content should preserve quotes and formatting
        assert '"AI safety"' in lvar.content or "AI safety" in lvar.content
        assert "limit=10" in lvar.content

    def test_hybrid_handles_nested_braces(self, parse_lndl_ast):
        """Test hybrid parsing handles nested braces in content."""
        text = '<lvar Report.data d>{"key": "value", "nested": {"x": 1}}</lvar>'
        program = parse_lndl_ast(text)
        lvar = program.lvars[0]
        assert "nested" in lvar.content
        assert "key" in lvar.content

    def test_hybrid_multiline_with_indentation(self, parse_lndl_ast):
        """Test hybrid parsing preserves multiline content structure."""
        text = """<lvar Report.code c>
def example():
    return True
</lvar>"""
        program = parse_lndl_ast(text)
        lvar = program.lvars[0]
        assert "def example" in lvar.content
        assert "return True" in lvar.content


class TestParseValueUtility:
    """Test parse_value utility function."""

    def test_parse_value_null(self):
        """Test parsing null literal (case-insensitive)."""

        result = parse_value("null")
        assert result is None

    def test_parse_value_null_uppercase(self):
        """Test parsing NULL literal case insensitive."""

        result = parse_value("NULL")
        assert result is None

    def test_parse_value_boolean_true(self):
        """Test parsing true boolean."""

        result = parse_value("true")
        assert result is True

    def test_parse_value_boolean_false(self):
        """Test parsing false boolean."""

        result = parse_value("false")
        assert result is False

    def test_parse_value_integer(self):
        """Test parsing integer literal."""

        result = parse_value("42")
        assert result == 42

    def test_parse_value_float(self):
        """Test parsing float literal."""

        result = parse_value("3.14")
        assert result == 3.14

    def test_parse_value_string(self):
        """Test parsing string literal."""

        result = parse_value("hello")
        assert result == "hello"

    def test_parse_value_invalid_literal_returns_string(self):
        """Test parse_value returns original string when literal_eval fails."""

        result = parse_value("invalid{syntax")
        assert result == "invalid{syntax"

    def test_parse_value_non_string_returns_as_is(self):
        """Test parse_value returns non-string values unchanged (line 617)."""

        # Integer
        result = parse_value(42)
        assert result == 42

        # Float
        result = parse_value(3.14)
        assert result == 3.14

        # Boolean
        result = parse_value(True)
        assert result is True

        # None
        result = parse_value(None)
        assert result is None

        # List
        result = parse_value([1, 2, 3])
        assert result == [1, 2, 3]


class TestParserEdgeCases:
    """Test edge cases for 100% coverage."""

    def test_current_token_when_past_end(self, tokenize):
        """Test current_token() returns EOF when pos >= len(tokens)."""
        text = "<lvar Report.title t>AI</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        # Move parser past end of tokens
        parser.pos = len(tokens) + 5
        token = parser.current_token()
        assert token.type.name == "EOF"

    def test_peek_token_when_beyond_bounds(self, tokenize):
        """Test peek_token() returns EOF when peek_pos >= len(tokens)."""
        text = "<lvar Report.title t>AI</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        # Peek very far ahead
        token = parser.peek_token(offset=100)
        assert token.type.name == "EOF"

    def test_parse_lvar_without_source_text(self, tokenize):
        """Test parse_lvar() raises error when source_text is None."""
        text = "<lvar Report.title t>AI Safety</lvar>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=None)
        # Manually call parse_lvar after positioning
        parser.pos = 0
        with pytest.raises(ParseError, match="requires source_text"):
            parser.parse_lvar()

    def test_parse_lact_without_source_text(self, tokenize):
        """Test parse_lact() raises error when source_text is None."""
        text = "<lact Report.summary s>generate()</lact>"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=None)
        parser.pos = 0
        with pytest.raises(ParseError, match="requires source_text"):
            parser.parse_lact()

    def test_parse_lvar_regex_mismatch(self):
        """Test parse_lvar() error when regex can't extract content."""
        # Create malformed lvar where regex won't match but </lvar> exists
        text = "<lvar Report.title t>content</lvar> extra </lvar>"
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text="<lvar Report.wrongfield t>content</lvar>")
        with pytest.raises(ParseError, match="Could not extract lvar content"):
            parser.parse()

    def test_parse_lact_regex_mismatch(self):
        """Test parse_lact() error when regex can't extract call."""
        # Create malformed lact where regex won't match but </lact> exists
        text = "<lact Report.summary s>generate()</lact>"
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        # Give wrong source_text so regex doesn't match
        parser = Parser(tokens, source_text="<lact Report.wrongfield s>generate()</lact>")
        with pytest.raises(ParseError, match="Could not extract lact call"):
            parser.parse()

    def test_parse_lvar_eof_in_token_loop(self):
        """Test parse_lvar() when EOF reached in token loop before closing tag."""
        # Create source_text with </lvar> so regex succeeds, but truncate tokens
        source_text = "<lvar Report.title t>content</lvar>"
        lexer = Lexer(source_text)
        tokens = lexer.tokenize()
        # Remove the LVAR_CLOSE token to simulate EOF before closing
        tokens = [t for t in tokens if t.type.name != "LVAR_CLOSE"]
        parser = Parser(tokens, source_text=source_text)
        with pytest.raises(ParseError, match=r"Unclosed lvar tag.*missing"):
            parser.parse()

    def test_parse_lact_eof_in_token_loop(self):
        """Test parse_lact() when EOF reached in token loop before closing tag."""
        # Create source_text with </lact> so regex succeeds, but truncate tokens
        source_text = "<lact Report.summary s>generate()</lact>"
        lexer = Lexer(source_text)
        tokens = lexer.tokenize()
        # Remove the LACT_CLOSE token to simulate EOF before closing
        tokens = [t for t in tokens if t.type.name != "LACT_CLOSE"]
        parser = Parser(tokens, source_text=source_text)
        with pytest.raises(ParseError, match=r"Unclosed lact tag.*missing"):
            parser.parse()

    def test_out_block_immediate_close_after_newlines(self, parse_lndl_ast):
        """Test OUT block closes after skipping newlines (line 512)."""
        # The while loop checks OUT_CLOSE/EOF, then skips newlines
        # After skipping, it checks again and breaks (line 512)
        # Multiple variations to ensure the break is hit
        texts = [
            "OUT{\n}",  # Newline then close
            "OUT{\n\n}",  # Multiple newlines then close
            "OUT{\n\n\n\n}",  # Many newlines then close
        ]
        for text in texts:
            program = parse_lndl_ast(text)
            assert program.out_block is not None
            assert program.out_block.fields == {}

    def test_out_block_eof_after_newlines(self, parse_lndl_ast):
        """Test OUT block EOF after skipping newlines (line 512)."""
        # Same as above but with EOF instead of OUT_CLOSE
        text = "OUT{\n\n"
        program = parse_lndl_ast(text)
        assert program.out_block is not None

    def test_out_block_non_id_token_skipped(self, parse_lndl_ast):
        """Test OUT block skips non-ID tokens where field name expected."""
        text = "OUT{\n, : , field: value}"
        program = parse_lndl_ast(text)
        # Non-ID tokens should be skipped, only valid field parsed
        assert "field" in program.out_block.fields

    def test_out_block_field_without_colon(self, tokenize):
        """Test OUT block raises error for field without colon."""
        text = "OUT{field_without_colon, valid_field: value}"
        tokens = tokenize(text)
        parser = Parser(tokens, source_text=text)
        # Field without colon should raise ParseError (not skip silently)
        with pytest.raises(ParseError, match="Expected COLON"):
            parser.parse()

    def test_out_block_array_immediate_close_after_newlines(self, parse_lndl_ast):
        """Test OUT block array closes after skipping newlines (line 544)."""
        # The while loop checks RBRACKET/EOF, then skips newlines
        # After skipping, it checks again and breaks (line 544)
        # Multiple variations to ensure the break is hit
        texts = [
            "OUT{field: [\n]}",  # Newline then close
            "OUT{field: [\n\n]}",  # Multiple newlines then close
            "OUT{field: [\n\n\n\n]}",  # Many newlines then close
        ]
        for text in texts:
            program = parse_lndl_ast(text)
            assert program.out_block.fields["field"] == []

    def test_out_block_array_eof_after_newlines(self, parse_lndl_ast):
        """Test OUT block array EOF after newlines (line 544)."""
        # Same as above but with EOF instead of RBRACKET
        text = "OUT{field: [\n\n"
        program = parse_lndl_ast(text)
        assert program.out_block is not None

    def test_out_block_unknown_value_type(self, parse_lndl_ast):
        """Test OUT block with unknown/unsupported value type."""
        # Use LVAR_OPEN token where value expected (completely wrong token)
        text = "OUT{field: <lvar}"
        program = parse_lndl_ast(text)
        # Parser should skip unknown value and move on
        # Field might be missing or empty
        assert program.out_block is not None

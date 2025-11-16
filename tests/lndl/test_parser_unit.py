"""Unit tests for parser.py - targeting uncovered code paths.

Focuses on:
- parse_lvar() with Model.field namespaced patterns (lines 293-357)
- parse_lact() with Model.field namespaced patterns (lines 379-441)
- Error handling for unclosed tags
- Token collection logic for complex content
"""

import pytest

from lionherd_core.lndl.ast import Lact, Lvar
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import ParseError, Parser


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

    def test_lvar_legacy_pattern(self):
        """Test <lvar alias>content</lvar> - legacy pattern without model.field"""
        source = "<lvar result>42</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert lvar.model is None
        assert lvar.field is None
        assert lvar.alias == "result"
        assert lvar.content == "42"

    def test_lvar_multiline_content(self):
        """Test lvar with multiline content including newlines"""
        source = """<lvar description>This is a
multi-line
description</lvar>"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert lvar.alias == "description"
        assert "multi-line" in lvar.content
        assert lvar.content.count("\n") >= 1

    def test_lvar_complex_content_with_punctuation(self):
        """Test lvar content with dots, commas, brackets"""
        source = "<lvar data>[1, 2, 3], {key: value}, etc.</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        lvar = parser.parse_lvar()

        assert "[" in lvar.content
        assert "," in lvar.content
        assert "{" in lvar.content
        assert ":" in lvar.content

    def test_lvar_unclosed_tag_error(self):
        """Test error handling for unclosed lvar tag"""
        source = "<lvar result>content without closing tag"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        with pytest.raises(ParseError, match="Unclosed lvar tag"):
            parser.parse_lvar()

    def test_lvar_empty_content(self):
        """Test lvar with empty content"""
        source = "<lvar result></lvar>"
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
        source = "<lvar test>\n\n\ncontent</lvar>"
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

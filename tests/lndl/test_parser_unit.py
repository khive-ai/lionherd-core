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

    def test_lvar_legacy_pattern_raises_error(self):
        """Test <lvar alias>content</lvar> - legacy pattern is not supported"""
        source = "<lvar result>42</lvar>"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source)

        with pytest.raises(ParseError, match="Legacy lvar syntax"):
            parser.parse_lvar()

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

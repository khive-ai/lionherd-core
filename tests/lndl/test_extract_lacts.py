# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.lndl import Lexer, Parser


def extract_lacts(text: str) -> dict[str, str]:
    """Adapter for old extract_lacts API using new Lexer + Parser.

    This function provides backward compatibility for tests by wrapping
    the new Lexer/Parser architecture.

    Args:
        text: LNDL text containing lact tags

    Returns:
        Dict mapping action alias to call string
    """
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    # Convert AST lacts to dict format (alias -> call)
    lacts: dict[str, str] = {}
    for lact in program.lacts:
        lacts[lact.alias] = lact.call

    return lacts


class TestExtractLacts:
    """Test <lact> action extraction."""

    def test_single_lact(self):
        """Test extracting single action."""
        text = '<lact search>search(query="AI", limit=5)</lact>'
        result = extract_lacts(text)
        assert result == {"search": 'search(query="AI", limit=5)'}

    def test_multiple_lacts(self):
        """Test extracting multiple actions."""
        text = """
        <lact search>search(query="AI")</lact>
        <lact validate>validate(data)</lact>
        """
        result = extract_lacts(text)
        assert "search" in result
        assert "validate" in result
        assert 'query="AI"' in result["search"]
        assert result["validate"] == "validate(data)"

    def test_lact_with_multiline(self):
        """Test action with multiline function call."""
        text = """
        <lact complex>process(
            query="AI safety",
            limit=10,
            filters=["academic"]
        )</lact>
        """
        result = extract_lacts(text)
        assert "complex" in result
        assert "process(" in result["complex"]
        assert '"AI safety"' in result["complex"]

    def test_lact_with_whitespace(self):
        """Test action content whitespace handling."""
        text = '<lact test>  search(query="test")  </lact>'
        result = extract_lacts(text)
        assert result == {"test": 'search(query="test")'}

    def test_empty_lacts(self):
        """Test when no actions present."""
        text = "no actions here"
        result = extract_lacts(text)
        assert result == {}

    def test_mixed_lvar_and_lact(self):
        """Test extracting actions when lvars are also present."""
        text = """
        <lvar Report.title t>Title</lvar>
        <lact search>search(query="AI")</lact>
        <lvar Report.summary s>Summary</lvar>
        """
        result = extract_lacts(text)
        # Should only extract lacts, not lvars
        assert len(result) == 1
        assert "search" in result

    def test_scratch_actions(self):
        """Test that all actions are extracted (execution filtering happens later)."""
        text = """
        <lact draft1>search(query="broad")</lact>
        <lact draft2>search(query="focused")</lact>
        <lact final>search(query="perfect")</lact>
        """
        result = extract_lacts(text)
        # All actions extracted (resolver decides which to execute)
        assert len(result) == 3
        assert "draft1" in result
        assert "draft2" in result
        assert "final" in result

    def test_complex_arguments(self):
        """Test action with complex arguments."""
        text = (
            '<lact api>fetch(url="https://api.com", headers={"Auth": "token"}, timeout=30)</lact>'
        )
        result = extract_lacts(text)
        assert "api" in result
        assert "fetch(" in result["api"]
        assert "https://api.com" in result["api"]

    def test_positional_args(self):
        """Test action with positional arguments."""
        text = "<lact calc>calculate(10, 20, 30)</lact>"
        result = extract_lacts(text)
        assert result == {"calc": "calculate(10, 20, 30)"}

    def test_name_collision_detection(self):
        """Test that duplicate aliases are detected during parsing."""
        from lionherd_core.lndl import ParseError

        text = """
        <lact data>search()</lact>
        <lvar Report.field data>value</lvar>
        """
        # New Parser detects duplicate aliases during parsing (fail-fast)
        with pytest.raises(ParseError, match="Duplicate alias"):
            extract_lacts(text)

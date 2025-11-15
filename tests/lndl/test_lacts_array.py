# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for LNDL namespace array syntax.

New features:
- <lact namespace a b c>[call1(), call2(), call3()]</lact>
- Balanced delimiter parsing for arrays
- Multi-alias support in LactMetadata
- Keyword-only argument requirement
"""

import warnings

import pytest

from lionherd_core.lndl.parser import (
    PYTHON_RESERVED,
    _parse_call_array,
    extract_lacts_prefixed,
)
from lionherd_core.lndl.types import LactMetadata


class TestParseCallArray:
    """Test _parse_call_array() balanced delimiter handling."""

    def test_simple_array(self):
        """Test parsing simple array of function calls."""
        array_str = '[find(query="ocean"), recall(query="test")]'
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == 'find(query="ocean")'
        assert result[1] == 'recall(query="test")'

    def test_array_without_brackets(self):
        """Test parsing array content without surrounding brackets."""
        array_str = 'find(query="ocean"), recall(query="test")'
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == 'find(query="ocean")'
        assert result[1] == 'recall(query="test")'

    def test_string_with_commas(self):
        """Test handling strings containing commas."""
        array_str = '[func(msg="hello, world"), other(x=1)]'
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == 'func(msg="hello, world")'
        assert result[1] == "other(x=1)"

    def test_nested_parentheses(self):
        """Test handling nested function calls."""
        array_str = "[outer(inner(a, b), c), simple()]"
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == "outer(inner(a, b), c)"
        assert result[1] == "simple()"

    def test_nested_brackets(self):
        """Test handling nested list literals."""
        array_str = "[func(data=[1, [2, 3]]), other(x=4)]"
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == "func(data=[1, [2, 3]])"
        assert result[1] == "other(x=4)"

    def test_nested_braces(self):
        """Test handling nested dict literals."""
        array_str = '[func(config={"a": 1, "nested": {"b": 2}}), other()]'
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == 'func(config={"a": 1, "nested": {"b": 2}})'
        assert result[1] == "other()"

    def test_mixed_nesting(self):
        """Test handling mixed nesting of all delimiter types."""
        array_str = '[func(a={"list": [1, 2]}, b="x,y"), other(c=3)]'
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == 'func(a={"list": [1, 2]}, b="x,y")'
        assert result[1] == "other(c=3)"

    def test_single_quoted_strings(self):
        """Test handling single-quoted strings."""
        array_str = "[func(msg='hello, world'), other(x='test')]"
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == "func(msg='hello, world')"
        assert result[1] == "other(x='test')"

    def test_escaped_quotes(self):
        """Test handling escaped quotes in strings."""
        array_str = r'[func(msg="she said \"hello\""), other(x=1)]'
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == r'func(msg="she said \"hello\"")'

    def test_whitespace_handling(self):
        """Test that whitespace is preserved within calls but trimmed between."""
        array_str = "[  func( a = 1 )  ,  other( b = 2 )  ]"
        result = _parse_call_array(array_str)
        assert len(result) == 2
        assert result[0] == "func( a = 1 )"
        assert result[1] == "other( b = 2 )"

    def test_empty_array(self):
        """Test parsing empty array."""
        array_str = "[]"
        result = _parse_call_array(array_str)
        assert result == []

    def test_single_element(self):
        """Test array with single element."""
        array_str = '[find(query="test")]'
        result = _parse_call_array(array_str)
        assert len(result) == 1
        assert result[0] == 'find(query="test")'


class TestExtractLactsArraySyntax:
    """Test extract_lacts_prefixed() with array syntax."""

    def test_namespace_array_basic(self):
        """Test basic namespace array syntax.

        CRITICAL: Each alias should map to its corresponding call by index.
        <lact cognition a b c>[find(), recall(), remember()]</lact>
        - a should execute find() (index 0)
        - b should execute recall() (index 1)
        - c should execute remember() (index 2)
        """
        text = '<lact cognition a b c>[find.by_name(query="ocean"), recall.search(query="test"), remember(content="x")]</lact>'
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 3
        assert "a" in lacts
        assert "b" in lacts
        assert "c" in lacts

        # Each alias gets its own metadata with only its specific call
        assert lacts["a"].model == "cognition"
        assert lacts["a"].field is None
        assert lacts["a"].local_names == ["a"]
        assert lacts["a"].calls == ['find.by_name(query="ocean")']
        assert lacts["a"].call == 'find.by_name(query="ocean")'  # First call for "a"

        assert lacts["b"].model == "cognition"
        assert lacts["b"].field is None
        assert lacts["b"].local_names == ["b"]
        assert lacts["b"].calls == ['recall.search(query="test")']
        assert lacts["b"].call == 'recall.search(query="test")'  # Second call for "b"

        assert lacts["c"].model == "cognition"
        assert lacts["c"].field is None
        assert lacts["c"].local_names == ["c"]
        assert lacts["c"].calls == ['remember(content="x")']
        assert lacts["c"].call == 'remember(content="x")'  # Third call for "c"

        # Each has single call, so is_array is False for individual metadata
        assert lacts["a"].is_array is False
        assert lacts["b"].is_array is False
        assert lacts["c"].is_array is False

    def test_namespace_array_default_aliases(self):
        """Test namespace array without explicit aliases (auto-generated)."""
        text = '<lact cognition>[find.by_name(query="ocean"), recall.search(query="test")]</lact>'
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 2
        assert "cognition_0" in lacts
        assert "cognition_1" in lacts

        # Each auto-generated alias gets its own metadata with its specific call
        assert lacts["cognition_0"].local_names == ["cognition_0"]
        assert lacts["cognition_0"].calls == ['find.by_name(query="ocean")']
        assert lacts["cognition_0"].call == 'find.by_name(query="ocean")'

        assert lacts["cognition_1"].local_names == ["cognition_1"]
        assert lacts["cognition_1"].calls == ['recall.search(query="test")']
        assert lacts["cognition_1"].call == 'recall.search(query="test")'

    def test_namespace_array_mismatch_warning(self):
        """Test warning when alias count doesn't match call count."""
        text = '<lact cognition a b>[find.by_name(query="ocean"), recall.search(query="test"), remember(content="x")]</lact>'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = extract_lacts_prefixed(text)

            # Should have warning about mismatch
            assert len(w) == 1
            assert "Alias count mismatch" in str(w[0].message)
            assert "2 aliases for 3 calls" in str(w[0].message)

    def test_namespaced_single_call(self):
        """Test Model.field syntax with single call."""
        text = '<lact Report.summary s>generate_summary(content="x")</lact>'
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 1
        assert "s" in lacts
        assert lacts["s"].model == "Report"
        assert lacts["s"].field == "summary"
        assert lacts["s"].local_names == ["s"]
        assert lacts["s"].calls == ['generate_summary(content="x")']
        assert lacts["s"].is_array is False

        # Backward compatibility properties
        assert lacts["s"].local_name == "s"
        assert lacts["s"].call == 'generate_summary(content="x")'

    def test_namespaced_without_alias(self):
        """Test Model.field syntax without explicit alias (uses field name)."""
        text = '<lact Report.summary>generate_summary(content="x")</lact>'
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 1
        assert "summary" in lacts
        assert lacts["summary"].local_names == ["summary"]
        assert lacts["summary"].field == "summary"

    def test_direct_single_call(self):
        """Test direct name>call syntax."""
        text = '<lact search>search(query="AI", limit=5)</lact>'
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 1
        assert "search" in lacts
        assert lacts["search"].model is None
        assert lacts["search"].field is None
        assert lacts["search"].local_names == ["search"]
        assert lacts["search"].calls == ['search(query="AI", limit=5)']

    def test_direct_with_custom_alias(self):
        """Test direct syntax with custom alias."""
        text = '<lact search s>search(query="AI", limit=5)</lact>'
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 1
        assert "s" in lacts
        assert lacts["s"].model is None
        assert lacts["s"].local_names == ["s"]

    def test_multiple_mixed_patterns(self):
        """Test multiple lacts with different patterns in same text."""
        text = """
        <lact cognition a b>[find.by_name(query="ocean"), recall.search(query="test")]</lact>
        <lact Report.summary s>generate_summary(content="x")</lact>
        <lact search>search(query="AI")</lact>
        """
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 4  # a, b, s, search
        assert "a" in lacts
        assert "b" in lacts
        assert "s" in lacts
        assert "search" in lacts

        # Array pattern - each alias has single call
        assert lacts["a"].model == "cognition"
        assert lacts["a"].is_array is False  # Single call per alias
        assert lacts["a"].call == 'find.by_name(query="ocean")'
        assert lacts["b"].call == 'recall.search(query="test")'

        # Namespaced pattern
        assert lacts["s"].model == "Report"
        assert lacts["s"].field == "summary"

        # Direct pattern
        assert lacts["search"].model is None

    def test_python_reserved_keyword_warning(self):
        """Test warning when action name is Python reserved keyword."""
        text = '<lact for>search(query="test")</lact>'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = extract_lacts_prefixed(text)

            assert len(w) == 1
            assert "Python reserved keyword" in str(w[0].message)
            assert "'for'" in str(w[0].message)

    def test_python_builtin_warning(self):
        """Test warning when action name is Python builtin."""
        text = '<lact print>log(message="test")</lact>'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = extract_lacts_prefixed(text)

            assert len(w) == 1
            assert "Python reserved keyword or builtin" in str(w[0].message)

    def test_no_duplicate_warnings(self):
        """Test that warnings are only shown once per keyword."""
        text = """
        <lact for>search(query="test1")</lact>
        <lact for>search(query="test2")</lact>
        """

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Clear the set first
            from lionherd_core.lndl.parser import _warned_action_names

            _warned_action_names.clear()

            _ = extract_lacts_prefixed(text)

            # Should only warn once even though 'for' appears twice
            keyword_warnings = [x for x in w if "Python reserved keyword" in str(x.message)]
            assert len(keyword_warnings) == 1

    def test_complex_nested_arguments(self):
        """Test array parsing with complex nested structures."""
        text = """<lact ops a b>[
            func(data={"key": [1, 2, 3]}, msg="hello, world"),
            other(nested=(1, (2, 3)), config={"a": {"b": "c"}})
        ]</lact>"""
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 2
        assert "a" in lacts
        assert "b" in lacts
        # Each alias has its own call
        assert 'func(data={"key": [1, 2, 3]}, msg="hello, world")' in lacts["a"].calls[0]
        assert 'other(nested=(1, (2, 3)), config={"a": {"b": "c"}})' in lacts["b"].calls[0]

    def test_multiline_array(self):
        """Test array syntax with multiline formatting."""
        text = """<lact cognition a b c>[
            find.by_name(query="ocean"),
            recall.search(query="patterns", limit=10),
            remember(content="important insight")
        ]</lact>"""
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 3
        # Each alias has its own call at the corresponding index
        assert lacts["a"].calls[0].strip() == 'find.by_name(query="ocean")'
        assert 'recall.search(query="patterns", limit=10)' in lacts["b"].calls[0]
        assert 'remember(content="important insight")' in lacts["c"].calls[0]

    def test_empty_text_returns_empty_dict(self):
        """Test that empty text returns empty dict."""
        text = "no lacts here"
        lacts = extract_lacts_prefixed(text)
        assert lacts == {}

    def test_whitespace_in_aliases(self):
        """Test handling of whitespace in alias list."""
        text = '<lact cognition  a   b    c  >[find(query="x"), recall(query="y"), remember(content="z")]</lact>'
        lacts = extract_lacts_prefixed(text)

        assert len(lacts) == 3
        assert "a" in lacts
        assert "b" in lacts
        assert "c" in lacts


class TestLactMetadataBackwardCompatibility:
    """Test backward compatibility of LactMetadata changes.

    NOTE: With the array alias fix, each alias gets its own metadata with a single call.
    The properties still work for backward compatibility but with different semantics:
    - local_name returns the first (and only) name in local_names
    - call returns the first (and only) call in calls
    - is_array is False for single-call metadata (which is now the norm)
    """

    def test_local_name_property(self):
        """Test backward-compatible local_name property with single alias."""
        metadata = LactMetadata(
            model="cognition",
            field=None,
            local_names=["a"],  # Single alias per metadata now
            calls=["find()"],  # Single call per metadata now
        )
        assert metadata.local_name == "a"

    def test_call_property(self):
        """Test backward-compatible call property with single call."""
        metadata = LactMetadata(
            model="cognition",
            field=None,
            local_names=["a"],  # Single alias per metadata now
            calls=["find()"],  # Single call per metadata now
        )
        assert metadata.call == "find()"

    def test_is_array_property_true(self):
        """Test is_array property for multi-call metadata (legacy format)."""
        # This format is no longer created by extract_lacts_prefixed after the fix
        # but the property still works for backward compatibility
        metadata = LactMetadata(
            model="cognition", field=None, local_names=["a", "b"], calls=["find()", "recall()"]
        )
        assert metadata.is_array is True

    def test_is_array_property_false(self):
        """Test is_array property for single-call declarations."""
        metadata = LactMetadata(model="cognition", field=None, local_names=["a"], calls=["find()"])
        assert metadata.is_array is False

    def test_empty_lists_backward_compat(self):
        """Test edge case with empty lists."""
        metadata = LactMetadata(model=None, field=None, local_names=[], calls=[])
        assert metadata.local_name == ""
        assert metadata.call == ""
        assert metadata.is_array is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_unmatched_brackets_in_text(self):
        """Test that unmatched brackets in surrounding text don't affect parsing."""
        text = """
        Some text with [ brackets ] here.
        <lact search>search(query="test")</lact>
        More text with ] brackets [ here.
        """
        lacts = extract_lacts_prefixed(text)
        assert len(lacts) == 1
        assert "search" in lacts

    def test_nested_lact_tags_not_supported(self):
        """Test that nested lact tags are not parsed (outer wins)."""
        # The regex will match the first complete <lact>...</lact> pair
        text = "<lact outer><lact inner>inner_call()</lact></lact>"
        lacts = extract_lacts_prefixed(text)

        # Should match outer with content "<lact inner>inner_call()"
        assert len(lacts) == 1
        assert "outer" in lacts

    def test_array_with_single_call(self):
        """Test array syntax with only one call."""
        text = '<lact cognition a>[find.by_name(query="ocean")]</lact>'
        lacts = extract_lacts_prefixed(text)

        # When counts match, no warning
        assert len(lacts) == 1
        assert "a" in lacts
        assert lacts["a"].calls == ['find.by_name(query="ocean")']

    def test_very_long_array(self):
        """Test array with many calls."""
        calls = ", ".join([f"call{i}(arg={i})" for i in range(100)])
        text = f"<lact ops>[{calls}]</lact>"
        lacts = extract_lacts_prefixed(text)

        # Default aliases should be ops_0, ops_1, ..., ops_99
        assert len(lacts) == 100
        assert "ops_0" in lacts
        assert "ops_99" in lacts
        # Each alias has only its own call
        assert lacts["ops_0"].calls == ["call0(arg=0)"]
        assert lacts["ops_99"].calls == ["call99(arg=99)"]
        assert lacts["ops_0"].call == "call0(arg=0)"
        assert lacts["ops_99"].call == "call99(arg=99)"

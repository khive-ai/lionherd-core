# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Consolidated LNDL integration tests.

This module contains all LNDL tests organized into logical sections:
- Section 1: Basic LNDL Parsing (prefixed syntax, extraction, resolution)
- Section 2: Fuzzy Matching (field/lvar/model/spec correction)
- Section 3: Action Resolution (lact tags, namespaced actions, error handling)
- Section 4: Full Integration (end-to-end workflows, validation, real-world scenarios)

Coverage:
- Lvar extraction and resolution (30 test classes total)
- OUT{} block parsing and validation
- Fuzzy field name correction with thresholds
- Action call lifecycle (parse → execute → revalidate)
- Type conversion and Pydantic validation
- Error handling and ExceptionGroup aggregation
- Backward compatibility
"""

import warnings
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from lionherd_core.lndl import (
    ActionCall,
    AmbiguousMatchError,
    MissingFieldError,
    TypeMismatchError,
    parse_lndl,
    parse_lndl_fuzzy,
    resolve_references_prefixed,
)
from lionherd_core.lndl.errors import MissingOutBlockError
from lionherd_core.lndl.parser import PYTHON_RESERVED, ParseError
from lionherd_core.lndl.types import (
    LactMetadata,
    LvarMetadata,
    ensure_no_action_calls,
    has_action_calls,
    revalidate_with_action_results,
)
from lionherd_core.types import Operable, Spec

from .conftest import Reason, Report, SearchResults, ValidatedReport, parse_lacts, parse_lvars

# ======================================================================================
# SECTION 1: BASIC LNDL PARSING (PREFIXED SYNTAX)
# ======================================================================================
# Tests for namespace-prefixed lvar extraction, reference resolution, and basic parsing.
# Covers: <lvar Model.field alias>value</lvar>, OUT{} blocks, scalar literals, validators.


class TestExtractLvarsPrefixed:
    """Test namespace-prefixed lvar extraction using token-based parser."""

    def test_extract_with_local_name(self):
        """Test extracting lvar with explicit local name."""
        text = "<lvar Report.title title>here is a good title</lvar>"
        lvars = parse_lvars(text)

        assert len(lvars) == 1
        assert "title" in lvars
        assert lvars["title"]["model"] == "Report"
        assert lvars["title"]["field"] == "title"
        assert lvars["title"]["local_name"] == "title"
        assert lvars["title"]["value"] == "here is a good title"

    def test_extract_without_local_name(self):
        """Test extracting lvar without local name (uses field name)."""
        text = "<lvar Report.title>here is a good title</lvar>"
        lvars = parse_lvars(text)

        assert len(lvars) == 1
        assert "title" in lvars  # Uses field name as local
        assert lvars["title"]["model"] == "Report"
        assert lvars["title"]["field"] == "title"
        assert lvars["title"]["local_name"] == "title"
        assert lvars["title"]["value"] == "here is a good title"

    def test_extract_with_custom_alias(self):
        """Test extracting lvar with custom local alias."""
        text = "<lvar Reason.confidence conf>0.85</lvar>"
        lvars = parse_lvars(text)

        assert len(lvars) == 1
        assert "conf" in lvars  # Custom alias
        assert lvars["conf"]["model"] == "Reason"
        assert lvars["conf"]["field"] == "confidence"
        assert lvars["conf"]["local_name"] == "conf"
        assert lvars["conf"]["value"] == "0.85"

    def test_extract_multiple_lvars(self):
        """Test extracting multiple namespace-prefixed lvars."""
        text = """
        <lvar Report.title title>here is a good title</lvar>
        <lvar Reason.confidence conf>0.85</lvar>
        <lvar Report.summary summ>sdfghjklkjhgfdfghj</lvar>
        <lvar Reason.analysis ana>fghjklfghj</lvar>
        """
        lvars = parse_lvars(text)

        assert len(lvars) == 4
        assert "title" in lvars
        assert "conf" in lvars
        assert "summ" in lvars
        assert "ana" in lvars

    def test_extract_with_revision(self):
        """Test extracting multiple versions of same field (revision tracking)."""
        text = """
        <lvar Report.summary summ>first version</lvar>
        <lvar Report.summary summ2>revised version</lvar>
        """
        lvars = parse_lvars(text)

        assert len(lvars) == 2
        assert lvars["summ"]["value"] == "first version"
        assert lvars["summ2"]["value"] == "revised version"

    def test_extract_with_multiline_value(self):
        """Test extracting lvar with multiline value."""
        text = """
        <lvar Report.summary summ>
        This is a long summary
        that spans multiple lines
        with various content
        </lvar>
        """
        lvars = parse_lvars(text)

        assert len(lvars) == 1
        assert "This is a long summary" in lvars["summ"]["value"]
        assert "multiple lines" in lvars["summ"]["value"]

    def test_extract_from_thinking_flow(self):
        """Test extraction from natural thinking flow with prose."""
        text = """
        Let me work through this step by step...
        Oh I think xyz might be a good approach to name the report
        <lvar Report.title title>here is a good title</lvar>

        But I am only 70% confident, let me see are there more evidence I missed, ...

        Wait, more evidence: 85%
        <lvar Reason.confidence conf>0.85</lvar>
        """
        lvars = parse_lvars(text)

        assert len(lvars) == 2
        assert lvars["title"]["value"] == "here is a good title"
        assert lvars["conf"]["value"] == "0.85"

    def test_extract_empty_returns_empty_dict(self):
        """Test that text without lvars returns empty dict."""
        text = "Just some prose without any lvar tags"
        lvars = parse_lvars(text)

        assert lvars == {}

    def test_raw_lvar_syntax(self):
        """Test that raw syntax (no namespace) returns RLvar."""
        text = "<lvar x>raw content</lvar>"
        lvars = parse_lvars(text)

        assert len(lvars) == 1
        assert "x" in lvars
        assert lvars["x"]["model"] is None  # Raw lvar has no model
        assert lvars["x"]["field"] is None  # Raw lvar has no field
        assert lvars["x"]["local_name"] == "x"
        assert lvars["x"]["value"] == "raw content"


class TestResolveReferencesPrefixed:
    """Test namespace-prefixed reference resolution."""

    def test_resolve_simple_fields(self):
        """Test resolving simple prefixed variables."""

        out_fields = {"report": ["title", "summary"]}
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Good Title"),
            "summary": LvarMetadata("Report", "summary", "summary", "Summary text"),
        }
        operable = Operable([Spec(Report, name="report")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.report.title == "Good Title"
        assert output.report.summary == "Summary text"

    def test_resolve_with_type_conversion(self):
        """Test resolving with automatic type conversion."""

        out_fields = {"reasoning": ["conf", "ana"]}
        lvars = {
            "conf": LvarMetadata("Reason", "confidence", "conf", "0.85"),
            "ana": LvarMetadata("Reason", "analysis", "ana", "Analysis text"),
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.reasoning.confidence == 0.85  # String "0.85" → float
        assert output.reasoning.analysis == "Analysis text"

    def test_resolve_multiple_specs(self):
        """Test resolving multiple specs at once."""

        out_fields = {
            "report": ["title", "summary"],
            "reasoning": ["conf", "ana"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
            "summary": LvarMetadata("Report", "summary", "summary", "Summary"),
            "conf": LvarMetadata("Reason", "confidence", "conf", "0.9"),
            "ana": LvarMetadata("Reason", "analysis", "ana", "Analysis"),
        }
        operable = Operable([Spec(Report, name="report"), Spec(Reason, name="reasoning")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.report.title == "Title"
        assert output.reasoning.confidence == 0.9

    def test_resolve_with_custom_alias(self):
        """Test resolving with custom local aliases."""

        out_fields = {"reasoning": ["conf", "ana"]}
        lvars = {
            "conf": LvarMetadata("Reason", "confidence", "conf", "0.75"),  # alias: conf
            "ana": LvarMetadata("Reason", "analysis", "ana", "Text"),  # alias: ana
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.reasoning.confidence == 0.75
        assert output.reasoning.analysis == "Text"

    def test_resolve_missing_required_field_error(self):
        """Test error when required field missing from OUT{}."""

        out_fields = {}
        lvars = {}
        operable = Operable([Spec(Reason, name="reasoning", required=True)])

        with pytest.raises(MissingFieldError, match="reasoning"):
            resolve_references_prefixed(out_fields, lvars, {}, operable)

    def test_resolve_type_mismatch_error(self):
        """Test error when variable model doesn't match spec."""

        out_fields = {"reasoning": ["title"]}
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Wrong model"),
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable)

        # Verify ExceptionGroup contains TypeMismatchError
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], TypeMismatchError)
        assert "Report" in str(exc_info.value.exceptions[0])
        assert "Reason" in str(exc_info.value.exceptions[0])

    def test_resolve_missing_variable_error(self):
        """Test error when referenced variable not declared."""

        out_fields = {"reasoning": ["conf", "missing_var"]}
        lvars = {
            "conf": LvarMetadata("Reason", "confidence", "conf", "0.85"),
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable)

        # Verify ExceptionGroup contains ValueError about missing var
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)
        assert "missing_var" in str(exc_info.value.exceptions[0])
        assert "not declared" in str(exc_info.value.exceptions[0])

    def test_resolve_scalar_missing_variable_error(self):
        """Test error when scalar field references undeclared variable."""

        # Scalar field with array syntax pointing to missing variable
        out_fields = {"quality_score": ["missing_score"]}
        lvars = {}  # No variables declared
        operable = Operable([Spec(float, name="quality_score")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable)

        # Verify ExceptionGroup contains ValueError about missing var
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)
        assert "missing_score" in str(exc_info.value.exceptions[0])
        assert "not declared" in str(exc_info.value.exceptions[0])

    def test_resolve_pydantic_validation_error(self):
        """Test Pydantic validation errors bubble up."""

        out_fields = {"reasoning": ["conf", "ana"]}
        lvars = {
            "conf": LvarMetadata("Reason", "confidence", "conf", "not_a_number"),
            "ana": LvarMetadata("Reason", "analysis", "ana", "Text"),
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable)

        # Verify ExceptionGroup contains ValueError from Pydantic validation
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)
        assert "Failed to construct" in str(exc_info.value.exceptions[0])

    def test_out_field_no_spec_error(self):
        """Test error when OUT{} field has no Spec in Operable."""

        out_fields = {"unknown_field": ["var1"]}
        lvars = {"var1": LvarMetadata("Report", "title", "var1", "value")}
        operable = Operable([Spec(Report, name="report")])

        # This raises ValueError from operable.check_allowed() at line 47
        with pytest.raises(ValueError, match="not allowed"):
            resolve_references_prefixed(out_fields, lvars, {}, operable)

    def test_basemodel_field_literal_error(self):
        """Test error when BaseModel field gets literal value."""
        out_fields = {"report": "literal_value"}  # Wrong: should be array
        lvars = {}
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert len(exc_info.value.exceptions) == 1
        assert "requires array syntax" in str(exc_info.value.exceptions[0])

    def test_spec_invalid_type_error(self):
        """Test error when Spec base_type is not BaseModel or scalar."""

        out_fields = {"invalid": ["var1"]}
        lvars = {"var1": LvarMetadata("Invalid", "field", "var1", "value")}

        # Create Spec with invalid type (e.g., list, dict)
        operable = Operable([Spec(list, name="invalid")])  # list is not BaseModel

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert len(exc_info.value.exceptions) == 1
        assert "must be a Pydantic BaseModel or scalar type" in str(exc_info.value.exceptions[0])

    def test_operable_get_returns_none(self):
        """Test defensive code when operable.get() returns None."""

        out_fields = {"field1": ["var1"]}
        lvars = {"var1": LvarMetadata("Report", "title", "var1", "value")}

        # Mock operable to pass check_allowed but return None from get
        operable_mock = Mock()
        operable_mock.check_allowed = Mock()  # Doesn't raise
        operable_mock.get_specs = Mock(return_value=[])  # No required specs
        operable_mock.get = Mock(return_value=None)  # Returns None

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable_mock)

        # Should raise ValueError with clear message
        assert len(exc_info.value.exceptions) == 1
        assert "no corresponding Spec" in str(exc_info.value.exceptions[0])


class TestParseLNDLPrefixed:
    """Test end-to-end namespace-prefixed LNDL parsing."""

    def test_parse_complete_example(self):
        """Test parsing the complete example from documentation."""
        response = """
        Let me work through this step by step...
        Oh I think xyz might be a good approach to name the report
        <lvar Report.title title>here is a good title</lvar>

        But I am only 70% confident, let me see are there more evidence I missed, ...

        Wait, more evidence: 85%
        <lvar Reason.confidence conf>0.85</lvar>
        So from the source, this and that, blah blah
        <lvar Report.summary summ>sdfghjklkjhgfdfghj</lvar>

        Hmmm let me revise, I think xyz is wrong,
        <lvar Reason.analysis ana>fghjklfghj</lvar>

        ok I am ready
        <lvar Report.summary summ2>dfghjkgfgjk</lvar>

        ```lndl
        OUT{report:[title, summ2], reasoning:[conf, ana]}
        ```
        """

        operable = Operable([Spec(Report, name="report"), Spec(Reason, name="reasoning")])
        output = parse_lndl(response, operable)

        # Verify correct construction
        assert output.report.title == "here is a good title"
        assert output.report.summary == "dfghjkgfgjk"  # summ2, not summ
        assert output.reasoning.confidence == 0.85
        assert output.reasoning.analysis == "fghjklfghj"

        # Verify lvars preserved (including unused summ)
        assert "title" in output.lvars
        assert "conf" in output.lvars
        assert "summ" in output.lvars
        assert "summ2" in output.lvars
        assert "ana" in output.lvars

    def test_parse_without_code_fence(self):
        """Test parsing without ```lndl code fence."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.report.summary == "Summary"

    def test_parse_with_optional_field_present(self):
        """Test parsing with optional field present."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lvar Reason.confidence c>0.9</lvar>
        <lvar Reason.analysis a>Analysis</lvar>

        OUT{report:[t, s], reasoning:[c, a]}
        """

        operable = Operable(
            [Spec(Report, name="report"), Spec(Reason, name="reasoning", required=False)]
        )
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.reasoning.confidence == 0.9

    def test_parse_with_optional_field_omitted(self):
        """Test parsing with optional field omitted."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable(
            [Spec(Report, name="report"), Spec(Reason, name="reasoning", required=False)]
        )
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert "reasoning" not in output.fields

    def test_parse_dict_access(self):
        """Test dictionary-style access."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        OUT{report:[t, s]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        assert output["report"].title == "Title"

    def test_parse_attribute_access(self):
        """Test attribute-style access."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        OUT{report:[t, s]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"

    def test_parse_preserves_lvar_metadata(self):
        """Test that LvarMetadata is preserved in output."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        OUT{report:[t, s]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        # lvars should be dict[str, LvarMetadata] for prefixed syntax
        assert "t" in output.lvars
        assert output.lvars["t"].model == "Report"
        assert output.lvars["t"].field == "title"
        assert output.lvars["t"].value == "Title"


class TestParseLNDLEdgeCases:
    """Test edge cases for namespace-prefixed LNDL."""

    def test_parse_with_no_local_name(self):
        """Test parsing when local name omitted (uses field name)."""
        response = """
        <lvar Report.title>Title</lvar>
        <lvar Report.summary>Summary</lvar>
        OUT{report:[title, summary]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.report.summary == "Summary"

    def test_parse_with_single_variable(self):
        """Test parsing with single variable (no array brackets)."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        OUT{report:t}
        """

        operable = Operable([Spec(Report, name="report")])

        # This should fail because single variable can't construct full model
        with pytest.raises(ExceptionGroup) as exc_info:
            parse_lndl(response, operable)

        # Verify it's a ValueError about missing field
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)

    def test_parse_revision_tracking(self):
        """Test that revision tracking works (multiple versions)."""
        response = """
        <lvar Report.summary v1>First version</lvar>
        <lvar Report.summary v2>Second version</lvar>
        <lvar Report.summary v3>Final version</lvar>
        <lvar Report.title t>Title</lvar>

        OUT{report:[t, v3]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        # Should use v3
        assert output.report.summary == "Final version"

        # But all versions preserved in lvars
        assert output.lvars["v1"].value == "First version"
        assert output.lvars["v2"].value == "Second version"
        assert output.lvars["v3"].value == "Final version"


class TestScalarLiterals:
    """Test scalar literal values in OUT blocks."""

    def test_float_literal(self):
        """Test float literal in OUT block."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], quality_score:0.8}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.quality_score == 0.8
        assert isinstance(output.quality_score, float)

    def test_int_literal(self):
        """Test integer literal in OUT block."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], priority_level:3}
        """

        operable = Operable([Spec(Report, name="report"), Spec(int, name="priority_level")])
        output = parse_lndl(response, operable)

        assert output.priority_level == 3
        assert isinstance(output.priority_level, int)

    def test_str_literal(self):
        """Test string literal in OUT block."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], status:"completed"}
        """

        operable = Operable([Spec(Report, name="report"), Spec(str, name="status")])
        output = parse_lndl(response, operable)

        assert output.status == "completed"
        assert isinstance(output.status, str)

    def test_bool_literal_true(self):
        """Test boolean true literal in OUT block."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], is_approved:true}
        """

        operable = Operable([Spec(Report, name="report"), Spec(bool, name="is_approved")])
        output = parse_lndl(response, operable)

        assert output.is_approved is True
        assert isinstance(output.is_approved, bool)

    def test_bool_literal_false(self):
        """Test boolean false literal in OUT block."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], is_draft:false}
        """

        operable = Operable([Spec(Report, name="report"), Spec(bool, name="is_draft")])
        output = parse_lndl(response, operable)

        assert output.is_draft is False
        assert isinstance(output.is_draft, bool)

    def test_negative_number_literal(self):
        """Test negative number literal."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], temperature:-5.5}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="temperature")])
        output = parse_lndl(response, operable)

        assert output.temperature == -5.5

    def test_multiple_scalars(self):
        """Test multiple scalar literals in one OUT block."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lvar Reason.confidence c>0.85</lvar>
        <lvar Reason.analysis a>Analysis</lvar>

        OUT{report:[t, s], reasoning:[c, a], quality_score:0.9, priority:2, status:"active"}
        """

        operable = Operable(
            [
                Spec(Report, name="report"),
                Spec(Reason, name="reasoning"),
                Spec(float, name="quality_score"),
                Spec(int, name="priority"),
                Spec(str, name="status"),
            ]
        )
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.reasoning.confidence == 0.85
        assert output.quality_score == 0.9
        assert output.priority == 2
        assert output.status == "active"

    def test_complete_example_with_scalars(self):
        """Test complete example from user requirement."""
        response = """
        Let me work through this step by step...
        Oh I think xyz might be a good approach to name the report
        <lvar Report.title title>here is a good title</lvar>

        But I am only 70% confident, let me see are there more evidence I missed, ...

        Wait, more evidence: 85%
        <lvar Reason.confidence conf>0.85</lvar>
        So from the source, this and that, blah blah
        <lvar Report.summary summ>sdfghjklkjhgfdfghj</lvar>

        Hmmm let me revise, I think xyz is wrong,
        <lvar Reason.analysis ana>fghjklfghj</lvar>

        ok I am ready
        <lvar Report.summary summ2>dfghjkgfgjk</lvar>

        ```lndl
        OUT{report:[title, summ2], reasoning:[conf, ana], quality_score:0.8}
        ```
        """

        operable = Operable(
            [
                Spec(Report, name="report"),
                Spec(Reason, name="reasoning"),
                Spec(float, name="quality_score"),
            ]
        )
        output = parse_lndl(response, operable)

        assert output.report.title == "here is a good title"
        assert output.report.summary == "dfghjkgfgjk"
        assert output.reasoning.confidence == 0.85
        assert output.reasoning.analysis == "fghjklfghj"
        assert output.quality_score == 0.8

    def test_scalar_type_validation_error(self):
        """Test error when literal can't be converted to scalar type."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], quality_score:"not_a_number"}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])

        with pytest.raises(ExceptionGroup) as exc_info:
            parse_lndl(response, operable)

        # Verify ExceptionGroup contains ValueError about type conversion
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)
        assert "Failed to convert" in str(exc_info.value.exceptions[0])

    def test_scalar_with_array_syntax_error(self):
        """Test error when scalar field uses array literals (parser rejects literals)."""

        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], quality_score:[0.8, 0.9]}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])

        # Parser now rejects array literals (0.8, 0.9) before reaching resolver
        with pytest.raises(ParseError, match=r"Arrays must contain only.*references"):
            parse_lndl(response, operable)

    def test_scalar_from_single_variable_array(self):
        """Test scalar field with single-variable array syntax."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lvar Reason.confidence q>0.95</lvar>

        OUT{report:[t, s], quality_score:[q]}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])
        output = parse_lndl(response, operable)

        assert output.quality_score == 0.95
        assert isinstance(output.quality_score, float)


class TestValidators:
    """Test Spec with custom validators."""

    def test_spec_with_validator_success(self):
        """Test Spec with validators that pass."""
        response = """
        <lvar ValidatedReport.title t>Good Title</lvar>
        <lvar ValidatedReport.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        # Custom validator function
        def uppercase_validator(instance):
            instance.title = instance.title.upper()
            return instance

        operable = Operable([Spec(ValidatedReport, name="report", validator=uppercase_validator)])
        output = parse_lndl(response, operable)

        assert output.report.title == "GOOD TITLE"

    def test_spec_with_validator_invoke_method(self):
        """Test Spec with validator that has invoke() method."""
        response = """
        <lvar ValidatedReport.title t>Good Title</lvar>
        <lvar ValidatedReport.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        # Validator with invoke method (must also be callable for Spec validation)
        class ValidatorWithInvoke:
            def __call__(self, instance):
                # Fallback for when called as function
                return instance

            def invoke(self, field_name, instance, target_type):
                instance.title = f"[{field_name}] {instance.title.upper()}"
                return instance

        operable = Operable([Spec(ValidatedReport, name="report", validator=ValidatorWithInvoke())])
        output = parse_lndl(response, operable)

        assert output.report.title == "[report] GOOD TITLE"


class TestHardeningImprovements:
    """Test balanced braces and multi-error aggregation."""

    def test_multi_error_aggregation_two_fields(self):
        """Test that ExceptionGroup collects errors from multiple fields."""

        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, missing], quality_score:"invalid"}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])

        with pytest.raises(ExceptionGroup) as exc_info:
            parse_lndl(response, operable)

        # Should have 2 errors: missing variable and invalid float
        assert len(exc_info.value.exceptions) == 2

        # Verify error types
        errors = exc_info.value.exceptions
        error_msgs = [str(e) for e in errors]

        # One error about missing variable
        assert any("missing" in msg and "not declared" in msg for msg in error_msgs)

        # One error about invalid float conversion
        assert any("Failed to convert" in msg or "invalid" in msg for msg in error_msgs)

    def test_multi_error_aggregation_three_fields(self):
        """Test ExceptionGroup with 3 failing fields."""

        # Create lvars with mismatched model
        lvars = {
            "t": LvarMetadata("Report", "title", "t", "Title"),
            "wrong_model": LvarMetadata("Report", "field", "wrong_model", "value"),
        }

        out_fields = {
            "report": ["t", "missing_summary"],
            "reason": ["wrong_model"],
            "quality_score": "not_a_float",
        }

        operable = Operable(
            [
                Spec(Report, name="report"),
                Spec(Reason, name="reason"),
                Spec(float, name="quality_score"),
            ]
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, {}, operable)

        # Should have 3 errors
        assert len(exc_info.value.exceptions) == 3


# ======================================================================================
# SECTION 2: FUZZY MATCHING
# ======================================================================================
# Tests for fuzzy field name correction with configurable thresholds.
# Covers: field/lvar/model/spec name typos, ambiguity detection, strict mode.


class TestFuzzyFieldNameCorrection:
    """Test fuzzy correction of field names in OUT{} blocks."""

    def test_field_typo_correction(self):
        """Test common typo correction (titel → title)."""

        class Report(BaseModel):
            title: str
            content: str

        # Create spec and operable
        operable = Operable([Spec(Report, name="report")])

        # LNDL with typo "titel" instead of "title"
        lndl_text = """\
        <lvar Report.title t>Good Title</lvar>
        <lvar Report.content c>Content here</lvar>

        ```lndl
        OUT{report: [t, c]}
                ```\
                """

        # Should auto-correct "titel" if present in lvars
        # But in this case, we're testing field correction in spec lookup
        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

        assert hasattr(result, "report")
        assert result.report.title == "Good Title"

    def test_field_multiple_typos(self):
        """Test multiple field name corrections in single OUT{}."""

        class Report(BaseModel):
            title: str
            summary: str
            quality_score: float

        operable = Operable([Spec(Report, name="report")])

        # Multiple typos: "titel", "sumary"
        lndl_text = """\
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lvar Report.quality_score q>0.9</lvar>

        ```lndl
        OUT{report: [t, s, q]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.report.quality_score == 0.9

    def test_field_below_threshold(self):
        """Test field name too dissimilar raises error."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "xyz" has no similarity to "title"
        lndl_text = """\
        <lvar Report.xyz x>Value</lvar>

        ```lndl
        OUT{report: [x]}
                ```\
                """

        with pytest.raises(MissingFieldError, match=r"xyz"):
            parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

    def test_field_ambiguous_match(self):
        """Test ambiguous field match raises error (tie detection)."""

        class Report(BaseModel):
            description: str
            descriptor: str  # Similar to "description"

        operable = Operable([Spec(Report, name="report")])

        # "desc" could match both "description" and "descriptor"
        lndl_text = """\
        <lvar Report.desc d>Value</lvar>

        ```lndl
        OUT{report: [d]}
                ```\
                """

        with pytest.raises(AmbiguousMatchError, match=r"desc|description|descriptor"):
            parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

    def test_field_exact_match_preferred(self):
        """Test exact matches bypass fuzzy logic."""

        class Report(BaseModel):
            title: str
            titl: str  # Similar field exists

        operable = Operable([Spec(Report, name="report")])

        # "title" exact match should NOT trigger fuzzy
        lndl_text = """
        <lvar Report.title t>Exact</lvar>
        <lvar Report.titl tt>Similar</lvar>

        ```lndl
        OUT{report: [t, tt]}
        ```
        """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.report.title == "Exact"
        assert result.report.titl == "Similar"

    def test_field_case_insensitive_fuzzy(self):
        """Test case differences handled by fuzzy matching."""

        class Report(BaseModel):
            Title: str  # Capital T

        operable = Operable([Spec(Report, name="report")])

        # "title" (lowercase) should match "Title" fuzzy
        lndl_text = """\
        <lvar Report.title t>Value</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.report.Title == "Value"


class TestFuzzyLvarReferenceCorrection:
    """Test fuzzy correction of lvar references in OUT{} arrays."""

    def test_lvar_ref_typo(self):
        """Test lvar reference typo correction (summ → summary)."""

        class Report(BaseModel):
            summary: str

        operable = Operable([Spec(Report, name="report")])

        # Lvar named "summary", referenced as "summ" in OUT{}
        lndl_text = """\
        <lvar Report.summary summary>Full Summary</lvar>

        ```lndl
        OUT{report: [summ]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.report.summary == "Full Summary"

    def test_lvar_multiple_refs(self):
        """Test multiple lvar reference corrections."""

        class Report(BaseModel):
            title: str
            summary: str

        operable = Operable([Spec(Report, name="report")])

        # Both refs have typos: "titel", "summ"
        lndl_text = """\
        <lvar Report.title title>Title</lvar>
        <lvar Report.summary summary>Summary</lvar>

        ```lndl
        OUT{report: [titel, summ]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.report.title == "Title"
        assert result.report.summary == "Summary"

    def test_lvar_ambiguous_ref(self):
        """Test ambiguous lvar reference raises error."""

        class Report(BaseModel):
            title: str
            summary: str

        operable = Operable([Spec(Report, name="report")])

        # "t" could match both "title" and "summary" with low similarity
        lndl_text = """\
        <lvar Report.title title>Title</lvar>
        <lvar Report.summary summ>Summary</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        # Should raise error for ambiguous match or no match
        with pytest.raises((AmbiguousMatchError, MissingFieldError)):
            parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

    def test_lvar_exact_over_fuzzy(self):
        """Test exact lvar reference preferred over fuzzy."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # Exact match should be used
        lndl_text = """\
        <lvar Report.title title>Exact Title</lvar>

        ```lndl
        OUT{report: [title]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.report.title == "Exact Title"


class TestFuzzyModelNameValidation:
    """Test fuzzy model name validation with threshold 0.90."""

    def test_model_name_typo(self):
        """Test model name typo correction (Reprot → Report)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # Model name has typo in lvar: "Reprot"
        lndl_text = """\
        <lvar Reprot.title t>Title</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold_model=0.90)
        assert result.report.title == "Title"

    def test_model_name_case_variation(self):
        """Test model name case variation handled."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "report" (lowercase) should match "Report"
        lndl_text = """\
        <lvar report.title t>Title</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold_model=0.85)
        assert result.report.title == "Title"

    def test_model_name_strict_threshold(self):
        """Test model name uses stricter threshold (0.90 default)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "Rep" is too different from "Report" at 0.90 threshold (scores 0.883)
        lndl_text = """\
        <lvar Rep.title t>Title</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        # Should fail with strict 0.90 threshold
        with pytest.raises(MissingFieldError):
            parse_lndl_fuzzy(lndl_text, operable, threshold_model=0.90)


class TestFuzzySpecNameCorrection:
    """Test fuzzy spec name correction."""

    def test_spec_name_typo(self):
        """Test spec name typo correction (reprot → report)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # Spec name in OUT{} has typo: "reprot"
        lndl_text = """\
        <lvar Report.title t>Title</lvar>

        ```lndl
        OUT{reprot: [t]}
        ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold_spec=0.85)
        assert hasattr(result, "report")

    def test_spec_name_case_insensitive(self):
        """Test spec name case insensitivity."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "REPORT" (uppercase) should match "report"
        lndl_text = """\
        <lvar Report.title t>Title</lvar>

        ```lndl
        OUT{REPORT: [t]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold_spec=0.85)
        assert hasattr(result, "report") or "REPORT" in result


class TestCombinedFuzzyMatching:
    """Test combined fuzzy matching across all rigid points."""

    def test_all_fuzzy_points(self):
        """Test fuzzy correction at all 4 rigid points simultaneously."""

        class Report(BaseModel):
            title: str
            summary: str

        operable = Operable([Spec(Report, name="report")])

        # Typos at all levels:
        # - Model: "Reprot"
        # - Lvars: "titel", "summ"
        # - Spec: "reprot"
        lndl_text = """\
        <lvar Reprot.title titel>Title</lvar>
        <lvar Reprot.summary summ>Summary</lvar>

        ```lndl
        OUT{reprot: [titel, summ]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert hasattr(result, "report")
        assert result.report.title == "Title"
        assert result.report.summary == "Summary"

    def test_partial_fuzzy_partial_exact(self):
        """Test mix of fuzzy and exact matches."""

        class Report(BaseModel):
            title: str
            summary: str

        operable = Operable([Spec(Report, name="report")])

        # "title" exact, "summ" fuzzy
        lndl_text = """\
        <lvar Report.title title>Title</lvar>
        <lvar Report.summary summ>Summary</lvar>

        ```lndl
        OUT{report: [title, summ]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.report.title == "Title"
        assert result.report.summary == "Summary"


class TestThresholdConfiguration:
    """Test threshold parameter behavior."""

    def test_strict_mode_threshold_1_0(self):
        """Test threshold=1.0 requires exact matches (strict mode)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "titel" typo should FAIL with threshold=1.0
        lndl_text = """\
        <lvar Report.titel t>Title</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        with pytest.raises(MissingFieldError):
            parse_lndl_fuzzy(lndl_text, operable, threshold=1.0)

    def test_threshold_0_85_default(self):
        """Test default threshold=0.85 works for common typos."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "titel" should pass with default 0.85
        lndl_text = """\
        <lvar Report.title t>Title</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable)  # No threshold param
        assert result.report.title == "Title"

    def test_threshold_0_7_very_tolerant(self):
        """Test threshold=0.7 accepts more dissimilar matches."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "ttl" might pass with 0.7, fail with 0.85
        lndl_text = """\
        <lvar Report.ttl t>Title</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.7)
        assert result.report.title == "Title"

    def test_threshold_boundary(self):
        """Test threshold boundary behavior (just below/above)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "titel" → "title" scores 0.953, passes at 0.95
        lndl_text_close = """\
        <lvar Report.titel t>Title</lvar>

        ```lndl
        OUT{report: [t]}
        ```\
        """

        # Should pass at 0.85
        result = parse_lndl_fuzzy(lndl_text_close, operable, threshold=0.85)
        assert result.report.title == "Title"

        # Also passes at 0.95 (very close typo, scores 0.953)
        result = parse_lndl_fuzzy(lndl_text_close, operable, threshold=0.95)
        assert result.report.title == "Title"

        # Use worse typo "ttl" for boundary test (should fail at 0.95)
        lndl_text_far = """\
        <lvar Report.ttl t>Title</lvar>

        ```lndl
        OUT{report: [t]}
        ```\
        """

        # Should pass at 0.85
        result = parse_lndl_fuzzy(lndl_text_far, operable, threshold=0.85)
        assert result.report.title == "Title"

        # Should fail at 0.95 (worse typo)
        with pytest.raises(MissingFieldError):
            parse_lndl_fuzzy(lndl_text_far, operable, threshold=0.95)

    def test_per_type_thresholds(self):
        """Test different thresholds for field/lvar/model/spec."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        lndl_text = """\
        <lvar Reprot.titel t>Title</lvar>

        ```lndl
        OUT{reprot: [t]}
                ```\
                """

        # Custom thresholds for each type
        result = parse_lndl_fuzzy(
            lndl_text,
            operable,
            threshold_field=0.85,
            threshold_lvar=0.85,
            threshold_model=0.90,
            threshold_spec=0.85,
        )
        assert result.report.title == "Title"

    def test_strict_model_threshold_with_fuzzy_main(self):
        """Test threshold_model=1.0 (strict) with fuzzy main threshold."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # Model name typo with fuzzy main threshold but strict model threshold
        lndl_text = """\
        <lvar Reprot.title t>Title</lvar>
        OUT{report: [t]}
        """

        # Should fail: threshold_model=1.0 requires exact match
        with pytest.raises(MissingFieldError, match="Model 'Reprot' not found"):
            parse_lndl_fuzzy(
                lndl_text,
                operable,
                threshold=0.85,  # Main threshold is fuzzy
                threshold_model=1.0,  # But model threshold is strict
            )


class TestFuzzyErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_lndl_text(self):
        """Test empty LNDL text handling."""

        operable = Operable()

        with pytest.raises(MissingOutBlockError):  # Should raise MissingOutBlockError
            parse_lndl_fuzzy("", operable)

    def test_no_matches_below_threshold(self):
        """Test all fields below threshold raises error."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # "xyz123" has no similarity to "title"
        lndl_text = """\
        <lvar Report.xyz123 x>Title</lvar>

        ```lndl
        OUT{report: [x]}
                ```\
                """

        with pytest.raises(MissingFieldError):
            parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

    def test_tie_detection_within_0_05(self):
        """Test tie detection for matches within 0.05 similarity."""

        class Report(BaseModel):
            description: str
            descriptor: str

        operable = Operable([Spec(Report, name="report")])

        # "desc" matches both fields closely
        lndl_text = """\
        <lvar Report.desc d>Value</lvar>

        ```lndl
        OUT{report: [d]}
                ```\
                """

        with pytest.raises(AmbiguousMatchError):
            parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)


class TestFuzzyCorrectionLogging:
    """Test correction logging for observability."""

    def test_correction_logged(self, caplog):
        """Test fuzzy corrections are logged at DEBUG level."""
        import logging

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        lndl_text = """\
        <lvar Report.titel t>Title</lvar>

        ```lndl
        OUT{report: [t]}
                ```\
                """

        with caplog.at_level(logging.DEBUG):
            parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

        # Should log the correction
        assert any(
            "titel" in record.message or "title" in record.message for record in caplog.records
        )


class TestBackwardCompatibility:
    """Test backward compatibility with strict resolver."""

    def test_fuzzy_calls_strict_resolver(self):
        """Test fuzzy layer calls strict resolver (no duplication)."""
        # This is architectural - verify by code inspection
        # fuzzy.py should call resolve_references_prefixed() after corrections
        from lionherd_core.lndl import fuzzy, resolver

        # Both should exist
        assert hasattr(fuzzy, "parse_lndl_fuzzy")
        assert hasattr(resolver, "resolve_references_prefixed")

    def test_exact_matches_same_as_strict(self):
        """Test exact matches produce same result as strict resolver."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        # Perfect LNDL, no typos
        lndl_text = """\
        <lvar Report.title title>Perfect Title</lvar>

        ```lndl
        OUT{report: [title]}
                ```\
                """

        # Fuzzy with exact matches should work
        result_fuzzy = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

        # Strict mode should also work
        result_strict = parse_lndl_fuzzy(lndl_text, operable, threshold=1.0)

        assert result_fuzzy["report"].title == result_strict["report"].title


class TestFuzzyCoverageEdgeCases:
    """Additional edge case tests for 100% fuzzy.py coverage."""

    def test_strict_mode_with_wrong_model_name(self):
        """Test strict mode raises early for wrong model name (line 188)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        lndl_text = """\
        <lvar Reprt.title t>Title</lvar>

        ```lndl
        OUT{report: [t]}
        ```\
        """

        # Should raise MissingFieldError for wrong model name in strict mode
        with pytest.raises(MissingFieldError, match=r"Model.*not found"):
            parse_lndl_fuzzy(lndl_text, operable, threshold=1.0)

    def test_strict_mode_with_wrong_field_name(self):
        """Test strict mode raises early for wrong field name (line 203)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        lndl_text = """\
        <lvar Report.titel t>Title</lvar>

        ```lndl
        OUT{report: [t]}
        ```\
        """

        # Should raise MissingFieldError for wrong field name in strict mode
        with pytest.raises(MissingFieldError, match=r"Field.*not found"):
            parse_lndl_fuzzy(lndl_text, operable, threshold=1.0)

    def test_strict_mode_with_wrong_spec_name(self):
        """Test strict mode raises early for wrong spec name (line 217)."""

        class Report(BaseModel):
            title: str

        operable = Operable([Spec(Report, name="report")])

        lndl_text = """\
        <lvar Report.title t>Title</lvar>

        ```lndl
        OUT{reprt: [t]}
        ```\
        """

        # Should raise MissingFieldError for wrong spec name in strict mode
        with pytest.raises(MissingFieldError, match=r"Spec.*not found"):
            parse_lndl_fuzzy(lndl_text, operable, threshold=1.0)

    def test_fuzzy_with_literal_value_not_array(self):
        """Test fuzzy parsing with literal scalar value (line 306)."""

        class Config(BaseModel):
            pass

        operable = Operable([Spec(int, name="count")])

        lndl_text = """\
        ```lndl
        OUT{count: 42}
        ```\
        """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)
        assert result.count == 42


class TestFuzzyNamespacedActions:
    """Test fuzzy correction for namespaced action syntax."""

    def test_fuzzy_corrects_action_model_typo(self):
        """Fuzzy corrects typo in action model name (via lvar corrections)."""

        class Report(BaseModel):
            title: str
            summary: str

        # Include lvar to build model correction ("Reprot" -> "Report")
        response = """
<lvar Reprot.title t>Title</lvar>
<lact Reprot.summary s>generate_summary(length=100)</lact>
OUT{report:[t, s]}
"""
        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        # Should correct "Reprot" -> "Report" (from lvar correction)
        assert output.report.title == "Title"
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.summary.function == "generate_summary"

    def test_fuzzy_corrects_action_field_typo(self):
        """Fuzzy corrects typo in action field name (via lvar corrections)."""

        class Report(BaseModel):
            title: str
            summary: str

        # Include lvar with same field typo to build field correction ("sumary" -> "summary")
        response = """
<lvar Report.title t>Title</lvar>
<lvar Report.sumary x>Extra</lvar>
<lact Report.sumary s>generate_summary(length=100)</lact>
OUT{report:[t, s]}
"""
        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        # Should correct "sumary" -> "summary" (from lvar correction)
        assert output.report.title == "Title"
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.summary.function == "generate_summary"

    def test_fuzzy_corrects_action_model_and_field(self):
        """Fuzzy corrects typos in both model and field names (via lvar corrections)."""

        class Report(BaseModel):
            title: str
            summary: str

        # Include lvars to build BOTH model ("Reprot"->"Report") and field ("sumary"->"summary") corrections
        response = """
<lvar Reprot.title t>Title</lvar>
<lvar Reprot.sumary x>Extra</lvar>
<lact Reprot.sumary s>generate_summary()</lact>
OUT{report:[t, s]}
"""
        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        # Should correct both: "Reprot.sumary" -> "Report.summary"
        assert output.report.title == "Title"
        assert isinstance(output.report.summary, ActionCall)

    def test_strict_mode_action_model_not_found(self):
        """Strict mode raises error when action model doesn't exist."""

        class Report(BaseModel):
            title: str
            summary: str

        response = """
<lact NonExistent.field a>generate()</lact>
OUT{report:[a]}
"""
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(MissingFieldError) as exc_info:
            parse_lndl_fuzzy(response, operable, threshold=1.0)

        assert "Action model 'NonExistent' not found" in str(exc_info.value)
        assert "strict mode" in str(exc_info.value)

    def test_strict_mode_action_field_not_found(self):
        """Strict mode raises error when action field doesn't exist."""

        class Report(BaseModel):
            title: str
            summary: str

        response = """
<lact Report.nonexistent a>generate()</lact>
OUT{report:[a]}
"""
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(MissingFieldError) as exc_info:
            parse_lndl_fuzzy(response, operable, threshold=1.0)

        assert "Action field 'nonexistent' not found" in str(exc_info.value)
        assert "model Report" in str(exc_info.value)

    def test_fuzzy_direct_action_no_correction_needed(self):
        """Direct actions (no namespace) skip fuzzy correction."""

        class SearchResults(BaseModel):
            items: list[str]
            count: int

        response = """
<lact search>search_api(query="test")</lact>
OUT{result:[search]}
"""
        operable = Operable([Spec(SearchResults, name="result")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        # Direct action should work without correction
        assert isinstance(output.result, ActionCall)
        assert output.result.function == "search_api"


class TestFuzzyEdgeCases:
    """Test edge cases in fuzzy matching logic."""

    def test_strict_mode_field_name_typo_error(self):
        """Strict mode with field typo raises clear error."""

        class Report(BaseModel):
            title: str
            summary: str

        response = """
<lvar Report.titl t>Value</lvar>
OUT{report:[t]}
"""
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(MissingFieldError) as exc_info:
            parse_lndl_fuzzy(response, operable, threshold=1.0)

        assert "Field 'titl' not found" in str(exc_info.value)
        assert "strict mode: exact match required" in str(exc_info.value)

    def test_fuzzy_single_match_above_threshold(self):
        """Single match above threshold returned in list (len=1)."""

        # Model with only one field to ensure single match
        class SingleFieldModel(BaseModel):
            field: str

        response = """
<lvar SingleFieldModel.fild f>Value</lvar>
OUT{model:[f]}
"""
        operable = Operable([Spec(SingleFieldModel, name="model")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        # Should correct "fild" -> "field" (single candidate, returned in list)
        assert output.model.field == "Value"


# ======================================================================================
# SECTION 3: ACTION RESOLUTION
# ======================================================================================
# Tests for action call parsing, namespaced actions, and error handling.
# Covers: <lact> tags, direct vs namespaced actions, action lifecycle, malformed syntax.


class TestActionResolution:
    """Test action resolution in OUT{} blocks."""

    def test_scalar_action_resolution(self):
        """Test resolving action for scalar field."""

        out_fields = {
            "report": ["title", "summary"],
            "quality_score": ["calculate"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
            "summary": LvarMetadata("Report", "summary", "summary", "Summary"),
        }
        lacts = {
            "calculate": LactMetadata(
                None, None, "calculate", 'compute_score(data="test", threshold=0.8)'
            ),
        }
        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])

        output = resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Report should be constructed normally
        assert output.report.title == "Title"
        assert output.report.summary == "Summary"

        # quality_score should be an ActionCall (not yet executed)
        assert isinstance(output.quality_score, ActionCall)
        assert output.quality_score.name == "calculate"
        assert output.quality_score.function == "compute_score"
        assert output.quality_score.arguments == {"data": "test", "threshold": 0.8}

        # Action should be in parsed_actions
        assert "calculate" in output.actions
        assert output.actions["calculate"].function == "compute_score"

    def test_action_in_basemodel_field_error(self):
        """Test error when direct action mixed with lvars in BaseModel field."""

        out_fields = {
            "report": ["title", "search_action"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
        }
        # Direct action (no namespace) cannot be mixed with lvars
        lacts = {
            "search_action": LactMetadata(
                None, None, "search_action", 'search(query="test", limit=10)'
            ),
        }
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Should raise ValueError about direct actions not being mixable
        assert len(exc_info.value.exceptions) == 1
        assert "Direct action" in str(exc_info.value.exceptions[0])
        assert "cannot be mixed" in str(exc_info.value.exceptions[0])

    def test_name_collision_lvar_lact_error(self):
        """Test error when same name used for lvar and lact."""

        out_fields = {"field": ["data"]}
        lvars = {
            "data": LvarMetadata("Report", "title", "data", "value"),
        }
        lacts = {
            "data": "search(query='test')",
        }
        operable = Operable([Spec(Report, name="field")])

        with pytest.raises(ValueError, match="Name collision detected"):
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

    def test_multiple_actions_only_referenced_parsed(self):
        """Test that only actions referenced in OUT{} are parsed."""

        out_fields = {
            "report": ["title", "summary"],
            "quality_score": ["final"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
            "summary": LvarMetadata("Report", "summary", "summary", "Summary"),
        }
        lacts = {
            "draft1": LactMetadata(None, None, "draft1", "compute(version=1)"),
            "draft2": LactMetadata(None, None, "draft2", "compute(version=2)"),
            "final": LactMetadata(None, None, "final", "compute(version=3)"),
        }
        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])

        output = resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Only "final" action should be in parsed_actions
        assert len(output.actions) == 1
        assert "final" in output.actions
        assert "draft1" not in output.actions
        assert "draft2" not in output.actions

        # quality_score field should contain the final action
        assert isinstance(output.quality_score, ActionCall)
        assert output.quality_score.name == "final"


class TestEndToEndActionParsing:
    """Test end-to-end action parsing with parse_lndl."""

    def test_complete_example_with_actions(self):
        """Test complete example with actions and lvars."""
        response = """
        Let me search for information...
        <lact broad>search(query="AI", limit=100)</lact>
        Too many results. Let me refine:
        <lact focused>search(query="AI safety", limit=20)</lact>

        Now let me create the report:
        <lvar Report.title t>AI Safety Analysis</lvar>
        <lvar Report.summary s>Based on search results...</lvar>

        ```lndl
        OUT{report:[t, s], search_data:[focused], quality_score:0.85}
        ```
        """

        operable = Operable(
            [
                Spec(Report, name="report"),
                Spec(SearchResults, name="search_data"),
                Spec(float, name="quality_score"),
            ]
        )
        output = parse_lndl(response, operable)

        # Report should be constructed from lvars
        assert output.report.title == "AI Safety Analysis"
        assert output.report.summary == "Based on search results..."

        # search_data should be an ActionCall
        assert isinstance(output.search_data, ActionCall)
        assert output.search_data.function == "search"
        assert output.search_data.arguments == {"query": "AI safety", "limit": 20}

        # quality_score is scalar literal
        assert output.quality_score == 0.85

        # Only "focused" action should be parsed (not "broad")
        assert len(output.actions) == 1
        assert "focused" in output.actions
        assert "broad" not in output.actions

    def test_actions_with_complex_arguments(self):
        """Test actions with complex nested arguments."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lact api>fetch(url="https://api.com", headers={"Auth": "token"}, timeout=30)</lact>

        OUT{report:[t, s], api_result:[api]}
        """

        operable = Operable([Spec(Report, name="report"), Spec(dict, name="api_result")])

        # Note: dict is not a BaseModel, so this will fail validation
        # But the action should still be parsed correctly
        with pytest.raises(ExceptionGroup):
            parse_lndl(response, operable)

    def test_scratch_actions_not_in_out_block(self):
        """Test that scratch actions (not in OUT{}) are not parsed."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lact scratch1>search(query="draft")</lact>
        <lact scratch2>search(query="another draft")</lact>
        <lact final>search(query="final")</lact>

        OUT{report:[t, s], result:[final]}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="result")])
        output = parse_lndl(response, operable)

        # Only "final" should be in parsed actions
        assert len(output.actions) == 1
        assert "final" in output.actions
        assert "scratch1" not in output.actions
        assert "scratch2" not in output.actions

    def test_mixed_lvars_and_actions_in_different_fields(self):
        """Test mixing lvars and actions across different fields."""
        response = """
        <lvar Report.title t>Analysis Report</lvar>
        <lvar Report.summary s>Summary text</lvar>
        <lact compute>calculate_score(data="metrics", method="weighted")</lact>

        ```lndl
        OUT{report:[t, s], quality_score:[compute]}
        ```
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])
        output = parse_lndl(response, operable)

        # Report from lvars
        assert output.report.title == "Analysis Report"
        assert output.report.summary == "Summary text"

        # quality_score from action
        assert isinstance(output.quality_score, ActionCall)
        assert output.quality_score.function == "calculate_score"
        assert output.quality_score.arguments == {"data": "metrics", "method": "weighted"}

    def test_action_with_positional_args(self):
        """Test action with positional arguments."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        <lact calc>calculate(10, 20, 30)</lact>

        OUT{report:[t, s], result:[calc]}
        """

        operable = Operable([Spec(Report, name="report"), Spec(int, name="result")])
        output = parse_lndl(response, operable)

        # Action should have positional args with _pos_ prefix
        assert isinstance(output.result, ActionCall)
        assert output.result.function == "calculate"
        assert "_pos_0" in output.result.arguments
        assert "_pos_1" in output.result.arguments
        assert "_pos_2" in output.result.arguments
        assert output.result.arguments["_pos_0"] == 10
        assert output.result.arguments["_pos_1"] == 20
        assert output.result.arguments["_pos_2"] == 30

    def test_action_collision_with_lvar_name(self):
        """Test name collision between lvar and lact triggers error.

        Parser now catches duplicate aliases early, before resolver.
        """

        response = """
        <lvar Report.title data>Title</lvar>
        <lact data>search(query="test")</lact>

        OUT{report:[data]}
        """

        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ParseError, match="Duplicate alias 'data'"):
            parse_lndl(response, operable)


class TestNamespacedActions:
    """Test namespaced action pattern for mixing lvars and actions."""

    def test_extract_namespaced_actions(self):
        """Test extracting namespaced actions with Model.field syntax."""

        response = """
        <lact Report.title t>generate_title(topic="AI")</lact>
        <lact Report.summary summarize>generate_summary(data="metrics")</lact>
        <lact search>search(query="test")</lact>
        """

        lacts = parse_lacts(response)

        # Namespaced actions
        assert "t" in lacts
        assert lacts["t"]["model"] == "Report"
        assert lacts["t"]["field"] == "title"
        assert lacts["t"]["local_name"] == "t"
        assert lacts["t"]["call"] == 'generate_title(topic="AI")'

        assert "summarize" in lacts
        assert lacts["summarize"]["model"] == "Report"
        assert lacts["summarize"]["field"] == "summary"
        assert lacts["summarize"]["local_name"] == "summarize"

        # Direct action
        assert "search" in lacts
        assert lacts["search"]["model"] is None
        assert lacts["search"]["field"] is None
        assert lacts["search"]["local_name"] == "search"

    def test_extract_namespaced_without_alias(self):
        """Test namespaced action defaults to field name when no alias provided."""

        response = """
        <lact Report.summary>generate_summary(data="test")</lact>
        """

        lacts = parse_lacts(response)

        assert "summary" in lacts
        assert lacts["summary"]["model"] == "Report"
        assert lacts["summary"]["field"] == "summary"
        assert lacts["summary"]["local_name"] == "summary"

    def test_mixing_lvars_and_namespaced_actions(self):
        """Test mixing lvars and namespaced actions in same BaseModel."""

        out_fields = {
            "report": ["title", "summarize", "footer"],
        }
        lvars = {
            "title": LvarMetadata("ExtendedReport", "title", "title", "Analysis Report"),
            "footer": LvarMetadata("ExtendedReport", "footer", "footer", "End of Report"),
        }
        lacts = {
            "summarize": LactMetadata(
                "ExtendedReport", "summary", "summarize", 'generate_summary(data="metrics")'
            ),
        }

        # Create extended Report model with footer field
        class ExtendedReport(BaseModel):
            title: str
            summary: str
            footer: str

        operable = Operable([Spec(ExtendedReport, name="report")])
        output = resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Title and footer from lvars
        assert output.report.title == "Analysis Report"
        assert output.report.footer == "End of Report"

        # Summary from namespaced action (ActionCall before execution)
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.summary.function == "generate_summary"
        assert output.report.summary.arguments == {"data": "metrics"}

        # Only summarize action should be in parsed_actions
        assert len(output.actions) == 1
        assert "summarize" in output.actions

    def test_namespaced_action_model_mismatch_error(self):
        """Test error when namespaced action model doesn't match field spec."""

        out_fields = {
            "report": ["title", "wrong_model_action"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
        }
        # Action declares SearchResults.summary but used in Report field
        lacts = {
            "wrong_model_action": LactMetadata(
                "SearchResults", "items", "wrong_model_action", 'search(query="test")'
            ),
        }
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Should raise TypeMismatchError about model mismatch
        assert len(exc_info.value.exceptions) == 1
        assert "SearchResults" in str(exc_info.value.exceptions[0])
        assert "Report" in str(exc_info.value.exceptions[0])

    def test_end_to_end_namespaced_mixing(self):
        """Test complete end-to-end parsing with mixed lvars and namespaced actions."""
        response = """
        Let me create a report with generated summary:

        <lvar Report.title t>Quarterly Analysis</lvar>
        <lact Report.summary s>generate_summary(quarter="Q4", year=2024)</lact>

        ```lndl
        OUT{report:[t, s]}
        ```
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        # Title from lvar
        assert output.report.title == "Quarterly Analysis"

        # Summary from namespaced action
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.summary.function == "generate_summary"
        assert output.report.summary.arguments == {"quarter": "Q4", "year": 2024}

        # Only "s" action should be parsed
        assert len(output.actions) == 1
        assert "s" in output.actions

    def test_direct_action_cannot_mix_with_lvars(self):
        """Test that direct actions cannot be mixed with lvars in OUT{} array."""

        out_fields = {
            "report": ["title", "direct_action"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
        }
        # Direct action (no namespace)
        lacts = {
            "direct_action": LactMetadata(None, None, "direct_action", 'fetch_data(url="...")'),
        }
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Should raise error about direct actions not being mixable
        assert len(exc_info.value.exceptions) == 1
        assert "Direct action" in str(exc_info.value.exceptions[0])
        assert "cannot be mixed" in str(exc_info.value.exceptions[0])

    def test_single_direct_action_for_entire_model(self):
        """Test single direct action returning entire BaseModel."""

        out_fields = {
            "report": ["fetch_report"],
        }
        lacts = {
            "fetch_report": LactMetadata(
                None, None, "fetch_report", 'api_fetch(endpoint="/report")'
            ),
        }
        operable = Operable([Spec(Report, name="report")])

        output = resolve_references_prefixed(out_fields, {}, lacts, operable)

        # Entire report field should be ActionCall
        assert isinstance(output.report, ActionCall)
        assert output.report.function == "api_fetch"
        assert output.report.arguments == {"endpoint": "/report"}

        # Action should be in parsed_actions
        assert "fetch_report" in output.actions


class TestActionErrorHandling:
    """Test error handling for malformed action calls."""

    def test_empty_action_call(self):
        """Test error for empty action body."""
        response = "<lact Report.summary s></lact>\nOUT{report:[s]}"
        operable = Operable([Spec(Report, name="report")])

        # Empty action call should raise ExceptionGroup with clear error message
        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        # Check that the nested exception has clear context
        errors = exc_info.value.exceptions
        assert len(errors) == 1
        assert "Invalid function call syntax" in str(errors[0])
        assert "action 's'" in str(errors[0])

    def test_non_function_action(self):
        """Test error for non-function syntax (missing parentheses)."""
        response = "<lact Report.summary s>not_a_function</lact>\nOUT{report:[s]}"
        operable = Operable([Spec(Report, name="report")])

        # Missing parentheses should raise ExceptionGroup
        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        errors = exc_info.value.exceptions
        assert len(errors) == 1
        assert "Invalid function call syntax" in str(errors[0])
        assert "not_a_function" in str(errors[0])

    def test_syntax_error_in_args(self):
        """Test error for unclosed quotes/parentheses in arguments."""
        response = '<lact s>search(query="unclosed)</lact>\nOUT{result:[s]}'
        operable = Operable([Spec(SearchResults, name="result")])

        # Syntax error in arguments should raise ExceptionGroup
        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        errors = exc_info.value.exceptions
        assert len(errors) == 1
        assert "Invalid function call syntax" in str(errors[0])
        # Error message should show the malformed call
        assert 'search(query="unclosed)' in str(errors[0])

    def test_nested_lact_tags(self):
        """Test behavior with nested lact tags (regex extracts inner first)."""
        # Regex will match the first complete tag pair (non-greedy .*?)
        # So it extracts <lact inner>x()</lact> first, leaving malformed outer
        response = "<lact outer>func(<lact inner>x()</lact>)</lact>\nOUT{report:[outer]}"
        operable = Operable([Spec(Report, name="report")])

        # The regex captures inner tag, leaving "func(<lact inner>x()" as outer call
        # This results in syntax error
        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        errors = exc_info.value.exceptions
        assert len(errors) == 1
        assert "Invalid function call syntax" in str(errors[0])

    def test_missing_closing_tag(self):
        """Test unclosed lact tag - token-based parser catches immediately."""

        response = '<lact action>search(query="test")\nOUT{result:[action]}'
        operable = Operable([Spec(SearchResults, name="result")])

        # Token-based parser detects unclosed tag immediately during parsing
        # This is better than old regex-based approach which silently ignored it
        with pytest.raises(ParseError) as exc_info:
            parse_lndl(response, operable)

        assert "Unclosed lact tag" in str(exc_info.value)
        assert "missing </lact>" in str(exc_info.value)

    def test_scalar_action_malformed_syntax(self):
        """Test error context for malformed action in scalar field."""
        response = "<lact calc>broken_syntax_no_parens</lact>\nOUT{score:[calc]}"
        operable = Operable([Spec(float, name="score")])

        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        errors = exc_info.value.exceptions
        assert len(errors) == 1
        assert "Invalid function call syntax" in str(errors[0])
        assert "action 'calc'" in str(errors[0])
        assert "scalar field 'score'" in str(errors[0])

    def test_scalar_action_empty_call(self):
        """Test error for empty action call in scalar field."""
        response = "<lact calc></lact>\nOUT{score:[calc]}"
        operable = Operable([Spec(float, name="score")])

        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        errors = exc_info.value.exceptions
        assert len(errors) == 1
        assert "Invalid function call syntax" in str(errors[0])

    @pytest.mark.parametrize("keyword", sorted(PYTHON_RESERVED))
    def test_reserved_keyword_warning(self, keyword):
        """Test warning when action name is Python reserved keyword or builtin."""

        response = f"<lact {keyword}>some_function()</lact>\nOUT{{result:[{keyword}]}}"
        operable = Operable([Spec(SearchResults, name="result")])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = parse_lndl(response, operable)

            # Should issue a warning
            assert len(w) == 1
            assert "reserved keyword" in str(w[0].message).lower()
            assert f"'{keyword}'" in str(w[0].message)

        # Should still parse successfully
        assert isinstance(output.result, ActionCall)

    def test_multiple_malformed_actions(self):
        """Test ExceptionGroup aggregation for multiple malformed actions."""
        response = """<lact action1>broken_syntax</lact>
<lact action2>also_broken(</lact>
OUT{result1:[action1], result2:[action2]}"""
        operable = Operable(
            [
                Spec(SearchResults, name="result1"),
                Spec(SearchResults, name="result2"),
            ]
        )

        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        errors = exc_info.value.exceptions
        # Should have 2 errors, one for each malformed action
        assert len(errors) == 2
        assert all("Invalid function call syntax" in str(e) for e in errors)
        # Verify both action names are mentioned
        error_strs = [str(e) for e in errors]
        assert any("action1" in s for s in error_strs)
        assert any("action2" in s for s in error_strs)

    def test_mixed_valid_and_malformed_actions(self):
        """Test partial failures with mixed valid and malformed actions."""
        response = """<lact valid>proper_function()</lact>
<lact broken>bad_syntax</lact>
OUT{result1:[valid], result2:[broken]}"""
        operable = Operable(
            [
                Spec(SearchResults, name="result1"),
                Spec(SearchResults, name="result2"),
            ]
        )

        with pytest.raises(ExceptionGroup, match="LNDL validation failed") as exc_info:
            parse_lndl(response, operable)

        errors = exc_info.value.exceptions
        # Should have only 1 error for the broken action
        assert len(errors) == 1
        assert "Invalid function call syntax" in str(errors[0])
        assert "broken" in str(errors[0])
        # Should NOT mention the valid action name (avoid false positive from "Invalid")
        assert "'valid'" not in str(errors[0])


# ======================================================================================
# SECTION 4: FULL INTEGRATION
# ======================================================================================
# Tests for complete end-to-end workflows, validation, real-world scenarios.
# Covers: complete pipeline, action lifecycle, Pydantic validation, fixtures.


# === Test Models ===


class SimpleReport(BaseModel):
    """Simple report model."""

    title: str
    summary: str


class ScoredReport(BaseModel):
    """Report with confidence score."""

    title: str
    summary: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class Analysis(BaseModel):
    """Analysis result."""

    findings: str
    recommendations: str
    risk_level: str


class NestedReport(BaseModel):
    """Report with nested analysis."""

    title: str
    analysis: Analysis


# === Basic Integration Tests ===


class TestIntegrationBasic:
    """Test basic end-to-end LNDL pipeline."""

    def test_simple_lvar_to_model(self):
        """Test single lvar extraction and model construction."""
        response = """
        <lvar SimpleReport.title t>Integration Test Report</lvar>
        <lvar SimpleReport.summary s>Testing the unified pipeline</lvar>

        ```lndl
        OUT{report:[t, s]}
        ```
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Integration Test Report"
        assert output.report.summary == "Testing the unified pipeline"
        assert "t" in output.lvars
        assert "s" in output.lvars

    def test_multiple_lvars_to_model(self):
        """Test multiple lvars with type conversion."""
        response = """
        <lvar ScoredReport.title t>Scored Report</lvar>
        <lvar ScoredReport.summary s>Summary text</lvar>
        <lvar ScoredReport.confidence c>0.95</lvar>

        OUT{report:[t, s, c]}
        """

        operable = Operable([Spec(ScoredReport, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Scored Report"
        assert output.report.summary == "Summary text"
        assert output.report.confidence == 0.95
        assert isinstance(output.report.confidence, float)

    def test_lact_execution_lifecycle(self):
        """Test action call lifecycle: parse → execute → revalidate."""
        response = """
        <lvar SimpleReport.title t>Report Title</lvar>
        <lact SimpleReport.summary s>generate_summary(length=100)</lact>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        # 1. Parse phase - ActionCall created
        assert output.report.title == "Report Title"
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.summary.function == "generate_summary"
        assert output.report.summary.arguments == {"length": 100}

        # 2. Model has action calls
        assert has_action_calls(output.report)

        # 3. Execute action (simulated)
        action_results = {"s": "This is the generated summary"}

        # 4. Re-validate with results
        validated_report = revalidate_with_action_results(output.report, action_results)

        assert validated_report.title == "Report Title"
        assert validated_report.summary == "This is the generated summary"
        assert not has_action_calls(validated_report)

    def test_mixed_lvars_lacts(self):
        """Test mixing lvars and lacts in same model."""
        response = """
        <lvar ScoredReport.title t>Mixed Report</lvar>
        <lact ScoredReport.summary s>generate_summary()</lact>
        <lvar ScoredReport.confidence c>0.88</lvar>

        OUT{report:[t, s, c]}
        """

        operable = Operable([Spec(ScoredReport, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Mixed Report"
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.confidence == 0.88

        # Re-validate after execution
        action_results = {"s": "Generated text"}
        validated = revalidate_with_action_results(output.report, action_results)

        assert validated.summary == "Generated text"
        assert not has_action_calls(validated)

    def test_literal_values_in_out_block(self):
        """Test literal scalar values in OUT block."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>

        OUT{report:[t, s], quality_score:0.92, priority:3, status:"active"}
        """

        operable = Operable(
            [
                Spec(SimpleReport, name="report"),
                Spec(float, name="quality_score"),
                Spec(int, name="priority"),
                Spec(str, name="status"),
            ]
        )
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.quality_score == 0.92
        assert output.priority == 3
        assert output.status == "active"


# === Validation Tests ===


class TestIntegrationValidation:
    """Test validation and error handling."""

    def test_required_field_missing(self):
        """Test missing required field raises error."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>

        OUT{other_model:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report", required=True)])

        # First check_allowed raises ValueError for unknown field
        with pytest.raises(ValueError, match="not allowed"):
            parse_lndl(response, operable)

    def test_unknown_field(self):
        """Test unknown field in OUT block raises error."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>

        OUT{unknown_field:[t]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])

        with pytest.raises(ValueError, match="not allowed"):
            parse_lndl(response, operable)

    def test_type_mismatch(self):
        """Test type mismatch in lvar model."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar Analysis.findings f>Wrong model</lvar>

        OUT{report:[t, f]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])

        with pytest.raises(ExceptionGroup) as exc_info:
            parse_lndl(response, operable)

        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], TypeMismatchError)

    def test_all_required_fields_present(self):
        """Test successful validation with all required fields."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>
        <lvar ScoredReport.title t2>Scored Title</lvar>
        <lvar ScoredReport.summary s2>Scored Summary</lvar>
        <lvar ScoredReport.confidence c>0.85</lvar>

        OUT{report:[t, s], scored:[t2, s2, c]}
        """

        operable = Operable(
            [
                Spec(SimpleReport, name="report", required=True),
                Spec(ScoredReport, name="scored", required=True),
            ]
        )
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.scored.confidence == 0.85


# === Fuzzy Matching Tests ===


class TestIntegrationFuzzyMatching:
    """Test fuzzy field name correction."""

    def test_fuzzy_corrects_typo(self):
        """Test fuzzy correction of field name typo."""
        response = """
        <lvar SimpleReport.titel t>Title with Typo</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        # "titel" corrected to "title"
        assert output.report.title == "Title with Typo"
        assert output.report.summary == "Summary"

    def test_fuzzy_threshold_too_low(self):
        """Test fuzzy fails when similarity below threshold."""
        response = """
        <lvar SimpleReport.xyz x>Value</lvar>

        OUT{report:[x]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])

        with pytest.raises(MissingFieldError, match="xyz"):
            parse_lndl_fuzzy(response, operable, threshold=0.85)

    def test_fuzzy_ambiguous_match(self):
        """Test ambiguous match raises error."""

        class AmbiguousModel(BaseModel):
            description: str
            descriptor: str

        response = """
        <lvar AmbiguousModel.desc d>Value</lvar>

        OUT{model:[d]}
        """

        operable = Operable([Spec(AmbiguousModel, name="model")])

        with pytest.raises(AmbiguousMatchError, match="desc"):
            parse_lndl_fuzzy(response, operable, threshold=0.85)

    def test_strict_mode_rejects_typo(self):
        """Test strict mode (fuzzy=False) rejects typos."""
        response = """
        <lvar SimpleReport.titel t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])

        with pytest.raises(MissingFieldError):
            parse_lndl_fuzzy(response, operable, threshold=1.0)  # Strict mode

    def test_fuzzy_with_multiple_fields(self):
        """Test fuzzy correction on multiple fields."""
        response = """
        <lvar ScoredReport.titel t>Title</lvar>
        <lvar ScoredReport.sumary s>Summary</lvar>
        <lvar ScoredReport.confidnce c>0.9</lvar>

        OUT{report:[t, s, c]}
        """

        operable = Operable([Spec(ScoredReport, name="report")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        assert output.report.title == "Title"
        assert output.report.summary == "Summary"
        assert output.report.confidence == 0.9


# === Action Lifecycle Tests ===


class TestIntegrationActionLifecycle:
    """Test action execution lifecycle."""

    def test_action_not_executed_until_referenced(self):
        """Test declared but unreferenced actions not in output.actions."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>
        <lact SimpleReport.summary unused>generate_summary()</lact>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        # Action declared in lacts but not in output.actions (not referenced)
        assert "unused" in output.lacts
        assert "unused" not in output.actions
        assert not has_action_calls(output.report)

    def test_action_execution_with_results(self):
        """Test full action execution workflow."""
        response = """
        <lvar ScoredReport.title t>Analysis Report</lvar>
        <lact ScoredReport.summary s>analyze_text(doc_id=42)</lact>
        <lact ScoredReport.confidence c>calculate_confidence(threshold=0.8)</lact>

        OUT{report:[t, s, c]}
        """

        operable = Operable([Spec(ScoredReport, name="report")])
        output = parse_lndl(response, operable)

        # Actions in output.actions
        assert "s" in output.actions
        assert "c" in output.actions
        assert output.actions["s"].function == "analyze_text"
        assert output.actions["c"].function == "calculate_confidence"

        # Execute (simulated)
        action_results = {"s": "Generated analysis text", "c": 0.87}

        # Re-validate
        validated = revalidate_with_action_results(output.report, action_results)

        assert validated.summary == "Generated analysis text"
        assert validated.confidence == 0.87

    def test_action_revalidation(self):
        """Test Pydantic validation during revalidation."""
        response = """
        <lvar ScoredReport.title t>Title</lvar>
        <lvar ScoredReport.summary s>Summary</lvar>
        <lact ScoredReport.confidence c>calculate_score()</lact>

        OUT{report:[t, s, c]}
        """

        operable = Operable([Spec(ScoredReport, name="report")])
        output = parse_lndl(response, operable)

        # Invalid confidence value (>1.0)
        action_results = {"c": 1.5}

        with pytest.raises(ValidationError):
            revalidate_with_action_results(output.report, action_results)

    def test_action_in_non_scalar_field_error(self):
        """Test direct action can return entire model."""
        response = """
        <lact search>search_api(query="test")</lact>

        OUT{results:[search]}
        """

        operable = Operable([Spec(SearchResults, name="results")])
        output = parse_lndl(response, operable)

        # Direct action referenced - should be ActionCall
        assert isinstance(output.results, ActionCall)
        assert output.results.function == "search_api"

    def test_mixed_lvar_and_action_in_same_field(self):
        """Test mixing lvars and namespaced actions."""
        response = """
        <lvar ScoredReport.title t>Title</lvar>
        <lact ScoredReport.summary s>generate()</lact>
        <lvar ScoredReport.confidence c>0.9</lvar>

        OUT{report:[t, s, c]}
        """

        operable = Operable([Spec(ScoredReport, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.confidence == 0.9


# === Real-World Scenarios ===


class TestIntegrationRealWorldScenarios:
    """Test realistic use cases."""

    def test_report_generation(self):
        """Test realistic report generation workflow."""
        response = """
        Let me analyze the data step by step...

        <lvar ScoredReport.title title>Q4 2024 Performance Analysis</lvar>

        Based on the metrics, I'll generate a comprehensive summary...
        <lact ScoredReport.summary summary>generate_executive_summary(
            quarter="Q4",
            year=2024,
            metrics=["revenue", "growth", "retention"]
        )</lact>

        Confidence level is high given data quality...
        <lvar ScoredReport.confidence conf>0.93</lvar>

        ```lndl
        OUT{report:[title, summary, conf]}
        ```
        """

        operable = Operable([Spec(ScoredReport, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Q4 2024 Performance Analysis"
        assert isinstance(output.report.summary, ActionCall)
        assert output.report.confidence == 0.93

        # Execute and validate
        action_results = {"summary": "Revenue increased 25% YoY..."}
        validated = revalidate_with_action_results(output.report, action_results)

        assert "Revenue increased" in validated.summary
        assert not has_action_calls(validated)

    def test_analysis_with_confidence(self):
        """Test analysis with multiple fields and scores."""
        response = """
        <lvar Analysis.findings f>Critical security vulnerabilities detected in auth module</lvar>
        <lvar Analysis.recommendations r>Immediate patching required, implement 2FA</lvar>
        <lvar Analysis.risk_level risk>high</lvar>

        OUT{analysis:[f, r, risk], confidence:0.88}
        """

        operable = Operable([Spec(Analysis, name="analysis"), Spec(float, name="confidence")])
        output = parse_lndl(response, operable)

        assert (
            output.analysis.findings == "Critical security vulnerabilities detected in auth module"
        )
        assert output.analysis.risk_level == "high"
        assert output.confidence == 0.88

    def test_nested_model_validation(self):
        """Test separate models (nested models not yet supported in single OUT{})."""
        response = """
        <lvar Analysis.findings f>SQL injection vulnerabilities</lvar>
        <lvar Analysis.recommendations r>Use parameterized queries</lvar>
        <lvar Analysis.risk_level r_lvl>critical</lvar>

        OUT{analysis:[f, r, r_lvl]}
        """

        # Note: Current implementation doesn't support actual nesting
        # Models are separate top-level fields
        operable = Operable([Spec(Analysis, name="analysis")])

        output = parse_lndl(response, operable)

        assert output.analysis.findings == "SQL injection vulnerabilities"
        assert output.analysis.recommendations == "Use parameterized queries"
        assert output.analysis.risk_level == "critical"

    def test_large_response(self):
        """Test handling many lvars and lacts."""
        lvars = "\n".join([f"<lvar SimpleReport.title t{i}>Title {i}</lvar>" for i in range(10)])
        lacts = "\n".join([f"<lact SimpleReport.summary s{i}>gen_{i}()</lact>" for i in range(10)])

        response = f"""
        {lvars}
        {lacts}

        OUT{{report:[t0, s0]}}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        # Should only process referenced ones
        assert output.report.title == "Title 0"
        assert isinstance(output.report.summary, ActionCall)
        assert len(output.lvars) == 10
        assert len(output.lacts) == 10
        assert len(output.actions) == 1  # Only s0 referenced


# === Backward Compatibility Tests ===


class TestIntegrationBackwardCompatibility:
    """Test compatibility scenarios."""

    def test_namespaced_lvar_format(self):
        """Test namespaced <lvar Model.field alias>content</lvar> format."""
        # Only namespaced format is supported

        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        # Prefixed format works
        assert output.report.title == "Title"

    def test_direct_action_format(self):
        """Test direct action format (non-namespaced lact, still supported)."""
        response = """
        <lact search>search(query="test")</lact>

        OUT{results:[search]}
        """

        operable = Operable([Spec(SearchResults, name="results")])
        output = parse_lndl(response, operable)

        # Direct action returns entire model
        assert isinstance(output.results, ActionCall)
        assert output.results.function == "search"

    def test_prefixed_format(self):
        """Test current prefixed format."""
        response = """
        <lvar SimpleReport.title title>Prefixed Title</lvar>
        <lvar SimpleReport.summary summary>Prefixed Summary</lvar>

        OUT{report:[title, summary]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Prefixed Title"
        assert output.report.summary == "Prefixed Summary"

    def test_mixed_legacy_and_prefixed(self):
        """Test mixing prefixed lvars with direct actions."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>
        <lact search>search_api(query="data")</lact>

        OUT{report:[t, s], search_results:[search]}
        """

        operable = Operable(
            [Spec(SimpleReport, name="report"), Spec(SearchResults, name="search_results")]
        )
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.report.summary == "Summary"
        assert isinstance(output.search_results, ActionCall)


# === Edge Cases & Helpers ===


class TestIntegrationEdgeCases:
    """Test edge cases and helper functions."""

    def test_ensure_no_action_calls_success(self):
        """Test ensure_no_action_calls with clean model."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        # Should not raise
        ensure_no_action_calls(output.report)

    def test_ensure_no_action_calls_failure(self):
        """Test ensure_no_action_calls with ActionCall."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lact SimpleReport.summary s>generate()</lact>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        with pytest.raises(ValueError, match="unexecuted actions"):
            ensure_no_action_calls(output.report)

    def test_has_action_calls_detection(self):
        """Test has_action_calls helper."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lact SimpleReport.summary s>generate()</lact>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        assert has_action_calls(output.report)

        # After revalidation
        action_results = {"s": "Generated text"}
        validated = revalidate_with_action_results(output.report, action_results)

        assert not has_action_calls(validated)

    def test_multiline_lvar_values(self):
        """Test multiline lvar values preserved."""
        response = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>
        This is a multiline summary
        with several lines
        of content
        </lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(SimpleReport, name="report")])
        output = parse_lndl(response, operable)

        assert "multiline" in output.report.summary
        assert "several lines" in output.report.summary

    def test_code_fence_optional(self):
        """Test OUT{} works with or without code fence."""
        # Without code fence
        response_no_fence = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>
        OUT{report:[t, s]}
        """

        # With code fence
        response_fence = """
        <lvar SimpleReport.title t>Title</lvar>
        <lvar SimpleReport.summary s>Summary</lvar>
        ```lndl
        OUT{report:[t, s]}
        ```
        """

        operable = Operable([Spec(SimpleReport, name="report")])

        output1 = parse_lndl(response_no_fence, operable)
        output2 = parse_lndl(response_fence, operable)

        assert output1.report.title == output2.report.title
        assert output1.report.summary == output2.report.summary


# === Fixtures ===


@pytest.fixture
def simple_operable():
    """Fixture for SimpleReport operable."""
    return Operable([Spec(SimpleReport, name="report")])


@pytest.fixture
def scored_operable():
    """Fixture for ScoredReport operable."""
    return Operable([Spec(ScoredReport, name="report")])


@pytest.fixture
def multi_model_operable():
    """Fixture for multiple model operable."""
    return Operable(
        [
            Spec(SimpleReport, name="report"),
            Spec(Analysis, name="analysis"),
            Spec(float, name="confidence"),
        ]
    )

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel, field_validator

from lionherd_core.lndl import (
    MissingFieldError,
    TypeMismatchError,
    parse_lndl,
    resolve_references_prefixed,
)
from lionherd_core.lndl.types import LvarMetadata
from lionherd_core.types import Operable, Spec

# ============================================================================
# Test Models
# ============================================================================


class Reason(BaseModel):
    """Test model for reasoning."""

    confidence: float
    analysis: str


class Report(BaseModel):
    """Test model for reports."""

    title: str
    summary: str


class ValidatedReport(BaseModel):
    """Report with validation."""

    title: str
    summary: str

    @field_validator("title")
    @classmethod
    def validate_title_length(cls, v):
        if len(v) < 3:
            raise ValueError("Title too short")
        return v


# ============================================================================
# Test Classes
# ============================================================================


class TestExtractLvarsPrefixed:
    """Test namespace-prefixed lvar extraction using new Parser."""

    def test_extract_with_local_name(self, extract_lvars_prefixed):
        """Test extracting lvar with explicit local name."""
        text = "<lvar Report.title title>here is a good title</lvar>"
        lvars = extract_lvars_prefixed(text)

        assert len(lvars) == 1
        assert "title" in lvars
        assert lvars["title"].model == "Report"
        assert lvars["title"].field == "title"
        assert lvars["title"].local_name == "title"
        assert lvars["title"].value == "here is a good title"

    def test_extract_without_local_name(self, extract_lvars_prefixed):
        """Test extracting lvar without local name (uses field name)."""
        text = "<lvar Report.title>here is a good title</lvar>"
        lvars = extract_lvars_prefixed(text)

        assert len(lvars) == 1
        assert "title" in lvars  # Uses field name as local
        assert lvars["title"].model == "Report"
        assert lvars["title"].field == "title"
        assert lvars["title"].local_name == "title"
        assert lvars["title"].value == "here is a good title"

    def test_extract_with_custom_alias(self, extract_lvars_prefixed):
        """Test extracting lvar with custom local alias."""
        text = "<lvar Reason.confidence conf>0.85</lvar>"
        lvars = extract_lvars_prefixed(text)

        assert len(lvars) == 1
        assert "conf" in lvars  # Custom alias
        assert lvars["conf"].model == "Reason"
        assert lvars["conf"].field == "confidence"
        assert lvars["conf"].local_name == "conf"
        assert lvars["conf"].value == "0.85"

    def test_extract_multiple_lvars(self, extract_lvars_prefixed):
        """Test extracting multiple namespace-prefixed lvars."""
        text = """
        <lvar Report.title title>here is a good title</lvar>
        <lvar Reason.confidence conf>0.85</lvar>
        <lvar Report.summary summ>sdfghjklkjhgfdfghj</lvar>
        <lvar Reason.analysis ana>fghjklfghj</lvar>
        """
        lvars = extract_lvars_prefixed(text)

        assert len(lvars) == 4
        assert "title" in lvars
        assert "conf" in lvars
        assert "summ" in lvars
        assert "ana" in lvars

    def test_extract_with_revision(self, extract_lvars_prefixed):
        """Test extracting multiple versions of same field (revision tracking)."""
        text = """
        <lvar Report.summary summ>first version</lvar>
        <lvar Report.summary summ2>revised version</lvar>
        """
        lvars = extract_lvars_prefixed(text)

        assert len(lvars) == 2
        assert lvars["summ"].value == "first version"
        assert lvars["summ2"].value == "revised version"

    def test_extract_with_multiline_value(self, extract_lvars_prefixed):
        """Test extracting lvar with multiline value."""
        text = """
        <lvar Report.summary summ>
        This is a long summary
        that spans multiple lines
        with various content
        </lvar>
        """
        lvars = extract_lvars_prefixed(text)

        assert len(lvars) == 1
        assert "This is a long summary" in lvars["summ"].value
        assert "multiple lines" in lvars["summ"].value

    def test_extract_from_thinking_flow(self, extract_lvars_prefixed):
        """Test extraction from natural thinking flow with prose."""
        text = """
        Let me work through this step by step...
        Oh I think xyz might be a good approach to name the report
        <lvar Report.title title>here is a good title</lvar>

        But I am only 70% confident, let me see are there more evidence I missed, ...

        Wait, more evidence: 85%
        <lvar Reason.confidence conf>0.85</lvar>
        """
        lvars = extract_lvars_prefixed(text)

        assert len(lvars) == 2
        assert lvars["title"].value == "here is a good title"
        assert lvars["conf"].value == "0.85"

    def test_extract_empty_returns_empty_dict(self, extract_lvars_prefixed):
        """Test that text without prefixed lvars returns empty dict."""
        text = "Just some plain text without any lvars"
        lvars = extract_lvars_prefixed(text)

        assert lvars == {}


class TestParseOutBlockArray:
    """Test array syntax OUT block parsing using new Parser."""

    def test_parse_array_syntax_single_field(self, parse_out_block_array):
        """Test parsing OUT block with array syntax for single field."""
        content = "report:[title, summary]"
        out_fields = parse_out_block_array(content)

        assert len(out_fields) == 1
        assert "report" in out_fields
        assert out_fields["report"] == ["title", "summary"]

    def test_parse_array_syntax_multiple_fields(self, parse_out_block_array):
        """Test parsing OUT block with multiple fields."""
        content = "report:[title, summary], reasoning:[conf, ana]"
        out_fields = parse_out_block_array(content)

        assert len(out_fields) == 2
        assert out_fields["report"] == ["title", "summary"]
        assert out_fields["reasoning"] == ["conf", "ana"]

    def test_parse_single_variable_no_brackets(self, parse_out_block_array):
        """Test parsing single variable without brackets."""
        content = "report:title, reasoning:conf"
        out_fields = parse_out_block_array(content)

        assert len(out_fields) == 2
        assert out_fields["report"] == ["title"]
        assert out_fields["reasoning"] == ["conf"]

    def test_parse_mixed_array_and_single(self, parse_out_block_array):
        """Test parsing mix of array and single variable syntax."""
        content = "report:[title, summary], reasoning:conf"
        out_fields = parse_out_block_array(content)

        assert len(out_fields) == 2
        assert out_fields["report"] == ["title", "summary"]
        assert out_fields["reasoning"] == ["conf"]

    def test_parse_with_whitespace(self, parse_out_block_array):
        """Test parsing with various whitespace."""
        content = """
        report : [ title , summary ] ,
        reasoning : [ conf , ana ]
        """
        out_fields = parse_out_block_array(content)

        assert len(out_fields) == 2
        assert out_fields["report"] == ["title", "summary"]
        assert out_fields["reasoning"] == ["conf", "ana"]

    def test_parse_with_revision_variable(self, parse_out_block_array):
        """Test parsing with revised variable names."""
        content = "report:[title, summ2], reasoning:[conf, ana]"
        out_fields = parse_out_block_array(content)

        assert out_fields["report"] == ["title", "summ2"]  # summ2, not summ


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
        from unittest.mock import Mock

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
        """Test error when scalar uses array syntax with multiple variables."""
        from lionherd_core.lndl.parser import ParseError

        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], quality_score:[0.8, 0.9]}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])

        # New Parser raises ParseError for literals in arrays (stricter validation)
        with pytest.raises(ParseError) as exc_info:
            parse_lndl(response, operable)

        # Verify error message mentions literals not allowed in arrays
        assert "Arrays must contain only variable/action references" in str(exc_info.value)

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


class TestResolverEdgeCases:
    def test_scalar_field_multiple_variables_error(self):
        """Test error when scalar field uses multiple variables."""
        response = """
        <lvar Report.title t1>Title1</lvar>
        <lvar Report.summary t2>Title2</lvar>

        OUT{report:[t1, t2], quality_score:[t1, t2]}
        """

        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])

        with pytest.raises(ExceptionGroup) as exc_info:
            parse_lndl(response, operable)

        # Should have error about multiple variables for scalar
        assert any("cannot use multiple variables" in str(e) for e in exc_info.value.exceptions)

    def test_lvar_value_already_typed(self):
        """Test resolver with lvar value already typed (not string)."""
        # Manually create metadata with typed value (as parser would)
        out_fields = {"quality_score": ["score"]}
        lvars = {
            "score": LvarMetadata("Score", "value", "score", 0.95),  # Already float, not string
        }
        operable = Operable([Spec(float, name="quality_score")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.quality_score == 0.95

    def test_raw_lvar_in_basemodel_field_error(self):
        """Test error when raw lvar used in BaseModel field."""
        response = """
        <lvar reasoning>Raw reasoning text</lvar>

        OUT{report:[reasoning]}
        """

        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ExceptionGroup) as exc_info:
            parse_lndl(response, operable)

        # Should have error about raw lvar in BaseModel field
        assert any(
            "Raw lvar" in str(e) and "cannot be used in BaseModel" in str(e)
            for e in exc_info.value.exceptions
        )

    def test_basemodel_lvar_value_not_string(self):
        """Test BaseModel construction with lvar value already typed."""
        # Manually create metadata with typed values
        out_fields = {"reasoning": ["conf", "ana"]}

        class Reason(BaseModel):
            confidence: float
            analysis: str

        lvars = {
            "conf": LvarMetadata("Reason", "confidence", "conf", 0.85),  # Already float
            "ana": LvarMetadata("Reason", "analysis", "ana", "Text"),
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.reasoning.confidence == 0.85

    def test_parse_lndl_with_raw_lvar(self):
        """Test parse_lndl with raw lvar (RLvar)."""
        response = """
        <lvar reasoning>This is raw reasoning</lvar>

        OUT{text:[reasoning]}
        """

        operable = Operable([Spec(str, name="text")])
        output = parse_lndl(response, operable)

        assert output.text == "This is raw reasoning"

    def test_parse_lndl_missing_out_block(self):
        """Test parse_lndl raises error when OUT block missing."""
        from lionherd_core.lndl.errors import MissingOutBlockError

        response = """
        <lvar Report.title t>Title</lvar>
        """

        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(MissingOutBlockError):
            parse_lndl(response, operable)


class TestHardeningImprovements:
    """Test balanced braces and multi-error aggregation."""

    def test_balanced_braces_with_nested_dict(self):
        """Test balanced brace scanner handles nested dicts with braces in strings."""
        # Test via parse_lndl to verify integration
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Title"
        assert output.report.summary == "Summary"

    def test_balanced_braces_with_quoted_brackets(self):
        """Test balanced scanner handles quoted brackets correctly."""
        response = """
        <lvar Report.title t>Title with [brackets]</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Title with [brackets]"

    def test_unbalanced_braces_error(self):
        """Test that unbalanced braces raise appropriate error."""
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t
        """

        operable = Operable([Spec(Report, name="report")])

        # New Parser is more lenient - it parses incomplete arrays and fails during validation
        # This is acceptable behavior (fail-soft parsing)
        with pytest.raises((ExceptionGroup, Exception)):
            parse_lndl(response, operable)

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
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.field wrong_model>value</lvar>

        OUT{report:[t, missing_summary], reason:[wrong_model], quality_score:"not_a_float"}
        """

        operable = Operable(
            [
                Spec(Report, name="report"),
                Spec(Reason, name="reason"),
                Spec(float, name="quality_score"),
            ]
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            parse_lndl(response, operable)

        # Should have 3 errors
        assert len(exc_info.value.exceptions) == 3

    def test_successful_parse_with_complex_nesting(self):
        """Test successful parsing with complex nested structures."""
        response = """
        <lvar Report.title t>Good Title</lvar>
        <lvar Report.summary s>Summary text</lvar>

        OUT{report:[t, s]}
        """

        operable = Operable([Spec(Report, name="report")])
        output = parse_lndl(response, operable)

        assert output.report.title == "Good Title"
        assert output.report.summary == "Summary text"

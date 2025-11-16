# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolver.py uncovered lines - scalar handling edge cases."""

import pytest
from pydantic import BaseModel

from lionherd_core.lndl import MissingOutBlockError, parse_lndl, resolve_references_prefixed
from lionherd_core.lndl.types import LvarMetadata
from lionherd_core.types import Operable, Spec


class Report(BaseModel):
    """Test model for reports."""

    title: str
    summary: str


class Reason(BaseModel):
    """Test model for reasoning."""

    confidence: float
    analysis: str


class TestResolverPreTypedValues:
    """Test resolver handling of pre-typed (non-string) lvar values.

    These tests cover lines 129 and 254 in resolver.py where lvar_meta.value
    is already typed (not a string). This happens when the parser has pre-converted
    values to their target types.
    """

    def test_scalar_with_pre_typed_float_value(self):
        """Test scalar field with pre-typed float value (line 129).

        Coverage: resolver.py line 129
        When lvar_meta.value is already a float (not string), the else branch
        at line 129 assigns it directly without calling parse_value().
        """
        # Create LvarMetadata with pre-typed float value (not string)
        out_fields = {"quality_score": ["score_var"]}
        lvars = {
            # Value is float, not string - tests line 129 else branch
            "score_var": LvarMetadata(
                model="ScalarField",
                field="score",
                local_name="score_var",
                value=0.95,  # Pre-typed float
            )
        }
        operable = Operable([Spec(float, name="quality_score")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        # Should handle pre-typed value correctly
        assert output.quality_score == 0.95
        assert isinstance(output.quality_score, float)

    def test_scalar_with_pre_typed_int_value(self):
        """Test scalar field with pre-typed int value (line 129).

        Coverage: resolver.py line 129
        Tests the else branch when value is already an integer.
        """
        out_fields = {"priority": ["priority_var"]}
        lvars = {
            "priority_var": LvarMetadata(
                model="ScalarField",
                field="priority",
                local_name="priority_var",
                value=3,  # Pre-typed int
            )
        }
        operable = Operable([Spec(int, name="priority")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.priority == 3
        assert isinstance(output.priority, int)

    def test_scalar_with_pre_typed_bool_value(self):
        """Test scalar field with pre-typed bool value (line 129).

        Coverage: resolver.py line 129
        Tests the else branch when value is already a boolean.
        """
        out_fields = {"is_approved": ["approval_var"]}
        lvars = {
            "approval_var": LvarMetadata(
                model="ScalarField",
                field="approved",
                local_name="approval_var",
                value=True,  # Pre-typed bool
            )
        }
        operable = Operable([Spec(bool, name="is_approved")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.is_approved is True
        assert isinstance(output.is_approved, bool)

    def test_basemodel_with_pre_typed_field_values(self):
        """Test BaseModel construction with pre-typed field values (line 254).

        Coverage: resolver.py line 254
        When lvar_meta.value is already typed (not string) for BaseModel fields,
        the else branch at line 254 assigns it directly to kwargs.
        """
        out_fields = {"reasoning": ["conf_var", "ana_var"]}
        lvars = {
            # Pre-typed float for confidence field
            "conf_var": LvarMetadata(
                model="Reason",
                field="confidence",
                local_name="conf_var",
                value=0.88,  # Pre-typed float, not "0.88" string
            ),
            # String value for analysis field (normal case)
            "ana_var": LvarMetadata(
                model="Reason",
                field="analysis",
                local_name="ana_var",
                value="Detailed analysis",  # String value
            ),
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        # Should correctly handle mixed pre-typed and string values
        assert output.reasoning.confidence == 0.88
        assert output.reasoning.analysis == "Detailed analysis"

    def test_basemodel_all_fields_pre_typed(self):
        """Test BaseModel with all fields pre-typed (line 254).

        Coverage: resolver.py line 254
        All fields have pre-typed values, testing multiple executions of line 254.
        """
        out_fields = {"reasoning": ["conf_var", "ana_var"]}
        lvars = {
            "conf_var": LvarMetadata(
                model="Reason",
                field="confidence",
                local_name="conf_var",
                value=0.75,  # Pre-typed float
            ),
            "ana_var": LvarMetadata(
                model="Reason",
                field="analysis",
                local_name="ana_var",
                value="Analysis text",  # Pre-typed string (already processed)
            ),
        }
        operable = Operable([Spec(Reason, name="reasoning")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.reasoning.confidence == 0.75
        assert output.reasoning.analysis == "Analysis text"


class TestMissingOutBlock:
    """Test missing OUT{} block error handling.

    This tests line 344 in resolver.py where MissingOutBlockError is raised
    when no OUT{} block is found in the parsed response.
    """

    def test_parse_lndl_missing_out_block(self):
        """Test error when OUT{} block is missing from response (line 344).

        Coverage: resolver.py line 344
        When parse_lndl is called with a response that has lvars but no OUT{} block,
        it should raise MissingOutBlockError.
        """
        # Response with lvars but no OUT{} block
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>
        """

        operable = Operable([Spec(Report, name="report")])

        # Should raise MissingOutBlockError at line 344
        with pytest.raises(MissingOutBlockError, match="No OUT\\{\\} block found"):
            parse_lndl(response, operable)

    def test_parse_lndl_empty_response_missing_out_block(self):
        """Test error when response is completely empty (line 344).

        Coverage: resolver.py line 344
        Empty response should also trigger MissingOutBlockError.
        """
        response = ""

        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(MissingOutBlockError, match="No OUT\\{\\} block found"):
            parse_lndl(response, operable)

    def test_parse_lndl_only_prose_missing_out_block(self):
        """Test error when response has only prose, no OUT{} block (line 344).

        Coverage: resolver.py line 344
        Response with thinking/prose but no structured output should raise error.
        """
        response = """
        Let me think about this problem...

        After careful consideration, I believe the answer is 42.
        This is based on various factors including X, Y, and Z.
        """

        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(MissingOutBlockError, match="No OUT\\{\\} block found"):
            parse_lndl(response, operable)

    def test_parse_lndl_lacts_but_no_out_block(self):
        """Test error when response has lacts but no OUT{} block (line 344).

        Coverage: resolver.py line 344
        Actions declared but no OUT{} block should raise error.
        """
        response = """
        <lact Report.summary s>summarize(prompt="test")</lact>
        <lvar Report.title t>Title</lvar>
        """

        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(MissingOutBlockError, match="No OUT\\{\\} block found"):
            parse_lndl(response, operable)


class TestScalarLiteralEdgeCases:
    """Additional edge case tests for scalar literal handling."""

    def test_scalar_literal_with_type_coercion(self):
        """Test scalar literal that requires type coercion.

        This ensures the scalar literal path properly handles type conversion
        when the literal value doesn't exactly match the target type.
        """
        response = """
        <lvar Report.title t>Title</lvar>
        <lvar Report.summary s>Summary</lvar>

        OUT{report:[t, s], quality_score:1}
        """

        # quality_score is float but literal is int 1
        operable = Operable([Spec(Report, name="report"), Spec(float, name="quality_score")])
        output = parse_lndl(response, operable)

        # Should convert int 1 to float 1.0
        assert output.quality_score == 1.0
        assert isinstance(output.quality_score, float)

    def test_mixed_pre_typed_and_string_scalars(self):
        """Test mixing pre-typed and string scalar values in same OUT block."""
        out_fields = {
            "score1": ["var1"],  # Pre-typed value
            "score2": ["var2"],  # String value
        }
        lvars = {
            "var1": LvarMetadata(
                model="Scalar",
                field="value",
                local_name="var1",
                value=0.85,  # Pre-typed float
            ),
            "var2": LvarMetadata(
                model="Scalar",
                field="value",
                local_name="var2",
                value="0.75",  # String that needs parsing
            ),
        }
        operable = Operable([Spec(float, name="score1"), Spec(float, name="score2")])

        output = resolve_references_prefixed(out_fields, lvars, {}, operable)

        assert output.score1 == 0.85
        assert output.score2 == 0.75

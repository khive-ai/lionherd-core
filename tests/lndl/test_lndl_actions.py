# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for LNDL action syntax (lact tags) and resolution."""

import pytest
from pydantic import BaseModel

from lionherd_core.lndl import (
    ActionCall,
    extract_lacts,
    parse_lndl,
    resolve_references_prefixed,
)
from lionherd_core.types import Operable, Spec


class SearchResults(BaseModel):
    """Test model for search results."""

    items: list[str]
    count: int


class Report(BaseModel):
    """Test model for reports."""

    title: str
    summary: str


class TestActionResolution:
    """Test action resolution in OUT{} blocks."""

    def test_scalar_action_resolution(self):
        """Test resolving action for scalar field."""
        from lionherd_core.lndl.types import LvarMetadata

        out_fields = {
            "report": ["title", "summary"],
            "quality_score": ["calculate"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
            "summary": LvarMetadata("Report", "summary", "summary", "Summary"),
        }
        lacts = {
            "calculate": 'compute_score(data="test", threshold=0.8)',
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
        """Test error when action referenced in BaseModel field."""
        from lionherd_core.lndl.types import LvarMetadata

        out_fields = {
            "report": ["title", "search_action"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
        }
        lacts = {
            "search_action": 'search(query="test", limit=10)',
        }
        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Should raise ValueError about actions in BaseModel fields
        assert len(exc_info.value.exceptions) == 1
        assert "Actions in BaseModel fields are not yet supported" in str(
            exc_info.value.exceptions[0]
        )

    def test_name_collision_lvar_lact_error(self):
        """Test error when same name used for lvar and lact."""
        from lionherd_core.lndl.types import LvarMetadata

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
        from lionherd_core.lndl.types import LvarMetadata

        out_fields = {
            "report": ["title", "summary"],
            "quality_score": ["final"],
        }
        lvars = {
            "title": LvarMetadata("Report", "title", "title", "Title"),
            "summary": LvarMetadata("Report", "summary", "summary", "Summary"),
        }
        lacts = {
            "draft1": 'compute(version=1)',
            "draft2": 'compute(version=2)',
            "final": 'compute(version=3)',
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
            output = parse_lndl(response, operable)

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
        """Test name collision between lvar and lact triggers error."""
        response = """
        <lvar Report.title data>Title</lvar>
        <lact data>search(query="test")</lact>

        OUT{report:[data]}
        """

        operable = Operable([Spec(Report, name="report")])

        with pytest.raises(ValueError, match="Name collision"):
            parse_lndl(response, operable)

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for LNDL action syntax (lact tags) and resolution."""

import pytest
from pydantic import BaseModel

from lionherd_core.lndl import (
    ActionCall,
    LactMetadata,
    Lexer,
    Parser,
    parse_lndl,
    resolve_references_prefixed,
)
from lionherd_core.lndl.parser import PYTHON_RESERVED
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
        from lionherd_core.lndl.types import LactMetadata, LvarMetadata

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
        from lionherd_core.lndl.types import LactMetadata, LvarMetadata

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
        from lionherd_core.lndl.types import LactMetadata, LvarMetadata

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
        """Test name collision between lvar and lact triggers error."""
        from lionherd_core.lndl import ParseError

        response = """
        <lvar Report.title data>Title</lvar>
        <lact data>search(query="test")</lact>

        OUT{report:[data]}
        """

        operable = Operable([Spec(Report, name="report")])

        # New Parser detects duplicate aliases during parsing (fail fast)
        with pytest.raises(ParseError, match="Duplicate alias"):
            parse_lndl(response, operable)


class TestNamespacedActions:
    """Test namespaced action pattern for mixing lvars and actions."""

    def _extract_lacts_from_response(self, response: str) -> dict[str, LactMetadata]:
        """Helper to extract lacts using new Parser API."""
        lexer = Lexer(response)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=response)
        program = parser.parse()

        # Convert AST lacts to LactMetadata dict
        lacts: dict[str, LactMetadata] = {}
        for lact in program.lacts:
            lacts[lact.alias] = LactMetadata(
                model=lact.model,
                field=lact.field,
                local_name=lact.alias,
                call=lact.call,
            )
        return lacts

    def test_extract_namespaced_actions(self):
        """Test extracting namespaced actions with Model.field syntax."""
        response = """
        <lact Report.title t>generate_title(topic="AI")</lact>
        <lact Report.summary summarize>generate_summary(data="metrics")</lact>
        <lact search>search(query="test")</lact>
        """

        lacts = self._extract_lacts_from_response(response)

        # Namespaced actions
        assert "t" in lacts
        assert lacts["t"].model == "Report"
        assert lacts["t"].field == "title"
        assert lacts["t"].local_name == "t"
        assert lacts["t"].call == 'generate_title(topic="AI")'

        assert "summarize" in lacts
        assert lacts["summarize"].model == "Report"
        assert lacts["summarize"].field == "summary"
        assert lacts["summarize"].local_name == "summarize"

        # Direct action
        assert "search" in lacts
        assert lacts["search"].model is None
        assert lacts["search"].field is None
        assert lacts["search"].local_name == "search"

    def test_extract_namespaced_without_alias(self):
        """Test namespaced action defaults to field name when no alias provided."""
        response = """
        <lact Report.summary>generate_summary(data="test")</lact>
        """

        lacts = self._extract_lacts_from_response(response)

        assert "summary" in lacts
        assert lacts["summary"].model == "Report"
        assert lacts["summary"].field == "summary"
        assert lacts["summary"].local_name == "summary"

    def test_mixing_lvars_and_namespaced_actions(self):
        """Test mixing lvars and namespaced actions in same BaseModel."""
        from lionherd_core.lndl.types import LactMetadata, LvarMetadata

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
        from lionherd_core.lndl.types import LactMetadata, LvarMetadata

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
        from lionherd_core.lndl.types import LactMetadata, LvarMetadata

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
        from lionherd_core.lndl.types import LactMetadata

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
        """Test unclosed lact tag (parser detects during parsing)."""
        from lionherd_core.lndl import ParseError

        response = '<lact action>search(query="test")\nOUT{result:[action]}'
        operable = Operable([Spec(SearchResults, name="result")])

        # New Parser detects unclosed tags during parsing
        with pytest.raises(ParseError, match="Unclosed lact tag"):
            parse_lndl(response, operable)

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
        import warnings

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

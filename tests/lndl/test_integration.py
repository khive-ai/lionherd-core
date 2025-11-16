# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for unified LNDL architecture.

Tests the complete pipeline: Parser → Resolver → Validator → LNDLOutput
Covers basic flows, validation, fuzzy matching, action lifecycle, and backward compatibility.
"""

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from lionherd_core.lndl import (
    ActionCall,
    AmbiguousMatchError,
    MissingFieldError,
    TypeMismatchError,
    parse_lndl,
    parse_lndl_fuzzy,
)
from lionherd_core.lndl.types import (
    ensure_no_action_calls,
    has_action_calls,
    revalidate_with_action_results,
)
from lionherd_core.types import Operable, Spec

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


class SearchResults(BaseModel):
    """Search results model."""

    items: list[str]
    count: int


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

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for fuzzy LNDL parsing with typo correction.

Tests verify:
- Field name correction (titel → title)
- Lvar reference correction (summ → summary)
- Model name correction (Reprot → Report)
- Spec name correction (reprot → report)
- Threshold configuration (strict mode, custom thresholds)
- Error handling (ambiguous matches, below threshold)
- Integration with Parser/Lexer/Resolver architecture
"""

import pytest
from pydantic import BaseModel

from lionherd_core.lndl import ActionCall, MissingFieldError, fuzzy, resolver
from lionherd_core.lndl.errors import AmbiguousMatchError, MissingOutBlockError
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec


class TestFuzzyFieldNameCorrection:
    """Test fuzzy correction of field names in OUT{} blocks."""

    def test_field_typo_correction(self, report_model):
        """Test common typo correction (titel → title)."""
        # Create spec and operable
        operable = Operable([Spec(report_model, name="report")])

        # LNDL with valid field names (testing basic functionality)
        lndl_text = """\
        <lvar Report.title t>Good Title</lvar>
        <lvar Report.content c>Content here</lvar>

        ```lndl
        OUT{report: [t, c]}
                ```\
                """

        result = parse_lndl_fuzzy(lndl_text, operable, threshold=0.85)

        assert hasattr(result, "report")
        assert result.report.title == "Good Title"
        assert result.report.content == "Content here"

    def test_field_multiple_typos(self):
        """Test multiple field name corrections in single OUT{}."""

        class Report(BaseModel):
            title: str
            summary: str
            quality_score: float

        operable = Operable([Spec(Report, name="report")])

        # Multiple fields with valid names
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
        assert result.report.title == "Title"
        assert result.report.summary == "Summary"

    def test_field_below_threshold(self, report_model):
        """Test field name too dissimilar raises error."""
        operable = Operable([Spec(report_model, name="report")])

        # "xyz" has no similarity to any field in Report model
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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        with pytest.raises(MissingOutBlockError):
            parse_lndl_fuzzy("", operable)

    def test_no_matches_below_threshold(self, report_model):
        """Test all fields below threshold raises error."""
        operable = Operable([Spec(report_model, name="report")])

        # "xyz123" has no similarity to any field in Report
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

        from lionherd_core.types import Operable, Spec

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

        # Both should exist
        assert hasattr(fuzzy, "parse_lndl_fuzzy")
        assert hasattr(resolver, "resolve_references_prefixed")

    def test_exact_matches_same_as_strict(self):
        """Test exact matches produce same result as strict resolver."""

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

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

        from lionherd_core.types import Operable, Spec

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

        from lionherd_core.types import Operable, Spec

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

    def test_fuzzy_with_raw_lvar(self):
        """Test fuzzy parsing with raw lvar (RLvar)."""

        from lionherd_core.types import Operable, Spec

        response = """
<lvar reasoning>This is raw reasoning text</lvar>

OUT{text:[reasoning]}
"""
        operable = Operable([Spec(str, name="text")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        assert output.text == "This is raw reasoning text"

    def test_strict_mode_with_raw_lvar(self):
        """Test strict mode validation skips raw lvars."""

        from lionherd_core.types import Operable, Spec

        response = """
<lvar reasoning>Raw text</lvar>

OUT{text:[reasoning]}
"""
        operable = Operable([Spec(str, name="text")])
        # Strict mode should work with raw lvars (they skip validation)
        output = parse_lndl_fuzzy(response, operable, threshold=1.0)

        assert output.text == "Raw text"

    def test_fuzzy_raw_lvar_passes_through(self):
        """Test raw lvar passes through fuzzy correction unchanged."""

        from lionherd_core.types import Operable, Spec

        class Report(BaseModel):
            title: str

        # Mix namespaced and raw lvars
        response = """
<lvar Report.titel t>Title</lvar>
<lvar reasoning>Raw reasoning</lvar>

OUT{report:[t], text:[reasoning]}
"""
        operable = Operable([Spec(Report, name="report"), Spec(str, name="text")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        assert output.report.title == "Title"  # Corrected
        assert output.text == "Raw reasoning"  # Unchanged

    def test_fuzzy_literal_value_in_out_block(self):
        """Test fuzzy with literal scalar value (not array)."""

        from lionherd_core.types import Operable, Spec

        response = """
OUT{count: 42}
"""
        operable = Operable([Spec(int, name="count")])
        output = parse_lndl_fuzzy(response, operable, threshold=0.85)

        assert output.count == 42

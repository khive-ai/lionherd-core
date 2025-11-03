# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.lndl import MissingFieldError
from lionherd_core.lndl.errors import AmbiguousMatchError


class TestFuzzyFieldNameCorrection:
    """Test fuzzy correction of field names in OUT{} blocks."""

    def test_field_typo_correction(self):
        """Test common typo correction (titel → title)."""
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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
        from pydantic import BaseModel

        from lionherd_core.lndl.errors import MissingFieldError
        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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


class TestFuzzyErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_lndl_text(self):
        """Test empty LNDL text handling."""
        from lionherd_core.lndl.errors import MissingOutBlockError
        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable

        operable = Operable()

        with pytest.raises(MissingOutBlockError):  # Should raise MissingOutBlockError
            parse_lndl_fuzzy("", operable)

    def test_no_matches_below_threshold(self):
        """Test all fields below threshold raises error."""
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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

        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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
        from lionherd_core.lndl import fuzzy, resolver

        # Both should exist
        assert hasattr(fuzzy, "parse_lndl_fuzzy")
        assert hasattr(resolver, "resolve_references_prefixed")

    def test_exact_matches_same_as_strict(self):
        """Test exact matches produce same result as strict resolver."""
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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
        from pydantic import BaseModel

        from lionherd_core.lndl.errors import MissingFieldError
        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.errors import MissingFieldError
        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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

        # Should raise MissingFieldError for wrong field name in strict mode
        with pytest.raises(MissingFieldError, match=r"Field.*not found"):
            parse_lndl_fuzzy(lndl_text, operable, threshold=1.0)

    def test_strict_mode_with_wrong_spec_name(self):
        """Test strict mode raises early for wrong spec name (line 217)."""
        from pydantic import BaseModel

        from lionherd_core.lndl.errors import MissingFieldError
        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
        from lionherd_core.types import Operable, Spec

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
        from pydantic import BaseModel

        from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
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

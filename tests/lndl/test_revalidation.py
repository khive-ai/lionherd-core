# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for action re-validation lifecycle (types.py)."""

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator

from lionherd_core.lndl import ActionCall, has_action_calls, revalidate_with_action_results

# ============================================================================
# Test Model Fixtures
# ============================================================================


@pytest.fixture
def simple_report_model():
    """Simple report model for action revalidation testing."""

    class Report(BaseModel):
        """Test model for reports."""

        title: str
        summary: str

    return Report


@pytest.fixture
def strict_report_model():
    """Report with field constraints for validation testing."""

    class StrictReport(BaseModel):
        """Report with field constraints for validation testing."""

        title: str
        summary: str = Field(min_length=10)

    return StrictReport


@pytest.fixture
def validated_report_model():
    """Report with custom field validators."""

    class ValidatedReport(BaseModel):
        """Report with custom field validators."""

        title: str
        summary: str

        @field_validator("summary")
        @classmethod
        def summary_must_contain_word(cls, v: str) -> str:
            if "important" not in v.lower():
                raise ValueError("summary must contain 'important'")
            return v

    return ValidatedReport


class TestHasActionCalls:
    """Test has_action_calls() detection function."""

    def test_returns_true_with_single_action_call(self, simple_report_model):
        """Returns True when model has one ActionCall field."""
        report = simple_report_model.model_construct(
            title="Report",
            summary=ActionCall(
                "summarize", "generate_summary", {"length": 100}, "generate_summary(length=100)"
            ),
        )
        assert has_action_calls(report) is True

    def test_returns_true_with_multiple_action_calls(self, simple_report_model):
        """Returns True when model has multiple ActionCall fields."""
        report = simple_report_model.model_construct(
            title=ActionCall("gen_title", "generate_title", {}, "generate_title()"),
            summary=ActionCall("gen_summary", "generate_summary", {}, "generate_summary()"),
        )
        assert has_action_calls(report) is True

    def test_returns_false_with_no_action_calls(self, simple_report_model):
        """Returns False when model has no ActionCall fields."""
        report = simple_report_model(title="Report", summary="Summary text")
        assert has_action_calls(report) is False

    def test_returns_true_with_mixed_fields(self, simple_report_model):
        """Returns True when some fields are ActionCall and others are normal."""
        report = simple_report_model.model_construct(
            title="Normal title",
            summary=ActionCall("summarize", "generate", {}, "generate()"),
        )
        assert has_action_calls(report) is True


class TestRevalidateWithActionResults:
    """Test revalidate_with_action_results() function."""

    def test_success_single_action(self, simple_report_model):
        """Successfully re-validates model with single action result."""
        # Construct with ActionCall placeholder
        report = simple_report_model.model_construct(
            title="Report",
            summary=ActionCall("summarize", "generate_summary", {}, "generate_summary()"),
        )

        # Execute actions (simulated)
        action_results = {"summarize": "This is the generated summary text"}

        # Re-validate
        validated = revalidate_with_action_results(report, action_results)

        # Verify
        assert isinstance(validated, simple_report_model)
        assert validated.title == "Report"
        assert validated.summary == "This is the generated summary text"
        assert not has_action_calls(validated)

    def test_success_multiple_actions(self, simple_report_model):
        """Successfully re-validates model with multiple action results."""
        report = simple_report_model.model_construct(
            title=ActionCall("gen_title", "generate_title", {}, "generate_title()"),
            summary=ActionCall("gen_summary", "generate_summary", {}, "generate_summary()"),
        )

        action_results = {
            "gen_title": "Generated Title",
            "gen_summary": "Generated Summary",
        }

        validated = revalidate_with_action_results(report, action_results)

        assert validated.title == "Generated Title"
        assert validated.summary == "Generated Summary"
        assert not has_action_calls(validated)

    def test_error_missing_action_result(self, simple_report_model):
        """Raises ValueError when action result missing from results dict."""
        report = simple_report_model.model_construct(
            title="Report",
            summary=ActionCall("summarize", "generate_summary", {}, "generate_summary()"),
        )

        # Missing "summarize" in results
        action_results = {"other_action": "value"}

        with pytest.raises(ValueError) as exc_info:
            revalidate_with_action_results(report, action_results)

        assert "Action 'summarize'" in str(exc_info.value)
        assert "has no execution result" in str(exc_info.value)
        assert "'other_action'" in str(exc_info.value)  # Shows available results

    def test_error_missing_multiple_action_results(self, simple_report_model):
        """Raises ValueError on first missing action result."""
        report = simple_report_model.model_construct(
            title=ActionCall("gen_title", "generate_title", {}, "generate_title()"),
            summary=ActionCall("gen_summary", "generate_summary", {}, "generate_summary()"),
        )

        # Missing both actions
        action_results = {}

        with pytest.raises(ValueError, match="has no execution result"):
            revalidate_with_action_results(report, action_results)

    def test_enforces_pydantic_field_constraints(self, strict_report_model):
        """Re-validation enforces Pydantic field constraints."""
        report = strict_report_model.model_construct(
            title="Report",
            summary=ActionCall("summarize", "generate", {}, "generate()"),
        )

        # Result violates min_length=10 constraint
        action_results = {"summarize": "short"}

        with pytest.raises(ValidationError) as exc_info:
            revalidate_with_action_results(report, action_results)

        # Verify it's the summary field with constraint violation
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("summary",) for e in errors)

    def test_enforces_field_validators(self, validated_report_model):
        """Re-validation runs custom field validators."""
        report = validated_report_model.model_construct(
            title="Report",
            summary=ActionCall("summarize", "generate", {}, "generate()"),
        )

        # Result doesn't pass validator
        action_results = {"summarize": "This is a regular summary"}

        with pytest.raises(ValidationError, match="must contain 'important'"):
            revalidate_with_action_results(report, action_results)

    def test_preserves_type_after_revalidation(self, simple_report_model):
        """Re-validated model is same type as original."""
        report = simple_report_model.model_construct(
            title="Report",
            summary=ActionCall("summarize", "generate", {}, "generate()"),
        )

        action_results = {"summarize": "Summary"}
        validated = revalidate_with_action_results(report, action_results)

        assert type(validated) is simple_report_model
        assert isinstance(validated, simple_report_model)

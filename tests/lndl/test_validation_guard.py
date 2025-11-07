"""Test LNDL validation guard - Issue #23.

The footgun: Users forget to call revalidate_with_action_results() after parsing,
leading to ActionCall objects being persisted/used instead of real values.

This test demonstrates the footgun and verifies the guard prevents it.
"""

import pytest
from pydantic import BaseModel, Field

from lionherd_core.lndl.types import ActionCall, ensure_no_action_calls, has_action_calls


class Report(BaseModel):
    """Test model for LNDL validation."""

    title: str
    summary: str = Field(..., min_length=10, max_length=500)
    score: int = Field(..., ge=0, le=100)


def test_has_action_calls_detects_placeholders():
    """Verify has_action_calls() correctly detects ActionCall objects."""
    # Fully validated model - no ActionCalls
    valid_report = Report(title="Test", summary="A valid summary string", score=85)
    assert has_action_calls(valid_report) is False

    # Model with ActionCall placeholder (bypassed validation)
    action_call = ActionCall(
        name="summarize",
        function="generate_summary",
        arguments={"text": "input"},
        raw_call="generate_summary(text='input')",
    )

    partial_report = Report.model_construct(
        title="Test",
        summary=action_call,  # ActionCall instead of string!
        score=85,
    )

    assert has_action_calls(partial_report) is True


def test_ensure_no_action_calls_passes_valid_model():
    """ensure_no_action_calls() should pass through fully validated models."""
    valid_report = Report(title="Test", summary="This is a proper summary", score=95)

    # Should return the same model without error
    result = ensure_no_action_calls(valid_report)
    assert result is valid_report
    assert isinstance(result.summary, str)


def test_ensure_no_action_calls_raises_on_placeholders():
    """ensure_no_action_calls() should raise ValueError if ActionCall present.

    This is the CRITICAL guard that prevents the footgun.
    """
    action_call = ActionCall(
        name="summarize",
        function="generate_summary",
        arguments={},
        raw_call="generate_summary()",
    )

    partial_report = Report.model_construct(
        title="Test",
        summary=action_call,  # BUG: ActionCall placeholder
        score=85,
    )

    # Should raise clear error
    with pytest.raises(ValueError, match="contains unexecuted actions"):
        ensure_no_action_calls(partial_report)


def test_footgun_scenario_database_save():
    """Demonstrate the EXACT footgun: User forgets revalidation before DB save.

    WITHOUT guard: Silent corruption (ActionCall saved to DB)
    WITH guard: Clear error prevents corruption
    """
    action_call = ActionCall(
        name="s",
        function="summarize",
        arguments={"text": "long text"},
        raw_call="summarize(text='long text')",
    )

    # User gets output from parse_lndl_fuzzy
    report = Report.model_construct(
        title="Analysis Report",
        summary=action_call,  # Forgot to execute actions!
        score=75,
    )

    # User tries to save to database
    # WITHOUT guard: report.summary is ActionCall object - DB corruption!
    # WITH guard: Raises error before corruption
    with pytest.raises(ValueError, match="contains unexecuted actions"):
        db_save(ensure_no_action_calls(report))


def test_guard_provides_helpful_error_message():
    """Error message should guide user to revalidate_with_action_results()."""
    action_call = ActionCall(
        name="summarize",
        function="generate_summary",
        arguments={},
        raw_call="generate_summary()",
    )

    partial = Report.model_construct(title="Test", summary=action_call, score=50)

    with pytest.raises(ValueError) as exc_info:
        ensure_no_action_calls(partial)

    error_msg = str(exc_info.value)
    assert "unexecuted actions" in error_msg
    assert "revalidate_with_action_results" in error_msg
    assert "Report" in error_msg  # Shows which model


def test_nested_models_with_action_calls():
    """Guard should detect ActionCalls in nested models too."""

    class NestedReport(BaseModel):
        main: Report
        appendix: str

    action_call = ActionCall(
        name="s", function="summarize", arguments={}, raw_call="summarize()"
    )

    nested = NestedReport.model_construct(
        main=Report.model_construct(title="Test", summary=action_call, score=80),
        appendix="Appendix text",
    )

    # Should detect ActionCall in nested model
    assert has_action_calls(nested.main) is True

    with pytest.raises(ValueError, match="contains unexecuted actions"):
        ensure_no_action_calls(nested.main)


# Mock database save function
def db_save(model: BaseModel):
    """Mock database save - would serialize model to JSON."""
    # In real code, this would do:
    # db.insert(model.model_dump_json())
    # If model has ActionCall, JSON would be:
    # {"summary": "ActionCall(name='summarize', ...)"} ← BUG!
    pass

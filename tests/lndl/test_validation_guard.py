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
    """Guard should detect ActionCalls in nested models recursively.

    CRITICAL: This tests recursive detection - ActionCall is in nested.main.summary,
    and the guard must detect it when called on the outer 'nested' object.
    """

    class NestedReport(BaseModel):
        main: Report
        appendix: str

    action_call = ActionCall(name="s", function="summarize", arguments={}, raw_call="summarize()")

    nested = NestedReport.model_construct(
        main=Report.model_construct(title="Test", summary=action_call, score=80),
        appendix="Appendix text",
    )

    # Should detect ActionCall in nested model (inner check)
    assert has_action_calls(nested.main) is True

    # CRITICAL: Should also detect when checking outer model (recursive detection)
    assert has_action_calls(nested) is True, "Recursive detection must work on outer model"

    # Guard should raise when checking inner model directly
    with pytest.raises(ValueError, match="contains unexecuted actions"):
        ensure_no_action_calls(nested.main)

    # CRITICAL: Guard should also raise when checking outer model (prevents bypass)
    with pytest.raises(ValueError, match="contains unexecuted actions"):
        ensure_no_action_calls(nested)

    # Error message should show field path
    with pytest.raises(ValueError, match="main\\.summary"):
        ensure_no_action_calls(nested)


def test_collection_fields_with_action_calls():
    """Guard should detect ActionCalls in collection fields (list, dict)."""

    class BatchReport(BaseModel):
        reports: list[Report]
        metadata: dict[str, Report]

    action_call = ActionCall(name="s", function="summarize", arguments={}, raw_call="summarize()")

    # ActionCall in list
    batch_with_list = BatchReport.model_construct(
        reports=[
            Report.model_construct(title="Report 1", summary=action_call, score=85),
            Report(title="Report 2", summary="Valid summary", score=90),
        ],
        metadata={},
    )

    assert has_action_calls(batch_with_list) is True
    with pytest.raises(ValueError, match="reports\\[0\\]\\.summary"):
        ensure_no_action_calls(batch_with_list)

    # ActionCall in dict
    batch_with_dict = BatchReport.model_construct(
        reports=[],
        metadata={"key1": Report.model_construct(title="Report", summary=action_call, score=75)},
    )

    assert has_action_calls(batch_with_dict) is True
    with pytest.raises(ValueError, match="metadata\\['key1'\\]\\.summary"):
        ensure_no_action_calls(batch_with_dict)


def test_multiple_action_calls_in_model():
    """Test error message with multiple ActionCall fields (>3 fields)."""

    class MultiFieldReport(BaseModel):
        field1: str
        field2: str
        field3: str
        field4: str
        field5: str

    action_call = ActionCall(name="a", function="action", arguments={}, raw_call="action()")

    # Model with 4 ActionCalls (more than 3)
    multi = MultiFieldReport.model_construct(
        field1=action_call,
        field2=action_call,
        field3=action_call,
        field4=action_call,
        field5="valid",
    )

    assert has_action_calls(multi) is True

    # Error message should show first 3 fields + count of remaining
    with pytest.raises(ValueError, match=r"\(and 1 more\)"):
        ensure_no_action_calls(multi)


def test_direct_action_call_in_collections():
    """Test ActionCall directly in collections (not nested in models)."""

    class DirectCollectionReport(BaseModel):
        items: list

    action_call = ActionCall(name="a", function="action", arguments={}, raw_call="action()")

    # ActionCall directly in list (not inside a Report)
    direct_list = DirectCollectionReport.model_construct(items=[action_call, "string"])

    assert has_action_calls(direct_list) is True
    with pytest.raises(ValueError, match="items\\[0\\]"):
        ensure_no_action_calls(direct_list)


def test_tuple_collections():
    """Test ActionCall detection in tuple collections.

    Note: ActionCall is unhashable (contains dict field), so cannot be added to sets.
    This test only covers tuples as they are the realistic collection type for ActionCall.
    """

    class TupleReport(BaseModel):
        tuple_field: tuple

    action_call = ActionCall(name="a", function="action", arguments={}, raw_call="action()")

    # ActionCall in tuple
    with_tuple = TupleReport.model_construct(tuple_field=(action_call, "other"))

    assert has_action_calls(with_tuple) is True
    with pytest.raises(ValueError, match="tuple_field\\[0\\]"):
        ensure_no_action_calls(with_tuple)


def test_deeply_nested_action_calls():
    """Test ActionCall detection in deeply nested structures."""

    class Level3(BaseModel):
        data: str

    class Level2(BaseModel):
        inner: Level3

    class Level1(BaseModel):
        middle: Level2

    action_call = ActionCall(name="a", function="action", arguments={}, raw_call="action()")

    # ActionCall 3 levels deep
    deep = Level1.model_construct(
        middle=Level2.model_construct(inner=Level3.model_construct(data=action_call))
    )

    assert has_action_calls(deep) is True
    with pytest.raises(ValueError, match="middle\\.inner\\.data"):
        ensure_no_action_calls(deep)


def test_empty_collections_pass():
    """Test that empty collections don't trigger false positives."""

    class EmptyCollectionsReport(BaseModel):
        empty_list: list
        empty_dict: dict
        empty_tuple: tuple
        empty_set: set

    empty = EmptyCollectionsReport(empty_list=[], empty_dict={}, empty_tuple=(), empty_set=set())

    assert has_action_calls(empty) is False
    # Should pass without raising
    result = ensure_no_action_calls(empty)
    assert result is empty


# Mock database save function
def db_save(model: BaseModel):
    """Mock database save - would serialize model to JSON."""
    # In real code, this would do:
    # db.insert(model.model_dump_json())
    # If model has ActionCall, JSON would be:
    # {"summary": "ActionCall(name='summarize', ...)"} ← BUG!
    pass

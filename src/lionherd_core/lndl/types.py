# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core types for LNDL (Lion Directive Language)."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

# Type alias for scalar values in OUT{} block
Scalar = float | int | str | bool


@dataclass(slots=True, frozen=True)
class LvarMetadata:
    """Metadata for namespaced lvar - maps to Pydantic model field.

    Example: <lvar Report.title t>Good Title</lvar>
    → LvarMetadata(model="Report", field="title", local_name="t", value="Good Title")
    """

    model: str  # Model name (e.g., "Report")
    field: str  # Field name (e.g., "title")
    local_name: str  # Local variable name (e.g., "t")
    value: str  # Raw string value


@dataclass(slots=True, frozen=True)
class RLvarMetadata:
    """Metadata for raw lvar - simple string capture without model mapping.

    Example: <lvar reasoning>The analysis shows...</lvar>
    → RLvarMetadata(local_name="reasoning", value="The analysis shows...")

    Usage:
        - Use for intermediate LLM outputs not mapped to Pydantic models
        - Can only resolve to scalar OUT{} fields (str, int, float, bool)
        - Cannot be used in BaseModel OUT{} fields
    """

    local_name: str  # Local variable name
    value: str  # Raw string value


@dataclass(slots=True, frozen=True)
class LactMetadata:
    """Metadata for action declaration (namespaced or direct).

    Examples:
        Namespaced: <lact Report.summary s>generate_summary(...)</lact>
        → LactMetadata(model="Report", field="summary", local_name="s", call="generate_summary(...)")

        Direct: <lact search>search(...)</lact>
        → LactMetadata(model=None, field=None, local_name="search", call="search(...)")
    """

    model: str | None  # Model name (e.g., "Report") or None for direct actions
    field: str | None  # Field name (e.g., "summary") or None for direct actions
    local_name: str  # Local reference name (e.g., "s", "search")
    call: str  # Raw function call string


@dataclass(slots=True, frozen=True)
class ParsedConstructor:
    """Parsed type constructor from OUT{} block."""

    class_name: str
    kwargs: dict[str, Any]
    raw: str

    @property
    def has_dict_unpack(self) -> bool:
        """Check if constructor uses **dict unpacking."""
        return any(k.startswith("**") for k in self.kwargs)


@dataclass(slots=True, frozen=True)
class ActionCall:
    """Parsed action call from <lact> tag.

    Represents a tool/function invocation declared in LNDL response.
    Actions are only executed if referenced in OUT{} block.

    Attributes:
        name: Local reference name (e.g., "search", "validate")
        function: Function/tool name to invoke
        arguments: Parsed arguments dict
        raw_call: Original Python function call string
    """

    name: str
    function: str
    arguments: dict[str, Any]
    raw_call: str


@dataclass(slots=True, frozen=True)
class LNDLOutput:
    """Validated LNDL output with action execution lifecycle.

    Fields can contain:
    - BaseModel instances (fully validated)
    - ActionCall objects (pre-execution, partially validated)
    - Scalar values (float, int, str, bool - fully validated)

    Action Execution Lifecycle:
    ---------------------------
    1. **Parse**: LNDL response parsed, ActionCall objects created for referenced actions
    2. **Partial Validation**: BaseModels with ActionCall fields use model_construct() to bypass validation
    3. **Execute**: Caller executes actions using .actions dict, collects results
    4. **Re-validate**: Caller replaces ActionCall objects with results and re-validates models

    Fields containing ActionCall objects have **partial validation** only:
    - Field constraints (validators, bounds, regex) are NOT enforced
    - Type checking is bypassed
    - Re-validation MUST occur after action execution

    Example:
        >>> output = parse_lndl(response, operable)
        >>> # Execute actions
        >>> action_results = {}
        >>> for name, action in output.actions.items():
        >>>     result = execute_tool(action.function, action.arguments)
        >>>     action_results[name] = result
        >>>
        >>> # Re-validate models with action results
        >>> for field_name, value in output.fields.items():
        >>>     if isinstance(value, BaseModel) and has_action_calls(value):
        >>>         value = revalidate_with_action_results(value, action_results)
        >>>         output.fields[field_name] = value
    """

    fields: dict[str, BaseModel | ActionCall | Scalar]  # BaseModel, ActionCall, or scalar values
    lvars: dict[str, str] | dict[str, LvarMetadata]  # Preserved for debugging
    lacts: dict[str, LactMetadata]  # All declared actions (for debugging/reference)
    actions: dict[str, ActionCall]  # Actions referenced in OUT{} (pending execution)
    raw_out_block: str  # Preserved for debugging

    def __getitem__(self, key: str) -> BaseModel | ActionCall | Scalar:
        return self.fields[key]

    def __getattr__(self, key: str) -> BaseModel | ActionCall | Scalar:
        if key in ("fields", "lvars", "lacts", "actions", "raw_out_block"):
            return object.__getattribute__(self, key)
        return self.fields[key]


def has_action_calls(model: BaseModel) -> bool:
    """Check if a BaseModel instance contains any ActionCall objects in its fields.

    Recursively checks nested BaseModel fields and collection fields (list, dict, tuple, set)
    to detect ActionCall objects at any depth.

    Args:
        model: Pydantic BaseModel instance to check

    Returns:
        True if any field value is an ActionCall (at any nesting level), False otherwise

    Example:
        >>> report = Report.model_construct(title="Report", summary=ActionCall(...))
        >>> has_action_calls(report)
        True
        >>> # Also detects in nested models
        >>> nested = NestedReport(main=report)
        >>> has_action_calls(nested)
        True
    """

    def _check_value(value: Any) -> bool:
        """Recursively check a value for ActionCall objects."""
        # Direct ActionCall
        if isinstance(value, ActionCall):
            return True

        # Nested BaseModel - recurse
        if isinstance(value, BaseModel):
            return has_action_calls(value)

        # Collections - check items
        if isinstance(value, (list, tuple, set)):
            return any(_check_value(item) for item in value)

        if isinstance(value, dict):
            return any(_check_value(v) for v in value.values())

        return False

    return any(_check_value(value) for value in model.__dict__.values())


def ensure_no_action_calls(model: BaseModel) -> BaseModel:
    """Validate that model contains no unexecuted ActionCall objects.

    Use this guard before persisting models to prevent database corruption or logic errors.
    Models with ActionCall placeholders must be re-validated with action results first.

    Recursively checks nested models and collections for ActionCall objects.

    Args:
        model: BaseModel instance to validate

    Returns:
        The same model instance if validation passes

    Raises:
        ValueError: If model contains any ActionCall objects, with field path details

    Example:
        >>> # CRITICAL: Always guard before persistence
        >>> output = parse_lndl_fuzzy(llm_response, operable)
        >>> report = output.report
        >>>
        >>> # Execute actions first
        >>> action_results = execute_actions(output.actions)
        >>> validated_report = revalidate_with_action_results(report, action_results)
        >>>
        >>> # Safe to persist - guard will pass
        >>> db.save(ensure_no_action_calls(validated_report))
        >>>
        >>> # BAD: Forgot revalidation - guard prevents corruption
        >>> db.save(ensure_no_action_calls(report))  # Raises ValueError!
    """

    def _find_action_call_fields(obj: Any, path: str = "") -> list[str]:
        """Find all field paths containing ActionCall objects."""
        paths = []

        if isinstance(obj, ActionCall):
            return [path] if path else ["<root>"]

        if isinstance(obj, BaseModel):
            for field_name, value in obj.__dict__.items():
                field_path = f"{path}.{field_name}" if path else field_name
                paths.extend(_find_action_call_fields(value, field_path))

        elif isinstance(obj, (list, tuple, set)):
            for idx, item in enumerate(obj):
                item_path = f"{path}[{idx}]"
                paths.extend(_find_action_call_fields(item, item_path))

        elif isinstance(obj, dict):
            for key, value in obj.items():
                dict_path = f"{path}[{key!r}]"
                paths.extend(_find_action_call_fields(value, dict_path))

        return paths

    if has_action_calls(model):
        model_name = type(model).__name__
        action_call_fields = _find_action_call_fields(model)
        fields_str = ", ".join(action_call_fields[:3])  # Show first 3 fields
        if len(action_call_fields) > 3:
            fields_str += f" (and {len(action_call_fields) - 3} more)"

        raise ValueError(
            f"{model_name} contains unexecuted actions in fields: {fields_str}. "
            f"Models with ActionCall placeholders must be re-validated after action execution. "
            f"Call revalidate_with_action_results() before using this model."
        )
    return model


def revalidate_with_action_results(
    model: BaseModel,
    action_results: dict[str, Any],
) -> BaseModel:
    """Replace ActionCall fields with execution results and re-validate the model.

    This function must be called after executing actions to restore full Pydantic validation.
    Models constructed with model_construct() have bypassed validation and may contain
    ActionCall objects where actual values are expected.

    Args:
        model: BaseModel instance with ActionCall placeholders
        action_results: Dict mapping action names to their execution results

    Returns:
        Fully validated BaseModel instance with action results substituted

    Raises:
        ValidationError: If action results don't satisfy field constraints

    Example:
        >>> # Model has ActionCall in summary field
        >>> report = Report.model_construct(title="Report", summary=action_call)
        >>>
        >>> # Execute action and get result
        >>> action_results = {"summarize": "Generated summary text"}
        >>>
        >>> # Re-validate with results
        >>> validated_report = revalidate_with_action_results(report, action_results)
        >>> isinstance(validated_report.summary, str)  # True, no longer ActionCall
        True
    """
    # Get current field values
    kwargs = model.model_dump()

    # Replace ActionCall objects with their execution results
    for field_name, value in model.__dict__.items():
        if isinstance(value, ActionCall):
            # Find result by action name
            if value.name not in action_results:
                raise ValueError(
                    f"Action '{value.name}' in field '{field_name}' has no execution result. "
                    f"Available results: {list(action_results.keys())}"
                )
            kwargs[field_name] = action_results[value.name]

    # Re-construct with full validation
    return type(model).model_validate(kwargs)

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core types for LNDL (Lion Directive Language)."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(slots=True, frozen=True)
class LvarMetadata:
    """Metadata for namespace-prefixed lvar.

    Example: <lvar Report.title title>Good Title</lvar>
    → LvarMetadata(model="Report", field="title", local_name="title", value="Good Title")
    """

    model: str  # Model name (e.g., "Report")
    field: str  # Field name (e.g., "title")
    local_name: str  # Local variable name (e.g., "title")
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

    fields: dict[str, BaseModel | ActionCall]  # BaseModel instances or ActionCall (pre-execution)
    lvars: dict[str, str] | dict[str, LvarMetadata]  # Preserved for debugging
    lacts: dict[str, LactMetadata]  # All declared actions (for debugging/reference)
    actions: dict[str, ActionCall]  # Actions referenced in OUT{} (pending execution)
    raw_out_block: str  # Preserved for debugging

    def __getitem__(self, key: str) -> BaseModel | ActionCall:
        return self.fields[key]

    def __getattr__(self, key: str) -> BaseModel | ActionCall:
        if key in ("fields", "lvars", "lacts", "actions", "raw_out_block"):
            return object.__getattribute__(self, key)
        return self.fields[key]


def has_action_calls(model: BaseModel) -> bool:
    """Check if a BaseModel instance contains any ActionCall objects in its fields.

    Args:
        model: Pydantic BaseModel instance to check

    Returns:
        True if any field value is an ActionCall, False otherwise

    Example:
        >>> report = Report.model_construct(title="Report", summary=ActionCall(...))
        >>> has_action_calls(report)
        True
    """
    return any(isinstance(value, ActionCall) for value in model.__dict__.values())


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
    return type(model)(**kwargs)

# LNDL Types

> Core types for LNDL (Language InterOperable Network Directive Language) parsing and execution

## Overview

The LNDL types module provides data structures for parsing, validating, and executing LNDL responses. LNDL is a structured directive language that enables LLM responses to declare variables, actions, and type constructors with deferred execution semantics.

**Key Capabilities:**

- **Variable Metadata**: Structured metadata for namespace-prefixed variables (`<lvar>`)
- **Action Metadata**: Metadata for action declarations with namespace support (`<lact>`)
- **Type Constructors**: Parsed constructor calls from `OUT{}` blocks
- **Action Execution**: Deferred action execution with validation lifecycle
- **Action Detection**: Recursive checking for unexecuted action placeholders
- **Re-validation**: Safe substitution of action results with full validation

**When to Use LNDL Types:**

- Parsing LLM responses with structured output directives
- Implementing deferred action execution (parse → execute → validate)
- Building operable systems with partial validation
- Creating type-safe LLM interaction patterns

**Action Execution Lifecycle:**

```text
1. Parse LNDL response → ActionCall objects created
2. Partial Validation → BaseModels use model_construct() (bypass validation)
3. Execute Actions → Caller executes ActionCall.function with ActionCall.arguments
4. Re-validate → Replace ActionCall objects with results, full validation enforced
```

## Types Reference

### LvarMetadata

Frozen dataclass representing metadata for namespace-prefixed variable declarations.

**Definition:**

```python
from lionherd_core.lndl import LvarMetadata

@dataclass(slots=True, frozen=True)
class LvarMetadata:
    """Metadata for namespace-prefixed lvar.

    Example: <lvar Report.title title>Good Title</lvar>
    → LvarMetadata(model="Report", field="title", local_name="title", value="Good Title")
    """

    model: str          # Model name (e.g., "Report")
    field: str          # Field name (e.g., "title")
    local_name: str     # Local variable name (e.g., "title")
    value: str          # Raw string value
```

**Attributes:**

| Attribute    | Type  | Description                                          |
| ------------ | ----- | ---------------------------------------------------- |
| `model`      | `str` | Model/class name the variable belongs to             |
| `field`      | `str` | Field name within the model                          |
| `local_name` | `str` | Local reference name used in OUT{} constructors      |
| `value`      | `str` | Raw string value from the lvar tag content           |

**Examples:**

```python
# LNDL directive:
# <lvar Report.title title>Quarterly Earnings Report</lvar>

metadata = LvarMetadata(
    model="Report",
    field="title",
    local_name="title",
    value="Quarterly Earnings Report"
)

print(metadata.model)       # "Report"
print(metadata.field)       # "title"
print(metadata.local_name)  # "title"
print(metadata.value)       # "Quarterly Earnings Report"
```

**Notes:**

Frozen dataclass ensures immutability - once created, metadata cannot be modified. This guarantees parse-time metadata integrity throughout the action execution lifecycle.

---

### LactMetadata

Frozen dataclass representing metadata for action declarations (namespaced or direct).

**Definition:**

```python
from lionherd_core.lndl import LactMetadata

@dataclass(slots=True, frozen=True)
class LactMetadata:
    """Metadata for action declaration (namespaced or direct).

    Examples:
        Namespaced: <lact Report.summary s>generate_summary(...)</lact>
        → LactMetadata(model="Report", field="summary", local_name="s", call="generate_summary(...)")

        Direct: <lact search>search(...)</lact>
        → LactMetadata(model=None, field=None, local_name="search", call="search(...)")
    """

    model: str | None       # Model name (e.g., "Report") or None for direct actions
    field: str | None       # Field name (e.g., "summary") or None for direct actions
    local_name: str         # Local reference name (e.g., "s", "search")
    call: str               # Raw function call string
```

**Attributes:**

| Attribute    | Type         | Description                                            |
| ------------ | ------------ | ------------------------------------------------------ |
| `model`      | `str \| None` | Model name for namespaced actions, None for direct     |
| `field`      | `str \| None` | Field name for namespaced actions, None for direct     |
| `local_name` | `str`        | Local reference name used in OUT{} constructors        |
| `call`       | `str`        | Raw Python function call string                        |

**Examples:**

```python
# Namespaced action (bound to model field)
# <lact Report.summary s>generate_summary(documents=docs)</lact>

namespaced = LactMetadata(
    model="Report",
    field="summary",
    local_name="s",
    call="generate_summary(documents=docs)"
)

# Direct action (standalone function call)
# <lact search>search(query="AI", limit=10)</lact>

direct = LactMetadata(
    model=None,
    field=None,
    local_name="search",
    call="search(query='AI', limit=10)"
)

# Check if action is namespaced
is_namespaced = namespaced.model is not None
# True
```

**Notes:**

Namespaced actions (`model` and `field` set) bind to specific model fields, enabling type-safe action execution where the action result must satisfy the field's type constraints. Direct actions (`model` and `field` are None) represent standalone operations.

---

### ParsedConstructor

Frozen dataclass representing a parsed type constructor from an `OUT{}` block.

**Definition:**

```python
from lionherd_core.lndl import ParsedConstructor

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
```

**Attributes:**

| Attribute    | Type             | Description                                      |
| ------------ | ---------------- | ------------------------------------------------ |
| `class_name` | `str`            | Name of the class being constructed              |
| `kwargs`     | `dict[str, Any]` | Parsed keyword arguments for constructor         |
| `raw`        | `str`            | Original raw constructor string from OUT{} block |

**Properties:**

**has_dict_unpack** : bool (read-only)

Returns True if constructor uses `**dict` unpacking in kwargs, False otherwise.

Detects kwargs keys starting with `"**"`, indicating dict unpacking syntax like `Report(**fields)`.

**Examples:**

```python
# LNDL directive:
# OUT{Report(title=title, summary=s, **metadata)}

constructor = ParsedConstructor(
    class_name="Report",
    kwargs={
        "title": "title",
        "summary": "s",
        "**metadata": "metadata"
    },
    raw="Report(title=title, summary=s, **metadata)"
)

print(constructor.class_name)      # "Report"
print(constructor.kwargs["title"])  # "title"
print(constructor.has_dict_unpack)  # True

# Constructor without unpacking
simple = ParsedConstructor(
    class_name="Config",
    kwargs={"setting": "value"},
    raw="Config(setting=value)"
)

print(simple.has_dict_unpack)  # False
```

**Notes:**

The `has_dict_unpack` property enables special handling during constructor evaluation, where unpacking syntax requires merging multiple dictionaries into the final kwargs before instantiation.

---

### ActionCall

Frozen dataclass representing a parsed action call from `<lact>` tag.

**Definition:**

```python
from lionherd_core.lndl import ActionCall

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
```

**Attributes:**

| Attribute   | Type             | Description                                        |
| ----------- | ---------------- | -------------------------------------------------- |
| `name`      | `str`            | Local reference name from lact tag                 |
| `function`  | `str`            | Function/tool name to invoke                       |
| `arguments` | `dict[str, Any]` | Parsed arguments dictionary for function call      |
| `raw_call`  | `str`            | Original Python function call string               |

**Examples:**

```python
# LNDL directive:
# <lact search>search(query="AI", limit=10)</lact>

action = ActionCall(
    name="search",
    function="search",
    arguments={"query": "AI", "limit": 10},
    raw_call="search(query='AI', limit=10)"
)

print(action.name)        # "search"
print(action.function)    # "search"
print(action.arguments)   # {"query": "AI", "limit": 10}

# Execute the action
result = execute_tool(action.function, **action.arguments)

# Action as placeholder in model
from pydantic import BaseModel

class Report(BaseModel):
    summary: str

# During partial validation, summary field contains ActionCall
report = Report.model_construct(summary=action)
print(isinstance(report.summary, ActionCall))  # True
```

**Notes:**

ActionCall objects serve as **placeholders** during partial validation. Models constructed with `model_construct()` can contain ActionCall objects where actual values are expected. These must be replaced with execution results via `revalidate_with_action_results()` before final use.

---

### LNDLOutput

Frozen dataclass representing validated LNDL output with complete action execution lifecycle.

**Definition:**

```python
from lionherd_core.lndl import LNDLOutput

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
    """

    fields: dict[str, BaseModel | ActionCall]       # BaseModel instances or ActionCall (pre-execution)
    lvars: dict[str, str] | dict[str, LvarMetadata] # Preserved for debugging
    lacts: dict[str, LactMetadata]                  # All declared actions (for debugging/reference)
    actions: dict[str, ActionCall]                  # Actions referenced in OUT{} (pending execution)
    raw_out_block: str                              # Preserved for debugging

    def __getitem__(self, key: str) -> BaseModel | ActionCall: ...
    def __getattr__(self, key: str) -> BaseModel | ActionCall: ...
```

**Attributes:**

| Attribute       | Type                                       | Description                                              |
| --------------- | ------------------------------------------ | -------------------------------------------------------- |
| `fields`        | `dict[str, BaseModel \| ActionCall]`       | Constructed model instances or ActionCall placeholders   |
| `lvars`         | `dict[str, str] \| dict[str, LvarMetadata]`| Variable declarations (name → value or metadata)         |
| `lacts`         | `dict[str, LactMetadata]`                  | All action declarations (for reference/debugging)        |
| `actions`       | `dict[str, ActionCall]`                    | Actions referenced in OUT{} requiring execution          |
| `raw_out_block` | `str`                                      | Raw OUT{} block string (preserved for debugging)         |

**Methods:**

**`__getitem__(key: str)`**

Dictionary-style access to fields.

```python
output = parse_lndl(response, operable)
report = output["report"]  # Access via dict syntax
```

**`__getattr__(key: str)`**

Attribute-style access to fields.

```python
output = parse_lndl(response, operable)
report = output.report  # Access via attribute syntax
```

**Examples:**

```python
from lionherd_core.lndl import parse_lndl

# Parse LNDL response
response = """
<lvar Report.title title>Quarterly Report</lvar>
<lact Report.summary s>generate_summary(docs=documents)</lact>

OUT{Report(title=title, summary=s)}
"""

output = parse_lndl(response, Report)

# Check what needs execution
print(output.actions)
# {'s': ActionCall(name='s', function='generate_summary', ...)}

# Execute actions
action_results = {}
for name, action in output.actions.items():
    result = execute_tool(action.function, action.arguments)
    action_results[name] = result

# Re-validate models with results
report = output.fields["report"]
if has_action_calls(report):
    report = revalidate_with_action_results(report, action_results)
    output.fields["report"] = report

# Now safe to use
final_report = output.report
print(final_report.title)   # "Quarterly Report"
print(final_report.summary) # Executed summary text
```

**Lifecycle Example:**

```python
# 1. Parse - ActionCall objects created
output = parse_lndl(llm_response, operable)
# output.actions = {'search': ActionCall(...)}

# 2. Partial Validation - Models constructed with ActionCall placeholders
report = output.report  # Report.model_construct(title="Report", results=ActionCall(...))

# 3. Execute - Run actions and collect results
results = {}
for name, action in output.actions.items():
    results[name] = execute_tool(action.function, **action.arguments)

# 4. Re-validate - Replace placeholders with results
validated_report = revalidate_with_action_results(report, results)
# Now validated_report.results is actual data, not ActionCall
```

**Notes:**

LNDLOutput is **frozen** - all attributes are immutable after creation. However, the dicts themselves contain mutable references (BaseModel instances), allowing in-place updates during the re-validation phase.

**Critical**: Models in `fields` may contain ActionCall placeholders. Always check with `has_action_calls()` and re-validate before persistence or production use.

---

## Functions Reference

### has_action_calls()

Check if a BaseModel instance contains any ActionCall objects in its fields.

**Signature:**

```python
def has_action_calls(model: BaseModel) -> bool: ...
```

**Parameters:**

**model** : BaseModel

Pydantic BaseModel instance to check for ActionCall objects.

**Returns:**

- **bool**: True if any field value is an ActionCall (at any nesting level), False otherwise.

**Examples:**

```python
from lionherd_core.lndl import has_action_calls, ActionCall
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str

# Model with ActionCall placeholder
action = ActionCall(name="s", function="summarize", arguments={}, raw_call="summarize()")
report = Report.model_construct(title="Report", summary=action)

print(has_action_calls(report))  # True

# Fully validated model
validated_report = Report(title="Report", summary="Actual summary text")
print(has_action_calls(validated_report))  # False

# Detects nested ActionCall objects
class NestedReport(BaseModel):
    main: Report

nested = NestedReport(main=report)
print(has_action_calls(nested))  # True (found in nested Report)
```

**Recursive Detection:**

```python
class ComplexModel(BaseModel):
    items: list[Report]
    metadata: dict[str, Report]

# ActionCall in list item
complex_model = ComplexModel.model_construct(
    items=[report],  # report has ActionCall
    metadata={}
)
print(has_action_calls(complex_model))  # True

# ActionCall in dict value
complex_model2 = ComplexModel.model_construct(
    items=[],
    metadata={"key": report}  # report has ActionCall
)
print(has_action_calls(complex_model2))  # True
```

**Notes:**

Recursively checks nested BaseModel fields and collections (list, tuple, set, dict) to detect ActionCall objects at any depth. Essential guard before persistence to prevent database corruption from unexecuted action placeholders.

---

### ensure_no_action_calls()

Validate that model contains no unexecuted ActionCall objects.

**Signature:**

```python
def ensure_no_action_calls(model: BaseModel) -> BaseModel: ...
```

**Parameters:**

**model** : BaseModel

BaseModel instance to validate for ActionCall placeholders.

**Returns:**

- **BaseModel**: The same model instance if validation passes (no ActionCall objects found).

**Raises:**

- **ValueError**: If model contains any ActionCall objects. Error message includes field paths showing where ActionCall objects were found (up to first 3 fields).

**Examples:**

```python
from lionherd_core.lndl import ensure_no_action_calls, ActionCall
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str

# SAFE: Fully validated model
report = Report(title="Report", summary="Summary text")
ensure_no_action_calls(report)  # OK, returns report

# UNSAFE: Model with ActionCall placeholder
action = ActionCall(name="s", function="summarize", arguments={}, raw_call="summarize()")
partial_report = Report.model_construct(title="Report", summary=action)

try:
    ensure_no_action_calls(partial_report)
except ValueError as e:
    print(e)
    # Report contains unexecuted actions in fields: summary.
    # Models with ActionCall placeholders must be re-validated after action execution.
    # Call revalidate_with_action_results() before using this model.
```

**Database Persistence Guard:**

```python
from lionherd_core.lndl import parse_lndl, ensure_no_action_calls

# Parse LNDL response
output = parse_lndl(llm_response, operable)
report = output.report

# BAD: Forgot to execute actions - guard prevents corruption
try:
    db.save(ensure_no_action_calls(report))
except ValueError:
    print("Cannot save: model has unexecuted actions")

# GOOD: Execute actions first, then save
action_results = execute_actions(output.actions)
validated_report = revalidate_with_action_results(report, action_results)
db.save(ensure_no_action_calls(validated_report))  # OK
```

**Multiple ActionCall Detection:**

```python
class MultiFieldReport(BaseModel):
    title: str
    summary: str
    metadata: dict[str, str]

# Multiple fields with ActionCall
report = MultiFieldReport.model_construct(
    title=action1,
    summary=action2,
    metadata={"key": action3}
)

try:
    ensure_no_action_calls(report)
except ValueError as e:
    print(e)
    # MultiFieldReport contains unexecuted actions in fields: title, summary, metadata['key'].
    # ...
```

**Notes:**

Use this guard **before any persistence operation** (database, file, cache) to prevent storing models with unexecuted action placeholders. The error message provides actionable feedback with field paths for debugging.

---

### revalidate_with_action_results()

Replace ActionCall fields with execution results and re-validate the model.

**Signature:**

```python
def revalidate_with_action_results(
    model: BaseModel,
    action_results: dict[str, Any],
) -> BaseModel: ...
```

**Parameters:**

**model** : BaseModel

BaseModel instance with ActionCall placeholders (created via `model_construct()`).

**action_results** : dict[str, Any]

Dictionary mapping action names to their execution results.

**Returns:**

- **BaseModel**: Fully validated BaseModel instance with action results substituted and all field constraints enforced.

**Raises:**

- **ValueError**: If an ActionCall's name is not found in `action_results` dictionary.
- **ValidationError**: If action results don't satisfy field constraints (type, validators, bounds, regex).

**Examples:**

```python
from lionherd_core.lndl import revalidate_with_action_results, ActionCall
from pydantic import BaseModel, Field

class Report(BaseModel):
    title: str
    summary: str = Field(..., min_length=10)  # Constraint: min 10 chars

# Model with ActionCall placeholder
action = ActionCall(
    name="summarize",
    function="generate_summary",
    arguments={"docs": "documents"},
    raw_call="generate_summary(docs='documents')"
)

report = Report.model_construct(
    title="Quarterly Report",
    summary=action  # Placeholder - constraint NOT enforced yet
)

# Execute action and get result
action_results = {
    "summarize": "This is a comprehensive summary of quarterly earnings."
}

# Re-validate with results
validated_report = revalidate_with_action_results(report, action_results)

print(isinstance(validated_report.summary, str))  # True
print(len(validated_report.summary) >= 10)        # True (constraint enforced)
```

**Validation Enforcement:**

```python
class StrictReport(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)  # Must be between 0 and 1

action = ActionCall(name="calc", function="calculate_score", arguments={}, raw_call="...")
report = StrictReport.model_construct(score=action)

# Invalid result - fails constraint
try:
    validated = revalidate_with_action_results(
        report,
        {"calc": 1.5}  # Outside valid range!
    )
except ValidationError as e:
    print("Validation failed: score must be <= 1.0")
```

**Missing Action Results:**

```python
report = Report.model_construct(title="Title", summary=action)

# Forgot to include "summarize" in results
try:
    validated = revalidate_with_action_results(
        report,
        {"wrong_name": "value"}
    )
except ValueError as e:
    print(e)
    # Action 'summarize' in field 'summary' has no execution result.
    # Available results: ['wrong_name']
```

**Complete Workflow:**

```python
from lionherd_core.lndl import parse_lndl, has_action_calls, ensure_no_action_calls

# 1. Parse LNDL response
output = parse_lndl(llm_response, Report)
report = output.report

# 2. Execute actions
action_results = {}
for name, action in output.actions.items():
    result = execute_tool(action.function, **action.arguments)
    action_results[name] = result

# 3. Re-validate with results
if has_action_calls(report):
    report = revalidate_with_action_results(report, action_results)

# 4. Guard before persistence
db.save(ensure_no_action_calls(report))  # Safe!
```

**Notes:**

This function restores **full Pydantic validation** after action execution. Models constructed with `model_construct()` bypass all field validators - re-validation ensures action results satisfy all type constraints, bounds, regex patterns, and custom validators before the model enters production use.

**Type Safety**: Re-validation occurs via `type(model)(**kwargs)`, creating a fresh instance with full validation. The original partially-validated model is not mutated.

---

## Usage Patterns

### Basic LNDL Parsing and Execution

```python
from lionherd_core.lndl import parse_lndl, has_action_calls, revalidate_with_action_results
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str

# LNDL response from LLM
response = """
<lvar Report.title title>Q4 Earnings Report</lvar>
<lact Report.summary s>generate_summary(quarter="Q4")</lact>

OUT{Report(title=title, summary=s)}
"""

# 1. Parse LNDL
output = parse_lndl(response, Report)

# 2. Check for actions
if output.actions:
    print(f"Actions to execute: {list(output.actions.keys())}")

    # 3. Execute actions
    action_results = {}
    for name, action in output.actions.items():
        result = execute_tool(action.function, **action.arguments)
        action_results[name] = result

    # 4. Re-validate models
    report = output.report
    if has_action_calls(report):
        report = revalidate_with_action_results(report, action_results)
else:
    report = output.report

# 5. Use validated model
print(report.title)   # "Q4 Earnings Report"
print(report.summary) # Generated summary text
```

### Multiple Models with Actions

```python
from lionherd_core.lndl import LNDLOutput

response = """
<lvar Config.name n>Production Config</lvar>
<lact Report.data d>fetch_data(source="db")</lact>

OUT{
    config: Config(name=n),
    report: Report(title="Report", data=d)
}
"""

output = parse_lndl(response, {"config": Config, "report": Report})

# Execute all actions
action_results = {}
for name, action in output.actions.items():
    action_results[name] = execute_tool(action.function, **action.arguments)

# Re-validate each model
for field_name, model in output.fields.items():
    if has_action_calls(model):
        output.fields[field_name] = revalidate_with_action_results(model, action_results)

# Access validated models
config = output.config   # Config instance
report = output.report   # Report instance
```

### Database Persistence Guard

```python
from lionherd_core.lndl import ensure_no_action_calls, parse_lndl

def save_report(llm_response: str, db: Database) -> None:
    """Parse LNDL response and safely persist to database."""
    # Parse
    output = parse_lndl(llm_response, Report)
    report = output.report

    # Execute actions
    action_results = {}
    for name, action in output.actions.items():
        action_results[name] = execute_tool(action.function, **action.arguments)

    # Re-validate
    if has_action_calls(report):
        report = revalidate_with_action_results(report, action_results)

    # Guard prevents saving models with unexecuted actions
    validated_report = ensure_no_action_calls(report)

    # Safe to persist
    db.save(validated_report)
```

### Handling Validation Errors

```python
from pydantic import ValidationError
from lionherd_core.lndl import revalidate_with_action_results

try:
    # Execute actions
    action_results = {"summary": "Too short"}  # Fails min_length constraint

    # Re-validate
    validated_report = revalidate_with_action_results(report, action_results)
except ValidationError as e:
    print("Action result failed validation:")
    for error in e.errors():
        print(f"  Field: {error['loc']}")
        print(f"  Error: {error['msg']}")
        print(f"  Value: {error['input']}")

    # Handle error: retry action, use fallback, raise to user
    raise
```

### Debugging with Metadata

```python
output = parse_lndl(response, operable)

# Inspect all declared variables
print("Variables:")
for name, metadata in output.lvars.items():
    if isinstance(metadata, LvarMetadata):
        print(f"  {name}: {metadata.model}.{metadata.field} = {metadata.value}")

# Inspect all declared actions
print("Actions:")
for name, metadata in output.lacts.items():
    print(f"  {name}: {metadata.model or 'direct'}.{metadata.field or 'N/A'}")
    print(f"    Call: {metadata.call}")

# Inspect which actions require execution
print("Pending Actions:")
for name, action in output.actions.items():
    print(f"  {name}: {action.function}({action.arguments})")

# View raw OUT{} block
print(f"Raw OUT block: {output.raw_out_block}")
```

---

## Design Rationale

### Why Frozen Dataclasses?

All LNDL types use frozen dataclasses (`frozen=True`) to ensure immutability:

1. **Parse Integrity**: Metadata captured during parsing cannot be accidentally modified during execution
2. **Thread Safety**: Frozen instances are inherently thread-safe for concurrent action execution
3. **Hash Stability**: Frozen dataclasses can be safely used in sets/dicts (though LNDL types don't implement `__hash__`)
4. **Clear Semantics**: Immutability signals these are data containers, not mutable state objects

### Why Deferred Action Execution?

LNDL separates action **declaration** (`<lact>`) from **execution** to enable:

1. **Partial Validation**: Models can be constructed and type-checked before expensive action execution
2. **Selective Execution**: Only actions referenced in `OUT{}` are executed (declared but unused actions are ignored)
3. **Batch Optimization**: All required actions can be collected and executed in parallel
4. **Error Recovery**: Parse errors are detected before any actions execute, preventing partial side effects

### Why Re-validation?

Models with ActionCall placeholders use `model_construct()` which **bypasses all Pydantic validation**:

1. **Type Safety Gap**: Field types, bounds, regex patterns, custom validators are NOT enforced
2. **Runtime Risk**: Using partially-validated models can cause type errors, database corruption, or logic bugs
3. **Explicit Guard**: `revalidate_with_action_results()` restores full validation guarantees
4. **Fail-Fast**: Validation errors surface immediately after action execution, not during production use

Re-validation transforms **optimistic parse-time construction** into **guaranteed runtime safety**.

### Why has_action_calls() and ensure_no_action_calls()?

These guards prevent common errors in action execution workflows:

1. **has_action_calls()**: Detects when re-validation is needed (avoids redundant validation of fully-validated models)
2. **ensure_no_action_calls()**: Prevents persistence of partially-validated models (database integrity)
3. **Recursive Detection**: Both functions check nested models and collections, preventing deep ActionCall leaks
4. **Explicit Contract**: Makes action execution lifecycle visible and enforceable in code

These guards enforce the **action execution lifecycle protocol** at the type level.

---

## See Also

- **Related Modules**:
  - [LNDL Parser](parser.md): LNDL response parsing and validation
  - [LNDL Resolver](resolver.md): Reference resolution and validation
  - [LNDL Fuzzy](fuzzy.md): Fuzzy matching for typo tolerance
  - [Spec](../types/spec.md): Type specifications
  - [Operable](../types/operable.md): Structured output integration

---

## Examples

### Example 1: Simple Action Execution

```python
from lionherd_core.lndl import parse_lndl, has_action_calls, revalidate_with_action_results
from pydantic import BaseModel

class SearchResult(BaseModel):
    query: str
    results: list[str]

response = """
<lact results>search(query="AI safety", limit=5)</lact>
OUT{SearchResult(query="AI safety", results=results)}
"""

# Parse
output = parse_lndl(response, SearchResult)

# Execute action
def search(query: str, limit: int) -> list[str]:
    # Simulate search
    return [f"Result {i}: {query}" for i in range(limit)]

action_results = {}
for name, action in output.actions.items():
    # action.function = "search"
    # action.arguments = {"query": "AI safety", "limit": 5}
    action_results[name] = search(**action.arguments)

# Re-validate
result = output.results
if has_action_calls(result):
    result = revalidate_with_action_results(result, action_results)

print(result.query)    # "AI safety"
print(result.results)  # ["Result 0: AI safety", "Result 1: AI safety", ...]
```

### Example 2: Nested Models with Actions

```python
from pydantic import BaseModel
from lionherd_core.lndl import parse_lndl, has_action_calls, revalidate_with_action_results

class Summary(BaseModel):
    text: str
    word_count: int

class Report(BaseModel):
    title: str
    summary: Summary

response = """
<lvar title>Annual Report</lvar>
<lact sum_text>generate_summary()</lact>
<lact word_cnt>count_words(text=sum_text)</lact>

OUT{Report(
    title=title,
    summary=Summary(text=sum_text, word_count=word_cnt)
)}
"""

output = parse_lndl(response, Report)

# Execute actions
action_results = {
    "sum_text": "This is a summary of the annual report.",
    "word_cnt": 8
}

# Re-validate (detects ActionCall in nested Summary model)
report = output.report
if has_action_calls(report):
    report = revalidate_with_action_results(report, action_results)

print(report.title)                 # "Annual Report"
print(report.summary.text)          # "This is a summary..."
print(report.summary.word_count)    # 8
```

### Example 3: Error Handling

```python
from pydantic import BaseModel, Field, ValidationError
from lionherd_core.lndl import parse_lndl, revalidate_with_action_results

class StrictReport(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., min_length=20)

response = """
<lact scr>calculate_score()</lact>
<lact sum>generate_summary()</lact>
OUT{StrictReport(score=scr, summary=sum)}
"""

output = parse_lndl(response, StrictReport)

# Execute with invalid results
action_results = {
    "scr": 1.5,        # Fails constraint: must be <= 1.0
    "sum": "Too short" # Fails constraint: must be >= 20 chars
}

try:
    report = revalidate_with_action_results(output.report, action_results)
except ValidationError as e:
    print("Validation failed:")
    for error in e.errors():
        print(f"  {error['loc'][0]}: {error['msg']}")
        # score: Input should be less than or equal to 1
        # summary: String should have at least 20 characters

    # Handle: retry actions with different parameters, use fallback values, etc.
```

### Example 4: Persistence Guard

```python
from lionherd_core.lndl import (
    parse_lndl,
    has_action_calls,
    ensure_no_action_calls,
    revalidate_with_action_results
)

class Database:
    def save(self, model: BaseModel) -> None:
        # Simulate database save
        print(f"Saving {type(model).__name__} to database")

db = Database()

# Parse LNDL
response = """
<lact data>fetch_data()</lact>
OUT{Report(title="Report", data=data)}
"""
output = parse_lndl(response, Report)

# BAD: Forgot to execute actions
try:
    db.save(ensure_no_action_calls(output.report))
except ValueError as e:
    print(f"ERROR: {e}")
    # Report contains unexecuted actions in fields: data.
    # Models with ActionCall placeholders must be re-validated...

# GOOD: Execute and re-validate first
action_results = {"data": ["item1", "item2", "item3"]}
report = revalidate_with_action_results(output.report, action_results)
db.save(ensure_no_action_calls(report))  # OK!
# Saving Report to database
```

### Example 5: Debugging Workflow

```python
from lionherd_core.lndl import parse_lndl

response = """
<lvar Report.title t>Debug Report</lvar>
<lvar Report.version v>1.0</lvar>
<lact Report.summary s>summarize(docs=documents)</lact>
<lact check>validate_data(data=raw_data)</lact>

OUT{Report(title=t, version=v, summary=s)}
"""

output = parse_lndl(response, Report)

# Debug: What variables were declared?
print("Variables declared:")
for name, meta in output.lvars.items():
    if isinstance(meta, LvarMetadata):
        print(f"  {name} → {meta.model}.{meta.field} = '{meta.value}'")
# Variables declared:
#   t → Report.title = 'Debug Report'
#   v → Report.version = '1.0'

# Debug: What actions were declared?
print("Actions declared:")
for name, meta in output.lacts.items():
    print(f"  {name}: {meta.call}")
# Actions declared:
#   s: summarize(docs=documents)
#   check: validate_data(data=raw_data)

# Debug: Which actions need execution?
print("Actions pending execution:")
for name, action in output.actions.items():
    print(f"  {name}: {action.function}(**{action.arguments})")
# Actions pending execution:
#   s: summarize(**{'docs': 'documents'})
# (Note: 'check' not in output.actions because it's not referenced in OUT{})

# Debug: Raw OUT{} block
print(f"Raw OUT block: {output.raw_out_block}")
# Raw OUT block: Report(title=t, version=v, summary=s)
```

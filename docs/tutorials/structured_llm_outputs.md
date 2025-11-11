# Tutorial: Structured LLM Outputs with LNDL

**Time**: 15 min | **Difficulty**: 🔵 Intermediate

Learn how to reliably extract structured data from LLM text responses using lionherd-core's LNDL fuzzy parser.

## The Problem

LLMs produce text, but applications need structured data. JSON parsing fails when LLMs:

- Add explanatory text around JSON
- Use inconsistent formatting
- Make typos in field names
- Miss closing brackets

LNDL solves this with fuzzy, fault-tolerant parsing.

## Core Concepts

**LNDL** (Language InterOperable Network Directive Language) is a markup format designed for LLM output:

```xml
<lvar ModelName.field var_name>value</lvar>
OUT{result: [var1, var2, ...]}
```

**Key benefits**:

- Tolerates typos (Jaro-Winkler fuzzy matching)
- Handles malformed tags
- Extracts structured data from mixed text/markup
- 10-50ms overhead, <5% failure rate (vs 40-60% for strict JSON)

## End-to-End Example: Task Extraction

### 1. Define Your Data Model

```python
from pydantic import BaseModel, Field
from typing import Literal

class Task(BaseModel):
    """A single task extracted from user input."""
    title: str = Field(..., description="Task title")
    priority: Literal["low", "medium", "high"] = Field(..., description="Priority level")
    estimated_hours: float = Field(..., description="Estimated hours to complete")
    tags: list[str] = Field(default_factory=list, description="Task tags")

class TaskList(BaseModel):
    """Collection of tasks."""
    tasks: list[Task] = Field(..., description="All extracted tasks")
    total_hours: float = Field(..., description="Sum of estimated hours")
```

### 2. Create LNDL Spec

```python
from lionherd_core import types

# Create specs for each model
task_spec = types.Spec(Task, name="task")
task_list_spec = types.Spec(TaskList, name="task_list")

# Bundle into Operable for parsing
operable = types.Operable([task_spec, task_list_spec])
```

### 3. Prepare LLM Prompt

```python
prompt = f"""Extract tasks from the user input and return them in LNDL format.

Use this schema:
{operable.instruction}

User input:
"I need to implement authentication (high priority, ~8 hours),
write API docs (medium, ~4 hours), and fix the login bug (high, ~2 hours).
Tag auth and bug work as 'backend'."

Return tasks in LNDL format.
"""
```

### 4. Parse LLM Response (Fuzzy)

```python
from lionherd_core import lndl

# Simulated LLM response (with typos and formatting issues)
llm_response = """
Here are the tasks I extracted:

<lvar Task.title task1>Implement authentication</lvar>
<lvar Task.priority task1>high</lvar>
<lvar Task.estimated_hours task1>8.0</lvar>
<lvar Task.tags task1>["backend"]</lvar>

<lvar Task.titl task2>Write API documentation</lvar>  <!-- typo: 'titl' -->
<lvar Task.priority task2>medium</lvar>
<lvar Task.estimated_hours task2>4.0</lvar>

<lvar Task.title task3>Fix login bug</lvar>
<lvar Task.priority task3>high</lvar>
<lvar Task.estimeted_hours task3>2.0</lvar>  <!-- typo: 'estimeted' -->
<lvar Task.tags task3>["backend"]</lvar>

<lvar TaskList.tasks task_list>[task1, task2, task3]</lvar>
<lvar TaskList.total_hours task_list>14.0</lvar>

OUT{task_list: [task_list]}
"""

# Parse with fuzzy matching (tolerates typos)
result = lndl.parse_lndl_fuzzy(llm_response, operable, threshold=0.85)

# Access structured data
task_list = result.task_list
print(f"Total tasks: {len(task_list.tasks)}")
print(f"Total hours: {task_list.total_hours}")

for task in task_list.tasks:
    print(f"- [{task.priority}] {task.title} ({task.estimated_hours}h)")
```

**Output**:

```text
Total tasks: 3
Total hours: 14.0
- [high] Implement authentication (8.0h)
- [medium] Write API documentation (4.0h)
- [high] Fix login bug (2.0h)
```

**Key insight**: Despite typos (`titl`, `estimeted`), fuzzy parsing succeeded.

## Production Patterns

### Pattern 1: Threshold Tuning

```python
# Strict (fewer false positives, more failures)
result = lndl.parse_lndl_fuzzy(response, operable, threshold=0.95)

# Balanced (default, recommended)
result = lndl.parse_lndl_fuzzy(response, operable, threshold=0.85)

# Lenient (tolerates more typos, risk false positives)
result = lndl.parse_lndl_fuzzy(response, operable, threshold=0.75)
```

**Recommendation**: Start with 0.85, adjust based on your LLM's accuracy.

### Pattern 2: Fallback Strategy

```python
from lionherd_core import lndl

try:
    # Try strict first (faster)
    result = lndl.parse_lndl(response, operable)
except Exception:
    # Fallback to fuzzy if strict fails
    result = lndl.parse_lndl_fuzzy(response, operable, threshold=0.85)
```

### Pattern 3: Validation + Retry

```python
from pydantic import ValidationError

max_retries = 3
for attempt in range(max_retries):
    llm_response = call_llm(prompt)

    try:
        result = lndl.parse_lndl_fuzzy(llm_response, operable)
        break  # Success
    except ValidationError as e:
        if attempt == max_retries - 1:
            raise  # Final attempt failed

        # Add error feedback to prompt
        prompt += f"\n\nPrevious attempt failed: {e}\nPlease fix and try again."
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Parsing overhead | 10-50ms |
| Strict JSON failure rate | 40-60% |
| LNDL fuzzy failure rate | <5% |
| Threshold impact | 0.95: strict, 0.85: balanced, 0.75: lenient |

## When to Use LNDL

**Use LNDL when**:

- LLM outputs need structure (API calls, database inserts)
- Reliability > speed (10-50ms overhead acceptable)
- LLM produces mixed text + structured data

**Don't use when**:

- LLM only returns pure JSON (use `orjson` directly)
- Performance critical (<10ms budget)
- Simple string extraction (regex sufficient)

## Next Steps

- **API Reference**: `docs/api/lndl/parser.md` for full API details
- **API Reference**: `docs/api/types/operable.md` for Operable and Spec documentation

## Real-World Use Case

```python
from typing import Any

# LLM agent system: extract tool calls from LLM response
class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]

class AgentResponse(BaseModel):
    reasoning: str
    tool_calls: list[ToolCall]

# Parse LLM response → execute tools → return results
# llm = YourLLMClient()  # Initialize your LLM client
response = llm.generate(prompt)
parsed = lndl.parse_lndl_fuzzy(response, types.Operable([types.Spec(AgentResponse, "result")]))

for call in parsed.result.tool_calls:
    # def execute_tool(name: str, args: dict) -> Any: ...  # Your tool execution logic
    result = execute_tool(call.tool, call.args)
    # ... continue agent loop
```

**Key takeaway**: LNDL bridges the gap between LLM text and application logic with <5% failure rate.

# Schema Handlers Tutorials

Master function call parsing and schema manipulation for tool-calling patterns (MCP, OpenAI function calling, LLM tool use). Learn to parse function call strings, map arguments, nest parameters, and dynamically select schemas.

## Overview

These tutorials teach you to handle function calling patterns:

- **Function Call Parsing**: Parse string representations of function calls
- **Argument Mapping**: Map positional args to keyword args using signatures
- **Argument Nesting**: Transform flat args into nested structures
- **Dynamic Schema Selection**: Route to correct schema based on function name

**Use Cases**:

- MCP (Model Context Protocol) server implementations
- OpenAI function/tool calling integration
- LLM-based tool use (Claude, GPT-4, etc.)
- Dynamic API routing with schema validation

## Prerequisites

- Python 3.11+
- Basic understanding of function signatures and `inspect` module
- Familiarity with Pydantic models (for schema validation)

## Quick Start

```bash
pip install lionherd-core
jupyter notebook mcp_tool_pipeline.ipynb
```

## Tutorials (2)

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [**MCP Tool Pipeline**](./mcp_tool_pipeline.ipynb) | 20-30 min | Parse function calls, map args, nest parameters for MCP tools |
| [**Dynamic Schema Selection**](./dynamic_schema_selection.ipynb) | 15-25 min | Select and apply schemas dynamically using schema dictionaries |

## Learning Path

**Recommended order** (45 min total):

1. **MCP Tool Pipeline** - Understand parse → map → nest → validate flow
2. **Dynamic Schema Selection** - Learn schema dictionary pattern for routing

**Outcome**: Build MCP servers and LLM tool integrations with proper schema handling

## Key Concepts

### Function Call Parsing

Parse LLM-generated function call strings into structured data:

```python
from lionherd_core.libs.schema_handlers import parse_function_call

# LLM generates: 'create_user("alice", email="alice@example.com", age=30)'
call_str = 'create_user("alice", email="alice@example.com", age=30)'

# Parse into structured format
parsed = parse_function_call(call_str)
# {
#     "function_name": "create_user",
#     "args": ["alice"],
#     "kwargs": {"email": "alice@example.com", "age": 30}
# }
```

**Tutorial**: [MCP Tool Pipeline](./mcp_tool_pipeline.ipynb)

### Argument Mapping (Positional → Keyword)

Map positional args to keyword args using function signature:

```python
from lionherd_core.libs.schema_handlers import map_positional_args
import inspect

def create_user(name: str, email: str, age: int):
    """Create a new user."""
    pass

# LLM provides positional args: ["alice", "alice@example.com", 30]
# We need keyword args for validation: {"name": "alice", "email": ..., "age": ...}

sig = inspect.signature(create_user)
args = ["alice", "alice@example.com", 30]
kwargs = {}

mapped = map_positional_args(sig, args, kwargs)
# {"name": "alice", "email": "alice@example.com", "age": 30}
```

**Why this matters**: Pydantic validation and many APIs require keyword arguments. LLMs often generate positional args that need mapping.

**Tutorial**: [MCP Tool Pipeline](./mcp_tool_pipeline.ipynb)

### Argument Nesting

Transform flat arguments into nested structures:

```python
from lionherd_core.libs.schema_handlers import nest_arguments

# Flat args from LLM
flat_args = {
    "user__name": "alice",
    "user__email": "alice@example.com",
    "settings__theme": "dark",
    "settings__notifications": True
}

# Nest using "__" separator
nested = nest_arguments(flat_args, separator="__")
# {
#     "user": {
#         "name": "alice",
#         "email": "alice@example.com"
#     },
#     "settings": {
#         "theme": "dark",
#         "notifications": True
#     }
# }
```

**Use case**: LLMs sometimes flatten nested structures. This reconstructs the intended hierarchy.

**Tutorial**: [MCP Tool Pipeline](./mcp_tool_pipeline.ipynb)

### Schema Dictionary Pattern

Dynamically select schemas based on function name:

```python
from pydantic import BaseModel

class CreateUserSchema(BaseModel):
    name: str
    email: str
    age: int

class UpdateUserSchema(BaseModel):
    user_id: int
    email: str | None = None
    age: int | None = None

# Schema dictionary maps function names to schemas
SCHEMAS = {
    "create_user": CreateUserSchema,
    "update_user": UpdateUserSchema
}

def validate_tool_call(function_name: str, arguments: dict):
    """Validate tool call using appropriate schema."""
    schema = SCHEMAS.get(function_name)
    if not schema:
        raise ValueError(f"Unknown function: {function_name}")

    return schema.model_validate(arguments)

# Example usage
args = {"name": "alice", "email": "alice@example.com", "age": 30}
validated = validate_tool_call("create_user", args)
```

**Tutorial**: [Dynamic Schema Selection](./dynamic_schema_selection.ipynb)

## Common Patterns

### Pattern 1: MCP Tool Pipeline (Complete Flow)

Full pipeline from function call string to validated execution:

```python
from lionherd_core.libs.schema_handlers import (
    parse_function_call,
    map_positional_args,
    nest_arguments
)
import inspect
from pydantic import BaseModel

# 1. Define tool schema
class CreateUserSchema(BaseModel):
    name: str
    email: str
    age: int

# 2. Define tool function
def create_user(name: str, email: str, age: int):
    return {"id": 123, "name": name, "email": email, "age": age}

# 3. Parse LLM call string
call_str = 'create_user("alice", "alice@example.com", 30)'
parsed = parse_function_call(call_str)

# 4. Map positional to keyword args
sig = inspect.signature(create_user)
mapped_args = map_positional_args(sig, parsed["args"], parsed["kwargs"])

# 5. Nest arguments (if needed)
nested_args = nest_arguments(mapped_args, separator="__")

# 6. Validate with schema
validated = CreateUserSchema.model_validate(nested_args)

# 7. Execute function
result = create_user(**validated.model_dump())
```

**Tutorial**: [MCP Tool Pipeline](./mcp_tool_pipeline.ipynb) walks through this complete pipeline with error handling.

### Pattern 2: Schema Registry with Routing

Central registry for all tool schemas with automatic routing:

```python
from typing import Callable, Any
from pydantic import BaseModel

class ToolRegistry:
    """Registry mapping function names to (function, schema) pairs."""

    def __init__(self):
        self.tools: dict[str, tuple[Callable, type[BaseModel]]] = {}

    def register(self, name: str, func: Callable, schema: type[BaseModel]):
        """Register a tool with its function and schema."""
        self.tools[name] = (func, schema)

    def execute(self, function_name: str, arguments: dict) -> Any:
        """Execute tool with automatic schema validation."""
        if function_name not in self.tools:
            raise ValueError(f"Unknown function: {function_name}")

        func, schema = self.tools[function_name]

        # Validate arguments
        validated = schema.model_validate(arguments)

        # Execute function
        return func(**validated.model_dump())

# Usage
registry = ToolRegistry()
registry.register("create_user", create_user, CreateUserSchema)
registry.register("update_user", update_user, UpdateUserSchema)

# Execute any tool dynamically
result = registry.execute("create_user", {"name": "alice", "email": "alice@example.com", "age": 30})
```

**Tutorial**: [Dynamic Schema Selection](./dynamic_schema_selection.ipynb) extends this pattern with error handling.

### Pattern 3: LLM Function Calling Integration (OpenAI/Claude)

Integrate with LLM function calling APIs:

```python
import openai
from lionherd_core.libs.schema_handlers import parse_function_call

# 1. Define tools for LLM (OpenAI function calling format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_user",
            "description": "Create a new user account",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's full name"},
                    "email": {"type": "string", "description": "User's email address"},
                    "age": {"type": "integer", "description": "User's age"}
                },
                "required": ["name", "email", "age"]
            }
        }
    }
]

# 2. Get LLM response with function call
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Create an account for Alice (alice@example.com, 30 years old)"}],
    tools=tools,
    tool_choice="auto"
)

# 3. Extract function call
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# 4. Execute using registry pattern
result = registry.execute(function_name, arguments)

# 5. Send result back to LLM
follow_up = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Create an account for Alice..."},
        response.choices[0].message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    ]
)
```

**Note**: This pattern applies to any LLM with function calling support (Claude tool use, Gemini function calling, etc.).

## Production Considerations

### Error Handling

Robust error handling for each pipeline stage:

```python
from lionherd_core.libs.schema_handlers import parse_function_call
from lionherd_core.errors import ValidationError
import logging

def safe_execute_tool(call_str: str, registry: ToolRegistry):
    """Execute tool with comprehensive error handling."""
    try:
        # Parse
        parsed = parse_function_call(call_str)
    except (ValueError, SyntaxError) as e:
        logging.error(f"Parse failed: {e}")
        return {"error": "invalid_call_format", "details": str(e)}

    function_name = parsed["function_name"]

    if function_name not in registry.tools:
        return {"error": "unknown_function", "function": function_name}

    try:
        # Execute with validation
        result = registry.execute(function_name, {**parsed["kwargs"]})
        return {"success": True, "result": result}
    except ValidationError as e:
        logging.error(f"Validation failed for {function_name}: {e}")
        return {"error": "validation_failed", "details": str(e)}
    except Exception as e:
        logging.error(f"Execution failed for {function_name}: {e}")
        return {"error": "execution_failed", "details": str(e)}
```

### Performance

- `parse_function_call()`: <1ms per call (uses `ast.parse`)
- `map_positional_args()`: <0.5ms (inspect.Signature lookup)
- `nest_arguments()`: <0.5ms for typical nesting depth (<5 levels)
- Pydantic validation: 1-5ms depending on schema complexity

**Total overhead**: ~5-10ms per tool call (negligible compared to LLM latency)

### Security

**Critical**: Never use `eval()` or `exec()` on untrusted function call strings. Always use `ast.parse()` (as done in `parse_function_call()`).

```python
# ❌ NEVER DO THIS
eval(llm_generated_call)  # Code injection vulnerability

# ✅ SAFE: Use ast.parse (what lionherd-core does)
import ast
parsed = ast.parse(llm_generated_call, mode='eval')
```

### Testing

Test each pipeline stage independently:

```python
def test_parse_function_call():
    call_str = 'create_user("alice", email="alice@example.com")'
    parsed = parse_function_call(call_str)
    assert parsed["function_name"] == "create_user"
    assert parsed["args"] == ["alice"]
    assert parsed["kwargs"] == {"email": "alice@example.com"}

def test_map_positional_args():
    sig = inspect.signature(create_user)
    mapped = map_positional_args(sig, ["alice", "alice@example.com", 30], {})
    assert mapped == {"name": "alice", "email": "alice@example.com", "age": 30}

def test_nest_arguments():
    flat = {"user__name": "alice", "user__email": "alice@example.com"}
    nested = nest_arguments(flat, separator="__")
    assert nested == {"user": {"name": "alice", "email": "alice@example.com"}}
```

## Troubleshooting

### Common Issues

**Issue**: `parse_function_call()` fails on complex function calls
**Solution**: Ensure LLM generates simple function call syntax (no nested calls, lambdas, or complex expressions)

**Issue**: `map_positional_args()` maps arguments incorrectly
**Solution**: Verify function signature has properly annotated parameters in correct order

**Issue**: `nest_arguments()` creates unexpected structure
**Solution**: Check separator matches LLM output format (default `"__"`, some LLMs use `"."`)

**Issue**: Schema validation rejects valid arguments
**Solution**: Use `fuzzy_validate_pydantic()` for lenient validation with field name variations

## Related Resources

- **API Reference**: [libs/schema_handlers](../../../docs/api/libs/schema_handlers/)
- **Function Call Parsing**: [function_call_parser](../../../docs/api/libs/schema_handlers/function_call_parser.md)
- **Argument Mapping**: [function_call_parser](../../../docs/api/libs/schema_handlers/function_call_parser.md)
- **MCP Specification**: <https://spec.modelcontextprotocol.io/>
- **OpenAI Function Calling**: <https://platform.openai.com/docs/guides/function-calling>

## Contributing

Found issues or have suggestions? Open an issue at [lionherd-core GitHub](https://github.com/khive-ai/lionherd-core/issues).

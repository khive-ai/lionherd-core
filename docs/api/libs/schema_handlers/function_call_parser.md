# Function Call Parser

> Parse Python function call syntax into JSON tool invocation format for MCP integration

## Overview

The `function_call_parser` module provides utilities for converting Pythonic function call syntax into JSON-based tool invocation format. This enables natural, readable function call expressions to be parsed and transformed into structured arguments for MCP (Model Context Protocol) tool invocations.

**Key Capabilities:**

- **Python AST Parsing**: Converts Python function call strings to structured data using `ast.parse()`
- **Positional Argument Mapping**: Maps positional arguments to parameter names based on schema
- **Schema-Based Nesting**: Restructures flat arguments into nested objects based on Pydantic schema structure
- **Union Type Handling**: Detects and maps arguments to nested union type members
- **Safe Evaluation**: Uses `ast.literal_eval()` for safe argument value extraction

**When to Use This Module:**

- Converting MCP tool requests from readable function syntax to JSON format
- Parsing user-provided function call strings for tool invocation
- Building developer-friendly APIs that accept Python-like function syntax
- Mapping flat argument dictionaries to nested schema structures

**When NOT to Use:**

- Direct function invocation (use native Python `call()` instead)
- Parsing arbitrary Python code (module is limited to function call expressions)
- Runtime code execution (no `eval()` support by design for security)
- Complex function calls with `*args`, `**kwargs` (not supported)

## Functions

### parse_function_call()

Parse Python function call syntax into JSON tool invocation format.

**Signature:**

```python
def parse_function_call(call_str: str) -> dict[str, Any]: ...
```

**Parameters:**

**call_str** : str

Python function call expression as a string. Must be valid Python syntax representing a single function call.

- Supported: `tool_name(arg1, arg2, key=value)`
- Supported: `client.search(query="test")` (attribute access)
- Not supported: Multiple statements, `**kwargs` syntax, complex expressions

**Returns:**

- dict[str, Any]: Dictionary with two keys:
  - `'tool'` (str): Extracted function/tool name
  - `'arguments'` (dict[str, Any]): Parsed arguments with:
    - Positional args as `_pos_0`, `_pos_1`, etc. (to be mapped later)
    - Keyword args with their original names

**Raises:**

- ValueError: If `call_str` is not valid Python syntax
- ValueError: If `call_str` is not a function call expression
- ValueError: If function type is unsupported (e.g., lambda, complex expressions)
- ValueError: If `**kwargs` syntax is used (not supported)
- SyntaxError: If `call_str` contains Python syntax errors

**Examples:**

```python
>>> from lionherd_core.libs.schema_handlers import parse_function_call

# Simple function call with keyword arguments
>>> parse_function_call('search(query="AI", limit=5)')
{'tool': 'search', 'arguments': {'query': 'AI', 'limit': 5}}

# Function call with positional arguments
>>> parse_function_call('add(1, 2, 3)')
{'tool': 'add', 'arguments': {'_pos_0': 1, '_pos_1': 2, '_pos_2': 3}}

# Mixed positional and keyword arguments
>>> parse_function_call('create("user", "Alice", age=30)')
{'tool': 'create', 'arguments': {'_pos_0': 'user', '_pos_1': 'Alice', 'age': 30}}

# Attribute access function call
>>> parse_function_call('client.search(query="test")')
{'tool': 'search', 'arguments': {'query': 'test'}}

# Complex data types
>>> parse_function_call('config(settings={"debug": true, "timeout": 30})')
{'tool': 'config', 'arguments': {'settings': {'debug': True, 'timeout': 30}}}

# Invalid syntax raises ValueError
>>> parse_function_call('not a function call')
ValueError: Not a function call

# **kwargs not supported
>>> parse_function_call('func(**kwargs)')
ValueError: **kwargs not supported
```

**See Also:**

- `map_positional_args()`: Map positional arguments to parameter names
- `nest_arguments_by_schema()`: Restructure arguments based on schema

**Notes:**

Uses Python's AST (Abstract Syntax Tree) parser for safe syntax analysis without executing code. Positional arguments are temporarily stored with `_pos_N` keys and must be mapped to actual parameter names using `map_positional_args()` with schema information.

**Security**: `ast.literal_eval()` restricts argument values to Python literals (strings, numbers, tuples, lists, dicts, booleans, None), preventing arbitrary code execution.

---

### map_positional_args()

Map positional arguments (`_pos_0`, `_pos_1`, ...) to actual parameter names.

**Signature:**

```python
def map_positional_args(
    arguments: dict[str, Any],
    param_names: list[str]
) -> dict[str, Any]: ...
```

**Parameters:**

**arguments** : dict of {str : Any}

Argument dictionary from `parse_function_call()` containing positional args as `_pos_N` keys and keyword args with their original names.

**param_names** : list of str

Ordered list of parameter names from the function schema. Positional arguments are mapped in order to these names.

**Returns:**

- dict[str, Any]: New dictionary with:
  - Positional args mapped to parameter names (e.g., `_pos_0` → `param_names[0]`)
  - Keyword args preserved with original names

**Raises:**

- ValueError: If more positional arguments provided than parameter names available

**Examples:**

```python
>>> from lionherd_core.libs.schema_handlers import (
...     parse_function_call,
...     map_positional_args
... )

# Parse function call with positional args
>>> args = parse_function_call('search("AI", 10)')
>>> args
{'tool': 'search', 'arguments': {'_pos_0': 'AI', '_pos_1': 10}}

# Map to parameter names
>>> mapped = map_positional_args(args['arguments'], ['query', 'limit'])
>>> mapped
{'query': 'AI', 'limit': 10}

# Mixed positional and keyword args
>>> args = parse_function_call('create("user", name="Alice")')
>>> map_positional_args(args['arguments'], ['entity_type', 'id', 'name'])
{'entity_type': 'user', 'name': 'Alice'}

# Too many positional args raises error
>>> args = parse_function_call('func(1, 2, 3)')
>>> map_positional_args(args['arguments'], ['a', 'b'])
ValueError: Too many positional arguments (expected 2)

# Keyword args don't need schema mapping
>>> args = parse_function_call('search(query="AI", limit=10)')
>>> map_positional_args(args['arguments'], [])
{'query': 'AI', 'limit': 10}
```

**See Also:**

- `parse_function_call()`: Initial parsing step
- `nest_arguments_by_schema()`: Next step for schema-based nesting

**Notes:**

Positional arguments are mapped in order: `_pos_0` → `param_names[0]`, `_pos_1` → `param_names[1]`, etc. Keyword arguments are preserved as-is since they already have correct names.

---

### nest_arguments_by_schema()

Restructure flat arguments into nested format based on Pydantic schema structure.

**Signature:**

```python
def nest_arguments_by_schema(
    arguments: dict[str, Any],
    schema_cls
) -> dict[str, Any]: ...
```

**Parameters:**

**arguments** : dict of {str : Any}

Flat argument dictionary with all parameters at top level (after positional mapping).

**schema_cls** : Pydantic model class or None

Schema class defining the structure. Must be a Pydantic `BaseModel` subclass with `model_fields` attribute. If None or not a Pydantic model, returns arguments unchanged.

**Returns:**

- dict[str, Any]: Restructured dictionary with:
  - Top-level fields at root
  - Nested model fields grouped into sub-dictionaries
  - Unknown fields preserved at top level (will fail validation later)

**Notes:**

Detects nested structure by inspecting `schema_cls.model_fields`:

- **Union types**: Collects fields from all union members (handles `Type1 | Type2`)
- **Pydantic models**: Extracts fields from nested BaseModel subclasses
- **Flat schemas**: Returns unchanged if no nested fields detected

Unknown fields (not in schema) are preserved at top level to enable Pydantic validation errors with clear field names.

**Examples:**

```python
>>> from pydantic import BaseModel
>>> from lionherd_core.libs.schema_handlers import nest_arguments_by_schema

# Define schema with nested structure
>>> class SearchOptions(BaseModel):
...     query: str
...     limit: int

>>> class SearchRequest(BaseModel):
...     action: str
...     options: SearchOptions

# Flat arguments
>>> flat_args = {'action': 'search', 'query': 'AI', 'limit': 10}

# Nest based on schema
>>> nested = nest_arguments_by_schema(flat_args, SearchRequest)
>>> nested
{'action': 'search', 'options': {'query': 'AI', 'limit': 10}}

# Validate nested structure works
>>> request = SearchRequest(**nested)
>>> request.options.query
'AI'

# Schema with union types
>>> class Option1(BaseModel):
...     field_a: str

>>> class Option2(BaseModel):
...     field_b: int

>>> class UnionRequest(BaseModel):
...     action: str
...     config: Option1 | Option2

# Fields from either union member get nested
>>> flat_args = {'action': 'test', 'field_a': 'value'}
>>> nest_arguments_by_schema(flat_args, UnionRequest)
{'action': 'test', 'config': {'field_a': 'value'}}

>>> flat_args = {'action': 'test', 'field_b': 42}
>>> nest_arguments_by_schema(flat_args, UnionRequest)
{'action': 'test', 'config': {'field_b': 42}}

# No nested fields - returns unchanged
>>> class FlatSchema(BaseModel):
...     query: str
...     limit: int

>>> flat_args = {'query': 'AI', 'limit': 10}
>>> nest_arguments_by_schema(flat_args, FlatSchema)
{'query': 'AI', 'limit': 10}

# Unknown fields preserved (for validation errors)
>>> flat_args = {'action': 'search', 'query': 'AI', 'unknown': 'value'}
>>> nest_arguments_by_schema(flat_args, SearchRequest)
{'action': 'search', 'options': {'query': 'AI'}, 'unknown': 'value'}
```

**See Also:**

- `parse_function_call()`: Parse function call syntax
- `map_positional_args()`: Map positional arguments first

**Design Notes:**

**Union Type Handling**: When a field is a union type (`Type1 | Type2`), the function collects all fields from all union members. This allows flat arguments to match any union member's structure. The actual type resolution happens during Pydantic validation.

**Unknown Field Strategy**: Unknown fields remain at top level rather than being discarded or raising errors immediately. This enables Pydantic's validation to provide clear error messages with field names.

## Usage Patterns

### Basic MCP Tool Invocation

```python
from lionherd_core.libs.schema_handlers import (
    parse_function_call,
    map_positional_args,
    nest_arguments_by_schema
)
from pydantic import BaseModel

# Define tool schema
class SearchOptions(BaseModel):
    query: str
    limit: int = 10

class SearchAction(BaseModel):
    action: str
    options: SearchOptions

# Parse user input
user_input = 'search(query="AI", limit=5)'
parsed = parse_function_call(user_input)
# {'tool': 'search', 'arguments': {'query': 'AI', 'limit': 5}}

# Map positional args (none in this case)
mapped = map_positional_args(parsed['arguments'], [])

# Nest by schema
nested = nest_arguments_by_schema(mapped, SearchAction)
# {'options': {'query': 'AI', 'limit': 5}}

# Add action field and validate
nested['action'] = parsed['tool']
request = SearchAction(**nested)
```

### Positional Argument Handling

```python
from lionherd_core.libs.schema_handlers import (
    parse_function_call,
    map_positional_args
)

# User provides positional args
user_input = 'create("user", "alice_id", "Alice Smith")'

# Parse
parsed = parse_function_call(user_input)
# {'tool': 'create', 'arguments': {'_pos_0': 'user', '_pos_1': 'alice_id', '_pos_2': 'Alice Smith'}}

# Define parameter order
param_names = ['entity_type', 'id', 'name']

# Map positional to named
mapped = map_positional_args(parsed['arguments'], param_names)
# {'entity_type': 'user', 'id': 'alice_id', 'name': 'Alice Smith'}
```

### Mixed Positional and Keyword Arguments

```python
from lionherd_core.libs.schema_handlers import (
    parse_function_call,
    map_positional_args
)

# Mixed args
user_input = 'search("AI research", limit=20, sort="relevance")'

parsed = parse_function_call(user_input)
# {'tool': 'search', 'arguments': {
#     '_pos_0': 'AI research',
#     'limit': 20,
#     'sort': 'relevance'
# }}

# Map positional (first param is 'query')
mapped = map_positional_args(parsed['arguments'], ['query'])
# {'query': 'AI research', 'limit': 20, 'sort': 'relevance'}
```

### Complex Nested Schema

```python
from pydantic import BaseModel
from lionherd_core.libs.schema_handlers import (
    parse_function_call,
    map_positional_args,
    nest_arguments_by_schema
)

# Define nested schema
class Filters(BaseModel):
    category: str
    min_score: float

class SearchConfig(BaseModel):
    timeout: int
    filters: Filters

class SearchRequest(BaseModel):
    action: str
    query: str
    config: SearchConfig

# Flat user input
user_input = 'search(query="AI", timeout=30, category="research", min_score=0.8)'

# Parse and map
parsed = parse_function_call(user_input)
mapped = map_positional_args(parsed['arguments'], [])

# Nest by schema (config.filters automatically nested)
nested = nest_arguments_by_schema(mapped, SearchRequest)
# {
#     'query': 'AI',
#     'config': {
#         'timeout': 30,
#         'filters': {
#             'category': 'research',
#             'min_score': 0.8
#         }
#     }
# }

nested['action'] = 'search'
request = SearchRequest(**nested)
```

### Union Type Schema

```python
from pydantic import BaseModel
from lionherd_core.libs.schema_handlers import (
    parse_function_call,
    nest_arguments_by_schema
)

# Union schema
class EmailConfig(BaseModel):
    email: str
    send_notifications: bool

class PhoneConfig(BaseModel):
    phone: str
    sms_enabled: bool

class UserRequest(BaseModel):
    action: str
    contact: EmailConfig | PhoneConfig

# Email variant
user_input = 'create(email="alice@example.com", send_notifications=true)'
parsed = parse_function_call(user_input)

nested = nest_arguments_by_schema(parsed['arguments'], UserRequest)
# {'contact': {'email': 'alice@example.com', 'send_notifications': True}}

nested['action'] = 'create'
request = UserRequest(**nested)
# request.contact will be EmailConfig instance

# Phone variant
user_input = 'create(phone="+1234567890", sms_enabled=false)'
parsed = parse_function_call(user_input)

nested = nest_arguments_by_schema(parsed['arguments'], UserRequest)
# {'contact': {'phone': '+1234567890', 'sms_enabled': False}}

nested['action'] = 'create'
request = UserRequest(**nested)
# request.contact will be PhoneConfig instance
```

## Common Pitfalls

### Pitfall 1: Forgetting to Map Positional Arguments

**Issue**: Positional arguments remain as `_pos_N` keys if not mapped.

```python
parsed = parse_function_call('search("AI", 10)')
# {'tool': 'search', 'arguments': {'_pos_0': 'AI', '_pos_1': 10}}

# Forgot to map!
# Schema validation will fail - no 'query' or 'limit' fields
request = SearchAction(**parsed['arguments'])
# ValidationError: Field required: query
```

**Solution**: Always call `map_positional_args()` after parsing, even if no positional args expected.

```python
mapped = map_positional_args(parsed['arguments'], ['query', 'limit'])
request = SearchAction(**mapped)  # Now works
```

### Pitfall 2: Using Unsupported Syntax

**Issue**: `**kwargs` and `*args` are not supported.

```python
# Raises ValueError
parse_function_call('search(**options)')
# ValueError: **kwargs not supported

# Also not supported
parse_function_call('create(*items)')
# ValueError: Invalid function call syntax
```

**Solution**: Use explicit keyword or positional arguments only.

```python
parse_function_call('search(query="AI", limit=10)')  # OK
parse_function_call('create("item1", "item2")')  # OK
```

### Pitfall 3: Wrong Parameter Name Order

**Issue**: Positional arguments mapped in wrong order produce incorrect results.

```python
parsed = parse_function_call('divide(10, 2)')
# {'tool': 'divide', 'arguments': {'_pos_0': 10, '_pos_1': 2}}

# Wrong order!
mapped = map_positional_args(parsed['arguments'], ['divisor', 'dividend'])
# {'divisor': 10, 'dividend': 2}  # Should be reversed
```

**Solution**: Ensure `param_names` matches function signature order.

```python
# Correct order
mapped = map_positional_args(parsed['arguments'], ['dividend', 'divisor'])
# {'dividend': 10, 'divisor': 2}  # Correct
```

### Pitfall 4: Schema Mismatch

**Issue**: Nesting arguments with wrong schema produces unexpected structure.

```python
class WrongSchema(BaseModel):
    query: str
    # Missing 'options' nested field

flat_args = {'query': 'AI', 'limit': 10}
nested = nest_arguments_by_schema(flat_args, WrongSchema)
# {'query': 'AI', 'limit': 10}  # 'limit' not nested (no nested fields detected)

# Validation fails
WrongSchema(**nested)
# ValidationError: Extra inputs are not permitted: limit
```

**Solution**: Ensure schema matches expected argument structure.

```python
class CorrectSchema(BaseModel):
    query: str
    limit: int

nested = nest_arguments_by_schema(flat_args, CorrectSchema)
# {'query': 'AI', 'limit': 10}

CorrectSchema(**nested)  # Works
```

### Pitfall 5: Complex Expressions in Arguments

**Issue**: Only literal values supported in arguments.

```python
# Raises ValueError (function call in argument)
parse_function_call('search(query=get_query())')
# ValueError: Invalid function call syntax

# Raises ValueError (variable reference)
parse_function_call('search(query=user_input)')
# ValueError: Invalid function call syntax
```

**Solution**: Use literal values only (strings, numbers, lists, dicts, booleans, None).

```python
parse_function_call('search(query="AI research")')  # OK
parse_function_call('config(settings={"debug": true})')  # OK
```

## Design Rationale

### Why AST Parsing Instead of eval()?

Using `ast.parse()` and `ast.literal_eval()` provides security and safety:

1. **No Code Execution**: AST parsing analyzes syntax without executing code
2. **Literal Values Only**: `ast.literal_eval()` restricts to safe literal types
3. **Syntax Validation**: AST parser catches invalid Python syntax before processing
4. **No Side Effects**: Cannot call functions, access variables, or execute arbitrary code

Alternative (`eval()`) would enable arbitrary code execution - a critical security risk.

### Why Separate Positional Mapping Step?

Positional argument mapping is separated from parsing because:

1. **Schema Agnostic Parsing**: Parser works without schema knowledge
2. **Flexible Integration**: Different schemas can be applied to same parsed result
3. **Clear Error Messages**: Mapping errors separate from syntax errors
4. **Reusable Parsing**: Parsed result can be mapped to different parameter orders

This separation follows single-responsibility principle.

### Why Schema-Based Nesting?

Automatic nesting based on schema provides:

1. **User-Friendly Input**: Users provide flat arguments naturally
2. **Schema Compliance**: Output matches Pydantic model structure automatically
3. **Union Support**: Handles union types by collecting all possible fields
4. **Validation Delegation**: Unknown fields preserved for Pydantic validation errors

Alternative (manual nesting) would require users to match exact schema structure in input.

### Why Preserve Unknown Fields?

Unknown fields remain at top level rather than being discarded because:

1. **Better Error Messages**: Pydantic validation identifies specific unknown fields
2. **Debugging Support**: Developers see which fields are unrecognized
3. **Forward Compatibility**: Schema evolution doesn't silently drop new fields
4. **Explicit Validation**: Errors happen at validation time with context

Discarding unknown fields silently would hide configuration mistakes.

## See Also

- **Related Modules**:
  - [TypeScript Schema Handler](./typescript.md): TypeScript notation for schemas
  - [Schema to Model](./schema_to_model.md): Dynamic Pydantic model generation
  - [Spec](../../types/spec.md): Validation framework using Pydantic models

## Examples

### Example 1: Complete MCP Tool Pipeline

> **Tutorial Available**: See [Tutorial #94](https://github.com/khive-ai/lionherd-core/issues/94) for a complete end-to-end example of building an MCP tool pipeline with nested schemas, showing the parse → map → nest workflow pattern.

### Example 2: Batch Tool Invocations

```python
from lionherd_core.libs.schema_handlers import (
    parse_function_call,
    map_positional_args
)

# Batch of user inputs
user_inputs = [
    'search("AI", limit=10)',
    'create("user", name="Alice")',
    'update("user_123", status="active")'
]

# Parse all
parsed_calls = [parse_function_call(inp) for inp in user_inputs]

# Process each
for parsed in parsed_calls:
    tool = parsed['tool']
    args = parsed['arguments']

    # Different parameter names per tool
    param_map = {
        'search': ['query'],
        'create': ['entity_type'],
        'update': ['id']
    }

    mapped = map_positional_args(args, param_map.get(tool, []))
    print(f"{tool}: {mapped}")

# Output:
# search: {'query': 'AI', 'limit': 10}
# create: {'entity_type': 'user', 'name': 'Alice'}
# update: {'id': 'user_123', 'status': 'active'}
```

### Example 3: Error Handling

```python
from lionherd_core.libs.schema_handlers import parse_function_call

def safe_parse(call_str: str):
    """Parse with comprehensive error handling."""
    try:
        result = parse_function_call(call_str)
        return result, None
    except ValueError as e:
        return None, f"Invalid function call: {e}"
    except SyntaxError as e:
        return None, f"Syntax error: {e}"

# Test cases
test_cases = [
    'search("AI")',              # Valid
    'not a function',            # Invalid
    'search(**kwargs)',          # Unsupported
    'search(key=value',          # Syntax error
]

for test in test_cases:
    result, error = safe_parse(test)
    if error:
        print(f"ERROR - {test}: {error}")
    else:
        print(f"OK - {test}: {result}")

# Output:
# OK - search("AI"): {'tool': 'search', 'arguments': {'_pos_0': 'AI'}}
# ERROR - not a function: Invalid function call: Not a function call
# ERROR - search(**kwargs): Invalid function call: **kwargs not supported
# ERROR - search(key=value: Syntax error: unexpected EOF while parsing
```

### Example 4: Dynamic Schema Selection

> **Tutorial Available**: See [Tutorial #95](https://github.com/khive-ai/lionherd-core/issues/95) for a production-ready example of implementing schema registry and dynamic routing patterns for multi-tool MCP systems.

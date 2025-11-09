# TypeScript Schema Handler

> Convert JSON Schema to TypeScript-style notation for optimal LLM comprehension

## Overview

The TypeScript schema handler provides utilities to convert JSON Schema definitions into TypeScript-style type notation, optimized for LLM tool use documentation. This conversion makes schema specifications more readable and familiar to developers while maintaining semantic accuracy.

**Key Capabilities:**

- **Type Mapping**: Converts JSON Schema types to TypeScript-like types (`string`, `int`, `float`, `bool`)
- **Enum Unions**: Formats enum values as TypeScript literal unions (`"value1" | "value2" | null`)
- **Complex Types**: Handles arrays, objects, unions (`anyOf`), and schema references (`$ref`)
- **Optional Fields**: Marks optional fields with `?` suffix and detects nullable types
- **Default Values**: Includes default value annotations in generated output
- **Inline Descriptions**: Appends field descriptions for inline documentation

**When to Use TypeScript Handler:**

- Generating tool documentation for LLM function calling
- Creating human-readable schema references
- Documenting API parameters in MCP server tools
- Converting Pydantic models to TypeScript-style signatures

**Output Format:**

```typescript
field_name?: type = default - description
```

## Module Functions

### `typescript_schema()`

Main entry point for converting JSON Schema to TypeScript-style notation.

**Signature:**

```python
from lionherd_core.libs.schema_handlers import typescript_schema

def typescript_schema(schema: dict, indent: int = 0) -> str: ...
```

**Parameters:**

- `schema` (dict): JSON Schema object with `properties` and optional `required` fields
- `indent` (int, default 0): Indentation level (each level = 2 spaces)

**Returns:**

- str: TypeScript-style formatted schema with one field per line

**Raises:**

- Silently returns empty string if `properties` not present in schema

**Examples:**

```python
>>> from lionherd_core.libs.schema_handlers import typescript_schema

# Simple schema
>>> schema = {
...     "properties": {
...         "name": {"type": "string", "description": "User name"},
...         "age": {"type": "integer", "default": 0}
...     },
...     "required": ["name"]
... }
>>> print(typescript_schema(schema))
name: string - User name
age?: int = 0

# Enum field
>>> schema = {
...     "properties": {
...         "status": {"enum": ["active", "inactive", "pending"]}
...     },
...     "required": ["status"]
... }
>>> print(typescript_schema(schema))
status: "active" | "inactive" | "pending"

# Array with items
>>> schema = {
...     "properties": {
...         "tags": {
...             "type": "array",
...             "items": {"type": "string"},
...             "description": "Tag list"
...         }
...     }
... }
>>> print(typescript_schema(schema))
tags?: string[] - Tag list

# Union type (anyOf)
>>> schema = {
...     "properties": {
...         "value": {
...             "anyOf": [
...                 {"type": "string"},
...                 {"type": "integer"},
...                 {"type": "null"}
...             ]
...         }
...     }
... }
>>> print(typescript_schema(schema))
value?: string | int | null

# Schema reference
>>> schema = {
...     "properties": {
...         "config": {"$ref": "#/definitions/Config"}
...     },
...     "required": ["config"]
... }
>>> print(typescript_schema(schema))
config: Config

# With indentation
>>> schema = {
...     "properties": {
...         "nested": {"type": "object"}
...     }
... }
>>> print(typescript_schema(schema, indent=2))
    nested?: object
```

**Notes:**

This function processes top-level properties only. For nested object schemas, recursively call `typescript_schema()` on nested property definitions with incremented `indent`.

## Usage Patterns

### Basic Schema Conversion

```python
from lionherd_core.libs.schema_handlers import typescript_schema

# Define JSON Schema
schema = {
    "properties": {
        "user_id": {
            "type": "string",
            "description": "Unique user identifier"
        },
        "email": {
            "type": "string",
            "description": "User email address"
        },
        "age": {
            "type": "integer",
            "default": 0,
            "description": "User age in years"
        },
        "role": {
            "enum": ["admin", "user", "guest"],
            "description": "User role"
        }
    },
    "required": ["user_id", "email"]
}

# Convert to TypeScript notation
ts_notation = typescript_schema(schema)
print(ts_notation)
# Output:
# user_id: string - Unique user identifier
# email: string - User email address
# age?: int = 0 - User age in years
# role?: "admin" | "user" | "guest" - User role
```

### MCP Tool Documentation

```python
from lionherd_core.libs.schema_handlers import typescript_schema

# Tool schema from MCP server
tool_schema = {
    "properties": {
        "action": {
            "enum": ["search", "get", "list"],
            "description": "Operation to perform"
        },
        "query": {
            "type": "string",
            "description": "Search query string"
        },
        "limit": {
            "type": "integer",
            "default": 10,
            "description": "Maximum results"
        },
        "filters": {
            "type": "object",
            "description": "Optional filters"
        }
    },
    "required": ["action"]
}

# Generate TypeScript-style docs
ts_docs = typescript_schema(tool_schema)

# Use in tool description
tool_description = f"""
Search tool with the following parameters:

{ts_docs}

Usage: search(action="search", query="...", limit=10)
"""
```

### Complex Nested Types

```python
from lionherd_core.libs.schema_handlers import typescript_schema

# Schema with arrays, refs, and unions
schema = {
    "properties": {
        "items": {
            "type": "array",
            "items": {"$ref": "#/definitions/Item"},
            "description": "List of items"
        },
        "metadata": {
            "anyOf": [
                {"type": "object"},
                {"type": "null"}
            ],
            "description": "Optional metadata"
        },
        "status": {
            "anyOf": [
                {"enum": ["active", "inactive"]},
                {"type": "null"}
            ],
            "default": None,
            "description": "Current status"
        }
    },
    "required": ["items"]
}

ts_output = typescript_schema(schema)
print(ts_output)
# Output:
# items: Item[] - List of items
# metadata?: object | null - Optional metadata
# status?: "active" | "inactive" | null = null - Current status
```

### Indented Output for Nested Schemas

```python
from lionherd_core.libs.schema_handlers import typescript_schema

def format_nested_schema(schema: dict, indent: int = 0) -> str:
    """Recursively format nested object schemas."""
    lines = []

    for field_name, field_spec in schema.get("properties", {}).items():
        if field_spec.get("type") == "object" and "properties" in field_spec:
            # Parent object field
            prefix = "  " * indent
            lines.append(f"{prefix}{field_name}: {{")

            # Nested properties
            nested = typescript_schema(field_spec, indent + 1)
            lines.append(nested)

            lines.append(f"{prefix}}}")
        else:
            # Regular field
            single_field_schema = {
                "properties": {field_name: field_spec},
                "required": schema.get("required", [])
            }
            lines.append(typescript_schema(single_field_schema, indent))

    return "\n".join(lines)

# Example nested schema
nested_schema = {
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
    }
}

print(format_nested_schema(nested_schema))
# Output:
# user: {
#   name: string
#   age?: int
# }
```

## Common Pitfalls

### Pitfall 1: Empty Schema Returns Empty String

**Issue**: Passing schema without `properties` returns empty string silently.

```python
>>> typescript_schema({})
''

>>> typescript_schema({"required": ["field"]})  # No properties
''
```

**Solution**: Always validate schema has `properties` before calling:

```python
if "properties" in schema:
    result = typescript_schema(schema)
else:
    result = "# No properties defined"
```

### Pitfall 2: Nested Objects Not Recursively Processed

**Issue**: Nested object schemas show as `object` without expanding properties.

```python
>>> schema = {
...     "properties": {
...         "config": {
...             "type": "object",
...             "properties": {
...                 "enabled": {"type": "boolean"}
...             }
...         }
...     }
... }
>>> typescript_schema(schema)
'config?: object'  # Doesn't expand nested properties
```

**Solution**: Manually recurse or use custom formatter (see "Indented Output" pattern above).

### Pitfall 3: Schema References Not Resolved

**Issue**: `$ref` values extracted as-is without following references.

```python
>>> schema = {
...     "properties": {
...         "item": {"$ref": "#/definitions/Item"}
...     }
... }
>>> typescript_schema(schema)
'item?: Item'  # Reference name extracted, not resolved
```

**Solution**: This is expected behavior. Handler extracts reference name for documentation. If full resolution needed, pre-process schema with JSON Schema resolver.

### Pitfall 4: Default Values Not Type-Checked

**Issue**: Default values formatted as-is without validation against field type.

```python
# Invalid default (string for integer field)
>>> schema = {
...     "properties": {
...         "count": {"type": "integer", "default": "invalid"}
...     }
... }
>>> typescript_schema(schema)
'count?: int = "invalid"'  # Type mismatch not caught
```

**Solution**: Handler focuses on documentation, not validation. Validate schema separately using JSON Schema validators before conversion.

## Design Rationale

### Why TypeScript Notation Over JSON Schema?

**Readability**: TypeScript syntax is more concise and familiar to developers:

```typescript
// TypeScript notation (generated)
name: string
age?: int = 0
status: "active" | "inactive"

// vs JSON Schema (input)
{
  "name": {"type": "string"},
  "age": {"type": "integer", "default": 0},
  "status": {"enum": ["active", "inactive"]}
}
```

**LLM Comprehension**: Modern LLMs are extensively trained on TypeScript documentation and understand the type syntax patterns better than verbose JSON Schema.

### Why Optional Marker (`?`) Over Explicit Flags?

TypeScript's `?` suffix provides visual clarity for optional fields at a glance:

```typescript
// Clear optionality
required_field: string
optional_field?: string

// vs verbose alternative
required_field: string (required)
optional_field: string (optional)
```

This matches developer expectations from TypeScript/JavaScript ecosystems.

### Why Not Full TypeScript Conversion?

This handler generates TypeScript-**style** notation, not valid TypeScript code:

- Uses `int`/`float` instead of TypeScript's `number` (more precise for LLM understanding)
- Doesn't generate interface/type declarations (focuses on inline documentation)
- Handles JSON Schema features without TypeScript equivalents (like `anyOf` → union translation)

**Use Case**: Tool documentation, not code generation. For actual TypeScript types, use dedicated schema-to-TypeScript converters.

### Why Inline Descriptions?

Appending descriptions directly to field definitions keeps documentation compact:

```typescript
name: string - User's full name
age?: int = 0 - Age in years
```

vs separated format:

```typescript
name: string
// User's full name

age?: int = 0
// Age in years
```

This improves readability in constrained contexts (LLM context windows, terminal output).

## See Also

- **Related Modules**:
  - [Schema to Model](./schema_to_model.md): Dynamic Pydantic model generation
  - [Function Call Parser](./function_call_parser.md): Parsing LLM function calls
  - [Spec](../../types/spec.md): Validation framework using Pydantic models
- **External Resources**:
  - [JSON Schema Specification](https://json-schema.org/specification.html)
  - [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)

## Examples

### Example 1: Simple API Parameters

```python
from lionherd_core.libs.schema_handlers import typescript_schema

# API endpoint schema
api_schema = {
    "properties": {
        "endpoint": {
            "type": "string",
            "description": "API endpoint path"
        },
        "method": {
            "enum": ["GET", "POST", "PUT", "DELETE"],
            "description": "HTTP method"
        },
        "headers": {
            "type": "object",
            "description": "Request headers"
        },
        "timeout": {
            "type": "number",
            "default": 30.0,
            "description": "Timeout in seconds"
        }
    },
    "required": ["endpoint", "method"]
}

print(typescript_schema(api_schema))
# Output:
# endpoint: string - API endpoint path
# method: "GET" | "POST" | "PUT" | "DELETE" - HTTP method
# headers?: object - Request headers
# timeout?: float = 30.0 - Timeout in seconds
```

### Example 2: Database Query Builder

```python
from lionherd_core.libs.schema_handlers import typescript_schema

query_schema = {
    "properties": {
        "table": {
            "type": "string",
            "description": "Table name"
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to select"
        },
        "where": {
            "type": "object",
            "description": "WHERE clause conditions"
        },
        "limit": {
            "type": "integer",
            "default": 100,
            "description": "Result limit"
        },
        "offset": {
            "type": "integer",
            "default": 0,
            "description": "Result offset"
        }
    },
    "required": ["table"]
}

docs = typescript_schema(query_schema)
print(f"Query Parameters:\n{docs}")
# Output:
# Query Parameters:
# table: string - Table name
# columns?: string[] - Columns to select
# where?: object - WHERE clause conditions
# limit?: int = 100 - Result limit
# offset?: int = 0 - Result offset
```

### Example 3: MCP Tool with Complex Types

```python
from lionherd_core.libs.schema_handlers import typescript_schema

mcp_tool_schema = {
    "properties": {
        "action": {
            "enum": ["create", "read", "update", "delete"],
            "description": "CRUD operation"
        },
        "resource_id": {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ],
            "description": "Resource identifier"
        },
        "data": {
            "anyOf": [
                {"$ref": "#/definitions/CreateData"},
                {"$ref": "#/definitions/UpdateData"},
                {"type": "null"}
            ],
            "description": "Operation payload"
        },
        "options": {
            "type": "object",
            "description": "Additional options"
        }
    },
    "required": ["action"]
}

ts_docs = typescript_schema(mcp_tool_schema)

# Use in MCP tool registration
tool_definition = {
    "name": "resource_manager",
    "description": f"Manage resources with CRUD operations.\n\nParameters:\n{ts_docs}",
    "inputSchema": mcp_tool_schema
}
```

### Example 4: Nullable Fields and Defaults

```python
from lionherd_core.libs.schema_handlers import typescript_schema

config_schema = {
    "properties": {
        "api_key": {
            "type": "string",
            "description": "API authentication key"
        },
        "base_url": {
            "type": "string",
            "default": "https://api.example.com",
            "description": "Base API URL"
        },
        "retry_attempts": {
            "type": "integer",
            "default": 3,
            "description": "Retry attempts on failure"
        },
        "proxy": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ],
            "default": None,
            "description": "Optional proxy URL"
        },
        "debug": {
            "type": "boolean",
            "default": False,
            "description": "Enable debug logging"
        }
    },
    "required": ["api_key"]
}

print(typescript_schema(config_schema))
# Output:
# api_key: string - API authentication key
# base_url?: string = "https://api.example.com" - Base API URL
# retry_attempts?: int = 3 - Retry attempts on failure
# proxy?: string | null = null - Optional proxy URL
# debug?: bool = false - Enable debug logging
```

### Example 5: Array Types

```python
from lionherd_core.libs.schema_handlers import typescript_schema

data_schema = {
    "properties": {
        "string_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of strings"
        },
        "int_array": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Array of integers"
        },
        "ref_list": {
            "type": "array",
            "items": {"$ref": "#/definitions/User"},
            "description": "User objects"
        },
        "enum_array": {
            "type": "array",
            "items": {
                "enum": ["read", "write", "execute"]
            },
            "description": "Permissions"
        },
        "any_array": {
            "type": "array",
            "description": "Untyped array"
        }
    },
    "required": ["string_list"]
}

print(typescript_schema(data_schema))
# Output:
# string_list: string[] - List of strings
# int_array?: int[] - Array of integers
# ref_list?: User[] - User objects
# enum_array?: ("read" | "write" | "execute")[] - Permissions
# any_array?: any[] - Untyped array
```

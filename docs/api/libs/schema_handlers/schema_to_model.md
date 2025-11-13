# Schema to Model Conversion

> Dynamic Pydantic model generation from JSON Schema at runtime

## Overview

The `schema_to_model` module provides **runtime Pydantic model generation** from JSON Schema definitions. It uses `datamodel-code-generator` to create Python code from schemas, dynamically imports the generated models, and rebuilds them with proper type resolution for immediate use.

**Key Capabilities:**

- **Dynamic Model Generation**: Convert JSON Schema to Pydantic BaseModel classes at runtime
- **Auto-Import**: Generated code is compiled and imported automatically (no file persistence)
- **Type Resolution**: Models are rebuilt with proper type namespace resolution
- **Flexible Input**: Accepts JSON Schema as string or dict
- **Python Version Detection**: Auto-detects Python version for code generation
- **Model Name Extraction**: Intelligently extracts model name from schema title or uses provided default

**When to Use:**

- Runtime model creation from API responses with JSON Schema
- Dynamic validation against evolving schemas
- Tool calling with structured outputs from LLM providers
- Schema-driven form generation
- Type-safe deserialization of external data sources

**When NOT to Use:**

- Static schemas known at development time (use Pydantic models directly)
- High-frequency operations (generation overhead ~100-300ms)
- Production hot paths requiring microsecond latency
- Simple validation where `TypedDict` or basic dicts suffice

**Dependencies:**

Requires optional dependency: `pip install 'lionherd-core[schema-gen]'` or `pip install datamodel-code-generator`

## Function Signature

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

def load_pydantic_model_from_schema(
    schema: str | dict[str, Any],
    model_name: str = "DynamicModel",
    /,
    pydantic_version: Any = None,
    python_version: Any = None,
) -> type[BaseModel]:
    """Generate Pydantic model dynamically from JSON schema.

    Creates model class via datamodel-code-generator, imports it,
    and rebuilds with proper type resolution.
    """
```

## Parameters

### Required Parameters

**schema** : str or dict[str, Any]

JSON Schema definition (as JSON string or Python dict).

- **Type coercion**: Dicts are serialized to JSON internally
- **Validation**: Must be valid JSON Schema format
- **Title extraction**: If schema contains `"title"` field, it's used as model name (sanitized)
- **Raises**: `ValueError` if invalid JSON string, `TypeError` if not str/dict

**Examples:**

```python
# Dict format
schema = {
    "title": "User",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    },
    "required": ["name"]
}

# JSON string format
schema = '{"title": "User", "type": "object", ...}'
```

**model_name** : str, default="DynamicModel"

Base name for generated model class (fallback if schema has no title).

- **Schema title precedence**: If schema contains `"title"`, it takes priority
- **Sanitization**: Non-alphanumeric characters removed, must start with letter
- **Validation**: Raises `ValueError` if cannot sanitize to valid Python identifier
- **Default**: `"DynamicModel"` used if no schema title provided

**Examples:**

```python
# Schema title takes precedence
schema = {"title": "UserProfile", "type": "object", ...}
Model = load_pydantic_model_from_schema(schema, model_name="Ignored")
# Model.__name__ == "UserProfile" (schema title wins)

# Fallback to model_name
schema = {"type": "object", ...}  # No title
Model = load_pydantic_model_from_schema(schema, model_name="CustomModel")
# Model.__name__ == "CustomModel"
```

### Optional Parameters

**pydantic_version** : DataModelType enum, optional

Pydantic version target for code generation.

- **Type**: `datamodel_code_generator.DataModelType` enum
- **Default**: `DataModelType.PydanticV2BaseModel` (Pydantic v2)
- **Common values**:
  - `DataModelType.PydanticV2BaseModel` - Pydantic v2 (recommended)
  - `DataModelType.PydanticBaseModel` - Pydantic v1 (legacy)
- **Use cases**: Generate v1-compatible models for legacy systems

**python_version** : PythonVersion enum, optional

Python version target for code generation.

- **Type**: `datamodel_code_generator.PythonVersion` enum
- **Default**: Auto-detected from `sys.version_info` (PY_311, PY_312, PY_313, PY_314)
- **Fallback**: `PythonVersion.PY_312` if version not in map
- **Common values**:
  - `PythonVersion.PY_311` - Python 3.11
  - `PythonVersion.PY_312` - Python 3.12 (default fallback)
  - `PythonVersion.PY_313` - Python 3.13
  - `PythonVersion.PY_314` - Python 3.14
- **Use cases**: Generate code for different Python runtime environments

## Returns

**type[BaseModel]**

Dynamically generated Pydantic model class ready for instantiation.

- **Type safety**: Fully type-hinted model with runtime validation
- **Subclass**: Inherits from `pydantic.BaseModel`
- **Validation**: Model is rebuilt with proper type resolution before return
- **Instantiation**: Can be used immediately: `instance = Model(**data)`

## Raises

### ImportError

`datamodel-code-generator` not installed.

- **Message**: Includes installation instructions
- **Fix**: `pip install 'lionherd-core[schema-gen]'` or `pip install datamodel-code-generator`

### ValueError

Invalid schema format or content.

- **Causes**:
  - Invalid JSON string provided
  - Invalid dictionary for schema
  - Model name cannot be sanitized to valid identifier
- **Context**: Error includes original exception details

### TypeError

Schema is neither string nor dict.

- **Message**: "Schema must be a JSON string or a dictionary"

### RuntimeError

Code generation or module loading failed.

- **Causes**:
  - `datamodel-code-generator.generate()` failed
  - Generated file not created
  - Module import failed
  - Type resolution failed during `model_rebuild()`
- **Context**: Error includes underlying exception and file path details

### AttributeError

Generated model class not found in module.

- **Causes**:
  - Neither specified `model_name` nor fallback `"Model"` found
  - Found class is not a Pydantic BaseModel
- **Message**: Lists available BaseModel classes in generated module for debugging

## Usage Patterns

### Basic Usage

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Define JSON Schema
schema = {
    "title": "User",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string"}
    },
    "required": ["name", "age"]
}

# Generate model
UserModel = load_pydantic_model_from_schema(schema)

# Use like normal Pydantic model
user = UserModel(name="Alice", age=30, email="alice@example.com")
print(user.name)  # "Alice"
print(user.model_dump())  # {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# Validation works
try:
    invalid = UserModel(name="Bob")  # Missing required 'age'
except ValueError as e:
    print(e)  # Pydantic validation error
```

### JSON String Input

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Schema as JSON string (from API response, file, etc.)
schema_json = '''
{
    "title": "Product",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"}
    },
    "required": ["id", "price"]
}
'''

ProductModel = load_pydantic_model_from_schema(schema_json)
product = ProductModel(id="ABC123", price=29.99, in_stock=True)
```

### Custom Model Name

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Schema without title
schema = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "value": {"type": "integer"}
    }
}

# Provide custom name
ConfigModel = load_pydantic_model_from_schema(schema, model_name="ConfigModel")
print(ConfigModel.__name__)  # "ConfigModel"

config = ConfigModel(key="timeout", value=30)
```

### LLM Tool Calling

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Get tool schema from LLM provider
tool_schema = {
    "title": "WeatherQuery",
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
}

# Generate model for structured output parsing
WeatherQuery = load_pydantic_model_from_schema(tool_schema)

# Parse LLM response
llm_output = {"location": "San Francisco", "unit": "celsius"}
query = WeatherQuery(**llm_output)  # Type-safe validation

# Use in tool execution
def get_weather(query: WeatherQuery) -> dict:
    return {"temp": 18, "unit": query.unit, "location": query.location}
```

### Nested Schema Handling

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Nested object schema
schema = {
    "title": "Order",
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name"]
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product": {"type": "string"},
                    "quantity": {"type": "integer"}
                }
            }
        }
    },
    "required": ["order_id", "customer"]
}

OrderModel = load_pydantic_model_from_schema(schema)

# Nested validation works
order = OrderModel(
    order_id="ORD-123",
    customer={"name": "Alice", "email": "alice@example.com"},
    items=[
        {"product": "Widget", "quantity": 2},
        {"product": "Gadget", "quantity": 1}
    ]
)

print(order.customer)  # Customer(name='Alice', email='alice@example.com')
print(order.items[0].product)  # "Widget"
```

### Schema Evolution

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Schema version 1
schema_v1 = {
    "title": "APIResponse",
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "data": {"type": "object"}
    }
}

ModelV1 = load_pydantic_model_from_schema(schema_v1)

# Schema version 2 (added field)
schema_v2 = {
    "title": "APIResponse",
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "data": {"type": "object"},
        "timestamp": {"type": "number"}  # New field
    }
}

ModelV2 = load_pydantic_model_from_schema(schema_v2)

# Both models coexist
response_v1 = ModelV1(status="ok", data={})
response_v2 = ModelV2(status="ok", data={}, timestamp=1699438200)
```

## Common Pitfalls

### Pitfall 1: Missing Optional Dependency

**Issue**: Import error when `datamodel-code-generator` not installed.

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

schema = {"type": "object", "properties": {"key": {"type": "string"}}}
Model = load_pydantic_model_from_schema(schema)
# ImportError: datamodel-code-generator not installed...
```

**Solution**: Install optional dependency via extras or directly.

```bash
# Via extras (recommended)
pip install 'lionherd-core[schema-gen]'

# Direct install
pip install datamodel-code-generator
```

### Pitfall 2: Invalid Schema Format

**Issue**: Schema doesn't conform to JSON Schema specification.

```python
schema = {
    "type": "object",
    "properties": {
        "age": {"type": "number", "minimum": "18"}  # Should be int, not str
    }
}

Model = load_pydantic_model_from_schema(schema)
# May generate incorrect model or fail validation
```

**Solution**: Validate schema against JSON Schema spec before generation.

```python
import jsonschema

# Validate schema first
jsonschema.Draft7Validator.check_schema(schema)  # Raises if invalid
Model = load_pydantic_model_from_schema(schema)
```

### Pitfall 3: Non-Sanitizable Model Names

**Issue**: Schema title contains characters that can't be converted to Python identifier.

```python
schema = {
    "title": "123-Invalid!",  # Starts with digit, contains special chars
    "type": "object"
}

Model = load_pydantic_model_from_schema(schema)
# ValueError: Cannot extract valid Python identifier from: '123-Invalid!'
```

**Solution**: Provide valid `model_name` or fix schema title.

```python
# Option 1: Provide fallback name
Model = load_pydantic_model_from_schema(schema, model_name="ValidModel")

# Option 2: Fix schema title
schema["title"] = "ValidModel"
Model = load_pydantic_model_from_schema(schema)
```

### Pitfall 4: Performance in Hot Paths

**Issue**: Generating models repeatedly in performance-critical code.

```python
# Example schema for user data
USER_SCHEMA = {
    "title": "User",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
    },
}

# BAD: Generating model on every request
def handle_request(data: dict):
    Model = load_pydantic_model_from_schema(USER_SCHEMA)  # 100-300ms overhead
    return Model(**data)

# Called 1000x/sec → massive overhead
```

**Solution**: Cache generated models by schema hash.

```python
from functools import lru_cache
import json

@lru_cache(maxsize=128)
def get_cached_model(schema_json: str):
    return load_pydantic_model_from_schema(schema_json)

# Convert schema to normalized JSON string for caching
SCHEMA_JSON = json.dumps(USER_SCHEMA, sort_keys=True)

def handle_request(data: dict):
    Model = get_cached_model(SCHEMA_JSON)  # Fast lookup after first call
    return Model(**data)
```

### Pitfall 5: Assuming Model Persistence

**Issue**: Expecting generated models to persist across processes.

```python
# Process 1
Model = load_pydantic_model_from_schema(schema)
pickle.dump(Model, open("model.pkl", "wb"))

# Process 2
Model = pickle.load(open("model.pkl", "rb"))
# PicklingError: Can't pickle dynamically created classes
```

**Solution**: Share schema, not model. Regenerate in each process.

```python
# noqa:validation
# Process 1
import json
schema = {"type": "object", ...}
json.dump(schema, open("schema.json", "w"))

# Process 2
schema = json.load(open("schema.json"))
Model = load_pydantic_model_from_schema(schema)  # Regenerate
```

## Design Rationale

### Why Runtime Generation?

Static Pydantic models require schemas known at development time. Runtime generation enables:

1. **Dynamic APIs**: Handle evolving API schemas without redeployment
2. **LLM Integration**: Create models from LLM-provided tool schemas
3. **Schema-Driven UIs**: Generate forms and validators from backend schemas
4. **Multi-Tenant Systems**: Different schemas per tenant without code duplication

**Trade-off**: ~100-300ms generation overhead vs. development-time compilation. Cache models to amortize cost.

### Why datamodel-code-generator?

Alternatives considered:

1. **`pydantic.create_model()`**: Doesn't support complex schemas (nested objects, $ref, allOf)
2. **Manual parsing**: Error-prone, doesn't handle JSON Schema edge cases
3. **datamodel-code-generator**: Battle-tested, handles full JSON Schema spec, generates idiomatic Pydantic code

**Trade-off**: External dependency vs. robust schema support. Optional dependency minimizes impact.

### Why Temporary Files?

Generated code is written to temporary files, imported, then deleted because:

1. **Import System**: Python's import machinery requires file-backed modules for reliable `spec_from_file_location()`
2. **Type Resolution**: `model_rebuild()` needs proper module namespace for forward references
3. **Clean Namespaces**: Temporary modules prevent global namespace pollution

**Trade-off**: File I/O overhead (~5-10ms) vs. reliable import semantics. Negligible compared to generation time.

### Why model_rebuild()?

After importing, models are rebuilt with `model_rebuild(_types_namespace=module.__dict__, force=True)` because:

1. **Forward References**: Generic types and self-references need resolved namespace
2. **Nested Models**: Child models must be in scope for parent validation
3. **Type Safety**: Ensures runtime validation matches schema intent

**Example needing rebuild:**

```python
# Generated code has forward reference
class Parent(BaseModel):
    children: list['Child']  # String annotation

class Child(BaseModel):
    name: str

# Without rebuild: NameError when validating Parent
# With rebuild: 'Child' resolved to actual class
```

## Internal Functions

The module provides these internal helper functions (not intended for direct use):

### `_get_python_version_enum()`

Auto-detects Python version from `sys.version_info` and maps to `datamodel_code_generator.PythonVersion` enum.

**Mapping:**

- Python 3.11 → `PY_311`
- Python 3.12 → `PY_312` (default fallback)
- Python 3.13 → `PY_313`
- Python 3.14 → `PY_314`

### `_sanitize_model_name()`

Extracts valid Python identifier from string by removing non-alphanumeric characters.

**Rules:**

- Keeps only letters, digits, underscores
- Must start with letter
- Raises `ValueError` if no valid identifier can be extracted

### `_extract_model_name_from_schema()`

Attempts to extract model name from schema `"title"` field, falling back to provided default.

### `_prepare_schema_input()`

Converts schema (str or dict) to JSON string and extracts final model name.

### `_generate_model_code()`

Invokes `datamodel_code_generator.generate()` with proper configuration.

### `_load_generated_module()`

Dynamically imports Python module from generated file using `importlib.util.spec_from_file_location()`.

### `_extract_model_class()`

Finds Pydantic BaseModel class in generated module by name, falling back to `"Model"` if not found.

### `_rebuild_model()`

Rebuilds model with proper type namespace resolution using `model_rebuild()`.

## See Also

- **Related Modules**:
  - [Spec](../../types/spec.md): Validation framework using Pydantic models
  - [Function Call Parser](./function_call_parser.md): Parsing LLM function calls
  - [TypeScript Schema Handler](./typescript.md): TypeScript notation for schemas
- **Related Guides**:
  - [Pydantic Documentation](https://docs.pydantic.dev/): Pydantic model usage
  - [JSON Schema](https://json-schema.org/): JSON Schema specification
  - [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator): Code generator documentation
- **Related Patterns**:
  - Caching generated models with `functools.lru_cache`
  - Schema versioning and migration strategies
  - LLM tool calling with structured outputs

## Examples

### Example 1: API Response Parsing

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Define API response schema
schema = {
    "title": "User",
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string"},
        "email": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
    },
    "required": ["id", "name", "email"],
}

# Generate model from schema
UserModel = load_pydantic_model_from_schema(schema)

# Parse API data (e.g., from requests/httpx response)
user_data = {
    "id": 123,
    "name": "Alice Smith",
    "email": "alice@example.com",
    "created_at": "2025-11-09T12:00:00Z",
}

# Validate with generated model
user = UserModel(**user_data)
print(f"User: {user.name}, Email: {user.email}")
```

### Example 2: Configuration Validation

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema
import json

# Load config schema
with open("config_schema.json") as f:
    schema = json.load(f)

ConfigModel = load_pydantic_model_from_schema(schema)

# Validate user config
with open("user_config.json") as f:
    user_config = json.load(f)

try:
    config = ConfigModel(**user_config)
    print("Config valid!")
except ValueError as e:
    print(f"Config validation failed: {e}")
```

### Example 3: Multi-Schema System

```python
# noqa:validation
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=100)
def get_model_for_schema(schema_hash: str, schema_json: str):
    """Cache models by schema hash."""
    return load_pydantic_model_from_schema(schema_json)

def validate_data(data: dict, schema: dict):
    """Validate data against schema with caching."""
    schema_json = json.dumps(schema, sort_keys=True)
    schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

    Model = get_model_for_schema(schema_hash, schema_json)
    return Model(**data)

# Use with different schemas
user_schema = {"title": "User", "type": "object", ...}
product_schema = {"title": "Product", "type": "object", ...}

user = validate_data(user_data, user_schema)  # First call: generates
user2 = validate_data(user_data2, user_schema)  # Cached: fast
product = validate_data(product_data, product_schema)  # Different schema: generates
```

### Example 4: Testing with Dynamic Fixtures

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema
import pytest

@pytest.fixture
def user_model():
    """Generate User model from schema."""
    schema = {
        "title": "User",
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "username": {"type": "string", "minLength": 3},
            "email": {"type": "string"}
        },
        "required": ["id", "username"]
    }
    return load_pydantic_model_from_schema(schema)

def test_user_validation(user_model):
    """Test user validation with generated model."""
    valid_user = user_model(id=1, username="alice", email="alice@example.com")
    assert valid_user.username == "alice"

    with pytest.raises(ValueError):
        user_model(id=1, username="ab")  # Too short (minLength=3)

    with pytest.raises(ValueError):
        user_model(username="bob")  # Missing required 'id'
```

### Example 5: GraphQL Schema to Model

```python
from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

# Convert GraphQL type to JSON Schema
graphql_type = """
type User {
    id: ID!
    name: String!
    email: String
    age: Int
    posts: [Post!]!
}
"""

# Manually convert to JSON Schema (or use graphql-to-json-schema)
schema = {
    "title": "User",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "email": {"type": "string"},
        "age": {"type": "integer"},
        "posts": {
            "type": "array",
            "items": {"$ref": "#/definitions/Post"}
        }
    },
    "required": ["id", "name", "posts"],
    "definitions": {
        "Post": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"}
            }
        }
    }
}

UserModel = load_pydantic_model_from_schema(schema)

# Use with GraphQL response
user = UserModel(
    id="user_123",
    name="Alice",
    email="alice@example.com",
    age=30,
    posts=[
        {"title": "Post 1", "content": "Content 1"},
        {"title": "Post 2", "content": "Content 2"}
    ]
)
```

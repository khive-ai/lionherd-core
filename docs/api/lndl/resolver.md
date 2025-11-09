# LNDL Resolver

> Core resolution engine for validating and constructing typed outputs from LNDL responses

## Overview

The LNDL Resolver module provides **namespace-aware variable resolution** and **spec-based validation** for Language InterOperable Network Directive Language (LNDL) responses. It transforms structured LLM outputs containing variable declarations (`<lvar>`), action declarations (`<lact>`), and output mappings (`OUT{}`) into validated Pydantic model instances.

**Key Capabilities:**

- **Namespace-Prefixed Resolution**: Maps `Model.field` variable syntax to Pydantic model fields
- **Spec Validation**: Validates output fields against Operable specs (required fields, allowed fields, type matching)
- **Action Integration**: Supports both direct actions (return full model) and namespaced actions (populate specific fields)
- **Scalar Type Handling**: Automatic type conversion for scalar fields (int, float, str, bool)
- **Partial Construction**: Uses `model_construct()` for action-containing models (deferred validation until execution)
- **Error Aggregation**: Collects all validation errors into ExceptionGroup for comprehensive diagnostics

**When to Use LNDL Resolver:**

- Parsing structured LLM responses with variable declarations and type constraints
- Validating LLM outputs against predefined schemas (Operable specs)
- Building workflows requiring type-safe, validated outputs from natural language interactions
- Implementing LNDL-based operations that mix static variables and dynamic actions

**When NOT to Use (Use Alternatives):**

- Simple string parsing without type validation → Use `lndl.parser` directly
- Direct Pydantic model construction from dicts → Use `model.model_validate()`
- Unstructured LLM outputs without LNDL syntax → Use plain text extraction
- Strict immediate validation (no actions) → Use `Operable.invoke()` with standard responses

See [LNDL Parser](parser.md) for raw extraction, [Operable](../types/operable.md) for spec definitions.

## Module Functions

### parse_lndl()

Parse complete LNDL response and validate against operable specs.

**Signature:**

```python
from lionherd_core.lndl.resolver import parse_lndl

def parse_lndl(response: str, operable: Operable) -> LNDLOutput: ...
```

**Parameters:**

- `response` (str): Full LLM response containing `<lvar>`, `<lact>`, and `OUT{}` blocks
- `operable` (Operable): Operable instance with spec definitions for validation

**Returns:**

- LNDLOutput: Validated output with populated fields, parsed actions, and metadata

**Raises:**

- ExceptionGroup: Aggregated validation errors (MissingFieldError, TypeMismatchError, ValueError)
- ValueError: Invalid LNDL syntax, missing variables, or field mismatches

**Examples:**

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Define allowed output spec
operable = Operable(specs=[Spec(name="user", base_type=User)])

# LNDL response from LLM
response = """
<lvar User.name>Alice</lvar>
<lvar User.age>30</lvar>

OUT{
  user: [name, age]
}
"""

# Parse and validate
output = parse_lndl(response, operable)
print(output.fields["user"])
# User(name='Alice', age=30)

print(output.lvars)
# {'name': LvarMetadata(model='User', field='name', local_name='name', value='Alice'),
#  'age': LvarMetadata(model='User', field='age', local_name='age', value='30')}
```

**Workflow:**

1. Extract namespace-prefixed `<lvar>` declarations via `extract_lvars_prefixed()`
2. Extract `<lact>` action declarations via `extract_lacts_prefixed()`
3. Extract and parse `OUT{}` block via `extract_out_block()` + `parse_out_block_array()`
4. Delegate to `resolve_references_prefixed()` for resolution and validation

**See Also:**

- `resolve_references_prefixed()`: Underlying resolution engine (called by this function)
- [extract_lvars_prefixed()](parser.md#extract_lvars_prefixed): Variable extraction
- [extract_lacts_prefixed()](parser.md#extract_lacts_prefixed): Action extraction
- [LNDLOutput](types.md#lndloutput): Return type with validation lifecycle

**Notes:**

This is the **primary high-level API** for LNDL resolution. Most users should call this function rather than `resolve_references_prefixed()` directly, as it handles complete extraction and parsing automatically.

### resolve_references_prefixed()

Resolve namespace-prefixed OUT{} fields and validate against operable specs.

**Signature:**

```python
from lionherd_core.lndl.resolver import resolve_references_prefixed

def resolve_references_prefixed(
    out_fields: dict[str, list[str] | str],
    lvars: dict[str, LvarMetadata],
    lacts: dict[str, LactMetadata],
    operable: Operable,
) -> LNDLOutput: ...
```

**Parameters:**

- `out_fields` (dict[str, list[str] | str]): Parsed OUT{} block mapping field names to variable name lists or literal values
- `lvars` (dict[str, LvarMetadata]): Extracted namespace-prefixed lvar declarations
- `lacts` (dict[str, LactMetadata]): Extracted action declarations with optional namespace
- `operable` (Operable): Operable containing allowed specs for validation

**Returns:**

- LNDLOutput: Validated output containing:
  - `fields`: dict[str, BaseModel | scalar | ActionCall] - Validated field values or action placeholders
  - `lvars`: dict[str, LvarMetadata] - Original lvar metadata
  - `lacts`: dict[str, LactMetadata] - Original lact metadata
  - `actions`: dict[str, ActionCall] - Parsed action calls for execution
  - `raw_out_block`: str - Original OUT{} block for debugging

**Raises:**

- MissingFieldError: Required spec field not present in OUT{}
- TypeMismatchError: Variable's model name doesn't match spec's base_type
- ValueError: Multiple issues including:
  - Name collision between lvars and lacts
  - Variable/action not declared but referenced in OUT{}
  - Field not allowed by operable specs
  - Scalar field using multiple variables (array syntax for single value)
  - Direct action mixed with lvars in BaseModel construction
  - Model name mismatch between variable/action and target spec
  - Invalid function call syntax in actions
  - Type conversion failure for scalar values
  - Pydantic validation errors during model construction
- ExceptionGroup: Aggregates all validation errors from all fields for batch reporting

**Examples:**

#### Example 1: Scalar Field Resolution

```python
from lionherd_core.lndl import resolve_references_prefixed, LvarMetadata
from lionherd_core.types import Operable, Spec

# Parsed inputs
out_fields = {"name": ["username"], "age": "25"}  # Mixed: var reference + literal
lvars = {"username": LvarMetadata(model=None, field=None, local_name="username", value="Alice")}
lacts = {}

# Define allowed specs
operable = Operable(specs=[
    Spec(name="name", base_type=str),
    Spec(name="age", base_type=int),
])

# Resolve
output = resolve_references_prefixed(out_fields, lvars, lacts, operable)
print(output.fields)
# {'name': 'Alice', 'age': 25}
```

#### Example 2: BaseModel Construction from lvars

```python
from lionherd_core.lndl import resolve_references_prefixed, LvarMetadata
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Namespace-prefixed lvars (local_name defaults to field name)
lvars = {
    "name": LvarMetadata(model="User", field="name", local_name="name", value="Alice"),
    "age": LvarMetadata(model="User", field="age", local_name="age", value="30"),
}
out_fields = {"user": ["name", "age"]}  # Reference local_name, not Model.field
lacts = {}

operable = Operable(specs=[Spec(name="user", base_type=User)])

output = resolve_references_prefixed(out_fields, lvars, lacts, operable)
print(output.fields["user"])
# User(name='Alice', age=30)
```

#### Example 3: Direct Action (Returns Full Model)

```python
from lionherd_core.lndl import resolve_references_prefixed, LactMetadata
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Direct action (no namespace - returns entire User model)
lacts = {
    "fetch_user": LactMetadata(
        model=None,  # No namespace = direct action
        field=None,
        local_name="fetch_user",
        call="get_user(user_id=123)",
    )
}
out_fields = {"user": ["fetch_user"]}
lvars = {}

operable = Operable(specs=[Spec(name="user", base_type=User)])

output = resolve_references_prefixed(out_fields, lvars, lacts, operable)
print(output.actions["fetch_user"])
# ActionCall(name='fetch_user', function='get_user', arguments={'user_id': 123}, raw_call='...')
print(output.fields["user"])
# ActionCall(...) - placeholder until execution
```

#### Example 4: Namespaced Actions Mixed with lvars

```python
from lionherd_core.lndl import resolve_references_prefixed, LvarMetadata, LactMetadata
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Mix static lvar with dynamic action for same model
lvars = {
    "name": LvarMetadata(model="User", field="name", local_name="name", value="Alice"),
}
lacts = {
    "get_age": LactMetadata(
        model="User",  # Namespaced: targets User.age
        field="age",
        local_name="get_age",
        call="calculate_age(birth_year=1990)",
    )
}
out_fields = {"user": ["name", "get_age"]}  # Reference local names

operable = Operable(specs=[Spec(name="user", base_type=User)])

output = resolve_references_prefixed(out_fields, lvars, lacts, operable)
# Constructs User with name='Alice' and age=ActionCall(...)
# Caller must execute action and revalidate
```

#### Example 5: Error Aggregation

```python
from lionherd_core.lndl import resolve_references_prefixed, LvarMetadata
from lionherd_core.lndl.errors import MissingFieldError, TypeMismatchError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# Multiple errors: missing required field, type mismatch
lvars = {
    "name": LvarMetadata(model="User", field="name", local_name="name", value="Alice"),
    "price": LvarMetadata(model="Product", field="price", local_name="price", value="99.99"),  # Wrong model!
}
out_fields = {"user": ["name", "price"]}  # Missing 'email', wrong type
lacts = {}

operable = Operable(specs=[
    Spec(name="user", base_type=User, required=True),
])

try:
    output = resolve_references_prefixed(out_fields, lvars, lacts, operable)
except ExceptionGroup as eg:
    print(f"Errors: {len(eg.exceptions)}")
    for e in eg.exceptions:
        print(f"  - {type(e).__name__}: {e}")
    # Errors: 2
    #   - MissingFieldError: Required field 'email' missing
    #   - TypeMismatchError: Variable 'price' is for model 'Product', but field 'user' expects 'User'
```

**Validation Workflow:**

1. **Name Collision Check**: Ensures no overlapping names between lvars and lacts
2. **Allowed Field Check**: Validates all OUT{} fields are allowed by operable specs
3. **Required Field Check**: Verifies all required specs are present in OUT{}
4. **Field Resolution Loop** (for each OUT{} field):
   - Get corresponding Spec from operable
   - Determine if scalar or BaseModel type
   - **Scalar Path**:
     - Resolve variable reference or parse literal value
     - Handle action references (store ActionCall placeholder)
     - Type convert and validate value
   - **BaseModel Path**:
     - Check for single direct action (returns entire model)
     - Build kwargs from variable list (mixing lvars and namespaced actions)
     - Validate model/field name matching for all variables/actions
     - Parse action function calls into ActionCall instances
     - Construct model instance:
       - `model_construct()` if contains actions (partial validation)
       - `model(**kwargs)` if pure lvars (full validation)
     - Apply spec validators/rules if defined
5. **Error Aggregation**: Collect all exceptions, raise as ExceptionGroup

**See Also:**

- `parse_lndl()`: High-level wrapper that extracts inputs and calls this function
- [LNDLOutput.revalidate_with_action_results()](types.md#revalidate): Re-validate after action execution
- [ActionCall](types.md#actioncall): Action placeholder type
- [LvarMetadata](types.md#lvarmetadata): Variable declaration metadata
- [LactMetadata](types.md#lactmetadata): Action declaration metadata

**Notes:**

#### Partial Validation with Actions

When a BaseModel field contains `ActionCall` placeholders (from `<lact>` references), `model_construct()` is used instead of `model(**kwargs)`. This **bypasses Pydantic validation** because action results aren't available yet.

**Critical**: Caller **MUST** re-validate after executing actions using `LNDLOutput.revalidate_with_action_results()`. See [LNDLOutput docstring](types.md#lndloutput) for complete action execution lifecycle.

#### Direct vs Namespaced Actions

- **Direct Actions** (`<lact action_name>func()</lact>`): No namespace, returns entire model
  - Use case: Single action populates all fields (e.g., `get_user()` returns complete User)
  - Cannot be mixed with lvars in same field
- **Namespaced Actions** (`<lact Model.field action_name>func()</lact>`): Targets specific field
  - Use case: Action populates one field, lvars populate others
  - Can be mixed with lvars in same OUT{} array

#### Spec Validators

If a Spec defines validators in metadata (`spec.get("validator")`), they are applied after model construction:

```python
# Spec with validator
spec = Spec(
    name="user",
    base_type=User,
    validator=lambda user: user if user.age >= 18 else raise ValueError("Underage")
)

# Validator invoked after construction
output = resolve_references_prefixed(...)
# Raises ValueError if age < 18
```

## Return Type

### LNDLOutput

Container for validated LNDL resolution results with action execution lifecycle support.

**Attributes:**

| Attribute       | Type                                       | Description                                                          |
| --------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| `fields`        | `dict[str, BaseModel \| scalar \| ActionCall]` | Validated output fields (may contain ActionCall placeholders)        |
| `lvars`         | `dict[str, LvarMetadata]`                  | Original lvar metadata (for debugging/inspection)                    |
| `lacts`         | `dict[str, LactMetadata]`                  | Original lact metadata (for debugging/inspection)                    |
| `actions`       | `dict[str, ActionCall]`                    | Parsed action calls for execution (name → ActionCall)                |
| `raw_out_block` | `str`                                      | Original OUT{} block string (for debugging)                          |

**Methods:**

- `revalidate_with_action_results(action_results: dict[str, Any]) -> LNDLOutput`: Replace ActionCall placeholders with actual results and re-validate models (see [LNDLOutput API](types.md#lndloutput))

**Usage:**

```python
output = parse_lndl(response, operable)

# Access validated fields
user = output.fields["user"]

# Execute actions if present
if output.actions:
    results = {}
    for name, action_call in output.actions.items():
        results[name] = execute_function(action_call.function, action_call.arguments)

    # Re-validate with action results
    final_output = output.revalidate_with_action_results(results)
    user = final_output.fields["user"]  # Now fully validated
```

## Error Types

### MissingFieldError

Raised when required spec field is absent from OUT{} block.

**Inheritance:** `ValueError`

**Message Format:** `"Required field '{field_name}' missing from OUT{}"`

**Example:**

```python
# Spec requires 'user' field
operable = Operable(specs=[Spec(name="user", base_type=User, required=True)])

# OUT{} missing 'user'
out_fields = {"other": ["value"]}

# Raises MissingFieldError
resolve_references_prefixed(out_fields, {}, {}, operable)
```

### TypeMismatchError

Raised when variable's model name doesn't match spec's expected base_type.

**Inheritance:** `ValueError`

**Message Format:** `"Variable '{var_name}' is for model '{var_model}', but field '{field_name}' expects '{expected_model}'"`

**Example:**

```python
# Spec expects User model
operable = Operable(specs=[Spec(name="user", base_type=User)])

# Variable declared for Product, not User
lvars = {"Product.price": LvarMetadata(model="Product", field="price", value="99.99")}
out_fields = {"user": ["Product.price"]}

# Raises TypeMismatchError
resolve_references_prefixed(out_fields, lvars, {}, operable)
```

## Usage Patterns

### Basic Variable Resolution

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec

# Simple scalar outputs (use literals for simplicity)
operable = Operable(specs=[
    Spec(name="name", base_type=str),
    Spec(name="age", base_type=int),
])

response = """
OUT{
  name: "Alice"
  age: 30
}
"""

output = parse_lndl(response, operable)
print(output.fields)
# {'name': 'Alice', 'age': 30}
```

### Nested Model Construction

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    name: str
    address: Address

operable = Operable(specs=[
    Spec(name="user", base_type=User),
    Spec(name="address", base_type=Address),  # Address constructed separately
])

response = """
<lvar User.name>Alice</lvar>
<lvar Address.street>123 Main St</lvar>
<lvar Address.city>NYC</lvar>

OUT{
  address: [street, city]
  user: [name]
}
"""

# Note: This constructs Address and User separately
# For nested models, construct them in separate OUT{} fields
# Then manually assign: user.address = address
```

### Action Execution Lifecycle

```python
from lionherd_core.lndl import parse_lndl, revalidate_with_action_results
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

operable = Operable(specs=[Spec(name="user", base_type=User)])

response = """
<lvar User.name>Alice</lvar>
<lact User.age get_age>calculate_age(birth_year=1990)</lact>

OUT{
  user: [name, get_age]
}
"""

# Step 1: Parse (partial validation)
output = parse_lndl(response, operable)

# Step 2: Execute actions
action_results = {}
for name, action in output.actions.items():
    if action.function == "calculate_age":
        # Execute function (pseudo-code)
        action_results[name] = 2025 - action.arguments["birth_year"]

# Step 3: Re-validate with results
validated_user = revalidate_with_action_results(output.fields["user"], action_results)
print(validated_user)
# User(name='Alice', age=35) - Fully validated
```

### Error Handling

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.lndl.errors import MissingFieldError, TypeMismatchError
from lionherd_core.types import Operable, Spec

operable = Operable(specs=[Spec(name="user", base_type=str, required=True)])

# Example: missing required field
response = """
OUT{
}
"""

try:
    output = parse_lndl(response, operable)
except ExceptionGroup as eg:
    # Handle multiple validation errors
    for exc in eg.exceptions:
        if isinstance(exc, MissingFieldError):
            print(f"Missing required field: {exc}")
        elif isinstance(exc, TypeMismatchError):
            print(f"Type mismatch: {exc}")
        else:
            print(f"Validation error: {exc}")
except (ValueError, MissingFieldError) as e:
    # Handle single critical error (name collision, syntax error, etc.)
    print(f"Critical error: {e}")
```

### Mixing Literals and Variables

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec

operable = Operable(specs=[
    Spec(name="name", base_type=str),
    Spec(name="status", base_type=str),
    Spec(name="count", base_type=int),
])

response = """
<lvar Scalar.name>Alice</lvar>

OUT{
  name: [name]
  status: "active"
  count: 42
}
"""

output = parse_lndl(response, operable)
print(output.fields)
# {'name': 'Alice', 'status': 'active', 'count': 42}
# Mixes variable reference with literal values
```

## Design Rationale

### Why Namespace-Prefixed Variables?

The `Model.field` syntax provides **explicit type mapping** that enables:

1. **Compile-time Validation**: Model/field names checked against specs before construction
2. **Mixing Multiple Models**: Same field names across models don't collide (`User.name` vs `Product.name`)
3. **Clear Intent**: Variable declarations explicitly state target model and field
4. **Type Safety**: Parser validates variable model matches spec's base_type

Alternative (untyped variables) would require runtime type inference and lose validation benefits.

### Why Partial Validation for Actions?

When models contain `ActionCall` placeholders, full Pydantic validation must be deferred because:

1. **Action Results Unknown**: Type constraints can't be validated without actual values
2. **Field Validators Need Real Data**: Pydantic validators (e.g., `@field_validator`) require concrete values
3. **Cross-Field Dependencies**: Validators referencing multiple fields fail with placeholders

`model_construct()` bypasses validation, caller re-validates with `revalidate_with_action_results()` after execution.

### Why Error Aggregation?

Collecting all validation errors into `ExceptionGroup` provides:

1. **Complete Diagnostics**: See all issues at once, not just first failure
2. **Better UX**: LLM can fix multiple errors in one iteration
3. **Debugging Efficiency**: Comprehensive error report accelerates troubleshooting
4. **Batch Reporting**: Workflows can collect and display all errors together

Single-error exceptions (ValueError) reserved for critical issues preventing continuation.

### Why Separate Direct and Namespaced Actions?

Two action modes support different use cases:

- **Direct Actions**: Single function returns complete model (e.g., database fetch)
  - Clean syntax for common case: `<lact fetch_user>get_user(id=123)</lact>`
  - Cannot mix with lvars (all fields from action or none)
- **Namespaced Actions**: Function populates specific field (e.g., computed value)
  - Mix with lvars: `<lact User.age compute_age>calculate_age()</lact>`
  - Supports partial model population (some fields static, some dynamic)

Separation prevents ambiguity and enables clear validation rules.

### Why Allow Literal Values in OUT{}?

Supporting literals (`OUT{status: "active"}`) alongside variable references provides:

1. **Convenience**: No need to declare lvar for constant values
2. **Readability**: Intent clearer when constant visible inline
3. **Flexibility**: Mix static and dynamic values in same output
4. **Backward Compatibility**: Matches common LLM output patterns

Type conversion still applied (validates literal converts to spec's base_type).

## See Also

- **Related Functions**:
  - [extract_lvars_prefixed()](parser.md#extract_lvars_prefixed): Extract namespace-prefixed lvars from response
  - [extract_lacts_prefixed()](parser.md#extract_lacts_prefixed): Extract action declarations
  - [extract_out_block()](parser.md#extract_out_block): Extract OUT{} block content
  - [parse_out_block_array()](parser.md#parse_out_block_array): Parse OUT{} into field mappings
- **Related Types**:
  - [LNDLOutput](types.md#lndloutput): Return type with action lifecycle methods
  - [ActionCall](types.md#actioncall): Parsed action representation
  - [LvarMetadata](types.md#lvarmetadata): Variable declaration metadata
  - [LactMetadata](types.md#lactmetadata): Action declaration metadata
  - [Operable](../types/operable.md): Spec container for validation
  - [Spec](../types/spec.md): Individual field specification
- **Related Errors**:
  - [MissingFieldError](errors.md#missingfielderror): Required field absent
  - [TypeMismatchError](errors.md#typemismatcherror): Variable/action model mismatch
- **Related Modules**:
  - [LNDL Parser](parser.md): LNDL response parsing
  - [LNDL Types](types.md): ActionCall and metadata types
  - [LNDL Fuzzy](fuzzy.md): Fuzzy matching for error tolerance

## Examples

### Example 1: End-to-End LNDL Workflow

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

# 1. Define output schema
class User(BaseModel):
    name: str
    age: int
    email: str

operable = Operable(specs=[
    Spec(name="user", base_type=User, required=True),
    Spec(name="greeting", base_type=str, required=True),
])

# 2. LLM generates LNDL response
response = """
Based on the user data:

<lvar User.name>Alice Johnson</lvar>
<lvar User.age>30</lvar>
<lvar User.email>alice@example.com</lvar>
<lvar Scalar.greeting>Hello Alice!</lvar>

OUT{
  user: [name, age, email]
  greeting: [greeting]
}
"""

# 3. Parse and validate
output = parse_lndl(response, operable)

# 4. Access validated outputs
user = output.fields["user"]
greeting = output.fields["greeting"]

print(f"{greeting}")
print(f"User: {user.name}, {user.age} years old, {user.email}")
# Hello Alice!
# User: Alice Johnson, 30 years old, alice@example.com
```

### Example 2: Handling Optional Fields

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str | None = None

operable = Operable(specs=[
    Spec(name="user", base_type=User, required=True),
])

# Response with missing optional field
response = """
<lvar User.name>Alice</lvar>

OUT{
  user: [name]
}
"""

output = parse_lndl(response, operable)
print(output.fields["user"])
# User(name='Alice', email=None)
# Optional field defaults to None when not provided
```

### Example 3: Multiple Models in Single Response

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

class Product(BaseModel):
    name: str
    price: float

operable = Operable(specs=[
    Spec(name="user", base_type=User),
    Spec(name="product", base_type=Product),
])

response = """
<lvar User.name u_name>Alice</lvar>
<lvar User.age u_age>30</lvar>
<lvar Product.name p_name>Laptop</lvar>
<lvar Product.price p_price>999.99</lvar>

OUT{
  user: [u_name, u_age]
  product: [p_name, p_price]
}
"""

output = parse_lndl(response, operable)
print(output.fields["user"])
# User(name='Alice', age=30)
print(output.fields["product"])
# Product(name='Laptop', price=999.99)
```

### Example 4: Spec Validators

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

def validate_adult(user: User) -> User:
    if user.age < 18:
        raise ValueError(f"User {user.name} is underage ({user.age})")
    return user

operable = Operable(specs=[
    Spec(
        name="user",
        base_type=User,
        validator=validate_adult,  # Applied after construction
    ),
])

response = """
<lvar User.name>Bob</lvar>
<lvar User.age>16</lvar>

OUT{
  user: [name, age]
}
"""

try:
    output = parse_lndl(response, operable)
except ExceptionGroup as eg:
    for exc in eg.exceptions:
        print(f"Validation failed: {exc}")
    # Validation failed: User Bob is underage (16)
```

### Example 5: Complex Action Workflow

```python
from lionherd_core.lndl import parse_lndl, revalidate_with_action_results
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    score: float
    recommendations: list[str]

operable = Operable(specs=[
    Spec(name="analysis", base_type=Analysis),
])

response = """
<lvar Analysis.summary>Code quality is good overall.</lvar>
<lact Analysis.score compute_score>calculate_code_score(repo="example")</lact>
<lact Analysis.recommendations get_recs>generate_recommendations(score=0.85)</lact>

OUT{
  analysis: [summary, compute_score, get_recs]
}
"""

# Step 1: Parse (creates placeholders)
output = parse_lndl(response, operable)
print(f"Actions to execute: {list(output.actions.keys())}")
# Actions to execute: ['compute_score', 'get_recs']

# Step 2: Execute actions
action_results = {
    "compute_score": 0.85,
    "get_recs": ["Add more tests", "Improve documentation"],
}

# Step 3: Re-validate
validated_analysis = revalidate_with_action_results(output.fields["analysis"], action_results)

print(validated_analysis.model_dump())
# {
#   'summary': 'Code quality is good overall.',
#   'score': 0.85,
#   'recommendations': ['Add more tests', 'Improve documentation']
# }
```

### Example 6: Debugging with Raw Metadata

```python
from lionherd_core.lndl import parse_lndl
from lionherd_core.types import Operable, Spec

operable = Operable(specs=[Spec(name="name", base_type=str)])

response = """
<lvar Scalar.name>Alice</lvar>

OUT{
  name: [name]
}
"""

output = parse_lndl(response, operable)

# Inspect metadata for debugging
print("Declared lvars:", output.lvars)
# Declared lvars: {'name': LvarMetadata(model='Scalar', field='name', local_name='name', value='Alice')}

print("Declared lacts:", output.lacts)
# Declared lacts: {}

print("Raw OUT block:", output.raw_out_block)
# Raw OUT block: {'name': ['name']}

print("Parsed actions:", output.actions)
# Parsed actions: {}
```

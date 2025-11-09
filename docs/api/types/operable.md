# Operable

> Ordered Spec collection for structured model generation with uniqueness validation

## Overview

`Operable` is a **framework-agnostic specification container** that manages ordered collections of `Spec` objects for dynamic model generation. It enforces **field name uniqueness**, provides **field filtering**, and enables **adapter-based model creation** for frameworks like Pydantic.

**Key Capabilities:**

- **Uniqueness Validation**: Enforces no duplicate field names across all Spec objects
- **Type Safety**: Validates all items are Spec instances at construction
- **Field Filtering**: Include/exclude patterns for selective field retrieval
- **Adapter Pattern**: Framework-agnostic model generation (currently supports Pydantic)
- **Protocol Implementation**: Implements Hashable and Allowable for collection operations
- **Immutability**: Frozen dataclass ensures specification integrity

**When to Use Operable:**

- **Dynamic Model Generation**: Create runtime models from specifications (LLM structured outputs, API schemas)
- **Schema Composition**: Build complex schemas from multiple Spec components
- **Field Validation**: Ensure field uniqueness and type correctness before model creation
- **Framework Abstraction**: Define schemas once, generate models for multiple frameworks

**When NOT to Use Operable:**

- Static models where dataclass/Pydantic suffices
- Single-field specifications (just use Spec directly)
- Scenarios where duplicate field names are valid
- Performance-critical tight loops (construction has validation overhead)

See [Spec](spec.md) for individual field specifications.

## Class Signature

```python
from lionherd_core.types import Operable, Spec

@implements(Hashable, Allowable)
@dataclass(frozen=True, slots=True, init=False)
class Operable:
    """Ordered Spec collection for model generation. Validates uniqueness, no duplicates."""

    # Constructor signature
    def __init__(
        self,
        specs: tuple[Spec, ...] | list[Spec] = (),
        *,
        name: str | None = None,
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**specs** : tuple[Spec, ...] or list[Spec], optional

Ordered collection of Spec objects defining the model fields. Converted to tuple if provided as list.

- Type coercion: Lists automatically converted to tuples (immutability)
- Validation: All items must be Spec instances (raises TypeError otherwise)
- Uniqueness: Field names must be unique across all Specs (raises ValueError if duplicates found)
- Default: Empty tuple `()`

**name** : str, optional

Optional name for the Operable collection. Used as default model name in `create_model()` if not explicitly provided.

- Default: `None`
- Usage: Model generation, documentation, debugging

## Attributes

| Attribute       | Type             | Frozen | Description                                    |
| --------------- | ---------------- | ------ | ---------------------------------------------- |
| `__op_fields__` | `tuple[Spec, ...]` | Yes    | Ordered tuple of Spec objects (internal storage) |
| `name`          | `str \| None`     | Yes    | Optional collection name                       |

## Methods

### Field Access

#### `allowed()`

Return set of allowed field names from all Specs.

**Signature:**

```python
def allowed(self) -> set[str]: ...
```

**Returns:**

- set[str]: Set of field names from all non-None Spec names

**Examples:**

```python
>>> from lionherd_core.types import Operable, Spec
>>> specs = [
...     Spec(str, name="username"),
...     Spec(int, name="age"),
...     Spec(bool, name="active"),
... ]
>>> op = Operable(specs)
>>> op.allowed()
{'username', 'age', 'active'}

# Unnamed specs excluded from allowed()
>>> specs_mixed = [
...     Spec(str, name="field1"),
...     Spec(int),  # No name
...     Spec(bool, name="field2"),
... ]
>>> op2 = Operable(specs_mixed)
>>> op2.allowed()
{'field1', 'field2'}  # Unnamed spec excluded
```

**Notes:**

Implements the `Allowable` protocol. Only Specs with non-None names are included in the allowed set.

#### `check_allowed()`

Check if field names are allowed, optionally return boolean instead of raising.

**Signature:**

```python
def check_allowed(self, *args, as_boolean: bool = False) -> bool: ...
```

**Parameters:**

- `*args` (str): Field names to check
- `as_boolean` (bool, default False): Return boolean instead of raising ValueError

**Returns:**

- bool: True if all args are allowed
  - If `as_boolean=False`: Always returns True (raises on failure)
  - If `as_boolean=True`: Returns False instead of raising

**Raises:**

- ValueError: If any field name not in allowed set (only when `as_boolean=False`)

**Examples:**

```python
>>> from lionherd_core.types import Operable, Spec
>>> specs = [
...     Spec(str, name="username"),
...     Spec(int, name="age"),
... ]
>>> op = Operable(specs)

# Check with exception (default)
>>> op.check_allowed("username", "age")
True

>>> op.check_allowed("invalid_field")
Traceback (most recent call last):
    ...
ValueError: Some specified fields are not allowed: {'invalid_field'}

# Check with boolean return
>>> op.check_allowed("username", as_boolean=True)
True
>>> op.check_allowed("invalid_field", as_boolean=True)
False  # No exception raised

# Multiple fields
>>> op.check_allowed("username", "age", "invalid", as_boolean=True)
False
```

**Notes:**

Use `as_boolean=True` for conditional logic without exception handling.

#### `get()`

Get Spec by field name with optional default.

**Signature:**

```python
def get(self, key: str, /, default=Unset) -> MaybeUnset[Spec]: ...
```

**Parameters:**

- `key` (str): Field name to look up (positional-only)
- `default` (Any, default Unset): Value to return if key not found

**Returns:**

- Spec or default: Spec object if found, otherwise default value
- MaybeUnset[Spec]: Type annotation indicating potential Unset sentinel

**Examples:**

```python
>>> from lionherd_core.types import Operable, Spec, Unset
>>> specs = [
...     Spec(str, name="username"),
...     Spec(int, name="age"),
... ]
>>> op = Operable(specs)

# Get existing field
>>> spec = op.get("username")
>>> spec.base_type
<class 'str'>

# Get missing field (returns default)
>>> result = op.get("missing")
>>> result is Unset
True

# Get with custom default
>>> result = op.get("missing", default=None)
>>> result is None
True
```

**Notes:**

Returns `Unset` by default for missing keys (not `None`) to distinguish "not found" from "explicitly None".

### Field Filtering

#### `get_specs()`

Get filtered Specs with include/exclude patterns.

**Signature:**

```python
def get_specs(
    self,
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> tuple[Spec, ...]: ...
```

**Parameters:**

- `include` (set[str], optional): Only include these field names. Cannot be used with `exclude`.
- `exclude` (set[str], optional): Exclude these field names. Cannot be used with `include`.

**Returns:**

- tuple[Spec, ...]: Filtered Specs matching criteria

**Raises:**

- ValueError: If both `include` and `exclude` provided, or if include contains invalid field names

**Examples:**

```python
>>> from lionherd_core.types import Operable, Spec
>>> specs = [
...     Spec(str, name="username"),
...     Spec(int, name="age"),
...     Spec(bool, name="active"),
...     Spec(str, name="email"),
... ]
>>> op = Operable(specs)

# Include pattern (whitelist)
>>> filtered = op.get_specs(include={"username", "email"})
>>> [s.name for s in filtered]
['username', 'email']

# Exclude pattern (blacklist)
>>> filtered = op.get_specs(exclude={"age"})
>>> [s.name for s in filtered]
['username', 'active', 'email']

# No filter (returns all)
>>> filtered = op.get_specs()
>>> len(filtered)
4

# Error: both include and exclude
>>> op.get_specs(include={"username"}, exclude={"age"})
Traceback (most recent call last):
    ...
ValueError: Cannot specify both include and exclude

# Error: invalid include field
>>> op.get_specs(include={"invalid_field"})
Traceback (most recent call last):
    ...
ValueError: Some specified fields are not allowed: {'invalid_field'}
```

**Notes:**

Use this method to:

- Build partial models (API versioning, permission filtering)
- Compose schemas from larger specifications
- Generate subset models for different contexts

### Model Generation

#### `create_model()`

Create framework-specific model from Specs using adapter pattern.

**Signature:**

```python
def create_model(
    self,
    adapter: Literal["pydantic"] = "pydantic",
    model_name: str | None = None,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    **kw,
) -> Any: ...
```

**Parameters:**

- `adapter` ({'pydantic'}, default 'pydantic'): Framework adapter to use
  - `'pydantic'`: Generate Pydantic BaseModel subclass
  - Future: `'dataclass'`, `'msgspec'`, `'attrs'` (planned)
- `model_name` (str, optional): Name for generated model class. Default: `self.name` or `"DynamicModel"`
- `include` (set[str], optional): Only include these fields in model
- `exclude` (set[str], optional): Exclude these fields from model
- `**kw` (Any): Additional adapter-specific keyword arguments

**Returns:**

- Any: Generated model class (type depends on adapter)
  - Pydantic: Returns `type[BaseModel]` subclass

**Raises:**

- ImportError: If adapter framework not installed (e.g., Pydantic missing)
- ValueError: If adapter not supported or if both include/exclude provided

**Examples:**

```python
>>> from lionherd_core.types import Operable, Spec
>>> from pydantic import ValidationError

# Define schema
>>> specs = [
...     Spec(str, name="username"),
...     Spec(int, name="age", default=0),
...     Spec(bool, name="active", default=True),
... ]
>>> op = Operable(specs, name="User")

# Generate Pydantic model
>>> UserModel = op.create_model(adapter="pydantic")
>>> UserModel.__name__
'User'

# Instantiate and validate
>>> user = UserModel(username="alice", age=30)
>>> user.username
'alice'
>>> user.active
True  # Default applied

# Validation enforced
>>> try:
...     UserModel(username="bob", age="invalid")
... except ValidationError as e:
...     print("Validation failed")
Validation failed

# Partial model with include
>>> PartialModel = op.create_model(
...     adapter="pydantic",
...     model_name="PartialUser",
...     include={"username", "age"}
... )
>>> fields = list(PartialModel.model_fields.keys())
>>> sorted(fields)
['age', 'username']  # 'active' excluded

# With custom name
>>> CustomModel = op.create_model(
...     adapter="pydantic",
...     model_name="CustomUser"
... )
>>> CustomModel.__name__
'CustomUser'
```

**Adapter-Specific Kwargs (Pydantic):**

```python
# Pass through to Pydantic create_model()
>>> StrictModel = op.create_model(
...     adapter="pydantic",
...     model_name="StrictUser",
...     __config__={"str_strip_whitespace": True}  # Pydantic config
... )
```

**Notes:**

**Adapter Pattern Benefits:**

- Framework-agnostic schema definition
- Generate models for multiple frameworks from same Specs
- Easy migration between validation libraries
- Testability (mock adapters)

**Current Limitations:**

- Only Pydantic adapter implemented (v1.0.0-alpha)
- Validation modes (strict/lenient/fuzzy) not yet exposed in API
- LNDL (Language InterOperable Network Directive Language) integration available via [lndl module](../lndl/types.md)

Pydantic adapter implementation is available in `src/lionherd_core/types/spec_adapters/pydantic_field.py`.

## Protocol Implementations

Operable implements two core protocols:

### Hashable

**Method**: `__hash__()` based on frozen dataclass

Operable instances are hashable because all attributes are frozen (immutable). Hash is computed from `(__op_fields__, name)` tuple.

**Usage**: Safe for use in sets and as dict keys.

**Examples:**

```python
>>> from lionherd_core.types import Operable, Spec
>>> specs = [Spec(str, name="field1")]
>>> op1 = Operable(specs, name="Schema1")
>>> op2 = Operable(specs, name="Schema1")

# Same specs and name = same hash
>>> hash(op1) == hash(op2)
True

# Use in sets
>>> schema_set = {op1, op2}
>>> len(schema_set)
1  # Deduplicated by hash

# Use as dict keys
>>> cache = {op1: "cached_model"}
>>> cache[op2]
'cached_model'  # Same hash retrieves value
```

### Allowable

**Method**: `allowed()` returns set of field names

Implements the `Allowable` protocol by extracting field names from all Specs with non-None names.

**Usage**: Field validation, schema introspection, access control.

**Examples:**

```python
>>> op.allowed()
{'field1', 'field2', 'field3'}

# Use with check_allowed()
>>> op.check_allowed("field1")
True
>>> op.check_allowed("invalid", as_boolean=True)
False
```

## Usage Patterns

### Basic Schema Definition

```python
from lionherd_core.types import Operable, Spec

# Define specifications
specs = [
    Spec(str, name="username", nullable=False),
    Spec(int, name="age", default=0),
    Spec(list[str], name="tags", default_factory=list),
]

# Create Operable
schema = Operable(specs, name="UserSchema")

# Inspect allowed fields
print(schema.allowed())  # {'username', 'age', 'tags'}

# Access individual specs
username_spec = schema.get("username")
print(username_spec.base_type)  # <class 'str'>
```

### Dynamic Model Generation

```python
from lionherd_core.types import Operable, Spec

# Define schema
specs = [
    Spec(str, name="id"),
    Spec(str, name="title"),
    Spec(str, name="content"),
    Spec(int, name="views", default=0),
]
schema = Operable(specs, name="Article")

# Generate Pydantic model
ArticleModel = schema.create_model(adapter="pydantic")

# Use model
article = ArticleModel(id="123", title="Hello", content="World")
print(article.model_dump())
# {'id': '123', 'title': 'Hello', 'content': 'World', 'views': 0}
```

### Schema Composition

```python
from lionherd_core.types import Operable, Spec

# Base user fields
base_specs = [
    Spec(str, name="id"),
    Spec(str, name="username"),
    Spec(str, name="email"),
]

# Admin-specific fields
admin_specs = [
    Spec(bool, name="is_admin", default=True),
    Spec(list[str], name="permissions", default_factory=list),
]

# Compose schemas
user_schema = Operable(base_specs, name="User")
admin_schema = Operable(base_specs + admin_specs, name="AdminUser")

# Generate models
UserModel = user_schema.create_model()
AdminModel = admin_schema.create_model()

# UserModel has 3 fields, AdminModel has 5 fields
```

### Field Filtering for API Versioning

```python
from lionherd_core.types import Operable, Spec

# Full schema (v2)
all_specs = [
    Spec(str, name="id"),
    Spec(str, name="username"),
    Spec(str, name="email"),
    Spec(str, name="phone"),      # Added in v2
    Spec(bool, name="verified"),  # Added in v2
]
full_schema = Operable(all_specs, name="UserV2")

# Generate v1 model (exclude new fields)
UserV1 = full_schema.create_model(
    model_name="UserV1",
    exclude={"phone", "verified"}
)

# Generate v2 model (all fields)
UserV2 = full_schema.create_model(model_name="UserV2")

# Both models from same schema definition
```

### Validation and Error Handling

```python
from lionherd_core.types import Operable, Spec

# Duplicate field names
try:
    specs = [
        Spec(str, name="field1"),
        Spec(int, name="field1"),  # Duplicate!
    ]
    schema = Operable(specs)
except ValueError as e:
    print(e)  # "Duplicate field names found: ['field1']"

# Invalid type (not a Spec)
try:
    schema = Operable([{"name": "field1"}])  # Dict, not Spec
except TypeError as e:
    print(e)  # "All specs must be Spec objects, got dict at index 0"

# Invalid field in include
schema = Operable([Spec(str, name="field1")])
try:
    filtered = schema.get_specs(include={"invalid_field"})
except ValueError as e:
    print(e)  # "Some specified fields are not allowed: {'invalid_field'}"
```

### LLM Structured Output Pattern

```python
from lionherd_core.types import Operable, Spec

# Define expected LLM output schema
llm_specs = [
    Spec(str, name="task", nullable=False),
    Spec(str, name="reasoning"),
    Spec(list[str], name="steps", default_factory=list),
    Spec(float, name="confidence", default=0.0),
]

# Create schema
schema = Operable(llm_specs, name="AgentResponse")

# Generate validation model
ResponseModel = schema.create_model(adapter="pydantic")

# Parse LLM response
llm_output = {
    "task": "Analyze sentiment",
    "reasoning": "Text contains positive indicators",
    "steps": ["tokenize", "classify", "aggregate"],
    "confidence": 0.92
}

# Validate and parse
response = ResponseModel(**llm_output)
print(response.task)  # "Analyze sentiment"
print(response.confidence)  # 0.92
```

## Common Pitfalls

### Pitfall 1: Duplicate Field Names

**Issue**: Attempting to create Operable with duplicate field names.

```python
from lionherd_core.types import Operable, Spec

# WRONG: Duplicate names
specs = [
    Spec(str, name="id"),
    Spec(int, name="id"),  # Duplicate!
]

try:
    schema = Operable(specs)
except ValueError as e:
    print(e)  # "Duplicate field names found: ['id']"
```

**Solution**: Ensure all Spec names are unique. Use different names or rename fields:

```python
# CORRECT: Unique names
specs = [
    Spec(str, name="id"),
    Spec(int, name="numeric_id"),  # Different name
]
schema = Operable(specs)
```

### Pitfall 2: Non-Spec Items

**Issue**: Passing non-Spec objects to Operable constructor.

```python
# WRONG: Mixed types
specs = [
    Spec(str, name="field1"),
    {"name": "field2", "type": int},  # Dict, not Spec!
]

try:
    schema = Operable(specs)
except TypeError as e:
    print(e)  # "All specs must be Spec objects, got dict at index 1"
```

**Solution**: Convert all items to Spec objects:

```python
# CORRECT: All Spec objects
specs = [
    Spec(str, name="field1"),
    Spec(int, name="field2"),  # Proper Spec
]
schema = Operable(specs)
```

### Pitfall 3: Both Include and Exclude

**Issue**: Providing both `include` and `exclude` parameters.

```python
schema = Operable([Spec(str, name="f1"), Spec(int, name="f2")])

# WRONG: Both parameters
try:
    filtered = schema.get_specs(include={"f1"}, exclude={"f2"})
except ValueError as e:
    print(e)  # "Cannot specify both include and exclude"
```

**Solution**: Use only one filtering mode:

```python
# CORRECT: Include OR exclude
filtered_include = schema.get_specs(include={"f1"})
filtered_exclude = schema.get_specs(exclude={"f2"})
```

### Pitfall 4: Missing Pydantic Dependency

**Issue**: Calling `create_model()` without Pydantic installed.

```python
schema = Operable([Spec(str, name="field1")])

# Raises ImportError if Pydantic not installed
try:
    model = schema.create_model(adapter="pydantic")
except ImportError as e:
    print(e)  # "PydanticSpecAdapter requires Pydantic. Install with: pip install pydantic"
```

**Solution**: Install required dependencies:

```bash
# Install Pydantic
pip install pydantic

# Or install with extras
pip install lionherd-core[pydantic]
```

### Pitfall 5: Unnamed Specs Not in allowed()

**Issue**: Expecting unnamed Specs to appear in `allowed()`.

```python
specs = [
    Spec(str, name="field1"),
    Spec(int),  # No name
]
schema = Operable(specs)

# Unnamed spec NOT in allowed()
print(schema.allowed())  # {'field1'} - unnamed spec missing
```

**Solution**: Always provide names for Specs that need to be accessible:

```python
# CORRECT: All specs named
specs = [
    Spec(str, name="field1"),
    Spec(int, name="field2"),  # Named
]
schema = Operable(specs)
print(schema.allowed())  # {'field1', 'field2'}
```

### Pitfall 6: Mutating Specs After Creation

**Issue**: Attempting to modify frozen Operable after construction.

```python
schema = Operable([Spec(str, name="field1")])

# WRONG: Operable is frozen
try:
    schema.name = "NewName"
except Exception as e:
    print(type(e).__name__)  # FrozenInstanceError
```

**Solution**: Operable is immutable by design. Create a new instance instead:

```python
# CORRECT: Create new instance
old_schema = Operable([Spec(str, name="field1")], name="OldSchema")
new_schema = Operable(old_schema.__op_fields__, name="NewSchema")
```

## Design Rationale

### Why Enforce Uniqueness?

Operable enforces field name uniqueness because:

1. **Model Generation**: Frameworks like Pydantic, dataclasses, and msgspec require unique field names
2. **Ambiguity Prevention**: Duplicate names create undefined behavior (which field wins?)
3. **Schema Integrity**: Clear 1:1 mapping between field names and Spec objects
4. **Error Detection**: Catch configuration errors early (construction time, not runtime)

If you need multiple specs for the same conceptual field (e.g., validation variants), use different names or separate Operables.

### Why Immutability (Frozen)?

Freezing Operable ensures:

1. **Hash Stability**: Safe for use in sets/dicts without hash corruption
2. **Schema Integrity**: Specifications don't change unexpectedly after creation
3. **Thread Safety**: No synchronization needed for concurrent access
4. **Functional Style**: Composable schemas without side effects

Mutable schemas would break hashing and introduce subtle bugs in multi-threaded or caching scenarios.

### Why Adapter Pattern?

The adapter pattern (`create_model(adapter=...)`) provides:

1. **Framework Agnostic**: Define specs once, target multiple validation libraries
2. **Future Proof**: Add new adapters (dataclass, msgspec, attrs) without changing Operable
3. **Testability**: Mock adapters for unit testing schema logic
4. **Migration**: Gradually migrate between frameworks without rewriting schemas

Single method call abstracts framework-specific model creation complexity.

### Why Include/Exclude Instead of Modification?

`get_specs()` filtering (include/exclude) rather than mutation enables:

1. **Immutability**: Original schema unchanged (functional style)
2. **Reusability**: Same schema generates multiple model variants
3. **Versioning**: API v1/v2 from same full schema
4. **Composition**: Build partial schemas without duplicating Spec definitions

Filtering is a query operation, not a transformation.

### Why List to Tuple Conversion?

Constructor converts `list[Spec]` to `tuple[Spec, ...]` because:

1. **Immutability**: Tuples are immutable, lists are not
2. **Hash Support**: Tuples are hashable, lists are not
3. **API Ergonomics**: Users can pass lists (convenient) without breaking guarantees
4. **Type Safety**: Internal storage always tuple (consistent)

This provides convenience without sacrificing safety.

## Validation Modes (Future)

**Status**: Planned for future releases (not yet implemented in API)

Operable will support multiple validation modes for different use cases:

- **Strict Mode**: All fields required, no defaults, validation errors fatal
- **Lenient Mode**: Optional fields allowed, defaults applied, warnings for issues
- **Fuzzy Mode**: Best-effort parsing, coercion enabled, silently ignore extra fields

**Current State**: Validation mode configuration not yet exposed in Operable API. Pydantic adapter uses framework defaults.

## LNDL Support

**Status**: Available in lionherd v1.0.0-alpha3+

LNDL (Language InterOperable Network Directive Language) integration enables:

- Structured LLM output parsing with `OUT{}` blocks
- Variable resolution and templating with `LVAR{}`
- Tool/ActionCall invocation with `ActionCall{}`
- Fuzzy matching for schema validation
- Template-based generation with metadata preservation

**Usage**: See comprehensive documentation in the [LNDL module](../lndl/parser.md):

- [Parser](../lndl/parser.md): Parse LVAR, ActionCall, and OUT blocks
- [Resolver](../lndl/resolver.md): Variable resolution and validation
- [Fuzzy](../lndl/fuzzy.md): Typo-tolerant parsing with Operable integration
- [Types](../lndl/types.md): Core LNDL data structures
- [Prompt](../lndl/prompt.md): System prompts for LLMs
- [Errors](../lndl/errors.md): Exception hierarchy

## See Also

- **Related Classes**:
  - [Spec](spec.md): Individual field specifications
  - [Meta](base.md#meta): Metadata key-value pairs
  - [Params](base.md#params): Frozen parameter objects
  - [DataClass](base.md#dataclass): Mutable validated dataclasses
- **Related Modules**:
  - [Sentinel Values](sentinel.md): Unset, Undefined sentinels
  - [Element](../base/element.md): Identity-based base class

## Examples

### Example 1: Basic CRUD Schema

```python
from lionherd_core.types import Operable, Spec
from datetime import datetime

# Define CRUD entity schema
specs = [
    Spec(str, name="id"),
    Spec(str, name="title", nullable=False),
    Spec(str, name="content"),
    Spec(datetime, name="created_at", default_factory=datetime.utcnow),
    Spec(datetime, name="updated_at", nullable=True),
]

schema = Operable(specs, name="Article")

# Generate model
ArticleModel = schema.create_model(adapter="pydantic")

# Create instance
article = ArticleModel(id="123", title="Hello World", content="...")
print(article.created_at)  # Auto-generated timestamp
print(article.updated_at)  # None (nullable)
```

### Example 2: Multi-Version API Schema

```python
from lionherd_core.types import Operable, Spec

# Full schema (v2 with all fields)
all_specs = [
    Spec(str, name="user_id"),
    Spec(str, name="username"),
    Spec(str, name="email"),
    Spec(str, name="phone"),      # v2 only
    Spec(bool, name="verified"),  # v2 only
    Spec(str, name="avatar_url"), # v2 only
]

schema = Operable(all_specs, name="User")

# API v1: Basic fields only
UserV1 = schema.create_model(
    model_name="UserV1",
    include={"user_id", "username", "email"}
)

# API v2: All fields
UserV2 = schema.create_model(model_name="UserV2")

# Same schema definition, different models
print(list(UserV1.model_fields.keys()))
# ['user_id', 'username', 'email']

print(list(UserV2.model_fields.keys()))
# ['user_id', 'username', 'email', 'phone', 'verified', 'avatar_url']
```

### Example 3: Filtered Schema Generation

```python
from lionherd_core.types import Operable, Spec

# Complete user profile schema
specs = [
    Spec(str, name="id"),
    Spec(str, name="username"),
    Spec(str, name="email"),
    Spec(str, name="password_hash"),  # Sensitive
    Spec(str, name="api_key"),        # Sensitive
    Spec(str, name="bio"),
    Spec(str, name="avatar_url"),
]

schema = Operable(specs, name="UserProfile")

# Public API model (exclude sensitive fields)
PublicUser = schema.create_model(
    model_name="PublicUser",
    exclude={"password_hash", "api_key"}
)

# Internal model (all fields)
InternalUser = schema.create_model(model_name="InternalUser")

# PublicUser safe for API responses
# InternalUser for database/internal use
```

### Example 4: Schema Introspection

```python
from lionherd_core.types import Operable, Spec

specs = [
    Spec(str, name="field1"),
    Spec(int, name="field2"),
    Spec(bool, name="field3"),
]

schema = Operable(specs, name="MySchema")

# Inspect allowed fields
print(schema.allowed())
# {'field1', 'field2', 'field3'}

# Access individual specs
field1_spec = schema.get("field1")
print(field1_spec.base_type)  # <class 'str'>

# Check field existence
print(schema.check_allowed("field1", as_boolean=True))  # True
print(schema.check_allowed("invalid", as_boolean=True))  # False

# Get filtered specs
subset = schema.get_specs(include={"field1", "field2"})
print([s.name for s in subset])  # ['field1', 'field2']
```

### Example 5: Hashing and Caching

```python
from lionherd_core.types import Operable, Spec

# Create schema
specs = [Spec(str, name="field1"), Spec(int, name="field2")]
schema1 = Operable(specs, name="Schema1")
schema2 = Operable(specs, name="Schema1")  # Same specs and name

# Hashing for deduplication
schema_set = {schema1, schema2}
print(len(schema_set))  # 1 (deduplicated by hash)

# Caching generated models
model_cache = {}

def get_or_create_model(schema: Operable):
    if schema not in model_cache:
        model_cache[schema] = schema.create_model(adapter="pydantic")
    return model_cache[schema]

# First call generates model
Model1 = get_or_create_model(schema1)

# Second call retrieves cached model (same hash)
Model2 = get_or_create_model(schema2)

# Same model instance
print(Model1 is Model2)  # True (cached)
```

### Example 6: Validation Error Handling

```python
from lionherd_core.types import Operable, Spec
from pydantic import ValidationError

# Schema with constraints
specs = [
    Spec(str, name="username", nullable=False),
    Spec(int, name="age"),
    Spec(str, name="email"),
]

schema = Operable(specs, name="User")
UserModel = schema.create_model(adapter="pydantic")

# Valid data
user = UserModel(username="alice", age=30, email="alice@example.com")
print(user.username)  # "alice"

# Invalid data (missing required field)
try:
    user = UserModel(age=30, email="bob@example.com")  # Missing username
except ValidationError as e:
    print("Validation error: missing username")

# Invalid type
try:
    user = UserModel(username="charlie", age="not_an_int", email="charlie@example.com")
except ValidationError as e:
    print("Validation error: age must be int")
```

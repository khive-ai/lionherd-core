# Fuzzy Validate

> Pydantic model validation and dictionary parsing with fuzzy JSON extraction and key matching

## Overview

The fuzzy validation module provides utilities for validating and parsing inputs into Pydantic models or dictionaries with robust error handling. It combines fuzzy JSON extraction, fuzzy key matching, and Pydantic validation to handle noisy inputs from LLMs, APIs, or user submissions.

**Key Capabilities:**

- **Fuzzy JSON Extraction**: Parse JSON from markdown code blocks, text with noise, or malformed strings
- **Pydantic Validation**: Convert text/dict to validated Pydantic models with fuzzy parsing
- **Fuzzy Key Matching**: Correct misspelled keys using similarity algorithms before validation
- **Multi-Format Parsing**: Handle dict, JSON string, XML, or arbitrary objects
- **Flexible Error Handling**: Suppress conversion errors or raise with detailed context
- **Strict/Lenient Modes**: Choose between strict validation or permissive parsing

**When to Use Fuzzy Validate:**

- Validating LLM-generated JSON with potential formatting issues
- Parsing API responses with inconsistent key naming
- Converting user-provided text to structured Pydantic models
- Handling noisy inputs where exact JSON format is not guaranteed
- Data pipelines requiring robust validation with fallback strategies

**When NOT to Use Fuzzy Validate:**

- Performance-critical paths where inputs are already validated
- Scenarios requiring strict schema enforcement without fuzzy matching
- Simple dict-to-model conversions with trusted inputs (use `model_validate()` directly)
- Cases where validation failures should always raise exceptions

## Module Contents

### Functions

- `fuzzy_validate_pydantic()`: Validate and parse text/dict into Pydantic model
- `fuzzy_validate_mapping()`: Validate any input into dictionary with expected keys

## Function: fuzzy_validate_pydantic

### Signature

```python
from lionherd_core.ln import fuzzy_validate_pydantic

def fuzzy_validate_pydantic(
    text,
    /,
    model_type: type[BaseModel],
    fuzzy_parse: bool = True,
    fuzzy_match: bool = False,
    fuzzy_match_params: FuzzyMatchKeysParams | dict = None,
) -> BaseModel: ...
```

### Parameters

#### Positional-Only Parameters

**text** : BaseModel or dict or str

Input data to validate and parse.

- Accepted types:
  - `BaseModel`: Returns as-is if already instance of `model_type`
  - `dict`: Validates directly (skips JSON extraction)
  - `str`: Extracts JSON, then validates
- Validation: Must be convertible to dict or valid JSON string
- Behavior: If already instance of `model_type`, returns immediately without revalidation
- Note: Strings may contain JSON in markdown code blocks, extra text, or malformed JSON

#### Keyword-Only Parameters

**model_type** : type[BaseModel]

Target Pydantic model class to validate against.

- Type: Pydantic `BaseModel` subclass
- Validation: Must be a valid Pydantic model with `model_validate()` and `model_fields`
- Behavior: Uses `model_type.model_validate()` for final validation
- Note: Model fields are used for fuzzy key matching if `fuzzy_match=True`

**fuzzy_parse** : bool, default True

Enable fuzzy JSON extraction from text inputs.

- True: Apply fuzzy JSON extraction (handles markdown blocks, noise, malformed JSON)
- False: Use strict `json.loads()` parsing
- Behavior: Only affects string inputs; dict inputs skip JSON extraction
- Note: Uses `extract_json()` with markdown code block detection

**fuzzy_match** : bool, default False

Enable fuzzy key matching for field names before validation.

- True: Correct misspelled keys using similarity algorithms
- False: Use exact key matching (default)
- Behavior: Compares input dict keys against `model_type.model_fields`
- Note: Requires `fuzzy_match_params` for custom matching configuration

**fuzzy_match_params** : FuzzyMatchKeysParams or dict or None, default None

Parameters for fuzzy key matching (only used if `fuzzy_match=True`).

- Accepted types:
  - `None`: Uses default params (`handle_unmatched="remove"`)
  - `dict`: Passed as `**kwargs` to `fuzzy_match_keys()`
  - `FuzzyMatchKeysParams`: Callable instance with configured params
- Default behavior (None):
  - `similarity_threshold=0.85`
  - `handle_unmatched="remove"` (discard unknown keys)
- Example dict: `{"similarity_threshold": 0.9, "handle_unmatched": "raise"}`
- Raises: `TypeError` if not dict, FuzzyMatchKeysParams, or None

### Returns

### BaseModel

Validated Pydantic model instance of type `model_type`.

- Type: Instance of `model_type` class
- Guarantees: All Pydantic validators and field constraints applied
- Behavior: Returns existing instance if input already matches `model_type`

### Raises

### ValidationError

- If JSON extraction fails (`fuzzy_parse=True` and invalid JSON string)
- If Pydantic validation fails (invalid field types, missing required fields, etc.)
- Error message includes detailed context about extraction or validation failure

### TypeError

- If `fuzzy_match_params` is not a dict, FuzzyMatchKeysParams instance, or None

## Function: fuzzy_validate_mapping

### Signature

```python
from lionherd_core.ln import fuzzy_validate_mapping

def fuzzy_validate_mapping(
    d: Any,
    keys: KeysLike,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
) -> dict[str, Any]: ...
```

### Parameters

#### Positional-Only Parameters

**d** : Any

Input to convert and validate.

- Accepted types: dict, JSON string, XML string, Pydantic model, arbitrary object
- Validation: Cannot be None (raises TypeError)
- Conversion:
  - `str`: Attempts JSON extraction, then `to_dict()` fallback
  - `dict`: Used directly
  - `BaseModel`: Converted via `model_dump()`
  - Other: Converted via `to_dict()` with fuzzy parsing
- Behavior: Conversion errors return empty dict if `suppress_conversion_errors=True`

**keys** : KeysLike (list[str] or dict-like with .keys())

Expected keys to validate against.

- Accepted types: List of strings, dict, Pydantic model with `.keys()` method
- Validation: Cannot be None (raises TypeError)
- Behavior: Extracted keys are used for fuzzy matching and validation

#### Keyword-Only Parameters

**similarity_algo** : SIMILARITY_TYPE or Callable[[str, str], float], default "jaro_winkler"

Algorithm for computing string similarity (see [fuzzy_match.md](fuzzy_match.md#similarity_algo) for details).

**similarity_threshold** : float, default 0.85

Minimum similarity score (0.0-1.0) for considering keys a match (see [fuzzy_match.md](fuzzy_match.md#similarity_threshold) for details).

**fuzzy_match** : bool, default True

Enable fuzzy key matching for keys that don't match exactly.

- True: Apply similarity algorithm to unmatched keys (default)
- False: Only exact matches considered
- Note: Differs from `fuzzy_validate_pydantic()` where default is False

**handle_unmatched** : Literal["ignore", "raise", "remove", "fill", "force"], default "ignore"

Strategy for handling keys that don't match expected keys (see [fuzzy_match.md](fuzzy_match.md#handle_unmatched) for details).

**fill_value** : Any, default None

Default value for missing expected keys when `handle_unmatched` is "fill" or "force" (see [fuzzy_match.md](fuzzy_match.md#fill_value) for details).

**fill_mapping** : dict of {str : Any} or None, default None

Custom values for specific missing expected keys (see [fuzzy_match.md](fuzzy_match.md#fill_mapping) for details).

**strict** : bool, default False

Raise ValueError if any expected keys are missing after matching (see [fuzzy_match.md](fuzzy_match.md#strict) for details).

**suppress_conversion_errors** : bool, default False

Return empty dict on conversion failure instead of raising exception.

- True: Conversion errors return `{}` (permissive)
- False: Conversion errors raise ValueError with detailed message (default)
- Use case: Pipelines where missing data should be handled gracefully
- Note: Validation errors (post-conversion) still raise if `strict=True` or `handle_unmatched="raise"`

### Returns

**dict[str, Any]**

Validated dictionary with corrected keys based on validation rules.

- Keys: Corrected to match expected keys via fuzzy matching
- Values: Preserved from original input
- Missing keys: Filled according to `fill_value` / `fill_mapping` if configured
- Unmatched keys: Handled according to `handle_unmatched` strategy

### Raises

### TypeError

- If `d` is None

### ValueError

- If conversion to dict fails and `suppress_conversion_errors=False`
- If `similarity_threshold` not in range [0.0, 1.0]
- If `similarity_algo` string not recognized
- If `handle_unmatched="raise"` and unmatched keys found
- If `strict=True` and expected keys missing

## Usage Patterns

### Basic Pydantic Validation

````python
from lionherd_core.ln import fuzzy_validate_pydantic
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    age: int

# JSON string with markdown code block (common LLM output)
llm_output = """
Here's the user data:
```json
{
    "username": "alice",
    "email": "alice@example.com",
    "age": 30
}
```

"""

# Parse and validate

user = fuzzy_validate_pydantic(llm_output, model_type=User)

# User(username='alice', email='<alice@example.com>', age=30)

````

### Pydantic Validation with Fuzzy Keys

```python
from lionherd_core.ln import fuzzy_validate_pydantic
from pydantic import BaseModel

class Task(BaseModel):
    title: str
    description: str
    priority: str

# LLM output with typos in field names
llm_json = {
    "titel": "Fix bug",           # Typo: "titel" → "title"
    "descr": "Fix login issue",   # Typo: "descr" → "description"
    "priority": "high"            # Exact match
}

# Validate with fuzzy key matching
task = fuzzy_validate_pydantic(
    llm_json,
    model_type=Task,
    fuzzy_match=True,
    fuzzy_match_params={"similarity_threshold": 0.85, "handle_unmatched": "remove"}
)
# Task(title='Fix bug', description='Fix login issue', priority='high')
```

### Dictionary Validation

```python
from lionherd_core.ln import fuzzy_validate_mapping

# Expected schema
expected_keys = ["user_id", "action", "timestamp"]

# Noisy input from API
api_response = """
{
    "userId": "12345",
    "acton": "login",
    "timestmp": 1699438200
}
"""

# Parse and validate
validated = fuzzy_validate_mapping(
    api_response,
    expected_keys,
    similarity_threshold=0.85,
    handle_unmatched="remove"
)
# {
#     'user_id': '12345',
#     'action': 'login',
#     'timestamp': 1699438200
# }
```

### Suppress Conversion Errors

```python
from lionherd_core.ln import fuzzy_validate_mapping

expected = ["name", "email"]

# Invalid input (not convertible to dict)
bad_inputs = [
    None,                    # TypeError (cannot suppress)
    "not valid json",        # Conversion error
    12345,                   # Conversion error
    ["list", "not", "dict"]  # Conversion error
]

for input_data in bad_inputs[1:]:  # Skip None (always raises)
    result = fuzzy_validate_mapping(
        input_data,
        expected,
        suppress_conversion_errors=True
    )
    print(result)  # {} (empty dict for all conversion failures)
```

### Strict Validation with Defaults

```python
from lionherd_core.ln import fuzzy_validate_mapping

expected = ["name", "age", "city"]

# Incomplete data
incomplete = {"name": "Alice"}

# Validate with strict mode + fill strategy
validated = fuzzy_validate_mapping(
    incomplete,
    expected,
    handle_unmatched="fill",
    fill_mapping={"age": 0, "city": "Unknown"},
    strict=True  # Ensure all keys present
)
# {
#     'name': 'Alice',
#     'age': 0,
#     'city': 'Unknown'
# }
```

### Multi-Format Input Handling

```python
from lionherd_core.ln import fuzzy_validate_mapping
from pydantic import BaseModel

class Config(BaseModel):
    setting1: str = "default1"
    setting2: str = "default2"

expected = ["setting1", "setting2"]

# Handle various input formats
inputs = [
    # Dict
    {"setting1": "value1", "setting2": "value2"},

    # JSON string
    '{"setting1": "value1", "setting2": "value2"}',

    # Pydantic model
    Config(setting1="value1", setting2="value2"),

    # Markdown JSON
    '```json\n{"setting1": "value1", "setting2": "value2"}\n```'
]

for input_data in inputs:
    result = fuzzy_validate_mapping(input_data, expected)
    print(result)
    # {'setting1': 'value1', 'setting2': 'value2'} (all produce same output)
```

## Common Pitfalls

### Pitfall 1: Forgetting to Enable Fuzzy Match

**Issue**: Expecting automatic key correction without enabling `fuzzy_match`.

```python
from lionherd_core.ln import fuzzy_validate_pydantic
from pydantic import BaseModel

class User(BaseModel):
    username: str

# Typo in key
data = {"usrname": "alice"}  # Typo: "usrname"

# Without fuzzy_match (default)
try:
    user = fuzzy_validate_pydantic(data, model_type=User)
except ValidationError as e:
    print(e)  # ValidationError: Field required [type=missing, input={'usrname': 'alice'}]
```

**Solution**: Enable `fuzzy_match=True` for key correction:

```python
user = fuzzy_validate_pydantic(
    data,
    model_type=User,
    fuzzy_match=True  # Enable fuzzy key matching
)
# User(username='alice')
```

### Pitfall 2: Using Strict Mode Without Fill Strategy

**Issue**: `strict=True` raises errors on missing keys, but no fill strategy configured.

```python
from lionherd_core.ln import fuzzy_validate_mapping

expected = ["name", "email", "age"]
data = {"name": "Bob"}

try:
    result = fuzzy_validate_mapping(data, expected, strict=True)
except ValueError as e:
    print(e)  # ValueError: Missing required keys: {'email', 'age'}
```

**Solution**: Combine `strict=True` with `handle_unmatched="fill"`:

```python
result = fuzzy_validate_mapping(
    data,
    expected,
    handle_unmatched="fill",
    fill_value=None,
    strict=True
)
# {'name': 'Bob', 'email': None, 'age': None}
```

### Pitfall 3: Confusing Default `fuzzy_match` Values

**Issue**: Different default values between `fuzzy_validate_pydantic` and `fuzzy_validate_mapping`.

```python
# fuzzy_validate_pydantic: fuzzy_match=False (default)
user = fuzzy_validate_pydantic(data, model_type=User)  # No key correction

# fuzzy_validate_mapping: fuzzy_match=True (default)
result = fuzzy_validate_mapping(data, expected)  # Key correction enabled
```

**Solution**: Explicitly set `fuzzy_match` to avoid confusion:

```python
# Be explicit
user = fuzzy_validate_pydantic(data, model_type=User, fuzzy_match=True)
result = fuzzy_validate_mapping(data, expected, fuzzy_match=True)
```

### Pitfall 4: Expecting `suppress_conversion_errors` to Catch All Errors

**Issue**: Suppression only affects conversion to dict, not validation errors.

```python
from lionherd_core.ln import fuzzy_validate_mapping

expected = ["name"]

# Conversion succeeds, but validation fails (strict mode)
data = {"wrong_key": "value"}

try:
    result = fuzzy_validate_mapping(
        data,
        expected,
        suppress_conversion_errors=True,  # Only suppresses conversion errors
        strict=True  # Validation error still raises
    )
except ValueError as e:
    print(e)  # ValueError: Missing required keys: {'name'}
```

**Solution**: `suppress_conversion_errors` only affects dict conversion, not fuzzy matching validation. Use `strict=False` or `handle_unmatched="fill"` to avoid validation errors.

### Pitfall 5: Mutating Already-Valid Models

**Issue**: Expecting validation to reprocess already-valid Pydantic instances.

```python
from lionherd_core.ln import fuzzy_validate_pydantic
from pydantic import BaseModel

class User(BaseModel):
    username: str

user = User(username="alice")

# Returns same instance (no revalidation)
result = fuzzy_validate_pydantic(user, model_type=User)
assert result is user  # True (same object)
```

**Solution**: `fuzzy_validate_pydantic` returns existing instances immediately without revalidation. If revalidation is needed, convert to dict first:

```python
# Force revalidation
result = fuzzy_validate_pydantic(user.model_dump(), model_type=User)
```

## Design Rationale

### Why Separate Functions for Pydantic vs Dict?

`fuzzy_validate_pydantic()` and `fuzzy_validate_mapping()` serve different use cases:

1. **Pydantic**: End-to-end validation for structured LLM outputs (JSON → Pydantic model)
2. **Dict**: Intermediate validation for pipelines where dict is needed before model conversion

Separation provides clear intent and avoids parameter bloat (Pydantic-specific vs dict-specific options).

### Why Default `fuzzy_match=False` for Pydantic?

Pydantic validation is already permissive (coerces types, handles aliases). Fuzzy key matching adds overhead and potential false corrections. Default to strict validation; users opt-in to fuzzy matching when needed (LLM outputs, user inputs).

### Why Default `fuzzy_match=True` for Mapping?

`fuzzy_validate_mapping()` targets noisy inputs (APIs, user submissions) where key variations are common. Default fuzzy matching improves usability for typical use cases while maintaining override option.

### Why `suppress_conversion_errors` Instead of Try/Except?

Explicit parameter provides:

1. **Clarity**: Intent is visible in function call, not hidden in exception handling
2. **Consistency**: Matches Pydantic's `validate_assignment` pattern
3. **Pipelines**: Enables graceful degradation without try/except boilerplate
4. **Debugging**: Clear distinction between conversion vs validation errors

### Why Combine Fuzzy Parse + Fuzzy Match?

LLM outputs often have **both** formatting issues (malformed JSON, markdown blocks) **and** key variations (typos, casing). Combining both strategies provides robust end-to-end parsing without manual pre-processing.

## See Also

- **Related Functions**:
  - [fuzzy_match_keys](fuzzy_match.md): Underlying fuzzy key matching implementation
  - [to_dict](to_dict.md): Universal dict conversion (used internally)
  <!-- TODO: Add link to extract_json documentation when available (issue #115) -->
- **Related Types**:
  - [Spec](../types/spec.md): Pydantic models with built-in fuzzy validation
  - [Operable](../types/operable.md): Structured LLM outputs with validation
  - [FuzzyMatchKeysParams](fuzzy_match.md#class-fuzzymatchkeysparams): Reusable fuzzy match configuration
  <!-- TODO: Add link to String Similarity documentation when available (issue #115) -->

## Examples

```python
# Standard imports for ln.fuzzy_validate examples
from lionherd_core.ln import (
    fuzzy_validate_pydantic,
    fuzzy_validate_mapping,
    FuzzyMatchKeysParams
)
from pydantic import BaseModel
```

### Example 1: LLM JSON Parsing Pipeline

````python

class AgentTask(BaseModel):
    task_name: str
    priority: str
    assigned_to: str
    estimated_hours: int

# LLM output with markdown, typos, and formatting issues
llm_response = """
I've created the task for you:

```json
{
    "taskName": "Implement authentication",
    "Priority": "HIGH",
    "assignedTo": "alice",
    "estimatedHours": 8
}
```

Let me know if you need changes!
"""

# Parse with fuzzy extraction and key matching

task = fuzzy_validate_pydantic(
    llm_response,
    model_type=AgentTask,
    fuzzy_parse=True,   # Extract JSON from markdown
    fuzzy_match=True,   # Correct key variations
    fuzzy_match_params={"similarity_threshold": 0.75}
)

print(task)

# AgentTask(

# task_name='Implement authentication'

# priority='HIGH'

# assigned_to='alice'

# estimated_hours=8

# )

````

### Example 2: API Response Normalization

```python
from lionherd_core.ln import fuzzy_validate_mapping

# Expected response schema
schema = ["user_id", "full_name", "email_address", "account_status"]

# Third-party API with inconsistent naming
api_responses = [
    # Response 1: camelCase
    '{"userId": "123", "fullName": "Alice", "emailAddress": "alice@ex.com", "accountStatus": "active"}',

    # Response 2: snake_case with typos
    '{"user_id": "456", "fullname": "Bob", "email_addr": "bob@ex.com", "acct_status": "active"}',

    # Response 3: Mixed with extra fields
    '{"UserId": "789", "full_name": "Charlie", "Email": "charlie@ex.com", "status": "inactive", "extra": "data"}',
]

# Normalize all responses
normalized = []
for response in api_responses:
    result = fuzzy_validate_mapping(
        response,
        schema,
        similarity_threshold=0.75,  # Lenient for abbreviations
        handle_unmatched="remove",  # Discard unknown keys
        fill_value="Unknown"        # Default for missing keys
    )
    normalized.append(result)

print(normalized)
# [
#     {'user_id': '123', 'full_name': 'Alice', 'email_address': 'alice@ex.com', 'account_status': 'active'},
#     {'user_id': '456', 'full_name': 'Bob', 'email_address': 'bob@ex.com', 'account_status': 'active'},
#     {'user_id': '789', 'full_name': 'Charlie', 'email_address': 'charlie@ex.com', 'account_status': 'inactive'}
# ]
```

### Example 3: Form Validation with Defaults

```python
from lionherd_core.ln import fuzzy_validate_mapping

# Required form fields
form_schema = ["name", "email", "age", "newsletter", "terms_accepted"]

# User submissions with missing/extra fields
submissions = [
    {"name": "Alice", "email": "alice@ex.com", "ag": 25, "newslettr": True},
    {"nam": "Bob", "email": "bob@ex.com"},
    {"name": "Charlie", "email": "charlie@ex.com", "age": 30, "extra_field": "ignored"}
]

# Validate with defaults
validated_submissions = []
for submission in submissions:
    validated = fuzzy_validate_mapping(
        submission,
        form_schema,
        similarity_threshold=0.85,
        handle_unmatched="force",  # Strict schema, discard unknowns
        fill_mapping={
            "newsletter": False,      # Default opt-out
            "terms_accepted": False   # Default not accepted
        },
        fill_value=None,  # Default for other missing fields
        strict=True       # All fields required
    )
    validated_submissions.append(validated)

print(validated_submissions)
# [
#     {'name': 'Alice', 'email': 'alice@ex.com', 'age': 25, 'newsletter': True, 'terms_accepted': False},
#     {'name': 'Bob', 'email': 'bob@ex.com', 'age': None, 'newsletter': False, 'terms_accepted': False},
#     {'name': 'Charlie', 'email': 'charlie@ex.com', 'age': 30, 'newsletter': False, 'terms_accepted': False}
# ]
```

### Example 4: Graceful Degradation Pipeline

```python
from lionherd_core.ln import fuzzy_validate_mapping
from typing import List

def process_batch(inputs: List[Any], schema: List[str]) -> List[dict]:
    """Process batch with graceful degradation."""
    results = []

    for i, input_data in enumerate(inputs):
        result = fuzzy_validate_mapping(
            input_data,
            schema,
            suppress_conversion_errors=True,  # Return {} on bad inputs
            handle_unmatched="fill",
            fill_value="MISSING"
        )

        # Track which inputs failed conversion
        if not result:
            print(f"Warning: Input {i} failed conversion, skipping")
            continue

        results.append(result)

    return results

# Mixed valid/invalid inputs
inputs = [
    {"name": "Alice", "age": 25},
    "not a valid json string",
    {"nam": "Bob", "age": 30},
    12345,  # Not convertible
    {"name": "Charlie"}
]

schema = ["name", "age"]

# Process with graceful degradation
results = process_batch(inputs, schema)
# Warning: Input 1 failed conversion, skipping
# Warning: Input 3 failed conversion, skipping

print(results)
# [
#     {'name': 'Alice', 'age': 25},
#     {'name': 'Bob', 'age': 30},
#     {'name': 'Charlie', 'age': 'MISSING'}
# ]
```

### Example 5: Multi-Stage Validation

```python
from lionherd_core.ln import fuzzy_validate_pydantic, fuzzy_validate_mapping
from pydantic import BaseModel, field_validator

class StrictUser(BaseModel):
    username: str
    email: str
    age: int

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v

# Stage 1: Normalize keys with fuzzy matching
raw_input = {
    "usrname": "alice",
    "Email": "alice@example.com",
    "ag": "30"  # String instead of int
}

# First normalize keys to dict
normalized = fuzzy_validate_mapping(
    raw_input,
    keys=["username", "email", "age"],
    fuzzy_match=True,
    handle_unmatched="remove"
)
# {'username': 'alice', 'email': 'alice@example.com', 'age': '30'}

# Stage 2: Validate with Pydantic (type coercion + custom validators)
try:
    user = fuzzy_validate_pydantic(
        normalized,
        model_type=StrictUser,
        fuzzy_parse=False  # Already a dict, no JSON extraction needed
    )
    print(user)
    # StrictUser(username='alice', email='alice@example.com', age=30)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

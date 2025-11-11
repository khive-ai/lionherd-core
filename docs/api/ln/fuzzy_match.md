# Fuzzy Match

> Dictionary key validation and correction using fuzzy string matching

## Overview

The fuzzy matching module provides utilities for validating and correcting dictionary keys using similarity algorithms. It handles common scenarios where dictionary keys may be misspelled, use different casing, or have slight variations from expected keys.

**Key Capabilities:**

- **Fuzzy String Matching**: Correct misspelled keys using similarity algorithms (Jaro-Winkler, Levenshtein, etc.)
- **Flexible Unmatched Handling**: Configurable strategies for keys that don't match (ignore, raise, remove, fill, force)
- **Exact Match Priority**: Exact matches take precedence before fuzzy matching
- **Fill Support**: Add missing expected keys with default or custom values
- **Strict Mode**: Enforce presence of all expected keys
- **Type-Safe Parameters**: Immutable dataclass configuration for reusable validation logic

**When to Use Fuzzy Match:**

- Validating user-provided dictionaries against a schema
- Correcting LLM output keys that may have slight variations
- Normalizing API request/response keys
- Data migration with inconsistent key naming
- Form validation where field names may vary

**When NOT to Use Fuzzy Match:**

- Performance-critical paths where exact matching suffices
- Keys that should never be auto-corrected (security tokens, IDs)
- Schemas where any variation is an error (use strict validation instead)
- Very large dictionaries (O(n*m) fuzzy matching overhead)

## Module Contents

### Types

**HandleUnmatched** : Literal type

Strategy for handling keys that don't match expected keys.

```python
HandleUnmatched = Literal["ignore", "raise", "remove", "fill", "force"]
```

**Values:**

- `"ignore"`: Keep unmatched keys in output (default)
- `"raise"`: Raise ValueError if unmatched keys found
- `"remove"`: Exclude unmatched keys from output
- `"fill"`: Add missing expected keys with fill values, keep unmatched keys
- `"force"`: Add missing expected keys with fill values, discard unmatched keys

## Function Signature

```python
from lionherd_core.ln import fuzzy_match_keys

def fuzzy_match_keys(
    d_: dict[str, Any],
    keys: KeysLike,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | SimilarityAlgo | SimilarityFunc = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: HandleUnmatched = "ignore",
    fill_value: Any = Unset,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]: ...
```

## Parameters

### Positional-Only Parameters

**d_** : dict of {str : Any}

Input dictionary to validate and correct.

- Validation: Must be a dict, raises TypeError otherwise
- Behavior: Original dict is not modified; function returns corrected copy
- Note: Empty dict returns empty dict copy

**keys** : KeysLike (list[str] or dict-like with .keys())

Expected keys to validate against.

- Accepted types: List of strings or any object with `.keys()` method (dict, Pydantic model)
- Validation: Cannot be None, raises TypeError if None
- Behavior: If empty, returns copy of original dict

### Keyword-Only Parameters

**similarity_algo** : SIMILARITY_TYPE or SimilarityAlgo or SimilarityFunc, default "jaro_winkler"

Algorithm for computing string similarity.

- String options: `"jaro_winkler"`, `"levenshtein"`, `"jaro"`, `"hamming"`, `"jaccard"`, etc.
- Enum: Can use `SimilarityAlgo` enum values
- Custom: Can pass custom similarity function `(str, str) -> float`
- Validation: Raises ValueError if string not in `SIMILARITY_ALGO_MAP`
- Default: Jaro-Winkler is optimized for short strings with typos

**similarity_threshold** : float, default 0.85

Minimum similarity score (0.0-1.0) for considering keys a match.

- Range: Must be between 0.0 (no similarity) and 1.0 (identical)
- Validation: Raises ValueError if out of range
- Behavior: Higher values require closer matches (more strict)
- Typical values: 0.7-0.8 (lenient), 0.85-0.9 (moderate), 0.95+ (strict)
- Default: 0.85 balances correction without false matches

**fuzzy_match** : bool, default True

Enable fuzzy matching for keys that don't match exactly.

- True: Apply similarity algorithm to unmatched keys (default)
- False: Only exact matches considered; unmatched keys handled by `handle_unmatched`
- Performance: Set to False for faster exact-match-only validation

**handle_unmatched** : HandleUnmatched, default "ignore"

Strategy for handling keys that don't match expected keys.

- `"ignore"`: Keep unmatched keys in output (permissive)
- `"raise"`: Raise ValueError with list of unmatched keys (strict validation)
- `"remove"`: Exclude unmatched keys from output (filter)
- `"fill"`: Add missing expected keys with fill values, keep unmatched keys (augment)
- `"force"`: Add missing expected keys, discard unmatched keys (strict schema)
- Default: `"ignore"` allows extra keys (backward compatible)

**fill_value** : Any, default Unset

Default value for missing expected keys when `handle_unmatched` is "fill" or "force".

- Used when: Expected key is missing and not in `fill_mapping`
- Default: `Unset` sentinel (check with `value is Unset`)
- Examples: `None`, `""`, `0`, `[]`, custom defaults
- Note: Applies to all missing keys unless overridden by `fill_mapping`

**fill_mapping** : dict of {str : Any} or None, default None

Custom values for specific missing expected keys (overrides `fill_value`).

- Format: `{"key_name": value, ...}`
- Behavior: Missing keys in mapping use `fill_value` instead
- Priority: `fill_mapping` value > `fill_value` > Unset
- Use case: Different defaults for different keys
- Example: `{"age": 0, "name": "Unknown", "tags": []}`

**strict** : bool, default False

Raise ValueError if any expected keys are missing after matching.

- True: Enforce all expected keys must be present (or filled)
- False: Allow missing expected keys (default)
- Validation: Raises ValueError with list of missing keys
- Note: Checked after fuzzy matching and filling

## Returns

**dict[str, Any]**

New dictionary with corrected keys based on validation rules.

- Original dict is not modified (creates new dict)
- Keys are renamed based on fuzzy matching results
- Unmatched keys handled according to `handle_unmatched`
- Missing expected keys filled according to `fill_value` / `fill_mapping`

## Raises

### TypeError

- If `d_` is not a dict
- If `keys` is None

### ValueError

- If `similarity_threshold` not in range [0.0, 1.0]
- If `similarity_algo` string not recognized
- If `handle_unmatched="raise"` and unmatched keys found
- If `strict=True` and expected keys missing

## Algorithm Workflow

The function applies a two-pass matching strategy:

### Pass 1: Exact Matching

```python
# First pass: exact matches take priority
for key in d_:
    if key in expected_keys:
        matched_keys.add(key)
        output[key] = d_[key]
```

### Pass 2: Fuzzy Matching (if enabled)

```python
# Second pass: fuzzy match remaining keys
if fuzzy_match:
    for unmatched_key in remaining_input_keys:
        best_match = find_most_similar(
            unmatched_key,
            remaining_expected_keys,
            algorithm=similarity_algo,
            threshold=similarity_threshold
        )
        if best_match:
            output[best_match] = d_[unmatched_key]  # Rename to expected key
```

### Pass 3: Unmatched Handling

Applies `handle_unmatched` strategy to keys that didn't match in Pass 1 or Pass 2.

## Class: FuzzyMatchKeysParams

Immutable dataclass configuration for `fuzzy_match_keys()` function.

### Signature

```python
from lionherd_core.ln import FuzzyMatchKeysParams

@dataclass(slots=True, init=False, frozen=True)
class FuzzyMatchKeysParams(Params):
    """Reusable parameter configuration for fuzzy_match_keys()."""

    similarity_algo: SIMILARITY_TYPE | SimilarityAlgo | SimilarityFunc = "jaro_winkler"
    similarity_threshold: float = 0.85
    fuzzy_match: bool = True
    handle_unmatched: HandleUnmatched = "ignore"
    fill_value: Any = Unset
    fill_mapping: dict[str, Any] | Any = Unset
    strict: bool = False

    def __call__(self, d_: dict[str, Any], keys: KeysLike) -> dict[str, Any]:
        """Apply configured parameters to fuzzy_match_keys()."""
```

### Attributes

All attributes match `fuzzy_match_keys()` keyword parameters (see [Parameters](#parameters) section).

### Methods

#### `__call__()`

Apply configured parameters to `fuzzy_match_keys()`.

**Signature:**

```python
def __call__(self, d_: dict[str, Any], keys: KeysLike) -> dict[str, Any]: ...
```

**Parameters:**

- `d_` (dict[str, Any]): Input dictionary to validate
- `keys` (KeysLike): Expected keys

**Returns:**

- dict[str, Any]: Corrected dictionary (delegates to `fuzzy_match_keys()`)

**Examples:**

```python
from lionherd_core.ln import FuzzyMatchKeysParams

# Create reusable validator
strict_validator = FuzzyMatchKeysParams(
    similarity_threshold=0.9,
    handle_unmatched="raise",
    strict=True
)

# Apply to multiple dictionaries
result1 = strict_validator({"usrname": "alice"}, keys=["username"])
result2 = strict_validator({"emaill": "bob@example.com"}, keys=["email"])
```

## Usage Patterns

### Basic Usage

```python
from lionherd_core.ln import fuzzy_match_keys

# Define expected keys
expected = ["username", "email", "age"]

# Input with typos
user_input = {
    "usrname": "alice",      # Misspelled
    "email": "alice@ex.com", # Exact match
    "ag": 30,                # Misspelled
}

# Correct keys
result = fuzzy_match_keys(user_input, expected)
# {
#     "username": "alice",  # Corrected from "usrname"
#     "email": "alice@ex.com",
#     "age": 30             # Corrected from "ag"
# }
```

### Strict Validation

```python
from lionherd_core.ln import fuzzy_match_keys

expected = ["id", "name", "status"]

# Missing required field
data = {"id": "123", "name": "Task A"}

try:
    result = fuzzy_match_keys(
        data,
        expected,
        strict=True,  # Enforce all keys present
        handle_unmatched="raise"  # No extra keys allowed
    )
except ValueError as e:
    print(e)  # "Missing required keys: {'status'}"
```

### Filling Missing Keys

```python
from lionherd_core.ln import fuzzy_match_keys

expected = ["name", "age", "city"]

# Incomplete data
data = {"name": "Bob"}

# Fill missing keys with defaults
result = fuzzy_match_keys(
    data,
    expected,
    handle_unmatched="fill",
    fill_value=None,
    fill_mapping={"age": 0, "city": "Unknown"}
)
# {
#     "name": "Bob",
#     "age": 0,          # From fill_mapping
#     "city": "Unknown"  # From fill_mapping
# }
```

### Filtering Unknown Keys

```python
from lionherd_core.ln import fuzzy_match_keys

expected = ["username", "email"]

# Data with extra fields
data = {
    "username": "alice",
    "email": "alice@example.com",
    "internal_id": "xyz",  # Unknown field
    "debug_flag": True     # Unknown field
}

# Remove unknown keys
result = fuzzy_match_keys(data, expected, handle_unmatched="remove")
# {
#     "username": "alice",
#     "email": "alice@example.com"
# }
```

### Custom Similarity Algorithm

```python
from lionherd_core.ln import fuzzy_match_keys
from lionherd_core.libs.string_handlers import SimilarityAlgo

# Use Levenshtein distance instead of Jaro-Winkler
result = fuzzy_match_keys(
    {"user_name": "alice"},
    ["username"],
    similarity_algo="levenshtein",
    similarity_threshold=0.8
)

# Or use enum
result = fuzzy_match_keys(
    {"user_name": "alice"},
    ["username"],
    similarity_algo=SimilarityAlgo.LEVENSHTEIN,
    similarity_threshold=0.8
)

# Or custom function
def custom_similarity(s1: str, s2: str) -> float:
    """Custom similarity: 1.0 if prefixes match, else 0.0."""
    return 1.0 if s1.startswith(s2[:3]) or s2.startswith(s1[:3]) else 0.0

result = fuzzy_match_keys(
    {"usr_id": "123"},
    ["user_id"],
    similarity_algo=custom_similarity,
    similarity_threshold=0.5
)
```

### Reusable Validator

```python
from lionherd_core.ln import FuzzyMatchKeysParams

# Configure once
api_validator = FuzzyMatchKeysParams(
    similarity_threshold=0.9,
    handle_unmatched="remove",  # Strip unknown keys
    strict=False
)

# Reuse across requests
requests = [
    {"usrname": "alice", "extra": "data"},
    {"email": "bob@ex.com", "debug": True},
]

expected_keys = ["username", "email"]

for req in requests:
    validated = api_validator(req, expected_keys)
    print(validated)
```

### LLM Output Normalization

```python
from lionherd_core.ln import fuzzy_match_keys

# Expected schema
schema = ["title", "description", "priority", "tags"]

# LLM sometimes varies keys slightly
llm_output = {
    "Title": "Bug fix",              # Capitalization
    "descr": "Fix login issue",      # Abbreviation
    "priority_level": "high",        # Extra suffix
    "tag": ["bug", "auth"]           # Singular vs plural
}

# Normalize with lenient matching
normalized = fuzzy_match_keys(
    llm_output,
    schema,
    similarity_threshold=0.75,  # More lenient for variations
    handle_unmatched="ignore"   # Keep extra fields
)
# {
#     "title": "Bug fix",
#     "description": "Fix login issue",
#     "priority": "high",
#     "tags": ["bug", "auth"]
# }
```

## Common Pitfalls

### Pitfall 1: Threshold Too Low

**Issue**: Low threshold causes false matches.

```python
result = fuzzy_match_keys(
    {"x": 1, "y": 2},
    ["username", "email"],
    similarity_threshold=0.3  # Too low!
)
# May incorrectly match "x" → "username" due to low threshold
```

**Solution**: Use threshold ≥ 0.7 for reasonable matching. Test with actual data.

### Pitfall 2: Forgetting `handle_unmatched`

**Issue**: Unmatched keys are kept by default, polluting output.

```python
result = fuzzy_match_keys(
    {"name": "alice", "internal_flag": True},
    ["name"]
)
# {"name": "alice", "internal_flag": True}  # Extra key kept
```

**Solution**: Use `handle_unmatched="remove"` or `"force"` to enforce schema strictly.

### Pitfall 3: Using `strict` Without `fill`

**Issue**: `strict=True` raises error when keys are missing, but no fill strategy.

```python
fuzzy_match_keys(
    {"name": "alice"},
    ["name", "age"],
    strict=True  # Raises ValueError: Missing required keys: {'age'}
)
```

**Solution**: Combine with `handle_unmatched="fill"` to auto-fill missing keys:

```python
fuzzy_match_keys(
    {"name": "alice"},
    ["name", "age"],
    handle_unmatched="fill",
    fill_value=None,
    strict=True  # Now passes: missing keys filled
)
```

### Pitfall 4: Performance on Large Dicts

**Issue**: Fuzzy matching is O(n*m) where n = input keys, m = expected keys.

```python
# Slow for large dicts
large_dict = {f"key_{i}": i for i in range(10000)}
large_expected = [f"field_{i}" for i in range(5000)]

result = fuzzy_match_keys(large_dict, large_expected)  # Slow!
```

**Solution**: Disable fuzzy matching if exact matches suffice:

```python
result = fuzzy_match_keys(
    large_dict,
    large_expected,
    fuzzy_match=False  # Fast exact-match only
)
```

### Pitfall 5: Mutating Original Dict

**Issue**: Expecting original dict to be modified.

```python
original = {"usrname": "alice"}
result = fuzzy_match_keys(original, ["username"])

print(original)  # {"usrname": "alice"} - unchanged!
print(result)    # {"username": "alice"} - corrected copy
```

**Solution**: Function returns new dict; original is never modified. Use returned value.

## Design Rationale

### Why Two-Pass Matching?

Exact matches take priority before fuzzy matching to prevent:

1. **False Corrections**: Exact match "email" shouldn't fuzzy-match to "email_backup"
2. **Performance**: Exact matches are O(1) hash lookups vs O(n) fuzzy comparisons
3. **Predictability**: Deterministic behavior when both exact and fuzzy matches possible

### Why Immutable Params Class?

`FuzzyMatchKeysParams` provides:

1. **Reusability**: Configure once, apply to many dicts
2. **Type Safety**: Frozen dataclass prevents accidental mutation
3. **Serialization**: Params can be serialized/deserialized for config files
4. **Composition**: Embed in larger validation pipelines

Follows lionherd-core pattern of separating data (Params) from logic (functions).

### Why `handle_unmatched` Modes?

Different use cases require different strategies:

- **API validation**: `"remove"` or `"raise"` (strict schema enforcement)
- **LLM output**: `"ignore"` (allow extra fields)
- **Data migration**: `"fill"` (add missing fields with defaults)
- **Form normalization**: `"force"` (strict schema + defaults)

Single parameter covers all scenarios without multiple functions.

### Why Default Jaro-Winkler?

Jaro-Winkler is optimal for dictionary keys because:

1. **Short Strings**: Keys are typically 5-20 characters
2. **Prefix Bias**: Common typos occur at end ("username" → "usrnam")
3. **Transpositions**: Handles swapped characters ("email" → "emial")
4. **Performance**: Faster than Levenshtein for short strings

Users can override with `similarity_algo` for specific needs.

## See Also

- **Related Modules**:
  <!-- TODO: Add link to String Similarity documentation when available (issue #115) -->
  - [Spec](../types/spec.md): Pydantic model validation with fuzzy key support
  - [Operable](../types/operable.md): Structured LLM outputs with validation
- **Related Types**:
  - [KeysLike](../types/base.md#keyslike): Type definition for key sources
  - [Unset](../types/sentinel.md): Sentinel value for fill_value default

## Examples

```python
# Standard imports for ln.fuzzy_match examples
from lionherd_core.ln import fuzzy_match_keys, FuzzyMatchKeysParams
from pydantic import BaseModel
```

### Example 1: API Request Validation

```python

# API expects specific keys
api_schema = ["user_id", "action", "timestamp"]

# Client sends with typos
request = {
    "userId": "12345",      # camelCase instead of snake_case
    "acton": "login",       # Typo
    "timestmp": 1234567890  # Typo
}

# Validate and correct
validated = fuzzy_match_keys(
    request,
    api_schema,
    similarity_threshold=0.85,
    handle_unmatched="raise"  # Reject unknown keys
)
# {
#     "user_id": "12345",
#     "action": "login",
#     "timestamp": 1234567890
# }
```

### Example 2: LLM JSON Parsing

```python
from lionherd_core.ln import fuzzy_match_keys
import json

# Expected LLM output schema
schema = ["task_name", "priority", "assignee", "due_date"]

# LLM returns slightly different keys
llm_json = """{
    "taskName": "Implement feature",
    "Priority": "HIGH",
    "assigned_to": "alice",
    "dueDate": "2025-11-15"
}"""

llm_output = json.loads(llm_json)

# Normalize keys
normalized = fuzzy_match_keys(
    llm_output,
    schema,
    similarity_threshold=0.75,  # Lenient for LLM variations
    handle_unmatched="remove"   # Discard unknown keys
)
# {
#     "task_name": "Implement feature",
#     "priority": "HIGH",
#     "assignee": "alice",
#     "due_date": "2025-11-15"
# }
```

### Example 3: Data Migration with Defaults

```python
from lionherd_core.ln import fuzzy_match_keys

# New schema requires additional fields
new_schema = ["id", "name", "email", "status", "created_at"]

# Legacy data missing some fields
legacy_records = [
    {"id": "1", "name": "Alice", "email": "alice@example.com"},
    {"id": "2", "nam": "Bob", "email": "bob@example.com"},  # Typo
    {"id": "3", "name": "Charlie"},  # Missing email
]

# Migrate with defaults
migrated = []
for record in legacy_records:
    migrated_record = fuzzy_match_keys(
        record,
        new_schema,
        handle_unmatched="fill",
        fill_mapping={
            "status": "active",
            "created_at": "2025-11-09T00:00:00Z"
        },
        fill_value=None
    )
    migrated.append(migrated_record)

# migrated[0]:
# {
#     "id": "1",
#     "name": "Alice",
#     "email": "alice@example.com",
#     "status": "active",
#     "created_at": "2025-11-09T00:00:00Z"
# }

# migrated[1]:
# {
#     "id": "2",
#     "name": "Bob",  # Corrected from "nam"
#     "email": "bob@example.com",
#     "status": "active",
#     "created_at": "2025-11-09T00:00:00Z"
# }

# migrated[2]:
# {
#     "id": "3",
#     "name": "Charlie",
#     "email": None,  # fill_value
#     "status": "active",
#     "created_at": "2025-11-09T00:00:00Z"
# }
```

### Example 4: Form Field Normalization

```python
from lionherd_core.ln import FuzzyMatchKeysParams

# Configure reusable form validator
form_validator = FuzzyMatchKeysParams(
    similarity_threshold=0.9,
    handle_unmatched="force",  # Strict schema, discard unknowns
    fill_mapping={
        "newsletter": False,
        "marketing_consent": False
    },
    strict=True  # All fields required
)

# Expected form fields
form_schema = ["name", "email", "newsletter", "marketing_consent"]

# User submissions with variations
submissions = [
    {"name": "Alice", "email": "alice@ex.com", "newslettr": True},
    {"nam": "Bob", "emal": "bob@ex.com"},
    {"name": "Charlie", "email": "charlie@ex.com", "extra": "data"},
]

for submission in submissions:
    try:
        validated = form_validator(submission, form_schema)
        print(f"Valid: {validated}")
    except ValueError as e:
        print(f"Invalid: {e}")

# Output:
# Valid: {'name': 'Alice', 'email': 'alice@ex.com', 'newsletter': True, 'marketing_consent': False}
# Valid: {'name': 'Bob', 'email': 'bob@ex.com', 'newsletter': False, 'marketing_consent': False}
# Valid: {'name': 'Charlie', 'email': 'charlie@ex.com', 'newsletter': False, 'marketing_consent': False}
```

### Example 5: Multi-Language Field Mapping

```python
from lionherd_core.ln import fuzzy_match_keys

# International API with field name variations
en_schema = ["first_name", "last_name", "phone_number"]

# French API response
fr_response = {
    "prénom": "Jean",
    "nom": "Dupont",
    "numéro_téléphone": "+33123456789"
}

# ASCII-normalized input for matching
normalized_input = {
    "prenom": "Jean",
    "nom": "Dupont",
    "numero_telephone": "+33123456789"
}

# Match to English schema
result = fuzzy_match_keys(
    normalized_input,
    en_schema,
    similarity_threshold=0.7,
    handle_unmatched="remove"
)
# {
#     "first_name": "Jean",    # From "prenom"
#     "last_name": "Dupont",   # From "nom"
#     "phone_number": "+33123456789"  # From "numero_telephone"
# }
```

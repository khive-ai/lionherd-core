# Fuzzy JSON Parser

> Fault-tolerant JSON parser with automatic error correction for quotes, spacing, and brackets

## Overview

The `fuzzy_json` module provides robust JSON parsing with automatic error correction for common formatting issues. It attempts multiple parsing strategies with progressive error correction, making it ideal for handling LLM-generated or user-provided JSON strings that may contain formatting errors.

**Key Capabilities:**

- **Multi-Stage Parsing**: Progressive error correction with fallback strategies
- **Quote Normalization**: Automatic conversion of single quotes to double quotes
- **Bracket Balancing**: Intelligent fixing of unmatched opening/closing brackets
- **Whitespace Handling**: Normalization of spacing and removal of trailing commas
- **Key Quoting**: Automatic quoting of unquoted object keys
- **Escape Sequence Awareness**: Proper handling of escaped characters in strings

**When to Use Fuzzy JSON:**

- Parsing LLM-generated JSON responses with formatting inconsistencies
- Processing user-provided JSON with unknown formatting quality
- Handling JSON from external APIs that may violate strict JSON specifications
- Recovering from JSON strings with missing closing brackets
- Converting Python dict literals (single quotes) to valid JSON

**When NOT to Use Fuzzy JSON:**

- Strict JSON validation where errors should fail fast
- Performance-critical paths where parsing overhead matters
- JSON from trusted sources known to be well-formed (use `orjson.loads()` directly)
- Scenarios requiring detailed error messages for debugging malformed JSON

## Function Reference

### Primary Function

#### `fuzzy_json()`

Parse JSON string with fuzzy error correction through multiple strategies.

**Signature:**

```python
def fuzzy_json(str_to_parse: str, /) -> dict[str, Any] | list[dict[str, Any]]: ...
```

**Parameters:**

- `str_to_parse` (str, positional-only): JSON string to parse, potentially malformed

**Returns:**

- dict[str, Any] or list[dict[str, Any]]: Parsed JSON as dictionary or list of dictionaries

**Raises:**

- TypeError: If `str_to_parse` is not a string
- ValueError: If input is empty or all parsing strategies fail

**Examples:**

```python
>>> from lionherd_core.libs.string_handlers import fuzzy_json

# Standard JSON (direct parse)
>>> fuzzy_json('{"key": "value"}')
{'key': 'value'}

# Single quotes (normalized to double quotes)
>>> fuzzy_json("{'name': 'Alice', 'age': 30}")
{'name': 'Alice', 'age': 30}

# Unquoted keys (automatically quoted)
>>> fuzzy_json('{name: "Bob", role: "admin"}')
{'name': 'Bob', 'role': 'admin'}

# Missing closing bracket (auto-fixed)
>>> fuzzy_json('{"items": [{"id": 1}, {"id": 2}')
{'items': [{'id': 1}, {'id': 2}]}

# Trailing commas (removed)
>>> fuzzy_json('{"a": 1, "b": 2,}')
{'a': 1, 'b': 2}

# Multiple issues (progressive correction)
>>> fuzzy_json("{name: 'Charlie', tags: ['ai', 'ml',]")
{'name': 'Charlie', 'tags': ['ai', 'ml']}
```

**Parsing Strategy:**

The function attempts parsing in this order:

1. **Direct Parse**: Try `orjson.loads()` on original string (fast path for valid JSON)
2. **Quote Normalization**: Replace single quotes with double quotes, clean whitespace, remove trailing commas
3. **Bracket Fixing**: Balance unmatched brackets by adding missing closing brackets

**See Also:**

- `fix_json_string()`: Bracket balancing utility (can be used standalone)
- `orjson.loads()`: Underlying fast JSON parser

**Notes:**

Uses `orjson` for high-performance JSON parsing once the string is corrected. The multi-stage approach minimizes overhead for well-formed JSON (single parsing attempt) while providing robust error recovery for malformed input.

**Limitations:**

- Cannot fix structurally invalid JSON (e.g., misplaced commas, invalid escape sequences not addressed by normalization)
- Does not validate JSON schema or data types
- May incorrectly fix intentionally malformed strings (e.g., strings containing literal bracket characters)
- Extra closing brackets or mismatched bracket types raise `ValueError`

### Helper Functions

#### `fix_json_string()`

> **⚠️ Internal API**: This function is an internal helper and not part of the public API. Use `fuzzy_json()` for general JSON parsing. Direct use of this function is not recommended unless you specifically need bracket balancing in isolation.

Fix JSON string by balancing unmatched brackets.

**Signature:**

```python
def fix_json_string(str_to_parse: str, /) -> str: ...
```

**Parameters:**

- `str_to_parse` (str, positional-only): JSON string to fix

**Returns:**

- str: JSON string with balanced brackets

**Raises:**

- ValueError: If input is empty, has extra closing brackets, or mismatched bracket types

**Examples:**

```python
>>> from lionherd_core.libs.string_handlers._fuzzy_json import fix_json_string  # Internal API

# Missing closing brackets (added)
>>> fix_json_string('{"a": [1, 2, 3')
'{"a": [1, 2, 3]}'

# Nested missing brackets
>>> fix_json_string('{"outer": {"inner": [1, 2')
'{"outer": {"inner": [1, 2]}}'

# Already balanced (no change)
>>> fix_json_string('{"complete": true}')
'{"complete": true}'

# Extra closing bracket (raises error)
>>> fix_json_string('{"a": 1}}')
ValueError: Extra closing bracket found.

# Mismatched brackets (raises error)
>>> fix_json_string('{"a": [1, 2}')
ValueError: Mismatched brackets.
```

**Algorithm:**

1. Traverse string character by character
2. Skip escaped characters and string contents (inside double quotes)
3. Track opening brackets `{`, `[` on a stack
4. Validate closing brackets `}`, `]` match stack top
5. Append missing closing brackets in reverse order

**See Also:**

- `fuzzy_json()`: Main parsing function that uses this for bracket fixing

**Notes:**

This function is exposed for standalone use when you need bracket balancing without full fuzzy parsing. It properly handles:

- Escaped characters (`\"`, `\\`)
- String contents (skips bracket-like characters inside strings)
- Nested structures (tracks stack of open brackets)

**String-Awareness Example:**

```python
# Brackets inside strings are ignored
>>> fix_json_string('{"msg": "use { and } carefully"')
'{"msg": "use { and } carefully"}'

# Escaped quotes don't break string detection
>>> fix_json_string('{"quote": "She said \\"hello\\""}')
'{"quote": "She said \\"hello\\""}'
```

### Private Functions

#### `_check_valid_str()`

Validate input is a non-empty string.

**Signature:**

```python
def _check_valid_str(str_to_parse: str, /) -> None: ...
```

**Parameters:**

- `str_to_parse` (str, positional-only): String to validate

**Raises:**

- TypeError: If input is not a string
- ValueError: If string is empty or whitespace-only

**Notes:**

Internal validation function used by `fuzzy_json()`. Not intended for external use.

#### `_clean_json_string()`

Normalize JSON string formatting.

**Signature:**

```python
def _clean_json_string(s: str) -> str: ...
```

**Parameters:**

- `s` (str): JSON string to clean

**Returns:**

- str: Cleaned JSON string

**Notes:**

Internal cleaning function that performs:

1. Replace unescaped single quotes with double quotes (`'` → `"`)
2. Collapse multiple whitespaces to single space
3. Remove trailing commas before closing brackets/braces
4. Quote unquoted object keys (e.g., `{key: value}` → `{"key": value}`)

Not intended for external use. Used by `fuzzy_json()` as part of the progressive correction strategy.

## Usage Patterns

### Basic LLM Response Parsing

```python
from lionherd_core.libs.string_handlers import fuzzy_json

# LLM returns JSON with formatting issues
llm_response = """
{
  name: 'GPT-4',
  capabilities: ['reasoning', 'coding', 'analysis',],
  config: {
    temperature: 0.7,
    max_tokens: 4000
  }
"""  # Missing closing brackets, single quotes, trailing comma

# Parse with automatic correction
parsed = fuzzy_json(llm_response)
print(parsed)
# {'name': 'GPT-4', 'capabilities': ['reasoning', 'coding', 'analysis'], 'config': {'temperature': 0.7, 'max_tokens': 4000}}
```

### API Response Recovery

```python
from lionherd_core.libs.string_handlers import fuzzy_json
import requests

# External API returns malformed JSON
response = requests.get("https://api.example.com/data")
raw_json = response.text  # May have formatting issues

try:
    # Attempt fuzzy parsing
    data = fuzzy_json(raw_json)
    process_data(data)
except ValueError:
    # All strategies failed - log and handle
    logger.error(f"Cannot parse JSON: {raw_json}")
    raise
```

### Batch Processing with Fallback

```python
from lionherd_core.libs.string_handlers import fuzzy_json
import orjson

def parse_json_safe(text: str) -> dict | None:
    """Parse JSON with fuzzy fallback."""
    # Try fast path first
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        pass

    # Fallback to fuzzy parsing
    try:
        return fuzzy_json(text)
    except (ValueError, TypeError):
        return None

# Process batch of potentially malformed JSON
json_strings = [...]
results = [parse_json_safe(s) for s in json_strings]
valid_results = [r for r in results if r is not None]
```

### Standalone Bracket Fixing

> **⚠️ Advanced Usage**: Shows internal API usage for bracket fixing only. Prefer `fuzzy_json()` for general use.

```python
from lionherd_core.libs.string_handlers._fuzzy_json import fix_json_string  # Internal API
import orjson

# Use bracket fixing separately (when you know other formatting is correct)
partial_json = '{"data": [1, 2, 3'  # Only missing closing brackets
fixed = fix_json_string(partial_json)  # '{"data": [1, 2, 3]}'
parsed = orjson.loads(fixed)
```

### Common Pitfalls

#### Pitfall 1: Assuming All Malformed JSON Is Fixable

**Issue**: Not all JSON errors are correctable by fuzzy parsing.

```python
# Structural errors that fuzzy_json cannot fix
invalid_json = '{"a": 1,, "b": 2}'  # Double comma
fuzzy_json(invalid_json)  # ValueError: Invalid JSON string

invalid_json = '{"a": 1 "b": 2}'  # Missing comma
fuzzy_json(invalid_json)  # ValueError: Invalid JSON string
```

**Solution**: Use fuzzy_json for common formatting issues (quotes, brackets, trailing commas). For structural errors, validate and sanitize input before parsing.

#### Pitfall 2: Performance Overhead in Hot Paths

**Issue**: Fuzzy parsing adds overhead even for valid JSON.

```python
# In performance-critical loops
for item in large_dataset:
    data = fuzzy_json(item)  # Overhead even if item is valid JSON
```

**Solution**: Use `orjson.loads()` directly when JSON is known to be well-formed. Reserve fuzzy_json for uncertain inputs.

```python
# Optimized version
for item in large_dataset:
    try:
        data = orjson.loads(item)  # Fast path
    except orjson.JSONDecodeError:
        data = fuzzy_json(item)  # Fallback for errors
```

#### Pitfall 3: Incorrect Bracket Fixing for Intentional Content

**Issue**: Bracket fixing may alter strings that intentionally contain unmatched brackets.

```python
# String content with intentional brackets
json_str = '{"message": "Use { for objects"'  # Missing closing bracket
fixed = fuzzy_json(json_str)
# Adds '}' which may not be intended location
```

**Solution**: Ensure JSON strings are properly escaped and quoted before fuzzy parsing, or use `fix_json_string()` separately with caution.

#### Pitfall 4: Not Handling TypeError for Non-String Input

**Issue**: Passing non-string types raises TypeError, not ValueError.

```python
# Passing dict instead of string
data = {"key": "value"}
fuzzy_json(data)  # TypeError: Input must be a string
```

**Solution**: Validate input type or catch TypeError separately from ValueError.

```python
def safe_parse(obj):
    if isinstance(obj, str):
        return fuzzy_json(obj)
    elif isinstance(obj, dict):
        return obj  # Already parsed
    else:
        raise TypeError(f"Cannot parse {type(obj)}")
```

## Design Rationale

### Why Progressive Error Correction?

The multi-stage parsing strategy (direct → clean → fix) provides optimal performance for well-formed JSON while maintaining robust error recovery:

1. **Fast Path Optimization**: Valid JSON parses immediately without correction overhead
2. **Incremental Correction**: Only apply transformations when needed
3. **Predictable Behavior**: Consistent correction order ensures reproducible results

### Why Quote Normalization?

Python dict literals use single quotes, but JSON requires double quotes. Automatic normalization enables:

1. **LLM Compatibility**: Many LLMs generate Python-style dicts instead of strict JSON
2. **Developer Convenience**: Copy-paste Python dicts without manual conversion
3. **Backward Compatibility**: Handle legacy systems that mix quote styles

### Why Bracket Balancing Instead of Rejection?

LLM-generated JSON frequently has truncated responses with missing closing brackets. Automatic balancing:

1. **Improves UX**: Recovers from common LLM output truncation
2. **Reduces Retry Overhead**: Avoids re-querying LLMs for simple formatting errors
3. **Maintains Structure**: Adds brackets in logical order (reverse of opening)

### Why Use orjson?

The underlying `orjson` parser provides:

1. **Performance**: 2-3x faster than stdlib `json` module
2. **Strict Validation**: Clear error messages when correction strategies fail
3. **Ecosystem Consistency**: Matches lionherd-core's serialization layer

### Why Positional-Only Parameters?

The `/` syntax enforces positional-only parameters to prevent accidental keyword usage:

```python
# Enforced usage
fuzzy_json('{"key": "value"}')  # Correct

# Prevented usage
fuzzy_json(str_to_parse='{"key": "value"}')  # TypeError
```

This improves:

1. **API Clarity**: Single-parameter functions are unambiguous
2. **Future Compatibility**: Allows adding keyword parameters without breaking changes

## See Also

- **Related Modules**:
  - `lionherd_core.libs.string_handlers`: String manipulation utilities
  - `orjson`: High-performance JSON serialization library
- **Related Functions**:
  - `pydapter.to_dict()`: Convert objects to dicts before JSON serialization
  - [extract_json](extract_json.md): Extract JSON from text with surrounding content
  - [string_similarity](string_similarity.md): Fuzzy string matching algorithms

## Examples

### Example 1: Handling LLM JSON with Multiple Issues

```python
from lionherd_core.libs.string_handlers import fuzzy_json

# Simulated LLM response with multiple formatting issues
llm_output = """
{
  'model': 'claude-3-opus',
  'parameters': {
    temperature: 0.7,
    max_tokens: 4096,
    'top_p': 0.9,
  },
  'capabilities': ['reasoning', 'coding', 'analysis',]
"""

# Single call handles all issues
parsed = fuzzy_json(llm_output)

print(parsed['model'])  # 'claude-3-opus'
print(parsed['parameters']['temperature'])  # 0.7
print(parsed['capabilities'])  # ['reasoning', 'coding', 'analysis']
```

### Example 2: Batch Processing API Responses

```python
from lionherd_core.libs.string_handlers import fuzzy_json
import logging

logger = logging.getLogger(__name__)

def process_api_batch(responses: list[str]) -> list[dict]:
    """Process batch of potentially malformed JSON responses."""
    results = []
    errors = []

    for i, response in enumerate(responses):
        try:
            parsed = fuzzy_json(response)
            results.append(parsed)
        except ValueError as e:
            logger.warning(f"Failed to parse response {i}: {e}")
            errors.append((i, response))

    logger.info(f"Parsed {len(results)}/{len(responses)} responses")
    return results

# Usage
api_responses = [
    '{"status": "ok", "data": [1, 2, 3]}',
    "{status: 'error', message: 'Failed'",  # Missing closing bracket
    '{"result": null, "count": 42,}',  # Trailing comma
]

parsed_data = process_api_batch(api_responses)
# Logs: "Parsed 3/3 responses" (all corrected successfully)
```

### Example 3: Two-Stage Parsing with Fast Path

```python
from lionherd_core.libs.string_handlers import fuzzy_json
import orjson

class JSONParser:
    """Optimized JSON parser with fuzzy fallback."""

    def __init__(self):
        self.fast_path_hits = 0
        self.fuzzy_path_hits = 0

    def parse(self, text: str) -> dict:
        """Parse JSON with performance tracking."""
        # Try fast path first (valid JSON)
        try:
            result = orjson.loads(text)
            self.fast_path_hits += 1
            return result
        except orjson.JSONDecodeError:
            pass

        # Fallback to fuzzy parsing
        try:
            result = fuzzy_json(text)
            self.fuzzy_path_hits += 1
            return result
        except ValueError:
            raise ValueError(f"Cannot parse JSON: {text[:100]}...")

    def stats(self):
        """Get parsing statistics."""
        total = self.fast_path_hits + self.fuzzy_path_hits
        if total == 0:
            return "No parses yet"

        fast_pct = (self.fast_path_hits / total) * 100
        return f"Fast: {self.fast_path_hits} ({fast_pct:.1f}%), Fuzzy: {self.fuzzy_path_hits}"

# Usage
parser = JSONParser()
responses = [
    '{"valid": true}',  # Fast path
    "{invalid: 'true'",  # Fuzzy path
    '{"another": "valid"}',  # Fast path
]

for resp in responses:
    parser.parse(resp)

print(parser.stats())  # "Fast: 2 (66.7%), Fuzzy: 1"
```

### Example 4: Bracket Fixing with Validation

> **⚠️ Advanced Usage**: Shows internal API usage. Prefer `fuzzy_json()` for general use.

```python
from lionherd_core.libs.string_handlers._fuzzy_json import fix_json_string  # Internal API
import orjson

def safe_fix_and_parse(partial_json: str) -> dict:
    """Fix brackets and parse, with validation."""
    try:
        # Attempt bracket fixing
        fixed = fix_json_string(partial_json)
        print(f"Fixed: {fixed}")

        # Validate by parsing
        parsed = orjson.loads(fixed)
        return parsed
    except ValueError as e:
        print(f"Cannot fix: {e}")
        raise

# Success case
partial = '{"users": [{"name": "Alice"}, {"name": "Bob"}'
result = safe_fix_and_parse(partial)
# Fixed: {"users": [{"name": "Alice"}, {"name": "Bob"}]}
# Returns: {'users': [{'name': 'Alice'}, {'name': 'Bob'}]}

# Failure case (mismatched brackets)
invalid = '{"data": [1, 2, 3}'
try:
    safe_fix_and_parse(invalid)
except ValueError:
    print("Bracket mismatch detected")
# Cannot fix: Mismatched brackets.
# Bracket mismatch detected
```

### Example 5: Converting Python Literals to JSON

```python
from lionherd_core.libs.string_handlers import fuzzy_json

# Python dict literal (single quotes, unquoted keys)
python_literal = """
{
    'name': 'Alice',
    age: 30,
    'roles': ['admin', 'user'],
    'config': {
        debug: True,
        'timeout': 5000,
    }
}
"""

# Convert to valid JSON dict
json_dict = fuzzy_json(python_literal)

# Now JSON-serializable
import orjson
json_bytes = orjson.dumps(json_dict)
print(json_bytes.decode())
# {"name":"Alice","age":30,"roles":["admin","user"],"config":{"debug":true,"timeout":5000}}
```

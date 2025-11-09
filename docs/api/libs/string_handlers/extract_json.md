# extract_json

> Extract and parse JSON content from strings or markdown code blocks

## Overview

`extract_json()` is a robust JSON extraction utility that handles multiple input formats, including raw JSON strings, markdown code blocks, and lists of strings. It provides intelligent fallback parsing with optional fuzzy JSON support for malformed input.

**Key Capabilities:**

- **Multi-Format Parsing**: Handles raw JSON strings, markdown ```json blocks, and string lists
- **Automatic Fallback**: Attempts direct parsing before extracting from markdown blocks
- **Fuzzy JSON Support**: Optional lenient parsing for malformed JSON via `fuzzy_json()`
- **Flexible Return Types**: Returns single dict or list based on content and configuration
- **Precompiled Regex**: Optimized markdown extraction with compiled pattern

**When to Use extract_json():**

- Parsing LLM responses that may contain JSON in markdown blocks
- Extracting structured data from mixed-format text
- Processing multiple JSON objects from a single string
- Handling potentially malformed JSON with fuzzy parsing

**When NOT to Use extract_json():**

- Strictly validated JSON with known format (use `orjson.loads()` directly)
- Performance-critical paths where markdown extraction overhead is unnecessary
- Binary JSON formats (use format-specific parsers)

See [fuzzy_json](fuzzy_json.md) for lenient JSON parsing alternative.

## Function Signature

```python
from lionherd_core.libs.string_handlers import extract_json

def extract_json(
    input_data: str | list[str],
    /,
    *,
    fuzzy_parse: bool = False,
    return_one_if_single: bool = True,
) -> Any | list[Any]: ...
```

## Parameters

### Positional Parameters

**input_data** : str or list of str

Input string(s) to parse for JSON content. If list provided, strings are joined with newlines before processing.

- Type coercion: Lists automatically joined with `"\n".join()`
- Processing order: Direct JSON parsing → Markdown block extraction
- Default: No default (required)

### Keyword-Only Parameters

**fuzzy_parse** : bool, default False

Enable fuzzy JSON parsing for malformed input. When True, uses `fuzzy_json()` for lenient parsing that handles common JSON errors (trailing commas, single quotes, etc.).

- Use when: Input may have JSON syntax errors (LLM-generated content)
- Performance: Slower than strict parsing due to additional processing
- Default: False (strict orjson parsing)

**return_one_if_single** : bool, default True

Control return type when exactly one JSON object is found. When True, returns the single dict/value directly instead of wrapping in a list.

- True: Single object → `dict`, multiple → `list[dict]`
- False: Always returns `list` (even for single object)
- Default: True (unwraps single results)

## Returns

`Any` or `list[Any]` - Parsed JSON content with type depending on input and parameters:

- **Empty list** (`[]`): No valid JSON found in input
- **Single value**: One JSON object found and `return_one_if_single=True`
  - Can be dict, list, str, int, float, bool, or None
- **List of values**: Multiple JSON objects found or `return_one_if_single=False`
  - Each element is a parsed JSON value (any JSON type)

## Parsing Logic

The function follows a three-stage fallback strategy:

### Stage 1: Direct Parsing

Attempts to parse entire `input_data` as JSON:

```python
# With fuzzy_parse=False (default)
return orjson.loads(input_str)

# With fuzzy_parse=True
return fuzzy_json(input_str)
```

If successful, returns immediately. If parsing fails, proceeds to Stage 2.

### Stage 2: Markdown Extraction

Searches for JSON code blocks using precompiled regex:

```python
# Pattern: ```json ... ```
_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
matches = _JSON_BLOCK_PATTERN.findall(input_str)
```

If no matches found, returns `[]`.

### Stage 3: Parse Extracted Blocks

Parses each markdown block:

- **Single match + `return_one_if_single=True`**: Returns first valid parse or `[]`
- **Multiple matches**: Returns list of all successfully parsed blocks (skips invalid)

Invalid blocks are silently suppressed (`contextlib.suppress(Exception)`).

## Examples

### Example 1: Direct JSON String

```python
from lionherd_core.libs.string_handlers import extract_json

# Valid JSON string
json_str = '{"name": "Alice", "age": 30}'
result = extract_json(json_str)
# {'name': 'Alice', 'age': 30}

# Array
json_array = '[1, 2, 3, 4]'
result = extract_json(json_array)
# [1, 2, 3, 4]

# Primitive value
json_primitive = '"hello"'
result = extract_json(json_primitive)
# 'hello'
```

### Example 2: Markdown Code Blocks

````python
from lionherd_core.libs.string_handlers import extract_json

# LLM response with JSON block
llm_response = """
Here's the user data:

```json
{
  "id": "123",
  "name": "Bob",
  "active": true
}
```

Hope that helps!
"""

result = extract_json(llm_response)

# Output: {'id': '123', 'name': 'Bob', 'active': True}

````

### Example 3: Multiple JSON Blocks

````python
from lionherd_core.libs.string_handlers import extract_json

# Multiple markdown blocks
multi_block = """
First user:
```json
{"name": "Alice", "role": "admin"}
```

Second user:

```json
{"name": "Bob", "role": "user"}
```

"""

result = extract_json(multi_block)

# Output: [

# {'name': 'Alice', 'role': 'admin'}

# {'name': 'Bob', 'role': 'user'}

# ]

# Force list return for single block

single_block = '```json\n{"key": "value"}\n```'
result = extract_json(single_block, return_one_if_single=False)

# Output: [{'key': 'value'}]  (wrapped in list)

````

### Example 4: Fuzzy JSON Parsing

````python
from lionherd_core.libs.string_handlers import extract_json

# Malformed JSON (trailing commas, single quotes)
malformed = """
{
  'name': 'Charlie',
  'tags': ['python', 'ai',],
}
"""

# Strict parsing fails
result = extract_json(malformed)
# []

# Fuzzy parsing succeeds
result = extract_json(malformed, fuzzy_parse=True)
# {'name': 'Charlie', 'tags': ['python', 'ai']}
````

### Example 5: List Input

````python
from lionherd_core.libs.string_handlers import extract_json

# List of strings (joined with newlines)
lines = [
    "Response:",
    '```json',
    '{"status": "success"}',
    '```'
]

result = extract_json(lines)
# {'status': 'success'}

# Mixed content across lines
mixed_lines = [
    '```json\n{"a": 1}\n```',
    'Some text',
    '```json\n{"b": 2}\n```'
]

result = extract_json(mixed_lines)
# [{'a': 1}, {'b': 2}]
````

### Example 6: Empty and Invalid Input

```python
from lionherd_core.libs.string_handlers import extract_json

# No JSON found
result = extract_json("Just plain text")
# []

# Invalid JSON (no markdown blocks)
result = extract_json("{invalid json}")
# []

# Empty string
result = extract_json("")
# []

# Markdown block with invalid JSON
invalid_block = '```json\n{broken: json}\n```'
result = extract_json(invalid_block)
# []

# With fuzzy parsing (may succeed depending on error)
result = extract_json(invalid_block, fuzzy_parse=True)
# {} or [] (depends on fuzzy_json behavior)
```

## Usage Patterns

### Pattern 1: LLM Response Parsing

````python
from lionherd_core.libs.string_handlers import extract_json

def parse_llm_response(response: str) -> dict | None:
    """Extract structured data from LLM response."""
    result = extract_json(response, fuzzy_parse=True)

    # Handle empty result
    if not result or result == []:
        return None

    # Handle list return (take first if multiple)
    if isinstance(result, list):
        return result[0] if result else None

    return result

# Usage
llm_output = """
I'll provide the user info:

```json
{"user_id": "abc123", "verified": true}
```

"""

data = parse_llm_response(llm_output)
# Output: {'user_id': 'abc123', 'verified': True}
````

### Pattern 2: Multi-Document Extraction

````python
from lionherd_core.libs.string_handlers import extract_json

def extract_all_json_objects(text: str) -> list[dict]:
    """Extract all JSON objects from text, always returning list."""
    result = extract_json(text, return_one_if_single=False)

    # Filter to dicts only (exclude primitives/arrays)
    return [item for item in result if isinstance(item, dict)]

# Usage
docs = """
```json
{"doc": 1}
```

Some text

```json
{"doc": 2}
```

```json
{"doc": 3}
```

"""

all_docs = extract_all_json_objects(docs)

# [{'doc': 1}, {'doc': 2}, {'doc': 3}]

````

### Pattern 3: Graceful Degradation

```python
from lionherd_core.libs.string_handlers import extract_json
import logging

def safe_extract_json(data: str, *, strict: bool = True) -> dict | list | None:
    """Extract JSON with graceful degradation from strict to fuzzy."""

    # Try strict parsing first
    result = extract_json(data, fuzzy_parse=False)
    if result and result != []:
        return result

    if not strict:
        # Fall back to fuzzy parsing
        logging.info("Strict parsing failed, trying fuzzy parsing")
        result = extract_json(data, fuzzy_parse=True)
        if result and result != []:
            logging.warning("Fuzzy parsing succeeded (may have data loss)")
            return result

    logging.error("No valid JSON found")
    return None
```

### Pattern 4: Batch Processing

```python
from lionherd_core.libs.string_handlers import extract_json

def batch_extract_json(
    texts: list[str],
    *,
    fuzzy: bool = False
) -> list[dict | list | None]:
    """Extract JSON from multiple texts."""
    results = []
    for text in texts:
        result = extract_json(text, fuzzy_parse=fuzzy)
        # Normalize empty results to None
        results.append(result if result and result != [] else None)
    return results

# Usage
responses = [
    '```json\n{"id": 1}\n```',
    'No JSON here',
    '{"id": 2}',
]

extracted = batch_extract_json(responses)
# [{'id': 1}, None, {'id': 2}]
```

## Common Pitfalls

### Pitfall 1: Not Checking for Empty Results

**Issue**: Assuming `extract_json()` always returns valid data.

```python
# Dangerous - may fail if no JSON found
result = extract_json("No JSON here")
name = result["name"]  # TypeError: list indices must be integers (result is [])
```

**Solution**: Always validate result before accessing:

```python
result = extract_json(text)
if result and result != []:
    name = result.get("name")  # Safe
else:
    # Handle missing JSON
    name = None
```

### Pitfall 2: Expecting Consistent Return Type

**Issue**: Forgetting that return type varies based on number of matches.

```python
# May be dict or list[dict] depending on input
result = extract_json(text)

# Breaks if result is list
print(result["key"])  # TypeError if result is list
```

**Solution**: Use `return_one_if_single=False` for consistent list return, or check type:

```python
# Force list return
result = extract_json(text, return_one_if_single=False)
for item in result:  # Always list
    print(item["key"])

# Or check type dynamically
result = extract_json(text)
items = [result] if isinstance(result, dict) else result
```

### Pitfall 3: Over-Relying on Fuzzy Parsing

**Issue**: Using `fuzzy_parse=True` by default and accepting data loss.

```python
# May silently corrupt data
result = extract_json(malformed_json, fuzzy_parse=True)
# Fuzzy parser might have "fixed" errors in unexpected ways
```

**Solution**: Use fuzzy parsing as fallback, not default. Validate output:

```python
# Try strict first
result = extract_json(data, fuzzy_parse=False)
if not result or result == []:
    # Only use fuzzy if strict fails
    result = extract_json(data, fuzzy_parse=True)
    if result:
        # Validate critical fields
        assert "required_field" in result
```

### Pitfall 4: Ignoring List Input Joining

**Issue**: Expecting list items to remain separate.

```python
lines = ['{"a": 1}', '{"b": 2}']
result = extract_json(lines)
# Lines joined with \n, parsed as single string
# If both are valid JSON, only first parse succeeds
```

**Solution**: Join with markdown blocks if expecting multiple objects:

```python
lines = [
    '```json\n{"a": 1}\n```',
    '```json\n{"b": 2}\n```'
]
result = extract_json(lines)
# [{'a': 1}, {'b': 2}]  (both extracted)
```

### Pitfall 5: Performance with Large Inputs

**Issue**: Using regex extraction on very large text with many markdown blocks.

```python
# Slow if text is multi-megabyte with hundreds of blocks
huge_text = "..." * 1000000
result = extract_json(huge_text)  # Regex may be slow
```

**Solution**: Pre-filter or chunk large inputs:

```python
# Limit search scope
if len(text) > 100000:
    # Only search first/last portions
    text = text[:50000] + text[-50000:]

result = extract_json(text)
```

## Implementation Details

### Regex Pattern

The markdown extraction uses a precompiled regex for performance:

```python
_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
```

- **Pattern**: Captures content between ```json and``` markers
- **Flags**: `re.DOTALL` allows `.` to match newlines (multiline JSON)
- **Non-greedy**: `.*?` ensures minimal matching (stops at first closing ```)
- **Whitespace**: `\s*` strips leading/trailing whitespace from captured content

### Error Handling

All parsing errors are silently suppressed using `contextlib.suppress(Exception)`:

```python
with contextlib.suppress(Exception):
    return orjson.loads(input_str)
```

**Rationale**: Allows graceful fallback through parsing stages without exception handling overhead.

**Trade-off**: Silent failures may hide genuine errors. Consider logging in production use.

### Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Direct JSON parse | O(n) | Where n = input string length (orjson) |
| Markdown extraction | O(n*m) | Where m = average block size (regex) |
| Fuzzy parsing | O(n*k) | Where k = repair iterations (varies) |

**Optimization**: Direct parsing attempt (`O(n)`) avoids regex overhead for well-formed JSON.

## Design Rationale

### Why Silent Error Suppression?

The function uses `contextlib.suppress(Exception)` instead of explicit try/except blocks for:

1. **Concise Fallback Logic**: Enables clean three-stage parsing without nested error handling
2. **Performance**: Avoids exception handling overhead in success cases
3. **Flexibility**: Works with any JSON parsing error type (not tied to specific exceptions)

**Trade-off**: Debugging is harder without error visibility. Production code may want logging.

### Why Precompiled Regex?

The `_JSON_BLOCK_PATTERN` is compiled at module import for:

1. **Performance**: Compilation happens once, not per function call
2. **Readability**: Separates pattern definition from extraction logic
3. **Reusability**: Pattern can be referenced by other functions if needed

### Why `return_one_if_single=True` Default?

Most use cases involve extracting single JSON objects from LLM responses:

```python
# Common pattern (single object expected)
response = 'Here it is: ```json\n{"key": "value"}\n```'
data = extract_json(response)  # dict, not [dict]
value = data["key"]  # No list unwrapping needed
```

Defaulting to True reduces boilerplate. Users needing consistent list returns can set `return_one_if_single=False`.

### Why Support List Input?

LLM responses are often received as lists of message chunks or lines. Supporting `list[str]` input eliminates manual joining:

```python
# Without list support
chunks = ["Line 1", "```json", '{"key": "value"}', "```"]
result = extract_json("\n".join(chunks))  # Manual join

# With list support
result = extract_json(chunks)  # Automatic join
```

## See Also

- **Related Functions**:
  - [fuzzy_json](fuzzy_json.md): Lenient JSON parsing for malformed input
  - [string_similarity](string_similarity.md): Fuzzy string matching algorithms
  - [to_num](to_num.md): String to number conversion
  - `orjson.loads()`: High-performance strict JSON parsing

## Performance Considerations

### Optimization Tips

1. **Skip Markdown Extraction**: If input is always valid JSON, use `orjson.loads()` directly
2. **Limit Fuzzy Parsing**: Only enable `fuzzy_parse=True` when necessary (slower)
3. **Pre-validate Input**: Check for markdown blocks before calling function
4. **Batch Processing**: Process multiple strings in parallel for large datasets

### Benchmarks (Approximate)

```python
import timeit

# Direct JSON parsing (best case)
timeit.timeit(lambda: extract_json('{"key": "value"}'), number=10000)
# ~0.02s (2μs per call)

# Markdown extraction (common case)
timeit.timeit(lambda: extract_json('```json\n{"key": "value"}\n```'), number=10000)
# ~0.15s (15μs per call)

# Fuzzy parsing (worst case)
timeit.timeit(lambda: extract_json('{key: "value"}', fuzzy_parse=True), number=10000)
# ~0.50s (50μs per call)
```

**Takeaway**: Direct JSON is ~7x faster than markdown extraction, ~25x faster than fuzzy parsing.

# Fuzzy LNDL Parser

> Fuzzy matching parser for LNDL with typo correction and strict mode support

## Overview

The fuzzy LNDL parser provides **typo-tolerant parsing** for LNDL (Language InterOperable Network Directive Language) responses, automatically correcting misspellings in model names, field names, spec names, and variable references. It supports both fuzzy matching (production default) and strict mode (exact matching only).

**Key Capabilities:**

- **Fuzzy Matching**: Jaro-Winkler similarity-based typo correction (default threshold: 0.85)
- **Strict Mode**: Exact matching when `threshold=1.0` (zero tolerance for typos)
- **Granular Control**: Per-category thresholds (field, lvar, model, spec)
- **Ambiguity Detection**: Prevents silent errors when multiple matches are equally plausible
- **Zero Duplication**: Pre-corrects typos before calling strict resolver (single validation path)

**When to Use Fuzzy Parser:**

- **Production LLM responses** - LLMs make typos, fuzzy matching recovers gracefully
- **User-written LNDL** - Humans make typos, especially in field/model names
- **Lenient validation** - Accept "titel" as "title" without forcing exact spelling
- **Interactive workflows** - Better UX when minor typos don't break execution

**When to Use Strict Mode (`threshold=1.0`):**

- **Testing/validation** - Enforce exact output format from LLMs
- **Critical workflows** - No tolerance for ambiguity or typos
- **Schema enforcement** - Validate LLM follows exact field names
- **Debugging** - Identify where LLM is making mistakes

## Module Exports

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy

__all__ = ("parse_lndl_fuzzy",)
```

## Functions

### `parse_lndl_fuzzy()`

Parse LNDL with fuzzy matching (default) or strict mode (threshold=1.0).

**Signature:**

```python
def parse_lndl_fuzzy(
    response: str,
    operable: Operable,
    /,
    *,
    threshold: float = 0.85,
    threshold_field: float | None = None,
    threshold_lvar: float | None = None,
    threshold_model: float | None = None,
    threshold_spec: float | None = None,
) -> LNDLOutput: ...
```

**Parameters:**

- **response** (str): Full LLM response containing lvars, lacts, and OUT{} block
- **operable** (Operable): Operable instance containing allowed specs for validation
- **threshold** (float, default 0.85): Global similarity threshold
  - `0.85`: Fuzzy matching (production-proven, balanced tolerance)
  - `1.0`: Strict mode (exact matches only, zero tolerance)
  - `0.7-0.95`: Custom tolerance (lower = more lenient, higher = stricter)
- **threshold_field** (float, optional): Override threshold for field names (default: use `threshold`)
- **threshold_lvar** (float, optional): Override threshold for lvar references (default: use `threshold`)
- **threshold_model** (float, optional): Override threshold for model names (default: `max(threshold, 0.90)` - stricter)
- **threshold_spec** (float, optional): Override threshold for spec names (default: use `threshold`)

**Returns:**

- **LNDLOutput**: Validated output with corrected field mappings and instantiated model instances

**Raises:**

- **MissingFieldError**: No match above threshold for a name
- **AmbiguousMatchError**: Multiple matches within 0.05 similarity (tie detection)
- **ValueError**: Validation errors from strict resolver (malformed LNDL, circular references, etc.)

**Examples:**

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.errors import MissingFieldError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str

operable = Operable([Spec(Report, name="report")])

# Default fuzzy matching (auto-corrects typos)
response = '''
<lvar Report.titel title>Good Title</lvar>
<lvar Report.sumary summary>Brief summary</lvar>
OUT{reprot: [titel, sumary]}
'''
result = parse_lndl_fuzzy(response, operable)
# Auto-corrects: "titel" → "title", "sumary" → "summary", "reprot" → "report"

# Strict mode (exact matches only)
try:
    result = parse_lndl_fuzzy(response, operable, threshold=1.0)
except MissingFieldError as e:
    print(e)  # "Field 'titel' not found. Available: ['title', ...] (strict mode)"

# Custom thresholds (stricter for models, lenient for fields)
result = parse_lndl_fuzzy(
    response,
    operable,
    threshold_model=0.95,  # Very strict for model names
    threshold_field=0.75,  # More lenient for field names
)

# Ambiguity detection
class User(BaseModel):
    username: str = ""  # Optional with default
    name: str

operable_user = Operable([Spec(User, name="user")])
response_ambiguous = '''
<lvar User.nam nam>John</lvar>
OUT{user: [nam]}
'''
# If both "username" and "name" score similarly (within 0.05), raises AmbiguousMatchError
```

**Architecture:**

The fuzzy parser follows a **pre-correction + strict validation** architecture:

1. **Parse LNDL**: Extract lvars, lacts, and OUT{} block using standard parser
2. **Pre-correct Typos**: Fuzzy match all names against expected values
   - Model names (from `operable.get_specs()`)
   - Field names (per model, from Pydantic schema)
   - Spec names (from `operable.allowed()`)
   - Lvar/lact references (from extracted variable names)
3. **Rebuild Metadata**: Create corrected `LvarMetadata` and `LactMetadata` with fixed names
4. **Call Strict Resolver**: Delegate to `resolve_references_prefixed()` with corrected inputs
5. **Return Validated Output**: LNDLOutput with instantiated models

**Design Benefits:**

- **Zero Duplication**: Validation logic lives in strict resolver, not fuzzy parser
- **Single Code Path**: Fuzzy mode and strict mode share same resolver
- **Composable**: Easy to add new correction types (e.g., action names)
- **Testable**: Fuzzy matching and strict validation tested independently

**See Also:**

- [`resolve_references_prefixed()`](resolver.md): Strict resolver (no typo correction)
- [`extract_lvars_prefixed()`](parser.md#extract_lvars_prefixed): LNDL parser (extracts lvars)
- [`LNDLOutput`](types.md#lndloutput): Output type with validated models

### `_correct_name()` (Internal)

Correct name using fuzzy matching with tie detection.

**Signature:**

```python
def _correct_name(
    target: str,
    candidates: list[str],
    threshold: float,
    context: str = "name",
) -> str: ...
```

**Parameters:**

- **target** (str): User-provided name (may have typo)
- **candidates** (list[str]): Valid names to match against
- **threshold** (float): Similarity threshold (0.0-1.0)
- **context** (str, default "name"): Context for error messages (e.g., "field", "lvar", "model")

**Returns:**

- **str**: Corrected name

**Raises:**

- **MissingFieldError**: No match above threshold
- **AmbiguousMatchError**: Multiple matches within 0.05 similarity (prevents silent errors)

**Algorithm:**

1. **Exact Match**: If `target` in `candidates`, return immediately (no fuzzy needed)
2. **Strict Mode**: If `threshold >= 1.0`, raise `MissingFieldError` (exact match required)
3. **Fuzzy Match**: Use Jaro-Winkler algorithm to score all candidates
4. **Threshold Filter**: Keep candidates with score >= threshold
5. **Tie Detection**: Check if multiple candidates score within 0.05 of max score
6. **Return Winner**: If single clear winner, return it (log correction if different from target)

**Examples:**

```python
from lionherd_core.lndl.fuzzy import _correct_name
from lionherd_core.lndl.errors import MissingFieldError, AmbiguousMatchError

# Exact match (no fuzzy needed)
result = _correct_name("title", ["title", "content"], 0.85, "field")
# Returns: "title" (immediate match)

# Fuzzy match (typo correction)
result = _correct_name("titel", ["title", "content"], 0.85, "field")
# Returns: "title" (Jaro-Winkler: 0.933 > 0.85)

# Strict mode (threshold=1.0)
try:
    _correct_name("titel", ["title", "content"], 1.0, "field")
except MissingFieldError as e:
    print(e)
    # "Field 'titel' not found. Available: ['title', 'content'] (strict mode)"

# Ambiguous match (tie)
try:
    _correct_name("nam", ["name", "number"], 0.85, "field")
except AmbiguousMatchError as e:
    print(e)
    # If scores: name=0.80, number=0.78 (within 0.05)
    # "Ambiguous match for field 'nam': ['name': 0.800, 'number': 0.780]"

# Below threshold
try:
    _correct_name("xyz", ["title", "content"], 0.85, "field")
except MissingFieldError as e:
    print(e)
    # "Field 'xyz' not found above threshold 0.85. Available: ['title', 'content']"
```

**Notes:**

- Uses **Jaro-Winkler** algorithm for similarity (favors prefix matches)
- **Tie threshold**: 0.05 (hardcoded) - prevents ambiguous corrections
- **Logging**: Logs corrections at DEBUG level when name is corrected
- **Context parameter**: Provides clear error messages ("Field 'titel' not found" vs "Model 'titel' not found")

## Usage Patterns

### Basic Fuzzy Parsing

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str

operable = Operable([Spec(Report, name="report")])

# LLM response with typos
response = '''
<lvar Report.titel title>Analysis Report</lvar>
<lvar Report.sumary summary>Key findings...</lvar>
OUT{reprot: [titel, sumary]}
'''

# Parse with fuzzy matching (auto-corrects typos)
result = parse_lndl_fuzzy(response, operable)

# Access corrected output
print(result.report.title)    # "Analysis Report"
print(result.report.summary)  # "Key findings..."
```

### Strict Mode (Testing/Validation)

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.errors import MissingFieldError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str = ""  # Optional with default for simple examples

operable = Operable([Spec(Report, name="report")])

# Strict mode: exact matches only
response_with_typo = '''
<lvar Report.titel title>Title</lvar>
OUT{report: [titel]}
'''

try:
    result = parse_lndl_fuzzy(response_with_typo, operable, threshold=1.0)
except MissingFieldError as e:
    print(f"Validation failed: {e}")
    # "Field 'titel' not found. Available: ['title', 'summary'] (strict mode: exact match required)"

# Correct version passes
response_correct = '''
<lvar Report.title title>Title</lvar>
OUT{report: [title]}
'''
result = parse_lndl_fuzzy(response_correct, operable, threshold=1.0)
# Success - exact match
```

### Granular Threshold Control

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str

class Config(BaseModel):
    api_key: str
    endpoint: str

operable = Operable([
    Spec(User, name="user"),
    Spec(Config, name="config"),
])

response = '''
<lvar User.usrname username>john_doe</lvar>
<lvar Config.apikey api_key>secret123</lvar>
OUT{usr: [usrname], cnfig: [apikey]}
'''

# Different tolerance levels per category
result = parse_lndl_fuzzy(
    response,
    operable,
    threshold=0.85,          # Default for most categories
    threshold_model=0.95,    # Very strict for model names (important)
    threshold_field=0.75,    # More lenient for field names (LLMs vary)
    threshold_spec=0.90,     # Stricter for spec names (output structure)
    threshold_lvar=0.80,     # Moderate for variable references
)
```

### Handling Ambiguity

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.errors import AmbiguousMatchError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

# Ambiguous field names
class User(BaseModel):
    username: str
    name: str  # Similar to username

operable = Operable([Spec(User, name="user")])

response = '''
<lvar User.nam username>John</lvar>
OUT{user: [nam]}
'''

try:
    result = parse_lndl_fuzzy(response, operable)
except AmbiguousMatchError as e:
    print(f"Ambiguous match: {e}")
    # "Ambiguous match for field 'nam': ['name': 0.667, 'username': 0.636]. Multiple candidates scored within 0.05."
    # Solution: User needs to be more specific
```

### Migration from Strict to Fuzzy

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.parser import (
    extract_lvars_prefixed,
    extract_lacts_prefixed,
    extract_out_block,
    parse_out_block_array,
)
from lionherd_core.lndl.resolver import resolve_references_prefixed
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str = ""  # Optional with default

operable = Operable([Spec(Report, name="report")])
response = '''<lvar Report.title title>Title</lvar>
OUT{report: [title]}'''

# Before: Using strict resolver (manual extraction)
lvars = extract_lvars_prefixed(response)
lacts = extract_lacts_prefixed(response)
out_fields = parse_out_block_array(extract_out_block(response))
result = resolve_references_prefixed(out_fields, lvars, lacts, operable)

# After: Using fuzzy parser (single call, auto-corrects typos)
result = parse_lndl_fuzzy(response, operable)

# Optionally enable strict mode for testing
result = parse_lndl_fuzzy(response, operable, threshold=1.0)
```

## Error Handling

### MissingFieldError

Raised when no match is found above threshold.

**Scenarios:**

1. **Typo too severe**: `"xyz"` vs `["title", "summary"]` (no candidate close enough)
2. **Strict mode**: Any typo when `threshold=1.0`
3. **Wrong model name**: `"Reporrt"` with threshold=0.95 (too strict)

**Example:**

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.errors import MissingFieldError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str = ""  # Optional with default

operable = Operable([Spec(Report, name="report")])
response = '''<lvar Report.titel title>Title</lvar>
OUT{report: [titel]}'''

try:
    result = parse_lndl_fuzzy(response, operable, threshold=0.95)
except MissingFieldError as e:
    print(e)
    # "Field 'titel' not found above threshold 0.95. Available: ['title', 'summary']"

    # Solutions:
    # 1. Lower threshold: threshold=0.85
    # 2. Fix typo in response: "titel" → "title"
    # 3. Add field to model: model_fields += ["titel"]
```

### AmbiguousMatchError

Raised when multiple candidates score within 0.05 of the best match.

**Scenarios:**

1. **Similar field names**: `"name"` and `"username"` both match `"nam"`
2. **Abbreviated names**: `"desc"` matches both `"description"` and `"descriptor"`
3. **Low threshold + ambiguity**: threshold=0.70 increases ambiguous matches

**Example:**

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.errors import AmbiguousMatchError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Data(BaseModel):
    description: str
    descriptor: str  # Very similar

operable = Operable([Spec(Data, name="data")])
response = '<lvar Data.desc description>Value</lvar>\nOUT{data: [desc]}'

try:
    result = parse_lndl_fuzzy(response, operable, threshold=0.75)
except AmbiguousMatchError as e:
    print(e)
    # "Ambiguous match for field 'desc': ['description': 0.812, 'descriptor': 0.785]. Be more specific."

    # Solutions:
    # 1. Be more specific: "desc" → "description"
    # 2. Increase threshold: threshold=0.85 (may eliminate one candidate)
    # 3. Rename model fields to be less ambiguous
```

### ValueError (from Strict Resolver)

Raised by strict resolver after fuzzy correction (validation errors).

**Scenarios:**

1. **Circular references**: lvar references itself transitively
2. **Missing variables**: OUT{} references undefined lvar
3. **Type validation errors**: Field value doesn't match Pydantic type
4. **Malformed LNDL**: Syntax errors in OUT{} block

**Example:**

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str

operable = Operable([Spec(Report, name="report")])

# Circular reference (fuzzy parser corrects names, resolver detects cycle)
response = '''
<lvar Report.title a><lvar:b></lvar>
<lvar Report.summary b><lvar:a></lvar>
OUT{report: [a, b]}
'''

try:
    result = parse_lndl_fuzzy(response, operable)
except ValueError as e:
    print(e)
    # "Circular reference detected: a → b → a"
```

## Design Rationale

### Why Fuzzy Matching by Default?

LLMs and humans make typos frequently, especially in structured outputs:

1. **Production Reality**: GPT-4, Claude, etc. occasionally misspell field names despite clear prompts
2. **User Experience**: Minor typos shouldn't break workflows ("titel" is clearly "title")
3. **Robustness**: Fuzzy matching recovers gracefully without retries or manual intervention
4. **Proven Threshold**: 0.85 (Jaro-Winkler) balances tolerance and accuracy across production use

**Alternatives Considered:**

- **Always Strict**: Too brittle for production LLM outputs
- **Edit Distance**: Jaro-Winkler performs better for field/model names (prefix-biased)
- **No Threshold**: Unsafe - would accept very poor matches

### Why Separate Thresholds per Category?

Different name categories have different error profiles:

1. **Model Names** (default: 0.90): Critical for routing, stricter to avoid misrouting
2. **Field Names** (default: 0.85): LLMs vary here, balanced tolerance
3. **Spec Names** (default: 0.85): Output structure matters, moderate strictness
4. **Lvar References** (default: 0.85): Local names, moderate tolerance

**Granular control** lets users tune tolerance per use case without global trade-offs.

### Why Tie Detection (0.05 threshold)?

Ambiguous matches cause **silent errors** - fuzzy parser "corrects" to wrong field:

```python
# Without tie detection (DANGEROUS)
_correct_name("nam", ["name", "number"], 0.85)
# Returns: "name" (score: 0.80) - but "number" scored 0.78!
# User expects "name" but model has different semantics

# With tie detection (SAFE)
# Raises AmbiguousMatchError - user must be more specific
```

**0.05 threshold** empirically balances:

- **Too strict** (0.01): False positives on clearly distinct matches
- **Too lenient** (0.10): Misses real ambiguities

### Why Pre-Correction Architecture?

**Pre-correction** (fuzzy parser) + **strict validation** (resolver) avoids duplication:

1. **Single Validation Path**: Resolver handles all validation (references, types, cycles)
2. **Composability**: Easy to add new correction types without touching resolver
3. **Testability**: Fuzzy matching and validation tested independently
4. **Maintainability**: Changes to validation logic don't affect fuzzy matching

**Alternative (Rejected)**: Inline fuzzy matching in resolver duplicates validation logic.

### Why Jaro-Winkler Algorithm?

Jaro-Winkler is **prefix-biased** - ideal for field/model names:

```python
# Jaro-Winkler favors prefix matches
similarity("titel", "title")   # 0.933 (common prefix "titl")
similarity("titel", "content") # 0.476 (no common prefix)

# Levenshtein (edit distance) is less prefix-aware
levenshtein("titel", "title")   # 0.833 (1 edit)
levenshtein("titel", "tidle")   # 0.833 (1 edit) - but "tidle" is nonsense
```

**Jaro-Winkler** correctly prioritizes "title" over "tidle" due to prefix similarity.

## Performance Considerations

### Complexity

- **Time**: O(N × M) where N = number of typos, M = number of candidates per category
  - Model names: O(num_models) per unique typo
  - Field names: O(num_fields_in_model) per unique typo
  - Lvars: O(num_lvars) per reference
- **Space**: O(N) for correction maps (model_corrections, field_corrections)

### Optimization Strategies

1. **Deduplicate Before Matching**: Only match unique names (not every occurrence)
2. **Cache Corrections**: Reuse correction map across multiple lvars
3. **Early Exit**: Exact matches skip fuzzy scoring
4. **Strict Mode Bypass**: `threshold=1.0` skips fuzzy matching entirely

### Benchmarks (Approximate)

- **Small Response** (5 lvars, 3 models): <1ms overhead vs strict parser
- **Medium Response** (20 lvars, 5 models): 2-5ms overhead
- **Large Response** (100 lvars, 10 models): 10-20ms overhead

**Conclusion**: Fuzzy matching overhead is negligible (<1% of total LLM response time).

## See Also

- **Related Functions**:
  - [`resolve_references_prefixed()`](resolver.md): Strict resolver (no fuzzy matching)
  - [`extract_lvars_prefixed()`](parser.md#extract_lvars_prefixed): Extracts lvars from LNDL
  - [`extract_lacts_prefixed()`](parser.md#extract_lacts_prefixed): Extracts lacts from LNDL
  - [`parse_out_block_array()`](parser.md#parse_out_block_array): Parses OUT{} block
- **Related Types**:
  - [`LNDLOutput`](types.md#lndloutput): Parser output type
  - [`LvarMetadata`](types.md#lvarmetadata): Lvar metadata structure
  - [`LactMetadata`](types.md#lactmetadata): Lact metadata structure
  - [`Operable`](../types/operable.md): Spec container for validation
- **Related Errors**:
  - [`MissingFieldError`](errors.md#missingfielderror): No match above threshold
  - [`AmbiguousMatchError`](errors.md#ambiguousmatcherror): Tie detection
- **Related Modules**:
  - [LNDL Resolver](resolver.md): Strict parsing and validation
  - [LNDL Parser](parser.md): LNDL response parsing
  - [LNDL Types](types.md): Type definitions

## Examples

### Example 1: Production Fuzzy Parsing

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class BlogPost(BaseModel):
    title: str
    content: str
    tags: list[str]

operable = Operable([Spec(BlogPost, name="blog_post")])

# LLM response with multiple typos
llm_response = '''
<lvar BlogPost.titel title>My First Post</lvar>
<lvar BlogPost.contnt content>Lorem ipsum...</lvar>
<lvar BlogPost.tgs tags>["python", "ai"]</lvar>
OUT{blog_pst: [titel, contnt, tgs]}
'''

# Fuzzy parser auto-corrects all typos
result = parse_lndl_fuzzy(llm_response, operable)

print(result.blog_post.title)   # "My First Post"
print(result.blog_post.content) # "Lorem ipsum..."
print(result.blog_post.tags)    # ["python", "ai"]
```

### Example 2: Strict Mode for Testing

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.errors import MissingFieldError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class BlogPost(BaseModel):
    title: str
    content: str = ""  # Optional with default
    tags: list[str] = []  # Optional with default

operable = Operable([Spec(BlogPost, name="blog_post")])

# Test harness: validate LLM produces exact field names
def test_llm_output_format(llm_response: str):
    """Strict validation - no typo tolerance."""
    try:
        result = parse_lndl_fuzzy(llm_response, operable, threshold=1.0)
        return {"status": "pass", "result": result}
    except MissingFieldError as e:
        return {"status": "fail", "error": str(e)}

# Test with correct output
correct_response = '''
<lvar BlogPost.title title>Title</lvar>
OUT{blog_post: [title]}
'''
assert test_llm_output_format(correct_response)["status"] == "pass"

# Test with typo (should fail in strict mode)
typo_response = '''
<lvar BlogPost.titel title>Title</lvar>
OUT{blog_post: [titel]}
'''
result = test_llm_output_format(typo_response)
assert result["status"] == "fail"
assert "strict mode" in result["error"]
```

### Example 3: Custom Threshold Tuning

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

# Use case: Multi-model response with varying tolerance
class User(BaseModel):
    username: str
    email: str

class Config(BaseModel):
    api_key: str
    endpoint: str

operable = Operable([
    Spec(User, name="user"),
    Spec(Config, name="config"),
])

response = '''
<lvar User.usrname username>john_doe</lvar>
<lvar Config.apikey api_key>secret123</lvar>
OUT{usr: [usrname], cnfig: [apikey]}
'''

# Strict for critical config, lenient for user fields
result = parse_lndl_fuzzy(
    response,
    operable,
    threshold_model=0.95,  # Strict model names (avoid misrouting)
    threshold_field=0.75,  # Lenient field names (user input varies)
    threshold_spec=0.90,   # Strict output structure
)
```

### Example 4: Handling Ambiguity

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.errors import AmbiguousMatchError
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Document(BaseModel):
    description: str
    descriptor: str = ""  # Ambiguous with description, optional with default

operable = Operable([Spec(Document, name="document")])

# Ambiguous typo
response = '''
<lvar Document.desc description>Some text</lvar>
OUT{document: [desc]}
'''

# Detect ambiguity
try:
    result = parse_lndl_fuzzy(response, operable)
except AmbiguousMatchError as e:
    print(f"Ambiguous: {e}")
    # Fix: Be more specific - "desc" → "description"

# Fixed version
response_fixed = '''
<lvar Document.description description>Some text</lvar>
OUT{document: [description]}
'''
result = parse_lndl_fuzzy(response_fixed, operable)
# Success
```

### Example 5: Migrating from Strict to Fuzzy

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.parser import (
    extract_lvars_prefixed,
    extract_lacts_prefixed,
    extract_out_block,
    parse_out_block_array,
)
from lionherd_core.lndl.resolver import resolve_references_prefixed
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str = ""  # Optional with default

operable = Operable([Spec(Report, name="report")])
response = '''<lvar Report.title title>Title</lvar>
OUT{report: [title]}'''

# Before: Manual extraction + strict resolver
# Step 1: Extract components
lvars = extract_lvars_prefixed(response)
lacts = extract_lacts_prefixed(response)
out_content = extract_out_block(response)
out_fields = parse_out_block_array(out_content)

# Step 2: Resolve (strict - fails on typos)
result = resolve_references_prefixed(out_fields, lvars, lacts, operable)

# After: Single fuzzy parser call (auto-handles typos, fewer lines)
result = parse_lndl_fuzzy(response, operable)

# Optionally enable strict mode for testing
result_strict = parse_lndl_fuzzy(response, operable, threshold=1.0)
```

### Example 6: Debugging Typo Corrections

```python
import logging
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str

operable = Operable([Spec(Report, name="report")])

# Enable DEBUG logging to see corrections
logging.basicConfig(level=logging.DEBUG)

response = '''
<lvar Report.titel title>Title</lvar>
<lvar Report.sumary summary>Summary</lvar>
OUT{reprot: [titel, sumary]}
'''

result = parse_lndl_fuzzy(response, operable)

# Console output:
# DEBUG:lionherd_core.lndl.fuzzy:Fuzzy corrected model: 'Reprot' → 'Report'
# DEBUG:lionherd_core.lndl.fuzzy:Fuzzy corrected field (model Report): 'titel' → 'title'
# DEBUG:lionherd_core.lndl.fuzzy:Fuzzy corrected field (model Report): 'sumary' → 'summary'
# DEBUG:lionherd_core.lndl.fuzzy:Fuzzy corrected spec: 'reprot' → 'report'
```

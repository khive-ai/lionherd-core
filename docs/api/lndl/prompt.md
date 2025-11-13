# LNDL Prompt

> System prompt for LNDL (Language InterOperable Network Directive Language) - Structured output with natural thinking

## Overview

The LNDL prompt module provides the system prompt that instructs LLMs on how to use the **LNDL syntax** for generating structured outputs while maintaining natural thinking patterns. LNDL is a lightweight markup language designed for LLM-driven workflows that combines:

- **Natural thinking**: Write thoughts in plain prose intermixed with structured tags
- **Explicit mapping**: `Model.field` prefixes eliminate ambiguity
- **Action support**: Execute tool/function calls with Pythonic syntax
- **Lazy execution**: Only actions referenced in `OUT{}` are executed
- **Flexible revision**: Declare multiple versions, select final in output block

**Key Features:**

- **Variables (`<lvar>`)**: Map LLM-generated values to Pydantic model fields
- **Actions (`<lact>`)**: Define tool/function calls with optional execution
- **Output Block (`OUT{}`)**: Declare final mappings with array syntax for models, literals for scalars
- **Two Action Patterns**: Namespaced (for mixing with lvars) vs Direct (entire output)
- **Scratch Work**: Unused variables/actions preserved for debugging but not executed

**When to Use LNDL:**

- Structured outputs from LLMs with strict Pydantic schema validation
- Workflows requiring tool/function calls mixed with generated content
- Natural thinking + structured output combination (not forced JSON)
- Iterative LLM reasoning where multiple versions are explored

**When NOT to Use LNDL:**

- Simple text generation without structure (use plain prompts)
- Pure JSON output without thinking process (use standard JSON mode)
- Non-Pydantic output validation (LNDL assumes Pydantic models)

## Module Contents

### Constants

#### `LNDL_SYSTEM_PROMPT`

The complete LNDL system prompt string containing syntax rules, examples, and error guidance.

**Type:** `str`

**Content Sections:**

1. **SYNTAX**: Variable, action, and output block syntax rules
2. **EXAMPLE 1**: Direct actions (entire output from single action)
3. **EXAMPLE 2**: Mixing lvars and namespaced actions
4. **KEY POINTS**: Core concepts and usage guidelines
5. **SCALARS vs MODELS vs ACTIONS**: Type-specific syntax
6. **ERRORS TO AVOID**: Common mistakes with corrections
7. **CORRECT**: Proper usage patterns

**Usage:**

```python
from lionherd_core.lndl.prompt import LNDL_SYSTEM_PROMPT

# Use in LLM system messages
system_message = {
    "role": "system",
    "content": LNDL_SYSTEM_PROMPT
}
```

**Notes:**

The prompt is designed to be self-contained and comprehensive, requiring no additional context. LLMs receive this as a system message before generating LNDL-formatted responses.

### Functions

#### `get_lndl_system_prompt()`

Get the LNDL system prompt for LLM guidance.

**Signature:**

```python
def get_lndl_system_prompt() -> str: ...
```

**Parameters:**

None

**Returns:**

- str: LNDL system prompt string (stripped of leading/trailing whitespace)

**Examples:**

```python
# noqa:validation
>>> from lionherd_core.lndl.prompt import get_lndl_system_prompt
>>> prompt = get_lndl_system_prompt()
>>> print(prompt[:50])
'LNDL - Structured Output with Natural Thinking\n\nS'

# Use with LLM clients
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": get_lndl_system_prompt()},
        {"role": "user", "content": "Generate a report about AI safety..."}
    ]
)
```

**See Also:**

- `LNDL_SYSTEM_PROMPT`: The underlying constant
- [LNDL Parser](parser.md): Parses LNDL-formatted LLM responses

**Notes:**

This is a simple accessor function that returns the stripped prompt constant. It exists to provide a consistent API pattern for retrieving the prompt programmatically.

## LNDL Syntax Overview

### Variables (`<lvar>`)

Map LLM-generated values to Pydantic model fields.

**Syntax:**

```xml
<lvar Model.field alias>value</lvar>
```

**Components:**

- `Model.field`: Explicit field mapping (e.g., `Report.title`)
- `alias`: Short reference name for use in `OUT{}` (optional, defaults to field name)
- `value`: LLM-generated content

**Examples:**

```xml
<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.summary s>Based on search results, AI safety research focuses on...</lvar>
<lvar Report.confidence c>0.85</lvar>
```

### Actions (`<lact>`)

Define tool/function calls with lazy execution.

**Two Patterns:**

**1. Namespaced (for mixing with lvars):**

```xml
<lact Model.field alias>function_call(args)</lact>
```

**2. Direct (entire output):**

```xml
<lact name>function_call(args)</lact>
```

**Key Difference:**

- **Namespaced**: Result fills a specific model field, can mix with lvars
- **Direct**: Result becomes the entire field value, cannot mix with lvars

**Examples:**

```xml
<!-- Namespaced: mixes with lvars -->
<lvar Report.title t>Analysis Report</lvar>
<lact Report.summary summarize>generate_summary(data="metrics")</lact>
<lvar Report.footer f>End of Report</lvar>
OUT{report:[t, summarize, f]}

<!-- Direct: entire output from action -->
<lact search_result>search(query="AI safety", limit=20)</lact>
OUT{search_data:[search_result]}
```

**Lazy Execution:**

Actions are **only executed** if referenced in `OUT{}`. Unreferenced actions are preserved as scratch work but never called.

```xml
<lact unused>expensive_call()</lact>
<lact used>cheap_call()</lact>
OUT{data:[used]}
<!-- Only "used" executes; "unused" is ignored -->
```

### Output Block (`OUT{}`)

Declares final mappings from variables/actions to model fields.

**Syntax:**

```lndl
OUT{field1:[var1, var2], field2:[action], scalar:literal}
```

**Rules:**

- **Models**: Use array syntax `field:[alias1, alias2, ...]`
- **Scalars**: Use literal `field:0.8` or single-element array `field:[q]`
- **Actions**: Reference by alias name `field:[action_name]`
- **Mixing**: Arrays can contain both lvar and lact aliases (if namespaced)

**Examples:**

```lndl
<!-- Models (array syntax) -->
OUT{report:[title, summary, footer]}

<!-- Scalars (literals) -->
OUT{quality_score:0.85, is_draft:false}

<!-- Scalars (from variables) -->
OUT{quality_score:[q]}

<!-- Mixing lvars and actions -->
OUT{report:[title, api_summary, footer]}

<!-- Direct action (entire output) -->
OUT{search_data:[search_call]}
```

## Usage Patterns

### Basic LNDL Prompt Setup

```python
from lionherd_core.lndl.prompt import get_lndl_system_prompt

# Add to LLM system messages
system_prompt = get_lndl_system_prompt()

# Example with OpenAI
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Generate a report with title, summary, and confidence score (0-1)"
        }
    ]
)

# LLM will respond with LNDL format:
# <lvar Report.title t>AI Safety Analysis</lvar>
# <lvar Report.summary s>...</lvar>
# <lvar Report.confidence c>0.85</lvar>
# OUT{report:[t, s, c]}
```

### Multi-Version Exploration

LNDL supports iterative thinking by allowing multiple versions:

```python
# LLM generates multiple options
"""
Let me try different titles:
<lvar Report.title t1>AI Safety: A Comprehensive Review</lvar>
Actually, this is better:
<lvar Report.title t2>Understanding AI Safety Challenges</lvar>

For the summary, let me start broad:
<lvar Report.summary s1>AI safety encompasses multiple research areas...</lvar>
No, too vague. More specific:
<lvar Report.summary s2>Three critical challenges in AI safety are...</lvar>

OUT{report:[t2, s2]}
"""

# Parser extracts only t2 and s2 (final selections)
```

### Action Execution Control

Only referenced actions execute:

```python
# LLM thinks through options
"""
Let me search broadly first:
<lact broad>search(query="AI", limit=100)</lact>
Too noisy. Let me refine:
<lact focused>search(query="AI safety", limit=20)</lact>

I'll use the focused search:
OUT{search_data:[focused]}
"""

# Only "focused" action executes
# "broad" is scratch work (not called)
```

### Mixing Variables and Actions

Namespaced actions enable mixing:

```python
# LLM combines generated text + API calls
"""
<lvar Report.title t>Research Analysis</lvar>
<lact Report.data api_data>fetch_metrics(source="api")</lact>
<lvar Report.footer f>Report generated on 2025-11-09</lvar>

OUT{report:[t, api_data, f]}
"""

# Result: Report with title (generated), data (from API), footer (generated)
```

### Scalar vs Model Outputs

```python
# Scalars: use literals or single references
"""
<lvar quality q>0.85</lvar>
OUT{quality_score:[q], is_draft:false}
"""

# Models: use arrays
"""
<lvar Report.title t>Title</lvar>
<lvar Report.summary s>Summary</lvar>
OUT{report:[t, s]}
"""
```

## Common Pitfalls

### Pitfall 1: Missing Model.field Prefix

**Issue:** Using alias without namespace.

```xml
<!-- ❌ WRONG -->
<lvar title>AI Safety</lvar>

<!-- ✅ CORRECT -->
<lvar Report.title title>AI Safety</lvar>
```

**Reason:** LNDL requires explicit model mapping to avoid ambiguity.

### Pitfall 2: Mismatched Tags

**Issue:** Opening tag doesn't match closing tag.

```xml
<!-- ❌ WRONG -->
<lvar Report.title t>AI Safety</var>
<lact search>search(...)</lvar>

<!-- ✅ CORRECT -->
<lvar Report.title t>AI Safety</lvar>
<lact search>search(...)</lact>
```

**Reason:** Parser expects matching tag types.

### Pitfall 3: Constructor Syntax in OUT{}

**Issue:** Using Pydantic constructor syntax.

```lndl
<!-- ❌ WRONG -->
OUT{report:Report(title=t, summary=s)}

<!-- ✅ CORRECT -->
OUT{report:[t, s]}
```

**Reason:** LNDL uses array mapping, not constructor calls.

### Pitfall 4: Scalar Arrays

**Issue:** Using array syntax for scalar values.

```lndl
<!-- ❌ WRONG -->
OUT{quality_score:[x, y]}

<!-- ✅ CORRECT -->
OUT{quality_score:0.85}
OUT{quality_score:[x]}
```

**Reason:** Scalars take single value or literal, not arrays.

### Pitfall 5: Name Collision

**Issue:** Same alias for lvar and lact.

```xml
<!-- ❌ WRONG -->
<lact Report.field data>search(...)</lact>
<lvar Report.field data>value</lvar>
OUT{field:[data]}  <!-- Ambiguous! -->

<!-- ✅ CORRECT -->
<lact Report.field api_data>search(...)</lact>
<lvar Report.field static_data>value</lvar>
OUT{field:[api_data]}  <!-- Clear -->
```

**Reason:** Parser can't resolve which "data" to use.

## Design Rationale

### Why LNDL Over Pure JSON?

LNDL combines **structured output** with **natural thinking**:

1. **LLMs think better in prose**: Forcing immediate JSON constrains reasoning
2. **Iterative refinement**: Multiple versions (s1, s2, s3) enable better outputs
3. **Debugging transparency**: Scratch work preserved for inspection
4. **Action laziness**: Avoid expensive API calls during exploration phase

**Comparison:**

```json
// Pure JSON: No thinking process visible
{
  "title": "AI Safety Analysis",
  "summary": "Three critical challenges..."
}
```

```xml
<!-- LNDL: Thinking process preserved -->
Let me think about the title...
<lvar Report.title t1>AI Safety</lvar>
Actually, more specific:
<lvar Report.title t2>AI Safety Analysis</lvar>

For the summary, I need to cover...
<lvar Report.summary s>Three critical challenges...</lvar>

OUT{report:[t2, s]}
```

### Why Model.field Prefix?

Explicit namespace avoids ambiguity in complex models:

```python
class Report(BaseModel):
    title: str
    summary: str

class Analysis(BaseModel):
    title: str
    conclusion: str
```

Without namespace, `title` is ambiguous. With `Report.title` vs `Analysis.title`, it's explicit.

### Why Two Action Patterns?

**Namespaced actions** (`<lact Model.field alias>`) enable **mixing** with lvars:

```xml
<lvar Report.title t>Title</lvar>
<lact Report.summary api_sum>generate_summary(...)</lact>
<lvar Report.footer f>Footer</lvar>
OUT{report:[t, api_sum, f]}
```

**Direct actions** (`<lact name>`) provide **cleaner syntax** when entire output comes from one action:

```xml
<lact search_result>search(...)</lact>
OUT{search_data:[search_result]}
```

### Why Lazy Action Execution?

LLMs explore multiple approaches during reasoning. Executing all actions wastes resources:

```xml
<lact try1>expensive_api_call(approach=1)</lact>
<lact try2>expensive_api_call(approach=2)</lact>
<lact try3>expensive_api_call(approach=3)</lact>

After thinking, approach 2 is best:
OUT{data:[try2]}
```

Only `try2` executes. Others are scratch work.

### Why OUT{} Block?

Explicit output declaration separates **thinking** (prose + tags) from **result** (final mapping):

- **Clarity**: Single source of truth for output structure
- **Validation**: Parser validates all references exist
- **Flexibility**: Change final selection without modifying tags

## See Also

- **Related Modules**:
  - [LNDL Parser](parser.md): Parses LNDL-formatted LLM responses
  - [LNDL Types](types.md): Pydantic models for LNDL structures
  - [LNDL Resolver](resolver.md): Reference resolution and validation
  - [LNDL Fuzzy](fuzzy.md): Fuzzy matching for error tolerance
  - [Spec](../types/spec.md): Type specifications
  - [Operable](../types/operable.md): Structured output integration

## Examples

### Example 1: Simple Report Generation

```python
from lionherd_core.lndl.prompt import get_lndl_system_prompt

# Setup LLM with LNDL prompt
system_prompt = get_lndl_system_prompt()

# User request
user_prompt = "Generate a research report on AI safety with title, summary, and confidence."

# LLM response (LNDL format):
"""
Let me research AI safety...

<lvar Report.title t>AI Safety: Current Challenges and Future Directions</lvar>
<lvar Report.summary s>AI safety research focuses on three critical areas:
alignment (ensuring AI systems pursue intended goals), robustness (preventing
unexpected failures), and interpretability (understanding AI decision-making).
Recent advances include mechanistic interpretability, scalable oversight, and
reward modeling.</lvar>
<lvar Report.confidence c>0.88</lvar>

OUT{report:[t, s], confidence:0.88}
"""

# Parser extracts:
# - report.title = "AI Safety: Current Challenges..."
# - report.summary = "AI safety research focuses..."
# - confidence = 0.88
```

### Example 2: Iterative Refinement

```python
# LLM explores multiple versions

"""
Let me draft a title...
<lvar Report.title t1>AI Safety</lvar>

Too generic. Let me be more specific:
<lvar Report.title t2>AI Safety Research Overview</lvar>

Actually, "Current Challenges" is more accurate:
<lvar Report.title t3>AI Safety: Current Challenges</lvar>

For the summary, initial attempt:
<lvar Report.summary s1>AI safety is important because...</lvar>

No, too vague. Let me focus on key areas:
<lvar Report.summary s2>Three critical areas: alignment, robustness, interpretability.</lvar>

OUT{report:[t3, s2]}
"""

# Parser uses final selections: t3, s2
# Earlier versions (t1, t2, s1) preserved for debugging
```

### Example 3: Action Execution with Tool Calls

```python
# LLM decides which search to use

"""
Let me search broadly:
<lact broad>search(query="AI", limit=100)</lact>

Too much noise. Let me refine:
<lact focused>search(query="AI safety research 2025", limit=20)</lact>

The focused search gives better results. Now I'll summarize:
<lvar Report.title t>AI Safety Research 2025</lvar>
<lvar Report.summary s>Based on recent findings...</lvar>

OUT{search_data:[focused], report:[t, s]}
"""

# Only "focused" action executes (lazy evaluation)
# "broad" action is never called
```

### Example 4: Mixing Variables and Actions

```python
# LLM combines generated content + API data

"""
<lvar Report.title t>Monthly Performance Report</lvar>

Let me fetch the latest metrics:
<lact Report.metrics api_metrics>fetch_metrics(period="monthly")</lact>

<lvar Report.interpretation i>Metrics show 15% improvement over last month,
driven primarily by reduced latency and increased throughput.</lvar>

<lvar Report.footer f>Report generated on 2025-11-09</lvar>

OUT{report:[t, api_metrics, i, f]}
"""

# Result:
# - title: generated by LLM
# - metrics: fetched from API
# - interpretation: generated by LLM
# - footer: generated by LLM
```

### Example 5: Scalar Outputs

```python
# LLM generates scalar values

"""
Let me evaluate quality...

<lvar quality q>0.85</lvar>
<lvar is_draft d>false</lvar>
<lvar word_count wc>1247</lvar>

OUT{quality_score:[q], is_draft:false, word_count:1247}
"""

# Scalars can use:
# - Variable reference: quality_score:[q]
# - Direct literal: is_draft:false, word_count:1247
```

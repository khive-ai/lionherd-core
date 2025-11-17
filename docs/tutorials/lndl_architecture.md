# LNDL Architecture Guide

> Complete guide to the Lexer/Parser/AST architecture for LNDL structured outputs

**Status**: Production (since PR #194, 2025-11-16)
**Complexity**: Intermediate
**Prerequisites**: Basic Python, understanding of Pydantic

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Layers](#architecture-layers)
3. [Complete Workflow](#complete-workflow)
4. [Design Philosophy](#design-philosophy)
5. [Migration from Regex API](#migration-from-regex-api)
6. [Common Patterns](#common-patterns)
7. [Performance Characteristics](#performance-characteristics)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**LNDL** (Language InterOperable Network Directive Language) is a structured output format for LLM responses, enabling type-safe variable declarations (`<lvar>`), lazy action invocations (`<lact>`), and output mappings (`OUT{}`).

### What Problem Does LNDL Solve?

**Problem**: LLMs produce unstructured text. We need **typed, validated outputs** without forcing JSON (which LLMs often break).

**Solution**: LNDL embeds structure within natural language:

```text
Here's my analysis:

<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.score s>0.95</lvar>

The system demonstrates strong safety properties...

OUT{
  title: [t],
  score: [s]
}
```

**Benefits**:

- ✅ **Natural for LLMs**: XML-like tags easier than JSON formatting
- ✅ **Type-safe**: Model/field namespacing enables validation
- ✅ **Flexible**: Mix narrative text with structured data
- ✅ **Validated**: Pydantic model construction with error reporting
- ✅ **Lazy Actions**: Deferred function execution until needed

---

## Architecture Layers

LNDL uses a **3-layer architecture** inspired by traditional compiler design:

```text
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Lexer (lionherd_core.lndl.lexer)                  │
│ ─────────────────────────────────────────────────────────── │
│ Text → Tokens                                               │
│ - Tokenization: 17 token types                              │
│ - Context-aware: Strings only inside OUT{}                  │
│ - Position tracking: Line/column for error messages         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Parser (lionherd_core.lndl.parser)                │
│ ─────────────────────────────────────────────────────────── │
│ Tokens → AST                                                │
│ - Recursive descent: Structured parsing                     │
│ - Hybrid approach: Tokens + regex for content               │
│ - Returns: Program (lvars, lacts, out_block)                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Resolver (lionherd_core.lndl.resolver)            │
│ ─────────────────────────────────────────────────────────── │
│ AST → Validated Models                                      │
│ - Validation: Against Operable specs                        │
│ - Construction: Pydantic model instances                    │
│ - Action parsing: ActionCall objects for execution          │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Separation of Concerns:**

- **Lexer**: Low-level character processing
- **Parser**: Structural analysis (syntax)
- **Resolver**: Semantic analysis (types, validation)

**Benefits:**

- **Maintainability**: Each layer has single responsibility
- **Testability**: Test tokenization, parsing, validation independently
- **Extensibility**: Add new features without touching other layers
- **Error Reporting**: Position tracking from lexer propagates to parser errors

---

## Complete Workflow

### Step-by-Step Example

Let's parse a complete LNDL response into validated Pydantic models.

#### Step 1: Define Output Schema

```python
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    score: float
    summary: str
```

#### Step 2: LLM Response (LNDL Format)

```python
llm_response = """
Here's my analysis:

<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.score s>0.95</lvar>
<lact Report.summary sum>generate_summary(data, max_words=100)</lact>

The system demonstrates strong safety properties...

OUT{
  report: [t, s, sum]
}
"""
```

#### Step 3: Tokenization (Lexer)

```python
from lionherd_core.lndl.lexer import Lexer

lexer = Lexer(llm_response)
tokens = lexer.tokenize()

# Tokens: [LVAR_OPEN, ID("Report"), DOT, ID("title"), ID("t"), GT, ...]
print(f"Generated {len(tokens)} tokens")
# Generated ~50 tokens (including EOF)
```

**What Happened:**

- Scanned text character by character
- Recognized LNDL tags (`<lvar`, `</lvar>`, `OUT{`)
- Tokenized identifiers, punctuation, literals
- Tracked position (line/column) for each token
- Ignored narrative text ("Here's my analysis...")

#### Step 4: Parsing (Parser)

```python
from lionherd_core.lndl.parser import Parser

parser = Parser(tokens, source_text=llm_response)
program = parser.parse()

# AST structure
print(f"Variables: {len(program.lvars)}")      # 2 (title, score)
print(f"Actions: {len(program.lacts)}")        # 1 (summary)
print(f"Output fields: {list(program.out_block.fields.keys())}")
# ['report']

# Inspect lvars
for lvar in program.lvars:
    print(f"{lvar.model}.{lvar.field} ({lvar.alias}): {lvar.content}")
# Report.title (t): AI Safety Analysis
# Report.score (s): 0.95

# Inspect lacts
for lact in program.lacts:
    print(f"{lact.model}.{lact.field} ({lact.alias}): {lact.call}")
# Report.summary (sum): generate_summary(data, max_words=100)
```

**What Happened:**

- Consumed tokens via recursive descent
- Built AST nodes: Lvar, Lact, OutBlock
- Extracted content via regex (preserves whitespace/quotes)
- Validated structure (closing tags, duplicate aliases)
- Returned Program (root AST node)

#### Step 5: Resolution & Validation (Resolver)

```python
from lionherd_core.types import Operable, Spec
from lionherd_core.lndl.resolver import parse_lndl

# Define allowed outputs
operable = Operable([
    Spec(Report, name="report")
])

# Parse & validate
output = parse_lndl(llm_response, operable)

# Access validated models
report_partial = output.fields["report"]
print(report_partial)
# Report(title='AI Safety Analysis', score=0.95, summary=ActionCall(...))

# Execute actions
if output.actions:
    action_results = {}
    for name, action in output.actions.items():
        # Execute function (simplified)
        action_results[name] = f"Summary of {report_partial.title}"

    # Re-validate with action results
    final_output = output.revalidate_with_action_results(action_results)
    final_report = final_output.fields["report"]
    print(final_report)
    # Report(title='AI Safety Analysis', score=0.95, summary='Summary of AI Safety Analysis')
```

**What Happened:**

- Validated OUT{} against Operable specs
- Constructed Pydantic models from AST
- Parsed actions into ActionCall objects
- Deferred full validation until action execution
- Re-validated after action results available

---

## Design Philosophy

### 1. Hybrid Parsing (Tokens + Regex)

**Why Not Pure Token-Based?**

- **Content Preservation**: Regex better for extracting content (preserves whitespace, quotes)
- **Structure Validation**: Tokens better for syntax checking (balanced tags, grammar)

**Hybrid Approach:**

```python
# Tokens for structure
<lvar Report.title t>   # Parsed via tokens
AI Safety Analysis      # Extracted via regex (content between tags)
</lvar>                 # Validated via tokens
```

**Benefits:**

- ✅ Tokens: Robust syntax validation, position tracking
- ✅ Regex: Preserves exact content (whitespace, escapes)

### 2. Context-Aware Lexing

**Problem**: Narrative text contains quotes that shouldn't be tokenized.

```text
This is "narrative" with quotes.
OUT{message: "This IS tokenized"}
```

**Solution**: `in_out_block` flag

- Set `True` when entering `OUT{`
- Set `False` when exiting `}`
- Strings tokenized **only** inside `OUT{}` blocks

**Benefit**: Prevents false positives in narrative text.

### 3. Pure Data AST

**Why Dataclasses Without Methods?**

- **Simple**: Easy to understand, serialize, debug
- **Flexible**: Operations (visitors, transformers) external
- **Testable**: Pure data structures have trivial equality

**Alternative (Rejected)**: Methods like `lvar.validate()` mix concerns.

### 4. Error Position Tracking

**All tokens have line/column:**

```python
Token(type=LVAR_OPEN, value="<lvar", line=3, column=1)
```

**Error messages:**

```text
Parse error at line 3, column 12: Unclosed lvar tag - missing </lvar>
```

**Benefit**: Direct correlation with editor UI (VS Code, Vim).

---

## Migration from Regex API

### Old API (Deprecated)

```python
# ❌ Deprecated: Regex-based extraction
from lionherd_core.lndl.parser import (
    extract_lvars_prefixed,
    extract_lacts_prefixed,
    extract_out_block,
    parse_out_block_array,
)

# Step 1: Extract components
lvars = extract_lvars_prefixed(response)
lacts = extract_lacts_prefixed(response)
out_content = extract_out_block(response)
out_fields = parse_out_block_array(out_content)

# Returns: dicts and strings (no structure)
```

**Problems:**

- No AST (just dicts/strings)
- Limited error reporting (no line/column)
- Hard to extend (regex patterns brittle)

### New API (Recommended)

```python
# ✅ Recommended: Lexer/Parser/AST
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser

# Parse to AST
lexer = Lexer(response)
parser = Parser(lexer.tokenize(), source_text=response)
program = parser.parse()

# Returns: Program (typed AST with lvars, lacts, out_block)
```

**Benefits:**

- ✅ Structured AST (typed, composable)
- ✅ Position tracking (line/column errors)
- ✅ Better validation (ParseError with context)
- ✅ Extensible (visitor pattern, transformations)

### Migration Checklist

- [ ] Replace `extract_lvars_prefixed()` with `Parser.parse()`
- [ ] Access lvars via `program.lvars` (list of Lvar/RLvar nodes)
- [ ] Access lacts via `program.lacts` (list of Lact nodes)
- [ ] Access OUT{} via `program.out_block` (OutBlock node)
- [ ] Update error handling (catch ParseError instead of ValueError)
- [ ] Test with position-aware error messages

---

## Common Patterns

### Pattern 1: Basic Parsing

```python
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser

def parse_lndl_response(response: str):
    """Parse LNDL response into AST."""
    lexer = Lexer(response)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=response)
    return parser.parse()

# Usage
program = parse_lndl_response(llm_response)
```

### Pattern 2: Error Handling with Position

```python
from lionherd_core.lndl.parser import ParseError

try:
    program = parse_lndl_response(response)
except ParseError as e:
    print(f"Syntax error: {e}")
    print(f"Location: line {e.token.line}, column {e.token.column}")
    # Syntax error: Parse error at line 3, column 12: Unclosed lvar tag
```

### Pattern 3: AST Traversal

```python
def extract_all_model_references(program):
    """Extract all unique model names from AST."""
    models = set()

    for lvar in program.lvars:
        if hasattr(lvar, 'model') and lvar.model:  # Lvar, not RLvar
            models.add(lvar.model)

    for lact in program.lacts:
        if lact.model:  # Namespaced lact
            models.add(lact.model)

    return models

# Usage
models = extract_all_model_references(program)
print(f"Models referenced: {models}")
# Models referenced: {'Report', 'User'}
```

### Pattern 4: High-Level Parsing (Fuzzy + Resolver)

```python
from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.types import Operable, Spec
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    score: float

# Define schema
operable = Operable([Spec(Report, name="report")])

# Parse with typo correction + validation
output = parse_lndl_fuzzy(llm_response, operable)

# Access validated model
report = output.fields["report"]
print(f"{report.title}: {report.score}")
```

**Note**: `parse_lndl_fuzzy()` internally uses Lexer → Parser → Fuzzy correction → Resolver.

### Pattern 5: Custom AST Transformation

```python
def rename_aliases(program, mapping: dict[str, str]):
    """Rename aliases in AST (returns new Program)."""
    from lionherd_core.lndl.ast import Program, Lvar, RLvar, Lact, OutBlock

    # Transform lvars
    new_lvars = []
    for lvar in program.lvars:
        if lvar.alias in mapping:
            if isinstance(lvar, Lvar):
                new_lvars.append(Lvar(
                    model=lvar.model,
                    field=lvar.field,
                    alias=mapping[lvar.alias],
                    content=lvar.content
                ))
            else:  # RLvar
                new_lvars.append(RLvar(
                    alias=mapping[lvar.alias],
                    content=lvar.content
                ))
        else:
            new_lvars.append(lvar)

    # Transform lacts (similar pattern)
    new_lacts = [...]

    # Transform OUT{} fields
    new_fields = {}
    for field, value in program.out_block.fields.items():
        if isinstance(value, list):
            new_fields[field] = [mapping.get(ref, ref) for ref in value]
        else:
            new_fields[field] = value

    new_out_block = OutBlock(fields=new_fields)

    return Program(lvars=new_lvars, lacts=new_lacts, out_block=new_out_block)

# Usage
renamed = rename_aliases(program, {"t": "title", "s": "score"})
```

---

## Performance Characteristics

### Complexity

| Layer | Time Complexity | Space Complexity | Notes |
|-------|----------------|------------------|-------|
| **Lexer** | O(n) | O(t) | n = text length, t = tokens (~10% of n) |
| **Parser** | O(t) | O(a) | t = tokens, a = AST nodes (~20% of t) |
| **Resolver** | O(f × m) | O(f) | f = fields, m = avg fields per model |
| **Total** | O(n) | O(n) | Linear in response size |

### Benchmarks (Approximate)

| Response Size | Lexer | Parser | Resolver | Total |
|--------------|-------|--------|----------|-------|
| Small (500 chars) | <1ms | <1ms | <1ms | ~2ms |
| Medium (5KB) | ~5ms | ~3ms | ~2ms | ~10ms |
| Large (50KB) | ~50ms | ~30ms | ~20ms | ~100ms |

**Optimization Strategies:**

- **Lexer**: Single-pass, minimal lookahead, no regex
- **Parser**: Recursive descent, O(1) token access
- **Resolver**: Early validation, batch error collection

**Bottlenecks:**

- Large responses (>50KB): Consider streaming parser
- Complex validation: Pydantic model construction dominates
- Action execution: External functions (not LNDL overhead)

---

## Troubleshooting

### Issue 1: ParseError with Unclosed Tag

**Error:**

```text
Parse error at line 5, column 1: Unclosed lvar tag - missing </lvar>
```

**Cause**: Forgot closing `</lvar>` tag.

**Fix:**

```python
# ❌ Wrong
<lvar Report.title t>Title

# ✅ Correct
<lvar Report.title t>Title</lvar>
```

### Issue 2: Duplicate Alias

**Error:**

```text
Parse error: Duplicate alias 'result' - aliases must be unique across lvars and lacts
```

**Cause**: Same alias used for lvar and lact.

**Fix:**

```python
# ❌ Wrong
<lvar Report.result result>0.95</lvar>
<lact Analysis.result result>compute()</lact>

# ✅ Correct (unique aliases)
<lvar Report.result r>0.95</lvar>
<lact Analysis.result a>compute()</lact>
```

### Issue 3: Missing source_text in Parser

**Error:**

```text
ParseError: Parser requires source_text for content extraction
```

**Cause**: Forgot to pass `source_text` to Parser.

**Fix:**

```python
# ❌ Wrong
parser = Parser(tokens)

# ✅ Correct
parser = Parser(tokens, source_text=response)
```

### Issue 4: Fuzzy Matching Failure

**Error:**

```text
MissingFieldError: Field 'titel' not found above threshold 0.85
```

**Cause**: Typo too severe for fuzzy matching.

**Fix 1**: Lower threshold

```python
parse_lndl_fuzzy(response, operable, threshold=0.75)
```

**Fix 2**: Use strict mode to identify exact error

```python
try:
    parse_lndl_fuzzy(response, operable, threshold=1.0)  # Strict
except MissingFieldError as e:
    print(f"Exact error: {e}")
    # Then correct typo in LLM prompt
```

### Issue 5: Type Validation Failure

**Error:**

```text
ValidationError: 1 validation error for Report
score
  Input should be a valid number [type=float_type]
```

**Cause**: Lvar content doesn't match Pydantic field type.

**Check:**

```python
# Lvar content is string "ninety-five" but field expects float
<lvar Report.score s>ninety-five</lvar>
```

**Fix**: Ensure LLM provides numeric string

```python
<lvar Report.score s>0.95</lvar>
```

---

## Advanced Topics

### Custom Token Types

To extend LNDL with new tags:

1. Add token type to `TokenType` enum (lexer.py)
2. Update `Lexer.tokenize()` to recognize new tag
3. Add AST node class (ast.py)
4. Update `Parser` to parse new syntax
5. Update `Resolver` to handle new node type

### AST Visitors

For complex AST analysis:

```python
from lionherd_core.lndl.ast import Program, Lvar, RLvar, Lact, OutBlock

class ASTVisitor:
    """Base visitor for AST traversal."""

    def visit_program(self, program: Program):
        for lvar in program.lvars:
            if isinstance(lvar, Lvar):
                self.visit_lvar(lvar)
            else:
                self.visit_rlvar(lvar)
        for lact in program.lacts:
            self.visit_lact(lact)
        if program.out_block:
            self.visit_out_block(program.out_block)

    def visit_lvar(self, lvar: Lvar):
        pass  # Override in subclass

    def visit_rlvar(self, rlvar: RLvar):
        pass

    def visit_lact(self, lact: Lact):
        pass

    def visit_out_block(self, out_block: OutBlock):
        pass
```

### Streaming Parser

For very large responses (>1MB):

```python
# Future: Streaming lexer
class StreamingLexer:
    def __init__(self, text_stream):
        self.stream = text_stream

    def tokenize_chunk(self, chunk_size=1024):
        """Yield tokens in chunks."""
        # Implementation: buffer-based tokenization
        pass
```

---

## Summary

**LNDL Architecture** provides a robust, extensible framework for parsing structured LLM outputs:

- **Layer 1 (Lexer)**: Text → Tokens (context-aware, position-tracked)
- **Layer 2 (Parser)**: Tokens → AST (recursive descent, hybrid approach)
- **Layer 3 (Resolver)**: AST → Models (Pydantic validation, action parsing)

**Key Benefits:**

- ✅ Type-safe LLM outputs
- ✅ Natural for LLMs (XML-like, not JSON)
- ✅ Comprehensive error reporting (line/column)
- ✅ Extensible (pure data AST, visitor pattern)
- ✅ Production-ready (comprehensive test coverage)

**Resources:**

- [Lexer API](../api/lndl/lexer.md): Tokenization reference
- [Parser API](../api/lndl/parser.md): Parsing reference
- [AST API](../api/lndl/ast.md): AST node reference
- [Resolver API](../api/lndl/resolver.md): Validation reference
- [Fuzzy Parser](../api/lndl/fuzzy.md): Typo-tolerant parsing

**Next Steps:**

1. Read [Lexer](../api/lndl/lexer.md) for tokenization details
2. Read [Parser](../api/lndl/parser.md) for parsing details
3. Read [AST](../api/lndl/ast.md) for node structure
4. Try [examples](../api/lndl/parser.md#examples) in parser docs
5. Migrate from [legacy API](../api/lndl/parser.md#legacy-api-deprecated)

---

**Version**: 1.0 (PR #194, 2025-11-16)
**Author**: HaiyangLi (Ocean)
**License**: Apache 2.0

# LNDL Lexer

> Tokenization for LNDL structured output tags with context-aware string handling

## Overview

The **LNDL Lexer** transforms LNDL (Language InterOperable Network Directive Language) responses into token streams for parsing. It provides single-pass tokenization with minimal lookahead, tracking line/column positions for robust error reporting. The Lexer is part of a **hybrid parsing approach** (tokens for structure validation, regex for content preservation) that balances robustness with flexibility.

**Key Capabilities:**

- **17 Token Types**: Tags (LVAR_OPEN, LACT_OPEN, OUT_OPEN), literals (ID, NUM, STR), punctuation, control
- **Context-Aware Lexing**: `in_out_block` flag prevents false positives (strings only tokenized inside OUT{})
- **Position Tracking**: Line/column information for every token (1-indexed)
- **Escape Sequences**: Full support for `\n`, `\t`, `\r`, `\\`, `\"`, `\'` in strings
- **Negative Numbers**: Recognized only inside OUT{} blocks (prevents narrative text interference)
- **Efficient Single-Pass**: O(n) complexity with minimal lookahead

**When to Use:**

- Tokenizing LNDL structured outputs before parsing
- Building AST-based parsers for LNDL
- Low-level LNDL processing with full position tracking
- Custom LNDL tooling and analysis

**When NOT to Use:**

- High-level LNDL parsing → Use [`parse_lndl_fuzzy()`](fuzzy.md) or [`parse_lndl()`](resolver.md)
- Simple regex-based extraction → Use [`parser.py`](parser.md) legacy functions
- Non-LNDL text → Use standard tokenizers

**Architecture Context:**

The Lexer is part of the **3-layer LNDL architecture**:

1. **Lexer** (this module): Text → Tokens
2. **Parser** ([parser.md](parser.md)): Tokens → AST
3. **Resolver** ([resolver.md](resolver.md)): AST → Validated Pydantic models

See [LNDL Architecture Guide](../../tutorials/lndl_architecture.md) for complete workflow.

**Thread Safety:**

The Lexer is **not thread-safe**. Create separate instances per thread.

```python
# ✓ Correct: Separate lexer per thread
def worker(text):
    lexer = Lexer(text)
    return lexer.tokenize()

# ✗ Wrong: Shared lexer across threads
lexer = Lexer(...)  # global
thread_pool.map(lambda t: lexer.tokenize(), texts)
```

**Reason**: Lexer maintains mutable state (`pos`, `line`, `column`, `tokens`) that would race under concurrent access.

## Classes

### TokenType

Enum defining 17 token types for LNDL structured outputs.

**Token Categories:**

| Category | Tokens | Description |
|----------|--------|-------------|
| **Tags** | `LVAR_OPEN`, `LVAR_CLOSE`, `LACT_OPEN`, `LACT_CLOSE`, `OUT_OPEN`, `OUT_CLOSE` | LNDL structure markers |
| **Literals** | `ID`, `NUM`, `STR` | Identifiers, numbers, quoted strings |
| **Punctuation** | `DOT`, `COMMA`, `COLON`, `LBRACKET`, `RBRACKET`, `LPAREN`, `RPAREN`, `GT` | Syntax elements |
| **Control** | `NEWLINE`, `EOF` | Whitespace and end-of-file |

**Tag Token Details:**

```python
# Tags (LNDL structure)
LVAR_OPEN   # <lvar
LVAR_CLOSE  # </lvar>
LACT_OPEN   # <lact
LACT_CLOSE  # </lact>
OUT_OPEN    # OUT{
OUT_CLOSE   # }
```

**Literal Token Details:**

```python
# Literals
ID    # Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
NUM   # Numbers: integers or floats (e.g., "42", "3.14", "-123")
STR   # Quoted strings: "..." or '...' with escape sequences
```

**Example:**

```python
from lionherd_core.lndl.lexer import TokenType

# Check token type
token_type = TokenType.LVAR_OPEN
print(token_type.name)  # "LVAR_OPEN"
print(token_type.value)  # Auto-assigned by Enum.auto()
```

**See Also:**

- [`Token`](#token): Token dataclass with type, value, and position
- [`Lexer.tokenize()`](#tokenize): Main tokenization method

---

### Token

Dataclass representing a single token with type, value, and position information.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | `TokenType` | Token type classification |
| `value` | `str` | Literal token value from source text |
| `line` | `int` | Line number (1-indexed) |
| `column` | `int` | Column number (1-indexed) |

**Example:**

```python
from lionherd_core.lndl.lexer import Token, TokenType

# Create token
token = Token(
    type=TokenType.LVAR_OPEN,
    value="<lvar",
    line=1,
    column=1
)

print(f"{token.type.name} at line {token.line}, column {token.column}: '{token.value}'")
# LVAR_OPEN at line 1, column 1: '<lvar'
```

**See Also:**

- [`Lexer.tokenize()`](#tokenize): Returns list of Token instances
- [`TokenType`](#tokentype): Enum of all token types

---

### Lexer

LNDL lexer for structured output tokenization with context-aware string handling.

The lexer performs **single-pass tokenization** with minimal lookahead, tracking line/column positions for error reporting. It uses an **`in_out_block` flag** to prevent tokenizing narrative text as strings (strings only recognized inside `OUT{}` blocks).

**Supported LNDL Tags:**

- `<lvar Model.field alias>content</lvar>` - Namespaced variable declarations
- `<lvar alias>content</lvar>` - Raw variable declarations
- `<lact Model.field alias>call()</lact>` - Namespaced action declarations
- `<lact alias>call()</lact>` - Direct action declarations
- `OUT{field: [refs], field2: value}` - Output specification blocks

**Constructor:**

```python
def __init__(self, text: str):
    """Initialize lexer with source text.

    Args:
        text: LNDL response text to tokenize
    """
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Source text being tokenized |
| `pos` | `int` | Current position in text (0-indexed) |
| `line` | `int` | Current line number (1-indexed) |
| `column` | `int` | Current column number (1-indexed) |
| `tokens` | `list[Token]` | Accumulated tokens |

**Methods:**

#### `tokenize()`

Tokenize LNDL source code into token stream.

**Signature:**

```python
def tokenize(self) -> list[Token]:
    """Tokenize LNDL source code into token stream.

    Returns:
        List of tokens including EOF token

    Example:
        >>> lexer = Lexer("OUT{title: [t]}")
        >>> tokens = lexer.tokenize()
        >>> len(tokens)
        8  # OUT_OPEN, ID, COLON, LBRACKET, ID, RBRACKET, OUT_CLOSE, EOF
    """
```

**Returns:**

- `list[Token]`: Token stream with EOF token at end

**Behavior:**

1. Maintains `in_out_block` flag (starts `False`, set `True` when entering `OUT{`, reset when exiting `}`)
2. Skips whitespace (space, tab, carriage return) but **preserves newlines** as tokens
3. Tokenizes strings **only inside OUT{} blocks** (prevents narrative text false positives)
4. Recognizes negative numbers **only inside OUT{} blocks** (prevents markdown/prose interference)
5. Adds `EOF` token at end for parser termination

**Examples:**

```python
from lionherd_core.lndl.lexer import Lexer

# Basic tokenization
lexer = Lexer("<lvar Report.title t>AI Safety Analysis</lvar>")
tokens = lexer.tokenize()

for token in tokens:
    print(f"{token.type.name:15} {token.value!r:20} (line {token.line}, col {token.column})")
# LVAR_OPEN       '<lvar'              (line 1, col 1)
# ID              'Report'             (line 1, col 7)
# DOT             '.'                  (line 1, col 14)
# ID              'title'              (line 1, col 15)
# ID              't'                  (line 1, col 21)
# GT              '>'                  (line 1, col 22)
# LVAR_CLOSE      '</lvar>'            (line 1, col 34)
# EOF             ''                   (line 1, col 41)

# OUT{} block with references
lexer = Lexer("OUT{title: [t], summary: [s]}")
tokens = lexer.tokenize()
print([t.type.name for t in tokens])
# ['OUT_OPEN', 'ID', 'COLON', 'LBRACKET', 'ID', 'RBRACKET', 'COMMA', ...]
```

**See Also:**

- [`Parser.parse()`](parser.md#parse): Uses token stream to build AST
- [`current_char()`](#current_char): Character access without advancing
- [`peek_char()`](#peek_char): Lookahead without advancing

---

#### `current_char()`

Get current character without advancing position.

**Signature:**

```python
def current_char(self) -> str | None:
    """Get current character without advancing.

    Returns:
        Current character or None if at end of input
    """
```

**Returns:**

- `str | None`: Character at current position, or `None` if EOF

**Example:**

```python
lexer = Lexer("hello")
print(lexer.current_char())  # 'h'
print(lexer.pos)             # 0 (unchanged)
```

---

#### `peek_char()`

Peek at character ahead without advancing position.

**Signature:**

```python
def peek_char(self, offset: int = 1) -> str | None:
    """Peek at character ahead without advancing.

    Args:
        offset: Number of characters to look ahead (default: 1)

    Returns:
        Character at offset or None if out of bounds
    """
```

**Parameters:**

- `offset` (int, default 1): Lookahead distance

**Returns:**

- `str | None`: Character at `pos + offset`, or `None` if beyond text end

**Example:**

```python
lexer = Lexer("hello")
print(lexer.peek_char(0))  # 'h' (current)
print(lexer.peek_char(1))  # 'e' (next)
print(lexer.peek_char(5))  # None (out of bounds)
```

---

#### `advance()`

Advance to next character, tracking line/column.

**Signature:**

```python
def advance(self) -> None:
    """Advance to next character, tracking line/column."""
```

**Behavior:**

- Increments `pos`
- If current character is `\n`: increments `line`, resets `column` to 0
- Otherwise: increments `column`

**Example:**

```python
lexer = Lexer("a\nb")
print(lexer.line, lexer.column)  # 1 1
lexer.advance()  # 'a'
print(lexer.line, lexer.column)  # 1 2
lexer.advance()  # '\n'
print(lexer.line, lexer.column)  # 2 0
```

---

#### `skip_whitespace()`

Skip whitespace characters except newlines.

**Signature:**

```python
def skip_whitespace(self) -> None:
    """Skip whitespace characters except newlines."""
```

**Skipped Characters:**

- Space (` `)
- Tab (`\t`)
- Carriage return (`\r`)

**NOT Skipped:**

- Newline (`\n`) - preserved as `NEWLINE` token

**Example:**

```python
lexer = Lexer("  \t  hello")
lexer.skip_whitespace()
print(lexer.current_char())  # 'h'
```

---

#### `read_identifier()`

Read identifier (alphanumeric + underscore).

**Signature:**

```python
def read_identifier(self) -> str:
    """Read identifier.

    Returns:
        Identifier string (alphanumeric + underscore)
    """
```

**Pattern:** `[a-zA-Z_][a-zA-Z0-9_]*`

**Returns:**

- `str`: Identifier string

**Example:**

```python
lexer = Lexer("user_name_123 next")
identifier = lexer.read_identifier()
print(identifier)  # "user_name_123"
print(lexer.current_char())  # ' ' (stopped at space)
```

---

#### `read_number()`

Read numeric literal (integer or float).

**Signature:**

```python
def read_number(self) -> str:
    """Read numeric literal (integer or float).

    Returns:
        Number string (digits + optional decimal point)
    """
```

**Pattern:** `[0-9]+(\.[0-9]+)?`

**Returns:**

- `str`: Number string (e.g., `"42"`, `"3.14"`)

**Note:** Returns string, not parsed number. Parser handles type conversion.

**Example:**

```python
lexer = Lexer("3.14159 next")
number = lexer.read_number()
print(number)  # "3.14159"
print(lexer.current_char())  # ' '
```

---

#### `read_string()`

Read quoted string with escape sequence handling.

**Signature:**

```python
def read_string(self) -> str:
    """Read quoted string with escape sequence handling.

    Supports escape sequences:
    - \\n: newline
    - \\t: tab
    - \\r: carriage return
    - \\\\: backslash
    - \\": double quote
    - \\': single quote

    Returns:
        String content without quotes, with escapes processed
    """
```

**Supported Escape Sequences:**

| Escape | Result | Description |
|--------|--------|-------------|
| `\n` | newline | Line feed |
| `\t` | tab | Horizontal tab |
| `\r` | carriage return | CR |
| `\\` | backslash | Literal backslash |
| `\"` | double quote | Literal `"` |
| `\'` | single quote | Literal `'` |
| `\X` (unknown) | `X` | Unknown escapes kept literal |

**Returns:**

- `str`: String content **without** surrounding quotes, escapes processed

**Behavior:**

- Advances past opening quote
- Reads until matching closing quote
- Processes escape sequences
- Advances past closing quote
- Supports both `"..."` and `'...'`

**Example:**

```python
lexer = Lexer('"Hello\\nWorld"')
lexer.advance()  # Skip opening quote
string_content = lexer.read_string()
print(string_content)  # "Hello\nWorld" (actual newline)
print(len(string_content))  # 11 (not 13)
```

---

## Usage Patterns

### Basic Tokenization

```python
from lionherd_core.lndl.lexer import Lexer

# Tokenize LNDL response
response = """
<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.score s>0.95</lvar>

OUT{
  title: [t],
  score: [s]
}
"""

lexer = Lexer(response)
tokens = lexer.tokenize()

# Inspect token types
for token in tokens:
    if token.type.name not in ["NEWLINE", "EOF"]:
        print(f"{token.type.name:15} {token.value!r}")
# LVAR_OPEN       '<lvar'
# ID              'Report'
# DOT             '.'
# ID              'title'
# ID              't'
# GT              '>'
# LVAR_CLOSE      '</lvar>'
# ...
```

### Context-Aware String Handling

```python
from lionherd_core.lndl.lexer import Lexer

# Narrative text is NOT tokenized as strings
narrative = """
This is a "story" about AI.
OUT{message: "This IS tokenized"}
"""

lexer = Lexer(narrative)
tokens = lexer.tokenize()

# Only string inside OUT{} is tokenized as STR
string_tokens = [t for t in tokens if t.type.name == "STR"]
print(len(string_tokens))  # 1
print(string_tokens[0].value)  # "This IS tokenized"
```

### Error Position Tracking

```python
from lionherd_core.lndl.lexer import Lexer

# Track token positions for error reporting
source = """<lvar Report.title t>
AI Safety Analysis
</lvar>"""

lexer = Lexer(source)
tokens = lexer.tokenize()

# Find LVAR_CLOSE token
lvar_close = next(t for t in tokens if t.type.name == "LVAR_CLOSE")
print(f"Found {lvar_close.value} at line {lvar_close.line}, column {lvar_close.column}")
# Found </lvar> at line 3, column 1
```

### Integration with Parser

```python
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser

# Lexer → Parser workflow
response = "<lvar Report.title t>Title</lvar>\nOUT{report: [t]}"

# Step 1: Lexer tokenizes
lexer = Lexer(response)
tokens = lexer.tokenize()

# Step 2: Parser builds AST
parser = Parser(tokens, source_text=response)
program = parser.parse()

print(f"Parsed {len(program.lvars)} lvars")
# Parsed 1 lvars
```

## Design Rationale

### Why Context-Aware String Tokenization?

LNDL responses mix **narrative text** with **structured outputs**:

```text
Here's my analysis: "The system is robust."

OUT{summary: "The system is robust."}
```

**Problem**: Naive lexer tokenizes both quoted strings, causing false positives.

**Solution**: `in_out_block` flag:

- Set `True` when entering `OUT{`
- Set `False` when exiting `}`
- Strings (`"..."`, `'...'`) tokenized **only** when `in_out_block == True`

**Benefit**: Narrative text ignored, only structured data tokenized.

---

### Why 1-Indexed Line/Column?

**Reason**: Matches **editor conventions** (VS Code, Vim, Emacs all use 1-indexed line numbers).

**Benefit**: Error messages directly match editor UI:

```text
Parse error at line 3, column 12
```

**Trade-off**: Internal `pos` is 0-indexed (standard for arrays), but `line` and `column` are 1-indexed for user-facing errors.

---

### Why Minimal Lookahead?

**Efficiency**: Single-pass O(n) tokenization with **at most 1-character lookahead** (for tags like `OUT{`).

**Benefits**:

- Fast: No backtracking or multi-pass
- Memory-efficient: Tokens generated on-the-fly
- Simple: Easy to understand and maintain

**Trade-off**: Tag-based design (not regex-based) requires explicit lookahead for multi-character tokens (`</lvar>`, `OUT{`).

---

### Why Separate NUM and STR Tokens?

**Type Safety**: Parser can validate literal types before Pydantic conversion:

```python
# Lexer identifies type
OUT{count: 42}       # NUM token
OUT{name: "Alice"}   # STR token

# Parser converts to Python types
count → int(42)
name → str("Alice")
```

**Benefit**: Early type detection prevents runtime errors.

---

### Why Negative Numbers Only in OUT{}?

**Narrative Interference**: Markdown/prose often uses hyphens that could be confused with negative signs:

```text
This is a sentence - with a dash.  # Would tokenize "-" + "with" as NUM if not context-aware
```

**Solution**: Negative numbers (`-123`) tokenized **only inside OUT{}** blocks.

**Benefit**: Prevents false positives in narrative text.

---

## Common Pitfalls

### Pitfall 1: Forgetting to Pass source_text to Parser

**Issue**: Parser needs original text for content extraction (regex-based hybrid approach).

```python
# ❌ WRONG: Parser receives tokens only
lexer = Lexer(response)
tokens = lexer.tokenize()
parser = Parser(tokens)  # Missing source_text!
program = parser.parse()  # Raises ParseError
```

**Solution**: Always pass source_text:

```python
# ✅ CORRECT
lexer = Lexer(response)
tokens = lexer.tokenize()
parser = Parser(tokens, source_text=response)
program = parser.parse()
```

---

### Pitfall 2: Assuming All Whitespace is Skipped

**Issue**: Newlines are **preserved** as NEWLINE tokens (for multi-line diagnostics).

```python
lexer = Lexer("OUT{\n  title: [t]\n}")
tokens = lexer.tokenize()

newlines = [t for t in tokens if t.type.name == "NEWLINE"]
print(len(newlines))  # 2 (not 0!)
```

**Solution**: Parser must explicitly skip newlines with `skip_newlines()`.

---

### Pitfall 3: Not Handling Escape Sequences

**Issue**: Raw string values include escape sequences as literals.

```python
lexer = Lexer('"Hello\\nWorld"')
# Token value is processed by read_string()
# Result: "Hello\nWorld" (actual newline, not literal "\\n")
```

**Note**: Lexer **processes** escape sequences. Parser receives processed strings.

---

## Performance Considerations

**Complexity:**

- **Time**: O(n) where n = response length
- **Space**: O(t) where t = number of tokens (~10% of n)

**Benchmarks (Approximate):**

- Small response (500 chars, 50 tokens): <1ms
- Medium response (5KB, 500 tokens): ~5ms
- Large response (50KB, 5000 tokens): ~50ms

**Optimization:**

- Single-pass: No backtracking
- Minimal lookahead: At most 1 character (for tag detection)
- No regex: Direct character-by-character scanning

---

## See Also

**Related Classes:**

- [`Parser`](parser.md#parser): Consumes token stream to build AST
- [`Program`](ast.md#program): Root AST node produced by parser
- [`TokenType`](#tokentype): Enum of all token types
- [`Token`](#token): Token dataclass

**Related Modules:**

- [LNDL Parser](parser.md): Token stream → AST
- [LNDL AST](ast.md): AST node hierarchy
- [LNDL Resolver](resolver.md): AST → Validated models
- [LNDL Architecture Guide](../../tutorials/lndl_architecture.md): Complete workflow

**External References:**

- [Lexical Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Lexical_analysis)
- [Tokenization](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)

---

## Examples

### Example 1: Complete Tokenization Workflow

```python
from lionherd_core.lndl.lexer import Lexer, TokenType

response = """
<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.score s>0.95</lvar>

OUT{
  title: [t],
  score: [s],
  confidence: 0.9
}
"""

# Tokenize
lexer = Lexer(response)
tokens = lexer.tokenize()

# Analyze token distribution
token_counts = {}
for token in tokens:
    token_counts[token.type.name] = token_counts.get(token.type.name, 0) + 1

print("Token Distribution:")
for token_type, count in sorted(token_counts.items()):
    print(f"  {token_type:15} {count:3}")
# Token Distribution:
#   COLON             3
#   COMMA             2
#   DOT               2
#   EOF               1
#   GT                2
#   ID                8
#   LBRACKET          2
#   LVAR_CLOSE        2
#   LVAR_OPEN         2
#   NEWLINE           5
#   NUM               1
#   OUT_CLOSE         1
#   OUT_OPEN          1
#   RBRACKET          2
```

### Example 2: Error Recovery with Position Tracking

```python
from lionherd_core.lndl.lexer import Lexer

# Malformed LNDL (unclosed lvar)
malformed = """
<lvar Report.title t>AI Safety
# Missing </lvar>

OUT{title: [t]}
"""

lexer = Lexer(malformed)
tokens = lexer.tokenize()

# Find OUT_OPEN position for error context
out_open = next(t for t in tokens if t.type.name == "OUT_OPEN")
print(f"OUT block starts at line {out_open.line}, column {out_open.column}")
# OUT block starts at line 4, column 1

# Parser will fail, but position info enables helpful error messages:
# "Parse error at line 4, column 1: Unclosed lvar tag - missing </lvar>"
```

### Example 3: Custom Token Filtering

```python
from lionherd_core.lndl.lexer import Lexer

response = """
<lvar Report.title t>Title</lvar>
OUT{report: [t]}
"""

lexer = Lexer(response)
tokens = lexer.tokenize()

# Filter: Get only structural tokens (tags)
structural = [t for t in tokens if t.type.name.endswith("_OPEN") or t.type.name.endswith("_CLOSE")]

print("Structural tokens:")
for token in structural:
    print(f"  {token.type.name:15} at line {token.line}")
# Structural tokens:
#   LVAR_OPEN       at line 2
#   LVAR_CLOSE      at line 2
#   OUT_OPEN        at line 3
#   OUT_CLOSE       at line 3
```

### Example 4: Debugging Tokenization Issues

```python
import logging
from lionherd_core.lndl.lexer import Lexer

# Enable debug logging to see tokenization process
logging.basicConfig(level=logging.DEBUG)

response = 'OUT{message: "Hello, World!"}'
lexer = Lexer(response)
tokens = lexer.tokenize()

# Print detailed token information
for i, token in enumerate(tokens):
    print(f"Token {i:2}: {token.type.name:15} = {token.value!r:20} (L{token.line}:C{token.column})")
# Token  0: OUT_OPEN       = 'OUT{'                (L1:C1)
# Token  1: ID             = 'message'             (L1:C5)
# Token  2: COLON          = ':'                   (L1:C12)
# Token  3: STR            = 'Hello, World!'       (L1:C14)
# Token  4: OUT_CLOSE      = '}'                   (L1:C30)
# Token  5: EOF            = ''                    (L1:C31)
```

### Example 5: Performance Profiling

```python
import time
from lionherd_core.lndl.lexer import Lexer

# Generate large LNDL response
large_response = "\n".join([
    f"<lvar Model.field{i} f{i}>value{i}</lvar>"
    for i in range(1000)
]) + "\nOUT{result: [f0]}"

# Measure tokenization time
start = time.perf_counter()
lexer = Lexer(large_response)
tokens = lexer.tokenize()
elapsed = time.perf_counter() - start

print(f"Tokenized {len(tokens)} tokens in {elapsed*1000:.2f}ms")
# Tokenized ~11000 tokens in ~50ms
# Performance: ~220,000 tokens/second
```

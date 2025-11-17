# LNDL Parser

> Recursive descent parser for LNDL structured outputs with hybrid token/regex approach

## Overview

The **LNDL parser module** provides parsing for LNDL (Language InterOperable Network Directive Language) structured outputs. Since **PR #194 (2025-11-16)**, the parser uses a **Lexer/Parser/AST architecture** for robust tokenization and structured parsing.

**Current Architecture** (Recommended):

- **[`Parser`](#parser-class)** class: Recursive descent parser consuming token streams from [`Lexer`](lexer.md)
- **Hybrid approach**: Tokens for structure, regex for content preservation (whitespace, quotes)
- **Returns**: AST ([`Program`](ast.md#program)) with typed nodes ([`Lvar`](ast.md#lvar), [`Lact`](ast.md#lact), [`OutBlock`](ast.md#outblock))

**Legacy API** (Deprecated):

- Regex-based extraction functions ([`extract_lvars()`](#extract_lvars), [`extract_lacts()`](#extract_lacts), etc.)
- Still functional but **use new Parser class** for new code

**Migration Path**:

```python
# ❌ Old (regex-based)
lvars = extract_lvars_prefixed(response)
lacts = extract_lacts_prefixed(response)
out_fields = parse_out_block_array(extract_out_block(response))

# ✅ New (Parser class)
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser

lexer = Lexer(response)
tokens = lexer.tokenize()
parser = Parser(tokens, source_text=response)
program = parser.parse()  # Returns AST with lvars, lacts, out_block
```

**When to Use:**

- Parsing LNDL responses into typed AST
- Building LNDL tooling with structured representations
- Validating LNDL syntax before resolution
- Custom LNDL transformations and analysis

**When NOT to Use:**

- High-level LNDL parsing → Use [`parse_lndl_fuzzy()`](fuzzy.md) or [`parse_lndl()`](resolver.md)
- Simple regex extraction → Use legacy functions below
- Non-LNDL formats → Use appropriate parsers

**Architecture Context:**

The Parser is part of the **3-layer LNDL architecture**:

1. **Lexer** ([lexer.md](lexer.md)): Text → Tokens
2. **Parser** (this module): Tokens → AST
3. **Resolver** ([resolver.md](resolver.md)): AST → Validated Pydantic models

See [LNDL Architecture Guide](../../tutorials/lndl_architecture.md) for complete workflow.

**Thread Safety:**

The Parser is **not thread-safe**. Create separate instances per thread.

```python
# ✓ Correct: Separate parser per thread
def worker(text):
    lexer = Lexer(text)
    parser = Parser(lexer.tokenize(), source_text=text)
    return parser.parse()

# ✗ Wrong: Shared parser across threads
parser = Parser(...)  # global
thread_pool.map(lambda t: parser.parse(), texts)
```

**Reason**: Parser maintains mutable state (`pos`) and references source text that would race under concurrent access.

---

## Parser Class

### Parser

Recursive descent parser for LNDL structured outputs.

The parser uses **recursive descent** to transform token streams from the Lexer into Abstract Syntax Trees containing [`Lvar`](ast.md#lvar), [`Lact`](ast.md#lact), and [`OutBlock`](ast.md#outblock) nodes.

**Supported LNDL Syntax:**

- **Namespaced lvars**: `<lvar Model.field alias>content</lvar>`
- **Raw lvars**: `<lvar alias>content</lvar>`
- **Namespaced lacts**: `<lact Model.field alias>func(...)</lact>`
- **Direct lacts**: `<lact alias>func(...)</lact>`
- **OUT blocks**: `OUT{field: [refs], field2: value}`

**Constructor:**

```python
def __init__(self, tokens: list[Token], source_text: str | None = None):
    """Initialize parser with token stream.

    Args:
        tokens: Token list from lexer (must include EOF token)
        source_text: Optional source text for raw content extraction
    """
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tokens` | `list[Token]` | ✅ | Token stream from Lexer (must include EOF) |
| `source_text` | `str \| None` | ✅ (for parsing) | Original LNDL response text for content extraction |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `tokens` | `list[Token]` | Token stream being parsed |
| `pos` | `int` | Current position in token stream (0-indexed) |
| `source_text` | `str \| None` | Original text for content extraction |

**Grammar:**

```text
Program    ::= (Lvar | Lact)* OutBlock?
Lvar       ::= '<lvar' (ID '.' ID ID? | ID) '>' Content '</lvar>'
Lact       ::= '<lact' (ID '.' ID ID? | ID) '>' FuncCall '</lact>'
OutBlock   ::= 'OUT{' OutFields '}'
OutFields  ::= OutField (',' OutField)*
OutField   ::= ID ':' OutValue
OutValue   ::= '[' RefList ']' | Literal
RefList    ::= ID (',' ID)*
```

**Key Methods:**

#### `parse()`

Parse token stream into AST.

**Signature:**

```python
def parse(self) -> Program:
    """Parse token stream into AST.

    Uses token-based parsing with regex content extraction to preserve whitespace and quotes.

    Returns:
        Program node containing lvars, lacts, and optional out_block

    Example:
        >>> parser = Parser(tokens, source_text)
        >>> program = parser.parse()
        >>> len(program.lvars)
        2
    """
```

**Returns:**

- [`Program`](ast.md#program): Root AST node containing:
  - `lvars`: List of [`Lvar`](ast.md#lvar) | [`RLvar`](ast.md#rlvar) instances
  - `lacts`: List of [`Lact`](ast.md#lact) instances
  - `out_block`: Optional [`OutBlock`](ast.md#outblock) instance

**Raises:**

- [`ParseError`](#parseerror): Syntax errors, unclosed tags, malformed LNDL

**Behavior:**

1. Tracks aliases to detect duplicates (lvars and lacts share namespace)
2. Skips narrative text (only parses LNDL tags)
3. Preserves newlines for error reporting
4. Extracts content via regex (preserves whitespace/quotes)

**Examples:**

```python
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser

response = """
<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.score s>0.95</lvar>

OUT{
  title: [t],
  score: [s]
}
"""

# Parse to AST
lexer = Lexer(response)
tokens = lexer.tokenize()
parser = Parser(tokens, source_text=response)
program = parser.parse()

# Access AST components
print(f"Variables: {len(program.lvars)}")       # 2
print(f"Actions: {len(program.lacts)}")         # 0
print(f"Output fields: {list(program.out_block.fields.keys())}")
# ['title', 'score']

# Inspect lvars
for lvar in program.lvars:
    print(f"{lvar.model}.{lvar.field} ({lvar.alias}): {lvar.content}")
# Report.title (t): AI Safety Analysis
# Report.score (s): 0.95
```

**See Also:**

- [`parse_lvar()`](#parse_lvar): Parse individual lvar declaration
- [`parse_lact()`](#parse_lact): Parse individual lact declaration
- [`parse_out_block()`](#parse_out_block): Parse OUT{} block

---

#### `parse_lvar()`

Parse lvar declaration (namespaced or raw).

**Signature:**

```python
def parse_lvar(self) -> Lvar | RLvar:
    """Parse lvar declaration (namespaced or raw).

    Grammar:
        Lvar  ::= '<lvar' ID '.' ID ID? '>' Content '</lvar>'  # Namespaced
        RLvar ::= '<lvar' ID '>' Content '</lvar>'              # Raw

    Namespaced pattern (maps to Pydantic model):
        <lvar Model.field alias>content</lvar>
        <lvar Model.field>content</lvar>  # Uses field as alias

    Raw pattern (simple string capture):
        <lvar alias>content</lvar>

    Returns:
        Lvar node (namespaced) or RLvar node (raw)

    Examples:
        <lvar Report.title t>AI Safety Analysis</lvar>
        → Lvar(model="Report", field="title", alias="t", content="AI Safety Analysis")

        <lvar reasoning>The analysis shows...</lvar>
        → RLvar(alias="reasoning", content="The analysis shows...")
    """
```

**Returns:**

- [`Lvar`](ast.md#lvar): If namespaced pattern (`Model.field`)
- [`RLvar`](ast.md#rlvar): If raw pattern (single identifier)

**Raises:**

- [`ParseError`](#parseerror): Syntax errors, unclosed tags, malformed content

**Behavior:**

1. Parses opening `<lvar` tag
2. Distinguishes namespaced vs raw via DOT token
3. Extracts content via regex (preserves whitespace)
4. Validates closing `</lvar>` tag

**Example:**

```python
# Namespaced lvar
lexer = Lexer("<lvar Report.title t>Title</lvar>")
parser = Parser(lexer.tokenize(), source_text=lexer.text)
program = parser.parse()

lvar = program.lvars[0]
assert lvar.model == "Report"
assert lvar.field == "title"
assert lvar.alias == "t"
assert lvar.content == "Title"
```

---

#### `parse_lact()`

Parse lact (action) declaration.

**Signature:**

```python
def parse_lact(self) -> Lact:
    """Parse lact (action) declaration.

    Grammar:
        Lact ::= '<lact' (ID '.' ID ID? | ID) '>' FuncCall '</lact>'

    Supports:
        Namespaced: <lact Model.field alias>func(...)</lact>
        Direct: <lact alias>func(...)</lact>

    Returns:
        Lact node with model, field, alias, and call string

    Example:
        <lact Report.summary s>generate_summary(prompt="...")</lact>
        → Lact(model="Report", field="summary", alias="s", call="generate_summary(...)")

        <lact search>search(query="AI")</lact>
        → Lact(model=None, field=None, alias="search", call="search(...)")
    """
```

**Returns:**

- [`Lact`](ast.md#lact): Action node with model, field, alias, call

**Raises:**

- [`ParseError`](#parseerror): Syntax errors, unclosed tags, invalid function calls
- `UserWarning`: If alias matches Python reserved keyword (deduplicated)

**Behavior:**

1. Parses opening `<lact` tag
2. Distinguishes namespaced vs direct via DOT token
3. Extracts call via regex (preserves exact syntax)
4. Warns if alias is Python reserved keyword
5. Validates closing `</lact>` tag

**Example:**

```python
# Namespaced lact
lexer = Lexer('<lact Report.summary s>generate_summary(prompt="...")</lact>')
parser = Parser(lexer.tokenize(), source_text=lexer.text)
program = parser.parse()

lact = program.lacts[0]
assert lact.model == "Report"
assert lact.field == "summary"
assert lact.alias == "s"
assert lact.call == 'generate_summary(prompt="...")'
```

---

#### `parse_out_block()`

Parse OUT{} block.

**Signature:**

```python
def parse_out_block(self) -> OutBlock:
    """Parse OUT{} block.

    Grammar:
        OutBlock  ::= 'OUT{' OutFields '}'
        OutFields ::= OutField (',' OutField)*
        OutField  ::= ID ':' OutValue
        OutValue  ::= '[' RefList ']' | Literal
        RefList   ::= ID (',' ID)*

    Returns:
        OutBlock node with fields dict

    Example:
        OUT{title: [t], summary: [s], confidence: 0.85}
        → OutBlock(fields={"title": ["t"], "summary": ["s"], "confidence": 0.85})
    """
```

**Returns:**

- [`OutBlock`](ast.md#outblock): Output specification with fields dict

**Field Value Types:**

| Type | Example | Description |
|------|---------|-------------|
| `list[str]` | `["t", "s"]` | Variable/action references (array or single) |
| `str` | `"active"` | String literal |
| `int` | `42` | Integer literal |
| `float` | `0.85` | Float literal |
| `bool` | `True` | Boolean literal (`true`/`false`) |

**Raises:**

- [`ParseError`](#parseerror): Syntax errors, malformed literals

**Behavior:**

1. Parses opening `OUT{`
2. Iterates field assignments (skip unexpected tokens)
3. Distinguishes array references from literals
4. Converts numbers to int/float, booleans to bool
5. Validates closing `}`

**Example:**

```python
lexer = Lexer("OUT{title: [t], score: 0.95, status: \"complete\"}")
parser = Parser(lexer.tokenize(), source_text=lexer.text)
program = parser.parse()

out = program.out_block
assert out.fields["title"] == ["t"]
assert out.fields["score"] == 0.95
assert out.fields["status"] == "complete"
```

---

### ParseError

Parser error with position information.

**Inheritance:** `Exception`

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Error description |
| `token` | `Token` | Token where error occurred |

**Message Format:**

```text
Parse error at line {line}, column {column}: {message}
```

**Example:**

```python
from lionherd_core.lndl.parser import Parser, ParseError
from lionherd_core.lndl.lexer import Lexer

# Unclosed lvar tag
malformed = "<lvar Report.title t>Title"  # Missing </lvar>

try:
    lexer = Lexer(malformed)
    parser = Parser(lexer.tokenize(), source_text=malformed)
    program = parser.parse()
except ParseError as e:
    print(e)
    # Parse error at line 1, column ...: Unclosed lvar tag - missing </lvar>
    print(f"Error at: line {e.token.line}, column {e.token.column}")
```

**See Also:**

- [`Parser.parse()`](#parse): Raises ParseError on syntax errors
- [`Token`](lexer.md#token): Token with position information

---

### Parser Helper Methods

#### `current_token()`

Get current token without advancing.

**Signature:**

```python
def current_token(self) -> Token:
    """Get current token without advancing.

    Returns:
        Current token or EOF if past end
    """
```

---

#### `peek_token()`

Peek at token ahead without advancing.

**Signature:**

```python
def peek_token(self, offset: int = 1) -> Token:
    """Peek at token ahead without advancing.

    Args:
        offset: Number of tokens to look ahead

    Returns:
        Token at offset or EOF if out of bounds
    """
```

---

#### `advance()`

Advance to next token.

**Signature:**

```python
def advance(self) -> None:
    """Advance to next token."""
```

---

#### `expect()`

Expect specific token type and advance.

**Signature:**

```python
def expect(self, token_type: TokenType) -> Token:
    """Expect specific token type and advance.

    Args:
        token_type: Expected token type

    Returns:
        Matched token

    Raises:
        ParseError: If current token doesn't match expected type
    """
```

---

#### `match()`

Check if current token matches any of given types.

**Signature:**

```python
def match(self, *token_types: TokenType) -> bool:
    """Check if current token matches any of given types.

    Args:
        *token_types: Token types to match against

    Returns:
        True if current token matches any type
    """
```

---

#### `skip_newlines()`

Skip newline tokens.

**Signature:**

```python
def skip_newlines(self) -> None:
    """Skip newline tokens."""
```

---

## Legacy API (Deprecated)

> **⚠️ DEPRECATED**: This API is deprecated since PR #194 (2025-11-16). Use the new [`Parser`](#parser-class) class for all new code.

**Why Deprecated:**

- **Regex-based parsing**: Limited error reporting (no line/column info)
- **No AST**: Direct dict/string extraction, not structured
- **Less robust**: Doesn't handle edge cases as well as token-based parser
- **Harder to extend**: Regex patterns difficult to maintain

**Status**: Functional but not recommended for new code. Will be maintained for backward compatibility but no new features.

**Migration Guide**:

```python
# ❌ Old: extract_lvars_prefixed() → dict
from lionherd_core.lndl.parser import extract_lvars_prefixed
lvars_dict = extract_lvars_prefixed(response)
# Returns: dict[str, LvarMetadata]

# ✅ New: Parser → AST
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser

lexer = Lexer(response)
parser = Parser(lexer.tokenize(), source_text=response)
program = parser.parse()
# Returns: Program with lvars, lacts, out_block (typed AST)

# Access lvars
for lvar in program.lvars:
    print(f"{lvar.model}.{lvar.field}: {lvar.content}")
```

**Benefits of Migration:**

- ✅ Structured AST (typed, composable)
- ✅ Position tracking (line/column error messages)
- ✅ Better error handling (ParseError with context)
- ✅ Easier testing (pure data structures)
- ✅ Extensible (visitor pattern, transformations)

---

## Module Constants

### `PYTHON_RESERVED`

Set of Python reserved keywords and common builtins that trigger warnings when used as action names.

**Type:** `set[str]`

**Contents:**

- Python keywords: `and`, `as`, `assert`, `async`, `await`, `break`, `class`, `continue`, `def`, `del`, `elif`, `else`, `except`, `finally`, `for`, `from`, `global`, `if`, `import`, `in`, `is`, `lambda`, `nonlocal`, `not`, `or`, `pass`, `raise`, `return`, `try`, `while`, `with`, `yield`
- Common builtins: `print`, `input`, `open`, `len`, `range`, `list`, `dict`, `set`, `tuple`, `str`, `int`, `float`, `bool`, `type`

**Purpose:** Prevent confusion when action names conflict with Python syntax (LNDL uses string keys, so conflicts don't break functionality but may confuse developers).

**Example:**

```python
# Action named "list" triggers warning (deduplicated)
text = '<lact list>get_items()</lact>'
lacts = extract_lacts_prefixed(text)
# UserWarning: Action name 'list' is a Python reserved keyword or builtin
```

## Functions

### Variable Extraction

#### `extract_lvars()`

Extract legacy-format lvar declarations without namespace support.

**Signature:**

```python
def extract_lvars(text: str) -> dict[str, str]: ...
```

**Parameters:**

- `text` (str): Response text containing `<lvar name>content</lvar>` declarations

**Returns:**

- dict[str, str]: Mapping of variable names to their content (whitespace-stripped)

**Examples:**

```python
>>> text = '<lvar name>Alice</lvar> <lvar age>30</lvar>'
>>> extract_lvars(text)
{'name': 'Alice', 'age': '30'}

# Preserves internal structure
>>> text = '<lvar data>line1\n  line2\n  line3</lvar>'
>>> extract_lvars(text)
{'data': 'line1\n  line2\n  line3'}
```

**Notes:**

- **Legacy format**: Superseded by `extract_lvars_prefixed()` for namespace support
- Pattern: `<lvar \w+>.*?</lvar>` (greedy match)
- Strips leading/trailing whitespace, preserves internal formatting
- Use for backward compatibility with non-namespaced LNDL responses

**See Also:**

- `extract_lvars_prefixed()`: Namespace-aware lvar extraction (recommended)

#### `extract_lvars_prefixed()`

Extract namespace-prefixed lvar declarations with model/field metadata.

**Signature:**

```python
def extract_lvars_prefixed(text: str) -> dict[str, LvarMetadata]: ...
```

**Parameters:**

- `text` (str): Response text with `<lvar Model.field alias>value</lvar>` declarations

**Returns:**

- dict[str, LvarMetadata]: Mapping of local names to metadata objects containing:
  - `model` (str): Model/class name
  - `field` (str): Field name within model
  - `local_name` (str): Local variable alias (defaults to field name if not provided)
  - `value` (str): Variable content (whitespace-stripped)

**Examples:**

```python
>>> text = '<lvar User.name>Alice</lvar>'
>>> extract_lvars_prefixed(text)
{'name': LvarMetadata(model='User', field='name', local_name='name', value='Alice')}

# With explicit alias
>>> text = '<lvar Report.summary s>Executive overview...</lvar>'
>>> extract_lvars_prefixed(text)
{'s': LvarMetadata(model='Report', field='summary', local_name='s', value='Executive overview...')}

# Multiple declarations
>>> text = '''
... <lvar User.name n>Alice</lvar>
... <lvar User.age a>30</lvar>
... '''
>>> extract_lvars_prefixed(text)
{
    'n': LvarMetadata(model='User', field='name', local_name='n', value='Alice'),
    'a': LvarMetadata(model='User', field='age', local_name='a', value='30')
}
```

**Pattern Details:**

- **Regex**: `<lvar (\w+)\.(\w+)(?:\s+(\w+))?\s*>(.*?)</lvar>`
- **Groups**: (1) model, (2) field, (3) optional alias, (4) value
- **Default alias**: If no alias provided, uses field name as local name

**See Also:**

- `LvarMetadata`: Dataclass in `lndl.types`
- `extract_lvars()`: Legacy non-namespaced extraction

### Action Extraction

#### `extract_lacts()`

Extract legacy-format lact declarations without namespace support.

**Status:** DEPRECATED - Use `extract_lacts_prefixed()` for namespace support.

**Signature:**

```python
def extract_lacts(text: str) -> dict[str, str]: ...
```

**Parameters:**

- `text` (str): Response text containing `<lact name>function_call</lact>` declarations

**Returns:**

- dict[str, str]: Mapping of action names to Python function call strings

**Examples:**

```python
>>> text = '<lact search>search(query="AI", limit=5)</lact>'
>>> extract_lacts(text)
{'search': 'search(query="AI", limit=5)'}

>>> text = '<lact analyze>analyze_data(df, method="regression")</lact>'
>>> extract_lacts(text)
{'analyze': 'analyze_data(df, method="regression")'}
```

**Notes:**

- **Legacy format**: Non-namespaced, no model/field metadata
- Actions are **lazy-evaluated**: Only executed if referenced in `OUT{}` block
- Pattern: `<lact \w+>.*?</lact>` (greedy match)
- Use `extract_lacts_prefixed()` for new code with namespace support

**See Also:**

- `extract_lacts_prefixed()`: Namespace-aware action extraction (recommended)

#### `extract_lacts_prefixed()`

Extract lact declarations with optional namespace prefix and reserved keyword warnings.

**Signature:**

```python
def extract_lacts_prefixed(text: str) -> dict[str, LactMetadata]: ...
```

**Parameters:**

- `text` (str): Response text containing `<lact>` declarations in either format:
  - Namespaced: `<lact Model.field alias>function_call()</lact>`
  - Direct: `<lact name>function_call()</lact>`

**Returns:**

- dict[str, LactMetadata]: Mapping of local names to metadata objects containing:
  - `model` (str | None): Model/class name (None for direct format)
  - `field` (str | None): Field name (None for direct format)
  - `local_name` (str): Local action alias (identifier or alias)
  - `call` (str): Python function call string (whitespace-stripped)

**Examples:**

```python
# Namespaced format
>>> text = "<lact Report.summary s>generate_summary(data, max_words=100)</lact>"
>>> extract_lacts_prefixed(text)
{
    's': LactMetadata(
        model='Report',
        field='summary',
        local_name='s',
        call='generate_summary(data, max_words=100)'
    )
}

# Direct format (no namespace)
>>> text = '<lact search>search(query="AI")</lact>'
>>> extract_lacts_prefixed(text)
{
    'search': LactMetadata(
        model=None,
        field=None,
        local_name='search',
        call='search(query="AI")'
    )
}

# Multiple actions
>>> text = '''
... <lact Query.search q>vector_search(query, top_k=5)</lact>
... <lact format>format_results(q)</lact>
... '''
>>> extract_lacts_prefixed(text)
{
    'q': LactMetadata(model='Query', field='search', local_name='q', call='vector_search(query, top_k=5)'),
    'format': LactMetadata(model=None, field=None, local_name='format', call='format_results(q)')
}
```

**Pattern Details:**

- **Regex**: `<lact ([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?(?:\s+([A-Za-z_]\w*))?>(.*?)</lact>`
- **Validation**: Strict identifier format (no leading digits, no multiple dots)
- **Default alias**: For namespaced format, uses field name if no alias provided

**Reserved Keyword Warnings:**

Action names matching `PYTHON_RESERVED` trigger `UserWarning` (deduplicated per session):

```python
>>> text = '<lact list>get_items()</lact>'
>>> extract_lacts_prefixed(text)
# UserWarning: Action name 'list' is a Python reserved keyword or builtin.
# While this works in LNDL (string keys), it may cause confusion.
{'list': LactMetadata(model=None, field=None, local_name='list', call='get_items()')}
```

**Performance:**

- **Complexity**: O(n) where n = response length
- **Regex**: Uses `(.*?)` with `DOTALL` for action body extraction
- **Limitation**: For very large responses (>100KB), parsing may be slow
- **Recommendation**: Maximum response size 50KB for optimal performance
- **Alternative**: For larger responses, consider streaming parsers

**See Also:**

- `LactMetadata`: Dataclass in `lndl.types`
- `extract_lacts()`: Legacy non-namespaced extraction
- `PYTHON_RESERVED`: Reserved keyword set

### Output Block Extraction

#### `extract_out_block()`

Extract `OUT{}` block content with balanced brace scanning and code fence support.

**Signature:**

```python
def extract_out_block(text: str) -> str: ...
```

**Parameters:**

- `text` (str): Response text containing `OUT{}` block (optionally within ` ```lndl ``` ` code fence)

**Returns:**

- str: Content inside `OUT{}` block (without outer braces, whitespace-stripped)

**Raises:**

- `MissingOutBlockError`: If no `OUT{}` block found or braces are unbalanced

**Examples:**

```python
# Basic extraction
>>> text = 'OUT{name: alice, age: 30}'
>>> extract_out_block(text)
'name: alice, age: 30'

# Code fence format
>>> text = '''
... ```lndl
... OUT{
...   summary: s,
...   results: [r1, r2]
... }
... ```
... '''
>>> extract_out_block(text)
'summary: s,\n  results: [r1, r2]'

# Nested braces (balanced)
>>> text = 'OUT{data: {nested: {deep: value}}}'
>>> extract_out_block(text)
'data: {nested: {deep: value}}'

# String-aware (ignores braces in strings)
>>> text = 'OUT{msg: "Message with {braces}"}'
>>> extract_out_block(text)
'msg: "Message with {braces}"'
```

**Behavior:**

1. **Code fence priority**: Searches for ` ```lndl ... ``` ` first
2. **Fallback**: If no code fence, searches entire text for `OUT{}`
3. **Balanced scanning**: Uses `_extract_balanced_curly()` to handle nested braces
4. **String awareness**: Ignores braces inside quoted strings

**Error Cases:**

```python
# Missing OUT block
>>> extract_out_block('No output block here')
# MissingOutBlockError: No OUT{} block found in response

# Unbalanced braces
>>> extract_out_block('OUT{missing close')
# MissingOutBlockError: Unbalanced OUT{} block
```

**See Also:**

- `_extract_balanced_curly()`: Internal balanced brace scanner
- `parse_out_block_array()`: Parse extracted block content
- `MissingOutBlockError`: Exception in `lndl.errors`

**Notes:**

- Case-insensitive: Matches `OUT`, `out`, `Out`, etc.
- Whitespace-tolerant: `OUT {` and `OUT{` both work
- Preserves internal whitespace and formatting

#### `_extract_balanced_curly()`

Internal utility for extracting balanced curly brace content, ignoring braces in strings.

**Status:** Private function - not part of public API.

**Signature:**

```python
def _extract_balanced_curly(text: str, open_idx: int) -> str: ...
```

**Parameters:**

- `text` (str): Full text containing the opening brace
- `open_idx` (int): Index of the opening `{` character

**Returns:**

- str: Content between balanced braces (without outer braces)

**Raises:**

- `MissingOutBlockError`: If braces are unbalanced

**Algorithm:**

1. Track brace depth starting at 1 (opening brace)
2. Scan character by character
3. Track string state (inside/outside quoted strings)
4. Handle escape sequences within strings
5. Increment depth on `{`, decrement on `}`
6. Return content when depth reaches 0

**String Handling:**

- Detects single (`'`) and double (`"`) quotes
- Tracks escape sequences (`\"`, `\'`, `\\`)
- Ignores braces inside strings: `{"key": "value with {braces}"}` → correctly parsed

**Example:**

```python
>>> text = 'prefix OUT{content with {nested} braces} suffix'
>>> open_idx = text.index('{')  # Index of first '{'
>>> _extract_balanced_curly(text, open_idx)
'content with {nested} braces'
```

**Notes:**

- Not intended for direct use - called by `extract_out_block()`
- Handles arbitrary nesting depth
- Robust against malformed strings (unmatched quotes)

### Output Block Parsing

#### `parse_out_block_array()`

Parse `OUT{}` block content with array syntax and literal value detection.

**Signature:**

```python
def parse_out_block_array(out_content: str) -> dict[str, list[str] | str]: ...
```

**Parameters:**

- `out_content` (str): Content inside `OUT{}` block (from `extract_out_block()`)

**Returns:**

- dict[str, list[str] | str]: Mapping of field names to either:
  - `list[str]`: Variable references (array syntax or single variable)
  - `str`: Literal scalar values (numbers, booleans, quoted strings)

**Syntax:**

```text
field_name: [var1, var2, ...]  # Array syntax → list[str]
field_name: variable_ref       # Variable reference → list[str] (single item)
field_name: "literal"          # String literal → str
field_name: 42                 # Number literal → str
field_name: true               # Boolean literal → str
```

**Examples:**

```python
# Array syntax
>>> parse_out_block_array('summary: [s1, s2, s3]')
{'summary': ['s1', 's2', 's3']}

# Single variable reference
>>> parse_out_block_array('name: user_name')
{'name': ['user_name']}

# String literal
>>> parse_out_block_array('msg: "Hello, world!"')
{'msg': '"Hello, world!"'}

# Number literal
>>> parse_out_block_array('count: 42, ratio: 3.14')
{'count': '42', 'ratio': '3.14'}

# Boolean literal
>>> parse_out_block_array('enabled: true, disabled: false')
{'enabled': 'true', 'disabled': 'false'}

# Mixed types
>>> parse_out_block_array('''
... results: [r1, r2],
... total: 100,
... status: "complete"
... ''')
{
    'results': ['r1', 'r2'],
    'total': '100',
    'status': '"complete"'
}
```

**Literal Detection Heuristic:**

Value is considered a literal if:

- Starts with `"` or `'` (quoted string)
- Matches number pattern (digits, `.`, `-`)
- Matches boolean/null keywords (`true`, `false`, `null`, case-insensitive)

Otherwise, treated as variable reference (wrapped in list for consistency).

**Parsing Rules:**

- **Delimiter**: Comma (`,`) separates field entries
- **Whitespace**: Ignored around field names, colons, and values
- **Quoted strings**: Escape sequences supported (`\"`, `\'`)
- **Nested brackets**: Supported (depth tracking)

**Edge Cases:**

```python
# Empty array
>>> parse_out_block_array('items: []')
{'items': []}

# Trailing comma (ignored)
>>> parse_out_block_array('a: x, b: y,')
{'a': ['x'], 'b': ['y']}

# Whitespace preservation in strings
>>> parse_out_block_array('msg: "  spaced  "')
{'msg': '"  spaced  "'}
```

**See Also:**

- `extract_out_block()`: Extract block content first
- `parse_value()`: Parse literal values to Python objects

**Notes:**

- Returns raw strings - use `parse_value()` to convert literals to Python types
- Designed for performance on typical LLM outputs (<10KB blocks)
- For very large blocks, consider streaming parsers

### Value Parsing

#### `parse_value()`

Parse string representation to Python object (numbers, booleans, lists, dicts, strings).

**Signature:**

```python
def parse_value(value_str: str) -> Any: ...
```

**Parameters:**

- `value_str` (str): String representation of value (e.g., `"42"`, `"true"`, `"[1, 2, 3]"`)

**Returns:**

- Any: Parsed Python object with appropriate type

**Parsing Rules:**

1. **Boolean literals** (case-insensitive):
   - `"true"` → `True`
   - `"false"` → `False`
   - `"null"` → `None`

2. **Structured literals** (via `ast.literal_eval`):
   - Numbers: `"42"` → `42`, `"3.14"` → `3.14`
   - Lists: `"[1, 2, 3]"` → `[1, 2, 3]`
   - Dicts: `"{'key': 'value'}"` → `{'key': 'value'}`
   - Tuples: `"(1, 2)"` → `(1, 2)`

3. **Fallback**: If parsing fails, returns original string

**Examples:**

```python
# Booleans (lowercase)
>>> parse_value("true")
True
>>> parse_value("false")
False
>>> parse_value("null")
None

# Numbers
>>> parse_value("42")
42
>>> parse_value("3.14159")
3.14159
>>> parse_value("-273.15")
-273.15

# Lists
>>> parse_value("[1, 2, 3]")
[1, 2, 3]
>>> parse_value("['a', 'b', 'c']")
['a', 'b', 'c']

# Dicts
>>> parse_value("{'name': 'Alice', 'age': 30}")
{'name': 'Alice', 'age': 30}

# Strings (fallback)
>>> parse_value("not a literal")
'not a literal'
>>> parse_value("hello world")
'hello world'

# Invalid literals (fallback to string)
>>> parse_value("[1, 2, invalid]")
'[1, 2, invalid]'
```

**Safety:**

Uses `ast.literal_eval()` for safe evaluation:

- **Allowed**: Literals only (numbers, strings, lists, dicts, tuples, booleans, None)
- **Forbidden**: Code execution, function calls, variable references
- **Security**: Safe for untrusted input (no arbitrary code execution)

**Error Handling:**

```python
# Syntax errors → fallback to string
>>> parse_value("{invalid: json}")
'{invalid: json}'

# Mixed valid/invalid → fallback
>>> parse_value("[1, 2, undefined_var]")
'[1, 2, undefined_var]'
```

**See Also:**

- `parse_out_block_array()`: Produces string values suitable for this function
- `ast.literal_eval()`: Underlying safe evaluation function

**Notes:**

- Handles lowercase boolean literals (`true`/`false`) which `ast.literal_eval` doesn't support
- Always returns a value (never raises exceptions)
- Useful for converting parsed OUT block literals to typed Python objects

## Usage Patterns

### Basic LNDL Response Parsing

```python
from lionherd_core.lndl.parser import (
    extract_lvars_prefixed,
    extract_lacts_prefixed,
    extract_out_block,
    parse_out_block_array,
)

# LLM response with LNDL format
response = '''
<lvar User.name n>Alice Johnson</lvar>
<lvar User.age a>30</lvar>
<lact Query.search q>vector_search(query="AI tools", top_k=5)</lact>
<lact format>format_results(q)</lact>

OUT{
  user_info: [n, a],
  search_results: format
}
'''

# Extract variables
lvars = extract_lvars_prefixed(response)
# {'n': LvarMetadata(..., value='Alice Johnson'), 'a': LvarMetadata(..., value='30')}

# Extract actions
lacts = extract_lacts_prefixed(response)
# {'q': LactMetadata(..., call='vector_search(...)'), 'format': LactMetadata(...)}

# Parse output mapping
out_content = extract_out_block(response)
# 'user_info: [n, a],\n  search_results: format'

fields = parse_out_block_array(out_content)
# {'user_info': ['n', 'a'], 'search_results': ['format']}
```

### Code Fence Format Support

````python
# LLM wraps LNDL in code fence
response = '''
Here's the structured output:

```lndl
<lvar Report.summary s>Executive summary of Q3 results...</lvar>

OUT{
  summary: s
}
```

This completes the analysis.
'''

# Extraction works with or without code fence

out_content = extract_out_block(response)

# 'summary: s'

fields = parse_out_block_array(out_content)

# {'summary': ['s']}

````

### Handling Literal Values

```python
from lionherd_core.lndl.parser import parse_out_block_array, parse_value

# OUT block with mixed literals and variables
out_content = '''
count: 42,
ratio: 3.14,
enabled: true,
message: "Processing complete",
results: [r1, r2, r3]
'''

fields = parse_out_block_array(out_content)
# {
#   'count': '42',
#   'ratio': '3.14',
#   'enabled': 'true',
#   'message': '"Processing complete"',
#   'results': ['r1', 'r2', 'r3']
# }

# Convert literals to Python types
for key, value in fields.items():
    if isinstance(value, str):  # Literal value
        fields[key] = parse_value(value)
    # else: list of variable references, leave as-is

# {
#   'count': 42,
#   'ratio': 3.14,
#   'enabled': True,
#   'message': 'Processing complete',
#   'results': ['r1', 'r2', 'r3']
# }
```

### Error Handling

```python
from lionherd_core.lndl.parser import extract_out_block
from lionherd_core.lndl.errors import MissingOutBlockError

response = "No LNDL output here"

try:
    out_content = extract_out_block(response)
except MissingOutBlockError as e:
    print(f"Failed to parse response: {e}")
    # Handle missing OUT block (e.g., retry, use default, log error)
```

### Reserved Keyword Warnings

```python
import warnings

# Capture warnings for action names
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    text = '<lact list>get_items()</lact> <lact range>get_range()</lact>'
    lacts = extract_lacts_prefixed(text)

    # Check for warnings
    for warning in w:
        print(f"Warning: {warning.message}")
    # Warning: Action name 'list' is a Python reserved keyword or builtin...
    # Warning: Action name 'range' is a Python reserved keyword or builtin...

# Warnings are deduplicated per session
text2 = '<lact list>another_call()</lact>'  # No warning (already warned)
lacts2 = extract_lacts_prefixed(text2)
```

## Common Pitfalls

### Pitfall 1: Forgetting to Strip Quotes from Parsed Literals

**Issue**: Quoted string literals include surrounding quotes.

```python
fields = parse_out_block_array('msg: "hello"')
# {'msg': '"hello"'}  # Quotes included

# Incorrect: Using quoted string directly
print(fields['msg'])  # "hello" (with quotes)
```

**Solution**: Use `parse_value()` to convert strings properly, or strip quotes manually.

```python
value = parse_value(fields['msg'])
# If parse_value returns the string as-is, strip quotes:
if value.startswith('"') and value.endswith('"'):
    value = value[1:-1]
# 'hello'
```

### Pitfall 2: Assuming All Fields Are Lists

**Issue**: `parse_out_block_array()` returns either `list[str]` or `str` depending on literal detection.

```python
fields = parse_out_block_array('count: 42, items: [a, b]')
# {'count': '42', 'items': ['a', 'b']}

# Incorrect: Assuming all fields are lists
for item in fields['count']:  # TypeError: str is not iterable
    print(item)
```

**Solution**: Check type before iteration or normalize to lists.

```python
for key, value in fields.items():
    items = value if isinstance(value, list) else [value]
    for item in items:
        print(f"{key}: {item}")
```

### Pitfall 3: Not Handling Missing OUT Blocks

**Issue**: Assuming `extract_out_block()` always succeeds.

```python
# May raise MissingOutBlockError
out_content = extract_out_block(response)
fields = parse_out_block_array(out_content)
```

**Solution**: Wrap in try/except for robust error handling.

```python
from lionherd_core.lndl.errors import MissingOutBlockError

try:
    out_content = extract_out_block(response)
    fields = parse_out_block_array(out_content)
except MissingOutBlockError:
    # Use default mapping or retry
    fields = {}
```

### Pitfall 4: Ignoring Namespace Information

**Issue**: Extracting prefixed lvars/lacts but only using local names, losing model/field context.

```python
lacts = extract_lacts_prefixed(text)
# {'q': LactMetadata(model='Query', field='search', ...)}

# Incorrect: Only using local name
action_call = lacts['q'].call  # Loses model/field info
```

**Solution**: Use full metadata for validation or model construction.

```python
for local_name, metadata in lacts.items():
    if metadata.model and metadata.field:
        # Validate against model schema
        validate_field(metadata.model, metadata.field)
    execute_action(metadata.call)
```

### Pitfall 5: Large Response Performance

**Issue**: Parsing very large responses (>100KB) may be slow due to regex greedy matching.

```python
# Slow for multi-megabyte responses
huge_response = generate_huge_response()  # 5MB
lacts = extract_lacts_prefixed(huge_response)  # May take seconds
```

**Solution**: Limit response size, use streaming parsers, or pre-filter text.

```python
# Limit response size
MAX_RESPONSE_SIZE = 50 * 1024  # 50KB
if len(response) > MAX_RESPONSE_SIZE:
    response = response[:MAX_RESPONSE_SIZE]

# Or extract only relevant sections
lndl_section = extract_lndl_code_fence(response)  # Custom function
lacts = extract_lacts_prefixed(lndl_section)
```

## Design Rationale

### Why Separate Extraction and Parsing?

**Extraction** (`extract_out_block`) and **parsing** (`parse_out_block_array`) are separate steps to enable:

1. **Flexibility**: Use different parsers for OUT block content (e.g., legacy vs. array syntax)
2. **Error Isolation**: Extraction errors (missing block) distinct from parsing errors (malformed content)
3. **Reusability**: Extracted content can be inspected/logged before parsing
4. **Performance**: Extract once, parse multiple times with different strategies

### Why String-Aware Brace Scanning?

Naive brace counting fails on valid outputs like:

```python
OUT{msg: "Message with {braces}"}
```

String-aware scanning (`_extract_balanced_curly`) correctly handles:

- Quoted strings with braces
- Escape sequences (`\"`, `\'`)
- Mixed single/double quotes

This ensures robust parsing of arbitrary LLM outputs.

### Why Reserved Keyword Warnings Instead of Errors?

LNDL uses **string keys** for action names, so Python reserved keywords don't break functionality:

```python
lacts = {'list': 'get_items()'}  # Valid dict key
```

However, developers may be confused seeing `list` as an action name. Warnings:

- **Educate** without blocking (non-breaking)
- **Deduplicate** to avoid spam (global `_warned_action_names` set)
- **Inform** design decisions (choose better action names)

### Why Literal vs. Variable Detection?

`parse_out_block_array()` distinguishes:

- **Literals** (`"hello"`, `42`, `true`): Direct values → return as `str`
- **Variables** (`user_name`, `result`): References → return as `list[str]`

This enables downstream code to:

1. Resolve variable references from extracted lvars/lacts
2. Use literals directly without resolution
3. Apply type conversion only to literals

**Heuristic design**: Prioritizes correctness (false positive = string, no harm) over optimization.

### Why Support Code Fence Format?

LLMs often wrap structured outputs in markdown code fences:

````markdown
Here's the result:

```lndl
OUT{...}
```

````

Supporting this format improves:

- **Robustness**: Works with typical LLM output patterns
- **Readability**: Code fences improve response formatting
- **Compatibility**: Handles responses from different prompting strategies

### Why Use `ast.literal_eval` for Safety?

Direct `eval()` enables arbitrary code execution (security risk):

```python
# UNSAFE
eval("[1, 2, os.system('rm -rf /')]")  # Executes malicious code
```

`ast.literal_eval()` only parses literals:

```python
# SAFE
ast.literal_eval("[1, 2, 3]")  # OK
ast.literal_eval("os.system('rm -rf /')")  # SyntaxError
```

This ensures **safe parsing** of untrusted LLM outputs.

## See Also

- **Related Modules**:
  - [lndl.types](types.md): `LvarMetadata`, `LactMetadata` dataclasses
  - [lndl.errors](errors.md): `MissingOutBlockError` exception
  - [lndl.resolver](resolver.md): Reference resolution and validation
  - [lndl.fuzzy](fuzzy.md): Fuzzy matching for error tolerance
  - [Spec](../types/spec.md): Type specifications
  - [Operable](../types/operable.md): Structured output integration
- **Standard Library**:
  - [`ast.literal_eval`](https://docs.python.org/3/library/ast.html#ast.literal_eval): Safe literal evaluation
  - [`re`](https://docs.python.org/3/library/re.html): Regular expressions module

## Examples

### Example 1: Complete LNDL Response Processing

```python
from lionherd_core.lndl.parser import (
    extract_lvars_prefixed,
    extract_lacts_prefixed,
    extract_out_block,
    parse_out_block_array,
    parse_value,
)

# LLM response
response = """
Analyzing user data...

<lvar User.name n>Alice Johnson</lvar>
<lvar User.age a>30</lvar>
<lvar User.active active>true</lvar>

<lact Query.fetch q>fetch_user_data(user_id=12345)</lact>
<lact analyze>analyze_activity(q, days=30)</lact>

```lndl
OUT{
  user_name: n,
  user_age: a,
  is_active: active,
  analysis: analyze
}```
Analysis complete.
"""

# Step 1: Extract variables

lvars = extract_lvars_prefixed(response)
print("Variables:")
for name, meta in lvars.items():
    print(f"  {name}: {meta.value} (from {meta.model}.{meta.field})")

# Step 2: Extract actions

lacts = extract_lacts_prefixed(response)
print("\nActions:")
for name, meta in lacts.items():
    ns = f"{meta.model}.{meta.field}" if meta.model else "direct"
    print(f"  {name}: {meta.call} ({ns})")

# Step 3: Parse output block

out_content = extract_out_block(response)
fields = parse_out_block_array(out_content)
print("\nOutput Mapping:")
for field, refs in fields.items():
    print(f"  {field}: {refs}")

# Step 4: Resolve output values

output = {}
for field, refs in fields.items():
    if isinstance(refs, str):  # Literal value
        output[field] = parse_value(refs)
    else:  # Variable references
        output[field] = [lvars[ref].value if ref in lvars else lacts[ref].call
                         for ref in refs]

print("\nResolved Output:")
print(output)

# {

# 'user_name': 'Alice Johnson'

# 'user_age': '30'

# 'is_active': 'true'

# 'analysis': ['analyze_activity(q, days=30)']

# }

```

### Example 2: Error Recovery

```python
from lionherd_core.lndl.parser import extract_out_block
from lionherd_core.lndl.errors import MissingOutBlockError

def safe_parse_response(response: str) -> dict:
    """Parse LNDL response with error recovery."""
    try:
        out_content = extract_out_block(response)
        fields = parse_out_block_array(out_content)
        return fields
    except MissingOutBlockError:
        # Log error and return default
        print("Warning: No OUT block found, using defaults")
        return {}
    except Exception as e:
        # Unexpected error
        print(f"Error parsing response: {e}")
        return {}

# Test with various responses
response1 = "OUT{result: x}"
response2 = "No LNDL output here"
response3 = "OUT{malformed"

print(safe_parse_response(response1))  # {'result': ['x']}
print(safe_parse_response(response2))  # {}
print(safe_parse_response(response3))  # {}
```

### Example 3: Namespace Validation

```python
from lionherd_core.lndl.parser import extract_lacts_prefixed

# Define expected schema
VALID_MODELS = {'Query', 'Report', 'Analysis'}
VALID_FIELDS = {
    'Query': {'search', 'filter', 'aggregate'},
    'Report': {'summary', 'detail', 'export'},
    'Analysis': {'compute', 'visualize'}
}

def validate_lacts(response: str) -> dict[str, str]:
    """Extract and validate namespaced lacts."""
    lacts = extract_lacts_prefixed(response)
    errors = []

    for name, meta in lacts.items():
        if meta.model:  # Namespaced
            if meta.model not in VALID_MODELS:
                errors.append(f"Invalid model: {meta.model}")
            elif meta.field not in VALID_FIELDS.get(meta.model, set()):
                errors.append(f"Invalid field: {meta.model}.{meta.field}")

    if errors:
        raise ValueError(f"Validation errors: {', '.join(errors)}")

    return lacts

# Valid
response1 = '<lact Query.search q>search()</lact>'
print(validate_lacts(response1))  # OK

# Invalid model
response2 = '<lact InvalidModel.field f>call()</lact>'
# Raises: ValueError: Validation errors: Invalid model: InvalidModel

# Invalid field
response3 = '<lact Query.invalid_field f>call()</lact>'
# Raises: ValueError: Validation errors: Invalid field: Query.invalid_field
```

### Example 4: Performance Optimization for Large Responses

```python
from lionherd_core.lndl.parser import extract_out_block, parse_out_block_array
import re

def extract_lndl_section(response: str, max_size: int = 50_000) -> str:
    """Extract only LNDL code fence section to reduce parsing overhead."""
    # Try to find code fence first
    pattern = r"```lndl\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        section = match.group(1)
    else:
        # No code fence, limit entire response
        section = response[:max_size]

    return section

def parse_large_response(response: str) -> dict:
    """Optimized parsing for large responses."""
    # Extract relevant section
    lndl_section = extract_lndl_section(response)

    # Parse only the section
    out_content = extract_out_block(lndl_section)
    fields = parse_out_block_array(out_content)

    return fields

# Test with large response
large_response = "..." + "```lndl\nOUT{result: x}\n```" + "..."*10000
fields = parse_large_response(large_response)  # Fast
```

### Example 5: Type-Safe Value Resolution

```python
from lionherd_core.lndl.parser import (
    extract_lvars_prefixed,
    parse_out_block_array,
    parse_value,
)
from typing import Any

def resolve_output_field(field_name: str, refs: list[str] | str,
                          lvars: dict[str, Any]) -> Any:
    """Resolve output field with type conversion."""
    if isinstance(refs, str):
        # Literal value - parse to Python type
        return parse_value(refs)
    else:
        # Variable references - resolve and convert
        values = []
        for ref in refs:
            if ref in lvars:
                raw_value = lvars[ref].value
                # Try to parse as typed value
                values.append(parse_value(raw_value))
            else:
                # Action reference (not resolved here)
                values.append(ref)

        # Return single value if list has one item
        return values[0] if len(values) == 1 else values

# Example usage
response = '''
<lvar Config.timeout t>30</lvar>
<lvar Config.enabled e>true</lvar>
<lvar Config.servers s>["srv1", "srv2"]</lvar>

OUT{
  timeout: t,
  enabled: e,
  servers: s,
  retry_count: 3
}
'''

lvars = extract_lvars_prefixed(response)
out_content = extract_out_block(response)
fields = parse_out_block_array(out_content)

# Resolve with type conversion
output = {
    field: resolve_output_field(field, refs, lvars)
    for field, refs in fields.items()
}

print(output)
# {
#   'timeout': 30,           # int (from string "30")
#   'enabled': True,         # bool (from string "true")
#   'servers': ['srv1', 'srv2'],  # list (from string "['srv1', 'srv2']")
#   'retry_count': 3         # int (literal)
# }
```

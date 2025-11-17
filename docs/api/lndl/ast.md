# LNDL AST Nodes

> Abstract Syntax Tree for LNDL structured outputs (simplified, data-only design)

## Overview

The **LNDL AST module** defines the Abstract Syntax Tree for LNDL (Language InterOperable Network Directive Language) structured outputs. It provides a **pure data representation** (dataclasses with no methods) for type-safe AST manipulation.

**Design Philosophy:**

- **Pure Data**: Dataclasses with `@dataclass(slots=True)`, no methods
- **Type-Safe**: Full type annotations for all fields
- **Simple and Clear**: No over-engineering, minimal abstraction
- **Structured Outputs Only**: Semantic operations deferred for future phases

**Key Capabilities:**

- **3 Base Node Types**: ASTNode, Expr (expressions), Stmt (statements)
- **5 Statement Types**: Lvar, RLvar, Lact, OutBlock, Program
- **2 Expression Types**: Literal, Identifier
- **Slots Optimization**: Memory-efficient via `__slots__`

**When to Use:**

- Building parsers for LNDL structured outputs
- AST-based analysis and transformation
- Custom LNDL tooling and validation
- Intermediate representation before model construction

**When NOT to Use:**

- High-level LNDL parsing → Use [`parse_lndl_fuzzy()`](fuzzy.md) or [`parse_lndl()`](resolver.md)
- Direct model construction → Use [`Resolver`](resolver.md)
- Simple string extraction → Use legacy [`parser.py`](parser.md) functions

**Architecture Context:**

The AST is part of the **3-layer LNDL architecture**:

1. **Lexer** ([lexer.md](lexer.md)): Text → Tokens
2. **Parser** ([parser.md](parser.md)): Tokens → **AST** (this module)
3. **Resolver** ([resolver.md](resolver.md)): AST → Validated Pydantic models

See [LNDL Architecture Guide](../../tutorials/lndl_architecture.md) for complete workflow.

---

## Node Hierarchy

```text
ASTNode (base, slots=())
├── Expr (expressions)
│   ├── Literal (int | float | str | bool)
│   └── Identifier (name: str)
└── Stmt (statements)
    ├── Lvar (model, field, alias, content)
    ├── RLvar (alias, content)
    ├── Lact (model, field, alias, call)
    ├── OutBlock (fields: dict)
    └── Program (lvars, lacts, out_block)
```

---

## Base Nodes

### ASTNode

Base class for all LNDL AST nodes.

**Signature:**

```python
class ASTNode:
    """Base AST node for all LNDL constructs."""

    __slots__ = ()  # Empty slots for proper inheritance
```

**Purpose:**

- Common base type for all AST nodes
- Enables type checking: `isinstance(node, ASTNode)`
- Empty `__slots__` allows proper slot inheritance in subclasses

**Usage:**

```python
from lionherd_core.lndl.ast import ASTNode, Lvar

# Check if node is AST node
node = Lvar(model="Report", field="title", alias="t", content="Title")
assert isinstance(node, ASTNode)
```

---

### Expr

Base class for expression nodes (evaluate to values).

**Signature:**

```python
class Expr(ASTNode):
    """Base expression node."""

    __slots__ = ()
```

**Subclasses:**

- [`Literal`](#literal): Scalar values (int, float, str, bool)
- [`Identifier`](#identifier): Variable references

**Example:**

```python
from lionherd_core.lndl.ast import Expr, Literal, Identifier

# All expressions inherit from Expr
lit = Literal(value=42)
ident = Identifier(name="title")

assert isinstance(lit, Expr)
assert isinstance(ident, Expr)
```

---

### Stmt

Base class for statement nodes (declarations, no return value).

**Signature:**

```python
class Stmt(ASTNode):
    """Base statement node."""

    __slots__ = ()
```

**Subclasses:**

- [`Lvar`](#lvar): Namespaced variable declaration
- [`RLvar`](#rlvar): Raw variable declaration
- [`Lact`](#lact): Action declaration
- [`OutBlock`](#outblock): Output specification
- [`Program`](#program): Root node (collection of statements)

**Example:**

```python
from lionherd_core.lndl.ast import Stmt, Lvar, OutBlock

# All statements inherit from Stmt
lvar = Lvar(model="Report", field="title", alias="t", content="Title")
out = OutBlock(fields={"title": ["t"]})

assert isinstance(lvar, Stmt)
assert isinstance(out, Stmt)
```

---

## Expression Nodes

### Literal

Scalar literal value (int, float, str, bool).

**Signature:**

```python
@dataclass(slots=True)
class Literal(Expr):
    """Literal scalar value.

    Examples:
        - "AI safety"
        - 42
        - 0.85
        - true
    """

    value: str | int | float | bool
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `value` | `str \| int \| float \| bool` | Scalar literal value |

**Examples:**

```python
from lionherd_core.lndl.ast import Literal

# String literal
str_lit = Literal(value="AI Safety")
print(str_lit.value)  # "AI Safety"

# Integer literal
int_lit = Literal(value=42)
print(int_lit.value)  # 42

# Float literal
float_lit = Literal(value=0.85)
print(float_lit.value)  # 0.85

# Boolean literal
bool_lit = Literal(value=True)
print(bool_lit.value)  # True
```

**Use Cases:**

- Literal values in `OUT{}` blocks: `OUT{count: 42, status: "active"}`
- Constant expressions in future semantic operations

**See Also:**

- [`Identifier`](#identifier): Variable references
- [`OutBlock`](#outblock): Uses literals in field values

---

### Identifier

Variable reference (e.g., `[title]`, `[summary]`).

**Signature:**

```python
@dataclass(slots=True)
class Identifier(Expr):
    """Variable reference.

    Examples:
        - [title]
        - [summary]
    """

    name: str
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Variable reference name |

**Examples:**

```python
from lionherd_core.lndl.ast import Identifier

# Variable reference
ident = Identifier(name="title")
print(ident.name)  # "title"

# Used in OUT{} blocks: OUT{result: [title]}
```

**Use Cases:**

- Variable references in `OUT{}` blocks: `OUT{title: [t]}`
- Future semantic operations (variable resolution)

**See Also:**

- [`Literal`](#literal): Scalar values
- [`OutBlock`](#outblock): Uses identifiers in field references

---

## Statement Nodes

### Lvar

Namespaced variable declaration - maps to Pydantic model field.

**Signature:**

```python
@dataclass(slots=True)
class Lvar(Stmt):
    """Namespaced variable declaration - maps to Pydantic model field.

    Syntax:
        <lvar Model.field alias>content</lvar>
        <lvar Model.field>content</lvar>  # Uses field as alias

    Examples:
        <lvar Report.title t>AI Safety Analysis</lvar>
        → Lvar(model="Report", field="title", alias="t", content="AI Safety Analysis")

        <lvar Report.score>0.95</lvar>
        → Lvar(model="Report", field="score", alias="score", content="0.95")
    """

    model: str  # Model name (e.g., "Report")
    field: str  # Field name (e.g., "title", "score")
    alias: str  # Local variable name (e.g., "t", defaults to field)
    content: str  # Raw string value
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model/class name (e.g., `"Report"`) |
| `field` | `str` | Field name within model (e.g., `"title"`, `"score"`) |
| `alias` | `str` | Local variable alias (e.g., `"t"`, defaults to field if not provided) |
| `content` | `str` | Raw string value from LNDL response |

**LNDL Syntax:**

```text
# Explicit alias
<lvar Model.field alias>content</lvar>

# Implicit alias (uses field name)
<lvar Model.field>content</lvar>
```

**Examples:**

```python
from lionherd_core.lndl.ast import Lvar

# Explicit alias
lvar1 = Lvar(
    model="Report",
    field="title",
    alias="t",
    content="AI Safety Analysis"
)
print(f"{lvar1.model}.{lvar1.field} ({lvar1.alias}): {lvar1.content}")
# Report.title (t): AI Safety Analysis

# Implicit alias (alias=field)
lvar2 = Lvar(
    model="Report",
    field="score",
    alias="score",
    content="0.95"
)
print(f"{lvar2.alias}: {lvar2.content}")
# score: 0.95
```

**Use Cases:**

- Namespaced variable declarations in LNDL responses
- Mapping LLM outputs to specific Pydantic model fields
- Type-safe variable resolution via model/field metadata

**See Also:**

- [`RLvar`](#rlvar): Raw (non-namespaced) variable declarations
- [`Program`](#program): Contains list of Lvar instances

---

### RLvar

Raw variable declaration - simple string capture without model mapping.

**Signature:**

```python
@dataclass(slots=True)
class RLvar(Stmt):
    """Raw variable declaration - simple string capture without model mapping.

    Syntax:
        <lvar alias>content</lvar>

    Examples:
        <lvar reasoning>The analysis shows...</lvar>
        → RLvar(alias="reasoning", content="The analysis shows...")

        <lvar score>0.95</lvar>
        → RLvar(alias="score", content="0.95")

    Usage:
        - Use for intermediate LLM outputs not mapped to Pydantic models
        - Can only resolve to scalar OUT{} fields (str, int, float, bool)
        - Cannot be used in BaseModel OUT{} fields (no type validation)
    """

    alias: str  # Local variable name
    content: str  # Raw string value
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `alias` | `str` | Local variable name (identifier) |
| `content` | `str` | Raw string value from LNDL response |

**LNDL Syntax:**

```text
<lvar alias>content</lvar>
```

**Examples:**

```python
from lionherd_core.lndl.ast import RLvar

# Raw variable (no model namespace)
rlvar = RLvar(
    alias="reasoning",
    content="The analysis shows strong performance."
)
print(f"{rlvar.alias}: {rlvar.content}")
# reasoning: The analysis shows strong performance.

# Scalar value
scalar = RLvar(alias="confidence", content="0.95")
print(f"{scalar.alias} = {scalar.content}")
# confidence = 0.95
```

**Use Cases:**

- Intermediate LLM outputs not mapped to Pydantic models
- Scalar OUT{} fields (str, int, float, bool)
- Temporary variables in multi-step reasoning

**Restrictions:**

- **Cannot** be used in BaseModel OUT{} fields (no model/field metadata for validation)
- **Can** be used in scalar OUT{} fields: `OUT{reasoning: [reasoning], score: [score]}`

**See Also:**

- [`Lvar`](#lvar): Namespaced variable declarations with model/field metadata
- [`Program`](#program): Contains list of Lvar | RLvar instances

---

### Lact

Action declaration (function call for lazy evaluation).

**Signature:**

```python
@dataclass(slots=True)
class Lact(Stmt):
    """Action declaration.

    Syntax:
        - Namespaced: <lact Model.field alias>func(...)</lact>
        - Direct: <lact alias>func(...)</lact>

    Examples:
        <lact Report.summary s>generate_summary(prompt="...")</lact>
        → Lact(model="Report", field="summary", alias="s", call="generate_summary(...)")

        <lact search>search(query="AI")</lact>
        → Lact(model=None, field=None, alias="search", call="search(...)")
    """

    model: str | None  # Model name or None for direct actions
    field: str | None  # Field name or None for direct actions
    alias: str  # Local reference name
    call: str  # Raw function call string
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str \| None` | Model name (e.g., `"Report"`) or `None` for direct actions |
| `field` | `str \| None` | Field name (e.g., `"summary"`) or `None` for direct actions |
| `alias` | `str` | Local action alias/reference name |
| `call` | `str` | Raw function call string (e.g., `"generate_summary(prompt=\"...\")"`) |

**LNDL Syntax:**

```text
# Namespaced action (targets specific field)
<lact Model.field alias>func(...)</lact>

# Direct action (returns entire model)
<lact alias>func(...)</lact>
```

**Examples:**

```python
from lionherd_core.lndl.ast import Lact

# Namespaced action (populates specific field)
namespaced = Lact(
    model="Report",
    field="summary",
    alias="s",
    call='generate_summary(data, max_words=100)'
)
print(f"{namespaced.model}.{namespaced.field} ({namespaced.alias}): {namespaced.call}")
# Report.summary (s): generate_summary(data, max_words=100)

# Direct action (returns entire model)
direct = Lact(
    model=None,
    field=None,
    alias="fetch_user",
    call='get_user(user_id=123)'
)
print(f"{direct.alias}: {direct.call}")
# fetch_user: get_user(user_id=123)
```

**Use Cases:**

- **Namespaced**: Mix static lvars with dynamic actions for same model
- **Direct**: Single action returns complete model instance

**See Also:**

- [`Program`](#program): Contains list of Lact instances
- [`OutBlock`](#outblock): References actions by alias

---

### OutBlock

Output specification block (`OUT{field: value, ...}`).

**Signature:**

```python
@dataclass(slots=True)
class OutBlock(Stmt):
    """Output specification block.

    Syntax: OUT{field: value, field2: [ref1, ref2]}

    Values can be:
        - Literal: 0.85, "text", true
        - Single reference: [alias]
        - Multiple references: [alias1, alias2]

    Example:
        OUT{title: [t], summary: [s], confidence: 0.85}
        → OutBlock(fields={"title": ["t"], "summary": ["s"], "confidence": 0.85})
    """

    fields: dict[str, list[str] | str | int | float | bool]
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `fields` | `dict[str, list[str] \| str \| int \| float \| bool]` | Field mappings (name → references or literals) |

**Field Value Types:**

| Type | Example | Description |
|------|---------|-------------|
| `list[str]` | `["t", "s"]` | Variable/action references (array syntax `[...]` or single variable) |
| `str` | `"active"` | String literal |
| `int` | `42` | Integer literal |
| `float` | `0.85` | Float literal |
| `bool` | `True` | Boolean literal |

**LNDL Syntax:**

```text
OUT{
  field1: [ref1, ref2],    # Multiple references
  field2: [ref],           # Single reference
  field3: "literal",       # String literal
  field4: 42,              # Integer literal
  field5: 0.85,            # Float literal
  field6: true             # Boolean literal
}
```

**Examples:**

```python
from lionherd_core.lndl.ast import OutBlock

# Mixed references and literals
out = OutBlock(fields={
    "title": ["t"],               # Single reference (wrapped in list)
    "summary": ["s1", "s2"],      # Multiple references
    "confidence": 0.85,           # Float literal
    "status": "complete",         # String literal
    "count": 100                  # Integer literal
})

# Access fields
print(out.fields["title"])       # ['t']
print(out.fields["confidence"])  # 0.85

# Iterate fields
for field_name, field_value in out.fields.items():
    if isinstance(field_value, list):
        print(f"{field_name}: refs={field_value}")
    else:
        print(f"{field_name}: literal={field_value}")
# title: refs=['t']
# summary: refs=['s1', 's2']
# confidence: literal=0.85
# status: literal=complete
# count: literal=100
```

**Use Cases:**

- Mapping LNDL variables/actions to output fields
- Specifying literal values for output fields
- Defining output structure for validation against Operable specs

**See Also:**

- [`Program`](#program): Contains optional OutBlock instance
- [`Resolver`](resolver.md): Validates OutBlock against Operable specs

---

### Program

Root AST node containing all LNDL declarations.

**Signature:**

```python
@dataclass(slots=True)
class Program:
    """Root AST node containing all declarations.

    A complete LNDL program consists of:
        - Variable declarations (lvars + rlvars)
        - Action declarations (lacts)
        - Output specification (out_block)

    Example:
        <lvar Report.title t>Title</lvar>
        <lvar reasoning>Analysis text</lvar>
        <lact Report.summary s>summarize()</lact>
        OUT{title: [t], summary: [s], reasoning: [reasoning]}

        → Program(
            lvars=[Lvar(...), RLvar(...)],
            lacts=[Lact(...)],
            out_block=OutBlock(...)
        )
    """

    lvars: list[Lvar | RLvar]  # Both namespaced and raw lvars
    lacts: list[Lact]
    out_block: OutBlock | None
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `lvars` | `list[Lvar \| RLvar]` | Variable declarations (namespaced + raw) |
| `lacts` | `list[Lact]` | Action declarations |
| `out_block` | `OutBlock \| None` | Output specification (optional) |

**Examples:**

```python
from lionherd_core.lndl.ast import Program, Lvar, RLvar, Lact, OutBlock

# Complete program
program = Program(
    lvars=[
        Lvar(model="Report", field="title", alias="t", content="Title"),
        RLvar(alias="reasoning", content="Analysis text")
    ],
    lacts=[
        Lact(model="Report", field="summary", alias="s", call="summarize()")
    ],
    out_block=OutBlock(fields={
        "title": ["t"],
        "summary": ["s"],
        "reasoning": ["reasoning"]
    })
)

# Access components
print(f"Variables: {len(program.lvars)}")       # Variables: 2
print(f"Actions: {len(program.lacts)}")         # Actions: 1
print(f"Output fields: {list(program.out_block.fields.keys())}")
# Output fields: ['title', 'summary', 'reasoning']

# Iterate lvars
for lvar in program.lvars:
    if isinstance(lvar, Lvar):
        print(f"  Namespaced: {lvar.model}.{lvar.field} = {lvar.content}")
    else:  # RLvar
        print(f"  Raw: {lvar.alias} = {lvar.content}")
# Namespaced: Report.title = Title
# Raw: reasoning = Analysis text
```

**Use Cases:**

- Root node of parsed LNDL AST
- Input to Resolver for validation and model construction
- Intermediate representation for AST transformations

**See Also:**

- [`Parser.parse()`](parser.md#parse): Returns Program instance
- [`Resolver`](resolver.md): Consumes Program to build validated models

---

## Usage Patterns

### Building AST Manually

```python
from lionherd_core.lndl.ast import Program, Lvar, OutBlock

# Construct AST manually
program = Program(
    lvars=[
        Lvar(
            model="User",
            field="name",
            alias="n",
            content="Alice"
        ),
        Lvar(
            model="User",
            field="age",
            alias="a",
            content="30"
        )
    ],
    lacts=[],
    out_block=OutBlock(fields={
        "user": ["n", "a"]
    })
)

# Validate structure
assert len(program.lvars) == 2
assert program.out_block is not None
assert "user" in program.out_block.fields
```

### AST Traversal

```python
from lionherd_core.lndl.ast import Program, Lvar, RLvar, Lact

# Parse LNDL (assume program is parsed AST)
# program = parser.parse()

# Traverse lvars
namespaced_lvars = [lv for lv in program.lvars if isinstance(lv, Lvar)]
raw_lvars = [lv for lv in program.lvars if isinstance(lv, RLvar)]

print(f"Namespaced: {len(namespaced_lvars)}, Raw: {len(raw_lvars)}")

# Collect all aliases
all_aliases = {lv.alias for lv in program.lvars} | {la.alias for la in program.lacts}
print(f"Total aliases: {len(all_aliases)}")

# Find actions targeting specific model
report_actions = [
    la for la in program.lacts
    if la.model == "Report"
]
print(f"Report actions: {len(report_actions)}")
```

### Type Checking

```python
from lionherd_core.lndl.ast import ASTNode, Expr, Stmt, Literal, Lvar

# Type hierarchy checks
lit = Literal(value=42)
assert isinstance(lit, Expr)
assert isinstance(lit, ASTNode)
assert not isinstance(lit, Stmt)

lvar = Lvar(model="Report", field="title", alias="t", content="Title")
assert isinstance(lvar, Stmt)
assert isinstance(lvar, ASTNode)
assert not isinstance(lvar, Expr)
```

### Pattern Matching (Python 3.10+)

```python
from lionherd_core.lndl.ast import Lvar, RLvar, Lact

def process_declaration(decl):
    """Process declaration using pattern matching."""
    match decl:
        case Lvar(model=m, field=f, alias=a, content=c):
            return f"Namespaced: {m}.{f} ({a}) = {c}"
        case RLvar(alias=a, content=c):
            return f"Raw: {a} = {c}"
        case Lact(model=m, field=f, alias=a, call=c) if m is not None:
            return f"Namespaced action: {m}.{f} ({a})"
        case Lact(alias=a, call=c):
            return f"Direct action: {a}"
        case _:
            return "Unknown declaration"

# Usage
lvar = Lvar(model="Report", field="title", alias="t", content="Title")
print(process_declaration(lvar))
# Namespaced: Report.title (t) = Title
```

---

## Design Rationale

### Why Pure Data (No Methods)?

**Reason**: Separation of concerns - **data** (AST structure) vs **operations** (traversal, validation, transformation).

**Benefits**:

- **Simple**: Easy to understand, serialize, and debug
- **Flexible**: Operations implemented externally (visitors, transformers)
- **Testable**: Pure data structures have trivial equality checks
- **Serializable**: Can be converted to JSON/dict without method serialization

**Alternative (Rejected)**: Methods like `lvar.validate()` or `program.resolve()` would mix concerns and reduce flexibility.

---

### Why Slots?

**Reason**: Memory efficiency via `__slots__` - **50% memory reduction** for large ASTs.

**Benefits**:

- Reduced memory footprint (no `__dict__` per instance)
- Faster attribute access (direct offset lookup)
- Prevents accidental attribute assignment (typo protection)

**Trade-off**: Slightly more verbose (must declare all attributes), but worth it for production use.

---

### Why Separate Lvar and RLvar?

**Reason**: **Type safety** - Lvar has model/field metadata, RLvar doesn't.

**Benefits**:

- **Explicit Intent**: Namespaced vs raw variables have different validation rules
- **Type Checking**: Resolver can validate Lvar against Pydantic model schema
- **Error Messages**: Clear distinction in error messages ("Lvar 'Report.title' not found" vs "RLvar 'reasoning' not found")

**Alternative (Rejected)**: Single `Lvar` with optional model/field would lose type safety and require runtime checks.

---

### Why OutBlock Uses dict, Not Structured Fields?

**Reason**: **Flexibility** - OUT{} fields are dynamic, schema unknown at parse time.

**Benefits**:

- Handles arbitrary field names
- Supports both references (`list[str]`) and literals (scalar types)
- No predefined schema required

**Alternative (Rejected)**: Structured fields (e.g., `@dataclass` with fixed attributes) would require schema definition before parsing.

---

## Common Pitfalls

### Pitfall 1: Mutating AST Nodes

**Issue**: Dataclasses are mutable by default (unless frozen).

```python
lvar = Lvar(model="Report", field="title", alias="t", content="Old")
lvar.content = "New"  # ⚠️ Mutates original
```

**Solution**: Treat AST as immutable. Create new nodes instead:

```python
# Correct: Create new node
new_lvar = Lvar(
    model=lvar.model,
    field=lvar.field,
    alias=lvar.alias,
    content="New"
)
```

---

### Pitfall 2: Assuming OutBlock Always Exists

**Issue**: `Program.out_block` is optional (can be `None`).

```python
program = Program(lvars=[], lacts=[], out_block=None)
fields = program.out_block.fields  # ❌ AttributeError: 'NoneType'
```

**Solution**: Check for `None`:

```python
if program.out_block is not None:
    fields = program.out_block.fields
else:
    fields = {}
```

---

### Pitfall 3: Mixing Lvar and RLvar Incorrectly

**Issue**: Treating RLvar as if it has model/field attributes.

```python
from lionherd_core.lndl.ast import RLvar

rlvar = RLvar(alias="reasoning", content="Text")
print(rlvar.model)  # ❌ AttributeError: 'RLvar' has no attribute 'model'
```

**Solution**: Check type before accessing attributes:

```python
if isinstance(rlvar, Lvar):
    print(f"{rlvar.model}.{rlvar.field}")
else:  # RLvar
    print(f"Raw variable: {rlvar.alias}")
```

---

## See Also

**Related Classes:**

- [`Lexer`](lexer.md): Tokenizes LNDL text
- [`Parser`](parser.md): Parses tokens into AST
- [`Resolver`](resolver.md): Validates AST and constructs models

**Related Modules:**

- [LNDL Lexer](lexer.md): Text → Tokens
- [LNDL Parser](parser.md): Tokens → AST
- [LNDL Resolver](resolver.md): AST → Validated models
- [LNDL Architecture Guide](../../tutorials/lndl_architecture.md): Complete workflow

**External References:**

- [Abstract Syntax Tree (Wikipedia)](https://en.wikipedia.org/wiki/Abstract_syntax_tree)
- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Slots in Python](https://docs.python.org/3/reference/datamodel.html#slots)

---

## Examples

### Example 1: Parsing and AST Inspection

```python
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser

response = """
<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.score s>0.95</lvar>
<lact Report.summary sum>generate_summary()</lact>

OUT{
  title: [t],
  score: [s],
  summary: [sum]
}
"""

# Parse to AST
lexer = Lexer(response)
tokens = lexer.tokenize()
parser = Parser(tokens, source_text=response)
program = parser.parse()

# Inspect AST
print(f"Variables: {len(program.lvars)}")
print(f"Actions: {len(program.lacts)}")
print(f"Output fields: {list(program.out_block.fields.keys())}")

# Iterate lvars
for lvar in program.lvars:
    print(f"  {lvar.model}.{lvar.field} ({lvar.alias}): {lvar.content}")
# Report.title (t): AI Safety Analysis
# Report.score (s): 0.95
```

### Example 2: AST Transformation

```python
from lionherd_core.lndl.ast import Program, Lvar, OutBlock

# Original AST
original = Program(
    lvars=[
        Lvar(model="User", field="name", alias="n", content="Alice")
    ],
    lacts=[],
    out_block=OutBlock(fields={"user": ["n"]})
)

# Transform: Add new lvar
transformed = Program(
    lvars=original.lvars + [
        Lvar(model="User", field="age", alias="a", content="30")
    ],
    lacts=original.lacts,
    out_block=OutBlock(fields={
        "user": ["n", "a"]  # Updated references
    })
)

print(f"Original lvars: {len(original.lvars)}")  # 1
print(f"Transformed lvars: {len(transformed.lvars)}")  # 2
```

### Example 3: AST Validation

```python
from lionherd_core.lndl.ast import Program, OutBlock

def validate_program(program: Program) -> list[str]:
    """Validate AST structure, return list of errors."""
    errors = []

    # Check: All OUT{} references must have corresponding lvars/lacts
    if program.out_block:
        all_aliases = {lv.alias for lv in program.lvars} | {la.alias for la in program.lacts}

        for field_name, field_value in program.out_block.fields.items():
            if isinstance(field_value, list):
                for ref in field_value:
                    if ref not in all_aliases:
                        errors.append(f"Field '{field_name}' references undefined alias '{ref}'")

    # Check: No duplicate aliases
    aliases_seen = set()
    for lvar in program.lvars:
        if lvar.alias in aliases_seen:
            errors.append(f"Duplicate alias: '{lvar.alias}'")
        aliases_seen.add(lvar.alias)

    for lact in program.lacts:
        if lact.alias in aliases_seen:
            errors.append(f"Duplicate alias: '{lact.alias}'")
        aliases_seen.add(lact.alias)

    return errors

# Valid program
valid_program = Program(
    lvars=[Lvar(model="User", field="name", alias="n", content="Alice")],
    lacts=[],
    out_block=OutBlock(fields={"user": ["n"]})
)
print(validate_program(valid_program))  # []

# Invalid program (undefined reference)
invalid_program = Program(
    lvars=[],
    lacts=[],
    out_block=OutBlock(fields={"user": ["undefined"]})
)
print(validate_program(invalid_program))
# ["Field 'user' references undefined alias 'undefined'"]
```

### Example 4: AST Serialization

```python
from dataclasses import asdict
from lionherd_core.lndl.ast import Program, Lvar, OutBlock

# Create AST
program = Program(
    lvars=[
        Lvar(model="Report", field="title", alias="t", content="Title")
    ],
    lacts=[],
    out_block=OutBlock(fields={"title": ["t"]})
)

# Serialize to dict
program_dict = asdict(program)
print(program_dict)
# {
#   'lvars': [
#     {'model': 'Report', 'field': 'title', 'alias': 't', 'content': 'Title'}
#   ],
#   'lacts': [],
#   'out_block': {'fields': {'title': ['t']}}
# }

# Can be serialized to JSON
import json
json_str = json.dumps(program_dict, indent=2)
print(json_str)
```

### Example 5: Custom AST Visitor

```python
from lionherd_core.lndl.ast import Program, Lvar, RLvar, Lact, OutBlock

class ASTVisitor:
    """Visitor pattern for AST traversal."""

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
        print(f"Visiting Lvar: {lvar.model}.{lvar.field}")

    def visit_rlvar(self, rlvar: RLvar):
        print(f"Visiting RLvar: {rlvar.alias}")

    def visit_lact(self, lact: Lact):
        if lact.model:
            print(f"Visiting Lact: {lact.model}.{lact.field}")
        else:
            print(f"Visiting Lact: {lact.alias}")

    def visit_out_block(self, out_block: OutBlock):
        print(f"Visiting OutBlock: {len(out_block.fields)} fields")

# Usage
program = Program(
    lvars=[Lvar(model="Report", field="title", alias="t", content="Title")],
    lacts=[],
    out_block=OutBlock(fields={"title": ["t"]})
)

visitor = ASTVisitor()
visitor.visit_program(program)
# Visiting Lvar: Report.title
# Visiting OutBlock: 1 fields
```

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL AST Nodes - Structured Output Only (Simplified).

This module defines the Abstract Syntax Tree for LNDL structured outputs.
Semantic operations and cognitive programming constructs are deferred for future phases.

AST Design Philosophy:
- Pure data (dataclasses, no methods)
- Type-safe (full annotations)
- Simple and clear (no over-engineering)

Node Hierarchy:
- ASTNode (base)
  - Expr (expressions)
    - Literal: Scalar values (int, float, str, bool)
    - Identifier: Variable references
  - Stmt (statements)
    - Lvar: Variable declarations
    - Lact: Action/function declarations
    - OutBlock: Output specification
  - Program: Root node (list of statements)
"""

from dataclasses import dataclass


# Base Nodes
class ASTNode:
    """Base AST node for all LNDL constructs."""

    __slots__ = ()  # Empty slots for proper inheritance


# Expressions (evaluate to values)
class Expr(ASTNode):
    """Base expression node."""

    __slots__ = ()


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


@dataclass(slots=True)
class Identifier(Expr):
    """Variable reference.

    Examples:
        - [title]
        - [summary]
    """

    name: str


# Statements (declarations, no return value)
class Stmt(ASTNode):
    """Base statement node."""

    __slots__ = ()


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


__all__ = (
    "ASTNode",
    "Expr",
    "Identifier",
    "Lact",
    "Literal",
    "Lvar",
    "OutBlock",
    "Program",
    "RLvar",
    "Stmt",
)

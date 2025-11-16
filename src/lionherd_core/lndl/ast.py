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

    pass


# Expressions (evaluate to values)
class Expr(ASTNode):
    """Base expression node."""

    pass


@dataclass
class Literal(Expr):
    """Literal scalar value.

    Examples:
        - "AI safety"
        - 42
        - 0.85
        - true
    """

    value: str | int | float | bool


@dataclass
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

    pass


@dataclass
class Lvar(Stmt):
    """Variable declaration.

    Syntax:
        - Namespaced: <lvar Model.field alias>content</lvar>
        - Legacy: <lvar alias>content</lvar>

    Examples:
        <lvar Report.title t>AI Safety Analysis</lvar>
        → Lvar(model="Report", field="title", alias="t", content="AI Safety Analysis")

        <lvar t>Simple content</lvar>
        → Lvar(model=None, field=None, alias="t", content="Simple content")
    """

    model: str | None  # Model name (e.g., "Report") or None for legacy
    field: str | None  # Field name (e.g., "title") or None for legacy
    alias: str  # Local variable name (e.g., "t")
    content: str  # Raw string value


@dataclass
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


@dataclass
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


@dataclass
class Program:
    """Root AST node containing all declarations.

    A complete LNDL program consists of:
        - Variable declarations (lvars)
        - Action declarations (lacts)
        - Output specification (out_block)

    Example:
        <lvar Report.title t>Title</lvar>
        <lact Report.summary s>summarize()</lact>
        OUT{title: [t], summary: [s]}

        → Program(
            lvars=[Lvar(...)],
            lacts=[Lact(...)],
            out_block=OutBlock(...)
        )
    """

    lvars: list[Lvar]
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
    "Stmt",
)

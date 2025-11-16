# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for LNDL AST nodes.

Tests all 10 AST node types with 100% coverage:
- Base classes: ASTNode, Expr, Stmt
- Expressions: Literal, Identifier
- Statements: Lvar, RLvar, Lact, OutBlock
- Program: Program

Coverage targets:
- All node types and their fields
- Dataclass features (equality, repr, slots)
- Inheritance hierarchy
- Edge cases (empty strings, zero, negatives, booleans, None, unicode)
"""

import pytest

from lionherd_core.lndl.ast import (
    ASTNode,
    Expr,
    Identifier,
    Lact,
    Literal,
    Lvar,
    OutBlock,
    Program,
    RLvar,
    Stmt,
)

# ============================================================================
# Base Classes
# ============================================================================


class TestASTNodeBase:
    """Test ASTNode base class."""

    def test_slots_defined(self):
        """Verify ASTNode has empty slots for proper inheritance."""
        assert ASTNode.__slots__ == ()

    def test_instantiation(self):
        """Verify ASTNode can be instantiated."""
        node = ASTNode()
        assert isinstance(node, ASTNode)


class TestExprBase:
    """Test Expr base class."""

    def test_inherits_astnode(self):
        """Verify Expr inherits from ASTNode."""
        expr = Expr()
        assert isinstance(expr, ASTNode)
        assert isinstance(expr, Expr)

    def test_slots_defined(self):
        """Verify Expr has empty slots."""
        assert Expr.__slots__ == ()


class TestStmtBase:
    """Test Stmt base class."""

    def test_inherits_astnode(self):
        """Verify Stmt inherits from ASTNode."""
        stmt = Stmt()
        assert isinstance(stmt, ASTNode)
        assert isinstance(stmt, Stmt)

    def test_slots_defined(self):
        """Verify Stmt has empty slots."""
        assert Stmt.__slots__ == ()


# ============================================================================
# Expression Nodes
# ============================================================================


class TestLiteralNode:
    """Test Literal scalar values (str, int, float, bool)."""

    @pytest.mark.parametrize(
        "value,expected_type",
        [
            ("AI safety", str),
            (42, int),
            (0.85, float),
            (True, bool),
            (False, bool),
            (-42, int),
            (-0.85, float),
            ("", str),
            ("line1\nline2", str),
            (0, int),
            (0.0, float),
        ],
    )
    def test_literal_scalar_types(self, value, expected_type):
        """Test Literal with various scalar types."""
        lit = Literal(value)
        assert lit.value == value
        assert isinstance(lit.value, expected_type)

    def test_literal_equality(self):
        """Test Literal dataclass equality."""
        lit1 = Literal(42)
        lit2 = Literal(42)
        lit3 = Literal(43)

        assert lit1 == lit2
        assert lit1 != lit3

    def test_literal_different_types_not_equal(self):
        """Test Literals with different types are not equal."""
        lit_int = Literal(42)
        lit_str = Literal("42")
        lit_float = Literal(3.14)

        # String vs numeric types should not be equal
        # Note: 42 == 42.0 in Python (int/float coercion), but 42 != "42"
        assert lit_int != lit_str
        assert lit_float != lit_str

    def test_literal_repr(self):
        """Test Literal repr includes value."""
        lit = Literal(42)
        repr_str = repr(lit)
        assert "Literal" in repr_str
        assert "42" in repr_str

    def test_literal_inheritance(self):
        """Test Literal inherits from Expr and ASTNode."""
        lit = Literal("test")
        assert isinstance(lit, Literal)
        assert isinstance(lit, Expr)
        assert isinstance(lit, ASTNode)

    # Edge cases
    def test_literal_very_large_int(self):
        """Test Literal with very large integer."""
        lit = Literal(999999999999999)
        assert lit.value == 999999999999999

    def test_literal_scientific_notation(self):
        """Test Literal with scientific notation float."""
        lit = Literal(1.5e-10)
        assert lit.value == 1.5e-10

    def test_literal_unicode_string(self):
        """Test Literal with unicode characters."""
        lit = Literal("Hello 世界 🌍")
        assert lit.value == "Hello 世界 🌍"

    def test_literal_empty_string(self):
        """Test Literal with empty string."""
        lit = Literal("")
        assert lit.value == ""
        assert isinstance(lit.value, str)

    def test_literal_multiline_string(self):
        """Test Literal with multiline string."""
        content = "line1\nline2\nline3"
        lit = Literal(content)
        assert lit.value == content


class TestIdentifierNode:
    """Test Identifier variable references."""

    @pytest.mark.parametrize(
        "name",
        [
            "title",
            "summary",
            "s",
            "var_with_underscore",
            "var123",
            "t",
            "very_long_variable_name_with_many_underscores",
            "_underscore_start",
        ],
    )
    def test_identifier_names(self, name):
        """Test Identifier with various names."""
        ident = Identifier(name)
        assert ident.name == name

    def test_identifier_equality(self):
        """Test Identifier dataclass equality."""
        ident1 = Identifier("x")
        ident2 = Identifier("x")
        ident3 = Identifier("y")

        assert ident1 == ident2
        assert ident1 != ident3

    def test_identifier_repr(self):
        """Test Identifier repr includes name."""
        ident = Identifier("variable_name")
        repr_str = repr(ident)
        assert "Identifier" in repr_str
        assert "variable_name" in repr_str

    def test_identifier_inheritance(self):
        """Test Identifier inherits from Expr and ASTNode."""
        ident = Identifier("test")
        assert isinstance(ident, Identifier)
        assert isinstance(ident, Expr)
        assert isinstance(ident, ASTNode)


# ============================================================================
# Statement Nodes
# ============================================================================


class TestLvarNode:
    """Test Lvar namespaced variable declaration."""

    def test_lvar_basic(self):
        """Test basic Lvar with all fields."""
        lvar = Lvar(model="Report", field="title", alias="t", content="Title")
        assert lvar.model == "Report"
        assert lvar.field == "title"
        assert lvar.alias == "t"
        assert lvar.content == "Title"

    def test_lvar_alias_equals_field(self):
        """Test Lvar where alias equals field name."""
        lvar = Lvar(model="Report", field="score", alias="score", content="0.95")
        assert lvar.alias == lvar.field
        assert lvar.content == "0.95"

    def test_lvar_complex_content(self):
        """Test Lvar with multiline content."""
        content = "Multi\nline\ncontent"
        lvar = Lvar(model="Report", field="content", alias="c", content=content)
        assert lvar.content == content

    def test_lvar_equality(self):
        """Test Lvar dataclass equality with all fields."""
        lvar1 = Lvar(model="M", field="f", alias="a", content="c")
        lvar2 = Lvar(model="M", field="f", alias="a", content="c")
        lvar3 = Lvar(model="M", field="f", alias="a", content="different")

        assert lvar1 == lvar2
        assert lvar1 != lvar3

    def test_lvar_repr(self):
        """Test Lvar repr includes all fields."""
        lvar = Lvar(model="Report", field="title", alias="t", content="Title")
        repr_str = repr(lvar)
        assert "Lvar" in repr_str
        assert "Report" in repr_str
        assert "title" in repr_str
        assert "t" in repr_str

    def test_lvar_inheritance(self):
        """Test Lvar inherits from Stmt and ASTNode."""
        lvar = Lvar(model="M", field="f", alias="a", content="c")
        assert isinstance(lvar, Lvar)
        assert isinstance(lvar, Stmt)
        assert isinstance(lvar, ASTNode)

    # Edge cases
    def test_lvar_empty_content(self):
        """Test Lvar with empty content."""
        lvar = Lvar(model="M", field="f", alias="a", content="")
        assert lvar.content == ""

    def test_lvar_numeric_content(self):
        """Test Lvar with numeric string content."""
        lvar = Lvar(model="M", field="f", alias="a", content="42")
        assert lvar.content == "42"
        assert isinstance(lvar.content, str)

    def test_lvar_special_chars(self):
        """Test Lvar with special characters in content."""
        content = "Content with \"quotes\" and 'apostrophes'"
        lvar = Lvar(model="M", field="f", alias="a", content=content)
        assert lvar.content == content


class TestRLvarNode:
    """Test RLvar raw variable declaration (no namespace)."""

    def test_rlvar_basic(self):
        """Test basic RLvar."""
        rlvar = RLvar(alias="reasoning", content="Analysis text")
        assert rlvar.alias == "reasoning"
        assert rlvar.content == "Analysis text"

    def test_rlvar_equality(self):
        """Test RLvar dataclass equality."""
        rlvar1 = RLvar(alias="a", content="c")
        rlvar2 = RLvar(alias="a", content="c")
        rlvar3 = RLvar(alias="b", content="c")

        assert rlvar1 == rlvar2
        assert rlvar1 != rlvar3

    def test_rlvar_repr(self):
        """Test RLvar repr includes fields."""
        rlvar = RLvar(alias="reasoning", content="text")
        repr_str = repr(rlvar)
        assert "RLvar" in repr_str
        assert "reasoning" in repr_str

    def test_rlvar_inheritance(self):
        """Test RLvar inherits from Stmt and ASTNode."""
        rlvar = RLvar(alias="a", content="c")
        assert isinstance(rlvar, RLvar)
        assert isinstance(rlvar, Stmt)
        assert isinstance(rlvar, ASTNode)

    # Edge cases
    def test_rlvar_empty_content(self):
        """Test RLvar with empty content."""
        rlvar = RLvar(alias="temp", content="")
        assert rlvar.content == ""

    def test_rlvar_numeric_string(self):
        """Test RLvar with numeric string content."""
        rlvar = RLvar(alias="score", content="123.45")
        assert rlvar.content == "123.45"

    def test_rlvar_multiline(self):
        """Test RLvar with multiline content."""
        content = "Line1\nLine2\nLine3"
        rlvar = RLvar(alias="text", content=content)
        assert rlvar.content == content


class TestLactNode:
    """Test Lact action declaration."""

    def test_lact_namespaced(self):
        """Test namespaced Lact (model and field)."""
        lact = Lact(
            model="Report",
            field="summary",
            alias="s",
            call="summarize(text='...')",
        )
        assert lact.model == "Report"
        assert lact.field == "summary"
        assert lact.alias == "s"
        assert lact.call == "summarize(text='...')"

    def test_lact_direct(self):
        """Test direct Lact (no model/field)."""
        lact = Lact(model=None, field=None, alias="search", call="search(query='AI')")
        assert lact.model is None
        assert lact.field is None
        assert lact.alias == "search"
        assert lact.call == "search(query='AI')"

    def test_lact_complex_call(self):
        """Test Lact with complex function call."""
        lact = Lact(
            model="Analysis",
            field="findings",
            alias="f",
            call="analyze(depth=3, threshold=0.8)",
        )
        assert lact.call == "analyze(depth=3, threshold=0.8)"

    def test_lact_equality(self):
        """Test Lact dataclass equality."""
        lact1 = Lact(model="M", field="f", alias="a", call="func()")
        lact2 = Lact(model="M", field="f", alias="a", call="func()")
        lact3 = Lact(model="M", field="f", alias="a", call="other()")

        assert lact1 == lact2
        assert lact1 != lact3

    def test_lact_repr(self):
        """Test Lact repr includes fields."""
        lact = Lact(model="M", field="f", alias="a", call="func()")
        repr_str = repr(lact)
        assert "Lact" in repr_str
        assert "func()" in repr_str

    def test_lact_inheritance(self):
        """Test Lact inherits from Stmt and ASTNode."""
        lact = Lact(model="M", field="f", alias="a", call="func()")
        assert isinstance(lact, Lact)
        assert isinstance(lact, Stmt)
        assert isinstance(lact, ASTNode)

    # Edge cases
    def test_lact_empty_call(self):
        """Test Lact with empty function call."""
        lact = Lact(model=None, field=None, alias="func", call="func()")
        assert lact.call == "func()"

    def test_lact_nested_parens(self):
        """Test Lact with nested parentheses."""
        lact = Lact(model=None, field=None, alias="f", call="func(nested(value))")
        assert lact.call == "func(nested(value))"

    def test_lact_none_fields_equality(self):
        """Test Lact equality with None fields."""
        lact1 = Lact(model=None, field=None, alias="a", call="c()")
        lact2 = Lact(model=None, field=None, alias="a", call="c()")
        assert lact1 == lact2


class TestOutBlockNode:
    """Test OutBlock output specification block."""

    def test_outblock_single_ref(self):
        """Test OutBlock with single reference."""
        out = OutBlock(fields={"title": ["t"]})
        assert out.fields == {"title": ["t"]}

    def test_outblock_multiple_refs(self):
        """Test OutBlock with multiple references."""
        out = OutBlock(fields={"summary": ["s1", "s2"]})
        assert out.fields == {"summary": ["s1", "s2"]}

    def test_outblock_literal_str(self):
        """Test OutBlock with literal string."""
        out = OutBlock(fields={"type": "analysis"})
        assert out.fields == {"type": "analysis"}

    def test_outblock_literal_int(self):
        """Test OutBlock with literal int."""
        out = OutBlock(fields={"count": 42})
        assert out.fields == {"count": 42}

    def test_outblock_literal_float(self):
        """Test OutBlock with literal float."""
        out = OutBlock(fields={"confidence": 0.85})
        assert out.fields == {"confidence": 0.85}

    def test_outblock_literal_bool(self):
        """Test OutBlock with literal bool."""
        out_true = OutBlock(fields={"valid": True})
        out_false = OutBlock(fields={"valid": False})
        assert out_true.fields == {"valid": True}
        assert out_false.fields == {"valid": False}

    def test_outblock_mixed_types(self):
        """Test OutBlock with mixed value types."""
        out = OutBlock(fields={"title": ["t"], "confidence": 0.95, "type": "report"})
        assert out.fields["title"] == ["t"]
        assert out.fields["confidence"] == 0.95
        assert out.fields["type"] == "report"

    def test_outblock_equality(self):
        """Test OutBlock dataclass equality."""
        out1 = OutBlock(fields={"x": ["a"], "y": 1})
        out2 = OutBlock(fields={"x": ["a"], "y": 1})
        out3 = OutBlock(fields={"x": ["b"], "y": 1})

        assert out1 == out2
        assert out1 != out3

    def test_outblock_repr(self):
        """Test OutBlock repr includes fields."""
        out = OutBlock(fields={"title": ["t"]})
        repr_str = repr(out)
        assert "OutBlock" in repr_str
        assert "title" in repr_str

    def test_outblock_inheritance(self):
        """Test OutBlock inherits from Stmt and ASTNode."""
        out = OutBlock(fields={})
        assert isinstance(out, OutBlock)
        assert isinstance(out, Stmt)
        assert isinstance(out, ASTNode)

    # Edge cases
    def test_outblock_empty_fields(self):
        """Test OutBlock with empty fields dict."""
        out = OutBlock(fields={})
        assert out.fields == {}

    def test_outblock_single_empty_ref(self):
        """Test OutBlock with empty reference list."""
        out = OutBlock(fields={"field": []})
        assert out.fields == {"field": []}

    def test_outblock_negative_number(self):
        """Test OutBlock with negative number."""
        out = OutBlock(fields={"score": -0.5})
        assert out.fields == {"score": -0.5}


# ============================================================================
# Program Node
# ============================================================================


class TestProgramNode:
    """Test Program root AST node combining all declarations."""

    def test_program_empty(self):
        """Test empty Program."""
        prog = Program(lvars=[], lacts=[], out_block=None)
        assert prog.lvars == []
        assert prog.lacts == []
        assert prog.out_block is None

    def test_program_only_lvars(self):
        """Test Program with only lvars."""
        lvar = Lvar(model="M", field="f", alias="a", content="c")
        rlvar = RLvar(alias="r", content="rc")
        prog = Program(lvars=[lvar, rlvar], lacts=[], out_block=None)
        assert len(prog.lvars) == 2
        assert prog.lacts == []
        assert prog.out_block is None

    def test_program_only_lacts(self):
        """Test Program with only lacts."""
        lact = Lact(model="M", field="f", alias="a", call="call()")
        prog = Program(lvars=[], lacts=[lact], out_block=None)
        assert prog.lvars == []
        assert len(prog.lacts) == 1
        assert prog.out_block is None

    def test_program_only_outblock(self):
        """Test Program with only outblock."""
        out = OutBlock(fields={"x": ["a"]})
        prog = Program(lvars=[], lacts=[], out_block=out)
        assert prog.lvars == []
        assert prog.lacts == []
        assert prog.out_block == out

    def test_program_complete(self):
        """Test complete Program with all components."""
        lvar = Lvar(model="M", field="f", alias="a", content="c")
        rlvar = RLvar(alias="r", content="rc")
        lact = Lact(model="M", field="f2", alias="act", call="call()")
        out = OutBlock(fields={"a": ["a"], "r": ["r"], "act": ["act"]})

        prog = Program(lvars=[lvar, rlvar], lacts=[lact], out_block=out)
        assert len(prog.lvars) == 2
        assert len(prog.lacts) == 1
        assert prog.out_block == out

    def test_program_equality(self):
        """Test Program dataclass equality."""
        lvar = Lvar(model="M", field="f", alias="a", content="c")
        prog1 = Program(lvars=[lvar], lacts=[], out_block=None)
        prog2 = Program(lvars=[lvar], lacts=[], out_block=None)

        assert prog1 == prog2

    def test_program_repr(self):
        """Test Program repr includes components."""
        lvar = Lvar(model="M", field="f", alias="a", content="c")
        prog = Program(lvars=[lvar], lacts=[], out_block=None)
        repr_str = repr(prog)
        assert "Program" in repr_str

    # Edge cases
    def test_program_many_lvars(self):
        """Test Program with many lvars."""
        lvars = [Lvar(model="M", field=f"f{i}", alias=f"a{i}", content=f"c{i}") for i in range(10)]
        prog = Program(lvars=lvars, lacts=[], out_block=None)
        assert len(prog.lvars) == 10

    def test_program_many_lacts(self):
        """Test Program with many lacts."""
        lacts = [Lact(model="M", field=f"f{i}", alias=f"a{i}", call=f"call{i}()") for i in range(5)]
        prog = Program(lvars=[], lacts=lacts, out_block=None)
        assert len(prog.lacts) == 5

    def test_program_mixed_lvars_rlvars(self):
        """Test Program with mixed Lvar and RLvar types."""
        lvars_mixed = [
            Lvar(model="M", field="f1", alias="a1", content="c1"),
            RLvar(alias="r1", content="rc1"),
            Lvar(model="M", field="f2", alias="a2", content="c2"),
            RLvar(alias="r2", content="rc2"),
        ]
        prog = Program(lvars=lvars_mixed, lacts=[], out_block=None)
        assert len(prog.lvars) == 4
        assert isinstance(prog.lvars[0], Lvar)
        assert isinstance(prog.lvars[1], RLvar)


# ============================================================================
# Type Validation & Inheritance
# ============================================================================


class TestInheritanceChain:
    """Verify inheritance relationships across all nodes."""

    def test_literal_inheritance_chain(self):
        """Test Literal inheritance chain."""
        lit = Literal(42)
        assert isinstance(lit, Literal)
        assert isinstance(lit, Expr)
        assert isinstance(lit, ASTNode)

    def test_identifier_inheritance_chain(self):
        """Test Identifier inheritance chain."""
        ident = Identifier("x")
        assert isinstance(ident, Identifier)
        assert isinstance(ident, Expr)
        assert isinstance(ident, ASTNode)

    def test_lvar_inheritance_chain(self):
        """Test Lvar inheritance chain."""
        lvar = Lvar(model="M", field="f", alias="a", content="c")
        assert isinstance(lvar, Lvar)
        assert isinstance(lvar, Stmt)
        assert isinstance(lvar, ASTNode)

    def test_rlvar_inheritance_chain(self):
        """Test RLvar inheritance chain."""
        rlvar = RLvar(alias="a", content="c")
        assert isinstance(rlvar, RLvar)
        assert isinstance(rlvar, Stmt)
        assert isinstance(rlvar, ASTNode)

    def test_lact_inheritance_chain(self):
        """Test Lact inheritance chain."""
        lact = Lact(model="M", field="f", alias="a", call="call()")
        assert isinstance(lact, Lact)
        assert isinstance(lact, Stmt)
        assert isinstance(lact, ASTNode)

    def test_outblock_inheritance_chain(self):
        """Test OutBlock inheritance chain."""
        out = OutBlock(fields={})
        assert isinstance(out, OutBlock)
        assert isinstance(out, Stmt)
        assert isinstance(out, ASTNode)

    def test_all_expr_subclasses(self):
        """Test all Expr subclasses are expressions."""
        lit = Literal(1)
        ident = Identifier("x")

        assert isinstance(lit, Expr)
        assert isinstance(ident, Expr)

    def test_all_stmt_subclasses(self):
        """Test all Stmt subclasses are statements."""
        lvar = Lvar(model="M", field="f", alias="a", content="c")
        rlvar = RLvar(alias="a", content="c")
        lact = Lact(model="M", field="f", alias="a", call="c()")
        out = OutBlock(fields={})

        assert isinstance(lvar, Stmt)
        assert isinstance(rlvar, Stmt)
        assert isinstance(lact, Stmt)
        assert isinstance(out, Stmt)


class TestDataclassFeatures:
    """Test dataclass-specific features across all nodes."""

    def test_dataclass_slots(self):
        """Test that dataclass nodes use slots."""
        # All dataclass nodes should have __slots__ defined
        assert hasattr(Literal, "__slots__")
        assert hasattr(Identifier, "__slots__")
        assert hasattr(Lvar, "__slots__")
        assert hasattr(RLvar, "__slots__")
        assert hasattr(Lact, "__slots__")
        assert hasattr(OutBlock, "__slots__")
        assert hasattr(Program, "__slots__")

    def test_dataclass_equality_symmetry(self):
        """Test equality is symmetric."""
        lit1 = Literal(42)
        lit2 = Literal(42)
        assert lit1 == lit2
        assert lit2 == lit1

    def test_dataclass_equality_transitivity(self):
        """Test equality is transitive."""
        lit1 = Literal(42)
        lit2 = Literal(42)
        lit3 = Literal(42)
        assert lit1 == lit2
        assert lit2 == lit3
        assert lit1 == lit3

    def test_dataclass_inequality(self):
        """Test inequality across different node types."""
        lit = Literal(42)
        ident = Identifier("42")
        # Different types should not be equal
        assert lit != ident

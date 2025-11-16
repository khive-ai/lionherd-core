"""Unit tests targeting resolver.py lines 81-129 (scalar Spec with array syntax).

Tests scalar Specs (Spec(str, ...), Spec(int, ...), etc.) with array syntax.
This is different from BaseModel fields - these are standalone scalar values.
"""

import pytest
from pydantic import BaseModel

from lionherd_core.lndl.fuzzy import parse_lndl_fuzzy
from lionherd_core.lndl.resolver import resolve_references_prefixed
from lionherd_core.lndl.types import LactMetadata, LvarMetadata
from lionherd_core.types import Operable, Spec


class TestScalarSpecWithArraySyntax:
    """Test scalar Spec with array syntax: OUT{field:[var]}"""

    def test_scalar_string_from_single_lvar(self):
        """Test Spec(str) with single-element array"""

        lvars = {"s": LvarMetadata(model="status", field="value", local_name="s", value="active")}
        lacts = {}
        out_fields = {"status": ["s"]}  # Array syntax for scalar Spec
        operable = Operable([Spec(str, name="status")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        assert result.fields["status"] == "active"

    def test_scalar_int_from_single_lvar(self):
        """Test Spec(int) with single-element array"""

        lvars = {"p": LvarMetadata(model="priority", field="value", local_name="p", value="3")}
        lacts = {}
        out_fields = {"priority": ["p"]}
        operable = Operable([Spec(int, name="priority")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        assert result.fields["priority"] == 3

    def test_scalar_float_from_single_lvar(self):
        """Test Spec(float) with single-element array"""

        lvars = {"q": LvarMetadata(model="quality", field="score", local_name="q", value="0.92")}
        lacts = {}
        out_fields = {"quality_score": ["q"]}
        operable = Operable([Spec(float, name="quality_score")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        assert result.fields["quality_score"] == 0.92

    def test_scalar_from_multiple_lvars_error(self):
        """Test ERROR when scalar Spec gets multiple variables (line 81-84)"""

        lvars = {
            "s1": LvarMetadata(model="status", field="value", local_name="s1", value="active"),
            "s2": LvarMetadata(model="status", field="value", local_name="s2", value="pending"),
        }
        lacts = {}
        out_fields = {"status": ["s1", "s2"]}  # Multiple vars for scalar Spec
        operable = Operable([Spec(str, name="status")])

        # ExceptionGroup wraps the ValueError
        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Check the wrapped exception
        assert len(exc_info.value.exceptions) == 1
        assert "Scalar field" in str(exc_info.value.exceptions[0])
        assert "cannot use multiple variables" in str(exc_info.value.exceptions[0])

    def test_scalar_undefined_variable_error(self):
        """Test ERROR when variable not found (line 119-122)"""

        lvars = {}
        lacts = {}
        out_fields = {"status": ["undefined"]}
        operable = Operable([Spec(str, name="status")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

        assert len(exc_info.value.exceptions) == 1
        assert "Variable or action" in str(exc_info.value.exceptions[0])
        assert "not declared" in str(exc_info.value.exceptions[0])


class TestScalarSpecWithAction:
    """Test scalar Spec with action reference (lines 87-116)"""

    def test_scalar_from_lact_simple(self):
        """Test scalar Spec with action call"""

        lvars = {}
        lacts = {
            "gen": LactMetadata(model="status", field="value", local_name="gen", call="generate()")
        }
        out_fields = {"status": ["gen"]}
        operable = Operable([Spec(str, name="status")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)

        # Should create ActionCall
        assert "gen" in result.actions
        assert result.actions["gen"].function == "generate"
        assert result.fields["status"] == result.actions["gen"]  # Field contains ActionCall

    def test_scalar_from_lact_with_args(self):
        """Test action with arguments"""

        lvars = {}
        lacts = {
            "fetch": LactMetadata(
                model="data", field="value", local_name="fetch", call='get(url="test.com")'
            )
        }
        out_fields = {"data": ["fetch"]}
        operable = Operable([Spec(str, name="data")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)

        action = result.actions["fetch"]
        assert action.function == "get"
        assert "url" in action.arguments
        assert action.arguments["url"] == "test.com"

    def test_scalar_from_lact_invalid_syntax_error(self):
        """Test ERROR on invalid function call syntax (line 93-100)"""

        lvars = {}
        lacts = {
            "broken": LactMetadata(
                model="result", field="value", local_name="broken", call="invalid[["
            )
        }
        out_fields = {"result": ["broken"]}
        operable = Operable([Spec(str, name="result")])

        with pytest.raises(ExceptionGroup) as exc_info:
            resolve_references_prefixed(out_fields, lvars, lacts, operable)

        assert len(exc_info.value.exceptions) == 1
        assert "Invalid function call syntax" in str(exc_info.value.exceptions[0])


class TestPreparsedValues:
    """Test already-parsed values from new parser (lines 126-129)"""

    def test_lvar_with_preparsed_int(self):
        """Test lvar.value already parsed as int"""

        lvars = {
            "p": LvarMetadata(model="priority", field="value", local_name="p", value=3)
        }  # Int not str
        lacts = {}
        out_fields = {"priority": ["p"]}
        operable = Operable([Spec(int, name="priority")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        assert result.fields["priority"] == 3

    def test_lvar_with_preparsed_float(self):
        """Test lvar.value already parsed as float"""

        lvars = {"q": LvarMetadata(model="quality", field="score", local_name="q", value=0.92)}
        lacts = {}
        out_fields = {"quality_score": ["q"]}
        operable = Operable([Spec(float, name="quality_score")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        assert result.fields["quality_score"] == 0.92

    def test_lvar_with_string_needs_parsing(self):
        """Test lvar.value as string gets parsed (line 127)"""

        lvars = {
            "p": LvarMetadata(model="priority", field="value", local_name="p", value="42")
        }  # String
        lacts = {}
        out_fields = {"priority": ["p"]}
        operable = Operable([Spec(int, name="priority")])

        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        assert result.fields["priority"] == 42
        assert isinstance(result.fields["priority"], int)

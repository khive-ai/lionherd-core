# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for LNDL tests.

This module provides common test fixtures for testing LNDL components:
- Sample Pydantic models (Report, Analysis, SearchResult)
- Sample LNDL text snippets (valid, invalid, edge cases)
- Operable instances with different Specs
- Lexer/Parser/AST helper instances
"""

import pytest
from pydantic import BaseModel

from lionherd_core.lndl import Lexer, Parser
from lionherd_core.lndl.ast import Lvar
from lionherd_core.lndl.types import LvarMetadata, RLvarMetadata
from lionherd_core.testing import get_sample_lndl_text, get_sample_pydantic_models
from lionherd_core.types import Operable, Spec

# ============================================================================
# Sample Pydantic Models
# ============================================================================


@pytest.fixture
def report_model():
    """Simple report model for basic testing."""
    return get_sample_pydantic_models()["Report"]


@pytest.fixture
def analysis_model():
    """Analysis model with multiple field types."""
    return get_sample_pydantic_models()["Analysis"]


@pytest.fixture
def search_result_model():
    """Search result model for action testing."""
    return get_sample_pydantic_models()["SearchResult"]


# ============================================================================
# Sample LNDL Text Snippets
# ============================================================================


@pytest.fixture
def simple_lndl_text():
    """Simple LNDL with one lvar and OUT block."""
    return get_sample_lndl_text("simple")


@pytest.fixture
def multi_lvar_lndl_text():
    """LNDL with multiple lvars and OUT block."""
    return get_sample_lndl_text("multi")


@pytest.fixture
def lact_lndl_text():
    """LNDL with action call."""
    return get_sample_lndl_text("lact")


@pytest.fixture
def raw_lvar_lndl_text():
    """LNDL with raw lvar (no namespace)."""
    return get_sample_lndl_text("raw")


@pytest.fixture
def mixed_lndl_text():
    """LNDL with mixed lvars, rlvars, lacts, and literals."""
    return get_sample_lndl_text("mixed")


@pytest.fixture
def invalid_lndl_text():
    """LNDL with syntax errors."""
    return get_sample_lndl_text("invalid")


@pytest.fixture
def empty_lndl_text():
    """Empty LNDL response."""
    return get_sample_lndl_text("empty")


@pytest.fixture
def lndl_with_code_block():
    """LNDL wrapped in markdown code block."""
    return """\
```lndl
<lvar Report.title t>Title</lvar>
OUT{title: [t]}
```
"""


# ============================================================================
# Operable Instances
# ============================================================================


@pytest.fixture
def report_operable(report_model):
    """Operable with Report spec."""
    return Operable([Spec(report_model, name="report")])


@pytest.fixture
def analysis_operable(analysis_model):
    """Operable with Analysis spec."""
    return Operable([Spec(analysis_model, name="analysis")])


@pytest.fixture
def multi_spec_operable(report_model, analysis_model):
    """Operable with multiple specs."""
    return Operable(
        [
            Spec(report_model, name="report"),
            Spec(analysis_model, name="analysis"),
        ]
    )


# ============================================================================
# Helper Functions
# ============================================================================


@pytest.fixture
def create_lexer():
    """Factory fixture for creating Lexer instances."""

    def _create_lexer(text: str) -> Lexer:
        return Lexer(text)

    return _create_lexer


@pytest.fixture
def create_parser():
    """Factory fixture for creating Parser instances."""

    def _create_parser(tokens, source_text=None) -> Parser:
        return Parser(tokens, source_text)

    return _create_parser


@pytest.fixture
def tokenize():
    """Helper to tokenize LNDL text."""

    def _tokenize(text: str):
        lexer = Lexer(text)
        return lexer.tokenize()

    return _tokenize


@pytest.fixture
def parse_lndl_ast():
    """Helper to parse LNDL text to AST."""

    def _parse(text: str):
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, text)
        return parser.parse()

    return _parse


@pytest.fixture
def extract_lvars_prefixed():
    """Helper to extract lvars using new Lexer/Parser architecture."""

    def _extract(text: str) -> dict[str, LvarMetadata | RLvarMetadata]:
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        program = parser.parse()

        lvars = {}
        for lvar in program.lvars:
            if isinstance(lvar, Lvar):
                # Namespaced lvar
                lvars[lvar.alias] = LvarMetadata(
                    model=lvar.model,
                    field=lvar.field,
                    local_name=lvar.alias,
                    value=lvar.content,
                )
            else:  # RLvar
                # Raw lvar
                lvars[lvar.alias] = RLvarMetadata(
                    local_name=lvar.alias,
                    value=lvar.content,
                )

        return lvars

    return _extract


@pytest.fixture
def parse_out_block_array():
    """Helper to parse OUT{} block using new Parser."""

    def _parse(content: str) -> dict[str, list[str] | str | int | float | bool]:
        text = content.strip()
        if not text.startswith("OUT{"):
            text = f"OUT{{{text}}}"

        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        program = parser.parse()

        return program.out_block.fields if program.out_block else {}

    return _parse

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
from lionherd_core.types import Operable, Spec

# ============================================================================
# Sample Pydantic Models
# ============================================================================


@pytest.fixture
def report_model():
    """Simple report model for basic testing."""

    class Report(BaseModel):
        title: str
        content: str
        score: float = 0.0

    return Report


@pytest.fixture
def analysis_model():
    """Analysis model with multiple field types."""

    class Analysis(BaseModel):
        summary: str
        findings: list[str]
        confidence: float
        metadata: dict[str, str] | None = None

    return Analysis


@pytest.fixture
def search_result_model():
    """Search result model for action testing."""

    class SearchResult(BaseModel):
        query: str
        results: list[str]
        count: int

    return SearchResult


# ============================================================================
# Sample LNDL Text Snippets
# ============================================================================


@pytest.fixture
def simple_lndl_text():
    """Simple LNDL with one lvar and OUT block."""
    return """\
<lvar Report.title t>AI Safety Analysis</lvar>

OUT{title: [t]}
"""


@pytest.fixture
def multi_lvar_lndl_text():
    """LNDL with multiple lvars and OUT block."""
    return """\
<lvar Report.title t>AI Safety</lvar>
<lvar Report.content c>Analysis of AI safety measures.</lvar>
<lvar Report.score s>0.95</lvar>

OUT{title: [t], content: [c], score: [s]}
"""


@pytest.fixture
def lact_lndl_text():
    """LNDL with action call."""
    return """\
<lact SearchResult.results r>search(query="AI safety")</lact>

OUT{results: [r]}
"""


@pytest.fixture
def raw_lvar_lndl_text():
    """LNDL with raw lvar (no namespace)."""
    return """\
<lvar reasoning>This is intermediate reasoning text.</lvar>

OUT{reasoning: [reasoning]}
"""


@pytest.fixture
def mixed_lndl_text():
    """LNDL with mixed lvars, rlvars, lacts, and literals."""
    return """\
<lvar Report.title t>Title</lvar>
<lvar reasoning>Reasoning text here</lvar>
<lact Analysis.summary s>summarize(text="...")</lact>

OUT{title: [t], reasoning: [reasoning], summary: [s], confidence: 0.85}
"""


@pytest.fixture
def invalid_lndl_text():
    """LNDL with syntax errors."""
    return """\
<lvar Report.title>Missing closing tag
OUT{title: [t]
"""


@pytest.fixture
def empty_lndl_text():
    """Empty LNDL response."""
    return ""


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

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and test models for LNDL tests."""

import pytest
from pydantic import BaseModel, Field, field_validator

from lionherd_core.lndl import parse_lndl
from lionherd_core.lndl.ast import Lact, Lvar, RLvar
from lionherd_core.lndl.lexer import Lexer
from lionherd_core.lndl.parser import Parser
from lionherd_core.lndl.types import LactMetadata, LvarMetadata, RLvarMetadata
from lionherd_core.types import Operable, Spec

# ============================================================================
# Common Test Models
# ============================================================================


class Report(BaseModel):
    """Basic report model with title and summary."""

    title: str
    summary: str


class Reason(BaseModel):
    """Reasoning model with confidence score and analysis."""

    confidence: float
    analysis: str


class SearchResults(BaseModel):
    """Search results with items and count."""

    items: list[str]
    count: int


class Analysis(BaseModel):
    """Analysis model with findings, recommendations, and risk level."""

    findings: str
    recommendations: str
    risk_level: str


class StrictReport(BaseModel):
    """Report with field length constraints."""

    title: str = Field(min_length=3)
    summary: str = Field(min_length=3)


class ValidatedReport(BaseModel):
    """Report with custom field validators."""

    title: str
    summary: str

    @field_validator("title", "summary")
    @classmethod
    def check_min_length(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError("must be at least 3 characters")
        return v


class ScoredReport(BaseModel):
    """Report with confidence score."""

    title: str
    summary: str
    confidence: float


class NestedReport(BaseModel):
    """Report with nested analysis."""

    title: str
    analysis: Analysis


# ============================================================================
# Test Utilities
# ============================================================================


def parse_lvars(text: str) -> dict:
    """Parse LNDL text and extract lvars as dict.

    Helper function for testing lvar extraction.
    Returns dict mapping alias to lvar metadata dict.
    """
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    # Convert to dict format matching old API
    result = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            # Namespaced lvar
            result[lvar.alias] = {
                "model": lvar.model,
                "field": lvar.field,
                "local_name": lvar.alias,
                "value": lvar.content,
            }
        else:  # RLvar
            # Raw lvar - no model/field
            result[lvar.alias] = {
                "model": None,
                "field": None,
                "local_name": lvar.alias,
                "value": lvar.content,
            }
    return result


def parse_lacts(text: str) -> dict:
    """Parse LNDL text and extract lacts as dict.

    Helper function for testing lact extraction.
    Returns dict mapping alias to lact metadata dict.
    """
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    # Convert to dict format
    result = {}
    for lact in program.lacts:
        result[lact.alias] = {
            "model": lact.model,
            "field": lact.field,
            "local_name": lact.alias,
            "call": lact.call,
        }
    return result


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def simple_operable():
    """Operable with single Report spec."""
    return Operable([Spec(Report, name="report")])


@pytest.fixture
def scored_operable():
    """Operable with ScoredReport spec."""
    return Operable([Spec(ScoredReport, name="report")])


@pytest.fixture
def multi_model_operable():
    """Operable with multiple specs."""
    return Operable(
        [
            Spec(Report, name="report"),
            Spec(Reason, name="reasoning"),
            Spec(float, name="quality_score"),
        ]
    )


@pytest.fixture
def strict_operable():
    """Operable with StrictReport (field constraints)."""
    return Operable([Spec(StrictReport, name="report")])


@pytest.fixture
def validated_operable():
    """Operable with ValidatedReport (custom validators)."""
    return Operable([Spec(ValidatedReport, name="report")])

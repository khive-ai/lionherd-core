# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL - Language Network Directive Language.

LNDL is a structured output format for LLM responses with fuzzy parsing tolerance.

Security Model:
    - **Threat model**: Designed for LLM output (structured but imperfect)
    - **NOT for**: Untrusted user input without validation
    - **ActionCalls**: Parsed but NOT executed (caller must sandbox)
    - **Fuzzy mode**: Trades strictness for tolerance (use strict mode for security-critical)

Usage:
    >>> from lionherd_core.lndl import parse_lndl_fuzzy
    >>> from lionherd_core.types import Operable
    >>> result = parse_lndl_fuzzy(llm_response, operable)

Thread Safety:
    Lexer and Parser are not thread-safe. Create separate instances per thread.
"""

from .ast import Identifier, Lact, Literal, Lvar, OutBlock, Program, RLvar
from .errors import (
    AmbiguousMatchError,
    InvalidConstructorError,
    LNDLError,
    MissingFieldError,
    MissingLvarError,
    MissingOutBlockError,
    TypeMismatchError,
)
from .fuzzy import parse_lndl_fuzzy
from .lexer import Lexer, Token, TokenType
from .parser import ParseError, Parser
from .prompt import LNDL_SYSTEM_PROMPT, get_lndl_system_prompt
from .resolver import parse_lndl, resolve_references_prefixed
from .types import (
    ActionCall,
    LactMetadata,
    LNDLOutput,
    LvarMetadata,
    ParsedConstructor,
    RLvarMetadata,
    Scalar,
    ensure_no_action_calls,
    has_action_calls,
    revalidate_with_action_results,
)

__all__ = (
    "LNDL_SYSTEM_PROMPT",
    "ActionCall",
    "AmbiguousMatchError",
    "Identifier",
    "InvalidConstructorError",
    "LNDLError",
    "LNDLOutput",
    "Lact",
    "LactMetadata",
    "Lexer",
    "Literal",
    "Lvar",
    "LvarMetadata",
    "MissingFieldError",
    "MissingLvarError",
    "MissingOutBlockError",
    "OutBlock",
    "ParseError",
    "ParsedConstructor",
    "Parser",
    "Program",
    "RLvar",
    "RLvarMetadata",
    "Scalar",
    "Token",
    "TokenType",
    "TypeMismatchError",
    "ensure_no_action_calls",
    "get_lndl_system_prompt",
    "has_action_calls",
    "parse_lndl",
    "parse_lndl_fuzzy",
    "resolve_references_prefixed",
    "revalidate_with_action_results",
)

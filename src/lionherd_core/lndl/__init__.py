# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

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
from .parser import (
    extract_lacts,
    extract_lacts_prefixed,
    extract_lvars,
    extract_lvars_prefixed,
    extract_out_block,
    parse_out_block_array,
)
from .prompt import LNDL_SYSTEM_PROMPT, get_lndl_system_prompt
from .resolver import parse_lndl, resolve_references_prefixed
from .types import (
    ActionCall,
    LactMetadata,
    LNDLOutput,
    LvarMetadata,
    ParsedConstructor,
    has_action_calls,
    revalidate_with_action_results,
)

__all__ = (
    "LNDL_SYSTEM_PROMPT",
    "ActionCall",
    "AmbiguousMatchError",
    "InvalidConstructorError",
    "LNDLError",
    "LNDLOutput",
    "LactMetadata",
    "LvarMetadata",
    "MissingFieldError",
    "MissingLvarError",
    "MissingOutBlockError",
    "ParsedConstructor",
    "TypeMismatchError",
    "extract_lacts",
    "extract_lacts_prefixed",
    "extract_lvars",  # backward compatibility
    "extract_lvars_prefixed",
    "extract_out_block",
    "get_lndl_system_prompt",
    "has_action_calls",
    "parse_lndl",
    "parse_lndl_fuzzy",
    "parse_out_block_array",
    "resolve_references_prefixed",
    "revalidate_with_action_results",
)

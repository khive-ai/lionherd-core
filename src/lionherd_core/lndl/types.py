# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core types for LNDL (Lion Directive Language)."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(slots=True, frozen=True)
class LvarMetadata:
    """Metadata for namespace-prefixed lvar.

    Example: <lvar Report.title title>Good Title</lvar>
    → LvarMetadata(model="Report", field="title", local_name="title", value="Good Title")
    """

    model: str  # Model name (e.g., "Report")
    field: str  # Field name (e.g., "title")
    local_name: str  # Local variable name (e.g., "title")
    value: str  # Raw string value


@dataclass(slots=True, frozen=True)
class ParsedConstructor:
    """Parsed type constructor from OUT{} block."""

    class_name: str
    kwargs: dict[str, Any]
    raw: str

    @property
    def has_dict_unpack(self) -> bool:
        """Check if constructor uses **dict unpacking."""
        return any(k.startswith("**") for k in self.kwargs)


@dataclass(slots=True, frozen=True)
class LNDLOutput:
    """Validated LNDL output."""

    fields: dict[str, BaseModel]
    lvars: dict[str, str] | dict[str, LvarMetadata]  # Preserved for debugging
    raw_out_block: str  # Preserved for debugging

    def __getitem__(self, key: str) -> BaseModel:
        return self.fields[key]

    def __getattr__(self, key: str) -> BaseModel:
        if key in ("fields", "lvars", "raw_out_block"):
            return object.__getattribute__(self, key)
        return self.fields[key]

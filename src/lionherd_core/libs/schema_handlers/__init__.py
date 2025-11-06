# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ._function_call_parser import (
    map_positional_args,
    nest_arguments_by_schema,
    parse_function_call,
)
from ._minimal_yaml import minimal_yaml
from ._typescript import typescript_schema

if TYPE_CHECKING:
    from ._schema_to_model import load_pydantic_model_from_schema


__all__ = (
    "load_pydantic_model_from_schema",
    "map_positional_args",
    "minimal_yaml",
    "nest_arguments_by_schema",
    "parse_function_call",
    "typescript_schema",
)


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "load_pydantic_model_from_schema":
        from ._schema_to_model import load_pydantic_model_from_schema

        return load_pydantic_model_from_schema
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

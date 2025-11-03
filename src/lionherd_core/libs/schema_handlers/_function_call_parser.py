# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import ast
from typing import Any, get_args, get_origin

from pydantic import BaseModel


def parse_function_call(call_str: str) -> dict[str, Any]:
    """Parse Python function call syntax into JSON tool invocation format."""
    try:
        # Parse the call as a Python expression
        tree = ast.parse(call_str, mode="eval")
        call = tree.body

        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # Extract function name
        if isinstance(call.func, ast.Name):
            tool_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            # Handle chained calls like client.search()
            tool_name = call.func.attr
        else:
            raise ValueError(f"Unsupported function type: {type(call.func)}")

        # Extract arguments
        arguments = {}

        # Positional arguments (will be mapped by parameter order in schema)
        for i, arg in enumerate(call.args):
            # For now, use position-based keys; will be mapped to param names later
            arguments[f"_pos_{i}"] = ast.literal_eval(arg)

        # Keyword arguments
        for keyword in call.keywords:
            if keyword.arg is None:
                # **kwargs syntax
                raise ValueError("**kwargs not supported")
            arguments[keyword.arg] = ast.literal_eval(keyword.value)

        return {"tool": tool_name, "arguments": arguments}

    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid function call syntax: {e}")


def map_positional_args(arguments: dict[str, Any], param_names: list[str]) -> dict[str, Any]:
    """Map positional arguments (_pos_0, _pos_1, ...) to actual parameter names."""
    mapped = {}
    pos_count = 0

    for key, value in arguments.items():
        if key.startswith("_pos_"):
            if pos_count >= len(param_names):
                raise ValueError(f"Too many positional arguments (expected {len(param_names)})")
            mapped[param_names[pos_count]] = value
            pos_count += 1
        else:
            # Keep keyword arguments as-is
            mapped[key] = value

    return mapped


def nest_arguments_by_schema(arguments: dict[str, Any], schema_cls) -> dict[str, Any]:
    """Restructure flat arguments into nested format based on schema structure."""
    if not schema_cls or not hasattr(schema_cls, "model_fields"):
        return arguments

    # Get top-level field names
    top_level_fields = set(schema_cls.model_fields.keys())

    # Find fields that are nested objects (Pydantic models or unions)
    nested_field_mappings = {}
    for field_name, field_info in schema_cls.model_fields.items():
        annotation = field_info.annotation

        # Check if it's a union type
        if get_origin(annotation) is type(int | str):  # UnionType check
            union_members = get_args(annotation)
            # Collect all fields from union members
            union_fields = set()
            for member in union_members:
                if hasattr(member, "model_fields"):
                    union_fields.update(member.model_fields.keys())
            if union_fields:
                nested_field_mappings[field_name] = union_fields
        # Check if it's a Pydantic model
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            nested_field_mappings[field_name] = set(annotation.model_fields.keys())

    # If no nested fields detected, return as-is
    if not nested_field_mappings:
        return arguments

    # Separate top-level args from nested args
    result = {}
    nested_args = {}

    for key, value in arguments.items():
        if key in top_level_fields:
            # This is a top-level field
            result[key] = value
        else:
            # Check if this belongs to a nested field
            for nested_field, nested_keys in nested_field_mappings.items():
                if key in nested_keys:
                    if nested_field not in nested_args:
                        nested_args[nested_field] = {}
                    nested_args[nested_field][key] = value
                    break
            else:
                # Unknown field - keep at top level (will fail validation later)
                result[key] = value

    # Add nested structures to result
    result.update(nested_args)

    return result

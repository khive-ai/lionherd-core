# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import contextlib
import re
from typing import Any

import orjson


def fuzzy_json(str_to_parse: str, /) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse JSON string with fuzzy error correction (quotes, spacing, brackets).

    Returns:
        Either a dict or a list of dicts. Will NOT return primitive types
        (int, float, str, bool, None) or lists of primitives.

    Raises:
        TypeError: If parsed JSON is a primitive type or list of primitives.
        ValueError: If JSON parsing fails after all correction attempts.
    """
    _check_valid_str(str_to_parse)

    # 1. Direct attempt
    with contextlib.suppress(orjson.JSONDecodeError):
        result = orjson.loads(str_to_parse)
        return _validate_return_type(result)

    # 2. Try cleaning: replace single quotes with double and normalize
    cleaned = _clean_json_string(str_to_parse.replace("'", '"'))
    with contextlib.suppress(orjson.JSONDecodeError):
        result = orjson.loads(cleaned)
        return _validate_return_type(result)

    # 3. Try fixing brackets
    fixed = fix_json_string(cleaned)
    with contextlib.suppress(orjson.JSONDecodeError):
        result = orjson.loads(fixed)
        return _validate_return_type(result)

    # If all attempts fail
    raise ValueError("Invalid JSON string")


def _check_valid_str(str_to_parse: str, /) -> None:
    if not isinstance(str_to_parse, str):
        raise TypeError("Input must be a string")
    if not str_to_parse.strip():
        raise ValueError("Input string is empty")


def _validate_return_type(result: Any) -> dict[str, Any] | list[dict[str, Any]]:
    """Validate that parsed JSON matches the declared return type.

    Args:
        result: The result from orjson.loads()

    Returns:
        The validated result (dict or list[dict])

    Raises:
        TypeError: If result is a primitive type or list of non-dict elements
    """
    # Check if it's a dict
    if isinstance(result, dict):
        return result

    # Check if it's a list
    if isinstance(result, list):
        # Ensure all elements are dicts
        if not result:
            # Empty list is valid as list[dict] (vacuous truth)
            return result

        for i, item in enumerate(result):
            if not isinstance(item, dict):
                raise TypeError(
                    f"fuzzy_json returns dict or list[dict], got list with "
                    f"non-dict element at index {i}: {type(item).__name__}"
                )
        return result

    # If we got here, it's a primitive type (int, float, str, bool, None)
    raise TypeError(
        f"fuzzy_json returns dict or list[dict], got primitive type: {type(result).__name__}"
    )


def _clean_json_string(s: str) -> str:
    """Basic normalization: replace unescaped single quotes, trim spaces, ensure keys are quoted."""
    # Replace unescaped single quotes with double quotes
    # '(?<!\\)'" means a single quote not preceded by a backslash
    s = re.sub(r"(?<!\\)'", '"', s)
    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s)
    # Remove trailing commas before closing brackets/braces
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Ensure keys are quoted
    # This attempts to find patterns like { key: value } and turn them into {"key": value}
    s = re.sub(r'([{,])\s*([^"\s]+)\s*:', r'\1"\2":', s)
    return s.strip()


def fix_json_string(str_to_parse: str, /) -> str:
    """Fix JSON string by balancing unmatched brackets."""
    if not str_to_parse:
        raise ValueError("Input string is empty")

    brackets = {"{": "}", "[": "]"}
    open_brackets = []
    pos = 0
    length = len(str_to_parse)

    while pos < length:
        char = str_to_parse[pos]

        if char == "\\":
            pos += 2  # Skip escaped chars
            continue

        if char == '"':
            pos += 1
            # skip string content
            while pos < length:
                if str_to_parse[pos] == "\\":
                    pos += 2
                    continue
                if str_to_parse[pos] == '"':
                    pos += 1
                    break
                pos += 1
            continue

        if char in brackets:
            open_brackets.append(brackets[char])
        elif char in brackets.values():
            if not open_brackets:
                # Extra closing bracket
                # Better to raise error than guess
                raise ValueError("Extra closing bracket found.")
            if open_brackets[-1] != char:
                # Mismatched bracket
                raise ValueError("Mismatched brackets.")
            open_brackets.pop()

        pos += 1

    # Add missing closing brackets if any
    if open_brackets:
        str_to_parse += "".join(reversed(open_brackets))

    return str_to_parse

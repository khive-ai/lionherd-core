# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import contextlib
import re
from typing import Any

import orjson

from ._fuzzy_json import fuzzy_json

# Precompile the regex for extracting JSON code blocks
_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_json(
    input_data: str | list[str],
    /,
    *,
    fuzzy_parse: bool = False,
    return_one_if_single: bool = True,
) -> Any | list[Any]:
    """Extract and parse JSON content from a string or markdown code blocks.
    Attempts direct JSON parsing first. If that fails, looks for JSON content
    within markdown code blocks denoted by ```json.

    Args:
        input_data (str | list[str]): The input string or list of strings to parse.
        fuzzy_parse (bool): If True, attempts fuzzy JSON parsing on failed attempts.
        return_one_if_single (bool): If True and only one JSON object is found,
            returns a dict instead of a list with one dict.
    """

    # If input_data is a list, join into a single string
    input_str = "\n".join(input_data) if isinstance(input_data, list) else input_data

    # 1. Try direct parsing

    with contextlib.suppress(Exception):
        if fuzzy_parse:
            return fuzzy_json(input_str)
        return orjson.loads(input_str)

    # 2. Attempt extracting JSON blocks from markdown
    matches = _JSON_BLOCK_PATTERN.findall(input_str)
    if not matches:
        return []

    # If only one match, return single dict; if multiple, return list of dicts
    if return_one_if_single and len(matches) == 1:
        data_str = matches[0]
        with contextlib.suppress(Exception):
            if fuzzy_parse:
                return fuzzy_json(data_str)
            return orjson.loads(data_str)
        return []

    # Multiple matches
    results: list[Any] = []
    for m in matches:
        with contextlib.suppress(Exception):
            parsed = fuzzy_json(m) if fuzzy_parse else orjson.loads(m)
            # Append valid JSON (dicts, lists, or primitives)
            results.append(parsed)
    return results

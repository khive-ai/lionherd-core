# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import ast
import re
import warnings
from typing import Any

from .errors import MissingOutBlockError
from .types import LactMetadata, LvarMetadata

# Track warned action names to prevent duplicate warnings
_warned_action_names: set[str] = set()

# Python reserved keywords and common builtins
# Action names matching these will trigger warnings (not errors)
PYTHON_RESERVED = {
    # Keywords
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    # Common builtins that might cause confusion
    "print",
    "input",
    "open",
    "len",
    "range",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "type",
}


def extract_lvars(text: str) -> dict[str, str]:
    """Extract <lvar name>content</lvar> declarations (legacy format).

    Args:
        text: Response text containing lvar declarations

    Returns:
        Dict mapping lvar names to their content
    """
    pattern = r"<lvar\s+(\w+)>(.*?)</lvar>"
    matches = re.findall(pattern, text, re.DOTALL)

    lvars = {}
    for name, content in matches:
        # Strip whitespace but preserve internal structure
        lvars[name] = content.strip()

    return lvars


def extract_lvars_prefixed(text: str) -> dict[str, LvarMetadata]:
    """Extract namespace-prefixed lvar declarations.

    Args:
        text: Response text with <lvar Model.field alias>value</lvar> declarations

    Returns:
        Dict mapping local names to LvarMetadata
    """
    # Pattern: <lvar Model.field optional_local_name>value</lvar>
    # Groups: (1) model, (2) field, (3) optional local_name, (4) value
    pattern = r"<lvar\s+(\w+)\.(\w+)(?:\s+(\w+))?\s*>(.*?)</lvar>"
    matches = re.findall(pattern, text, re.DOTALL)

    lvars = {}
    for model, field, local_name, value in matches:
        # If no local_name provided, use field name
        local = local_name if local_name else field

        lvars[local] = LvarMetadata(model=model, field=field, local_name=local, value=value.strip())

    return lvars


def extract_lacts(text: str) -> dict[str, str]:
    """Extract <lact name>function_call</lact> action declarations (legacy, non-namespaced).

    DEPRECATED: Use extract_lacts_prefixed() for namespace support.

    Actions represent tool/function invocations using pythonic syntax.
    They are only executed if referenced in the OUT{} block.

    Args:
        text: Response text containing <lact> declarations

    Returns:
        Dict mapping action names to Python function call strings

    Examples:
        >>> text = '<lact search>search(query="AI", limit=5)</lact>'
        >>> extract_lacts(text)
        {'search': 'search(query="AI", limit=5)'}
    """
    pattern = r"<lact\s+(\w+)>(.*?)</lact>"
    matches = re.findall(pattern, text, re.DOTALL)

    lacts = {}
    for name, call_str in matches:
        # Strip whitespace but preserve the function call structure
        lacts[name] = call_str.strip()

    return lacts


def extract_lacts_prefixed(text: str) -> dict[str, LactMetadata]:
    """Extract <lact> action declarations with optional namespace prefix.

    Supports two patterns:
        Namespaced: <lact Model.field alias>function_call()</lact>
        Direct: <lact name>function_call()</lact>

    Args:
        text: Response text containing <lact> declarations

    Returns:
        Dict mapping local names to LactMetadata

    Note:
        Performance: The regex pattern uses (.*?) with DOTALL for action body extraction.
        For very large responses (>100KB), parsing may be slow. Recommended maximum
        response size: 50KB. For larger responses, consider streaming parsers.

    Examples:
        >>> text = "<lact Report.summary s>generate_summary(...)</lact>"
        >>> extract_lacts_prefixed(text)
        {'s': LactMetadata(model="Report", field="summary", local_name="s", call="generate_summary(...)")}

        >>> text = '<lact search>search(query="AI")</lact>'
        >>> extract_lacts_prefixed(text)
        {'search': LactMetadata(model=None, field=None, local_name="search", call='search(query="AI")')}
    """
    # Pattern matches both forms with strict identifier validation:
    # <lact Model.field alias>call</lact>  OR  <lact name>call</lact>
    # Groups: (1) identifier (Model or name), (2) optional .field, (3) optional alias, (4) call
    # Rejects: multiple dots, leading/trailing dots, numeric prefixes
    # Note: \w* allows single-character identifiers (e.g., alias="t")
    pattern = r"<lact\s+([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?(?:\s+([A-Za-z_]\w*))?>(.*?)</lact>"
    matches = re.findall(pattern, text, re.DOTALL)

    lacts = {}
    for identifier, field, alias, call_str in matches:
        # Check if field group is present (namespaced pattern)
        if field:
            # Namespaced: <lact Model.field alias>
            model = identifier
            local_name = alias if alias else field  # Use alias or default to field name
        else:
            # Direct: <lact name>
            model = None
            field = None
            local_name = identifier  # identifier is the name

        # Warn if action name conflicts with Python reserved keywords (deduplicated)
        if local_name in PYTHON_RESERVED and local_name not in _warned_action_names:
            _warned_action_names.add(local_name)
            warnings.warn(
                f"Action name '{local_name}' is a Python reserved keyword or builtin. "
                f"While this works in LNDL (string keys), it may cause confusion.",
                UserWarning,
                stacklevel=2,
            )

        lacts[local_name] = LactMetadata(
            model=model,
            field=field,
            local_name=local_name,
            call=call_str.strip(),
        )

    return lacts


def extract_out_block(text: str) -> str:
    """Extract OUT{...} block content with balanced brace scanning.

    Args:
        text: Response text containing OUT{} block

    Returns:
        Content inside OUT{} block (without outer braces)

    Raises:
        MissingOutBlockError: If no OUT{} block found or unbalanced
    """
    # First try to extract from ```lndl code fence
    lndl_fence_pattern = r"```lndl\s*(.*?)```"
    lndl_match = re.search(lndl_fence_pattern, text, re.DOTALL | re.IGNORECASE)

    if lndl_match:
        # Extract from code fence content using balanced scanner
        fence_content = lndl_match.group(1)
        out_match = re.search(r"OUT\s*\{", fence_content, re.IGNORECASE)
        if out_match:
            return _extract_balanced_curly(fence_content, out_match.end() - 1).strip()

    # Fallback: try to find OUT{} anywhere in text
    out_match = re.search(r"OUT\s*\{", text, re.IGNORECASE)

    if not out_match:
        raise MissingOutBlockError("No OUT{} block found in response")

    return _extract_balanced_curly(text, out_match.end() - 1).strip()


def _extract_balanced_curly(text: str, open_idx: int) -> str:
    """Extract balanced curly brace content, ignoring braces in strings.

    Args:
        text: Full text containing the opening brace
        open_idx: Index of the opening '{'

    Returns:
        Content between balanced braces (without outer braces)

    Raises:
        MissingOutBlockError: If braces are unbalanced
    """
    depth = 1
    i = open_idx + 1
    in_str = False
    quote = ""
    esc = False

    while i < len(text):
        ch = text[i]

        if in_str:
            # Inside string: handle escapes and track quote end
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            # Outside string: track quotes and braces
            if ch in ('"', "'"):
                in_str = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Found matching closing brace
                    return text[open_idx + 1 : i]

        i += 1

    raise MissingOutBlockError("Unbalanced OUT{} block")


def parse_out_block_array(out_content: str) -> dict[str, list[str] | str]:
    """Parse OUT{} block with array syntax and literal values.

    Args:
        out_content: Content inside OUT{} block

    Returns:
        Dict mapping field names to lists of variable names or literal values
    """
    fields: dict[str, list[str] | str] = {}

    # Pattern: field_name:[var1, var2, ...] or field_name:value
    # Split by comma at top level (not inside brackets or quotes)
    i = 0
    while i < len(out_content):
        # Skip whitespace
        while i < len(out_content) and out_content[i].isspace():
            i += 1

        if i >= len(out_content):
            break

        # Extract field name
        field_start = i
        while i < len(out_content) and (out_content[i].isalnum() or out_content[i] == "_"):
            i += 1

        if i >= len(out_content):
            break

        field_name = out_content[field_start:i].strip()

        # Skip whitespace and colon
        while i < len(out_content) and out_content[i].isspace():
            i += 1

        if i >= len(out_content) or out_content[i] != ":":
            break

        i += 1  # Skip colon

        # Skip whitespace
        while i < len(out_content) and out_content[i].isspace():
            i += 1

        # Check if array syntax [var1, var2] or value
        if i < len(out_content) and out_content[i] == "[":
            # Array syntax
            i += 1  # Skip opening bracket
            bracket_start = i

            # Find matching closing bracket
            depth = 1
            while i < len(out_content) and depth > 0:
                if out_content[i] == "[":
                    depth += 1
                elif out_content[i] == "]":
                    depth -= 1
                i += 1

            # Extract variable names from inside brackets
            vars_str = out_content[bracket_start : i - 1].strip()
            var_names = [v.strip() for v in vars_str.split(",") if v.strip()]
            fields[field_name] = var_names

        else:
            # Single value (variable or literal)
            value_start = i

            # Handle quoted strings
            if i < len(out_content) and out_content[i] in ('"', "'"):
                quote = out_content[i]
                i += 1
                while i < len(out_content) and out_content[i] != quote:
                    if out_content[i] == "\\":
                        i += 2  # Skip escaped character
                    else:
                        i += 1
                if i < len(out_content):
                    i += 1  # Skip closing quote
            else:
                # Read until comma or newline
                while i < len(out_content) and out_content[i] not in ",\n":
                    i += 1

            value = out_content[value_start:i].strip()
            if value:
                # Detect if this is a literal scalar (number, boolean) or variable name
                # Heuristic: literals contain non-alphanumeric chars or are numbers/booleans
                is_likely_literal = (
                    value.startswith('"')
                    or value.startswith("'")
                    or value.replace(".", "", 1).replace("-", "", 1).isdigit()  # number
                    or value.lower() in ("true", "false", "null")  # boolean/null
                )

                if is_likely_literal:
                    # Literal value (scalar)
                    fields[field_name] = value
                else:
                    # Variable reference - wrap in list for consistency
                    fields[field_name] = [value]

        # Skip optional comma
        while i < len(out_content) and out_content[i].isspace():
            i += 1
        if i < len(out_content) and out_content[i] == ",":
            i += 1

    return fields


def parse_value(value_str: str) -> Any:
    """Parse string value to Python object (numbers, booleans, lists, dicts, strings).

    Args:
        value_str: String representation of value

    Returns:
        Parsed Python object
    """
    value_str = value_str.strip()

    # Handle lowercase boolean literals
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null":
        return None

    # Try literal_eval for numbers, lists, dicts
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # Return as string
        return value_str

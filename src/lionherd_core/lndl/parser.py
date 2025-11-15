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


def _parse_call_array(array_str: str) -> list[str]:
    """Parse array of function calls with balanced delimiter handling.

    Handles:
    - String literals with commas: func("hello, world")
    - Nested parentheses: func(a, nested(b, c))
    - Nested brackets: func([1, [2, 3]])
    - Mixed nesting: func({key: [1, 2]}, "test")

    Args:
        array_str: String like "[call1(), call2(), call3()]" or "call1(), call2()"

    Returns:
        List of individual call strings

    Example:
        >>> _parse_call_array('[find.by_name(query="ocean"), search(query="AI, ML")]')
        ['find.by_name(query="ocean")', 'search(query="AI, ML")']
    """
    # Strip array brackets if present
    content = array_str.strip()
    if content.startswith("[") and content.endswith("]"):
        content = content[1:-1].strip()

    calls = []
    current_call = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_string = False
    string_char = None

    for i, char in enumerate(content):
        # Track string literals
        if char in ('"', "'") and (i == 0 or content[i - 1] != "\\"):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None

        # Track nesting depth (only outside strings)
        if not in_string:
            if char == "(":
                depth_paren += 1
            elif char == ")":
                depth_paren -= 1
            elif char == "[":
                depth_bracket += 1
            elif char == "]":
                depth_bracket -= 1
            elif char == "{":
                depth_brace += 1
            elif char == "}":
                depth_brace -= 1
            elif char == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
                # Top-level comma - split here
                calls.append("".join(current_call).strip())
                current_call = []
                continue

        current_call.append(char)

    # Add last call
    if current_call:
        calls.append("".join(current_call).strip())

    return [c for c in calls if c]  # Filter empty strings


def extract_lacts_prefixed(text: str) -> dict[str, LactMetadata]:
    """Extract <lact> action declarations with optional namespace prefix and array support.

    Supports three patterns:
        1. Namespaced: <lact Model.field alias>function_call()</lact>
        2. Direct: <lact name>function_call()</lact>
        3. Array: <lact namespace a b c>[call1(), call2(), call3()]</lact>

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
        {'s': LactMetadata(model="Report", field="summary", local_names=["s"], calls=["generate_summary(...)"])}

        >>> text = '<lact search>search(query="AI")</lact>'
        >>> extract_lacts_prefixed(text)
        {'search': LactMetadata(model=None, field=None, local_names=["search"], calls=['search(query="AI")'])}

        >>> text = '<lact cognition a b c>[find.by_name(query="ocean"), recall.search(...), remember(...)]</lact>'
        >>> result = extract_lacts_prefixed(text)
        >>> result["a"].model
        'cognition'
        >>> len(result["a"].calls)
        3
    """
    # Pattern matches all forms:
    # <lact Model.field alias>call</lact>  OR  <lact name>call</lact>  OR  <lact namespace a b c>[...]</lact>
    # Groups: (1) identifier, (2) optional .field, (3) optional aliases, (4) call content (can be empty)
    pattern = r"<lact\s+([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?(?:\s+([A-Za-z_][\w\s]*))?>(.*?)</lact>"
    matches = re.findall(pattern, text, re.DOTALL)

    lacts = {}
    for identifier, field, aliases_str, call_content in matches:
        call_content = call_content.strip()

        # Parse aliases (space-separated)
        alias_list = [a.strip() for a in aliases_str.split() if a.strip()] if aliases_str else []

        # Detect array syntax
        is_array = call_content.startswith("[") and "]" in call_content

        if is_array:
            # Array pattern: <lact namespace a b c>[call1(), call2()]</lact>
            calls = _parse_call_array(call_content)

            # Validate: number of aliases must match number of calls
            if alias_list and len(alias_list) != len(calls):
                warnings.warn(
                    f"Alias count mismatch: {len(alias_list)} aliases for {len(calls)} calls. "
                    f"Aliases: {alias_list}, Calls: {len(calls)}",
                    UserWarning,
                    stacklevel=2,
                )

            # If no aliases provided, generate default ones
            if not alias_list:
                alias_list = [f"{identifier}_{i}" for i in range(len(calls))]

            # Namespace pattern (no .field)
            model = identifier
            field = None
            local_names = alias_list
        elif field:
            # Namespaced: <lact Model.field alias>call</lact>
            model = identifier
            local_names = [alias_list[0]] if alias_list else [field]
            calls = [call_content]
        else:
            # Direct: <lact name>call</lact>
            model = None
            field = None
            local_names = [alias_list[0]] if alias_list else [identifier]
            calls = [call_content]

        # Create metadata for each alias
        # CRITICAL: Each alias gets its own metadata with only its specific call
        # to support array syntax like <lact cognition a b>[find(), remember()]</lact>
        # where "a" should execute find() and "b" should execute remember()
        for i, local_name in enumerate(local_names):
            # Warn if action name conflicts with Python reserved keywords
            if local_name in PYTHON_RESERVED and local_name not in _warned_action_names:
                _warned_action_names.add(local_name)
                warnings.warn(
                    f"Action name '{local_name}' is a Python reserved keyword or builtin. "
                    f"While this works in LNDL (string keys), it may cause confusion.",
                    UserWarning,
                    stacklevel=2,
                )

            # Each alias gets its own metadata with only its specific call (at index i)
            # This ensures lacts["a"].call returns the first call, lacts["b"].call returns the second, etc.
            lacts[local_name] = LactMetadata(
                model=model,
                field=field,
                local_names=[local_name],  # Single alias for this metadata
                calls=[calls[i]],  # Corresponding call at same index
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

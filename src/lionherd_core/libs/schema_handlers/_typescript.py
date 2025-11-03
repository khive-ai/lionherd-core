# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0


def _type_map(json_type: str) -> str:
    """Map JSON Schema types to TypeScript-like types."""
    mapping = {
        "string": "string",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "array",  # Will be handled specially
        "object": "object",
        "null": "null",
    }
    return mapping.get(json_type, json_type)


def _format_enum_union(enum_values: list) -> str:
    """Format enum values as TypeScript union of literals."""
    formatted = []
    for val in enum_values:
        if isinstance(val, str):
            formatted.append(f'"{val}"')
        elif val is None:
            formatted.append("null")
        else:
            formatted.append(str(val))
    return " | ".join(formatted)


def _extract_type_signature(field_spec: dict, required: bool) -> tuple[str, bool]:
    """Extract TypeScript-style type signature from JSON Schema field."""
    # Handle enums first (most specific)
    if "enum" in field_spec:
        type_sig = _format_enum_union(field_spec["enum"])
        # Check if enum includes null
        has_null = None in field_spec["enum"]
        is_optional = not required or has_null
        return type_sig, is_optional

    # Handle anyOf (unions)
    if "anyOf" in field_spec:
        options = field_spec["anyOf"]
        type_parts = []
        has_null = False

        for opt in options:
            if opt.get("type") == "null":
                has_null = True
                continue

            if "type" in opt:
                if opt["type"] == "array" and "items" in opt:
                    item_type = _type_map(opt["items"].get("type", "any"))
                    if "$ref" in opt["items"]:
                        item_type = opt["items"]["$ref"].split("/")[-1]
                    type_parts.append(f"{item_type}[]")
                else:
                    type_parts.append(_type_map(opt["type"]))
            elif "$ref" in opt:
                ref_name = opt["$ref"].split("/")[-1]
                type_parts.append(ref_name)
            elif "enum" in opt:
                type_parts.append(_format_enum_union(opt["enum"]))

        if has_null:
            type_parts.append("null")

        type_sig = " | ".join(type_parts) if type_parts else "any"
        is_optional = not required or has_null
        return type_sig, is_optional

    # Handle arrays
    if "type" in field_spec and field_spec["type"] == "array":
        if "items" in field_spec:
            items_spec = field_spec["items"]
            if "type" in items_spec:
                item_type = _type_map(items_spec["type"])
            elif "$ref" in items_spec:
                item_type = items_spec["$ref"].split("/")[-1]
            elif "enum" in items_spec:
                item_type = f"({_format_enum_union(items_spec['enum'])})"
            else:
                item_type = "any"
        else:
            item_type = "any"
        type_sig = f"{item_type}[]"
        is_optional = not required
        return type_sig, is_optional

    # Handle $ref
    if "$ref" in field_spec:
        ref_name = field_spec["$ref"].split("/")[-1]
        is_optional = not required
        return ref_name, is_optional

    # Handle simple types
    if "type" in field_spec:
        base_type = field_spec["type"]
        # Handle nullable simple types (type? suffix from simplification)
        if isinstance(base_type, str) and base_type.endswith("?"):
            type_sig = _type_map(base_type[:-1])
            is_optional = True
        else:
            type_sig = _type_map(base_type)
            is_optional = not required
        return type_sig, is_optional

    # Fallback
    return "any", not required


def typescript_schema(schema: dict, indent: int = 0) -> str:
    """Convert JSON Schema to TypeScript-style notation for optimal LLM comprehension."""
    lines = []
    prefix = "  " * indent

    if "properties" not in schema:
        return ""

    required_fields = set(schema.get("required", []))

    for field_name, field_spec in schema["properties"].items():
        is_required = field_name in required_fields

        # Extract type signature
        type_sig, is_optional = _extract_type_signature(field_spec, is_required)

        # Add default value if present
        default_str = ""
        if "default" in field_spec:
            default_val = field_spec["default"]
            if isinstance(default_val, str):
                default_str = f' = "{default_val}"'
            elif default_val is None:
                default_str = " = null"
            elif isinstance(default_val, bool):
                default_str = f" = {'true' if default_val else 'false'}"
            else:
                default_str = f" = {default_val}"

        # Build field definition
        optional_marker = "?" if is_optional else ""
        field_def = f"{prefix}{field_name}{optional_marker}: {type_sig}{default_str}"

        # Add description if present
        if field_spec.get("description"):
            field_def += f" - {field_spec['description']}"

        lines.append(field_def)

    return "\n".join(lines)

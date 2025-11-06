# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import string
import sys
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, PydanticUserError

from lionherd_core import ln

B = TypeVar("B", bound=BaseModel)


def _get_python_version_enum(python_version_module: Any) -> Any:
    """Auto-detect Python version from environment."""
    version_info = sys.version_info
    version_map = {
        (3, 11): "PY_311",
        (3, 12): "PY_312",
        (3, 13): "PY_313",
        (3, 14): "PY_314",
    }
    version_key = (version_info.major, version_info.minor)
    enum_name = version_map.get(version_key, "PY_312")
    return getattr(python_version_module, enum_name, python_version_module.PY_312)


def _sanitize_model_name(name: str) -> str:
    """Extract valid Python identifier from string.

    Raises:
        ValueError: If name cannot be converted to valid Python identifier.
    """
    valid_chars = string.ascii_letters + string.digits + "_"
    sanitized = "".join(c for c in name.replace(" ", "") if c in valid_chars)

    if not sanitized or not sanitized[0].isalpha():
        msg = f"Cannot extract valid Python identifier from: {name!r}"
        raise ValueError(msg)

    return sanitized


def _extract_model_name_from_schema(schema_dict: dict[str, Any], default: str) -> str:
    """Extract model name from schema title or use default."""
    title = schema_dict.get("title")
    if title and isinstance(title, str):
        try:
            return _sanitize_model_name(title)
        except ValueError:
            pass  # Fall back to default if title cannot be sanitized
    return default


def _prepare_schema_input(
    schema: str | dict[str, Any],
    model_name: str,
) -> tuple[str, dict[str, Any], str]:
    """Convert schema to JSON string and extract model name."""
    if isinstance(schema, dict):
        try:
            schema_dict = schema
            schema_json = ln.json_dumps(schema_dict)
        except TypeError as e:
            msg = "Invalid dictionary provided for schema"
            raise ValueError(msg) from e
    elif isinstance(schema, str):
        try:
            schema_dict = ln.to_dict(schema)
        except Exception as e:
            msg = "Invalid JSON schema string provided"
            raise ValueError(msg) from e
        schema_json = schema
    else:
        msg = "Schema must be a JSON string or a dictionary"
        raise TypeError(msg)

    resolved_name = _extract_model_name_from_schema(schema_dict, model_name)
    return schema_json, schema_dict, resolved_name


def _generate_model_code(
    schema_json: str,
    output_file: Path,
    pydantic_version: Any,
    python_version: Any,
    generate_func: Any,
    input_file_type_enum: Any,
) -> None:
    """Generate Pydantic model code from schema."""
    try:
        generate_func(
            schema_json,
            input_file_type=input_file_type_enum.JsonSchema,
            input_filename="schema.json",
            output=output_file,
            output_model_type=pydantic_version,
            target_python_version=python_version,
            base_class="pydantic.BaseModel",
        )
    except Exception as e:
        msg = "Failed to generate model code from schema"
        raise RuntimeError(msg) from e


def _load_generated_module(output_file: Path, module_name: str) -> Any:
    """Dynamically import generated Python module."""
    if not output_file.exists():
        msg = f"Generated model file not created: {output_file}"
        raise FileNotFoundError(msg)

    spec = importlib.util.spec_from_file_location(module_name, str(output_file))
    if spec is None or spec.loader is None:
        msg = f"Could not create module spec for {output_file}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        msg = f"Failed to load generated module from {output_file}"
        raise RuntimeError(msg) from e

    return module


def _extract_model_class(
    module: Any,
    model_name: str,
    output_file: Path,
) -> type[BaseModel]:
    """Find BaseModel class in generated module."""

    def _is_valid_model(obj: Any) -> bool:
        return isinstance(obj, type) and issubclass(obj, BaseModel)

    try:
        model_class = getattr(module, model_name)
        if not _is_valid_model(model_class):
            msg = f"'{model_name}' is not a Pydantic BaseModel class"
            raise TypeError(msg)
        return model_class
    except AttributeError:
        pass

    # Fallback to 'Model'
    try:
        model_class = module.Model
        if not _is_valid_model(model_class):
            msg = "Fallback 'Model' is not a Pydantic BaseModel class"
            raise TypeError(msg)
        return model_class
    except AttributeError:
        pass

    # List available models for debugging
    available = [
        attr
        for attr in dir(module)
        if _is_valid_model(getattr(module, attr, None)) and getattr(module, attr) is not BaseModel
    ]
    msg = (
        f"Could not find '{model_name}' or 'Model' in {output_file}. "
        f"Available BaseModel classes: {available}"
    )
    raise AttributeError(msg)


def _rebuild_model(model_class: type[BaseModel], module: Any, model_name: str) -> None:
    """Rebuild model with proper type resolution."""
    try:
        model_class.model_rebuild(_types_namespace=module.__dict__, force=True)
    except (PydanticUserError, NameError) as e:
        msg = f"Type resolution failed during rebuild for {model_name}"
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Unexpected error during model_rebuild for {model_name}"
        raise RuntimeError(msg) from e


def load_pydantic_model_from_schema(
    schema: str | dict[str, Any],
    model_name: str = "DynamicModel",
    /,
    pydantic_version: Any = None,
    python_version: Any = None,
) -> type[BaseModel]:
    """Generate Pydantic model dynamically from JSON schema.

    Creates model class via datamodel-code-generator, imports it,
    and rebuilds with proper type resolution.

    Args:
        schema: JSON schema (string or dict)
        model_name: Base name for model (schema title takes precedence)
        pydantic_version: DataModelType enum (default: PydanticV2BaseModel)
        python_version: PythonVersion enum (default: PY_312)

    Returns:
        Dynamically created BaseModel class

    Raises:
        ImportError: datamodel-code-generator not installed
        ValueError: Invalid schema format
        TypeError: Invalid schema type
        RuntimeError: Generation or loading failed

    Example:
        >>> schema = {"title": "User", "type": "object", ...}
        >>> UserModel = load_pydantic_model_from_schema(schema)
        >>> user = UserModel(name="Alice", age=30)
    """
    try:
        from datamodel_code_generator import (
            DataModelType,
            InputFileType,
            PythonVersion,
            generate,
        )
    except ImportError as e:
        msg = (
            "datamodel-code-generator not installed. "
            "Install with: pip install 'lionherd-core[schema-gen]' "
            "or: pip install datamodel-code-generator"
        )
        raise ImportError(msg) from e

    pydantic_version = pydantic_version or DataModelType.PydanticV2BaseModel
    python_version = python_version or _get_python_version_enum(PythonVersion)

    schema_json, _, resolved_name = _prepare_schema_input(schema, model_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_file = tmpdir_path / f"{resolved_name.lower()}_model_{hash(schema_json)}.py"
        module_name = output_file.stem

        _generate_model_code(
            schema_json, output_file, pydantic_version, python_version, generate, InputFileType
        )
        module = _load_generated_module(output_file, module_name)
        model_class = _extract_model_class(module, resolved_name, output_file)
        _rebuild_model(model_class, module, resolved_name)

        return model_class

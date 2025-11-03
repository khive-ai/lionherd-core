# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from pydantic import (
    BaseModel,
    ValidationError as PydanticValidationError,
)

from lionherd_core.types import Operable

from .errors import MissingFieldError, TypeMismatchError
from .parser import parse_value
from .types import LNDLOutput, LvarMetadata


def resolve_references_prefixed(
    out_fields: dict[str, list[str] | str],
    lvars: dict[str, LvarMetadata],
    operable: Operable,
) -> LNDLOutput:
    """Resolve namespace-prefixed OUT{} fields and validate against operable specs.

    Args:
        out_fields: Parsed OUT{} block (field -> list of var names OR literal value)
        lvars: Extracted namespace-prefixed lvar declarations
        operable: Operable containing allowed specs

    Returns:
        LNDLOutput with validated Pydantic model instances or scalar values

    Raises:
        MissingFieldError: Required spec field not in OUT{}
        TypeMismatchError: Variable model doesn't match spec type
        ValueError: Variable not found or field mismatch
    """
    # Check all fields in OUT{} are allowed by operable
    operable.check_allowed(*out_fields.keys())

    # Check all required specs present
    for spec in operable.get_specs():
        is_required = spec.get("required", True)
        if is_required and spec.name not in out_fields:
            raise MissingFieldError(f"Required field '{spec.name}' missing from OUT{{}}")

    # Resolve and validate each field (collect all errors)
    validated_fields = {}
    errors: list[Exception] = []

    for field_name, value in out_fields.items():
        try:
            # Get spec for this field
            spec = operable.get(field_name)
            if spec is None:
                raise ValueError(
                    f"OUT{{}} field '{field_name}' has no corresponding Spec in Operable"
                )

            # Get type from spec
            target_type = spec.base_type

            # Check if this is a scalar type (float, str, int, bool)
            is_scalar = target_type in (float, str, int, bool)

            if is_scalar:
                # Handle scalar assignment
                if isinstance(value, list):
                    # Array syntax for scalar - should be single variable
                    if len(value) != 1:
                        raise ValueError(
                            f"Scalar field '{field_name}' cannot use multiple variables, got {value}"
                        )
                    var_name = value[0]

                    # Look up variable in lvars
                    if var_name not in lvars:
                        raise ValueError(
                            f"Variable '{var_name}' referenced in OUT{{}} but not declared"
                        )

                    lvar_meta = lvars[var_name]
                    parsed_value = parse_value(lvar_meta.value)
                else:
                    # Literal value (string)
                    parsed_value = parse_value(value)

                # Type conversion and validation
                try:
                    validated_value = target_type(parsed_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Failed to convert value for field '{field_name}' to {target_type.__name__}: {e}"
                    ) from e

                validated_fields[field_name] = validated_value

            else:
                # Handle Pydantic BaseModel construction
                if not isinstance(value, list):
                    raise ValueError(
                        f"BaseModel field '{field_name}' requires array syntax, got literal: {value}"
                    )

                var_list = value

                # Validate it's a BaseModel
                if not isinstance(target_type, type) or not issubclass(target_type, BaseModel):
                    raise TypeError(
                        f"Spec base_type for '{field_name}' must be a Pydantic BaseModel or scalar type, "
                        f"got {target_type}"
                    )

                # Build kwargs from variable list
                kwargs = {}
                for var_name in var_list:
                    # Look up variable in lvars
                    if var_name not in lvars:
                        raise ValueError(
                            f"Variable '{var_name}' referenced in OUT{{}} but not declared"
                        )

                    lvar_meta = lvars[var_name]

                    # Validate: model name matches
                    if lvar_meta.model != target_type.__name__:
                        raise TypeMismatchError(
                            f"Variable '{var_name}' is for model '{lvar_meta.model}', "
                            f"but field '{field_name}' expects '{target_type.__name__}'"
                        )

                    # Map field name to kwargs
                    kwargs[lvar_meta.field] = parse_value(lvar_meta.value)

                # Construct Pydantic model instance
                try:
                    instance = target_type(**kwargs)
                except PydanticValidationError as e:
                    raise ValueError(
                        f"Failed to construct {target_type.__name__} for field '{field_name}': {e}"
                    ) from e

                # Apply validators/rules if specified in spec metadata
                validators = spec.get("validator")
                if validators:
                    validators = validators if isinstance(validators, list) else [validators]
                    for validator in validators:
                        if hasattr(validator, "invoke"):
                            instance = validator.invoke(field_name, instance, target_type)
                        else:
                            instance = validator(instance)

                validated_fields[field_name] = instance

        except Exception as e:
            # Collect errors for aggregation
            errors.append(e)

    # Raise all collected errors as ExceptionGroup
    if errors:
        raise ExceptionGroup("LNDL validation failed", errors)

    return LNDLOutput(fields=validated_fields, lvars=lvars, raw_out_block=str(out_fields))


def parse_lndl(response: str, operable: Operable) -> LNDLOutput:
    """Parse LNDL response and validate against operable specs.

    Args:
        response: Full LLM response containing lvars and OUT{}
        operable: Operable containing allowed specs

    Returns:
        LNDLOutput with validated fields
    """
    from .parser import extract_lvars_prefixed, extract_out_block, parse_out_block_array

    # 1. Extract namespace-prefixed lvars
    lvars_prefixed = extract_lvars_prefixed(response)

    # 2. Extract and parse OUT{} block with array syntax
    out_content = extract_out_block(response)
    out_fields = parse_out_block_array(out_content)

    # 3. Resolve references and validate
    return resolve_references_prefixed(out_fields, lvars_prefixed, operable)

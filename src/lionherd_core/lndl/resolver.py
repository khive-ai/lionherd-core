# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from pydantic import (
    BaseModel,
    ValidationError as PydanticValidationError,
)

from lionherd_core.libs.schema_handlers._function_call_parser import parse_function_call
from lionherd_core.types import Operable

from .errors import MissingFieldError, MissingOutBlockError, TypeMismatchError
from .parser import parse_value
from .types import ActionCall, LactMetadata, LNDLOutput, LvarMetadata, RLvarMetadata


def resolve_references_prefixed(
    out_fields: dict[str, list[str] | str],
    lvars: dict[str, LvarMetadata | RLvarMetadata],
    lacts: dict[str, LactMetadata],
    operable: Operable,
) -> LNDLOutput:
    """Resolve namespace-prefixed OUT{} fields and validate against operable specs.

    Args:
        out_fields: Parsed OUT{} block (field -> list of var names OR literal value)
        lvars: Extracted namespace-prefixed lvar declarations
        lacts: Extracted action declarations (name -> LactMetadata)
        operable: Operable containing allowed specs

    Returns:
        LNDLOutput with validated Pydantic model instances or scalar values

    Raises:
        MissingFieldError: Required spec field not in OUT{}
        TypeMismatchError: Variable model doesn't match spec type
        ValueError: Variable not found, field mismatch, or name collision
    """
    # Check for name collisions between lvars and lacts
    lvar_names = set(lvars.keys())
    lact_names = set(lacts.keys())
    collisions = lvar_names & lact_names

    if collisions:
        raise ValueError(
            f"Name collision detected: {collisions} used in both <lvar> and <lact> declarations"
        )
    # Check all fields in OUT{} are allowed by operable
    operable.check_allowed(*out_fields.keys())

    # Check all required specs present
    for spec in operable.get_specs():
        is_required = spec.get("required", True)
        if is_required and spec.name not in out_fields:
            raise MissingFieldError(f"Required field '{spec.name}' missing from OUT{{}}")

    # Resolve and validate each field (collect all errors)
    validated_fields = {}
    parsed_actions: dict[str, ActionCall] = {}  # Actions referenced in OUT{}
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
                    # Array syntax for scalar - should be single variable or action
                    if len(value) != 1:
                        raise ValueError(
                            f"Scalar field '{field_name}' cannot use multiple variables, got {value}"
                        )
                    var_name = value[0]

                    # Check if this is an action reference
                    if var_name in lacts:
                        # Get action metadata
                        lact_meta = lacts[var_name]

                        # Parse function call with context
                        try:
                            parsed_call = parse_function_call(lact_meta.call)
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid function call syntax in action '{var_name}' for scalar field '{field_name}':\n"
                                f"  Action call: {lact_meta.call}\n"
                                f"  Parse error: {e}"
                            ) from e

                        # Create ActionCall instance
                        action_call = ActionCall(
                            name=var_name,
                            function=parsed_call["tool"],
                            arguments=parsed_call["arguments"],
                            raw_call=lact_meta.call,
                        )
                        parsed_actions[var_name] = action_call

                        # For scalar actions, we mark this field for later execution
                        # The actual execution happens externally, we just store the action
                        # For now, we can't validate the result type, so we skip type conversion
                        # The field will be populated with the action result during execution
                        validated_fields[field_name] = action_call
                        continue

                    # Look up variable in lvars
                    if var_name not in lvars:
                        raise ValueError(
                            f"Variable or action '{var_name}' referenced in OUT{{}} but not declared"
                        )

                    lvar_meta = lvars[var_name]
                    # Value might already be typed by new parser
                    if isinstance(lvar_meta.value, str):
                        parsed_value = parse_value(lvar_meta.value)
                    else:
                        parsed_value = lvar_meta.value
                else:
                    # Literal value - might already be typed by new parser
                    parsed_value = parse_value(value) if isinstance(value, str) else value

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

                # Special case: single action reference that returns entire model
                if len(var_list) == 1 and var_list[0] in lacts:
                    action_name = var_list[0]
                    lact_meta = lacts[action_name]

                    # Direct actions (no namespace) can return entire model
                    if lact_meta.model is None:
                        # Parse function call with context
                        try:
                            parsed_call = parse_function_call(lact_meta.call)
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid function call syntax in direct action '{action_name}' for BaseModel field '{field_name}':\n"
                                f"  Action call: {lact_meta.call}\n"
                                f"  Parse error: {e}"
                            ) from e

                        # Create ActionCall instance
                        action_call = ActionCall(
                            name=action_name,
                            function=parsed_call["tool"],
                            arguments=parsed_call["arguments"],
                            raw_call=lact_meta.call,
                        )
                        parsed_actions[action_name] = action_call

                        # Store action as field value (will be executed to get model instance)
                        validated_fields[field_name] = action_call
                        continue
                    # If namespaced, fall through to mixing logic below

                # Build kwargs from variable list - supports mixing lvars and namespaced actions
                kwargs = {}
                for var_name in var_list:
                    # Check if this is an action reference
                    if var_name in lacts:
                        lact_meta = lacts[var_name]

                        # Namespaced actions must specify which field they populate
                        if lact_meta.model is None or lact_meta.field is None:
                            raise ValueError(
                                f"Direct action '{var_name}' cannot be mixed with lvars in BaseModel field '{field_name}'. "
                                f"Use namespaced syntax: <lact {target_type.__name__}.fieldname {var_name}>...</lact>"
                            )

                        # Validate: model name matches
                        if lact_meta.model != target_type.__name__:
                            raise TypeMismatchError(
                                f"Action '{var_name}' is for model '{lact_meta.model}', "
                                f"but field '{field_name}' expects '{target_type.__name__}'"
                            )

                        # Parse action function call with context
                        try:
                            parsed_call = parse_function_call(lact_meta.call)
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid function call syntax in namespaced action '{var_name}' for field '{lact_meta.model}.{lact_meta.field}':\n"
                                f"  Action call: {lact_meta.call}\n"
                                f"  Parse error: {e}"
                            ) from e

                        # Create ActionCall instance
                        action_call = ActionCall(
                            name=var_name,
                            function=parsed_call["tool"],
                            arguments=parsed_call["arguments"],
                            raw_call=lact_meta.call,
                        )
                        parsed_actions[var_name] = action_call

                        # Use the namespaced field to map action result
                        kwargs[lact_meta.field] = action_call
                        continue

                    # Look up variable in lvars
                    if var_name not in lvars:
                        raise ValueError(
                            f"Variable or action '{var_name}' referenced in OUT{{}} but not declared"
                        )

                    lvar_meta = lvars[var_name]

                    # Raw lvars cannot be used in BaseModel fields (no type validation)
                    if isinstance(lvar_meta, RLvarMetadata):
                        raise ValueError(
                            f"Raw lvar '{var_name}' cannot be used in BaseModel field '{field_name}'. "
                            f"Use namespaced format: <lvar Model.field alias>"
                        )

                    # Validate: model name matches
                    if lvar_meta.model != target_type.__name__:
                        raise TypeMismatchError(
                            f"Variable '{var_name}' is for model '{lvar_meta.model}', "
                            f"but field '{field_name}' expects '{target_type.__name__}'"
                        )

                    # Map field name to kwargs (value might already be typed)
                    if isinstance(lvar_meta.value, str):
                        kwargs[lvar_meta.field] = parse_value(lvar_meta.value)
                    else:
                        kwargs[lvar_meta.field] = lvar_meta.value

                # Construct Pydantic model instance
                # WARNING: model_construct() bypasses Pydantic validation when ActionCall objects present.
                # Caller MUST re-validate after executing actions using revalidate_with_action_results().
                # See LNDLOutput docstring for complete action execution lifecycle.
                has_actions = any(isinstance(v, ActionCall) for v in kwargs.values())
                try:
                    if has_actions:
                        # PARTIAL VALIDATION: Field constraints, validators, and type checking bypassed
                        instance = target_type.model_construct(**kwargs)
                    else:
                        # FULL VALIDATION: Normal Pydantic validation for models without actions
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

    return LNDLOutput(
        fields=validated_fields,
        lvars=lvars,
        lacts=lacts,
        actions=parsed_actions,
        raw_out_block=str(out_fields),
    )


def parse_lndl(response: str, operable: Operable) -> LNDLOutput:
    """Parse LNDL response and validate against operable specs.

    Args:
        response: Full LLM response containing lvars, lacts, and OUT{}
        operable: Operable containing allowed specs

    Returns:
        LNDLOutput with validated fields and parsed actions
    """
    from .lexer import Lexer
    from .parser import Parser

    # 1. Tokenize using new lexer
    lexer = Lexer(response)
    tokens = lexer.tokenize()

    # 2. Parse into AST (hybrid: regex for content, tokens for structure)
    parser = Parser(tokens, source_text=response)
    program = parser.parse()

    # 3. Convert AST to resolver input format (supports both Lvar and RLvar)
    from .ast import Lvar

    lvars_prefixed: dict[str, LvarMetadata | RLvarMetadata] = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            # Namespaced lvar
            lvars_prefixed[lvar.alias] = LvarMetadata(
                model=lvar.model,
                field=lvar.field,
                local_name=lvar.alias,
                value=lvar.content,
            )
        else:  # RLvar
            # Raw lvar
            lvars_prefixed[lvar.alias] = RLvarMetadata(
                local_name=lvar.alias,
                value=lvar.content,
            )

    lacts_prefixed: dict[str, LactMetadata] = {}
    for lact in program.lacts:
        lacts_prefixed[lact.alias] = LactMetadata(
            model=lact.model,
            field=lact.field,
            local_name=lact.alias,
            call=lact.call,
        )

    # 4. Extract OUT{} fields (already in correct format from parser)
    if not program.out_block:
        raise MissingOutBlockError("No OUT{} block found in response")

    out_fields = program.out_block.fields

    # 5. Resolve references and validate
    return resolve_references_prefixed(out_fields, lvars_prefixed, lacts_prefixed, operable)

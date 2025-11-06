# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from lionherd_core.errors import ValidationError

from ..libs.string_handlers._extract_json import extract_json
from ..libs.string_handlers._string_similarity import SIMILARITY_TYPE
from ..types import KeysLike
from ._fuzzy_match import FuzzyMatchKeysParams, fuzzy_match_keys
from ._to_dict import to_dict

if TYPE_CHECKING:
    from pydantic import BaseModel


__all__ = ("fuzzy_validate_pydantic",)


def fuzzy_validate_pydantic(
    text,
    /,
    model_type: "type[BaseModel]",
    fuzzy_parse: bool = True,
    fuzzy_match: bool = False,
    fuzzy_match_params: FuzzyMatchKeysParams | dict = None,
):
    """Validate and parse text/dict into Pydantic model with fuzzy parsing.

    Args:
        text: Input data (BaseModel instance, dict, or JSON string)
        model_type: Target Pydantic model class
        fuzzy_parse: Enable fuzzy JSON extraction from text
        fuzzy_match: Enable fuzzy key matching for field names
        fuzzy_match_params: Parameters for fuzzy matching (dict or FuzzyMatchKeysParams)

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If JSON extraction or model validation fails
        TypeError: If fuzzy_match_params is invalid type
    """
    # Handle already-valid model instances
    if isinstance(text, model_type):
        return text

    # Handle dict inputs directly (skip JSON extraction)
    if isinstance(text, dict):
        model_data = text
    else:
        # Handle string inputs (JSON strings, markdown, etc.)
        try:
            model_data = extract_json(text, fuzzy_parse=fuzzy_parse)
        except Exception as e:
            raise ValidationError(f"Failed to extract valid JSON from model response: {e}") from e

    d = model_data
    if fuzzy_match:
        if fuzzy_match_params is None:
            model_data = fuzzy_match_keys(d, model_type.model_fields, handle_unmatched="remove")
        elif isinstance(fuzzy_match_params, dict):
            model_data = fuzzy_match_keys(d, model_type.model_fields, **fuzzy_match_params)
        elif isinstance(fuzzy_match_params, FuzzyMatchKeysParams):
            model_data = fuzzy_match_params(d, model_type.model_fields)
        else:
            raise TypeError("fuzzy_keys_params must be a dict or FuzzyMatchKeysParams instance")

    try:
        return model_type.model_validate(model_data)
    except Exception as e:
        raise ValidationError(f"Validation failed: {e}") from e


def fuzzy_validate_mapping(
    d: Any,
    keys: KeysLike,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
) -> dict[str, Any]:
    """Validate any input into dict with expected keys and fuzzy matching.

    Converts input (dict, JSON string, XML, object) to dict and validates keys.

    Args:
        d: Input to convert and validate
        keys: Expected keys (list or dict-like)
        similarity_algo: String similarity algorithm
        similarity_threshold: Minimum similarity score (0.0-1.0)
        fuzzy_match: Enable fuzzy key matching
        handle_unmatched: How to handle unmatched keys
        fill_value: Default value for missing keys
        fill_mapping: Custom values for specific keys
        strict: Raise if expected keys are missing
        suppress_conversion_errors: Return empty dict on conversion failure

    Returns:
        Validated dictionary with corrected keys

    Raises:
        TypeError: If d is None
        ValueError: If conversion fails and suppress_conversion_errors is False
    """
    if d is None:
        raise TypeError("Input cannot be None")

    # Try converting to dictionary
    try:
        if isinstance(d, str):
            try:
                json_result = extract_json(d, fuzzy_parse=True, return_one_if_single=True)
                dict_input = json_result[0] if isinstance(json_result, list) else json_result
            except Exception:
                dict_input = to_dict(d, fuzzy_parse=True, suppress=True)
        else:
            dict_input = to_dict(d, prioritize_model_dump=True, fuzzy_parse=True, suppress=True)

        if not isinstance(dict_input, dict):
            if suppress_conversion_errors:
                dict_input = {}
            else:
                raise ValueError(f"Failed to convert input to dictionary: {type(dict_input)}")

    except Exception as e:
        if suppress_conversion_errors:
            dict_input = {}
        else:
            raise ValueError(f"Failed to convert input to dictionary: {e}")

    # Validate the dictionary
    return fuzzy_match_keys(
        dict_input,
        keys,
        similarity_algo=similarity_algo,
        similarity_threshold=similarity_threshold,
        fuzzy_match=fuzzy_match,
        handle_unmatched=handle_unmatched,
        fill_value=fill_value,
        fill_mapping=fill_mapping,
        strict=strict,
    )

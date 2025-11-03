# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging

from lionherd_core.libs.string_handlers._string_similarity import string_similarity
from lionherd_core.types import Operable

from .errors import AmbiguousMatchError, MissingFieldError
from .parser import extract_lvars_prefixed, extract_out_block, parse_out_block_array
from .resolver import resolve_references_prefixed
from .types import LNDLOutput, LvarMetadata

__all__ = ("parse_lndl_fuzzy",)

logger = logging.getLogger(__name__)


def _correct_name(
    target: str,
    candidates: list[str],
    threshold: float,
    context: str = "name",
) -> str:
    """Correct name using fuzzy matching with tie detection.

    Args:
        target: User-provided name (may have typo)
        candidates: Valid names to match against
        threshold: Similarity threshold (0.0-1.0)
        context: Context for error messages (e.g., "field", "lvar")

    Returns:
        Corrected name

    Raises:
        MissingFieldError: No match above threshold
        AmbiguousMatchError: Multiple matches within 0.05 similarity

    Example:
        >>> _correct_name("titel", ["title", "content"], 0.85, "field")
        "title"  # Jaro-Winkler: 0.933
    """
    # Exact match - no fuzzy needed
    if target in candidates:
        return target

    # Strict mode (threshold=1.0) - exact match only
    if threshold >= 1.0:
        raise MissingFieldError(
            f"{context.capitalize()} '{target}' not found. "
            f"Available: {candidates} (strict mode: exact match required)"
        )

    # Fuzzy match with tie detection
    result = string_similarity(
        word=target,
        correct_words=candidates,
        algorithm="jaro_winkler",
        threshold=threshold,
        return_most_similar=False,  # Get ALL matches for tie detection
    )

    if not result:
        raise MissingFieldError(
            f"{context.capitalize()} '{target}' not found above threshold {threshold}. "
            f"Available: {candidates}"
        )

    # result is str | list[str] | None
    if isinstance(result, list):
        # Calculate scores for tie detection
        # Get similarity scores directly from algorithms
        from lionherd_core.libs.string_handlers._string_similarity import SIMILARITY_ALGO_MAP

        algo_func = SIMILARITY_ALGO_MAP["jaro_winkler"]
        scores = {candidate: algo_func(target, candidate) for candidate in result}

        # Find max score
        max_score = max(scores.values())

        # Check for ties (matches within 0.05)
        ties = [k for k, v in scores.items() if abs(v - max_score) < 0.05]

        if len(ties) > 1:
            scores_str = ", ".join(f"'{k}': {scores[k]:.3f}" for k in ties)
            raise AmbiguousMatchError(
                f"Ambiguous match for {context} '{target}': [{scores_str}]. "
                f"Multiple candidates scored within 0.05. Be more specific."
            )

        # Single clear winner
        match = result[0]
    else:
        # Single match returned (str)
        match = str(result)

    # Log correction
    if match != target:
        logger.debug(f"Fuzzy corrected {context}: '{target}' → '{match}'")

    return match


def parse_lndl_fuzzy(
    response: str,
    operable: Operable,
    /,
    *,
    threshold: float = 0.85,
    threshold_field: float | None = None,
    threshold_lvar: float | None = None,
    threshold_model: float | None = None,
    threshold_spec: float | None = None,
) -> LNDLOutput:
    """Parse LNDL with fuzzy matching (default) or strict mode (threshold=1.0).

    Args:
        response: Full LLM response containing lvars and OUT{}
        operable: Operable containing allowed specs
        threshold: Global similarity threshold (default: 0.85)
                   - 0.85: Fuzzy matching (production-proven)
                   - 1.0: Strict mode (exact matches only)
                   - 0.7-0.95: Custom tolerance
        threshold_field: Override threshold for field names (default: use threshold)
        threshold_lvar: Override threshold for lvar references (default: use threshold)
        threshold_model: Override threshold for model names (default: use threshold or 0.90)
        threshold_spec: Override threshold for spec names (default: use threshold)

    Returns:
        LNDLOutput with validated fields

    Raises:
        MissingFieldError: No match above threshold
        AmbiguousMatchError: Multiple matches within 0.05 similarity
        ValueError: Validation errors from strict resolver

    Example:
        >>> # Default: Fuzzy matching
        >>> response = '''
        ... <lvar Report.titel title>Good Title</lvar>
        ... OUT{reprot: [titel]}
        ... '''
        >>> parse_lndl_fuzzy(response, operable)  # Auto-corrects typos

        >>> # Strict mode
        >>> parse_lndl_fuzzy(response, operable, threshold=1.0)  # Raises error

    Architecture:
        1. Parse LNDL (extract lvars and OUT{})
        2. Pre-correct typos in lvar names, model names, field names, spec names
        3. Call strict resolver with corrected inputs (zero duplication)
    """
    # Set default thresholds
    threshold_field = threshold_field if threshold_field is not None else threshold
    threshold_lvar = threshold_lvar if threshold_lvar is not None else threshold
    threshold_model = (
        threshold_model if threshold_model is not None else max(threshold, 0.90)
    )  # Stricter for model names
    threshold_spec = threshold_spec if threshold_spec is not None else threshold

    # 1. Extract namespace-prefixed lvars and OUT{} block (same as strict)
    lvars_raw = extract_lvars_prefixed(response)
    out_content = extract_out_block(response)
    out_fields_raw = parse_out_block_array(out_content)

    # If threshold is 1.0 (strict mode), validate strictly then call resolver
    if threshold >= 1.0:
        # Validate model names exist
        expected_models = {spec.base_type.__name__ for spec in operable.get_specs()}
        for lvar in lvars_raw.values():
            if lvar.model not in expected_models:
                raise MissingFieldError(
                    f"Model '{lvar.model}' not found. "
                    f"Available: {list(expected_models)} (strict mode: exact match required)"
                )

        # Validate field names exist for each model
        for lvar in lvars_raw.values():
            # Get spec for this model
            spec = None
            for s in operable.get_specs():
                if s.base_type.__name__ == lvar.model:
                    spec = s
                    break

            if spec is None:
                continue  # Already caught above

            # Check if field exists
            expected_fields = list(spec.base_type.model_fields.keys())
            if lvar.field not in expected_fields:
                raise MissingFieldError(
                    f"Field '{lvar.field}' not found in model {lvar.model}. "
                    f"Available: {expected_fields} (strict mode: exact match required)"
                )

        # Validate spec names in OUT{} block
        expected_spec_names = list(operable.allowed())
        for spec_name in out_fields_raw:
            if spec_name not in expected_spec_names:
                raise MissingFieldError(
                    f"Spec '{spec_name}' not found. "
                    f"Available: {expected_spec_names} (strict mode: exact match required)"
                )

        return resolve_references_prefixed(out_fields_raw, lvars_raw, operable)

    # 2. Pre-correct lvar metadata (model names and field names)
    # Collect all unique model names and field names from lvars
    raw_model_names = {lvar.model for lvar in lvars_raw.values()}
    raw_field_names_by_model: dict[str, set[str]] = {}
    for lvar in lvars_raw.values():
        if lvar.model not in raw_field_names_by_model:
            raw_field_names_by_model[lvar.model] = set()
        raw_field_names_by_model[lvar.model].add(lvar.field)

    # Get expected model names from operable specs
    expected_models = {spec.base_type.__name__ for spec in operable.get_specs()}

    # Correct model names in lvars
    model_corrections: dict[str, str] = {}  # raw_model → corrected_model
    for raw_model in raw_model_names:
        corrected_model = _correct_name(raw_model, list(expected_models), threshold_model, "model")
        model_corrections[raw_model] = corrected_model

    # Correct field names in lvars (per model)
    field_corrections: dict[tuple[str, str], str] = {}  # (model, raw_field) → corrected_field
    for raw_model, raw_fields in raw_field_names_by_model.items():
        corrected_model = model_corrections[raw_model]

        # Get expected fields for this model from spec
        spec = None
        for s in operable.get_specs():
            if s.base_type.__name__ == corrected_model:
                spec = s
                break

        if spec is None:
            # Model not in operable - let strict resolver handle error
            continue

        # Get field names from Pydantic model
        expected_fields = list(spec.base_type.model_fields.keys())

        for raw_field in raw_fields:
            corrected_field = _correct_name(
                raw_field, expected_fields, threshold_field, f"field (model {corrected_model})"
            )
            field_corrections[(raw_model, raw_field)] = corrected_field

    # Rebuild lvars with corrected model and field names
    lvars_corrected: dict[str, LvarMetadata] = {}
    for local_name, lvar in lvars_raw.items():
        corrected_model = model_corrections.get(lvar.model, lvar.model)
        corrected_field = field_corrections.get((lvar.model, lvar.field), lvar.field)

        lvars_corrected[local_name] = LvarMetadata(
            model=corrected_model,
            field=corrected_field,
            local_name=lvar.local_name,
            value=lvar.value,
        )

    # 3. Pre-correct OUT{} spec names (keys in out_fields_raw)
    expected_spec_names = list(operable.allowed())
    out_fields_corrected: dict[str, list[str] | str] = {}

    for raw_spec_name, value in out_fields_raw.items():
        corrected_spec_name = _correct_name(
            raw_spec_name, expected_spec_names, threshold_spec, "spec"
        )
        out_fields_corrected[corrected_spec_name] = value

    # 4. Pre-correct lvar references in OUT{} arrays
    available_lvar_names = list(lvars_corrected.keys())
    out_fields_final: dict[str, list[str] | str] = {}

    for spec_name, value in out_fields_corrected.items():
        if isinstance(value, list):
            # Array of lvar references - correct each reference
            corrected_refs = []
            for raw_ref in value:
                corrected_ref = _correct_name(
                    raw_ref, available_lvar_names, threshold_lvar, "lvar reference"
                )
                corrected_refs.append(corrected_ref)
            out_fields_final[spec_name] = corrected_refs
        else:
            # Literal value - no correction needed
            out_fields_final[spec_name] = value

    # 5. Call strict resolver with corrected inputs (REUSE existing logic)
    return resolve_references_prefixed(out_fields_final, lvars_corrected, operable)

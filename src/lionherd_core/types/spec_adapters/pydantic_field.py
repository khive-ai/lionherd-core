# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import TYPE_CHECKING, Any

from .._sentinel import Unset, is_sentinel
from ._protocol import SpecAdapter

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    from ..operable import Operable
    from ..spec import Spec


@functools.lru_cache(maxsize=1)
def _get_pydantic_field_params() -> set[str]:
    """Get valid Pydantic Field params (cached)."""
    import inspect

    from pydantic import Field as PydanticField

    params = set(inspect.signature(PydanticField).parameters.keys())
    params.discard("kwargs")
    return params


class PydanticSpecAdapter(SpecAdapter["BaseModel"]):
    """Pydantic adapter: Spec → FieldInfo → BaseModel."""

    @classmethod
    def create_field(cls, spec: "Spec") -> "FieldInfo":
        """Create Pydantic FieldInfo from Spec."""
        from pydantic import Field as PydanticField

        # Get valid Pydantic Field parameters (cached)
        pydantic_field_params = _get_pydantic_field_params()

        # Extract metadata for FieldInfo
        field_kwargs = {}

        if not is_sentinel(spec.metadata, none_as_sentinel=True):
            for meta in spec.metadata:
                if meta.key == "default":
                    # Handle callable defaults as default_factory
                    if callable(meta.value):
                        field_kwargs["default_factory"] = meta.value
                    else:
                        field_kwargs["default"] = meta.value
                elif meta.key == "validator":
                    # Validators are handled separately in create_model
                    continue
                elif meta.key in pydantic_field_params:
                    # Pass through standard Pydantic field attributes
                    field_kwargs[meta.key] = meta.value
                elif meta.key in {"nullable", "listable"}:
                    # These are FieldTemplate markers, don't pass to FieldInfo
                    pass
                else:
                    # Filter out unserializable objects from json_schema_extra
                    if isinstance(meta.value, type):
                        # Skip type objects - can't be serialized
                        continue

                    # Any other metadata goes in json_schema_extra
                    if "json_schema_extra" not in field_kwargs:
                        field_kwargs["json_schema_extra"] = {}
                    field_kwargs["json_schema_extra"][meta.key] = meta.value

        # Handle nullable case - ensure default is set if not already
        if (
            spec.is_nullable
            and "default" not in field_kwargs
            and "default_factory" not in field_kwargs
        ):
            field_kwargs["default"] = None

        field_info = PydanticField(**field_kwargs)
        field_info.annotation = spec.annotation

        return field_info

    @classmethod
    def create_validator(cls, spec: "Spec") -> dict[str, Any] | None:
        """Create Pydantic field_validator from Spec metadata."""
        from .._sentinel import Undefined

        v = spec.get("validator")
        if v is Unset or v is Undefined:
            return None

        from pydantic import field_validator

        field_name = spec.name or "field"
        # check_fields=False allows the validator to be defined in a base class before the field exists
        return {f"{field_name}_validator": field_validator(field_name, check_fields=False)(v)}

    @classmethod
    def create_model(
        cls,
        op: "Operable",
        model_name: str,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        base_type: type["BaseModel"] | None = None,
        doc: str | None = None,
    ) -> type["BaseModel"]:
        """Generate Pydantic BaseModel from Operable using pydantic.create_model()."""
        from pydantic import BaseModel, create_model

        use_specs = op.get_specs(include=include, exclude=exclude)
        use_fields = {i.name: cls.create_field(i) for i in use_specs if i.name}

        # Convert fields to (type, FieldInfo) tuples for create_model
        field_definitions = {
            name: (field_info.annotation, field_info) for name, field_info in use_fields.items()
        }

        # Collect validators
        validators = {}
        for spec in use_specs:
            if spec.name and (validator := cls.create_validator(spec)):
                validators.update(validator)

        # If we have validators, create a base class with them
        # Otherwise use the provided base_type or BaseModel
        if validators:
            # Create a temporary base class with validators as class attributes
            base_with_validators = type(
                f"{model_name}Base",
                (base_type or BaseModel,),
                validators,
            )
            actual_base = base_with_validators
        else:
            actual_base = base_type or BaseModel

        # Create model using pydantic's create_model
        model_cls = create_model(
            model_name,
            __base__=actual_base,
            __doc__=doc,
            **field_definitions,
        )

        model_cls.model_rebuild()
        return model_cls

    @classmethod
    def fuzzy_match_fields(
        cls, data: dict, model_cls: type["BaseModel"], strict: bool = False
    ) -> dict[str, Any]:
        """Match data keys to Pydantic fields (fuzzy). Filters sentinels. Args: data, model_cls, strict."""
        from lionherd_core.ln._fuzzy_match import fuzzy_match_keys

        from .._sentinel import not_sentinel

        # "ignore" mode only includes successfully matched fields (no sentinel injection)
        # "raise" mode raises on unmatched keys for strict validation
        handle_mode = "raise" if strict else "ignore"

        matched = fuzzy_match_keys(data, model_cls.model_fields, handle_unmatched=handle_mode)

        # Filter out sentinel values (Unset, Undefined)
        return {k: v for k, v in matched.items() if not_sentinel(v)}

    @classmethod
    def validate_model(cls, model_cls: type["BaseModel"], data: dict) -> "BaseModel":
        """Validate dict → Pydantic model via model_validate()."""
        return model_cls.model_validate(data)

    @classmethod
    def dump_model(cls, instance: "BaseModel") -> dict[str, Any]:
        """Dump Pydantic model → dict via model_dump()."""
        return instance.model_dump()

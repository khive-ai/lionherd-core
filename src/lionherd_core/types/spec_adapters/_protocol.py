# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

if TYPE_CHECKING:
    from lionherd_core.types.operable import Operable
    from lionherd_core.types.spec import Spec

__all__ = ("SpecAdapter",)

M = TypeVar("M")  # Model instance type
JSONLike: TypeAlias = dict[str, Any] | list[Any] | str | int | float | bool | None


class SpecAdapter(ABC, Generic[M]):
    """Abstract adapter: Spec → framework-specific formats. Implement create_field, create_model, validate_model, dump_model."""

    # ---- Abstract Methods ----

    @classmethod
    @abstractmethod
    def create_field(cls, spec: "Spec") -> Any:
        """Convert Spec to framework field (FieldInfo, Attribute, etc.)."""
        ...

    @classmethod
    @abstractmethod
    def create_model(
        cls,
        operable: "Operable",
        model_name: str,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        **kwargs: Any,
    ) -> type[M]:
        """Generate model class from Operable. Args: operable, model_name, include/exclude, framework kwargs."""
        ...

    @classmethod
    @abstractmethod
    def validate_model(cls, model_cls: type[M], data: dict) -> M:
        """Validate dict → model instance. Framework-specific (Pydantic: model_validate, attrs/dataclasses: __init__)."""
        ...

    @classmethod
    @abstractmethod
    def dump_model(cls, instance: M) -> dict[str, Any]:
        """Dump model → dict. Framework-specific (Pydantic: model_dump, attrs: asdict, dataclasses: asdict)."""
        ...

    @classmethod
    def create_validator(cls, spec: "Spec") -> Any:
        """Generate framework-specific validators from Spec metadata. Returns None if not supported."""
        return None

    # ---- Concrete Methods (Shared) ----

    @classmethod
    def parse_json(cls, text: str, fuzzy: bool = True) -> JSONLike:
        """Extract/parse JSON from text. Args: text, fuzzy (markdown extraction)."""
        from lionherd_core.libs.string_handlers import extract_json

        data = extract_json(text, fuzzy_parse=fuzzy)

        # Unwrap single-item lists/tuples
        if isinstance(data, list | tuple) and len(data) == 1:
            data = data[0]

        return data

    @classmethod
    @abstractmethod
    def fuzzy_match_fields(cls, data: dict, model_cls: type[M], strict: bool = False) -> dict:
        """Match data keys to model fields (fuzzy). Framework-specific. Args: data, model_cls, strict."""
        ...

    @classmethod
    def validate_response(
        cls,
        text: str,
        model_cls: type[M],
        strict: bool = False,
        fuzzy_parse: bool = True,
    ) -> M | None:
        """Validate text → model. Pipeline: parse_json → fuzzy_match → validate. Returns None on fail (strict=False)."""
        try:
            # Step 1: Parse JSON
            data = cls.parse_json(text, fuzzy=fuzzy_parse)

            # Step 2: Fuzzy match fields
            matched_data = cls.fuzzy_match_fields(data, model_cls, strict=strict)

            # Step 3: Validate with framework-specific method
            instance = cls.validate_model(model_cls, matched_data)

            return instance

        except (ValueError, TypeError, KeyError, AttributeError):
            # Catch validation-related exceptions only
            # ValueError: JSON/parsing errors, validation failures
            # TypeError: Type mismatches during validation
            # KeyError: Missing required fields
            # AttributeError: Field access errors
            if strict:
                raise
            return None

    @classmethod
    def update_model(
        cls,
        instance: M,
        updates: dict,
        model_cls: type[M] | None = None,
    ) -> M:
        """Update model with new data. Merges existing + updates, returns new validated instance."""
        model_cls = model_cls or type(instance)  # type: ignore[assignment]

        # Merge existing data with updates
        current_data = cls.dump_model(instance)
        current_data.update(updates)

        # Validate merged data
        return cls.validate_model(model_cls, current_data)

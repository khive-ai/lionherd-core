# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from lionherd_core.protocols import Allowable, Hashable, implements

from ._sentinel import MaybeUnset, Unset

if TYPE_CHECKING:
    from .spec import Spec

__all__ = ("Operable",)


@implements(Hashable, Allowable)
@dataclass(frozen=True, slots=True, init=False)
class Operable:
    """Ordered Spec collection for model generation. Validates uniqueness, no duplicates."""

    __op_fields__: tuple[Spec, ...]
    name: str | None

    def __init__(
        self,
        specs: tuple[Spec, ...] | list[Spec] = (),
        *,
        name: str | None = None,
    ):
        """Init with specs. Raises: TypeError (non-Spec), ValueError (duplicate names)."""
        # Import here to avoid circular import
        from .spec import Spec

        # Convert to tuple if list
        if isinstance(specs, list):
            specs = tuple(specs)

        # Validate all items are Spec objects
        for i, item in enumerate(specs):
            if not isinstance(item, Spec):
                raise TypeError(
                    f"All specs must be Spec objects, got {type(item).__name__} at index {i}"
                )

        # Check for duplicate names
        names = [s.name for s in specs if s.name is not None]
        if len(names) != len(set(names)):
            from collections import Counter

            duplicates = [name for name, count in Counter(names).items() if count > 1]
            raise ValueError(
                f"Duplicate field names found: {duplicates}. Each spec must have a unique name."
            )

        object.__setattr__(self, "__op_fields__", specs)
        object.__setattr__(self, "name", name)

    def allowed(self) -> set[str]:
        """Get set of allowed field names from specs."""
        return {i.name for i in self.__op_fields__}  # type: ignore[misc]

    def check_allowed(self, *args, as_boolean: bool = False):
        """Check field names allowed. Args: field names, as_boolean. Raises ValueError if not allowed."""
        if not set(args).issubset(self.allowed()):
            if as_boolean:
                return False
            raise ValueError(
                f"Some specified fields are not allowed: {set(args).difference(self.allowed())}"
            )
        return True

    def get(self, key: str, /, default=Unset) -> MaybeUnset[Spec]:
        """Get Spec by field name. Returns default if not found."""
        if not self.check_allowed(key, as_boolean=True):
            return default
        for i in self.__op_fields__:
            if i.name == key:
                return i

    def get_specs(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[Spec, ...]:
        """Get filtered Specs. Args: include/exclude sets. Raises ValueError if both or invalid names."""
        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude")

        if include:
            if self.check_allowed(*include, as_boolean=True) is False:
                raise ValueError(
                    "Some specified fields are not allowed: "
                    f"{set(include).difference(self.allowed())}"
                )
            return tuple(self.get(i) for i in include if self.get(i) is not Unset)  # type: ignore[misc]

        if exclude:
            _discards = {self.get(i) for i in exclude if self.get(i) is not Unset}
            return tuple(s for s in self.__op_fields__ if s not in _discards)

        return self.__op_fields__

    def create_model(
        self,
        adapter: Literal["pydantic"] = "pydantic",
        model_name: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        **kw,
    ):
        """Create framework model from specs. Args: adapter, model_name, include/exclude. Raises: ImportError, ValueError."""
        match adapter:
            case "pydantic":
                try:
                    from .spec_adapters.pydantic_field import PydanticSpecAdapter
                except ImportError as e:
                    raise ImportError(
                        "PydanticSpecAdapter requires Pydantic. Install with: pip install pydantic"
                    ) from e

                kws = {
                    "model_name": model_name or self.name or "DynamicModel",
                    "include": include,
                    "exclude": exclude,
                    **kw,
                }
                return PydanticSpecAdapter.create_model(self, **kws)
            case _:
                raise ValueError(f"Unsupported adapter: {adapter}")

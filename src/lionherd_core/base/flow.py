# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import Field, PrivateAttr, field_validator

from ..protocols import Serializable, implements
from ._utils import extract_types
from .element import Element
from .pile import Pile
from .progression import Progression

__all__ = ("Flow",)

E = TypeVar("E", bound=Element)  # Element type for items
P = TypeVar("P", bound=Progression)  # Progression type


@implements(Serializable)
class Flow(Element, Generic[E, P]):
    """Workflow state machine with ordered progressions and referenced items.

    Flow uses composition: two Pile instances for clear separation.
    - progressions: Named sequences of item UUIDs (workflow stages)
    - items: Referenced elements (Nodes, Agents, etc.)

    Generic Parameters:
        E: Element type for items
        P: Progression type
    """

    name: str | None = Field(
        default=None,
        description="Optional name for this flow (e.g., 'task_workflow')",
    )
    progressions: Pile[P] = Field(
        default_factory=Pile,
        description="Workflow stages as named progressions",
    )
    items: Pile[E] = Field(
        default_factory=Pile,
        description="Items that progressions reference",
    )
    _progression_names: dict[str, UUID] = PrivateAttr(default_factory=dict)

    @field_validator("items", "progressions", mode="wrap")
    @classmethod
    def _validate_piles(cls, v: Any, handler: Any) -> Any:
        """Convert dict to Pile during deserialization."""
        if isinstance(v, dict):
            return Pile.from_dict(v)
        # Let Pydantic handle it
        return handler(v)

    def model_post_init(self, __context: Any) -> None:
        """Rebuild _progression_names index after deserialization."""
        super().model_post_init(__context)
        # Rebuild name index from progressions
        for progression in self.progressions:
            if progression.name:
                self._progression_names[progression.name] = progression.id

    def __init__(
        self,
        items: list[E] | None = None,
        name: str | None = None,
        item_type: type[E] | set[type] | list[type] | None = None,
        strict_type: bool = False,
        **data,
    ):
        """Initialize Flow with optional items and type validation.

        Args:
            items: Initial items to add to items pile
            name: Flow name
            item_type: Type(s) for validation
            strict_type: Enforce exact type match (no subclasses)
            **data: Additional Element fields
        """
        # Let Pydantic create default piles, then populate
        super().__init__(name=name, **data)

        # Normalize item_type to set and extract types from unions
        if item_type is not None:
            item_type = extract_types(item_type)

        # Set item_type and strict_type on items pile if provided
        if item_type:
            self.items.item_type = item_type
        if strict_type:
            self.items.strict_type = strict_type

        # Add items after initialization (only if items is a list, not during deserialization)
        if items and isinstance(items, list):
            for item in items:
                self.items.add(item)

    # ==================== Progression Management ====================

    def add_progression(self, progression: P) -> None:
        """Add progression with name registration. Raises ValueError if UUID or name exists."""
        # Check name uniqueness
        if progression.name and progression.name in self._progression_names:
            raise ValueError(
                f"Progression with name '{progression.name}' already exists. Names must be unique."
            )

        # Add to progressions pile
        self.progressions.add(progression)

        # Register name if present
        if progression.name:
            self._progression_names[progression.name] = progression.id

    def remove_progression(self, progression_id: UUID | str | P) -> P:
        """Remove progression by UUID or name. Raises ValueError if not found."""
        # Resolve name to UUID if needed
        if isinstance(progression_id, str) and progression_id in self._progression_names:
            uid = self._progression_names[progression_id]
            del self._progression_names[progression_id]
            return self.progressions.remove(uid)

        # Convert to UUID for type-safe removal
        from ._utils import to_uuid

        uid = to_uuid(progression_id)
        prog: P = self.progressions[uid]

        if prog.name and prog.name in self._progression_names:
            del self._progression_names[prog.name]
        return self.progressions.remove(uid)

    def get_progression(self, key: UUID | str | P) -> P:
        """Get progression by UUID or name. Raises KeyError if not found."""
        if isinstance(key, str):
            # Check name index first
            if key in self._progression_names:
                uid = self._progression_names[key]
                return self.progressions[uid]

            # Try parsing as UUID string
            from ._utils import to_uuid

            try:
                uid = to_uuid(key)
                return self.progressions[uid]
            except (ValueError, TypeError):
                raise KeyError(f"Progression '{key}' not found in flow")

        # UUID or Progression instance
        return self.progressions[key]

    # ==================== Item Management ====================

    def add_item(
        self,
        item: E,
        progression_ids: list[UUID | str] | UUID | str | None = None,
    ) -> None:
        """Add item to items pile and optionally to progressions. Raises ValueError if exists."""
        # Add to items pile
        self.items.add(item)

        # Add to specified progressions
        if progression_ids is not None:
            # Normalize to list
            ids = [progression_ids] if not isinstance(progression_ids, list) else progression_ids

            for prog_id in ids:
                progression = self.get_progression(prog_id)
                progression.append(item)

    def remove_item(
        self,
        item_id: UUID | str | Element,
        remove_from_progressions: bool = True,
    ) -> E:
        """Remove item from items pile and optionally from progressions. Raises ValueError if not found."""
        from ._utils import to_uuid

        uid = to_uuid(item_id)

        # Remove from progressions first
        if remove_from_progressions:
            for progression in self.progressions:
                if uid in progression:
                    progression.remove(uid)

        # Remove from items pile
        return self.items.remove(uid)

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Flow(items={len(self.items)}, progressions={len(self.progressions)}{name_str})"

    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
        meta_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize Flow with proper Pile serialization for items and progressions.

        Overrides Element.to_dict() to ensure Pile fields are properly serialized
        with their items, not just metadata.
        """
        # Exclude items and progressions from parent serialization
        exclude = kwargs.pop("exclude", set())
        if isinstance(exclude, set):
            exclude = exclude | {"items", "progressions"}
        else:
            exclude = set(exclude) | {"items", "progressions"}

        # Get base Element serialization (without Pile fields)
        data = super().to_dict(
            mode=mode,
            created_at_format=created_at_format,
            meta_key=meta_key,
            exclude=exclude,
            **kwargs,
        )

        # Add Pile fields with their proper serialization (includes items)
        data["items"] = self.items.to_dict(
            mode=mode, created_at_format=created_at_format, meta_key=meta_key
        )
        data["progressions"] = self.progressions.to_dict(
            mode=mode, created_at_format=created_at_format, meta_key=meta_key
        )

        return data

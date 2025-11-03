# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload
from uuid import UUID

from pydantic import Field, PrivateAttr, field_validator

from ..protocols import Containable, implements
from ._utils import extract_types
from .element import Element
from .pile import Pile
from .progression import Progression

__all__ = ("Flow",)

E = TypeVar("E", bound=Element)  # Element type for items pile
P = TypeVar("P", bound=Progression)  # Progression type (Flow IS Pile[P])


@implements(Containable)
class Flow(Pile[P], Generic[E, P]):
    """Pile of progressions + pile of items for workflow state machines.

    Flow IS Pile[P] with additional items pile. Progressions reference items.

    Generic Parameters:
        E: Element type for items
        P: Progression type
    """

    name: str | None = Field(
        default=None,
        description="Optional name for this flow (e.g., 'task_workflow')",
    )
    pile: Pile[E] = Field(
        default_factory=Pile,
        description="Items that progressions reference",
    )
    _progression_names: dict[str, UUID] = PrivateAttr(default_factory=dict)

    @field_validator("pile", mode="before")
    @classmethod
    def _validate_pile(cls, v: Any) -> Any:
        """Convert dict to Pile during deserialization."""
        if isinstance(v, dict):
            return Pile.from_dict(v)
        return v

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
            items: Initial items to add to pile
            name: Flow name
            item_type: Type(s) for validation
            strict_type: Enforce exact type match (no subclasses)
            **data: Additional Element fields
        """
        # Let Pydantic create default pile, then populate it
        super().__init__(name=name, **data)

        # Normalize item_type to set and extract types from unions
        if item_type is not None:
            item_type = extract_types(item_type)

        # Set item_type and strict_type on pile if provided
        if item_type:
            self.pile.item_type = item_type
        if strict_type:
            self.pile.strict_type = strict_type

        # Add items after initialization
        if items:
            for item in items:
                self.pile.add(item)

    # ==================== Progression Management ====================
    # Flow IS Pile[P], so progression operations are inherited
    # Override add/remove to manage name index

    def add(self, progression: P) -> None:
        """Add progression with name registration. Raises ValueError if UUID or name exists."""
        # Check name uniqueness
        if progression.name and progression.name in self._progression_names:
            raise ValueError(
                f"Progression with name '{progression.name}' already exists. Names must be unique."
            )

        # Add to pile (Flow IS Pile[P])
        super().add(progression)

        # Register name if present
        if progression.name:
            self._progression_names[progression.name] = progression.id

    def remove(self, progression_id: UUID | str | P) -> P:
        """Remove progression by UUID or name. Raises ValueError if not found."""
        # Resolve name to UUID if needed
        if isinstance(progression_id, str) and progression_id in self._progression_names:
            uid = self._progression_names[progression_id]
            del self._progression_names[progression_id]
            return super().remove(uid)

        # Convert to UUID for type-safe removal
        from ._utils import to_uuid

        uid = to_uuid(progression_id)
        prog: P = super().__getitem__(uid)

        if prog.name and prog.name in self._progression_names:
            del self._progression_names[prog.name]
        return super().remove(uid)

    # ==================== Item Management ====================

    def add_item(
        self,
        item: E,
        progression_ids: list[UUID | str] | UUID | str | None = None,
    ) -> None:
        """Add item to pile and optionally to progressions. Raises ValueError if exists."""
        # Add to items pile
        self.pile.add(item)

        # Add to specified progressions
        if progression_ids is not None:
            # Normalize to list
            ids = [progression_ids] if not isinstance(progression_ids, list) else progression_ids

            for prog_id in ids:
                progression = self[prog_id]  # Flow IS Pile[P]
                progression.append(item)

    def remove_item(
        self,
        item_id: UUID | str | Element,
        remove_from_progressions: bool = True,
    ) -> E:
        """Remove item from pile and optionally from progressions. Raises ValueError if not found."""
        from ._utils import to_uuid

        uid = to_uuid(item_id)

        # Remove from progressions first (Flow IS Pile[P])
        if remove_from_progressions:
            for progression in self:
                if uid in progression:
                    progression.remove(uid)

        # Remove from items pile
        return self.pile.remove(uid)

    # ==================== Operators ====================

    @overload
    def __getitem__(self, key: UUID | str) -> P:
        """Get progression by UUID or name."""
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: Progression) -> Pile[P]:
        """Filter by progression - returns new Pile."""
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: int) -> P:
        """Get progression by index."""
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: slice) -> list[P]:
        """Get multiple progressions by slice."""
        ...  # pragma: no cover

    @overload
    def __getitem__(self, key: Callable[[P], bool]) -> Pile[P]:
        """Filter by function - returns new Pile."""
        ...  # pragma: no cover

    def __getitem__(self, key: Any) -> P | list[P] | Pile[P]:
        """Get progression by UUID/name/int/slice/callable. Raises KeyError if not found.

        Flow IS Pile[P]: flow["name"] → progression. For items: flow.pile[item_id].
        """
        # String: check name index first
        if isinstance(key, str):
            if key in self._progression_names:
                uid = self._progression_names[key]
                # Type narrowing: uid is UUID, super().__getitem__(UUID) returns P
                return super().__getitem__(uid)

            # Try parsing as UUID string
            from ._utils import to_uuid

            try:
                uid = to_uuid(key)
                # Type narrowing: uid is UUID, super().__getitem__(UUID) returns P
                return super().__getitem__(uid)
            except (ValueError, TypeError):
                raise KeyError(f"Progression '{key}' not found in flow")

        # All other cases (UUID, int, slice, Progression, callable): delegate to Pile
        return super().__getitem__(key)

    def __contains__(self, item: str | UUID | Element) -> bool:
        """Check if progression (by name/UUID) or item exists in piles."""
        # String: check name index first, then items pile
        if isinstance(item, str):
            return item in self._progression_names or item in self.pile

        # UUID/Element: check progressions (Flow IS Pile[P]) or items pile
        return super().__contains__(item) or item in self.pile

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Flow(items={len(self.pile)}, progressions={len(self)}{name_str})"

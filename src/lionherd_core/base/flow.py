# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import Field, PrivateAttr, field_validator

from ..errors import ExistsError, NotFoundError
from ..protocols import Serializable, implements
from ._utils import extract_types, synchronized
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
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @field_validator("items", "progressions", mode="wrap")
    @classmethod
    def _validate_piles(cls, v: Any, handler: Any) -> Any:
        """Convert dict to Pile during deserialization."""
        if isinstance(v, dict):
            return Pile.from_dict(v)
        # If it's already a Pile, use it directly (don't let Pydantic recreate it)
        if isinstance(v, Pile):
            return v
        # Let Pydantic handle it
        return handler(v)

    def model_post_init(self, __context: Any) -> None:
        """Rebuild _progression_names index after deserialization."""
        super().model_post_init(__context)
        # Rebuild name index from progressions
        for progression in self.progressions:
            if progression.name:
                self._progression_names[progression.name] = progression.id

    def _check_item_exists(self, item_id: UUID) -> E:
        """Verify item exists, re-raising NotFoundError with flow context.

        Args:
            item_id: Item UUID to check

        Returns:
            Item if found

        Raises:
            NotFoundError: With flow context and preserved metadata
        """
        try:
            return self.items[item_id]
        except NotFoundError as e:
            raise NotFoundError(
                f"Item {item_id} not found in flow",
                details=e.details,
                retryable=e.retryable,
                cause=e,
            )

    def _check_progression_exists(self, progression_id: UUID) -> P:
        """Verify progression exists, re-raising NotFoundError with flow context.

        Args:
            progression_id: Progression UUID to check

        Returns:
            Progression if found

        Raises:
            NotFoundError: With flow context and preserved metadata
        """
        try:
            return self.progressions[progression_id]
        except NotFoundError as e:
            raise NotFoundError(
                f"Progression {progression_id} not found in flow",
                details=e.details,
                retryable=e.retryable,
                cause=e,
            )

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
        # Let Pydantic create default piles
        super().__init__(name=name, **data)

        # Normalize item_type to set and extract types from unions
        if item_type is not None:
            item_type = extract_types(item_type)

        # If type validation is requested, create new Pile with validation
        # and replace the default one
        if item_type is not None or strict_type:
            self.items = Pile(items=items, item_type=item_type, strict_type=strict_type)
        else:
            # No type validation - add items to default Pile
            if items and isinstance(items, list):
                for item in items:
                    self.items.add(item)

    # ==================== Progression Management ====================

    @synchronized
    def add_progression(self, progression: P) -> None:
        """Add progression with name registration. Raises ExistsError if UUID or name exists."""
        # Check name uniqueness
        if progression.name and progression.name in self._progression_names:
            raise ExistsError(
                f"Progression with name '{progression.name}' already exists. Names must be unique."
            )

        # Add to progressions pile
        self.progressions.add(progression)

        # Register name if present
        if progression.name:
            self._progression_names[progression.name] = progression.id

    @synchronized
    def remove_progression(self, progression_id: UUID | str | P) -> P:
        """Remove progression by UUID or name. Raises NotFoundError if not found."""
        # Resolve name to UUID if needed
        if isinstance(progression_id, str) and progression_id in self._progression_names:
            uid = self._progression_names[progression_id]
            del self._progression_names[progression_id]
            # Use helper to verify and remove (not catching - let NotFoundError bubble with context)
            prog = self._check_progression_exists(uid)
            return self.progressions.remove(uid)

        # Convert to UUID for type-safe removal
        uid = self._coerce_id(progression_id)
        prog = self._check_progression_exists(uid)

        if prog.name and prog.name in self._progression_names:
            del self._progression_names[prog.name]
        return self.progressions.remove(uid)

    @synchronized
    def get_progression(self, key: UUID | str | P) -> P:
        """Get progression by UUID or name. Raises KeyError if not found."""
        if isinstance(key, str):
            # Check name index first
            if key in self._progression_names:
                uid = self._progression_names[key]
                return self.progressions[uid]

            # Try parsing as UUID string
            try:
                uid = self._coerce_id(key)
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
        """Add item to items pile and optionally to progressions. Raises ExistsError if exists."""
        # Add to items pile (let ExistsError bubble)
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
        """Remove item from items pile and optionally from progressions. Raises NotFoundError if not found."""
        uid = self._coerce_id(item_id)

        # Verify item exists first
        self._check_item_exists(uid)

        # Remove from progressions first
        if remove_from_progressions:
            for progression in self.progressions:
                if uid in progression:
                    progression.remove(uid)

        # Remove from items pile
        return self.items.remove(uid)

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
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

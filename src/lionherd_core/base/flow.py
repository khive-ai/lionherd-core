# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import Field, PrivateAttr, field_validator, model_validator

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

    def __init__(
        self,
        items: list[E] | Pile[E] | Element | None = None,
        progressions: list[P] | Pile[P] | None = None,
        name: str | None = None,
        item_type: type[E] | set[type] | list[type] | None = None,
        strict_type: bool = False,
        **data,
    ):
        """Initialize Flow with optional items and type validation.

        Field validator handles dict/list conversion. Model validator validates
        referential integrity after construction.

        Args:
            items: Initial items (Element, list[Element], Pile, or list[dict])
            progressions: Initial progressions (Progression, list, Pile, or list[dict])
            name: Flow name
            item_type: Type(s) for validation (passed to items Pile)
            strict_type: Enforce exact type match (passed to items Pile)
            **data: Additional Element fields
        """
        # Extract and normalize item_type
        item_type = extract_types(item_type) if item_type else None

        # Handle items - create configured Pile if needed
        if isinstance(items, Pile):
            data["items"] = items
        elif isinstance(items, dict):
            # Dict from deserialization - let field validator handle it
            data["items"] = items
        elif isinstance(items, list) and items and isinstance(items[0], dict):
            # List of dicts from deserialization - let field validator handle it
            data["items"] = items
        elif items is not None or item_type is not None or strict_type:
            # Normalize to list
            if isinstance(items, Element):
                items = [items]

            # Create Pile with items and type validation (item_type/strict_type are frozen)
            # Even if items=None, create Pile if item_type/strict_type specified
            data["items"] = Pile(items=items, item_type=item_type, strict_type=strict_type)

        # Handle progressions - let field validator convert dict/list to Pile
        if progressions is not None:
            data["progressions"] = progressions

        if name is not None:
            data["name"] = name

        super().__init__(**data)

    @field_validator("items", "progressions", mode="wrap")
    @classmethod
    def _validate_piles(cls, v: Any, handler: Any, info) -> Any:
        """Handle Pile, dict, and list inputs - convert to Pile as needed."""
        if isinstance(v, Pile):
            # Already a Pile from __init__, use it directly
            return v
        if isinstance(v, dict):
            # Dict from deserialization, convert to Pile
            return Pile.from_dict(v)
        if isinstance(v, list):
            # List input (can be list[Element] or list[dict]), convert to Pile
            pile = Pile()
            for item in v:
                if isinstance(item, dict):
                    pile.add(Element.from_dict(item))
                else:
                    pile.add(item)
            return pile
        # Let Pydantic handle other cases (default_factory)
        return handler(v)

    @model_validator(mode="after")
    def _validate_referential_integrity(self) -> Flow:
        """Validate that all progression UUIDs exist in items pile.

        Runs after model construction (both __init__ and deserialization).
        Uses set operations for O(1) membership checks.
        """
        item_ids = set(self.items.keys())

        # Validate each progression
        for prog in self.progressions:
            missing_ids = set(list(prog)) - item_ids
            if missing_ids:
                raise NotFoundError(
                    f"Progression '{prog.name}' contains UUIDs not in items pile: {missing_ids}"
                )

        return self

    def model_post_init(self, __context: Any) -> None:
        """Rebuild _progression_names index after deserialization."""
        super().model_post_init(__context)
        # Rebuild name index from progressions
        for progression in self.progressions:
            if progression.name:
                self._progression_names[progression.name] = progression.id

    def _check_item_exists(self, item_id: UUID) -> E:
        """Verify item exists, re-raising NotFoundError with flow context."""
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
        """Verify progression exists, re-raising NotFoundError with flow context."""
        try:
            return self.progressions[progression_id]
        except NotFoundError as e:
            raise NotFoundError(
                f"Progression {progression_id} not found in flow",
                details=e.details,
                retryable=e.retryable,
                cause=e,
            )

    # ==================== Progression Management ====================

    @synchronized
    def add_progression(self, progression: P) -> None:
        """Add progression with name registration. Raises ExistsError if UUID or name exists."""
        # Check name uniqueness
        if progression.name and progression.name in self._progression_names:
            raise ExistsError(
                f"Progression with name '{progression.name}' already exists. Names must be unique."
            )

        # Validate referential integrity before adding to pile
        item_ids = set(self.items.keys())
        missing_ids = set(list(progression)) - item_ids
        if missing_ids:
            raise NotFoundError(
                f"Progression '{progression.name or progression.id}' contains UUIDs not in items pile: {missing_ids}"
            )

        # Add to progressions pile (safe - validation passed)
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
        progressions: list[UUID | str] | UUID | str | None = None,
    ) -> None:
        """Add item to items pile and optionally to progressions.

        Args:
            item: Item to add
            progressions: Progression ID(s) or name(s) to add item to

        Raises:
            ExistsError: If item already exists
        """
        # Add to items pile (let ExistsError bubble)
        self.items.add(item)

        # Add to specified progressions
        if progressions is not None:
            # Normalize to list - treat string/UUID as single value, convert other iterables
            ids = [progressions] if isinstance(progressions, (str, UUID)) else list(progressions)

            for prog_id in ids:
                progression = self.get_progression(prog_id)
                progression.append(item)

    def remove_item(self, item_id: UUID | str | Element) -> E:
        """Remove item from items pile and all progressions.

        Args:
            item_id: Item ID, UUID string, or Element instance

        Returns:
            Removed item

        Raises:
            NotFoundError: If item not found in pile
        """
        uid = self._coerce_id(item_id)

        # Remove from all progressions first
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

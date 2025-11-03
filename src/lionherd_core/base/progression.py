# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, overload
from uuid import UUID

from pydantic import Field, field_validator

from ..protocols import Containable, implements
from .element import Element

__all__ = ("Progression",)


@implements(Containable)
class Progression(Element):
    """Ordered sequence of UUIDs with Element identity.

    Attributes:
        name: Optional progression name
        order: Ordered UUIDs (allows duplicates)

    Supports list-like operations (append/insert/remove/pop/extend), reordering (move/swap/reverse),
    and idempotent set-like operations (include/exclude).
    """

    name: str | None = Field(
        default=None,
        description="Optional name for this progression (e.g., 'execution_order')",
    )
    order: list[UUID] = Field(
        default_factory=list,
        description="Ordered sequence of UUIDs",
    )

    def __init__(
        self, order: list[UUID] | list[Element] | None = None, name: str | None = None, **data
    ):
        """Initialize Progression.

        Args:
            order: Initial items (UUIDs or Elements)
            name: Optional name for this progression
            **data: Additional Element fields
        """
        # Convert Elements to UUIDs
        if order:
            from ._utils import to_uuid

            order = [to_uuid(item) for item in order]

        # Pass all field values through **kwargs to satisfy mypy
        super().__init__(**{"name": name, "order": order or [], **data})

    @field_validator("order", mode="before")
    @classmethod
    def _validate_order(cls, value: Any) -> list[UUID]:
        """Validate and coerce order field."""
        if value is None:
            return []

        from ._utils import to_uuid

        if not isinstance(value, list):
            value = [value]

        result = []
        for item in value:
            try:
                result.append(to_uuid(item))
            except Exception:
                # Skip invalid items
                continue

        return result

    # ==================== Core Operations ====================

    def append(self, item_id: UUID | Element) -> None:
        """Add item to end of progression."""
        from ._utils import to_uuid

        uid = to_uuid(item_id)
        self.order.append(uid)

    def insert(self, index: int, item_id: UUID | Element) -> None:
        """Insert item at specific position."""
        from ._utils import to_uuid

        uid = to_uuid(item_id)
        self.order.insert(index, uid)

    def remove(self, item_id: UUID | Element) -> None:
        """Remove first occurrence of item from progression."""
        from ._utils import to_uuid

        uid = to_uuid(item_id)
        self.order.remove(uid)

    def pop(self, index: int = -1) -> UUID:
        """Remove and return item at index."""
        return self.order.pop(index)

    def popleft(self) -> UUID:
        """Remove and return first item (queue behavior)."""
        if not self.order:
            raise IndexError("Progression is empty")
        return self.order.pop(0)

    def clear(self) -> None:
        """Remove all items from progression."""
        self.order.clear()

    def extend(self, items: list[UUID | Element]) -> None:
        """Extend progression with multiple items."""
        from ._utils import to_uuid

        for item in items:
            uid = to_uuid(item)
            self.order.append(uid)

    # ==================== Query Operations ====================

    def __contains__(self, item: UUID | Element) -> bool:
        """Check if item is in progression."""
        from ._utils import to_uuid

        try:
            uid = to_uuid(item)
            return uid in self.order
        except Exception:
            return False

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.order)

    def __iter__(self):
        """Iterate over UUIDs in order."""
        return iter(self.order)

    @overload
    def __getitem__(self, index: int) -> UUID:
        """Get single item by index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> list[UUID]:
        """Get multiple items by slice."""
        ...

    def __getitem__(self, index: int | slice) -> UUID | list[UUID]:
        """Get item(s) by index."""
        return self.order[index]

    def __setitem__(self, index: int | slice, value: UUID | Element | list) -> None:
        """Set item(s) at index."""
        from ._utils import to_uuid

        if isinstance(index, slice):
            # Type guard: ensure value is a list when using slice
            if not isinstance(value, list):
                raise TypeError(f"Cannot assign {type(value).__name__} to slice, expected list")
            self.order[index] = [to_uuid(v) for v in value]
        else:
            self.order[index] = to_uuid(value)

    def index(self, item_id: UUID | Element) -> int:
        """Get index of item in progression."""
        from ._utils import to_uuid

        uid = to_uuid(item_id)
        return self.order.index(uid)

    def __reversed__(self):
        """Iterate over UUIDs in reverse order."""
        return reversed(self.order)

    def _validate_index(self, index: int, allow_end: bool = False) -> int:
        """Validate and normalize index (supports negative). Raises IndexError if out of bounds."""
        length = len(self.order)
        if length == 0 and not allow_end:
            raise IndexError("Progression is empty")

        # Normalize negative indices
        if index < 0:
            index = length + index

        # Check bounds
        max_index = length if allow_end else length - 1
        if index < 0 or index > max_index:
            raise IndexError(f"Index {index} out of range for progression of length {length}")

        return index

    # ==================== Workflow Operations ====================

    def move(self, from_index: int, to_index: int) -> None:
        """Move item from one position to another.

        Args:
            from_index: Current position (supports negative indexing)
            to_index: Target position (supports negative indexing)
        """
        from_index = self._validate_index(from_index)
        # For to_index, allow insertion at end
        to_index = self._validate_index(to_index, allow_end=True)

        item = self.order.pop(from_index)
        # Adjust to_index if we removed item before it
        if from_index < to_index:
            to_index -= 1
        self.order.insert(to_index, item)

    def swap(self, index1: int, index2: int) -> None:
        """Swap two items by index.

        Args:
            index1: First position (supports negative indexing)
            index2: Second position (supports negative indexing)
        """
        index1 = self._validate_index(index1)
        index2 = self._validate_index(index2)

        self.order[index1], self.order[index2] = self.order[index2], self.order[index1]

    def reverse(self) -> None:
        """Reverse the progression in-place."""
        self.order.reverse()

    # ==================== Set-like Operations ====================

    def include(self, item: UUID | Element) -> bool:
        """Include item in progression (idempotent).

        Returns:
            bool: True if item was added, False if already present
        """
        from ._utils import to_uuid

        uid = to_uuid(item)
        if uid not in self.order:
            self.order.append(uid)
            return True
        return False

    def exclude(self, item: UUID | Element) -> bool:
        """Exclude item from progression (idempotent).

        Returns:
            bool: True if item was removed, False if not present
        """
        from ._utils import to_uuid

        uid = to_uuid(item)
        if uid in self.order:
            self.order.remove(uid)
            return True
        return False

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Progression(len={len(self)}{name_str})"

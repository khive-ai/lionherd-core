# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Any, overload
from uuid import UUID

from pydantic import Field, field_validator

from ..errors import NotFoundError
from ..protocols import Containable, implements
from .element import Element

__all__ = ("Progression",)


@implements(Containable)
class Progression(Element):
    """Ordered sequence of UUIDs with Element identity.

    Attributes:
        name: Optional progression name
        order: Ordered sequence of UUIDs (allows duplicates)
    """

    name: str | None = Field(
        default=None,
        description="Optional name for this progression (e.g., 'execution_order')",
    )
    order: list[UUID] = Field(
        default_factory=list,
        description="Ordered sequence of UUIDs",
    )

    @field_validator("order", mode="before")
    @classmethod
    def _validate_order(cls, value: Any) -> list[UUID]:
        """Validate and coerce order field.

        Handles:
        - None -> []
        - Single UUID/str/Element -> [coerced_uuid]
        - List/tuple/iterable -> [coerced_uuid, ...]

        All items are coerced to UUID via _coerce_id.
        """
        if value is None:
            return []

        # Normalize single values to list
        if isinstance(value, (UUID, str, Element)):
            value = [value]
        elif not isinstance(value, list):
            value = list(value)

        # Coerce all items to UUIDs (let coercion errors raise)
        return [cls._coerce_id(item) for item in value]

    # ==================== Core Operations ====================

    def append(self, item_id: UUID | Element) -> None:
        """Add item to end of progression."""
        uid = self._coerce_id(item_id)
        self.order.append(uid)

    def insert(self, index: int, item_id: UUID | Element) -> None:
        """Insert item at specific position."""
        uid = self._coerce_id(item_id)
        self.order.insert(index, uid)

    def remove(self, item_id: UUID | Element) -> None:
        """Remove first occurrence of item from progression."""
        uid = self._coerce_id(item_id)
        self.order.remove(uid)

    def pop(self, index: int = -1, default: Any = ...) -> UUID | Any:
        """Remove and return item at index.

        Args:
            index: Position to pop (default: -1)
            default: Return if index not found (default: raise)

        Returns:
            UUID or default

        Raises:
            NotFoundError: If index not found and no default
        """
        try:
            return self.order.pop(index)
        except IndexError as e:
            if default is ...:
                raise NotFoundError(
                    f"Index {index} not found in progression of length {len(self)}",
                    details={"index": index, "length": len(self)},
                ) from e
            return default

    def popleft(self) -> UUID:
        """Remove and return first item.

        Raises:
            NotFoundError: If progression is empty
        """
        if not self.order:
            raise NotFoundError("Cannot pop from empty progression")
        return self.order.pop(0)

    def clear(self) -> None:
        """Remove all items from progression."""
        self.order.clear()

    def extend(self, items: list[UUID | Element]) -> None:
        """Extend with multiple items (batch operation)."""
        self.order.extend(self._coerce_id(item) for item in items)

    # ==================== Query Operations ====================

    def __contains__(self, item: UUID | Element) -> bool:
        """Check if item is in progression."""
        with contextlib.suppress(Exception):
            uid = self._coerce_id(item)
            return uid in self.order
        return False

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.order)

    def __bool__(self) -> bool:
        """Return False if progression is empty, True otherwise."""
        return len(self.order) > 0

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
        if isinstance(index, slice):
            # Type guard: ensure value is a list when using slice
            if not isinstance(value, list):
                raise TypeError(f"Cannot assign {type(value).__name__} to slice, expected list")
            self.order[index] = [self._coerce_id(v) for v in value]
        else:
            self.order[index] = self._coerce_id(value)

    def index(self, item_id: UUID | Element) -> int:
        """Get index of item in progression."""
        uid = self._coerce_id(item_id)
        return self.order.index(uid)

    def __reversed__(self):
        """Iterate over UUIDs in reverse order."""
        return reversed(self.order)

    def __list__(self) -> list[UUID]:
        """Return items as list."""
        return list(self.order)

    def _validate_index(self, index: int, allow_end: bool = False) -> int:
        """Validate and normalize index.

        Args:
            index: Index to validate (supports negative)
            allow_end: Allow index == len (for insertion)

        Returns:
            Normalized index

        Raises:
            NotFoundError: If index out of bounds or progression empty
        """
        length = len(self.order)
        if length == 0 and not allow_end:
            raise NotFoundError("Progression is empty")

        # Normalize negative indices
        if index < 0:
            index = length + index

        # Check bounds
        max_index = length if allow_end else length - 1
        if index < 0 or index > max_index:
            raise NotFoundError(
                f"Index {index} out of range for progression of length {length}",
                details={"index": index, "length": length, "allow_end": allow_end},
            )

        return index

    # ==================== Workflow Operations ====================

    def move(self, from_index: int, to_index: int) -> None:
        """Move item from one position to another.

        Args:
            from_index: Current position (supports negative)
            to_index: Target position (supports negative)
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
            index1: First position (supports negative)
            index2: Second position (supports negative)
        """
        index1 = self._validate_index(index1)
        index2 = self._validate_index(index2)

        self.order[index1], self.order[index2] = self.order[index2], self.order[index1]

    def reverse(self) -> None:
        """Reverse progression in-place."""
        self.order.reverse()

    # ==================== Set-like Operations ====================

    def include(self, item: UUID | Element) -> bool:
        """Include item (idempotent).

        Returns:
            True if added, False if already present
        """
        uid = self._coerce_id(item)
        if uid not in self.order:
            self.order.append(uid)
            return True
        return False

    def exclude(self, item: UUID | Element) -> bool:
        """Exclude item (idempotent).

        Returns:
            True if removed, False if not present
        """
        uid = self._coerce_id(item)
        if uid in self.order:
            self.order.remove(uid)
            return True
        return False

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Progression(len={len(self)}{name_str})"

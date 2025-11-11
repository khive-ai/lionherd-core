# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, TypeVar, overload
from uuid import UUID

from pydantic import Field, PrivateAttr, field_serializer, field_validator
from pydapter import (
    Adaptable as PydapterAdaptable,
    AsyncAdaptable as PydapterAsyncAdaptable,
)
from typing_extensions import override

from ..errors import ExistsError, NotFoundError
from ..libs.concurrency import Lock as AsyncLock
from ..protocols import (
    Adaptable,
    AdapterRegisterable,
    AsyncAdaptable,
    AsyncAdapterRegisterable,
    Containable,
    Deserializable,
    Serializable,
    implements,
)
from ._utils import (
    extract_types,
    load_type_from_string,
    synchronized,
)
from .element import Element
from .progression import Progression

__all__ = ("Pile",)

T = TypeVar("T", bound=Element)


@implements(
    Containable,
    Adaptable,
    AdapterRegisterable,
    AsyncAdaptable,
    AsyncAdapterRegisterable,
    Serializable,
    Deserializable,
)
class Pile(Element, PydapterAdaptable, PydapterAsyncAdaptable, Generic[T]):
    """Thread-safe typed collection with rich query interface.

    Type-dispatched __getitem__: pile[uuid], pile[int/slice], pile[progression], pile[callable].

    Args:
        items: Initial items
        item_type: Type(s) for validation (single/set/list/Union)
        strict_type: Enforce exact type match (no subclasses)

    Adapter Registration (Rust-like isolated pattern):
        Each Pile subclass has its own independent adapter registry. No auto-registration.
        Must explicitly register adapters on each class that needs them:

        ```python
        from pydapter.adapters import TomlAdapter


        class CustomPile(Pile):
            pass


        # Must register explicitly (no inheritance from parent)
        CustomPile.register_adapter(TomlAdapter)
        custom_pile.adapt_to("toml")  # Now works
        ```

        This prevents adapter pollution and ensures explicit control per class.
    """

    # Private internal state - excluded from serialization
    _items: dict[UUID, T] = PrivateAttr(default_factory=dict)
    _progression: Progression = PrivateAttr(default_factory=Progression)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    _async_lock: AsyncLock = PrivateAttr(default_factory=AsyncLock)

    # Properties for internal access (return immutable views)
    @property
    def progression(self) -> Progression:
        """Progression order as read-only copy."""
        # Return copy to prevent external modification
        return Progression(order=list(self._progression.order), name=self._progression.name)

    # Type validation config
    item_type: set[type] | None = Field(
        default=None,
        frozen=True,
        description="Set of allowed types for validation (None = any Element subclass)",
    )
    strict_type: bool = Field(
        default=False,
        frozen=True,
        description="If True, enforce exact type match (no subclasses allowed)",
    )

    @field_validator("item_type", mode="before")
    @classmethod
    def _normalize_item_type(cls, v: Any) -> set[type] | None:
        """Normalize item_type to set[type] (handles deserialization and runtime Union/list/set)."""
        if v is None:
            return None

        # Deserialization case: ["module.ClassName", ...] → {type, ...}
        if isinstance(v, list) and v and isinstance(v[0], str):
            return {load_type_from_string(type_str) for type_str in v}

        # Runtime case: Union[A, B] | [A, B] | {A, B} | A → {A, B}
        return extract_types(v)

    @override
    def __init__(
        self,
        items: list[T] | None = None,
        item_type: type[T] | set[type] | list[type] | None = None,
        order: list[UUID] | Progression | None = None,
        strict_type: bool = False,
        **kwargs,
    ):
        """Initialize Pile with optional items.

        Args:
            items: Initial items to add to the pile
            item_type: Type(s) for validation (single type, set, list, or Union)
            order: Order of items (list of UUIDs or Progression instance)
            strict_type: If True, enforce exact type match (no subclasses)
            **kwargs: Additional Element fields (id, created_at, metadata, etc.)
        """
        # Initialize Pydantic model with fields (pass through **kwargs for mypy)
        super().__init__(**{"item_type": item_type, "strict_type": strict_type, **kwargs})

        # Add items after initialization (uses _items PrivateAttr)
        if items:
            for item in items:
                self.add(item)

        # Set custom order if provided (overrides insertion order)
        if order:
            order_list = list(order.order) if isinstance(order, Progression) else order

            # Validate that all UUIDs in order are in items
            for uid in order_list:
                if uid not in self._items:
                    raise NotFoundError(f"UUID {uid} in order not found in items")
            # Set progression order
            self._progression = Progression(order=order_list)

    # ==================== Serialization ====================

    @field_serializer("item_type")
    def _serialize_item_type(self, v: set[type] | None) -> list[str] | None:
        """Serialize item_type set to list of module paths."""
        if v is None:
            return None
        return [f"{t.__module__}.{t.__name__}" for t in v]

    @override
    def to_dict(
        self,
        mode: Literal["python", "json", "db"] = "python",
        created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
        meta_key: str | None = None,
        item_meta_key: str | None = None,
        item_created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize pile with items in progression order.

        Args:
            mode: python/json/db
            created_at_format: Timestamp format for Pile
            meta_key: Rename Pile metadata field
            item_meta_key: Pass to each item's to_dict for metadata renaming
            item_created_at_format: Pass to each item's to_dict for timestamp format
            **kwargs: Passed to model_dump()
        """
        # Get base Element serialization (will handle meta_key renaming)
        data = super().to_dict(
            mode=mode, created_at_format=created_at_format, meta_key=meta_key, **kwargs
        )

        # Determine the actual metadata key name in the output
        # (will be renamed if meta_key was specified, or "node_metadata" for db mode)
        actual_meta_key = (
            meta_key if meta_key else ("node_metadata" if mode == "db" else "metadata")
        )

        # Store progression metadata in pile's metadata
        if self._progression.name and actual_meta_key in data:
            if data[actual_meta_key] is None:
                data[actual_meta_key] = {}
            data[actual_meta_key]["progression_name"] = self._progression.name

        # Serialize items in progression order (progression order is implicit)
        if mode == "python":
            # Python mode: keep objects as-is
            data["items"] = [
                i.to_dict(
                    mode="python",
                    meta_key=item_meta_key,
                    created_at_format=item_created_at_format,
                )
                for i in self
            ]
        else:
            # JSON/DB mode: convert to JSON-safe
            data["items"] = [
                i.to_dict(
                    mode="json",
                    meta_key=item_meta_key,
                    created_at_format=item_created_at_format,
                )
                for i in self
            ]

        return data

    # ==================== Core Operations ====================

    @synchronized
    def add(self, item: T) -> None:
        """Add item to pile.

        Args:
            item: Element to add

        Raises:
            ExistsError: If item with same ID already exists
            TypeError: If item type not allowed (when item_type set)
        """
        self._validate_type(item)

        if item.id in self._items:
            raise ExistsError(f"Item {item.id} already exists in pile")

        self._items[item.id] = item
        self._progression.append(item.id)

    @synchronized
    def remove(self, item_id: UUID | str | Element) -> T:
        """Remove item from pile.

        Args:
            item_id: Item ID or Element instance

        Returns:
            Removed item

        Raises:
            NotFoundError: If item not found
        """
        uid = self._coerce_id(item_id)

        try:
            item = self._items.pop(uid)
        except KeyError:
            raise NotFoundError(f"Item {uid} not found in pile") from None

        self._progression.remove(uid)
        return item

    @synchronized
    def pop(self, item_id: UUID | str | Element, default: Any = ...) -> T | Any:
        """Remove and return item from pile with optional default.

        Args:
            item_id: Item ID or Element instance
            default: Default value if not found (default: raise NotFoundError)

        Returns:
            Removed item or default

        Raises:
            NotFoundError: If item not found and no default provided
        """
        uid = self._coerce_id(item_id)

        try:
            item = self._items.pop(uid)
            self._progression.remove(uid)
            return item
        except KeyError:
            if default is ...:
                raise NotFoundError(f"Item {uid} not found in pile") from None
            return default

    @synchronized
    def get(self, item_id: UUID | str | Element, default: Any = ...) -> T | Any:
        """Get item by ID with optional default.

        Args:
            item_id: Item ID or Element instance
            default: Default value if not found

        Returns:
            Item or default

        Raises:
            NotFoundError: If item not found and no default
        """
        uid = self._coerce_id(item_id)

        try:
            return self._items[uid]
        except KeyError:
            if default is ...:
                raise NotFoundError(f"Item {uid} not found in pile") from None
            return default

    @synchronized
    def update(self, item: T) -> None:
        """Update existing item.

        Args:
            item: Updated item (must have same ID)

        Raises:
            NotFoundError: If item not found
            TypeError: If item type not allowed (when item_type set)
        """
        self._validate_type(item)

        if item.id not in self._items:
            raise NotFoundError(f"Item {item.id} not found in pile")

        self._items[item.id] = item

    @synchronized
    def clear(self) -> None:
        """Remove all items."""
        self._items.clear()
        self._progression.clear()

    # ==================== Set-like Operations ====================

    def include(self, item: T) -> bool:
        """Include item in pile (idempotent).

        Returns:
            True if item IS in pile (membership guaranteed).
            False only if validation fails.
            Use pattern: `if pile.include(x): ...` guarantees x is in pile.
        """
        with contextlib.suppress(Exception):
            if item.id not in self._items:
                self.add(item)
            return True
        return False

    def exclude(self, item: UUID | str | Element) -> bool:
        """Exclude item from pile (idempotent).

        Returns:
            True if item IS NOT in pile (absence guaranteed).
            False only if ID coercion fails.
            Use pattern: `if pile.exclude(x): ...` guarantees x is not in pile.
        """
        with contextlib.suppress(Exception):
            uid = self._coerce_id(item)
            if uid in self._items:
                self.remove(uid)
            return True
        return False

    # ==================== Rich __getitem__ (Type Dispatch) ====================

    @overload
    def __getitem__(self, key: UUID | str) -> T:
        """Get single item by UUID or string ID."""
        ...

    @overload
    def __getitem__(self, key: Progression) -> Pile[T]:
        """Filter by progression - returns new Pile."""
        ...

    @overload
    def __getitem__(self, key: int) -> T:
        """Get item by index."""
        ...

    @overload
    def __getitem__(self, key: slice) -> list[T]:
        """Get multiple items by slice."""
        ...

    @overload
    def __getitem__(self, key: Callable[[T], bool]) -> Pile[T]:
        """Filter by function - returns new Pile."""
        ...

    def __getitem__(self, key: Any) -> T | list[T] | Pile[T]:
        """Type-dispatched query interface.

        Args:
            key: UUID/str (get by ID), Progression (filter), int/slice (index), or callable (predicate)

        Returns:
            Element, list[Element], or new Pile (depends on key type)

        Raises:
            TypeError: If key type is not supported
            NotFoundError: If item not found (UUID/str access)
            ValueError: If no items match filter or slice is empty
        """
        # Type 1: UUID/str - Get by ID
        if isinstance(key, (UUID, str)):
            return self.get(key)

        # Type 2: Progression - Filter by progression (RETURNS NEW PILE!)
        elif isinstance(key, Progression):
            return self._filter_by_progression(key)

        # Type 3: int - Index access
        elif isinstance(key, int):
            return self._get_by_index(key)

        # Type 4: slice - Multiple items
        elif isinstance(key, slice):
            return self._get_by_slice(key)

        # Type 5: callable - Filter function
        elif callable(key):
            return self._filter_by_function(key)

        else:
            raise TypeError(
                f"Invalid key type: {type(key)}. Expected UUID, Progression, int, slice, or callable"
            )

    def _filter_by_progression(self, prog: Progression) -> Pile[T]:
        """Filter pile by progression order, returns new Pile.

        Raises:
            NotFoundError: If any UUID in progression not found in pile
        """
        if any(uid not in self._items for uid in prog):
            raise NotFoundError("Some items from progression not found in pile")

        return Pile(
            items=[self._items[uid] for uid in prog],
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    @synchronized
    def _get_by_index(self, index: int) -> T:
        """Get item by index in progression order."""
        # With overloaded __getitem__, mypy knows int index returns UUID
        uid: UUID = self._progression[index]
        return self._items[uid]

    @synchronized
    def _get_by_slice(self, s: slice) -> list[T]:
        """Get multiple items by slice."""
        # With overloaded __getitem__, mypy knows slice returns list[UUID]
        uids: list[UUID] = self._progression[s]
        return [self._items[uid] for uid in uids]

    def _filter_by_function(self, func: Callable[[T], bool]) -> Pile[T]:
        """Filter pile by function - returns NEW Pile."""
        filtered_items = [item for item in self if func(item)]

        return Pile(
            items=filtered_items,
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    def filter_by_type(self, item_type: type[T] | set[type] | list[type]) -> Pile[T]:
        """Filter by type(s), returns new Pile.

        Args:
            item_type: Type(s) to filter by

        Returns:
            New Pile with filtered items

        Raises:
            TypeError: If requested type is not allowed
            ValueError: If no items of requested type exist
        """
        # Normalize to set
        types_to_filter = extract_types(item_type)

        # Check if types are allowed
        if self.item_type is not None:
            if self.strict_type:
                # Strict mode: exact type match (efficient set operation)
                invalid_types = types_to_filter - self.item_type
                if invalid_types:
                    raise TypeError(
                        f"Types {invalid_types} not allowed in pile (allowed: {self.item_type})"
                    )
            else:
                # Permissive mode: check subclass relationships
                # Build set of all compatible types (requested types + their subclasses/superclasses in allowed set)
                for t in types_to_filter:
                    is_compatible = any(
                        issubclass(t, allowed) or issubclass(allowed, t)
                        for allowed in self.item_type
                    )
                    if not is_compatible:
                        raise TypeError(
                            f"Type {t} not compatible with allowed types {self.item_type}"
                        )

        # Filter items by type(s) - iterate over self to reuse synchronized iterator
        filtered_items = [
            item for item in self if any(isinstance(item, t) for t in types_to_filter)
        ]

        # Check if any items found
        if not filtered_items:
            raise NotFoundError(f"No items of type(s) {types_to_filter} found in pile")

        return Pile(
            items=filtered_items,
            item_type=self.item_type,
            strict_type=self.strict_type,
        )

    # ==================== Context Managers ====================

    async def __aenter__(self) -> Pile[T]:
        """Acquire lock for async context manager."""
        await self._async_lock.acquire()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Release lock for async context manager."""
        self._async_lock.release()

    # ==================== Query Operations ====================

    @synchronized
    def __contains__(self, item: UUID | str | Element) -> bool:
        """Check if item exists in pile."""
        with contextlib.suppress(Exception):
            uid = self._coerce_id(item)
            return uid in self._items
        return False

    @synchronized
    def __len__(self) -> int:
        """Return number of items."""
        return len(self._items)

    def __bool__(self) -> bool:
        """Return False if pile is empty, True otherwise."""
        return len(self._items) > 0

    @synchronized
    def __iter__(self) -> Iterator[T]:
        """Iterate items in insertion order."""
        for uid in self._progression:
            yield self._items[uid]

    def keys(self) -> Iterator[UUID]:
        """Iterate over UUIDs in insertion order.

        Returns:
            Iterator of UUIDs in the order items were added.
        """
        return iter(self._progression)

    def items(self) -> Iterator[tuple[UUID, T]]:
        """Iterate over (UUID, item) pairs in insertion order.

        Returns:
            Iterator of (UUID, item) tuples in the order items were added.
        """
        for i in self:
            yield (i.id, i)

    def __list__(self) -> list[T]:
        """Return items as list in insertion order."""
        return [i for i in self]

    def is_empty(self) -> bool:
        """Check if pile is empty."""
        return len(self._items) == 0

    # ==================== Validation ====================

    def _validate_type(self, item: T) -> None:
        """Validate item type with set-based checking and strict mode."""
        if not isinstance(item, Element):
            raise TypeError(f"Item must be Element subclass, got {type(item)}")

        if self.item_type is not None:
            item_type_actual = type(item)

            if self.strict_type:
                # Strict mode: exact type match only (no subclasses)
                if item_type_actual not in self.item_type:
                    raise TypeError(
                        f"Item type {item_type_actual} not in allowed types {self.item_type} "
                        "(strict_type=True, no subclasses allowed)"
                    )
            else:
                # Permissive mode: allow subclasses
                if not any(issubclass(item_type_actual, t) for t in self.item_type):
                    raise TypeError(
                        f"Item type {item_type_actual} is not a subclass of any allowed type {self.item_type}"
                    )

    # ==================== Deserialization ====================

    @classmethod
    @override
    def from_dict(
        cls,
        data: dict[str, Any],
        meta_key: str | None = None,
        item_meta_key: str | None = None,
        **kwargs: Any,
    ) -> Pile[T]:
        """Deserialize Pile from dict.

        Args:
            data: Serialized pile data
            meta_key: If provided, rename this key back to "metadata" (for db mode deserialization)
            item_meta_key: If provided, pass to Element.from_dict for item deserialization
            **kwargs: Additional arguments, including optional item_type

        Returns:
            Reconstructed Pile
        """
        from .element import Element

        # Make a copy to avoid mutating input
        data = data.copy()

        # Restore metadata from custom key if specified (db mode deserialization)
        if meta_key and meta_key in data:
            data["metadata"] = data.pop(meta_key)

        # Extract pile configuration
        item_type_data = data.get("item_type") or kwargs.get("item_type")
        strict_type = data.get("strict_type", False)

        # FAIL FAST: Validate ALL item types before deserializing ANY items
        items_data = data.get("items", [])
        if item_type_data is not None and items_data:
            # Normalize item_type to set of types (handle serialized strings)
            if (
                isinstance(item_type_data, list)
                and item_type_data
                and isinstance(item_type_data[0], str)
            ):
                # Deserialization case: convert strings to types
                allowed_types = {load_type_from_string(type_str) for type_str in item_type_data}
            else:
                # Runtime case: use extract_types
                allowed_types = extract_types(item_type_data)

            # Validate all lion_class values upfront
            for item_dict in items_data:
                lion_class = item_dict.get("metadata", {}).get("lion_class")
                if lion_class:
                    try:
                        item_type_actual = load_type_from_string(lion_class)
                    except ValueError:
                        # Let Element.from_dict handle invalid types
                        continue

                    # Check type compatibility
                    if strict_type:
                        # Strict: exact type match
                        if item_type_actual not in allowed_types:
                            raise TypeError(
                                f"Item type {lion_class} not in allowed types {allowed_types} "
                                "(strict_type=True)"
                            )
                    else:
                        # Permissive: allow subclasses
                        if not any(issubclass(item_type_actual, t) for t in allowed_types):
                            raise TypeError(
                                f"Item type {lion_class} is not a subclass of any allowed type {allowed_types}"
                            )

        # Create pile with Element fields (id, created_at, metadata preserved)
        # Remove items/progression/item_type/strict_type from data to avoid duplication
        pile_data = data.copy()
        pile_data.pop("items", None)
        pile_data.pop("item_type", None)
        pile_data.pop("strict_type", None)
        pile = cls(item_type=item_type_data, strict_type=strict_type, **pile_data)

        # Extract and restore progression metadata
        metadata = data.get("metadata", {})
        progression_name = metadata.get("progression_name")
        if progression_name:
            pile._progression.name = progression_name

        # Deserialize items (type validation already done above)
        for item_dict in items_data:
            item = Element.from_dict(item_dict, meta_key=item_meta_key)
            pile.add(item)  # Adds to _items dict + _progression (maintains order)

        return pile

    # ==================== Adapter Methods ====================

    @classmethod
    def register_adapter(cls, adapter: Any) -> None:
        """Register adapter for this class."""
        super().register_adapter(adapter)

    @classmethod
    def register_async_adapter(cls, adapter: Any) -> None:
        """Register async adapter for this class."""
        super().register_async_adapter(adapter)

    def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Convert to external format via pydapter adapter.

        Args:
            obj_key: Adapter key (e.g., "postgres", "qdrant", "mongo")
            many: Adaptation mode:
                - False (default): Treat entire Pile as single object (export Pile + all items)
                - True: Bulk operation on items within Pile (export items individually)
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return super().adapt_to(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Pile:
        """Create from external format via pydapter adapter.

        Args:
            obj: Source object
            obj_key: Adapter key (e.g., "postgres", "qdrant", "mongo")
            many: Adaptation mode:
                - False (default): Deserialize entire Pile from single object
                - True: Bulk load items from external source (construct Pile from items)
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "from_dict")
        return super().adapt_from(obj, obj_key=obj_key, many=many, **kwargs)

    async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any:
        """Async convert to external format via pydapter async adapter.

        Args:
            obj_key: Adapter key
            many: Adaptation mode (see adapt_to for details)
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "to_dict")
        kwargs.setdefault("adapt_kw", {"mode": "db"})
        return await super().adapt_to_async(obj_key=obj_key, many=many, **kwargs)

    @classmethod
    async def adapt_from_async(
        cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any
    ) -> Pile:
        """Async create from external format via pydapter async adapter.

        Args:
            obj: Source object
            obj_key: Adapter key
            many: Adaptation mode (see adapt_from for details)
            **kwargs: Passed to adapter
        """
        kwargs.setdefault("adapt_meth", "from_dict")
        return await super().adapt_from_async(obj, obj_key=obj_key, many=many, **kwargs)

    def __repr__(self) -> str:
        return f"Pile(len={len(self)})"

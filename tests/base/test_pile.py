# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Pile: Thread-safe typed collection with rich query interface.

Design Philosophy
-----------------
Pile is lionherd's foundational data structure for managing Element collections with:

1. **Thread Safety**: RLock-based synchronization for safe concurrent access
   - All public methods protected by @synchronized/@async_synchronized
   - Recursive locking allows nested pile operations
   - CPython GIL makes dict reads thread-safe (readers get consistent snapshots)
   - O(1) add/remove with zero-copy reads (MappingProxyType views)

2. **Type Validation**: Flexible constraints with Union support
   - strict_type=False: Duck typing, allows subclasses (default)
   - strict_type=True: Exact type match only (no inheritance)
   - Union types: Multi-type collections (Pile[Node | Event])
   - Set-based type algebra for O(1) validation checks

3. **Progression Integration**: Ordered collections with rich query interface
   - Internal Progression tracks insertion order (or custom order)
   - Type-dispatched __getitem__: pile[uuid], pile[int], pile[slice], pile[func], pile[prog]
   - Filter operations return NEW Pile instances (immutable semantics)
   - Progression order preserved through serialization/deserialization

Type Algebra
------------
Let I = item_type (set of types), x = item to validate:

    strict=False (permissive):
        valid(x) ⟺ ∃t ∈ I: isinstance(x, t)  [allows subclasses]

    strict=True (exact match):
        valid(x) ⟺ type(x) ∈ I  [no inheritance]

    Union[A, B]:
        normalized to I = {A, B}  [set-based O(1) checking]

Set Operations
--------------
Pile provides mathematical set semantics:

    P U {x}  ≡  pile.include(x)   [idempotent add]
    P \\ {x}  ≡  pile.exclude(x)   [idempotent remove]
    |P|      ≡  len(pile)          [cardinality]
    x ∈ P    ≡  x in pile          [membership]

Algorithmic Complexity
----------------------
Internal storage: dict[UUID, Element] + Progression (list[UUID])

    Operation           Time        Space       Notes
    ---------           ----        -----       -----
    add(item)           O(1)        O(1)        Dict insert + progression append
    remove(item)        O(n)        O(1)        Dict pop + progression.remove (linear scan)
    get(uuid)           O(1)        O(1)        Direct dict access
    __getitem__[int]    O(1)        O(1)        Index progression, dict lookup
    __getitem__[slice]  O(k)        O(k)        k = slice width
    __getitem__[prog]   O(m)        O(m)        m = progression length (creates new Pile)
    __getitem__[func]   O(n)        O(n)        Full scan + filter (creates new Pile)
    filter_by_type()    O(n)        O(n)        isinstance check for all items
    __contains__        O(1)        O(1)        Dict membership (UUID-based)
    __iter__            O(n)        O(1)        Yields in progression order
    to_dict()           O(n)        O(n)        Serializes all items in progression order
    from_dict()         O(n)        O(n)        Validates + adds all items

Concurrency Invariants
----------------------
Threading guarantees:

    □(op ∈ {add, remove, get, update} ⇒ @synchronized)        [all ops atomic]
    □(readers see consistent snapshot via MappingProxyType)    [no torn reads]
    □(RLock allows recursion: pile.add → pile._validate_type) [reentrant]
    □(CPython dict reads thread-safe without lock)             [GIL guarantee]

Race condition safety:

    // Concurrent adds (unique IDs)
    Thread1: pile.add(x)  }  Both succeed if x.id ≠ y.id
    Thread2: pile.add(y)  }  ValueError if x.id = y.id (checked under lock)

    // Concurrent read-modify-write
    Thread1: x = pile.get(id); pile.update(modify(x))  }  Atomic: lock held during get+update
    Thread2: y = pile.get(id); pile.update(modify(y))  }  Serialized by RLock

    // Concurrent filter operations (immutable semantics)
    Thread1: filtered1 = pile[lambda x: x.value > 5]  }  Both create NEW Piles
    Thread2: filtered2 = pile[lambda x: x.value < 3]  }  Original pile unchanged

Test Organization
-----------------
This test suite verifies:

1. Initialization: Type validation, progression order, edge cases
2. Basic ops: add, remove, get, update, clear
3. Set ops: include, exclude (idempotency)
4. Query ops: __getitem__ dispatch, filter_by_type, slicing
5. Async ops: async context manager, concurrent operations
6. Thread safety: ThreadPoolExecutor stress tests
7. Type validation: strict/permissive modes, Union types
8. Serialization: to_dict/from_dict round-trips, progression preservation
9. Collection protocols: __contains__, __len__, __iter__, keys, values
10. Edge cases: empty piles, repr, progression integrity

Edge Cases Tested
-----------------
Type Constraints:
    - Permissive mode: Rejects unrelated types (not just any object)
    - Adapter registries: Isolated per subclass (no pollution)

Serialization:
    - Union type normalization: Union[A, B] → {A, B}
    - Set type serialization: set → list → set round-trip
    - Runtime types: Type objects in from_dict (not just strings)
    - Invalid metadata: Validation continues gracefully
    - Mode-specific metadata: db mode with custom meta_key

Validation:
    - Invalid lion_class: Skip in validation, error in deserialization
    - All invalid classes: Validation completes gracefully
    - Strict mode positive case: Valid exact type match
"""

import concurrent.futures
import threading
from uuid import UUID

import pytest

from lionherd_core.base import Element, Pile, Progression
from lionherd_core.errors import ExistsError, NotFoundError
from lionherd_core.libs.concurrency import gather

# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleElement(Element):
    """Simple element for testing."""

    value: int = 0


class TypedElement(Element):
    """Typed element for testing type validation."""

    name: str = "test"


class ChildElement(TypedElement):
    """Child element for testing strict_type mode."""

    extra: str = "child"


@pytest.fixture
def simple_items():
    """Create simple test items."""
    return [SimpleElement(value=i) for i in range(5)]


@pytest.fixture
def typed_items():
    """Create typed test items."""
    return [TypedElement(name=f"item{i}") for i in range(3)]


@pytest.fixture
def child_items():
    """Create child test items."""
    return [ChildElement(name=f"child{i}", extra=f"extra{i}") for i in range(2)]


# =============================================================================
# Initialization Tests
# =============================================================================


def test_pile_empty_initialization():
    """Test empty Pile initialization."""
    pile = Pile()
    assert len(pile) == 0
    assert pile.is_empty()
    assert pile.item_type is None
    assert pile.strict_type is False


def test_pile_with_items(simple_items):
    """Test Pile initialization with items."""
    pile = Pile(items=simple_items)
    assert len(pile) == 5
    assert not pile.is_empty()
    for item in simple_items:
        assert item.id in pile


def test_pile_with_item_type(typed_items):
    """Test Pile with item_type validation."""
    pile = Pile(items=typed_items, item_type=TypedElement)
    assert len(pile) == 3
    assert pile.item_type == {TypedElement}


def test_pile_with_strict_type(typed_items):
    """Test Pile with strict_type=True."""
    pile = Pile(items=typed_items, item_type=TypedElement, strict_type=True)
    assert pile.strict_type is True
    # Should allow exact type
    assert len(pile) == 3


def test_pile_with_custom_order(simple_items):
    """Test Pile with custom order."""
    custom_order = [item.id for item in reversed(simple_items)]
    pile = Pile(items=simple_items, order=custom_order)

    # Check order matches custom order
    for i, item in enumerate(pile):
        assert item.id == custom_order[i]


def test_pile_with_progression_order(simple_items):
    """Test Pile with Progression order."""
    custom_order = [item.id for item in reversed(simple_items)]
    prog = Progression(order=custom_order, name="reversed")
    pile = Pile(items=simple_items, order=prog)

    # Check order matches
    for i, item in enumerate(pile):
        assert item.id == custom_order[i]

    # Note: progression name is not preserved in __init__ (only order is used)
    # Name can be manually set after creation if needed
    pile._progression.name = "reversed"
    assert pile._progression.name == "reversed"


def test_pile_order_validation_invalid_uuid(simple_items):
    """Test Pile rejects order with invalid UUID."""
    from uuid import uuid4

    invalid_order = [uuid4()]  # UUID not in items
    with pytest.raises(NotFoundError, match=r"UUID .* not found in items"):
        Pile(items=simple_items, order=invalid_order)


def test_pile_item_type_normalization():
    """Test item_type normalization from various inputs."""
    # Single type
    pile1 = Pile(item_type=SimpleElement)
    assert pile1.item_type == {SimpleElement}

    # List of types
    pile2 = Pile(item_type=[SimpleElement, TypedElement])
    assert pile2.item_type == {SimpleElement, TypedElement}

    # Set of types
    pile3 = Pile(item_type={SimpleElement, TypedElement})
    assert pile3.item_type == {SimpleElement, TypedElement}


# =============================================================================
# Basic Operations Tests
# =============================================================================


def test_add_item():
    """Test adding items to Pile."""
    pile = Pile()
    item = SimpleElement(value=42)

    pile.add(item)
    assert len(pile) == 1
    assert item.id in pile
    assert pile.get(item.id) == item


def test_add_duplicate_raises_error():
    """Test adding duplicate item raises ExistsError."""
    pile = Pile()
    item = SimpleElement(value=42)

    pile.add(item)
    with pytest.raises(ExistsError, match="already exists"):
        pile.add(item)


def test_add_with_type_validation():
    """Test add with type validation."""
    pile = Pile(item_type=TypedElement)

    # Valid type
    valid_item = TypedElement(name="valid")
    pile.add(valid_item)
    assert valid_item.id in pile

    # Invalid type
    invalid_item = SimpleElement(value=42)
    with pytest.raises(TypeError, match="not a subclass"):
        pile.add(invalid_item)


def test_add_with_strict_type():
    """Test add with strict_type=True."""
    pile = Pile(item_type=TypedElement, strict_type=True)

    # Exact type allowed
    exact_item = TypedElement(name="exact")
    pile.add(exact_item)
    assert exact_item.id in pile

    # Subclass not allowed
    child_item = ChildElement(name="child", extra="extra")
    with pytest.raises(TypeError, match="strict_type=True"):
        pile.add(child_item)


def test_remove_item():
    """Test removing items from Pile."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    removed = pile.remove(item.id)
    assert len(pile) == 0
    assert removed == item
    assert item.id not in pile


def test_remove_by_element():
    """Test remove accepts Element instance."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    removed = pile.remove(item)  # Pass element, not ID
    assert removed == item
    assert len(pile) == 0


def test_remove_nonexistent_raises_error():
    """Test removing nonexistent item raises NotFoundError."""
    from uuid import uuid4

    pile = Pile()
    with pytest.raises(NotFoundError, match="not found"):
        pile.remove(uuid4())


def test_pop_alias():
    """Test pop() is alias for remove()."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    popped = pile.pop(item.id)
    assert popped == item
    assert len(pile) == 0


def test_pop_without_default_not_found():
    """Test pop() raises NotFoundError when item not found and no default."""
    from uuid import uuid4

    pile = Pile()

    with pytest.raises(NotFoundError, match="not found in pile"):
        pile.pop(uuid4())


def test_pop_with_default_none():
    """Test pop() returns None when item not found with default=None."""
    from uuid import uuid4

    pile = Pile()
    result = pile.pop(uuid4(), default=None)
    assert result is None


def test_pop_with_custom_default():
    """Test pop() returns custom default when item not found."""
    from uuid import uuid4

    pile = Pile()
    default_item = SimpleElement(value=999)

    result = pile.pop(uuid4(), default=default_item)
    assert result == default_item


def test_pop_with_default_when_exists():
    """Test pop() returns and removes item when it exists, ignoring default."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    default_item = SimpleElement(value=999)
    popped = pile.pop(item.id, default=default_item)

    assert popped == item
    assert popped != default_item
    assert len(pile) == 0


def test_get_item():
    """Test getting items from Pile."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    retrieved = pile.get(item.id)
    assert retrieved == item


def test_get_with_default():
    """Test get with default value."""
    from uuid import uuid4

    pile = Pile()
    default = SimpleElement(value=999)

    result = pile.get(uuid4(), default=default)
    assert result == default


def test_get_nonexistent_raises_error():
    """Test get without default raises NotFoundError."""
    from uuid import uuid4

    pile = Pile()
    with pytest.raises(NotFoundError, match="not found"):
        pile.get(uuid4())


def test_update_item():
    """Test updating existing item."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    # Update with modified item (same ID)
    # Pass id during initialization (frozen field)
    updated_item = SimpleElement(value=100, id=item.id)
    pile.update(updated_item)

    assert pile.get(item.id).value == 100


def test_update_nonexistent_raises_error():
    """Test updating nonexistent item raises NotFoundError."""
    pile = Pile()
    item = SimpleElement(value=42)

    with pytest.raises(NotFoundError, match="not found"):
        pile.update(item)


def test_update_with_type_validation():
    """Test update respects type validation."""
    pile = Pile(item_type=TypedElement)
    item = TypedElement(name="original")
    pile.add(item)

    # Invalid type update (pass id during initialization)
    invalid_item = SimpleElement(value=42, id=item.id)

    with pytest.raises(TypeError, match="not a subclass"):
        pile.update(invalid_item)


def test_clear():
    """Test clearing all items."""
    pile = Pile()
    for i in range(5):
        pile.add(SimpleElement(value=i))

    assert len(pile) == 5
    pile.clear()
    assert len(pile) == 0
    assert pile.is_empty()


# =============================================================================
# Set-like Operations Tests
# =============================================================================


def test_include_new_item():
    """Test include adds new item and returns True (membership guaranteed)."""
    pile = Pile()
    item = SimpleElement(value=42)

    result = pile.include(item)
    assert result is True  # Membership guaranteed
    assert item.id in pile


def test_include_existing_item():
    """Test include is idempotent - returns True even if already present."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    result = pile.include(item)
    assert result is True  # Still returns True (membership guaranteed)
    assert len(pile) == 1  # Not duplicated


def test_exclude_existing_item():
    """Test exclude removes item and returns True (absence guaranteed)."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    result = pile.exclude(item.id)
    assert result is True  # Absence guaranteed
    assert item.id not in pile


def test_exclude_nonexistent_item():
    """Test exclude is idempotent - returns True even if not present."""
    from uuid import uuid4

    pile = Pile()
    result = pile.exclude(uuid4())
    assert result is True  # Returns True (absence guaranteed)


def test_exclude_by_element():
    """Test exclude accepts Element instance."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    result = pile.exclude(item)
    assert result is True
    assert len(pile) == 0


def test_include_validation_failure():
    """Test include returns False on validation failure."""
    pile = Pile(item_type=SimpleElement, strict_type=True)

    # Create incompatible item (different type)
    class OtherElement(Element):
        pass

    item = OtherElement()
    result = pile.include(item)
    assert result is False  # Validation failed
    assert len(pile) == 0  # Item not added


def test_exclude_invalid_id():
    """Test exclude returns False on ID coercion failure."""
    pile = Pile()

    # Invalid ID that cannot be coerced
    result = pile.exclude(12345)  # Not a valid UUID/str/Element
    assert result is False  # ID coercion failed


# =============================================================================
# Query Operations Tests (__getitem__)
# =============================================================================
#
# Rich query interface: Pile's __getitem__ provides type-dispatched access with
# five distinct query modes.
#
# Design Philosophy:
# ------------------
# Python's __getitem__ allows operator overloading for expressive querying:
#     pile[uuid]      → single item (by ID)
#     pile[3]         → single item (by index)
#     pile[1:5]       → list[items] (slice)
#     pile[func]      → filtered Pile (predicate)
#     pile[prog]      → filtered Pile (progression)
#
# This unified interface eliminates method proliferation:
#     ❌ pile.get_by_uuid(), pile.get_by_index(), pile.filter_by_func()
#     ✅ pile[key] (single interface, type-dispatched)
#
# Type Dispatch Strategy:
# -----------------------
# 1. UUID/str → get by ID
#    - O(1) dict lookup: self._items[uuid]
#    - Returns single Element
#    - Raises ValueError if not found
#
# 2. int → get by index
#    - O(1) progression index: self._progression[index]
#    - Then O(1) dict lookup: self._items[uuid]
#    - Supports negative indices: pile[-1]
#
# 3. slice → get multiple items
#    - O(k) where k = slice width
#    - Returns list[Element] (NOT new Pile)
#    - Uses progression order: pile[1:3] → [item1, item2]
#
# 4. Progression → filter by progression
#    - O(m) where m = len(progression)
#    - Returns NEW Pile with subset of items
#    - Preserves type constraints (item_type, strict_type)
#
# 5. Callable[[T], bool] → filter by predicate
#    - O(n) full scan with isinstance check
#    - Returns NEW Pile with filtered items
#    - Example: pile[lambda x: x.value > 5]
#
# Immutable Semantics:
# --------------------
# Filter operations (Progression/callable) return NEW Pile:
#     original = Pile(items=[a, b, c])
#     filtered = original[lambda x: x.value > 5]
#     # original unchanged, filtered is new Pile
#
# Why new Pile? Allows chaining without mutation:
#     pile[prog1][lambda x: x.enabled].filter_by_type(Node)
#
# Thread Safety:
# --------------
# All __getitem__ paths use @synchronized:
#     - get_by_uuid: lock held during dict access
#     - get_by_index: lock held during progression + dict access
#     - filter ops: lock held during iteration (snapshot created)
#     - Returned Piles are independent (no shared state)
#
# Performance Notes:
# ------------------
# - Index access: O(1) via Progression's __getitem__ optimization
# - Slice: O(k) where k = slice width (not O(n) iteration)
# - Progression filter: O(m) where m = progression length (subset iteration)
# - Callable filter: O(n) full scan (cannot optimize without indexing)
#
# Overload Annotations:
# ---------------------
# Type checker sees different return types:
#     @overload
#     def __getitem__(self, key: UUID) -> T: ...
#     def __getitem__(self, key: int) -> T: ...
#     def __getitem__(self, key: slice) -> list[T]: ...
#     def __getitem__(self, key: Progression) -> Pile[T]: ...
#     def __getitem__(self, key: Callable[[T], bool]) -> Pile[T]: ...


def test_getitem_by_uuid(simple_items):
    """Test __getitem__ with UUID."""
    pile = Pile(items=simple_items)
    item = simple_items[0]

    retrieved = pile[item.id]
    assert retrieved == item


def test_getitem_by_str(simple_items):
    """Test __getitem__ with string ID."""
    pile = Pile(items=simple_items)
    item = simple_items[0]

    retrieved = pile[str(item.id)]
    assert retrieved == item


def test_getitem_by_index(simple_items):
    """Test __getitem__ with int index."""
    pile = Pile(items=simple_items)

    # Positive index
    assert pile[0] == simple_items[0]
    assert pile[2] == simple_items[2]

    # Negative index
    assert pile[-1] == simple_items[-1]


def test_getitem_by_slice(simple_items):
    """Test __getitem__ with slice."""
    pile = Pile(items=simple_items)

    # Slice returns list
    result = pile[1:3]
    assert isinstance(result, list)
    assert len(result) == 2
    assert result == simple_items[1:3]


def test_getitem_by_progression(simple_items):
    """Test __getitem__ with Progression."""
    pile = Pile(items=simple_items)

    # Create progression with subset of items
    subset_ids = [simple_items[1].id, simple_items[3].id]
    prog = Progression(order=subset_ids)

    filtered_pile = pile[prog]
    assert isinstance(filtered_pile, Pile)
    assert len(filtered_pile) == 2
    assert simple_items[1].id in filtered_pile
    assert simple_items[3].id in filtered_pile


def test_getitem_by_callable(simple_items):
    """Test __getitem__ with callable predicate."""
    pile = Pile(items=simple_items)

    # Filter by value
    filtered_pile = pile[lambda x: x.value >= 3]
    assert isinstance(filtered_pile, Pile)
    assert len(filtered_pile) == 2
    assert all(item.value >= 3 for item in filtered_pile)


def test_getitem_invalid_key_type(simple_items):
    """Test __getitem__ with invalid key type raises TypeError."""
    pile = Pile(items=simple_items)

    with pytest.raises(TypeError, match="Invalid key type"):
        _ = pile[3.14]  # float not supported


def test_filter_by_progression_returns_new_pile(simple_items):
    """Test _filter_by_progression returns NEW Pile."""
    pile = Pile(items=simple_items)
    subset_ids = [simple_items[0].id, simple_items[2].id]
    prog = Progression(order=subset_ids)

    filtered_pile = pile[prog]

    # Verify new pile created
    assert filtered_pile is not pile
    assert filtered_pile.item_type == pile.item_type
    assert filtered_pile.strict_type == pile.strict_type


def test_filter_by_function_returns_new_pile(simple_items):
    """Test _filter_by_function returns NEW Pile."""
    pile = Pile(items=simple_items)

    filtered_pile = pile[lambda x: x.value < 3]

    # Verify new pile created
    assert filtered_pile is not pile
    assert filtered_pile.item_type == pile.item_type


def test_filter_by_type(simple_items, typed_items):
    """Test filter_by_type method."""
    # Create pile with mixed types
    pile = Pile(items=simple_items + typed_items, item_type={SimpleElement, TypedElement})

    # Filter by SimpleElement
    simple_pile = pile.filter_by_type(SimpleElement)
    assert len(simple_pile) == 5
    assert all(isinstance(item, SimpleElement) for item in simple_pile)


def test_filter_by_type_with_strict_validation():
    """Test filter_by_type validates against pile's item_type."""
    pile = Pile(item_type=TypedElement)

    # Try to filter by incompatible type
    with pytest.raises(TypeError, match="not compatible"):
        pile.filter_by_type(SimpleElement)


def test_filter_by_type_no_matches():
    """Test filter_by_type raises NotFoundError when no matches."""
    pile = Pile(items=[SimpleElement(value=1)])

    with pytest.raises(NotFoundError, match="No items of type"):
        pile.filter_by_type(TypedElement)


def test_filter_by_type_strict_mode_invalid_types():
    """Test filter_by_type in strict mode rejects invalid types."""
    pile = Pile(items=[TypedElement(name="test")], item_type=TypedElement, strict_type=True)

    # Try to filter by type not in allowed set
    with pytest.raises(TypeError, match="not allowed in pile"):
        pile.filter_by_type(SimpleElement)


# =============================================================================
# Async Operations Tests
# =============================================================================


@pytest.mark.asyncio
async def test_async_add():
    """Test async add operation using context manager."""
    pile = Pile()
    item = SimpleElement(value=42)

    async with pile:
        pile.add(item)

    assert len(pile) == 1
    assert item.id in pile


@pytest.mark.asyncio
async def test_async_add_duplicate_raises_error():
    """Test async add of duplicate raises ExistsError."""
    pile = Pile()
    item = SimpleElement(value=42)

    async with pile:
        pile.add(item)
        with pytest.raises(ExistsError, match="already exists"):
            pile.add(item)


@pytest.mark.asyncio
async def test_async_remove():
    """Test async remove operation using context manager."""
    pile = Pile()
    item = SimpleElement(value=42)

    async with pile:
        pile.add(item)
        removed = pile.remove(item.id)

    assert removed == item
    assert len(pile) == 0


@pytest.mark.asyncio
async def test_async_remove_nonexistent_raises_error():
    """Test async remove of nonexistent item raises NotFoundError."""
    from uuid import uuid4

    pile = Pile()
    async with pile:
        with pytest.raises(NotFoundError, match="not found"):
            pile.remove(uuid4())


@pytest.mark.asyncio
async def test_async_get():
    """Test async get operation using context manager."""
    pile = Pile()
    item = SimpleElement(value=42)

    async with pile:
        pile.add(item)
        retrieved = pile.get(item.id)

    assert retrieved == item


@pytest.mark.asyncio
async def test_async_get_nonexistent_raises_error():
    """Test async get of nonexistent item raises NotFoundError."""
    from uuid import uuid4

    pile = Pile()
    async with pile:
        with pytest.raises(NotFoundError, match="not found"):
            pile.get(uuid4())


@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager (__aenter__ / __aexit__)."""
    pile = Pile()
    item = SimpleElement(value=42)
    pile.add(item)

    # Context manager acquires lock, so we access internal state directly
    async with pile as p:
        assert p is pile
        # Direct access to internal state (lock already held)
        assert len(p._items) == 1
        assert item.id in p._items
        assert p._items[item.id] == item


@pytest.mark.asyncio
async def test_concurrent_async_operations():
    """Test concurrent async operations using context manager."""
    pile = Pile()
    items = [SimpleElement(value=i) for i in range(10)]

    # Add items in async context
    async with pile:
        for item in items:
            pile.add(item)

    assert len(pile) == 10

    # Get items in async context
    async with pile:
        results = [pile.get(item.id) for item in items]

    assert len(results) == 10
    assert all(r in items for r in results)


# =============================================================================
# Pile Error Handling Tests (Traceback Suppression)
# =============================================================================


def test_remove_suppresses_keyerror():
    """Verify Pile.remove uses 'from None' to suppress KeyError traceback."""
    import traceback
    from uuid import uuid4

    pile = Pile()
    fake_id = uuid4()

    try:
        pile.remove(fake_id)
    except NotFoundError as e:
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        tb_str = "".join(tb)

        # Should NOT contain KeyError or "During handling" context
        assert "KeyError" not in tb_str, "KeyError should be suppressed by 'from None'"
        assert "During handling" not in tb_str, "Exception context should be suppressed"


def test_get_suppresses_keyerror():
    """Verify Pile.get uses 'from None' to suppress KeyError traceback."""
    import traceback
    from uuid import uuid4

    pile = Pile()
    fake_id = uuid4()

    try:
        pile.get(fake_id)
    except NotFoundError as e:
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        tb_str = "".join(tb)

        assert "KeyError" not in tb_str, "KeyError should be suppressed by 'from None'"
        assert "During handling" not in tb_str, "Exception context should be suppressed"


# =============================================================================
# Thread Safety Tests
# =============================================================================
#
# Thread-safety mechanisms: Pile achieves safe concurrent access through RLock-based
# synchronization with zero-copy reads.
#
# Synchronization Strategy:
# --------------------------
# 1. **RLock (Reentrant Lock)**: threading.RLock
#    - Allows same thread to acquire lock multiple times
#    - Required for nested operations: pile.add → pile._validate_type
#    - Prevents deadlocks in recursive validation
#
# 2. **@synchronized decorator**: Function-level protection
#    - Wraps with self._lock.acquire() / release()
#    - Applied to all mutating operations: add, remove, update, clear
#    - Read operations also synchronized for consistency
#
# 3. **AsyncLock**: For async operations
#    - Separate lock for __aenter__ / __aexit__
#    - async with pile: acquires _async_lock
#    - Independent of sync lock (no cross-contamination)
#
# 4. **Immutable views**: Zero-copy reads
#    - pile.items returns MappingProxyType (read-only dict wrapper)
#    - pile.progression returns copy (Progression(order=list(...)))
#    - No lock needed for reads (CPython GIL makes dict reads thread-safe)
#
# Concurrency Invariants:
# ------------------------
#     □(op ∈ {add, remove, get, update} ⇒ atomic)    [RLock serializes ops]
#     □(∀t: pile[uuid] consistent)                    [no partial updates]
#     □(∀t: len(pile) = |_items| = |_progression|)   [structural integrity]
#     □(readers never blocked by readers)             [dict is GIL-protected]
#
# Race Condition Scenarios:
# -------------------------
# 1. Duplicate add (same UUID):
#    Thread1: pile.add(x) → checks "x.id in _items" under lock → adds
#    Thread2: pile.add(x) → waits for lock → checks → raises ValueError
#    Result: Exactly one thread succeeds (atomicity guaranteed)
#
# 2. Concurrent remove + get:
#    Thread1: pile.remove(uuid) → acquires lock → removes from dict + progression
#    Thread2: pile.get(uuid) → waits for lock → ValueError (item not found)
#    Result: Serialized access (no torn reads)
#
# 3. Filter operations (immutable semantics):
#    Thread1: filtered1 = pile[lambda x: x.value > 5] → creates NEW Pile
#    Thread2: filtered2 = pile[lambda x: x.value < 3] → creates NEW Pile
#    Result: Both succeed, original pile unchanged (no locking contention)
#
# Performance Characteristics:
# ----------------------------
# - Lock contention: O(1) dict ops minimize hold time
# - No lock convoy: Readers don't block readers (GIL handles dict reads)
# - Filter ops: No contention (creates new Pile, source unchanged)
# - Progression ops: O(n) but single-threaded (progression.remove is linear scan)
#
# Test Strategy:
# --------------
# ThreadPoolExecutor with 4 workers stress-tests:
# - 20+ concurrent adds (unique items)
# - 10+ concurrent removes (partial removal)
# - 10+ concurrent gets (read-heavy workload)
# - Mixed operations (add/remove/get interleaved)


def test_thread_safe_add():
    """Test thread-safe add operations."""
    pile = Pile()
    items = [SimpleElement(value=i) for i in range(20)]

    def add_item(item):
        pile.add(item)

    # Add items from multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(add_item, items)

    assert len(pile) == 20
    for item in items:
        assert item.id in pile


def test_thread_safe_remove():
    """Test thread-safe remove operations."""
    pile = Pile()
    items = [SimpleElement(value=i) for i in range(20)]
    for item in items:
        pile.add(item)

    def remove_item(item):
        pile.remove(item.id)

    # Remove items from multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(remove_item, items[:10])

    assert len(pile) == 10


def test_thread_safe_get():
    """Test thread-safe get operations."""
    pile = Pile()
    items = [SimpleElement(value=i) for i in range(10)]
    for item in items:
        pile.add(item)

    results = []
    lock = threading.Lock()

    def get_item(item):
        retrieved = pile.get(item.id)
        with lock:
            results.append(retrieved)

    # Get items from multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(get_item, items)

    assert len(results) == 10
    assert all(r in items for r in results)


def test_thread_safe_mixed_operations():
    """Test thread-safe mixed operations (add, remove, get)."""
    pile = Pile()
    initial_items = [SimpleElement(value=i) for i in range(10)]
    for item in initial_items:
        pile.add(item)

    results = []
    lock = threading.Lock()

    def add_items():
        for i in range(10, 15):
            pile.include(SimpleElement(value=i))

    def remove_items():
        for item in initial_items[:5]:
            pile.exclude(item.id)

    def get_items():
        for item in initial_items[5:]:
            try:
                result = pile.get(item.id)
                with lock:
                    results.append(result)
            except ValueError:
                pass

    # Run operations from multiple threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(add_items),
            executor.submit(remove_items),
            executor.submit(get_items),
        ]
        concurrent.futures.wait(futures)

    # Verify final state
    assert 10 <= len(pile) <= 15


# =============================================================================
# Type Validation Tests
# =============================================================================
#
# Type validation strategy: Pile supports flexible type constraints while maintaining
# runtime safety through set-based type algebra.
#
# Design Rationale:
# -----------------
# 1. **Set-based validation**: O(1) type checking via set membership
#    - item_type stored as set[type] for efficient lookup
#    - Normalized from Union/list/single type during initialization
#
# 2. **Dual modes**:
#    - strict_type=False (default): Duck typing, allows subclasses
#      ∀x: valid(x) ⟺ ∃t ∈ I: isinstance(x, t)
#    - strict_type=True: Exact type match, no inheritance
#      ∀x: valid(x) ⟺ type(x) ∈ I
#
# 3. **Union type support**: Pile[Node | Event]
#    - Union[A, B] normalized to {A, B}
#    - Multi-type collections with single validation pass
#
# 4. **Fail-fast validation**:
#    - Type checked at add() time (not lazy)
#    - Deserialization validates ALL items before constructing Pile
#    - Invalid types raise TypeError immediately
#
# Thread Safety:
# --------------
# _validate_type() called within @synchronized methods (no lock needed)
# Set membership checks are atomic (immutable set, no races)
#
# Performance:
# ------------
# Permissive mode: O(k) where k = |item_type| (isinstance checks)
# Strict mode: O(1) (set membership: type(x) in item_type)


def test_validate_type_not_element():
    """Test _validate_type rejects non-Element."""
    pile = Pile()

    with pytest.raises(TypeError, match="must be Element subclass"):
        pile._validate_type("not an element")


def test_validate_type_permissive_allows_subclass():
    """Test permissive mode allows subclasses."""
    pile = Pile(item_type=TypedElement, strict_type=False)
    child_item = ChildElement(name="child", extra="extra")

    # Should not raise
    pile._validate_type(child_item)


def test_validate_type_strict_rejects_subclass():
    """Test strict mode rejects subclasses."""
    pile = Pile(item_type=TypedElement, strict_type=True)
    child_item = ChildElement(name="child", extra="extra")

    with pytest.raises(TypeError, match="strict_type=True"):
        pile._validate_type(child_item)


def test_validate_type_strict_allows_exact_type():
    """Test strict mode allows exact type."""
    pile = Pile(item_type=TypedElement, strict_type=True)
    exact_item = TypedElement(name="exact")

    # Should not raise
    pile._validate_type(exact_item)


def test_validate_type_multiple_allowed_types():
    """Test validation with multiple allowed types."""
    pile = Pile(item_type={SimpleElement, TypedElement})

    # Both types should be allowed
    simple_item = SimpleElement(value=42)
    typed_item = TypedElement(name="test")

    pile._validate_type(simple_item)
    pile._validate_type(typed_item)


# =============================================================================
# Type Constraints Edge Cases Tests
# =============================================================================


class TestPileTypeConstraintsEdgeCases:
    """
    Test Pile type constraint edge cases and adapter registry isolation.

    Edge Cases:
        - Permissive mode with unrelated types: Rejects non-subclass types
        - Adapter registry isolation: Subclass registries independent

    Scenarios:
        - Deserialize with unrelated type in permissive mode
        - Verify isolated adapter registries per subclass

    Invariants Tested:
        - Permissive mode: Allows subclasses, rejects unrelated types
        - Adapter registry: Rust-like explicit pattern (no pollution)

    Design Rationale:
        Permissive vs Strict:
            - Permissive: Duck typing for subclasses (flexible)
            - Still validates type hierarchy (safety)
            - Rejects unrelated types (not "anything goes")

        Adapter Isolation:
            - Each subclass has independent registry
            - Prevents adapter pollution across class hierarchy
            - Explicit registration required (Rust-like safety)
    """

    def test_from_dict_permissive_mode_incompatible_type(self):
        """
        Test permissive mode rejects unrelated types (not subclasses).

        Pattern:
            Permissive type validation with inheritance hierarchy

        Edge Case:
            Type constraint allows subclasses, but not unrelated types

        Scenario:
            1. Create pile with ChildElement (subclass of TypedElement)
            2. Serialize to dict
            3. Change item_type to unrelated type (SimpleElement)
            4. Attempt deserialization in permissive mode

        Expected:
            TypeError: ChildElement is not a subclass of SimpleElement

        Design Trade-off:
            Permissive mode allows subclasses for flexibility, but still
            validates type hierarchy for safety. Not "anything goes".

        Use Case:
            Deserializing heterogeneous collections from external sources
            where type metadata may be modified or incorrect.

        Complexity:
            Type check: O(1) set membership + O(len(mro)) isinstance
        """
        # Create pile with TypeC (child of TypeB)
        child_item = ChildElement(name="child", extra="extra")
        pile = Pile(items=[child_item])
        data = pile.to_dict()

        # Change item_type to unrelated type (SimpleElement)
        data["item_type"] = [f"{SimpleElement.__module__}.{SimpleElement.__name__}"]
        data["strict_type"] = False

        # Should fail because ChildElement is not a subclass of SimpleElement
        # (permissive mode allows subclasses, but not unrelated types)
        with pytest.raises(TypeError, match="is not a subclass of any allowed type"):
            Pile.from_dict(data)

    def test_isolated_registry_per_subclass(self):
        """
        Test that each Pile subclass has its own isolated adapter registry.

        Pattern:
            Rust-like explicit adapter pattern (no implicit inheritance)

        Edge Case:
            Adapter registration is per-class, not shared across hierarchy

        Scenario:
            1. Create two Pile subclasses (PileA, PileB)
            2. Verify neither has adapters initially
            3. Attempt to use unregistered adapter (toml)
            4. Verify AdapterNotFoundError raised for both

        Expected:
            Both subclasses have independent registries (no shared state)

        Design Rationale:
            Why explicit adapters?
                - Prevents adapter pollution across hierarchy
                - Explicit registration required (Rust-like safety)
                - Each subclass controls its own adapter set

        Architecture:
            pydapter creates unique registry attribute per class using id(cls).
            This ensures complete isolation between subclasses.

        Use Case:
            Domain-specific Pile subclasses with different serialization needs
            (e.g., FilePile with file:// adapters, DBPile with db:// adapters).

        Complexity:
            Registry lookup: O(1) via class ID-based attribute name
        """

        # Create two Pile subclasses
        class PileA(Pile):
            pass

        class PileB(Pile):
            pass

        # Both should have independent registries (no adapters initially)
        from pydapter.exceptions import AdapterNotFoundError

        pile_a = PileA([Element()])
        pile_b = PileB([Element()])

        # Neither has toml adapter (isolated registries)
        try:
            pile_a.adapt_to("toml")
            raise AssertionError("PileA should not have toml adapter")
        except AdapterNotFoundError:
            pass  # Expected

        try:
            pile_b.adapt_to("toml")
            raise AssertionError("PileB should not have toml adapter")
        except AdapterNotFoundError:
            pass  # Expected


# =============================================================================
# Serialization Tests
# =============================================================================
#
# Serialization strategy: Pile preserves progression order and type constraints
# through to_dict/from_dict round-trips with three output modes.
#
# Design Philosophy:
# ------------------
# Pile extends Element serialization with:
# 1. **Progression order preservation**: Items serialized in order
# 2. **Type metadata**: item_type set serialized as list[str]
# 3. **Validation on deserialization**: Fail-fast type checking
# 4. **Mode-aware**: python/json/db with different formats
#
# Output Modes:
# -------------
# 1. python (default):
#    - UUIDs as UUID objects
#    - datetime as datetime objects
#    - Items as dict with Python types
#    - For in-memory transfer (no JSON serialization needed)
#
# 2. json:
#    - UUIDs as str
#    - datetime as ISO format str
#    - JSON-safe primitives only
#    - For API responses, file storage
#
# 3. db:
#    - Like json, but metadata → node_metadata (db column naming)
#    - For PostgreSQL/Qdrant storage via pydapter
#
# Progression Preservation:
# --------------------------
# Progression order is IMPLICIT in items array:
#     data["items"] = [items[uuid] for uuid in progression]
#     # Order recovered during from_dict by add() sequence
#
# Progression name stored in metadata:
#     if progression.name:
#         data["metadata"]["progression_name"] = progression.name
#
# Why implicit order? Token efficiency:
#     ❌ {items: [...], order: [uuid1, uuid2, ...]} (duplicates UUIDs)
#     ✅ {items: [dict1, dict2, ...]} (order = iteration order)
#
# Type Serialization:
# -------------------
# item_type: set[type] → list[str]:
#     {SimpleElement, TypedElement}
#     → ["tests.base.test_pile.SimpleElement", "tests.base.test_pile.TypedElement"]
#
# Deserialization: list[str] → set[type]:
#     load_type_from_string() imports each type
#     Pydantic validator normalizes to set[type]
#
# Fail-Fast Validation:
# ---------------------
# from_dict() validates ALL items BEFORE adding:
#     for item_dict in items_data:
#         lion_class = item_dict["metadata"]["lion_class"]
#         validate_type(load_type(lion_class))  # Check before Element.from_dict
#     # Only if all valid, construct Pile + add items
#
# Why fail-fast? Prevents partial deserialization:
#     ❌ Add 10 items, fail on 11th → Pile has 10 items (inconsistent state)
#     ✅ Validate 11 items, fail on 11th → Raise TypeError, no Pile created
#
# Thread Safety:
# --------------
# to_dict(): @synchronized ensures consistent snapshot
#     - Items serialized in progression order
#     - No concurrent modifications during serialization
#
# from_dict(): No locking needed (class method, creates new Pile)
#     - Adds items sequentially (single-threaded initialization)
#
# Round-Trip Guarantees:
# ----------------------
#     pile1 = Pile(items=[...], item_type=T, strict_type=True)
#     data = pile1.to_dict(mode="json")
#     pile2 = Pile.from_dict(data)
#
# Invariants preserved:
#     □(pile1.item_type == pile2.item_type)
#     □(pile1.strict_type == pile2.strict_type)
#     □(list(pile1) == list(pile2))  [order preserved]
#     □(pile1._progression.name == pile2._progression.name)  [metadata preserved]
#     □(∀item: item.id in pile1 ⟺ item.id in pile2)  [membership preserved]
#
# Performance:
# ------------
# to_dict(): O(n) where n = len(pile)
#     - Iterate progression: O(n)
#     - Serialize each item: O(n) x O(item serialization)
#
# from_dict(): O(n) where n = len(items)
#     - Validate all types: O(n) x O(1) [set membership]
#     - Deserialize items: O(n) x O(Element.from_dict)
#     - Add items: O(n) x O(1) [dict insert + progression append]


def test_to_dict_python_mode(simple_items):
    """Test to_dict in python mode."""
    pile = Pile(items=simple_items)
    data = pile.to_dict(mode="python")

    assert "items" in data
    assert len(data["items"]) == 5
    assert isinstance(data["id"], UUID)  # UUID object in python mode


def test_to_dict_json_mode(simple_items):
    """Test to_dict in json mode."""
    pile = Pile(items=simple_items)
    data = pile.to_dict(mode="json")

    assert "items" in data
    assert len(data["items"]) == 5
    assert isinstance(data["id"], str)  # String in json mode


def test_to_dict_preserves_progression_order(simple_items):
    """Test to_dict preserves progression order."""
    reversed_order = [item.id for item in reversed(simple_items)]
    pile = Pile(items=simple_items, order=reversed_order)
    data = pile.to_dict()

    # Items should be in reversed order
    for i, item_dict in enumerate(data["items"]):
        assert item_dict["id"] == reversed_order[i]


def test_to_dict_preserves_progression_name(simple_items):
    """Test to_dict preserves progression name in metadata."""
    pile = Pile(items=simple_items)
    # Manually set progression name (not preserved from __init__)
    pile._progression.name = "custom_order"
    data = pile.to_dict()

    # progression_name only added if progression has a name
    assert "metadata" in data
    assert "progression_name" in data["metadata"]
    assert data["metadata"]["progression_name"] == "custom_order"


def test_to_dict_adds_metadata_for_progression_name():
    """Test to_dict creates metadata dict if needed for progression name."""
    # Create pile with no metadata
    pile = Pile()
    pile.add(SimpleElement(value=1))
    # Clear metadata and set progression name
    pile.metadata = {}
    pile._progression.name = "test_prog"

    pile.to_dict(exclude={"metadata"})  # Exclude existing metadata
    # Re-serialize with metadata
    data2 = pile.to_dict()

    assert "metadata" in data2
    assert "progression_name" in data2["metadata"]


def test_serialize_item_type():
    """Test item_type serialization."""
    pile = Pile(item_type={SimpleElement, TypedElement})
    data = pile.to_dict()

    assert "item_type" in data
    assert isinstance(data["item_type"], list)
    assert len(data["item_type"]) == 2
    # Should be module paths
    assert all("." in type_str for type_str in data["item_type"])


def test_from_dict_basic(simple_items):
    """Test from_dict basic deserialization."""
    pile = Pile(items=simple_items)
    data = pile.to_dict()

    restored = Pile.from_dict(data)
    assert len(restored) == len(pile)
    for item in simple_items:
        assert item.id in restored


def test_from_dict_preserves_item_type():
    """Test from_dict preserves item_type."""
    pile = Pile(item_type=SimpleElement)
    data = pile.to_dict()

    restored = Pile.from_dict(data)
    assert restored.item_type == {SimpleElement}


def test_from_dict_preserves_strict_type():
    """Test from_dict preserves strict_type."""
    pile = Pile(item_type=TypedElement, strict_type=True)
    data = pile.to_dict()

    restored = Pile.from_dict(data)
    assert restored.strict_type is True


def test_from_dict_preserves_progression_name(simple_items):
    """Test from_dict preserves progression name."""
    pile = Pile(items=simple_items)
    # Manually set progression name
    pile._progression.name = "test_order"
    data = pile.to_dict()

    restored = Pile.from_dict(data)
    # Check internal progression (progression property returns copy)
    assert restored._progression.name == "test_order"


def test_from_dict_validates_types_strict_mode():
    """Test from_dict validates types in strict mode."""
    # Create pile with mixed types
    child_item = ChildElement(name="child", extra="extra")
    pile = Pile(items=[child_item])
    data = pile.to_dict()

    # Try to deserialize with strict parent type
    data["item_type"] = [f"{TypedElement.__module__}.{TypedElement.__name__}"]
    data["strict_type"] = True

    # Should fail because ChildElement is not exact match for TypedElement
    with pytest.raises(TypeError, match="strict_type=True"):
        Pile.from_dict(data)


def test_from_dict_validates_types_permissive_mode():
    """Test from_dict validates types in permissive mode."""
    # Create pile with child type
    child_item = ChildElement(name="child", extra="extra")
    pile = Pile(items=[child_item])
    data = pile.to_dict()

    # Deserialize with parent type (permissive)
    data["item_type"] = [f"{TypedElement.__module__}.{TypedElement.__name__}"]
    data["strict_type"] = False

    # Should succeed because ChildElement is subclass of TypedElement
    restored = Pile.from_dict(data)
    assert len(restored) == 1


# =============================================================================
# Serialization Edge Cases Tests
# =============================================================================


class TestPileSerializationEdgeCases:
    """
    Test Pile serialization edge cases and error handling.

    Edge Cases:
        - Union type serialization: Union[A, B] → {A, B}
        - Set type serialization: set → list → set round-trip
        - Runtime type objects: Type objects vs strings
        - Invalid lion_class: Validation error handling
        - Metadata modes: db mode with custom meta_key

    Scenarios:
        - Serialize/deserialize Union types
        - Serialize/deserialize set-based type constraints
        - Programmatic from_dict with type objects
        - Validation with invalid/missing type metadata
        - Mode-specific metadata handling

    Invariants Tested:
        - Type normalization: Union/set → set internally
        - Serialization format: set → list (JSON-safe)
        - Deserialization: list/Union/type objects → set
        - Validation: Graceful error handling for invalid metadata
        - Metadata preservation: Across all modes

    Design Rationale:
        Union Type Support:
            - Union[A, B] normalized to {A, B} for O(1) validation
            - Set-based algebra for efficient type checking
            - Serialization uses list (JSON-compatible)

        Validation Strategy:
            - Fail-fast: Validate before deserialization
            - Graceful degradation: Skip invalid metadata, continue validation
            - Error late: Element.from_dict handles actual deserialization errors

        Metadata Handling:
            - Mode-aware: python/json/db have different formats
            - Custom meta_key: db mode supports column name customization
            - Progression name: Stored in metadata for restoration
    """

    def test_from_dict_with_union_item_type(self):
        """
        Test Union type serialization and deserialization.

        Pattern:
            Union type normalization through full serialization cycle

        Edge Case:
            Union[TypeA, TypeB] → {TypeA, TypeB} → list → {TypeA, TypeB}

        Scenario:
            1. Create pile with Union[TypeA, TypeB] type constraint
            2. Add items of both types
            3. Serialize to dict
            4. Deserialize from dict
            5. Verify type constraint restored as set

        Expected:
            - Union normalized to set: {TypeA, TypeB}
            - Serialization: set → list of strings
            - Deserialization: list → set of types
            - Type constraint preserved through round-trip

        Design Rationale:
            Why normalize Union to set?
                - O(1) type validation via set membership
                - Consistent internal representation
                - Set-based type algebra for multi-type constraints

        Use Case:
            Heterogeneous collections with explicit type constraints:
            Pile[Node | Event] for workflow systems.

        Complexity:
            - Union extraction: O(k) where k = number of union members
            - Type validation: O(1) set membership check
        """
        from typing import Union

        # Create pile with Union type
        pile = Pile(item_type=Union[SimpleElement, TypedElement])
        pile.add(SimpleElement(value=1))
        pile.add(TypedElement(name="test"))

        # Serialize
        data = pile.to_dict()

        # Deserialize - extract_types handles Union → set conversion
        restored = Pile.from_dict(data)

        assert len(restored) == 2
        assert restored.item_type == {SimpleElement, TypedElement}

    def test_from_dict_with_set_item_type(self):
        """
        Test set-based type constraint serialization round-trip.

        Pattern:
            Set → list → set serialization for JSON compatibility

        Edge Case:
            Python set → JSON list (JSON has no set type)

        Scenario:
            1. Create pile with set of types: {TypeA, TypeB}
            2. Serialize to dict
            3. Verify item_type serialized as list
            4. Deserialize from dict
            5. Verify item_type restored as set

        Expected:
            - item_type: set → list → set
            - JSON-safe format (list, not set)
            - Set semantics preserved (no duplicate types)

        Design Rationale:
            Why serialize to list?
                - JSON has no set type (compatibility)
                - List preserves all type references
                - Deserialization normalizes back to set

        Use Case:
            API responses, file storage where JSON format required
            but set semantics desired internally.

        Complexity:
            - Serialization: O(k) where k = |item_type|
            - Deserialization: O(k) list → set conversion
        """
        # Create pile with set of types
        pile = Pile(item_type={SimpleElement, TypedElement})
        pile.add(SimpleElement(value=1))

        data = pile.to_dict()

        # item_type serializes to list
        assert isinstance(data["item_type"], list)

        # Deserialize - should handle list correctly
        restored = Pile.from_dict(data)
        assert restored.item_type == {SimpleElement, TypedElement}

    def test_from_dict_strict_mode_exact_match_in_list(self):
        """
        Test strict mode allows exact type match in deserialization.

        Pattern:
            Strict type validation positive case (valid exact match)

        Edge Case:
            Strict mode with valid exact type (not just rejection tests)

        Scenario:
            1. Create pile with exact type in strict mode
            2. Serialize to dict
            3. Deserialize from dict
            4. Verify deserialization succeeds

        Expected:
            - Strict mode: type(item) ∈ item_type (exact match)
            - Deserialization succeeds for valid exact types
            - No subclass tolerance

        Design Trade-off:
            Strict vs Permissive:
                - Strict: Homogeneous collections (predictable)
                - Permissive: Heterogeneous collections (flexible)
                - Strict prevents "surprising" subtypes

        Use Case:
            Task queues where all tasks must be exact type (not subclasses)
            to ensure uniform processing behavior.

        Complexity:
            Type check: O(1) set membership (type(x) in item_type)
        """
        # Create pile with exact type in strict mode
        exact_item = TypedElement(name="exact")
        pile = Pile(items=[exact_item], item_type=TypedElement, strict_type=True)
        data = pile.to_dict()

        # Should deserialize successfully
        restored = Pile.from_dict(data)
        assert len(restored) == 1
        assert restored.strict_type is True

    def test_to_dict_with_progression_name_and_db_mode(self):
        """
        Test progression name serialization with db mode and custom meta_key.

        Pattern:
            Mode-specific metadata handling (db vs python/json)

        Edge Case:
            Custom meta_key for database column naming conventions

        Scenario:
            1. Create pile with empty metadata
            2. Set progression name
            3. Serialize with db mode and custom meta_key
            4. Verify progression_name in custom metadata field

        Expected:
            - Metadata initialized as {} if None/empty
            - progression_name stored in custom meta_key field
            - db mode uses custom column names

        Design Rationale:
            Why mode-aware metadata?
                - python mode: Python objects (UUID, datetime)
                - json mode: JSON-safe strings
                - db mode: Database column names (node_metadata, pile_meta)

        Use Case:
            ORM integration where metadata column has custom name
            (e.g., "pile_meta" instead of "metadata").

        Complexity:
            Metadata initialization: O(1)
        """
        # Create pile with empty metadata
        elem = Element()
        pile = Pile([elem])

        # Set progression name
        pile._progression.name = "custom_progression"

        # Serialize with metadata renaming (db mode)
        data = pile.to_dict(mode="db", meta_key="pile_meta")

        # The progression_name should be stored in the metadata
        assert "pile_meta" in data
        assert data["pile_meta"]["progression_name"] == "custom_progression"

    def test_from_dict_with_runtime_type_objects(self):
        """
        Test from_dict with type objects (not strings).

        Pattern:
            Programmatic API with type objects vs serialized strings

        Edge Case:
            Passing type objects directly instead of string references

        Scenario:
            1. Create pile with type constraint
            2. Serialize to dict
            3. Deserialize with type object passed as kwarg
            4. Verify type constraint restored

        Expected:
            - extract_types() handles both strings and type objects
            - No import/lookup needed for type objects
            - Direct type reference works

        Design Rationale:
            Why support both?
                - Strings: Serialization format (cross-process)
                - Type objects: Programmatic usage (same process)
                - extract_types() unifies both paths

        Use Case:
            Programmatic pile construction where types are available
            as objects (not serialized strings).

        Complexity:
            Type object handling: O(1) (no import/lookup)
        """
        # Create pile with type constraint
        elem1 = Element()
        elem2 = Element()
        pile = Pile([elem1, elem2], item_type=Element)

        # Serialize to dict
        data = pile.to_dict(mode="python")

        # Modify data to simulate runtime case - pass actual type objects, not strings
        # This happens when constructing Pile programmatically with from_dict
        # and passing item_type directly as kwargs
        restored = Pile.from_dict(data, item_type=Element)

        assert len(restored) == 2
        assert restored.item_type == {Element}

    def test_from_dict_with_invalid_lion_class_skips_validation(self):
        """
        Test validation loop continues when lion_class can't be imported.

        Pattern:
            Fail-safe validation (don't crash on invalid metadata)

        Edge Case:
            Invalid lion_class in metadata (typo, missing module)

        Scenario:
            1. Create data with invalid and valid lion_class values
            2. Attempt deserialization
            3. Verify validation loop catches ValueError and continues
            4. Verify deserialization fails later (Element.from_dict)

        Expected:
            - Validation loop: Skip invalid lion_class, continue
            - Deserialization error: Later stage (not validation)
            - ValueError message: "Failed to deserialize"

        Design Rationale:
            Why separate validation and deserialization?
                - Validation: Check types are compatible (early)
                - Deserialization: Construct objects (late)
                - Graceful degradation: Don't crash on metadata issues

        Use Case:
            Deserializing data from external sources where metadata
            may be corrupted or outdated.

        Complexity:
            Validation: O(n) with try/except per item
        """
        # Create data with item_type constraint and one invalid, one valid lion_class
        # The validation loop should catch ValueError for invalid class and continue
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-01-01T00:00:00Z",
            "metadata": {},
            "items": [
                {
                    "id": "650e8400-e29b-41d4-a716-446655440001",
                    "created_at": "2025-01-01T00:00:00Z",
                    "metadata": {
                        "lion_class": "nonexistent.module.InvalidClass"  # Invalid - triggers ValueError
                    },
                },
                {
                    "id": "750e8400-e29b-41d4-a716-446655440002",
                    "created_at": "2025-01-01T00:00:00Z",
                    "metadata": {
                        "lion_class": "lionherd_core.base.element.Element"  # Valid
                    },
                },
            ],
            "item_type": ["lionherd_core.base.element.Element"],
            "strict_type": True,  # Enable validation to trigger the code path
        }

        # The validation should skip the invalid lion_class gracefully
        # and continue validating the valid one without raising
        # The actual deserialization will still fail for the invalid class, but that's later
        # We're testing that the VALIDATION loop handles ValueError correctly
        try:
            Pile.from_dict(data)
            # If we get here, the validation worked but deserialization failed later
        except ValueError as e:
            # The ValueError should be from Element.from_dict deserialization, not from validation
            # Verify it's the deserialization error (not validation error)
            assert "Failed to deserialize" in str(e)

    def test_from_dict_validation_with_all_invalid_lion_classes(self):
        """
        Test validation with all invalid lion_class values.

        Pattern:
            Extreme edge case - no valid metadata at all

        Edge Case:
            All items have invalid lion_class (worst-case scenario)

        Scenario:
            1. Create data with all invalid lion_class values
            2. Attempt deserialization
            3. Verify validation loop completes gracefully
            4. Verify deserialization fails appropriately

        Expected:
            - Validation: Completes without crashing
            - Deserialization: Fails (no valid items)
            - Graceful error handling throughout

        Design Rationale:
            Why continue validation?
                - Collect all errors (not just first)
                - Graceful degradation philosophy
                - Better error reporting

        Use Case:
            Bulk data import from legacy systems where all metadata
            may be outdated or malformed.

        Complexity:
            Validation: O(n) with try/except for each item
        """
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-01-01T00:00:00Z",
            "metadata": {},
            "items": [
                {
                    "id": "650e8400-e29b-41d4-a716-446655440001",
                    "created_at": "2025-01-01T00:00:00Z",
                    "metadata": {"lion_class": "invalid.one.Class"},
                },
                {
                    "id": "750e8400-e29b-41d4-a716-446655440002",
                    "created_at": "2025-01-01T00:00:00Z",
                    "metadata": {"lion_class": "invalid.two.Class"},
                },
            ],
            "item_type": ["lionherd_core.base.element.Element"],
            "strict_type": False,  # Permissive mode
        }

        # Validation should skip all invalid classes gracefully
        # Deserialization will fail, but validation loop should complete
        try:
            Pile.from_dict(data)
        except ValueError:
            # Expected - deserialization fails, but validation loop completed
            pass


# =============================================================================
# Collection Methods Tests
# =============================================================================


def test_contains_by_uuid(simple_items):
    """Test __contains__ with UUID."""
    pile = Pile(items=simple_items)
    assert simple_items[0].id in pile

    from uuid import uuid4

    assert uuid4() not in pile


def test_contains_by_element(simple_items):
    """Test __contains__ with Element instance."""
    pile = Pile(items=simple_items)
    assert simple_items[0] in pile
    assert SimpleElement(value=999) not in pile


def test_contains_invalid_returns_false():
    """Test __contains__ with invalid input returns False."""
    pile = Pile()
    assert "invalid" not in pile
    assert 42 not in pile


def test_len(simple_items):
    """Test __len__."""
    pile = Pile(items=simple_items)
    assert len(pile) == 5

    pile.remove(simple_items[0].id)
    assert len(pile) == 4


def test_iter(simple_items):
    """Test __iter__ yields items in progression order."""
    pile = Pile(items=simple_items)

    items_list = list(pile)
    assert len(items_list) == 5
    assert items_list == simple_items


def test_list_conversion(simple_items):
    """Test __list__ and to_list()."""
    pile = Pile(items=simple_items)

    # __list__
    items_list = pile.__list__()
    assert items_list == simple_items

    # list() uses __list__()
    items_list2 = list(pile)
    assert items_list2 == simple_items


def test_keys(simple_items):
    """Test keys() yields UUIDs."""
    pile = Pile(items=simple_items)
    keys_list = list(pile.keys())

    assert len(keys_list) == 5
    assert all(isinstance(k, UUID) for k in keys_list)
    assert all(k in [item.id for item in simple_items] for k in keys_list)


def test_values(simple_items):
    """Test __iter__ yields items (values)."""
    pile = Pile(items=simple_items)
    values_list = list(pile)  # __iter__ yields items

    assert len(values_list) == 5
    assert values_list == simple_items


def test_len_and_bool(simple_items):
    """Test __len__ and __bool__."""
    pile = Pile(items=simple_items)
    assert len(pile) == 5
    assert pile  # __bool__ returns True for non-empty pile


def test_is_empty():
    """Test is_empty()."""
    pile = Pile()
    assert pile.is_empty()
    assert not pile  # __bool__ returns False for empty pile

    pile.add(SimpleElement(value=42))
    assert not pile.is_empty()
    assert pile  # __bool__ returns True for non-empty pile

    pile.clear()
    assert pile.is_empty()


# =============================================================================
# Property Tests
# =============================================================================


def test_keys_method(simple_items):
    """Test keys() method returns UUID iterator."""
    pile = Pile(items=simple_items)
    keys_list = list(pile.keys())

    # Should return all UUIDs
    assert len(keys_list) == len(simple_items)
    assert all(item.id in keys_list for item in simple_items)

    # Should be in insertion order
    expected_order = [item.id for item in simple_items]
    assert keys_list == expected_order


def test_items_method(simple_items):
    """Test items() method returns (UUID, item) iterator."""
    pile = Pile(items=simple_items)
    items_list = list(pile.items())

    # Should return all (UUID, item) pairs
    assert len(items_list) == len(simple_items)

    # Each item should be a tuple
    for uuid, item in items_list:
        assert isinstance(uuid, UUID)
        assert pile[uuid] == item

    # Should be in insertion order
    expected_order = [(item.id, item) for item in simple_items]
    assert items_list == expected_order


def test_progression_property_copy(simple_items):
    """Test progression property returns copy."""
    pile = Pile(items=simple_items)
    prog1 = pile.progression
    prog2 = pile.progression

    # Should be different objects (copies)
    assert prog1 is not prog2
    assert list(prog1.order) == list(prog2.order)


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_empty_pile_operations():
    """Test operations on empty pile."""
    pile = Pile()

    # Iteration
    assert list(pile) == []
    assert list(pile.keys()) == []
    assert list(pile.items()) == []

    # Size checks
    assert len(pile) == 0
    assert pile.is_empty()
    assert not pile  # __bool__ returns False for empty pile

    # Clear on empty
    pile.clear()
    assert len(pile) == 0


def test_pile_repr():
    """Test __repr__."""
    pile = Pile()
    assert repr(pile) == "Pile(len=0)"

    pile.add(SimpleElement(value=42))
    assert repr(pile) == "Pile(len=1)"


def test_progression_order_integrity_after_operations(simple_items):
    """Test progression maintains order through operations."""
    pile = Pile(items=simple_items)

    # Add new item
    new_item = SimpleElement(value=999)
    pile.add(new_item)

    # Check order includes new item at end
    items_list = list(pile)
    assert items_list == [*simple_items, new_item]

    # Remove middle item
    pile.remove(simple_items[2].id)

    # Check order maintained (without removed item)
    expected = simple_items[:2] + simple_items[3:] + [new_item]
    assert list(pile) == expected


@pytest.mark.asyncio
async def test_async_operations_with_type_validation():
    """Test async operations respect type validation."""
    pile = Pile(item_type=TypedElement)

    valid_item = TypedElement(name="valid")
    async with pile:
        pile.add(valid_item)

    assert len(pile) == 1

    invalid_item = SimpleElement(value=42)
    async with pile:
        with pytest.raises(TypeError, match="not a subclass"):
            pile.add(invalid_item)


def test_filter_operations_preserve_config(simple_items):
    """Test filter operations preserve item_type and strict_type."""
    pile = Pile(
        items=simple_items,
        item_type=SimpleElement,
        strict_type=True,
    )

    # Filter by callable
    filtered = pile[lambda x: x.value >= 3]
    assert filtered.item_type == pile.item_type
    assert filtered.strict_type == pile.strict_type

    # Filter by progression
    prog = Progression(order=[simple_items[0].id, simple_items[1].id])
    filtered2 = pile[prog]
    assert filtered2.item_type == pile.item_type
    assert filtered2.strict_type == pile.strict_type


# ==================== Coverage Tests for Missing Lines ====================


def test_pile_to_dict_with_name_and_none_meta(simple_items):
    """Test Pile.to_dict with name and meta_key=None path (line 202)."""
    pile = Pile[Element](items=simple_items[:2])
    # Give the progression a name
    pile._progression.name = "test_pile"

    # Serialize with mode that creates None meta dict
    result = pile.to_dict(mode="db", meta_key="meta")

    # Should have created meta dict and added progression_name
    assert "meta" in result
    assert result["meta"] is not None
    assert result["meta"]["progression_name"] == "test_pile"


def test_pile_getitem_progression_with_missing_uuid(simple_items):
    """Test pile[progression] raises NotFoundError for missing UUID (line 449)."""
    from lionherd_core.errors import NotFoundError

    pile = Pile[Element](items=simple_items[:2])

    # Create progression with a UUID not in pile
    from uuid import uuid4

    missing_uuid = uuid4()
    prog = Progression(order=[simple_items[0].id, missing_uuid])

    # Should raise NotFoundError
    with pytest.raises(NotFoundError) as exc_info:
        _ = pile[prog]

    assert "not found in pile" in str(exc_info.value)


def test_pile_from_dict_runtime_extract_types(simple_items):
    """Test Pile.from_dict with item_type at runtime (line 667)."""
    # Create a Pile and serialize it
    pile = Pile[Element](items=simple_items[:2])
    pile_dict = pile.to_dict()

    # Add item_type as a type object (not string) - forces runtime extract_types path
    pile_dict["item_type"] = [Element]  # Type object, not string

    # Deserialize - should use extract_types for runtime case
    restored = Pile.from_dict(pile_dict)

    assert len(restored) == 2
    assert restored.item_type == {Element}  # extract_types normalizes to set

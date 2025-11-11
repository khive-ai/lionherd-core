# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Progression: ordered UUID sequences with list/workflow operations.

Mathematical Model:
    Progression is a finite sequence: P = (u₁, u₂, ..., uₙ) where uᵢ ∈ UUID
    Total order with index: P[i] = uᵢ, 0 ≤ i < n
    Operations preserve ordering semantics:
        - append(u): P → P + (u)  [O(1)]
        - insert(i, u): P → P[:i] + (u) + P[i:]  [O(n)]
        - remove(u): P → P without first occurrence of u  [O(n)]
        - move(i, j): P → reordered with P[i] at position j  [O(n)]

    Stability: Relative order of unaffected elements is preserved across mutations.
    Duplicates: Allowed. remove() and index() target first occurrence only.

Workflow Semantics:
    Progressions represent execution order in state machines:
        State = {pending: Progression, active: Progression, completed: Progression}
        Transitions:
            - start(uuid): move(uuid, pending, active)
            - finish(uuid): move(uuid, active, completed)
            - retry(uuid): move(uuid, completed, pending)

    Set-like operations (include/exclude) enable idempotent state management:
        - include(uuid): Ensures uuid ∈ P without duplicates
        - exclude(uuid): Ensures uuid ∉ P (no-op if absent)

    Workflow invariants:
        - ∀u: u appears in at most one state progression
        - completed.order preserves execution history
        - pending.order defines execution priority

Async Support:
    Progressions support async iteration patterns:
        async for item in progression:  # Not implemented, but conceptually O(1) per item
            process(item)

    Thread-safety: NOT guaranteed. Use external synchronization for concurrent access.
    Concurrent operations (append, include) may race. Use `include()` for idempotency.

    Integration with asyncio:
        - gather(*[async_op(item) for item in progression])
        - create_task_group() for structured concurrency
        - Progression state mutations are synchronous (no await needed)

Test Coverage:
    - Instantiation: empty, with UUIDs/Elements, name field, Element inheritance
    - Validation: None handling, UUID string coercion, list conversion
    - Core operations: append, insert, remove, pop, popleft, clear, extend
    - Query operations: contains, len, iter, getitem, setitem, index, reversed
    - Workflow operations: move, swap, reverse, index validation
    - Set-like operations: include, exclude (idempotent)
    - Error handling: NotFoundError (semantic exceptions), ValueError, ExceptionGroup for batch operations
    - Async: concurrent append/include, task groups
    - Serialization: to_dict/from_dict roundtrips, ln integration
    - Edge cases: empty, single item, duplicates, negative indices
"""

from uuid import UUID, uuid4

import pytest

from lionherd_core.base import Element, Progression
from lionherd_core.errors import NotFoundError
from lionherd_core.libs.concurrency import create_task_group, gather
from lionherd_core.ln import to_dict, to_list


class TestProgressionInstantiation:
    """Progression creation: empty, with UUIDs/Elements, name field, Element inheritance.

    Instantiation Patterns:
        Progression is a Pydantic BaseModel subclass inheriting from Element.
        Supports multiple construction patterns for flexibility.

    Construction Signatures:
        Progression() → Empty progression (order=[], name=None)
        Progression(order=[uuid1, uuid2]) → With UUIDs
        Progression(order=[elem1, elem2]) → With Elements (auto-converts to UUIDs)
        Progression(name="execution_order") → Named progression (empty)
        Progression(name="tasks", order=[...]) → Named with items

    Element Inheritance:
        Inherited fields from Element:
            - id: UUID (auto-generated via uuid4)
            - created_at: datetime (auto-set to now)
            - metadata: dict (default {}, mutable instance)

        Progression-specific fields:
            - order: list[UUID] (default [])
            - name: str | None (optional label)

    Validation on Construction:
        Field validators run on instantiation:
            - order: _validate_order coerces UUIDs/Elements/strings
            - name: Optional string, no validation
            - Element fields: Pydantic handles defaults

    Representation:
        __repr__ format: "Progression(len=N, name='X')" or "Progression(len=N)"
        Compact: Shows size and name, not full order (could be large)
        Debug-friendly: Easy to identify in logs

    Design Trade-offs:
        Flexible input: Accepts UUIDs or Elements (convenience)
        Auto-conversion: Elements → UUIDs via validation (normalized storage)
        Named vs unnamed: Optional name field for semantic labeling
        Pydantic: Validation + serialization for free
    """

    def test_empty_progression(self):
        """Empty progression should have no items."""
        prog = Progression()
        assert len(prog) == 0
        assert prog.order == []
        assert prog.name is None

    def test_with_uuids(self):
        """Can create progression with UUIDs."""
        uid1, uid2 = uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        assert len(prog) == 2
        assert prog.order == [uid1, uid2]

    def test_with_elements(self):
        """Can create progression with Elements (converts to UUIDs)."""
        elem1, elem2 = Element(), Element()
        prog = Progression(order=[elem1, elem2])
        assert len(prog) == 2
        assert prog.order == [elem1.id, elem2.id]

    def test_with_name(self):
        """Can create named progression."""
        prog = Progression(name="execution_order")
        assert prog.name == "execution_order"

    def test_inherits_element_fields(self):
        """Progression should inherit Element fields."""
        prog = Progression(name="test")
        assert isinstance(prog.id, UUID)
        assert prog.created_at is not None
        assert isinstance(prog.metadata, dict)

    def test_repr(self):
        """__repr__ should show length and name."""
        prog = Progression(name="test", order=[uuid4(), uuid4()])
        repr_str = repr(prog)
        assert "Progression" in repr_str
        assert "len=2" in repr_str
        assert "name='test'" in repr_str

    def test_repr_no_name(self):
        """__repr__ without name should not show name field."""
        prog = Progression(order=[uuid4()])
        repr_str = repr(prog)
        assert repr_str == "Progression(len=1)"


class TestProgressionValidation:
    """Validation: _validate_order with None/list/coercion.

    Validation Strategy:
        Input normalization via Pydantic field_validator.
        Goal: Accept diverse inputs, normalize to canonical form (list[UUID]).

    _validate_order(order) → list[UUID]:
        Input types:
            None → []  (empty list)
            list[UUID] → list[UUID]  (passthrough)
            list[str] → list[UUID]  (coerce UUID strings, common in JSON deserialization)
            list[Element] → list[UUID]  (extract .id field via _get_id)

        Coercion logic:
            Uses _get_id(item) helper to extract UUID from UUID/str/Element.
            UUID strings converted via UUID(str) constructor.
            Invalid UUIDs raise ValueError (fails fast, no partial state).

        Design Intent:
            Permissive input: Simplifies API usage (accept Elements or UUIDs)
            Strict validation: No silent failures, errors propagate
            JSON-friendly: Handles UUID string deserialization from to_dict/from_dict

    Serialization Contract:
        Progression.to_dict(mode="json") serializes UUIDs as strings (JSON compat).
        Progression.from_dict() deserializes strings back to UUID objects.
        Validator ensures roundtrip correctness.
    """

    def test_validate_none(self):
        """None should convert to empty list."""
        prog = Progression(order=None)
        assert prog.order == []

    def test_validate_with_uuid_strings(self):
        """Validator should coerce UUID strings to UUID objects."""
        uid1, uid2 = uuid4(), uuid4()
        # When deserializing from JSON, UUIDs come as strings
        data = {
            "order": [str(uid1), str(uid2)],  # UUID strings
            "metadata": {},
        }
        prog = Progression.from_dict(data)
        assert len(prog.order) == 2
        assert prog.order[0] == uid1
        assert prog.order[1] == uid2
        # Verify they're UUID objects, not strings
        assert isinstance(prog.order[0], UUID)
        assert isinstance(prog.order[1], UUID)

    def test_progression_with_no_order_argument(self):
        """
        Test Progression() with no arguments defaults to empty list.

        Pattern:
            Default parameter handling for optional fields

        Edge Case:
            Constructor called without order parameter (common initialization pattern)

        Expected:
            order=[] (empty list)

        Design Intent:
            Ergonomics: Simplifies empty progression creation
            Consistency: None and omitted parameter both → []
        """
        prog = Progression()

        assert prog.order == []
        assert len(prog) == 0

    def test_progression_init_with_mixed_valid_invalid_elements(self):
        """
        Test Progression initialization with Element objects.

        Pattern:
            Polymorphic input handling (UUIDs or Elements)

        Edge Case:
            Elements passed to constructor instead of UUIDs

        Expected:
            Elements converted to UUIDs via _validate_order

        Use Case:
            Workflow initialization: `Progression(order=[task1, task2])` where
            task1/task2 are Element instances (more intuitive than extracting .id)

        Complexity:
            O(n) validation with UUID extraction per item
        """
        elem1 = Element()
        elem2 = Element()

        # Use __init__ with Element objects - should extract .id via to_uuid()
        prog = Progression(order=[elem1, elem2])

        # Should have 2 items: the element IDs
        assert len(prog) == 2
        assert elem1.id in prog.order
        assert elem2.id in prog.order

    def test_progression_with_empty_and_none_order(self):
        """
        Test Progression handles empty list, None, and omitted parameter identically.

        Pattern:
            Consistent coercion across multiple null-like inputs

        Edge Cases:
            - Empty list: Explicit empty
            - None: Null coercion
            - No argument: Default parameter

        Expected:
            All three result in order=[] (canonical empty state)

        Design Intent:
            Flexibility: Accept diverse empty representations
            Normalization: Single internal representation (empty list)
            Ergonomics: All patterns produce same result (no surprises)
        """
        # Empty list
        prog1 = Progression(order=[])
        assert len(prog1) == 0

        # None (validator coerces None → [])
        prog2 = Progression(order=None)
        assert len(prog2) == 0

        # No argument
        prog3 = Progression()
        assert len(prog3) == 0


class TestProgressionCoreOperations:
    """List operations: append, insert, remove, pop, popleft, clear, extend.

    Ordering Stability:
        All operations preserve relative order of unaffected elements.
        Stability guarantee: If e₁ precedes e₂ before operation, and neither is moved/removed,
        then e₁ precedes e₂ after operation.

    Operation Semantics:
        append(item): Add to end
            Complexity: O(1) amortized (Python list append)
            Stability: Trivial (only appends, no reordering)
            Accepts: UUID or Element (converts to UUID)

        insert(index, item): Add at position
            Complexity: O(n) due to list shift
            Stability: Preserved (items after index shift right)
            Index: Supports negative indices, 0 ≤ idx ≤ len

        remove(item): Delete first occurrence
            Complexity: O(n) due to search + shift
            Stability: Preserved (items after removed element shift left)
            Error: Raises ValueError if not found

        pop(index=-1): Remove and return item
            Complexity: O(1) for last item (index=-1), O(n) for others
            Stability: Preserved (items after index shift left)
            Error: Raises IndexError if empty or out of bounds

        popleft(): Remove and return first item
            Complexity: O(n) due to list shift (not deque)
            Stability: Preserved (all items shift left)
            Error: Raises IndexError if empty
            Note: Use collections.deque if frequent popleft needed

        clear(): Remove all items
            Complexity: O(1) (replaces internal list)
            Result: Empty progression (len == 0)

        extend(items): Add multiple items
            Complexity: O(k) where k = len(items)
            Stability: Preserved (all new items appended)
            Accepts: Iterable of UUIDs/Elements

    Design Trade-offs:
        List-based: O(n) insert/remove, but O(1) indexing and cache-friendly
        Alternative: Deque would give O(1) popleft, but O(n) indexing
        Choice: Indexing and slicing more common than popleft in workflow patterns
    """

    def test_append(self):
        """append should add item to end."""
        prog = Progression()
        uid = uuid4()
        prog.append(uid)
        assert len(prog) == 1
        assert prog.order[-1] == uid

    def test_append_element(self):
        """append should accept Elements."""
        prog = Progression()
        elem = Element()
        prog.append(elem)
        assert prog.order[-1] == elem.id

    def test_insert(self):
        """insert should add item at specific position."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid3])
        prog.insert(1, uid2)
        assert prog.order == [uid1, uid2, uid3]

    def test_insert_element(self):
        """insert should accept Elements."""
        elem1, elem2 = Element(), Element()
        prog = Progression(order=[elem1])
        prog.insert(0, elem2)
        assert prog.order[0] == elem2.id

    def test_remove(self):
        """remove should delete first occurrence."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        prog.remove(uid2)
        assert prog.order == [uid1, uid3]

    def test_pop_default(self):
        """pop without index should remove last item."""
        uid1, uid2 = uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        result = prog.pop()
        assert result == uid2
        assert prog.order == [uid1]

    def test_pop_with_index(self):
        """pop with index should remove item at position."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        result = prog.pop(1)
        assert result == uid2
        assert prog.order == [uid1, uid3]

    def test_pop_with_default(self):
        """pop with default should return default on NotFoundError.

        Design Intent: Allow graceful fallback when index doesn't exist,
        consistent with dict.get(key, default) pattern.
        """
        uid1 = uuid4()
        prog = Progression(order=[uid1])
        default_value = uuid4()
        # Pop out of range with default
        result = prog.pop(10, default=default_value)
        assert result == default_value
        assert prog.order == [uid1]  # Unchanged

    def test_pop_without_default_raises(self):
        """pop without default should raise NotFoundError on invalid index.

        Design Intent: Semantic exception (NotFoundError) instead of IndexError
        for consistency with Pile/Graph/Flow patterns. Index "not found" is
        conceptually same as item "not found".
        """
        prog = Progression(order=[uuid4()])
        with pytest.raises(NotFoundError, match="not found"):
            prog.pop(10)

    def test_popleft_success(self):
        """popleft should remove and return first item."""
        uid1, uid2 = uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        result = prog.popleft()
        assert result == uid1
        assert prog.order == [uid2]

    def test_popleft_empty(self):
        """popleft on empty progression should raise NotFoundError.

        Design Intent: Semantic exception for empty collection operations.
        "Cannot pop from empty" is a "not found" semantic.
        """
        prog = Progression()
        with pytest.raises(NotFoundError, match="empty"):
            prog.popleft()

    def test_clear(self):
        """clear should remove all items."""
        prog = Progression(order=[uuid4(), uuid4(), uuid4()])
        prog.clear()
        assert len(prog) == 0
        assert prog.order == []

    def test_extend(self):
        """extend should add multiple items."""
        uid1, uid2 = uuid4(), uuid4()
        elem1, elem2 = Element(), Element()
        prog = Progression(order=[uid1])
        prog.extend([uid2, elem1, elem2])
        assert len(prog) == 4
        assert prog.order == [uid1, uid2, elem1.id, elem2.id]


class TestProgressionQueryOperations:
    """Query operations: contains, len, iter, getitem, setitem, index, reversed.

    Query Complexity:
        All query operations are read-only (no state mutation).
        Performance characteristics match Python list semantics.

    contains (in operator): Membership test
        Syntax: uuid in progression
        Complexity: O(n) linear search
        Returns: bool
        Accepts: UUID or Element
        Tolerance: Returns False for invalid types (no error)

    len: Count items
        Syntax: len(progression)
        Complexity: O(1) (Python list maintains length)
        Returns: int ≥ 0

    iter: Sequential iteration
        Syntax: for uuid in progression: ...
        Complexity: O(1) per item, O(n) total
        Yields: UUID objects in order
        Pattern: Enables comprehensions, generators, all/any checks

    getitem: Index access
        Syntax: progression[i] or progression[start:stop:step]
        Complexity: O(1) for single index, O(k) for slice
        Single index: Returns UUID, supports negative indices
        Slice: Returns list[UUID], not Progression (Python list semantics)
        Error: IndexError if out of bounds

    setitem: Index mutation
        Syntax: progression[i] = uuid or progression[start:stop] = [uuids]
        Complexity: O(1) for single index, O(k) for slice
        Single index: Replaces one item, accepts UUID/Element
        Slice: Replaces range with list[UUID/Element]
        Error: TypeError if slice value is not list
        Note: setitem is mutation, not query, but grouped for dunder symmetry

    index: Find position
        Syntax: progression.index(uuid)
        Complexity: O(n) linear search
        Returns: int (first occurrence index)
        Accepts: UUID or Element
        Error: ValueError if not found
        Note: Only returns first occurrence (duplicates allowed)

    reversed: Reverse iteration
        Syntax: reversed(progression) or list(reversed(progression))
        Complexity: O(1) to create iterator, O(n) to consume
        Returns: Iterator yielding UUIDs in reverse order
        Note: Does not mutate (contrast with progression.reverse())

    Design Patterns:
        Read path optimization: No locking needed (GIL sufficient for reads)
        Slicing returns list, not Progression (follows Python convention)
        Error tolerance in __contains__ (graceful for invalid types)
    """

    def test_contains_present(self):
        """__contains__ should return True for present items."""
        uid = uuid4()
        prog = Progression(order=[uid])
        assert uid in prog

    def test_contains_element(self):
        """__contains__ should accept Elements."""
        elem = Element()
        prog = Progression(order=[elem])
        assert elem in prog

    def test_contains_absent(self):
        """__contains__ should return False for absent items."""
        prog = Progression(order=[uuid4()])
        assert uuid4() not in prog

    def test_contains_invalid(self):
        """__contains__ should return False for invalid items (error tolerance)."""
        prog = Progression(order=[uuid4()])
        assert "invalid" not in prog

    def test_len(self):
        """__len__ should return number of items."""
        prog = Progression(order=[uuid4(), uuid4(), uuid4()])
        assert len(prog) == 3

    def test_iter(self):
        """__iter__ should iterate over UUIDs in order."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        result = list(prog)
        assert result == [uid1, uid2, uid3]

    def test_getitem_single(self):
        """__getitem__ with int should return single UUID."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        assert prog[0] == uid1
        assert prog[1] == uid2
        assert prog[-1] == uid3

    def test_getitem_slice(self):
        """__getitem__ with slice should return list of UUIDs."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        result = prog[0:2]
        assert result == [uid1, uid2]
        assert isinstance(result, list)

    def test_setitem_single(self):
        """__setitem__ with int should replace single item."""
        uid1, uid2, uid_new = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        prog[0] = uid_new
        assert prog.order[0] == uid_new

    def test_setitem_element(self):
        """__setitem__ should accept Elements."""
        elem = Element()
        prog = Progression(order=[uuid4()])
        prog[0] = elem
        assert prog.order[0] == elem.id

    def test_setitem_slice(self):
        """__setitem__ with slice should replace multiple items."""
        uid1, uid2, uid3, uid4 = uuid4(), uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        prog[0:2] = [uid3, uid4]
        assert prog.order == [uid3, uid4]

    def test_setitem_slice_type_error(self):
        """__setitem__ with slice requires list value."""
        prog = Progression(order=[uuid4(), uuid4()])
        with pytest.raises(TypeError, match="expected list"):
            prog[0:2] = uuid4()  # Should be list, not single UUID

    def test_index(self):
        """index should return position of item."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        assert prog.index(uid2) == 1

    def test_index_element(self):
        """index should accept Elements."""
        elem1, elem2 = Element(), Element()
        prog = Progression(order=[elem1, elem2])
        assert prog.index(elem2) == 1

    def test_reversed(self):
        """__reversed__ should iterate in reverse order."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        result = list(reversed(prog))
        assert result == [uid3, uid2, uid1]

    def test__list__(self):
        """__list__ should return items as list in order."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        result = prog.__list__()
        assert result == [uid1, uid2, uid3]
        assert isinstance(result, list)

    def test__list__empty(self):
        """__list__ should return empty list for empty progression."""
        prog = Progression()
        result = prog.__list__()
        assert result == []
        assert isinstance(result, list)


class TestProgressionWorkflowOperations:
    """Workflow operations: move, swap, reverse, index validation.

    Workflow Theory:
        Progressions are primitives for state machine construction. Key operations:

        move(from_idx, to_idx): State transition operator
            Semantics: Remove element at from_idx, insert at adjusted to_idx
            Adjustment: If from_idx < to_idx, then to_idx -= 1 after removal
            Use case: Task priority reordering, workflow stage transitions

        swap(i, j): Symmetric exchange operator
            Semantics: P[i], P[j] = P[j], P[i]
            Properties: Commutative (swap(i,j) == swap(j,i)), idempotent if i==j
            Use case: Dependency resolution, deadlock prevention

        reverse(): Total order inversion
            Semantics: P → P[::-1]
            Properties: Involutory (reverse ∘ reverse = identity)
            Use case: Undo stacks, LIFO↔FIFO conversion

    Index Validation:
        _validate_index(idx, allow_end=False): Bounds checking with negative index support
            Normalizes: idx ∈ [-len, len-1] → idx ∈ [0, len-1]
            allow_end: Enables idx == len for insertion operations
            Raises: IndexError if out of bounds or empty progression

    Workflow Patterns:
        Queue-based execution: popleft() from pending → process → append() to completed
        Priority scheduling: move() to reorder pending by priority
        Rollback/retry: move() from failed back to pending
        Audit trail: completed.order maintains execution history
    """

    def test_validate_index_empty(self):
        """_validate_index should raise NotFoundError for empty progression.

        Design Intent: Empty progression has no valid indices - "not found" semantic.
        """
        prog = Progression()
        with pytest.raises(NotFoundError, match="empty"):
            prog._validate_index(0)

    def test_validate_index_allow_end(self):
        """_validate_index with allow_end=True should accept len as index."""
        prog = Progression(order=[uuid4(), uuid4()])
        # Without allow_end, len is out of bounds
        with pytest.raises(NotFoundError):
            prog._validate_index(2, allow_end=False)
        # With allow_end, len is valid (for insertion)
        result = prog._validate_index(2, allow_end=True)
        assert result == 2

    def test_validate_index_negative(self):
        """_validate_index should handle negative indices."""
        prog = Progression(order=[uuid4(), uuid4(), uuid4()])
        assert prog._validate_index(-1) == 2
        assert prog._validate_index(-2) == 1

    def test_validate_index_out_of_bounds(self):
        """_validate_index should raise NotFoundError for invalid indices.

        Design Intent: Out-of-range index is semantically "not found" - the
        requested position doesn't exist in the progression.
        """
        prog = Progression(order=[uuid4(), uuid4()])
        with pytest.raises(NotFoundError, match="out of range"):
            prog._validate_index(5)

    def test_move_forward(self):
        """move should relocate item forward."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        prog.move(0, 2)  # Move first to position 2 (becomes position 1 after removal)
        # After removing uid1 from 0: [uid2, uid3]
        # Adjusted to_index: 2 - 1 = 1
        # Insert at 1: [uid2, uid1, uid3]
        assert prog.order == [uid2, uid1, uid3]

    def test_move_backward(self):
        """move should relocate item backward."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        prog.move(2, 0)  # Move last to first position
        assert prog.order == [uid3, uid1, uid2]

    def test_move_negative_indices(self):
        """move should support negative indices."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        prog.move(-1, 0)  # Move last to first
        assert prog.order == [uid3, uid1, uid2]

    def test_swap(self):
        """swap should exchange two items."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        prog.swap(0, 2)
        assert prog.order == [uid3, uid2, uid1]

    def test_swap_negative_indices(self):
        """swap should support negative indices."""
        uid1, uid2 = uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        prog.swap(0, -1)
        assert prog.order == [uid2, uid1]

    def test_reverse(self):
        """reverse should flip order in-place."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])
        prog.reverse()
        assert prog.order == [uid3, uid2, uid1]


class TestProgressionSetLikeOperations:
    """Set-like operations: include, exclude (idempotent).

    Idempotency Theory:
        Operations that satisfy: f(f(x)) = f(x)
        Safe for retries, concurrent calls, and eventual consistency patterns.

    include(item) → bool:
        Semantics: item ∈ P (set membership)
        Returns: True if added (new), False if already present
        Idempotent: include(x); include(x) ≡ include(x)
        Complexity: O(n) due to membership check

        Properties:
            - Order preservation: New items appended to end
            - Uniqueness: No duplicates introduced
            - Commutativity: include(a); include(b) ≡ include(b); include(a) (if both new)

        Use cases:
            - Deduplication in data pipelines
            - Async task registration (safe for redundant calls)
            - Event subscription lists (no duplicate handlers)

    exclude(item) → bool:
        Semantics: item ∉ P (set non-membership)
        Returns: True if removed (was present), False if absent
        Idempotent: exclude(x); exclude(x) ≡ exclude(x)
        Complexity: O(n) due to list.remove()

        Properties:
            - Order preservation: Remaining items retain relative order
            - First-occurrence removal: Only first match removed
            - Graceful degradation: No error if absent

        Use cases:
            - Safe cleanup in error handlers
            - Task cancellation (idempotent)
            - Resource deregistration

    Contrast with List Operations:
        append(x): NOT idempotent. Repeated calls → duplicates.
        remove(x): NOT idempotent. Second call raises ValueError if absent.

        include/exclude trade performance (O(n) membership check) for safety.

    Workflow Integration:
        State machines benefit from idempotency:
            - Retry logic: include(task) safe to call multiple times
            - Cleanup: exclude(resource) won't fail if already removed
            - Distributed systems: Out-of-order message handling
    """

    def test_include_new_item(self):
        """include should add item if not present."""
        uid = uuid4()
        prog = Progression()
        result = prog.include(uid)
        assert result is True
        assert uid in prog.order

    def test_include_existing_item(self):
        """include should not add duplicate (idempotent)."""
        uid = uuid4()
        prog = Progression(order=[uid])
        result = prog.include(uid)
        assert result is False
        assert len(prog) == 1

    def test_include_element(self):
        """include should accept Elements."""
        elem = Element()
        prog = Progression()
        result = prog.include(elem)
        assert result is True
        assert elem.id in prog.order

    def test_exclude_present_item(self):
        """exclude should remove item if present."""
        uid1, uid2 = uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        result = prog.exclude(uid1)
        assert result is True
        assert uid1 not in prog.order

    def test_exclude_absent_item(self):
        """exclude should do nothing if item absent (idempotent)."""
        uid = uuid4()
        prog = Progression(order=[uuid4()])
        result = prog.exclude(uid)
        assert result is False

    def test_exclude_element(self):
        """exclude should accept Elements."""
        elem = Element()
        prog = Progression(order=[elem])
        result = prog.exclude(elem)
        assert result is True
        assert elem.id not in prog.order


class TestProgressionErrorHandling:
    """Error handling: ExceptionGroup for batch operations.

    Error Propagation Strategy:
        Individual operations raise specific exceptions (NotFoundError, ValueError).
        Batch operations collect errors and group them via Python 3.11+ ExceptionGroup.

    Exception Types:
        NotFoundError: Out-of-bounds access, empty progression operations, index not found
            Raised by: pop, popleft, _validate_index (semantic "not found" vs positional IndexError)
        IndexError: Still raised by __getitem__ (Python list semantic preserved)
            Message: Descriptive (e.g., "empty progression", "index out of range")

        ValueError: Item not found in progression
            Raised by: remove, index
            Message: Includes context (e.g., "UUID not in list")

        TypeError: Invalid operation arguments
            Raised by: __setitem__ with slice (requires list value)
            Message: Expected type information

    ExceptionGroup Pattern (Python 3.11+):
        Use case: Multiple operations in batch fail independently
        Construction: ExceptionGroup("message", [exception_list])
        Benefits:
            - Preserves all error contexts (no information loss)
            - Enables except* syntax for selective handling
            - Useful for async gather + error aggregation

        Example:
            errors = []
            for op in batch_operations:
                try: op()
                except (IndexError, ValueError) as e: errors.append(e)
            if errors:
                raise ExceptionGroup("Batch errors", errors)

    Error Recovery Patterns:
        Idempotent operations: Use include/exclude instead of append/remove
        Try-except at call site: Handle individual operation failures
        Transaction pattern: Validate all inputs before mutations (no partial state)

    Design Philosophy:
        Fail fast: Errors propagate immediately, no silent failures
        Context-rich: Error messages include operation details
        Composable: Exceptions work with standard Python error handling
    """

    def test_batch_operations_with_errors(self):
        """Batch operations should collect errors in ExceptionGroup.

        Design Intent: Demonstrate that NotFoundError integrates cleanly with
        ExceptionGroup for batch error handling, maintaining semantic meaning.
        """
        prog = Progression(order=[uuid4(), uuid4()])
        errors = []

        # Try multiple operations, some will fail
        operations = [
            (lambda: prog.pop(10), "Invalid index 10"),
            (lambda: prog.remove(uuid4()), "UUID not in list"),
            (lambda: prog.index(uuid4()), "UUID not in list"),
        ]

        for op, _ in operations:
            try:
                op()
            except (NotFoundError, ValueError) as e:
                errors.append(e)

        # Verify we collected multiple errors
        assert len(errors) >= 2

        # Create ExceptionGroup with all errors
        if errors:
            exc_group = ExceptionGroup("Progression operation errors", errors)
            assert len(exc_group.exceptions) >= 2

    def test_validation_errors_grouped(self):
        """Multiple validation errors should be groupable.

        Design Intent: Show how NotFoundError from multiple operations can be
        collected and grouped for batch error reporting. Semantic exceptions
        make error groups more meaningful than generic IndexError.
        """
        prog = Progression()
        errors = []

        # Empty progression - multiple operations will fail
        try:
            prog.pop()
        except NotFoundError as e:
            errors.append(e)

        try:
            prog.popleft()
        except NotFoundError as e:
            errors.append(e)

        try:
            prog[0]
        except IndexError as e:
            errors.append(e)

        assert len(errors) == 3

        # Group them
        exc_group = ExceptionGroup("Empty progression errors", errors)
        assert isinstance(exc_group, ExceptionGroup)
        assert len(exc_group.exceptions) == 3


class TestProgressionAsync:
    """Async operations: concurrent append/include, task groups.

    Async Patterns:
        Progressions are synchronous data structures that integrate with async workflows.
        State mutations (append, insert, remove) are immediate (no await).
        Async coordination occurs at the workflow level, not within Progression itself.

    Concurrent Access Patterns:
        append(): Not thread-safe. Race conditions possible.
            Pattern: Use gather() for fire-and-forget appends
            Result: All items eventually present, order non-deterministic

        include(): Idempotent set-like operation. Safe for redundant concurrent calls.
            Pattern: Multiple tasks can safely include() same UUID
            Result: Exactly one returns True, len == 1 guaranteed

        Race Condition Handling:
            Strategy 1: Use include() instead of append() for idempotency
            Strategy 2: Serialize mutations via asyncio.Lock()
            Strategy 3: Accept non-deterministic order (append with gather)

    Task Group Integration:
        create_task_group() provides structured concurrency:
            - Automatic cancellation propagation on error
            - Exception aggregation via ExceptionGroup
            - Resource cleanup guarantees

        Usage Pattern:
            async with create_task_group() as tg:
                tg.start_soon(task1)  # No return value capture
                tg.start_soon(task2)
            # All tasks complete before context exit

    Async Iteration (Future):
        Progressions don't implement __aiter__/__anext__ (not needed for sync list).
        Pattern: Use sync iteration with async operations:
            for uuid in progression:
                await async_process(uuid)

        Alternative: Async comprehension with gather:
            await gather(*[async_process(uuid) for uuid in progression])

    Performance:
        Concurrent append: O(1) per operation, but contention on shared list
        gather() overhead: ~1-2ms per task spawn (asyncio.create_task)
        Task group: Minimal overhead for structured concurrency (~0.1ms)

    Thread Safety:
        ⚠️ Progressions are NOT thread-safe. Use asyncio.Lock or queue.Queue for
        cross-thread coordination. Python GIL provides some protection but not atomicity
        guarantees for compound operations (e.g., check-then-append).
    """

    @pytest.mark.asyncio
    async def test_concurrent_append(self):
        """Concurrent append operations should all succeed."""
        prog = Progression()
        uids = [uuid4() for _ in range(10)]

        async def append_item(uid):
            prog.append(uid)

        # Concurrent appends using gather
        await gather(*[append_item(uid) for uid in uids])

        # All items should be present
        assert len(prog) == 10
        for uid in uids:
            assert uid in prog.order

    @pytest.mark.asyncio
    async def test_concurrent_include(self):
        """Concurrent include operations should handle duplicates correctly."""
        prog = Progression()
        uid = uuid4()

        async def include_item():
            return prog.include(uid)

        # Try to include same UUID concurrently (only first should succeed)
        results = await gather(*[include_item() for _ in range(5)])

        # Only one should return True (first to add)
        assert results.count(True) == 1
        assert len(prog) == 1

    @pytest.mark.asyncio
    async def test_task_group_operations(self):
        """TaskGroup should coordinate multiple progression operations."""
        prog = Progression()

        async def batch_append():
            for _ in range(3):
                prog.append(uuid4())

        async def batch_query():
            # Query operations during modifications
            return len(prog)

        # Use create_task_group for structured concurrency
        async with create_task_group() as tg:
            # start_soon expects a callable, not a coroutine
            tg.start_soon(batch_append)
            tg.start_soon(batch_append)
            # Note: start_soon doesn't return values, so we can't capture query result
            tg.start_soon(batch_query)

        # After all tasks complete, verify state
        assert len(prog) == 6


class TestProgressionSerialization:
    """Serialization: to_dict/from_dict, integration with ln.to_list.

    Serialization Modes:
        Progression inherits Element serialization (to_dict/from_dict).
        Two modes: "python" (native types) and "json" (JSON-compatible strings).

    to_dict(mode="python") → dict:
        Fields: {id, created_at, metadata, name, order}
        Types: id→UUID, order→list[UUID]
        Use case: Internal representation, no serialization needed

    to_dict(mode="json") → dict:
        Fields: Same as python mode
        Types: id→str, order→list[str] (UUIDs as strings)
        Use case: JSON serialization, API responses, file storage

    from_dict(data: dict) → Progression:
        Deserializes both modes via _validate_order:
            - list[UUID] → passthrough (python mode)
            - list[str] → list[UUID] (json mode, coerced)
        Preserves: id, created_at, metadata, name, order
        Roundtrip guarantee: from_dict(to_dict(prog)) == prog (field equality)

    Integration with ln Module:
        ln.to_dict(progression): Universal dict conversion
            Delegates to progression.to_dict() if available
            Fallback: vars(progression) or custom logic

        ln.to_list(progression.order): List normalization
            Ensures list type (handles iterables, single items)
            Useful for coercing order field to list

    Serialization Invariants:
        1. Roundtrip: prog == Progression.from_dict(prog.to_dict(mode="python"))
        2. JSON-compatible: json.dumps(prog.to_dict(mode="json")) succeeds
        3. Field preservation: All Element fields (id, created_at, metadata) preserved
        4. Order preservation: order field maintains sequence after roundtrip

    Performance:
        to_dict: O(n) where n = len(order) (copies list)
        from_dict: O(n) due to _validate_order coercion
        JSON serialization: Additional O(n) for UUID→str conversion
    """

    def test_to_dict_basic(self):
        """to_dict should serialize Progression correctly."""
        uid1, uid2 = uuid4(), uuid4()
        prog = Progression(name="test", order=[uid1, uid2])
        data = prog.to_dict(mode="python")

        assert data["name"] == "test"
        assert data["order"] == [uid1, uid2]
        assert "id" in data
        assert "created_at" in data

    def test_from_dict_roundtrip(self):
        """Roundtrip through to_dict/from_dict should preserve data."""
        uid1, uid2 = uuid4(), uuid4()
        original = Progression(name="test", order=[uid1, uid2])
        data = original.to_dict(mode="python")
        restored = Progression.from_dict(data)

        assert restored.name == original.name
        assert restored.order == original.order
        assert restored.id == original.id

    def test_ln_to_dict_integration(self):
        """ln.to_dict should work with Progression."""
        prog = Progression(name="test", order=[uuid4()])
        # ln.to_dict can handle various object types
        data = to_dict(prog)
        assert isinstance(data, dict)
        assert "name" in data or "order" in data

    def test_ln_to_list_on_order(self):
        """ln.to_list should convert order field to list."""
        uid1, uid2 = uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2])
        # to_list can ensure we have a proper list
        order_list = to_list(prog.order)
        assert isinstance(order_list, list)
        assert len(order_list) == 2
        assert order_list == [uid1, uid2]

    def test_serialization_with_elements(self):
        """Serialization should work with Elements in order."""
        elem1, elem2 = Element(), Element()
        prog = Progression(order=[elem1, elem2])

        # Serialize
        data = prog.to_dict(mode="json")
        assert isinstance(data["order"], list)
        assert len(data["order"]) == 2

        # Deserialize
        restored = Progression.from_dict(data)
        assert restored.order == [elem1.id, elem2.id]


class TestProgressionEdgeCases:
    """Edge cases: empty, single item, duplicates, negative indices.

    Boundary Conditions:
        Edge case testing ensures robustness at limits of input domain.
        Focus: Empty progressions, single items, duplicates, negative indexing.

    Empty Progression (len == 0):
        Valid state: Progression() constructs empty instance
        Query operations: Safe (len→0, list→[], contains→False)
        Mutation operations: Mostly safe (clear, append, extend work)
        Error operations: pop, popleft, __getitem__ raise IndexError
        Workflow: Common initial state, valid for empty task queues

    Single Item (len == 1):
        Indexing: prog[0] and prog[-1] both return same UUID
        Boundary: Only valid indices are 0, -1
        Operations: All operations supported (remove→empty, pop→empty)
        Workflow: Minimal non-empty state, useful for testing

    Duplicates:
        Allowed: Progressions permit duplicate UUIDs (contrast with set)
        Semantics: remove() and index() target first occurrence only
        Use case: Workflow retry (same task multiple times), event logs
        Trade-off: Duplicates add complexity but enable flexible modeling

    Negative Indices:
        Python convention: -1 is last item, -2 is second-to-last, etc.
        Support: All index-based operations (getitem, setitem, pop, swap, move)
        Normalization: _validate_index converts negative to positive
        Range: -len ≤ idx < len for access, -len ≤ idx ≤ len for insertion

    Invalid Operations:
        remove(absent_uuid): Raises ValueError (strict, not idempotent)
        index(absent_uuid): Raises ValueError (not found)
        pop(out_of_bounds): Raises IndexError
        Rationale: Fail fast, clear error messages, no silent failures

    Design Intent:
        Empty as valid state: Simplifies initialization, no special cases
        Duplicates allowed: Flexibility for workflow modeling (vs set semantics)
        Negative indices: Pythonic, common pattern for end-relative access
        Strict errors: Fail fast prevents bugs from propagating
    """

    def test_empty_operations(self):
        """Operations on empty progression should handle gracefully."""
        prog = Progression()

        # Query operations
        assert len(prog) == 0
        assert list(prog) == []
        assert uuid4() not in prog

        # Modification operations that should work
        prog.clear()  # Should not raise
        assert len(prog) == 0

    def test_single_item_operations(self):
        """Single-item progression should support all operations."""
        uid = uuid4()
        prog = Progression(order=[uid])

        assert len(prog) == 1
        assert prog[0] == uid
        assert prog[-1] == uid
        assert uid in prog

        # Remove and verify empty
        prog.remove(uid)
        assert len(prog) == 0

    def test_duplicate_items(self):
        """Progression should allow duplicate UUIDs."""
        uid = uuid4()
        prog = Progression(order=[uid, uid, uid])
        assert len(prog) == 3
        assert prog.order.count(uid) == 3

        # Remove should only remove first occurrence
        prog.remove(uid)
        assert len(prog) == 2

    def test_negative_index_access(self):
        """Negative indices should work for all operations."""
        uid1, uid2, uid3 = uuid4(), uuid4(), uuid4()
        prog = Progression(order=[uid1, uid2, uid3])

        # Access
        assert prog[-1] == uid3
        assert prog[-2] == uid2

        # Pop
        result = prog.pop(-1)
        assert result == uid3
        assert len(prog) == 2

    def test_invalid_remove(self):
        """remove on absent item should raise ValueError."""
        prog = Progression(order=[uuid4()])
        with pytest.raises(ValueError):
            prog.remove(uuid4())

    def test_invalid_index(self):
        """index on absent item should raise ValueError."""
        prog = Progression(order=[uuid4()])
        with pytest.raises(ValueError):
            prog.index(uuid4())


# ==================== Coverage Tests for Missing Lines ====================


def test_progression_order_none_validator():
    """Test order=None is normalized to empty list (line 59)."""
    progression = Progression(order=None)
    assert progression.order == []
    assert len(progression) == 0


def test_progression_order_single_value_validator():
    """Test single UUID normalized to list (line 53)."""
    single_uuid = uuid4()

    # Pass single UUID via from_dict (triggers validator before __init__)
    progression = Progression.from_dict(
        {
            "id": str(uuid4()),
            "created_at": "2024-01-01T00:00:00Z",
            "order": str(single_uuid),  # Single UUID string, not list
        }
    )

    # Should be normalized to list by validator
    assert isinstance(progression.order, list)
    assert len(progression.order) == 1
    assert progression.order[0] == single_uuid


def test_progression_order_non_list_iterable_validator():
    """Test non-list iterable normalized to list (line 56)."""
    uuid1 = uuid4()
    uuid2 = uuid4()

    # Pass tuple (non-list iterable) via from_dict
    progression = Progression.from_dict(
        {
            "id": str(uuid4()),
            "created_at": "2024-01-01T00:00:00Z",
            "order": (str(uuid1), str(uuid2)),  # Tuple, not list
        }
    )

    # Should be converted to list by validator
    assert isinstance(progression.order, list)
    assert len(progression.order) == 2
    assert progression.order[0] == uuid1
    assert progression.order[1] == uuid2


def test_progression_getitem_slice():
    """Test __getitem__ with slice (line 152)."""
    uuids = [uuid4() for _ in range(5)]
    progression = Progression(order=uuids)

    # Test slice access
    result = progression[1:3]

    # Should return list of UUIDs
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == uuids[1]
    assert result[1] == uuids[2]


# ==================== Validator Regression Tests ====================


def test_progression_validator_handles_uuid_strings():
    """Regression: Validator should accept UUID strings and coerce to UUID objects."""
    uuid_strs = [str(uuid4()) for _ in range(3)]

    # Pass UUID strings via validator
    progression = Progression.from_dict(
        {"id": str(uuid4()), "created_at": "2024-01-01T00:00:00Z", "order": uuid_strs}
    )

    # Should coerce to UUID objects
    assert len(progression.order) == 3
    for i, uuid_str in enumerate(uuid_strs):
        assert progression.order[i] == UUID(uuid_str)


def test_progression_validator_handles_uuid_objects():
    """Regression: Validator should handle UUID objects directly in from_dict."""
    uuid_objs = [uuid4() for _ in range(3)]

    # Pass UUID objects directly (no need to serialize)
    progression = Progression.from_dict(
        {
            "id": str(uuid4()),
            "created_at": "2024-01-01T00:00:00Z",
            "order": uuid_objs,  # UUID objects work directly
        }
    )

    # Should preserve as UUID objects
    assert len(progression.order) == 3
    for i, uuid_obj in enumerate(uuid_objs):
        assert progression.order[i] == uuid_obj


def test_progression_validator_raises_on_invalid_uuid():
    """Regression: Validator should raise on invalid UUID string."""
    with pytest.raises((ValueError, AttributeError)):  # Invalid UUID string
        Progression.from_dict(
            {
                "id": str(uuid4()),
                "created_at": "2024-01-01T00:00:00Z",
                "order": ["not-a-valid-uuid"],
            }
        )


def test_progression_validator_empty_list():
    """Regression: Validator should handle empty list."""
    progression = Progression.from_dict(
        {"id": str(uuid4()), "created_at": "2024-01-01T00:00:00Z", "order": []}
    )

    assert progression.order == []
    assert len(progression) == 0

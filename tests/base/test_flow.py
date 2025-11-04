# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Flow as workflow state machine with dual-pile architecture.

Architecture - Dual-Pile Design:
    Flow combines two specialized data structures:
    1. Progressions container (Pile[Progression]): Ordered workflow stages
    2. Items pile (Pile[Element]): Shared item storage

    This separation enables:
    - M:N relationships (items can exist in multiple progressions)
    - Independent lifecycle management (items persist across stage transitions)
    - Named state access (flow.get_progression("pending") → Progression)
    - Flexible ordering (same items, different orders per progression)

Workflow State Machine Pattern:
    ```python
    # 1. Create flow with shared item storage
    flow = Flow[WorkItem, WorkProgression](name="deployment")

    # 2. Define workflow stages as named progressions
    flow.add(Progression(name="pending"))
    flow.add(Progression(name="active"))
    flow.add(Progression(name="completed"))

    # 3. Add items to shared pile
    task = WorkItem(name="deploy_api")
    flow.add_item(task, progression_ids="pending")

    # 4. State transitions: move between progressions
    flow.get_progression("pending").remove(task.id)
    flow.get_progression("active").append(task.id)

    # 5. Query current state
    active_items = [flow.items[id] for id in flow.get_progression("active").order]
    ```

Named Access Semantics:
    - flow.get_progression("stage_name") → Progression (O(1) lookup via name index)
    - flow[uuid] → Progression (O(1) lookup via items dict)
    - flow[progression] → Pile[Element] (filtered items from progression.order)
    - Enables ergonomic workflow queries: flow.get_progression("failed").order

Exception Aggregation for Batch Workflows:
    Flow operations collect errors into ExceptionGroup for batch reporting:

    ```python
    errors = []
    for item in items:
        try:
            flow.add_item(item, progression_ids="stage1")
        except ValueError as e:
            errors.append(e)

    if errors:
        raise ExceptionGroup("Batch validation errors", errors)
    ```

    Used for:
    - Bulk item insertion (collect all validation failures)
    - Multi-progression updates (aggregate inconsistencies)
    - Workflow integrity checks (report all constraint violations)

Async Workflow Execution:
    Flow.pile supports async operations for concurrent workflows:

    ```python
    async def process_batch(items):
        # Add items concurrently
        await gather(*[flow.items.add_async(item) for item in items])

        # Retrieve concurrently
        results = await gather(*[flow.items.get_async(id) for id in ids])
    ```

Design Rationale:
    1. **Dual-pile over single container**:
       - Progressions and items have different lifecycle semantics
       - Progressions define structure (workflow stages)
       - Items contain data (work units)
       - Separation enables independent evolution

    2. **UUID references over object references**:
       - Progressions store item.id, not item itself
       - Enables serialization (UUIDs are JSON-safe)
       - Allows lazy loading (fetch items on demand)
       - Supports distributed workflows (items in separate storage)

    3. **Named progressions over indexed access**:
       - flow.get_progression("pending") more readable than flow.progressions[0]
       - Enforces unique names (prevents accidental overwrites)
       - Enables workflow introspection (what stages exist?)
       - Natural mapping to domain concepts (stages, phases, states)

See Also:
    - Progression: Ordered container for workflow stages
    - Pile: Generic container with async support
    - Element: Base class for workflow items
"""

from __future__ import annotations

from uuid import UUID

import pytest

from lionherd_core.base import Element, Flow, Pile, Progression
from lionherd_core.libs.string_handlers._to_dict import to_dict

# ==================== Fixtures ====================


class FlowTestItem(Element):
    """Test item for Flow tests."""

    value: str = "test"


class FlowTestProgression(Progression):
    """Test progression for Flow tests."""

    pass


@pytest.fixture
def items():
    """Create test items."""
    return [FlowTestItem(value=f"item{i}") for i in range(5)]


@pytest.fixture
def progressions():
    """Create test progressions."""
    return [FlowTestProgression(name=f"prog{i}", order=[]) for i in range(3)]


@pytest.fixture
def flow(items, progressions):
    """Create Flow with items and progressions."""
    f = Flow[FlowTestItem, FlowTestProgression](
        items=items,
        name="test_flow",
        item_type=FlowTestItem,
    )
    # Add progressions
    for prog in progressions:
        f.add_progression(prog)
    return f


# ==================== Initialization Tests ====================


def test_flow_init_empty():
    """Test Flow initialization without items."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    assert len(f.progressions) == 0
    assert len(f.items) == 0
    assert f.name is None


def test_flow_init_with_items(items):
    """Test Flow initialization with items pre-populates the pile.

    Design Philosophy:
        Flow construction accepts items as a convenience parameter, immediately
        adding them to the items pile. This design choice prioritizes ergonomics
        (one-line flow creation with data) over explicitness (create flow, then
        add items separately).

    Architectural Decision:
        Items are added to pile during __init__, not stored as separate field.
        This ensures single source of truth: pile.items contains all items, no
        separate tracking needed. The items parameter is initialization-only.

    Why This Matters:
        Pre-population enables declarative flow construction:
        `flow = Flow(items=[...])` vs `flow = Flow(); for i in items: flow.add_item(i)`

        This pattern is consistent with Pile initialization and reduces boilerplate
        in common use cases (workflow initialization with known items).
    """
    f = Flow[FlowTestItem, FlowTestProgression](items=items, name="test")
    assert len(f.items) == 5
    assert f.name == "test"
    # Verify all items are in pile
    for item in items:
        assert item.id in f.items


def test_flow_init_with_item_type():
    """Test Flow initialization with item_type validation."""
    f = Flow[FlowTestItem, FlowTestProgression](
        item_type=FlowTestItem,
        strict_type=True,
    )
    # Should be able to add FlowTestItem
    item = FlowTestItem(value="test")
    f.items.add(item)
    assert len(f.items) == 1


def test_flow_init_normalizes_item_type():
    """Test Flow initialization normalizes item_type to set."""
    # Single type
    f1 = Flow[FlowTestItem, FlowTestProgression](item_type=FlowTestItem)
    assert f1.items.item_type == {FlowTestItem}

    # List of types
    f2 = Flow[Element, FlowTestProgression](item_type=[FlowTestItem, Element])
    assert f2.items.item_type == {FlowTestItem, Element}


def test_flow_validate_piles_converts_dict():
    """Test _validate_piles converts dict to Pile during deserialization."""
    # Create pile dict
    pile_dict = {
        "id": str(UUID("12345678-1234-5678-1234-567812345678")),
        "items": [],
        "item_type": None,
        "strict_type": False,
    }

    # Validate conversion (validator uses mode="wrap" so needs handler)
    def mock_handler(v):  # Mock handler that returns input
        return v

    result = Flow._validate_piles(pile_dict, mock_handler)
    assert isinstance(result, Pile)


def test_flow_validate_piles_preserves_pile():
    """Test _validate_piles delegates to handler for non-dict inputs."""
    pile = Pile[FlowTestItem]()

    # Validator delegates to handler for non-dict inputs
    def mock_handler(v):  # Mock handler that returns input
        return v

    result = Flow._validate_piles(pile, mock_handler)
    assert result is pile


# ==================== Progression Management Tests ====================


def test_flow_add_progression():
    """Test adding progression to Flow as workflow stage.

    Workflow Semantics:
        Progressions represent workflow stages (pending, active, completed).
        Adding a progression defines a new state in the workflow state machine.

    Pattern:
        ```python
        flow.add(Progression(name="pending"))  # Define stage
        flow.get_progression("pending")  # Access by name
        ```

    Name Registration:
        Named progressions are registered in `_progression_names` index for
        O(1) lookup. This enables ergonomic state queries in workflow code.
    """
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name="test_prog")

    f.add_progression(prog)
    assert len(f.progressions) == 1
    assert prog.id in f.progressions
    assert "test_prog" in f._progression_names


def test_flow_add_progression_duplicate_name_raises():
    """Test adding progression with duplicate name raises ValueError.

    Design Rationale - Name Uniqueness:
        Progression names serve as ergonomic access keys (`flow.get_progression("stage_name")`).
        Allowing duplicate names would create ambiguity: which progression should
        `flow.get_progression("duplicate")` return?

    Architecture Decision:
        Enforce uniqueness at insertion time (fail fast) rather than:
        1. Last-write-wins (silently overwrites) - loses data unexpectedly
        2. List of progressions with same name - breaks O(1) name lookup
        3. No enforcement (allow duplicates) - runtime errors during access

    Why This Matters:
        Name-based access is a core Flow ergonomic feature. Unique names enable
        deterministic behavior and prevent subtle bugs from name collisions.

        Trade-off: Slightly more restrictive API (must use unique names) for
        much better debugging experience (explicit error vs mysterious overwrites).

    Edge Case Handling:
        Progressions without names bypass uniqueness check (name=None is valid).
        This enables workflows where only some progressions need named access.
    """
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog1 = FlowTestProgression(name="duplicate")
    prog2 = FlowTestProgression(name="duplicate")

    f.add_progression(prog1)
    with pytest.raises(ValueError, match="Progression with name 'duplicate' already exists"):
        f.add_progression(prog2)


def test_flow_add_progression_without_name():
    """Test adding progression without name (no name registration)."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name=None)

    f.add_progression(prog)
    assert len(f.progressions) == 1
    assert prog.id in f.progressions
    assert len(f._progression_names) == 0


def test_flow_remove_progression_by_uuid(flow, progressions):
    """Test removing progression by UUID."""
    prog = progressions[0]
    removed = flow.remove_progression(prog.id)

    assert removed is prog
    assert prog.id not in flow.progressions
    assert prog.name not in flow._progression_names


def test_flow_remove_progression_by_name(flow, progressions):
    """Test removing progression by name."""
    prog = progressions[0]
    removed = flow.remove_progression(prog.name)

    assert removed is prog
    assert prog.id not in flow.progressions
    assert prog.name not in flow._progression_names


def test_flow_remove_progression_by_str_uuid(flow, progressions):
    """Test removing progression by string UUID."""
    prog = progressions[0]
    removed = flow.remove_progression(str(prog.id))

    assert removed is prog
    assert prog.id not in flow.progressions


def test_flow_remove_progression_by_instance(flow, progressions):
    """Test removing progression by Progression instance."""
    prog = progressions[0]
    removed = flow.remove_progression(prog)

    assert removed is prog
    assert prog.id not in flow.progressions


def test_flow_remove_progression_cleans_name_index(flow, progressions):
    """Test removing progression cleans up name index."""
    prog = progressions[0]
    name = prog.name

    flow.remove_progression(prog.id)
    assert name not in flow._progression_names


def test_flow_remove_progression_without_name():
    """Test removing progression that has no name."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name=None)
    f.add_progression(prog)

    removed = f.remove_progression(prog.id)
    assert removed is prog


# ==================== Item Management Tests ====================


def test_flow_add_item_to_pile_only():
    """Test adding item to pile without progressions."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    item = FlowTestItem(value="test")

    f.add_item(item)
    assert len(f.items) == 1
    assert item.id in f.items


def test_flow_add_item_to_single_progression():
    """Test adding item to pile and single progression (workflow state assignment).

    Workflow Semantics:
        Adding an item to a progression assigns it to that workflow state.
        The item exists in flow.progressions.items (shared storage) and is referenced by
        the progression's order list.

    Pattern:
        ```python
        flow.add_item(task, progression_ids="pending")  # Assign to state
        # Task now in "pending" stage, retrievable via flow.get_progression("pending")
        ```

    Two-Phase Addition:
        1. Item added to flow.items (shared storage, lifecycle managed here)
        2. Item.id added to progression.order (state membership)

    This design enables state transitions by moving UUIDs between progressions
    without copying/moving actual item data.
    """
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name="test")
    f.add_progression(prog)

    item = FlowTestItem(value="test")
    f.add_item(item, progression_ids=prog.id)

    assert item.id in f.items
    assert item.id in prog


def test_flow_add_item_to_multiple_progressions():
    """Test adding item to multiple progressions (M:N relationship).

    Workflow Semantics:
        Items can exist in multiple workflow stages simultaneously.
        This enables complex workflows where work units span multiple contexts.

    Use Cases:
        1. Cross-cutting concerns:
           - Item in both "active" and "needs_review" progressions
           - Represents work that's in-progress AND awaiting review

        2. Multi-phase workflows:
           - Deployment task in "qa_testing" and "staging_deploy"
           - Different teams track same work unit in different stages

        3. Tagging/categorization:
           - Use progressions as tags: "high_priority", "customer_facing"
           - Item can have multiple tags without duplication

    Architecture:
        Single item in flow.progressions.items, multiple progressions reference its UUID.
        Removing from one progression doesn't affect others (independent lifecycle).
    """
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog1 = FlowTestProgression(name="prog1")
    prog2 = FlowTestProgression(name="prog2")
    f.add_progression(prog1)
    f.add_progression(prog2)

    item = FlowTestItem(value="test")
    f.add_item(item, progression_ids=[prog1.id, prog2.id])

    assert item.id in f.items
    assert item.id in prog1
    assert item.id in prog2


def test_flow_add_item_by_progression_name():
    """Test adding item using progression name."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name="test_prog")
    f.add_progression(prog)

    item = FlowTestItem(value="test")
    f.add_item(item, progression_ids="test_prog")

    assert item.id in prog


def test_flow_remove_item_from_pile_only():
    """Test removing item from pile without removing from progressions."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name="test")
    f.add_progression(prog)

    item = FlowTestItem(value="test")
    f.add_item(item, progression_ids=prog.id)

    removed = f.remove_item(item.id, remove_from_progressions=False)
    assert removed is item
    assert item.id not in f.items
    assert item.id in prog  # Still in progression


def test_flow_remove_item_from_pile_and_progressions():
    """Test removing item from pile and all progressions.

    Design Philosophy - Item-Progression Lifecycle Independence:
        Items and progressions have independent lifecycles by default. Removing an
        item from the pile doesn't automatically remove it from progressions, and
        vice versa. This design respects the M:N relationship between items and
        progressions.

    Architectural Decision - Cascade Control:
        The `remove_from_progressions` parameter provides explicit control over
        cascade behavior:
        - False (default): Remove from pile only, leave progression references
          → Enables "soft delete" patterns (mark as deleted but preserve ordering)
        - True: Remove from pile AND scan all progressions for cleanup
          → Ensures referential integrity when item truly deleted

    Why This Matters:
        Different use cases require different cascade semantics:

        1. Soft deletion: Item marked inactive but workflows preserve it
           ```python
           item.metadata["deleted"] = True
           flow.items[item.id] = item  # Update in place
           # Progressions still reference it for audit trail
           ```

        2. Hard deletion: Item completely removed
           ```python
           flow.remove_item(item.id, remove_from_progressions=True)
           # No dangling references anywhere
           ```

    Performance Considerations:
        Cascade removal (remove_from_progressions=True) is O(P) where P is number
        of progressions, as each progression must be scanned for the item ID.
        For large flows with many progressions, prefer soft deletion or batch
        removal strategies.

    Alternative Designs Rejected:
        1. Always cascade (automatic removal from progressions)
           - ❌ Prevents soft deletion patterns
           - ❌ No way to preserve progression ordering for audit

        2. Never cascade (manual cleanup required)
           - ❌ Easy to create dangling references
           - ❌ Burden on caller to track all progressions

        3. Reference counting (remove from pile when no progressions reference it)
           - ❌ Complex bookkeeping overhead
           - ❌ Unclear lifecycle (when does item "belong" to flow?)
    """
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog1 = FlowTestProgression(name="prog1")
    prog2 = FlowTestProgression(name="prog2")
    f.add_progression(prog1)
    f.add_progression(prog2)

    item = FlowTestItem(value="test")
    f.add_item(item, progression_ids=[prog1.id, prog2.id])

    removed = f.remove_item(item.id, remove_from_progressions=True)
    assert removed is item
    assert item.id not in f.items
    assert item.id not in prog1
    assert item.id not in prog2


def test_flow_remove_item_by_str_uuid():
    """Test removing item by string UUID."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    item = FlowTestItem(value="test")
    f.add_item(item)

    removed = f.remove_item(str(item.id))
    assert removed is item


def test_flow_remove_item_by_element():
    """Test removing item by Element instance."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    item = FlowTestItem(value="test")
    f.add_item(item)

    removed = f.remove_item(item)
    assert removed is item


# ==================== __getitem__ Tests ====================


def test_flow_get_progression_by_uuid(flow, progressions):
    """Test getting progression by UUID."""
    prog = progressions[0]
    result = flow.get_progression(prog.id)
    assert result is prog


def test_flow_get_progression_by_name(flow, progressions):
    """Test getting progression by name."""
    prog = progressions[0]
    result = flow.get_progression(prog.name)
    assert result is prog


def test_flow_get_progression_by_str_uuid(flow, progressions):
    """Test getting progression by string UUID."""
    prog = progressions[0]
    result = flow.get_progression(str(prog.id))
    assert result is prog


def test_flow_get_progression_invalid_string_raises():
    """Test getting progression by invalid string raises KeyError."""
    f = Flow[FlowTestItem, FlowTestProgression]()

    with pytest.raises(KeyError, match="Progression 'nonexistent' not found"):
        _ = f.get_progression("nonexistent")


def test_flow_get_progression_checks_name_index_first():
    """Test get_progression checks name index before parsing UUID."""
    f = Flow[FlowTestItem, FlowTestProgression]()

    # Add progression with name that looks like UUID
    prog = FlowTestProgression(name="12345678-1234-5678-1234-567812345678")
    f.add_progression(prog)

    # Should find by name, not try to parse as UUID
    result = f.get_progression("12345678-1234-5678-1234-567812345678")
    assert result is prog


# ==================== __contains__ Tests ====================


def test_flow_contains_progression_by_uuid(flow, progressions):
    """Test checking if progression exists by UUID."""
    prog = progressions[0]
    assert prog.id in flow.progressions


def test_flow_contains_progression_by_name(flow, progressions):
    """Test checking if progression name is registered."""
    prog = progressions[0]
    assert prog.name in flow._progression_names


def test_flow_contains_item_by_uuid(flow, items):
    """Test checking if item exists in items pile by UUID."""
    item = items[0]
    assert item.id in flow.items


def test_flow_contains_item_by_str_uuid(flow, items):
    """Test checking if item exists in items pile by string UUID."""
    item = items[0]
    assert str(item.id) in flow.items


# ==================== __repr__ Tests ====================


def test_flow_repr_with_name(flow):
    """Test Flow repr with name."""
    repr_str = repr(flow)
    assert "test_flow" in repr_str
    assert "items=5" in repr_str
    assert "progressions=3" in repr_str


def test_flow_repr_without_name():
    """Test Flow repr without name."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    repr_str = repr(f)
    assert "items=0" in repr_str
    assert "progressions=0" in repr_str
    assert "name=" not in repr_str


# ==================== Serialization Tests ====================


def test_flow_to_dict(flow):
    """Test Flow serialization to dict."""
    data = flow.to_dict()

    assert "items" in data
    assert "progressions" in data
    # Both piles should be serialized as dicts
    assert isinstance(data["items"], dict)
    assert isinstance(data["progressions"], dict)
    assert data["name"] == "test_flow"


def test_flow_to_dict_with_exclude_list(flow):
    """Test Flow.to_dict() with exclude as list (not set)."""
    # Test the else branch when exclude is not a set
    data = flow.to_dict(exclude=["metadata"])

    # Should still have items and progressions
    assert "items" in data
    assert "progressions" in data
    # metadata should be excluded
    assert "metadata" not in data


def test_flow_to_dict_with_exclude_set(flow):
    """Test Flow.to_dict() with exclude as set."""
    # Test the if branch when exclude is already a set
    data = flow.to_dict(exclude={"metadata"})

    # Should still have items and progressions
    assert "items" in data
    assert "progressions" in data
    # metadata should be excluded
    assert "metadata" not in data


def test_flow_from_dict():
    """Test Flow deserialization from dict."""
    # Create flow and serialize
    f1 = Flow[FlowTestItem, FlowTestProgression](
        items=[FlowTestItem(value=f"item{i}") for i in range(3)],
        name="test",
    )
    prog = FlowTestProgression(name="prog1")
    f1.add_progression(prog)

    data = to_dict(f1)

    # Deserialize
    f2 = Flow.from_dict(data)

    # Check items pile is deserialized
    assert isinstance(f2.items, Pile)
    assert f2.name == "test"
    # Note: Flow.from_dict only deserializes Element fields, not items/progressions
    # Those need to be handled separately by subclasses if needed


def test_flow_from_dict_with_piles_as_dicts():
    """Test Flow deserialization with items and progressions as dicts."""
    data = {
        "id": str(UUID("12345678-1234-5678-1234-567812345678")),
        "name": "test",
        "items": {
            "id": str(UUID("87654321-4321-8765-4321-876543218765")),
            "items": [],
            "item_type": None,
            "strict_type": False,
        },
        "progressions": {
            "id": str(UUID("11111111-1111-1111-1111-111111111111")),
            "items": [],
            "item_type": None,
            "strict_type": False,
        },
    }

    f = Flow.from_dict(data)
    assert isinstance(f.items, Pile)
    assert isinstance(f.progressions, Pile)
    assert f.name == "test"


def test_flow_progression_names_persisted_after_deserialization():
    """Test _progression_names index is rebuilt after deserialization.

    Critical Bug Fix:
        The _progression_names dict is a PrivateAttr (not serialized). Without
        model_post_init() rebuilding it from progressions, name-based access fails
        after deserialization with KeyError.

    Design Pattern:
        Pydantic model_post_init() hook rebuilds derived state after deserialization.
        This is standard for caching/indexing structures that aren't persisted.

    Verification:
        1. Create Flow with named progressions
        2. Serialize to dict
        3. Deserialize from dict
        4. Verify name-based access works (get_progression by name)

    This test catches the blocking bug identified by architect + tester reviews.
    """
    from lionherd_core.libs.string_handlers._to_dict import to_dict

    # Create flow with named progressions
    f1 = Flow[FlowTestItem, FlowTestProgression](name="workflow")
    prog1 = FlowTestProgression(name="stage1")
    prog2 = FlowTestProgression(name="stage2")
    prog3 = FlowTestProgression(name="stage3")
    f1.add_progression(prog1)
    f1.add_progression(prog2)
    f1.add_progression(prog3)

    # Verify name index is populated
    assert len(f1._progression_names) == 3
    assert "stage1" in f1._progression_names
    assert "stage2" in f1._progression_names
    assert "stage3" in f1._progression_names

    # Serialize
    data = to_dict(f1)

    # Deserialize
    f2 = Flow.from_dict(data)

    # CRITICAL: Verify name index is rebuilt (was broken before model_post_init)
    assert len(f2._progression_names) == 3
    assert "stage1" in f2._progression_names
    assert "stage2" in f2._progression_names
    assert "stage3" in f2._progression_names

    # Verify name-based access works
    retrieved_prog1 = f2.get_progression("stage1")
    retrieved_prog2 = f2.get_progression("stage2")
    retrieved_prog3 = f2.get_progression("stage3")

    assert retrieved_prog1.name == "stage1"
    assert retrieved_prog2.name == "stage2"
    assert retrieved_prog3.name == "stage3"

    # Verify UUIDs match
    assert retrieved_prog1.id == prog1.id
    assert retrieved_prog2.id == prog2.id
    assert retrieved_prog3.id == prog3.id


# ==================== Integration Tests ====================


def test_flow_end_to_end_workflow():
    """Test complete Flow workflow demonstrating state machine pattern.

    Workflow State Machine Lifecycle:
        1. Define workflow: Create Flow container
        2. Define states: Add named progressions (stages)
        3. Add work units: Items to shared pile
        4. Assign states: Items to progressions (state membership)
        5. Query states: Access progressions by name
        6. Evolve workflow: Remove obsolete stages

    Pattern Demonstrated:
        ```python
        # Workflow definition
        flow = Flow(name="deployment_pipeline")
        flow.add(Progression(name="pending"))  # Stage 1
        flow.add(Progression(name="deploying"))  # Stage 2

        # Work units
        tasks = [Task(...) for _ in range(5)]
        for task in tasks:
            flow.items.add(task)

        # State assignment
        for task in tasks[:3]:
            flow.get_progression("pending").append(task.id)  # 3 tasks pending

        # State transitions (move between progressions)
        task_id = flow.get_progression("pending").order[0]
        flow.get_progression("pending").remove(task_id)
        flow.get_progression("deploying").append(task_id)  # Transition to deploying

        # State queries
        pending_count = len(flow.get_progression("pending"))
        deploying_tasks = [flow.items[id] for id in flow.get_progression("deploying").order]
        ```

    Design Notes:
        - Items in pile persist across progression changes (single source of truth)
        - Progressions reference items by UUID (enables state transitions)
        - Named access enables ergonomic workflow queries
        - Progression lifecycle independent from items (can remove stages)
    """
    # Create flow
    f = Flow[FlowTestItem, FlowTestProgression](name="workflow")

    # Add items
    items = [FlowTestItem(value=f"item{i}") for i in range(5)]
    for item in items:
        f.items.add(item)

    # Create progressions
    prog1 = FlowTestProgression(name="stage1")
    prog2 = FlowTestProgression(name="stage2")
    f.add_progression(prog1)
    f.add_progression(prog2)

    # Add items to progressions
    for item in items[:3]:
        prog1.append(item.id)
    for item in items[2:]:
        prog2.append(item.id)

    # Verify structure
    assert len(f.items) == 5
    assert len(f.progressions) == 2
    assert len(prog1) == 3
    assert len(prog2) == 3

    # Access by name
    assert f.get_progression("stage1") is prog1
    assert f.get_progression("stage2") is prog2

    # Remove progression by name
    removed = f.remove_progression("stage1")
    assert removed is prog1
    assert len(f.progressions) == 1
    assert "stage1" not in f._progression_names


def test_flow_with_multiple_item_types():
    """Test Flow with multiple item types in pile."""

    class ItemA(Element):
        value_a: str = "a"

    class ItemB(Element):
        value_b: str = "b"

    f = Flow[Element, FlowTestProgression](
        item_type=[ItemA, ItemB],
        strict_type=False,
    )

    # Add different types
    item_a = ItemA()
    item_b = ItemB()
    f.items.add(item_a)
    f.items.add(item_b)

    assert len(f.items) == 2


def test_flow_progression_order_independence():
    """Test Flow progressions have independent ordering from items pile.

    Workflow Ordering Semantics:
        Each progression maintains its own order, independent from pile insertion order.
        This enables different views of the same items for different workflow contexts.

    Use Cases:
        1. Priority ordering:
           - Pile: items in creation order
           - "high_priority" progression: sorted by urgency
           - "low_priority" progression: sorted by effort

        2. Multi-stage processing:
           - Pile: all tasks (insertion order)
           - "deploy_order" progression: sorted by dependencies
           - "test_order" progression: sorted by risk

        3. Team views:
           - Pile: all work items (chronological)
           - "frontend_team" progression: UI tasks by priority
           - "backend_team" progression: API tasks by sprint

    Architecture:
        - Pile._progression: Internal ordering (insertion/addition order)
        - Named progressions: Workflow-specific ordering
        - Both reference same items, different order lists
    """
    f = Flow[FlowTestItem, FlowTestProgression]()

    # Add items
    items = [FlowTestItem(value=f"item{i}") for i in range(3)]
    for item in items:
        f.items.add(item)

    # Create progression with different order
    prog = FlowTestProgression(name="custom_order")
    prog.append(items[2].id)
    prog.append(items[0].id)
    prog.append(items[1].id)
    f.add_progression(prog)

    # Verify progression order is independent
    assert list(prog.order) == [items[2].id, items[0].id, items[1].id]
    assert list(f.items._progression.order) == [items[0].id, items[1].id, items[2].id]


# ==================== Error Handling Tests ====================


def test_flow_add_item_duplicate_raises():
    """Test adding duplicate item raises ValueError."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    item = FlowTestItem(value="test")
    f.add_item(item)

    with pytest.raises(ValueError, match="already exists"):
        f.add_item(item)


def test_flow_remove_nonexistent_progression_raises():
    """Test removing nonexistent progression raises ValueError."""
    f = Flow[FlowTestItem, FlowTestProgression]()

    with pytest.raises(ValueError, match="not found"):
        f.remove_progression(UUID("12345678-1234-5678-1234-567812345678"))


def test_flow_remove_nonexistent_item_raises():
    """Test removing nonexistent item raises ValueError."""
    f = Flow[FlowTestItem, FlowTestProgression]()

    with pytest.raises(ValueError, match="not found"):
        f.remove_item(UUID("12345678-1234-5678-1234-567812345678"))


def test_flow_add_item_invalid_progression_raises():
    """Test adding item to nonexistent progression raises error."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    item = FlowTestItem(value="test")

    # Should raise when trying to access nonexistent progression
    with pytest.raises((ValueError, KeyError)):
        f.add_item(item, progression_ids="nonexistent")


# ==================== ExceptionGroup Tests ====================


def test_flow_exception_group_collection():
    """Test ExceptionGroup for batch workflow error handling.

    Workflow Error Handling Pattern:
        In batch workflows, individual operation failures shouldn't stop processing.
        Instead, collect all errors and report them together using ExceptionGroup.

    Use Cases:
        1. Bulk item insertion:
           - Try adding 100 tasks
           - 3 fail validation (duplicates, invalid state)
           - Report all 3 failures together, not just first

        2. Multi-progression updates:
           - Transition 50 items to "completed"
           - Some items missing from source progression
           - Collect all failures, report which items couldn't transition

        3. Workflow integrity checks:
           - Validate all items have required metadata
           - Multiple violations found
           - Report all violations for batch fixing

    Pattern:
        ```python
        errors = []
        for item in batch:
            try:
                flow.add_item(item, progression_ids="stage")
            except ValueError as e:
                errors.append(e)  # Collect, don't raise immediately

        if errors:
            raise ExceptionGroup("Batch validation errors", errors)
        ```

    Why ExceptionGroup:
        - Preserves all error context (individual failures)
        - Enables batch retry strategies (retry failed subset)
        - Better error reporting (see all issues at once)
        - Pythonic (built-in in 3.11+, backported to 3.9+)
    """

    def collect_errors():
        f = Flow[FlowTestItem, FlowTestProgression]()
        errors = []

        # Try adding duplicate items
        item1 = FlowTestItem(value="item1")
        f.add_item(item1)
        try:
            f.add_item(item1)
        except ValueError as e:
            errors.append(e)

        # Try removing nonexistent item
        try:
            f.remove_item(UUID("12345678-1234-5678-1234-567812345678"))
        except ValueError as e:
            errors.append(e)

        # Try adding progression with duplicate name
        prog1 = FlowTestProgression(name="duplicate")
        f.add_progression(prog1)
        try:
            prog2 = FlowTestProgression(name="duplicate")
            f.add_progression(prog2)
        except ValueError as e:
            errors.append(e)

        # Raise ExceptionGroup if any errors
        if errors:
            raise ExceptionGroup("Multiple Flow validation errors", errors)

    with pytest.raises(ExceptionGroup) as exc_info:
        collect_errors()

    eg = exc_info.value
    assert len(eg.exceptions) == 3
    assert all(isinstance(e, ValueError) for e in eg.exceptions)


# ==================== Async-Related Tests ====================


@pytest.mark.asyncio
async def test_flow_with_async_operations():
    """Test Flow pile supports async operations for concurrent workflows.

    Async Workflow Pattern:
        Flow.pile provides async methods (add_async, get_async) for concurrent
        workflow execution. This enables non-blocking operations when workflows
        involve I/O (database, network, file system).

    Use Cases:
        1. I/O-bound workflows:
           - Fetch items from database concurrently
           - Add to flow without blocking other operations

        2. Distributed workflows:
           - Items pulled from remote queue
           - Process and add to flow asynchronously

        3. Reactive workflows:
           - Listen to event stream
           - Add items to flow as events arrive

    Pattern:
        ```python
        async def process_stream(flow, items):
            for item in items:
                await flow.items.add_async(item)  # Non-blocking
                # Continue processing while item is added
        ```

    Thread Safety:
        Pile uses threading.RLock for thread-safe async operations.
        Multiple coroutines can safely add/get items concurrently.
    """
    f = Flow[FlowTestItem, FlowTestProgression]()

    # Use async operations directly (without context manager)
    item = FlowTestItem(value="async_test")
    await f.items.add_async(item)
    assert len(f.items) == 1

    # Verify async get
    retrieved = await f.items.get_async(item.id)
    assert retrieved is item


@pytest.mark.asyncio
async def test_flow_concurrent_operations():
    """Test Flow handles concurrent batch operations correctly.

    Concurrent Workflow Pattern:
        Batch workflows can process multiple items concurrently using gather().
        This pattern maximizes throughput for I/O-bound workflow operations.

    Use Cases:
        1. Batch task ingestion:
           - Receive 100 tasks from API
           - Add all to flow concurrently
           - Complete in time of slowest operation, not sum of all

        2. Multi-source workflows:
           - Pull items from multiple queues simultaneously
           - Aggregate into single flow
           - Maintain ordering per source (progressions)

        3. Parallel state checks:
           - Check status of 50 running jobs concurrently
           - Update flow based on results
           - Don't block on sequential checks

    Pattern:
        ```python
        async def batch_add(flow, items):
            async def add_one(item):
                await flow.items.add_async(item)
                return item.id

            # All adds happen concurrently
            ids = await gather(*[add_one(item) for item in items])
            return ids
        ```

    Performance:
        For 10 items with 100ms I/O each:
        - Sequential: 10 * 100ms = 1000ms
        - Concurrent: max(100ms operations) ≈ 100ms
        10x throughput improvement
    """
    from lionherd_core.libs.concurrency import gather

    f = Flow[FlowTestItem, FlowTestProgression]()

    # Create items concurrently
    items = [FlowTestItem(value=f"item{i}") for i in range(10)]

    # Add items to pile concurrently
    async def add_item(item):
        await f.items.add_async(item)

    await gather(*[add_item(item) for item in items])

    assert len(f.items) == 10


@pytest.mark.asyncio
async def test_flow_async_operations_with_progressions():
    """Test Flow async operations with multi-stage workflow progressions.

    Concurrent Multi-Stage Workflow:
        Combine async operations with progressions to build concurrent
        multi-stage workflows where items flow through stages asynchronously.

    Use Cases:
        1. Pipeline workflows:
           - Items added concurrently to "intake" stage
           - Worker coroutines move items through stages
           - Each stage processes independently and concurrently

        2. Fan-out/fan-in:
           - Single input progression
           - Multiple processing progressions (parallel stages)
           - Results aggregated in output progression

        3. Priority lanes:
           - High/medium/low priority progressions
           - Items routed based on priority
           - Each lane processes concurrently

    Pattern:
        ```python
        async def multi_stage_pipeline(flow, items):
            # Stage 1: Concurrent ingestion
            await gather(*[flow.items.add_async(item) for item in items])

            # Stage 2: Assign to stages (can be concurrent)
            for item in items:
                stage = determine_stage(item)
                flow[stage].append(item.id)

            # Stage 3: Process each stage concurrently
            results = await gather(
                *[process_stage(flow[stage]) for stage in ["stage1", "stage2", "stage3"]]
            )
        ```

    Architecture Benefits:
        - Progressions isolate stages (failure in one doesn't affect others)
        - Async enables concurrent processing across stages
        - Shared pile enables zero-copy state transitions
    """
    from lionherd_core.libs.concurrency import gather

    f = Flow[FlowTestItem, FlowTestProgression]()

    # Add progressions
    progs = [FlowTestProgression(name=f"prog{i}") for i in range(3)]
    for prog in progs:
        f.add_progression(prog)

    # Add items concurrently
    items = [FlowTestItem(value=f"item{i}") for i in range(5)]

    async def add_item(item):
        await f.items.add_async(item)

    await gather(*[add_item(item) for item in items])

    # Verify structure
    assert len(f.items) == 5
    assert len(f.progressions) == 3

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
    flow.add_item(task, progressions="pending")

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
            flow.add_item(item, progressions="stage1")
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
    Flow supports async context manager for thread-safe operations:

    ```python
    async def process_batch(items):
        async with flow.items:
            for item in items:
                flow.add_item(item)  # Optionally: flow.add_item(item, progressions="stage1")
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

from uuid import UUID, uuid4

import pytest

from lionherd_core.base import Element, Flow, Pile, Progression
from lionherd_core.errors import ExistsError, NotFoundError
from lionherd_core.ln import to_dict

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

    # Mock info object
    class MockInfo:
        field_name = "items"

    result = Flow._validate_piles(pile_dict, mock_handler, MockInfo())
    assert isinstance(result, Pile)


def test_flow_validate_piles_preserves_pile():
    """Test _validate_piles delegates to handler for non-dict inputs."""
    pile = Pile[FlowTestItem]()

    # Validator delegates to handler for non-dict inputs
    def mock_handler(v):  # Mock handler that returns input
        return v

    # Mock info object
    class MockInfo:
        field_name = "items"

    result = Flow._validate_piles(pile, mock_handler, MockInfo())
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
    """Test adding progression with duplicate name raises ExistsError.

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
    with pytest.raises(ExistsError, match="Progression with name 'duplicate' already exists"):
        f.add_progression(prog2)


def test_flow_add_progression_without_name():
    """Test adding progression without name (no name registration)."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name=None)

    f.add_progression(prog)
    assert len(f.progressions) == 1
    assert prog.id in f.progressions
    assert len(f._progression_names) == 0


def test_flow_add_progression_validates_referential_integrity():
    """Test adding progression with invalid UUIDs raises NotFoundError.

    Design Rationale - Fail Fast:
        Validate referential integrity BEFORE adding to pile to prevent
        inconsistent state. If validation happens after adding, a failure
        would leave the progression in the pile but unusable.

    Consistency with __init__:
        Both __init__ (via @model_validator) and add_progression() enforce
        the same referential integrity constraint: all progression UUIDs
        must exist in items pile.

    Related:
        - Issue #164: Bug fix for validation order
        - PR #162: Added @model_validator for __init__ validation
    """
    # Create flow with items
    items = [FlowTestItem(value=f"item{i}") for i in range(3)]
    f = Flow[FlowTestItem, FlowTestProgression](items=items)

    # Create progression with non-existent UUID
    missing_uuid = uuid4()
    prog = FlowTestProgression(name="invalid_prog", order=[missing_uuid])

    # Should raise NotFoundError before adding to pile
    with pytest.raises(NotFoundError, match="contains UUIDs not in items pile"):
        f.add_progression(prog)

    # Verify progression was NOT added to pile (no inconsistent state)
    assert len(f.progressions) == 0
    assert "invalid_prog" not in f._progression_names


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
        flow.add_item(task, progressions="pending")  # Assign to state
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
    f.add_item(item, progressions=prog.id)

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
    f.add_item(item, progressions=[prog1.id, prog2.id])

    assert item.id in f.items
    assert item.id in prog1
    assert item.id in prog2


def test_flow_add_item_by_progression_name():
    """Test adding item using progression name."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog = FlowTestProgression(name="test_prog")
    f.add_progression(prog)

    item = FlowTestItem(value="test")
    f.add_item(item, progressions="test_prog")

    assert item.id in prog


def test_flow_remove_item_from_pile_and_progressions():
    """Test removing item automatically removes from all progressions.

    Design Philosophy - Data Integrity:
        Removing an item from the flow always removes it from all progressions
        to prevent dangling references. This ensures referential integrity and
        follows the principle of least surprise.

    Why Always Cascade:
        1. **Prevents bugs**: Dangling UUID references would cause KeyError later
        2. **Clear semantics**: remove_item() means "fully remove from flow"
        3. **Simpler API**: No parameter to think about

    Alternative Pattern - Soft Delete:
        If you need to preserve audit trail while marking items inactive:
        ```python
        # Keep item in flow but mark as deleted
        item.metadata["deleted"] = True
        flow.items[item.id] = item  # Update in place
        # Progressions still reference it for history
        ```

    Performance:
        O(P) where P is number of progressions. Each progression is scanned once.
    """
    f = Flow[FlowTestItem, FlowTestProgression]()
    prog1 = FlowTestProgression(name="prog1")
    prog2 = FlowTestProgression(name="prog2")
    f.add_progression(prog1)
    f.add_progression(prog2)

    item = FlowTestItem(value="test")
    f.add_item(item, progressions=[prog1.id, prog2.id])

    removed = f.remove_item(item.id)
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
    from lionherd_core.ln import to_dict

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

    with pytest.raises(ExistsError, match="already exists"):
        f.add_item(item)


def test_flow_remove_nonexistent_progression_raises():
    """Test removing nonexistent progression raises NotFoundError."""
    f = Flow[FlowTestItem, FlowTestProgression]()

    with pytest.raises(NotFoundError, match="not found"):
        f.remove_progression(UUID("12345678-1234-5678-1234-567812345678"))


def test_flow_remove_nonexistent_item_raises():
    """Test removing nonexistent item raises NotFoundError."""
    f = Flow[FlowTestItem, FlowTestProgression]()

    with pytest.raises(NotFoundError, match="not found"):
        f.remove_item(UUID("12345678-1234-5678-1234-567812345678"))


def test_flow_add_item_invalid_progression_raises():
    """Test adding item to nonexistent progression raises error."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    item = FlowTestItem(value="test")

    # Should raise when trying to access nonexistent progression
    with pytest.raises((ValueError, KeyError)):
        f.add_item(item, progressions="nonexistent")


# ==================== Exception Transformation Tests ====================


def test_flow_add_item_raises_existserror():
    """Test add_item raises ExistsError when item already exists."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    item = FlowTestItem(value="test")
    f.add_item(item)

    # Adding again should raise ExistsError
    with pytest.raises(ExistsError, match=f"Item {item.id} already exists"):
        f.add_item(item)


def test_flow_remove_item_raises_notfounderror():
    """Test remove_item raises NotFoundError from pile."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    fake_id = UUID("12345678-1234-5678-1234-567812345678")

    # Should raise NotFoundError from pile
    with pytest.raises(NotFoundError, match=f"Item {fake_id} not found in pile"):
        f.remove_item(fake_id)


def test_flow_remove_progression_raises_notfounderror_with_metadata():
    """Test remove_progression raises NotFoundError with preserved metadata."""
    f = Flow[FlowTestItem, FlowTestProgression]()
    fake_id = UUID("12345678-1234-5678-1234-567812345678")

    # Should raise NotFoundError with better message
    with pytest.raises(NotFoundError, match=f"Progression {fake_id} not found in flow"):
        f.remove_progression(fake_id)

    # Verify metadata is preserved via __cause__
    try:
        f.remove_progression(fake_id)
    except NotFoundError as e:
        assert e.__cause__ is not None
        assert hasattr(e, "details")
        assert hasattr(e, "retryable")


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
                flow.add_item(item, progressions="stage")
            except (ExistsError, NotFoundError) as e:
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

        # Try adding duplicate items (raises ExistsError)
        item1 = FlowTestItem(value="item1")
        f.add_item(item1)
        try:
            f.add_item(item1)
        except ExistsError as e:
            errors.append(e)

        # Try removing nonexistent item (raises NotFoundError)
        try:
            f.remove_item(UUID("12345678-1234-5678-1234-567812345678"))
        except NotFoundError as e:
            errors.append(e)

        # Try adding progression with duplicate name (raises ExistsError - name uniqueness check)
        prog1 = FlowTestProgression(name="duplicate")
        f.add_progression(prog1)
        try:
            prog2 = FlowTestProgression(name="duplicate")
            f.add_progression(prog2)
        except ExistsError as e:
            errors.append(e)

        # Raise ExceptionGroup if any errors
        if errors:
            raise ExceptionGroup("Multiple Flow validation errors", errors)

    with pytest.raises(ExceptionGroup) as exc_info:
        collect_errors()

    eg = exc_info.value
    assert len(eg.exceptions) == 3
    # Mixed exception types: ExistsError, NotFoundError, ExistsError
    assert isinstance(eg.exceptions[0], ExistsError)
    assert isinstance(eg.exceptions[1], NotFoundError)
    assert isinstance(eg.exceptions[2], ExistsError)


# ==================== Async-Related Tests ====================


@pytest.mark.asyncio
async def test_flow_with_async_operations():
    """Test Flow pile supports async operations for concurrent workflows.

    Async Workflow Pattern:
        Flow.pile supports async context manager for concurrent workflow
        execution. This enables non-blocking operations when workflows involve
        I/O (database, network, file system).

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
                async with flow.items:
                    flow.add_item(item)
                # Continue processing after adding item
        ```

    Thread Safety:
        Pile uses asyncio.Lock for thread-safe async operations.
        Multiple coroutines safely coordinate via context manager.
    """
    f = Flow[FlowTestItem, FlowTestProgression]()

    # Use async context manager for thread-safe operations
    item = FlowTestItem(value="async_test")
    async with f.items:
        f.items.add(item)

    assert len(f.items) == 1

    # Verify async get
    async with f.items:
        retrieved = f.items.get(item.id)

    assert retrieved is item


@pytest.mark.asyncio
async def test_flow_concurrent_operations():
    """Test Flow handles concurrent lock acquisition correctly.

    Concurrent Lock Pattern:
        Multiple coroutines can safely acquire the async lock using context manager.
        Operations are serialized by the lock, ensuring thread safety.

    Use Cases:
        1. Multiple async workers:
           - Multiple coroutines processing items
           - Each acquires lock when adding results
           - Lock ensures no race conditions

        2. Concurrent readers/writers:
           - Some coroutines adding items
           - Others reading items
           - Lock coordinates access

    Pattern:
        ```python
        async def worker(flow, item):
            async with flow.items:
                flow.add_item(item)
                # Do other work with exclusive access


        # All workers compete for lock
        await gather(*[worker(flow, item) for item in items])
        ```

    Thread Safety:
        Lock ensures operations are serialized even when called concurrently.
    """
    from lionherd_core.libs.concurrency import gather

    f = Flow[FlowTestItem, FlowTestProgression]()
    items = [FlowTestItem(value=f"item{i}") for i in range(10)]

    # Multiple coroutines concurrently acquiring lock
    async def add_with_lock(item):
        async with f.items:
            f.items.add(item)

    await gather(*[add_with_lock(item) for item in items])

    assert len(f.items) == 10


@pytest.mark.asyncio
async def test_flow_async_operations_with_progressions():
    """Test Flow async operations with multi-stage workflow progressions.

    Concurrent Multi-Stage Workflow:
        Multiple coroutines can safely add items and update progressions
        using async context manager for coordination.

    Use Cases:
        1. Pipeline workflows:
           - Multiple workers adding items with lock coordination
           - Worker coroutines move items through stages
           - Lock ensures consistent state

        2. Fan-out/fan-in:
           - Single input progression
           - Multiple workers processing stages
           - Lock coordinates updates

        3. Priority lanes:
           - High/medium/low priority progressions
           - Items routed based on priority
           - Lock ensures atomic routing

    Pattern:
        ```python
        async def worker(flow, item):
            async with flow.items:
                stage = determine_stage(item)
                flow.add_item(item, progression_id=stage)


        # Multiple workers coordinate via lock
        await gather(*[worker(flow, item) for item in items])
        ```

    Architecture Benefits:
        - Progressions isolate stages (failure in one doesn't affect others)
        - Async context manager ensures thread safety
        - Shared pile enables zero-copy state transitions
    """
    from lionherd_core.libs.concurrency import gather

    f = Flow[FlowTestItem, FlowTestProgression]()

    # Add progressions
    progs = [FlowTestProgression(name=f"prog{i}") for i in range(3)]
    for prog in progs:
        f.add_progression(prog)

    # Multiple workers adding items with lock coordination
    items = [FlowTestItem(value=f"item{i}") for i in range(5)]

    async def add_with_lock(item):
        async with f.items:
            f.items.add(item)

    await gather(*[add_with_lock(item) for item in items])

    # Verify structure
    assert len(f.items) == 5
    assert len(f.progressions) == 3


# ==================== Coverage Tests for Missing Lines ====================


def test_flow_init_with_pile_instance():
    """Test passing a Pile instance directly to Flow __init__ (line 80)."""
    # Create a pre-populated Pile
    pile = Pile[FlowTestItem]()
    item1 = FlowTestItem(value="item1")
    item2 = FlowTestItem(value="item2")
    pile.add(item1)
    pile.add(item2)

    # Pass Pile instance directly
    flow = Flow[FlowTestItem, FlowTestProgression](items=pile)

    # Verify it uses the same Pile instance
    assert flow.items is pile
    assert len(flow.items) == 2
    assert item1.id in flow.items
    assert item2.id in flow.items


def test_flow_init_with_single_element():
    """Test passing a single Element to Flow __init__ (line 87)."""
    item = FlowTestItem(value="single")

    # Pass single Element (not in a list)
    flow = Flow[FlowTestItem, FlowTestProgression](items=item)

    # Verify it was normalized to a list and added
    assert len(flow.items) == 1
    assert item.id in flow.items
    assert flow.items[item.id] == item


def test_flow_field_validator_with_list_of_elements():
    """Test field validator with list of Element instances (lines 112-122)."""
    item1 = FlowTestItem(value="item1")
    item2 = FlowTestItem(value="item2")

    # Create Flow with list of elements directly to validator
    # (bypassing __init__ logic by using pydantic constructor)
    flow = Flow[FlowTestItem, FlowTestProgression].model_validate(
        {"items": [item1, item2], "progressions": []}
    )

    assert len(flow.items) == 2
    assert item1.id in flow.items
    assert item2.id in flow.items


def test_flow_field_validator_with_list_of_dicts():
    """Test field validator with list[dict] deserialization (lines 112-122)."""
    # Create serialized Flow dict that will go through deserialization path
    item1 = FlowTestItem(value="item1")
    item2 = FlowTestItem(value="item2")

    # Serialize items to dicts
    flow_dict = {
        "id": str(uuid4()),
        "created_at": "2024-01-01T00:00:00Z",
        "items": [item1.to_dict(), item2.to_dict()],
        "progressions": [],
    }

    # Deserialize - field validator should handle list[dict]
    flow = Flow[FlowTestItem, FlowTestProgression].from_dict(flow_dict)

    assert len(flow.items) == 2
    # Verify deserialized correctly
    for item in flow.items:
        assert isinstance(item, Element)
        assert hasattr(item, "value")


def test_flow_referential_integrity_validation_failure():
    """Test model validator raises NotFoundError for missing UUIDs (line 137)."""
    item1 = FlowTestItem(value="item1")
    missing_uuid = uuid4()  # UUID not in items

    # Create progression with missing UUID
    prog = FlowTestProgression(name="test", order=[item1.id, missing_uuid])

    # Should raise NotFoundError during validation
    with pytest.raises(NotFoundError) as exc_info:
        Flow[FlowTestItem, FlowTestProgression](items=[item1], progressions=[prog])

    assert "not in items pile" in str(exc_info.value)
    assert str(missing_uuid) in str(exc_info.value)


def test_flow_check_item_exists_error_path():
    """Test _check_item_exists re-raises NotFoundError with context (lines 153-156)."""
    flow = Flow[FlowTestItem, FlowTestProgression]()
    missing_uuid = uuid4()

    # Call _check_item_exists with missing UUID
    with pytest.raises(NotFoundError) as exc_info:
        flow._check_item_exists(missing_uuid)

    # Verify error message includes flow context
    assert "not found in flow" in str(exc_info.value)
    assert str(missing_uuid) in str(exc_info.value)


def test_flow_field_validator_default_factory():
    """Test field validator default factory path (line 125).

    When neither items nor progressions are provided (not even None),
    Pydantic should call default_factory=Pile, which goes through
    handler(v) path in the validator.
    """
    # Create Flow using model_validate with minimal fields
    # This forces Pydantic to use default_factory for items/progressions
    flow = Flow[FlowTestItem, FlowTestProgression].model_validate(
        {
            "id": str(uuid4()),
            "created_at": "2024-01-01T00:00:00Z",
            # No items or progressions fields -> default_factory triggered
        }
    )

    # Verify default factories created empty Piles
    assert isinstance(flow.items, Pile)
    assert isinstance(flow.progressions, Pile)
    assert len(flow.items) == 0
    assert len(flow.progressions) == 0

# Flow

> Dual-pile workflow state machine with named progression stages and shared item storage.

---

> **⚠️ Migration Notice**: Breaking changes in v1.0.0-alpha5. See the [comprehensive migration guide](../../migration/v1.0.0-alpha5.md) for detailed upgrade instructions.

---

## Overview

**Flow** is a state machine built from two specialized [Pile](pile.md) instances:

- **`progressions`**: Named workflow stages (Progression instances in ordered sequence)
- **`items`**: Shared work units (Element instances referenced by UUID)

This dual-pile architecture enables M:N relationships where items can exist in multiple workflow stages simultaneously while maintaining a single source of truth for item data.

**Thread Safety**: All operations are thread-safe via RLock synchronization.

---

## When to Use Flow

**Use Flow when you need:**

- **Multi-stage workflows**: Task pipelines, deployment stages, approval processes
- **M:N state relationships**: Items in multiple stages (e.g., task in "testing" and "documentation")

- **Named stage access**: Ergonomic `flow.get_progression("pending")` vs UUID lookup
- **Independent lifecycles**: Items persist when removed from stages
- **Workflow introspection**: Query all stages, count items per stage

**Don't use Flow when:**

- Single linear sequence is sufficient → use [Progression](progression.md)
- No named stages needed → use [Pile](pile.md) directly
- Simple state enum is adequate → use Element with state field

---

## Migration Guide (v1.0.0a4 → v1.0.0a5)

### 1. Constructor - New `progressions` Parameter

```python
# Before: flow.add_progression(Progression(name="pending"))
# After:  Flow(progressions=[Progression(name="pending")])
```

### 2. Type Configuration - Frozen Fields

```python
# item_type/strict_type now properly applied to items Pile and frozen
flow = Flow(items=[task1], item_type=Task, strict_type=True)
```

### 3. Referential Integrity Validation

```python
# Validates progression UUIDs exist in items at construction
# Raises NotFoundError if progression contains UUIDs not in items
```

### 4. add_item() Parameter Renamed

```python
# Before: flow.add_item(task, progression_ids="pending")
# After:
flow.add_item(task, progressions="pending")  # Parameter renamed
flow.add_item(task, progressions=["pending", "active"])
```

### 5. remove_item() - Always Removes from Progressions

```python
# Before: flow.remove_item(task_id, remove_from_progressions=False)
# After:  flow.remove_item(task_id)  # Always removes from pile AND progressions
```

---

## Class Signature

```python
class Flow(Element, Generic[E, P]):
    """Workflow state machine with ordered progressions and referenced items."""
```

**Generic Parameters:**

- `E`: Element type for items (work units)
- `P`: Progression type for workflow stages

---

## Constructor

```python
def __init__(
    self,
    items: list[E] | Pile[E] | Element | None = None,
    progressions: list[P] | Pile[P] | None = None,  # New in v1.0.0a5 (PR #156)
    name: str | None = None,
    item_type: type[E] | set[type] | list[type] | None = None,
    strict_type: bool = False,
    **data: Any,  # id, created_at, metadata
) -> None: ...
```

**Parameters:**

- `items` (list[E] | Pile[E] | Element | None): Initial items to add to items pile
- **`progressions` (list[P] | Pile[P] | None)**: Initial workflow stages (progressions). **New in v1.0.0a5**
- `name` (str | None): Flow identifier (e.g., "deployment_pipeline")
- `item_type` (type[E] | set[type] | list[type] | None): Type constraint for items pile validation (creates configured Pile with frozen fields)
- `strict_type` (bool): Enforce exact type match (no subclasses) for items. Default: False
- `**data`: Additional Element fields (id, created_at, metadata)

**⚠️ BREAKING CHANGE (PR #156):**

- Constructor now accepts `progressions` parameter for initializing workflow stages upfront
- `item_type` and `strict_type` are now correctly applied to items Pile (frozen fields)
- Referential integrity is validated at construction via `@model_validator` (all progression UUIDs must exist in items)

**Example:**

```python
from lionherd_core.base import Flow, Progression, Element

class Task(Element):
    description: str = "task"
    priority: int = 0

task1 = Task(description="task1")
task2 = Task(description="task2")

# Create flow with type validation and initial progressions
flow = Flow[Task, Progression](
    items=[task1, task2],
    progressions=[
        Progression(name="pending", order=[task1.id]),
        Progression(name="active", order=[task2.id])
    ],
    name="workflow",
    item_type=Task,
    strict_type=False
)

# Referential integrity validated at construction
# All UUIDs in progressions must exist in items pile
```

---

## Attributes

### Core Piles

#### `progressions`

```python
progressions: Pile[P]
```

Pile of Progression instances representing workflow stages. Each progression should have a unique `name` for O(1) named access.

**Type:** Pile[P] (default: Pile[Progression])

**Access:** Pydantic Field (mutable Pile instance)

#### `items`

```python
items: Pile[E]
```

Pile of Element instances that progressions reference by UUID. Items are the single source of truth for work unit data.

**Type:** Pile[E]

**Access:** Pydantic Field (mutable Pile instance)

### Metadata

#### `name`

```python
name: str | None
```

Optional flow identifier for introspection and debugging.

**Type:** str | None

---

## Methods

### Progression Management

#### `add_progression()`

Add named progression to workflow stages.

**Signature:**

```python
def add_progression(self, progression: P) -> None
```

**Parameters:**

- `progression` (P): Progression instance with unique name

**Raises:**

- `ExistsError`: If progression UUID or name already exists
- `TypeError`: If progression type doesn't match Pile constraint

**Thread Safety:** Synchronized with @synchronized decorator

**Example:**

```python
flow = Flow[Task, Progression](name="workflow")

flow.add_progression(Progression(name="pending"))
flow.add_progression(Progression(name="active"))
flow.add_progression(Progression(name="completed"))

print(len(flow.progressions))  # 3
```

**Note**: Progression names must be unique. Attempting to add a progression with an existing name raises ExistsError.

---

#### `remove_progression()`

Remove progression by UUID or name.

**Signature:**

```python
def remove_progression(self, progression_id: UUID | str | P) -> P
```

**Parameters:**

- `progression_id` (UUID | str | P): Progression UUID, name, or instance

**Returns:** P - The removed progression

**Raises:**

- `NotFoundError`: If progression not found in flow

**Thread Safety:** Synchronized with @synchronized decorator

**Time Complexity:** O(1) for name lookup, O(n) for UUID removal from pile progression

**Example:**

```python
# Remove by name
removed = flow.remove_progression("completed")

# Remove by UUID
removed = flow.remove_progression(prog_uuid)
```

**Note**: Removing a progression does NOT remove referenced items from the items pile. Items persist independently.

---

#### `get_progression()`

Get progression by UUID or name (O(1) for name lookup).

**Signature:**

```python
def get_progression(self, key: UUID | str | P) -> P
```

**Parameters:**

- `key` (UUID | str | P): Progression UUID, name, or instance

**Returns:** P - The progression instance

**Raises:** KeyError if progression not found

**Thread Safety:** Synchronized with @synchronized decorator

**Time Complexity:** O(1) for name lookup (via _progression_names index), O(1) for UUID lookup

**Example:**

```python
# Get by name (most common pattern)
pending = flow.get_progression("pending")
print(f"Pending items: {len(pending)}")

# Get by UUID
prog = flow.get_progression(uuid_obj)
```

**Design Note**: Named access is the primary API pattern. The internal `_progression_names` index provides O(1) lookup without scanning.

---

### Item Management

#### `add_item()`

Add item to items pile and optionally assign to progression stages.

**Signature:**

```python
def add_item(
    self,
    item: E,
    progressions: list[UUID | str] | UUID | str | None = None,  # Renamed in v1.0.0a5
) -> None
```

**Parameters:**

- `item` (E): Element to add to items pile
- **`progressions`** (list[UUID | str] | UUID | str | None): Optional progression(s) to add item to (by UUID or name). **Parameter renamed from `progression_ids` in PR #156**

**Raises:**

- `ExistsError`: If item already exists in items pile
- `KeyError`: If progressions references non-existent progression

**Time Complexity:** O(1) for pile add, O(k) for k progression assignments

**⚠️ BREAKING CHANGE (PR #156):** Parameter renamed `progression_ids` → `progressions`

**Example:**

```python
task = Task(description="Deploy API")

# Add to pile only (no stage assignment)
flow.add_item(task)

# Add to pile and specific stage (parameter name changed)
flow.add_item(task, progressions="pending")  # Before: progression_ids="pending"

# Add to multiple stages (M:N relationship)
flow.add_item(task, progressions=["testing", "documentation"])
```

**Pattern**: Items exist in pile independently of progression membership. This enables flexible state transitions without data duplication.

---

#### `remove_item()`

Remove item from items pile and all progressions.

**Signature:**

```python
def remove_item(
    self,
    item_id: UUID | str | Element,
) -> E
```

**Parameters:**

- `item_id` (UUID | str | Element): Item UUID, string UUID, or Element instance

**Returns:** E - The removed item

**Raises:**

- `NotFoundError`: If item not found in flow

**Time Complexity:** O(p × n) where p = number of progressions, n = items per progression

**⚠️ BREAKING CHANGE (PR #156):** `remove_from_progressions` parameter removed. Method now **always** removes item from all progressions before removing from pile.

**Example:**

```python
# After PR #156: Always removes from pile + all progressions
removed = flow.remove_item(task_id)

# Before PR #156: Had optional parameter
# removed = flow.remove_item(task_id, remove_from_progressions=False)  # No longer supported
```

**Migration:** If you need to keep item in progressions (not recommended - creates orphan references), manually remove from pile only:

```python
# Not recommended: Manual pile removal (leaves orphan UUIDs in progressions)
removed = flow.items.remove(task_id)
```

---

### Serialization

#### `to_dict()`

Serialize flow to dictionary with proper Pile serialization for both piles.

**Signature:**

```python
def to_dict(
    self,
    mode: Literal["python", "json", "db"] = "python",
    created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
    meta_key: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]
```

**Parameters:**

- `mode`: Serialization mode (python/json/db)
- `created_at_format`: Timestamp format for Flow
- `meta_key`: Rename metadata field (e.g., "node_metadata" for db mode)
- `**kwargs`: Additional serialization options

**Returns:** dict[str, Any]

**Example:**

```python
data = flow.to_dict(mode="json")
# Contains: id, created_at, metadata, name, items (full pile), progressions (full pile)
```

**Implementation Note**: Overrides Element.to_dict() to ensure both `items` and `progressions` Pile fields are fully serialized with their contents, not just metadata.

---

#### `from_dict()`

Deserialize flow from dictionary (inherited from Element, with post-init index rebuild).

**Signature:**

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Flow
```

**Parameters:**

- `data` (dict): Serialized flow data from to_dict()

**Returns:** Flow instance with rebuilt _progression_names index

**Example:**

```python
data = flow.to_dict(mode="json")
restored = Flow.from_dict(data)

# Named access works after deserialization
pending = restored.get_progression("pending")
```

**Implementation Note**: The `model_post_init()` hook rebuilds the `_progression_names` index from deserialized progressions, ensuring O(1) named access after round-trip.

---

## Design Rationale

### Dual-Pile Architecture

**Why two Piles instead of embedded progressions?**

1. **Separation of Concerns**: Structure (progressions) vs data (items)
2. **Independent Lifecycles**: Items persist when removed from stages
3. **Type Safety**: Generic constraints on both E and P types
4. **Serialization**: Each pile handles its own to_dict/from_dict
5. **Performance**: Separate locks avoid contention between structure and data mutations

### UUID References Instead of Embedded Items

**Why UUIDs in progression.order instead of items?**

1. **M:N Relationships**: Same item in multiple progressions without duplication
2. **Lazy Loading**: Can serialize progressions without full item data
3. **Distributed Workflows**: UUIDs can reference items across processes
4. **Memory Efficiency**: Single item instance, multiple references

### Named Progression Access

**Why _progression_names index?**

1. **Ergonomics**: `flow.get_progression("pending")` vs UUID lookup
2. **Domain Mapping**: Stage names match business vocabulary
3. **Introspection**: Enumerate stage names for debugging/UI
4. **O(1) Lookup**: Dictionary index avoids scanning progressions pile

---

## Usage Patterns

### Pattern 1: Task Workflow State Machine

```python
# Define workflow
flow = Flow[Task, Progression](name="tasks")
for stage in ["backlog", "in_progress", "review", "done"]:
    flow.add_progression(Progression(name=stage))

# Add task to backlog
task = Task(description="Implement feature X")
flow.add_item(task, progression_ids="backlog")

# Transition: backlog → in_progress
backlog = flow.get_progression("backlog")
in_progress = flow.get_progression("in_progress")

backlog.remove(task.id)
in_progress.append(task.id)

# Query stage
tasks_in_review = [
    flow.items[uid] for uid in flow.get_progression("review").order
]
```

### Pattern 2: M:N Cross-Cutting Concerns

```python
# Item in multiple stages simultaneously
task = Task(description="Update docs")
flow.add_item(task, progression_ids=["development", "documentation", "translation"])

# Check membership
dev = flow.get_progression("development")
docs = flow.get_progression("documentation")

print(task.id in dev)   # True
print(task.id in docs)  # True
```

### Pattern 3: Deployment Pipeline

```python
# Create pipeline
pipeline = Flow[Deployment, Progression](name="deploy")
for env in ["testing", "staging", "production"]:
    pipeline.add_progression(Progression(name=env))

# Promote high-priority to staging
testing = pipeline.get_progression("testing")
staging = pipeline.get_progression("staging")

for deploy_id in list(testing.order):
    deploy = pipeline.items[deploy_id]
    if deploy.priority == 0:  # High priority
        testing.remove(deploy_id)
        staging.append(deploy_id)
```

### Pattern 4: Workflow Introspection

```python
# Report all stages
print(f"Pipeline: {flow.name}")
for prog in flow.progressions:
    items = [flow.items[uid].description for uid in prog.order]
    print(f"  {prog.name}: {items}")

# Count items per stage
stage_counts = {
    prog.name: len(prog) for prog in flow.progressions
}
```

---

## Common Pitfalls

### Pitfall 1: Forgetting to Remove from Progressions

**Issue**: Removing item from pile but not from progressions creates orphan UUID references.

```python
# ❌ WRONG: Manual pile removal leaves orphan references
removed = flow.items.remove(task_id)
# progressions still reference task_id, will raise NotFoundError on access
```

**Solution**: Use `remove_item()` which ensures clean removal from both pile and progressions.

```python
# ✓ CORRECT: Clean removal from everywhere
flow.remove_item(task_id)  # Removes from pile + all progressions
```

---

### Pitfall 2: Duplicate Progression Names

**Issue**: Adding progressions with duplicate names raises ExistsError.

```python
flow.add_progression(Progression(name="pending"))
flow.add_progression(Progression(name="pending"))  # ❌ ExistsError
```

**Solution**: Check existence before adding or use unique naming scheme.

```python
# ✓ CORRECT: Check before adding
if "pending" not in flow._progression_names:
    flow.add_progression(Progression(name="pending"))
```

---

### Pitfall 3: Modifying Progression Order Without Items Pile

**Issue**: Adding UUIDs to progression.order for items not in flow.items pile.

```python
pending = flow.get_progression("pending")
pending.append(random_uuid)  # ❌ UUID not in flow.items
# Later: flow.items[random_uuid] raises KeyError
```

**Solution**: Always add items to pile first, then reference in progressions.

```python
# ✓ CORRECT: Add to pile, then progression
task = Task(description="New task")
flow.add_item(task, progression_ids="pending")
```

---

## Examples

### Example 1: Complete Task Workflow

```python
from lionherd_core.base import Flow, Progression, Element

class Task(Element):
    description: str = ""
    priority: int = 0

# Create workflow
workflow = Flow[Task, Progression](name="sprint")

# Define stages
for stage in ["todo", "doing", "review", "done"]:
    workflow.add_progression(Progression(name=stage))

# Add tasks
tasks = [
    Task(description="Implement login", priority=1),
    Task(description="Write tests", priority=2),
    Task(description="Deploy", priority=0),
]

for task in tasks:
    workflow.add_item(task, progression_ids="todo")

# Start work on high-priority task
todo = workflow.get_progression("todo")
doing = workflow.get_progression("doing")

high_priority_id = min(
    todo.order,
    key=lambda uid: workflow.items[uid].priority
)
todo.remove(high_priority_id)
doing.append(high_priority_id)

print(f"Now working on: {workflow.items[high_priority_id].description}")
```

---

### Example 2: Multi-Environment Deployment

```python
# Deployment pipeline
pipeline = Flow[Deployment, Progression](name="deploy_v2.0")

# Environments
for env in ["dev", "staging", "prod"]:
    pipeline.add_progression(Progression(name=env))

# Services to deploy
services = [
    Deployment(name="api-gateway"),
    Deployment(name="auth-service"),
    Deployment(name="database"),
]

# Start all in dev
for svc in services:
    pipeline.add_item(svc, progression_ids="dev")

# Promote critical services to staging
dev = pipeline.get_progression("dev")
staging = pipeline.get_progression("staging")

for svc_id in list(dev.order):
    svc = pipeline.items[svc_id]
    if svc.name == "database":  # Critical first
        dev.remove(svc_id)
        staging.append(svc_id)

# Report status
for env_name in ["dev", "staging", "prod"]:
    env = pipeline.get_progression(env_name)
    services = [pipeline.items[uid].name for uid in env.order]
    print(f"{env_name}: {services}")
```

---

### Example 3: M:N Cross-Cutting Concerns

```python
# Feature development with parallel tracks
flow = Flow[Feature, Progression](name="feature_x")

# Define tracks
for track in ["backend", "frontend", "docs", "tests"]:
    flow.add_progression(Progression(name=track))

# Feature spans multiple tracks
feature = Feature(name="User Authentication")
flow.add_item(
    feature,
    progression_ids=["backend", "frontend", "docs"]  # M:N
)

# Check where feature is active
for track_name in ["backend", "frontend", "docs", "tests"]:
    track = flow.get_progression(track_name)
    if feature.id in track:
        print(f"Feature active in {track_name}")
```

---

### Example 4: Workflow Serialization

```python
# Create and populate workflow
flow = Flow[Task, Progression](name="project_alpha")
flow.add_progression(Progression(name="pending"))
flow.add_progression(Progression(name="active"))

task = Task(description="Setup CI/CD")
flow.add_item(task, progression_ids="pending")

# Serialize
data = flow.to_dict(mode="json")
print(f"Serialized flow: {len(data['items']['items'])} items")

# Deserialize
restored = Flow.from_dict(data)

# Verify named access works
pending = restored.get_progression("pending")
print(f"Restored pending stage: {len(pending)} items")
print(f"Task preserved: {restored.items[task.id].description}")
```

---

## Summary

**Flow** provides a dual-pile workflow state machine:

**Key Features**:

- Dual-pile architecture (progressions + items)
- M:N relationships (items in multiple stages)
- Named progression access (O(1) lookup)
- Independent lifecycles (items persist independently)
- Thread-safe operations (RLock synchronization)
- Full serialization support (with name index rebuild)

**Common Patterns**:

- `flow.get_progression("stage")` - Named stage access
- `flow.add_item(item, progression_ids="stage")` - Add to pile and stage
- `stage.remove(id); next_stage.append(id)` - State transitions
- `for prog in flow.progressions` - Workflow introspection

**Performance**:

- O(1) named progression lookup
- O(1) item add/get
- O(n) item remove from pile progression
- O(p × n) full item removal (p progressions, n items each)

**Design Principles**:

- UUID references enable M:N without duplication
- Named stages map to business vocabulary
- Dual piles separate structure from data
- Thread-safe for concurrent workflows

See `src/lionherd_core/base/flow.py` for full implementation details.

---

## See Also

- [Element](element.md) - Base class with identity and serialization
- [Pile](pile.md) - Thread-safe typed collection
- [Progression](progression.md) - Ordered sequence of UUIDs
- [Node](node.md) - Content-bearing Element
- [Graph](graph.md) - Graph structure using Flow patterns

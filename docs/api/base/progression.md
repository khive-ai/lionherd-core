# Progression

> Ordered sequence of UUIDs with Element identity for workflow construction

---

> **⚠️ Migration Notice**: Breaking changes in v1.0.0-alpha5. See the [comprehensive migration guide](../../migration/v1.0.0-alpha5.md) for detailed upgrade instructions.

---

## Overview

`Progression` is an ordered sequence container that combines list-like operations with workflow-specific functionality. It extends Element to provide UUID ordering with identity, serialization, and metadata tracking.

**Key Capabilities:**

- **Element Inheritance**: Auto-generated UUID, timestamps, metadata (from Element base class)
- **List Operations**: `append`, `extend`, `insert`, `remove`, `pop`, `popleft`, `clear`
- **Workflow Operations**: `move`, `swap`, `reverse` for dynamic reordering
- **Idempotent Operations**: `include`, `exclude` for safe retry (operations that can be called multiple times with the same effect as calling once)
- **Query Operations**: `len`, `contains`, `getitem`, `setitem`, `index`, iteration
- **Serialization**: JSON roundtrip with UUID string conversion

## When to Use Progression

**Use Progression for:**

- Workflow state machines (pending/active/completed task progressions)
- Task execution order tracking
- Priority scheduling with dynamic reordering
- Event sourcing with ordered event sequences
- Idempotent task registration systems

**When NOT to Use Progression:**

- Unordered collections (use `set` or `Pile`)
- Content-bearing entities (use `Node`)
- Key-value mappings (use `dict`)
- Single-value state tracking (use simple attributes)

See [Element](element.md) for identity-based base class.

## Migration Guide (v1.0.0a4 → v1.0.0a5)

### 1. Field Validator - Stricter Validation

```python
# Before: Progression(order=[uuid1, "invalid", None])  # Silently dropped invalid
# After:  Progression(order=[uuid1, "invalid", None])  # ValidationError
# Use:    Progression(order=[uuid1, uuid2, element])   # Valid UUIDs/Elements only
```

### 2. Exception Changes - IndexError → NotFoundError

```python
# Before: except IndexError  # pop(), popleft(), __bool__
# After:  except NotFoundError
from lionherd_core.errors import NotFoundError
try:
    item = prog.pop()
except NotFoundError:  # Changed from IndexError
    pass

try:
    item = prog.popleft()
except NotFoundError:  # Changed from IndexError
    pass
```

### 3. **bool**() Protocol Added

```python
# Before: if len(prog) == 0
# After:  if not prog  # Pythonic empty check
```

## Class Signature

```python
from lionherd_core.base import Progression

class Progression(Element):
    """Ordered sequence of UUIDs with workflow operations.

    Combines list semantics with idempotent operations for state machines.
    """

    # Constructor signature
    def __init__(
        self,
        order: list[UUID] | list[Element] | None = None,
        name: str | None = None,
        # Inherited from Element (keyword-only):
        **data: Any,  # id, created_at, metadata
    ) -> None: ...
```

## Parameters

**order**: `list[UUID | Element] | None = None` - Ordered sequence (auto-converts Elements to UUIDs, duplicates allowed)

- Default: `[]` (empty progression)

**name** : str, optional

Descriptive name for the progression (e.g., "pending_tasks", "execution_order").

- Type: `str`
- Usage: Workflow state identification, logging, debugging
- Default: `None`

**id** : UUID or str, optional (inherited from Element)

Unique identifier. Auto-generated via `uuid4()` if not provided.

**created_at** : datetime or str or int or float, optional (inherited from Element)

Creation timestamp. Auto-generated if not provided.

**metadata** : dict, optional (inherited from Element)

Additional metadata dictionary.

## Attributes

| Attribute    | Type                | Mutable | Inherited | Description                        |
|--------------|---------------------|---------|-----------|-----------------------------------|
| `order`      | `list[UUID]`        | Yes     | No        | Ordered sequence of UUIDs         |
| `name`       | `str \| None`       | Yes     | No        | Descriptive name                  |
| `id`         | `UUID`              | No      | Yes       | Unique identifier (frozen)        |
| `created_at` | `datetime`          | No      | Yes       | Creation timestamp (frozen)       |
| `metadata`   | `dict[str, Any]`    | Yes     | Yes       | Additional metadata               |

## Methods

### List Operations

#### `append()`

Add item to end of progression.

**Signature:**

```python
def append(self, item: UUID | Element) -> None
```

**Parameters:**

- `item` (UUID | Element): Item to append (auto-converts Element to UUID)

**Returns:** None (modifies in place)

**Example:**

```python
from lionherd_core.base import Progression
from uuid import uuid4

prog = Progression(name="tasks")
prog.append(uuid4())
prog.append(uuid4())
print(len(prog))  # 2
```

**Time Complexity:** O(1) - amortized constant time

#### `extend()`

Add multiple items to end of progression.

**Signature:**

```python
def extend(self, items: list[UUID | Element]) -> None
```

**Parameters:**

- `items` (list[UUID | Element]): Items to append

**Returns:** None (modifies in place)

**Example:**

```python
tasks = [uuid4() for _ in range(3)]
prog.extend(tasks)
print(len(prog))  # 5 (2 + 3)
```

**Time Complexity:** O(k) where k = number of items to extend

#### `insert()`

Insert item at specific position.

**Signature:**

```python
def insert(self, index: int, item: UUID | Element) -> None
```

**Parameters:**

- `index` (int): Position to insert at (supports negative indices)
- `item` (UUID | Element): Item to insert

**Returns:** None (modifies in place)

**Example:**

```python
prog = Progression(order=[uuid4(), uuid4()])
prog.insert(1, uuid4())  # Insert in middle
print(len(prog))  # 3
```

**Time Complexity:** O(n) - linear time due to list shift

#### `remove()`

Remove first occurrence of item.

**Signature:**

```python
def remove(self, item: UUID | Element) -> None
```

**Parameters:**

- `item` (UUID | Element): Item to remove

**Raises:**

- `ValueError`: If item not in progression

**Returns:** None (modifies in place)

**Example:**

```python
uid = uuid4()
prog.append(uid)
prog.remove(uid)
# prog.remove(uid)  # ValueError: not in list
```

**Time Complexity:** O(n) - must search for item, then shift remaining elements

#### `pop()`

Remove and return item at index (default: last item). Optionally provide a default value for safe fallback.

**Signature:**

```python
def pop(self, index: int = -1, default: Any = ...) -> UUID | Any
```

**Parameters:**

- `index` (int, optional): Position to pop from. Default: `-1` (last item)
- `default` (Any, optional): Value to return if index not found. If not provided, raises `NotFoundError`

**Raises:**

- **`NotFoundError`**: If progression is empty or index out of range **and no default provided** (Changed from `IndexError` in PR #156)

**Returns:** UUID | Any - Removed item, or default value if index not found

**⚠️ BREAKING CHANGE (PR #156):** Exception changed from `IndexError` to `NotFoundError` for semantic consistency with Pile/Graph/Flow.

**Example:**

```python
from lionherd_core.errors import NotFoundError

prog = Progression(order=[uuid4(), uuid4()])
last = prog.pop()           # Remove last item
first = prog.pop(0)          # Remove first item

# Safe fallback with default
task = prog.pop(default=None)  # Returns None if empty, no exception

# Production pattern: safe queue processing
while (task := prog.pop(default=None)) is not None:
    process(task)

# Exception handling (updated)
try:
    item = prog.pop()
except NotFoundError:  # Before: IndexError
    handle_empty()
```

**Time Complexity:** O(1) for pop(), O(n) for pop(0) due to list shift

#### `popleft()`

Remove and return first item (queue behavior).

**Signature:**

```python
def popleft(self) -> UUID
```

**Raises:**

- **`NotFoundError`**: If progression is empty (Changed from `IndexError` in PR #156)

**Returns:** UUID - First item

**⚠️ BREAKING CHANGE (PR #156):** Exception changed from `IndexError` to `NotFoundError` for semantic consistency.

**Example:**

```python
from lionherd_core.errors import NotFoundError

# FIFO queue
pending = Progression(order=[uuid4(), uuid4()])
next_task = pending.popleft()

# Exception handling (updated)
try:
    task = pending.popleft()
except NotFoundError:  # Before: IndexError
    handle_empty()
```

**Time Complexity:** O(n) - removes from front, requires shifting all elements

**See Also:**

- `pop()`: Remove from arbitrary position
- Standard `collections.deque` for dedicated queue operations

#### `clear()`

Remove all items from progression.

**Signature:**

```python
def clear(self) -> None
```

**Returns:** None (modifies in place)

**Example:**

```python
prog.clear()
print(len(prog))  # 0
```

---

### Workflow Operations

#### `move()`

Move item from one position to another.

**Signature:**

```python
def move(self, from_index: int, to_index: int) -> None
```

**Parameters:**

- `from_index` (int): Source position (supports negative indices)
- `to_index` (int): Destination position (supports negative indices)

**Returns:** None (modifies in place)

**Example:**

```python
# Priority queue - move task to front
prog = Progression(order=[task1, task2, task3])
prog.move(2, 0)  # Move task3 to front
# Result: [task3, task1, task2]
```

**Use Cases:**

- Priority scheduling (move high-priority task to front)
- Task reordering in workflow editors
- Undo/redo stacks with position changes

**Time Complexity:** O(n) in worst case (moving between distant positions)

#### `swap()`

Swap positions of two items.

**Signature:**

```python
def swap(self, i: int, j: int) -> None
```

**Parameters:**

- `i` (int): First position (supports negative indices)
- `j` (int): Second position (supports negative indices)

**Returns:** None (modifies in place)

**Example:**

```python
prog = Progression(order=[task1, task2, task3])
prog.swap(0, -1)  # Swap first and last
# Result: [task3, task2, task1]
```

**Use Cases:**

- Manual task reordering
- Swapping execution order of dependent tasks
- UI drag-and-drop implementations

**Time Complexity:** O(1) - constant time swap

#### `reverse()`

Reverse order of all items.

**Signature:**

```python
def reverse(self) -> None
```

**Returns:** None (modifies in place)

**Example:**

```python
prog = Progression(order=[task1, task2, task3])
prog.reverse()
# Result: [task3, task2, task1]
```

**Use Cases:**

- Undo stacks (reverse chronological order)
- Workflow direction changes
- LIFO to FIFO conversion

**Time Complexity:** O(n) - reverses all n elements

---

### Idempotent Operations

#### `include()`

Add item if not present (idempotent, safe for retries).

**Signature:**

```python
def include(self, item: UUID | Element) -> bool
```

**Parameters:**

- `item` (UUID | Element): Item to add

**Returns:** bool - `True` if item was added, `False` if already present

**Example:**

```python
task_id = uuid4()
added1 = prog.include(task_id)  # True (added)
added2 = prog.include(task_id)  # False (already present)
print(len(prog))  # 1 (only one instance)
```

**Use Cases:**

- Retry-safe task registration
- Idempotent event processing
- Task deduplication in workflow systems

**Time Complexity:** O(n) - must check if item exists

**Note**: Not thread-safe. For concurrent access, use external synchronization.

**See Also:**

- `append()`: Non-idempotent (allows duplicates)
- `exclude()`: Idempotent removal

#### `exclude()`

Remove item if present (idempotent, safe for retries).

**Signature:**

```python
def exclude(self, item: UUID | Element) -> bool
```

**Parameters:**

- `item` (UUID | Element): Item to remove

**Returns:** bool - `True` if item was removed, `False` if not present

**Example:**

```python
task_id = uuid4()
prog.append(task_id)

removed1 = prog.exclude(task_id)  # True (removed)
removed2 = prog.exclude(task_id)  # False (not present)
print(len(prog))  # 0
```

**Use Cases:**

- Retry-safe task cancellation
- Cleanup operations that can be called multiple times
- Distributed systems with at-least-once delivery

**Time Complexity:** O(n) - must search for item, then shift if found

**See Also:**

- `remove()`: Non-idempotent (raises ValueError if not present)
- `include()`: Idempotent addition

---

### Query Operations

#### `__len__()`

Get number of items in progression.

**Signature:**

```python
def __len__(self) -> int
```

**Returns:** int - Number of items

**Example:**

```python
print(len(prog))  # 5
```

#### `__bool__()`

Check if progression is non-empty (supports `if prog:` idiom).

**Signature:**

```python
def __bool__(self) -> bool
```

**Returns:** bool - `False` if progression is empty, `True` otherwise

**Example:**

```python
prog = Progression()
if not prog:
    print("Progression is empty")

prog.append(uuid4())
if prog:
    print("Progression has items")
```

**Time Complexity:** O(1)

**Added in:** v1.0.0a5 (PR #156)

**Pattern:** Prefer `if prog:` over `if len(prog) > 0:` for empty checks

#### `__contains__()`

Check if item is in progression.

**Signature:**

```python
def __contains__(self, item: UUID | Element) -> bool
```

**Parameters:**

- `item` (UUID | Element): Item to check

**Returns:** bool - `True` if present

**Example:**

```python
task_id = uuid4()
prog.append(task_id)

if task_id in prog:
    print("Task found")
```

#### `__getitem__()`

Access item by index or slice.

**Signature:**

```python
def __getitem__(self, index: int | slice) -> UUID | list[UUID]
```

**Parameters:**

- `index` (int | slice): Index or slice object

**Raises:**

- `IndexError`: If index out of range (standard list behavior)

**Returns:** UUID (for int) or list[UUID] (for slice)

**Example:**

```python
first = prog[0]
last = prog[-1]
middle_three = prog[1:4]
```

**Time Complexity:** O(1) for index access, O(k) for slice where k = slice length

#### `__setitem__()`

Set item at index or slice.

**Signature:**

```python
def __setitem__(self, index: int | slice, value: UUID | Element | list) -> None
```

**Parameters:**

- `index` (int | slice): Index or slice object
- `value` (UUID | Element | list): Item(s) to set (list required for slice assignment)

**Raises:**

- `IndexError`: If index out of range (standard list behavior)
- `TypeError`: If assigning non-list to slice

**Returns:** None (modifies in place)

**Example:**

```python
prog = Progression(order=[uuid4(), uuid4(), uuid4()])

# Set single item
prog[0] = uuid4()

# Set slice
prog[1:3] = [uuid4(), uuid4()]
```

**Time Complexity:** O(1) for index assignment, O(k) for slice where k = slice length

#### `index()`

Find position of item.

**Signature:**

```python
def index(self, item: UUID | Element) -> int
```

**Parameters:**

- `item` (UUID | Element): Item to find

**Raises:**

- `ValueError`: If item not in progression

**Returns:** int - Zero-based index of item

**Example:**

```python
task_id = uuid4()
prog.append(task_id)
pos = prog.index(task_id)  # 0
```

#### `__iter__()`

Iterate over items.

**Signature:**

```python
def __iter__(self) -> Iterator[UUID]
```

**Returns:** Iterator[UUID]

**Example:**

```python
for task_id in prog:
    print(f"Task: {task_id}")
```

---

### Serialization

Progression inherits Element serialization with UUID string conversion for JSON compatibility.

#### `to_dict()`

Serialize to dictionary (inherited from Element).

**Signature:**

```python
def to_dict(
    self,
    *,
    mode: Literal["python", "json", "db"] = "python",
    **kwargs: Any,
) -> dict[str, Any]
```

**Parameters:**

- `mode` (str): Serialization mode
  - `"python"`: Python objects (datetime, UUID objects)
  - `"json"`: JSON-safe types (UUIDs as strings)
  - `"db"`: Database mode (uses `node_metadata` key)

**Returns:** dict[str, Any]

**Example:**

```python
prog = Progression(order=[uuid4(), uuid4()], name="tasks")

# JSON mode - UUIDs as strings
data = prog.to_dict(mode="json")
# {'id': '...', 'order': ['uuid-str1', 'uuid-str2'], 'name': 'tasks'}
```

#### `from_dict()`

Deserialize from dictionary (class method, inherited from Element).

**Signature:**

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Progression
```

**Parameters:**

- `data` (dict[str, Any]): Serialized progression

**Returns:** Progression

**Example:**

```python
data = prog.to_dict(mode="json")
restored = Progression.from_dict(data)

print(restored.order == prog.order)  # True
print(restored.id == prog.id)        # True
```

---

## Design Rationale

### Why Progression Exists

**Problem**: Workflow systems need ordered sequences with:

- Identity and metadata (Element base)
- List operations (append, remove, pop)
- Workflow operations (move, swap, reverse)
- Idempotent operations (include, exclude)
- Serialization for persistence

**Solution**: Progression combines these requirements in a single primitive.

### Design Decisions

#### 1. Duplicates Allowed (List Semantics)

**Decision**: Allow duplicate UUIDs in progression

**Rationale**:

- Matches list semantics users expect
- Enables retry tracking (same task appears multiple times)
- Event sourcing patterns (duplicate events with different timestamps)

**Alternative Rejected**: Set semantics (no duplicates)

- Would require separate tracking for retry counts
- Loses order information for duplicates
- Incompatible with event sourcing patterns

**Use `include()` for set-like behavior when needed**

#### 2. Idempotent Operations (include/exclude)

**Decision**: Provide `include()` and `exclude()` alongside `append()` and `remove()`

**Rationale**:

- Distributed systems need retry-safe operations
- At-least-once delivery requires idempotency
- Concurrent workflow agents benefit from safe operations

**Pattern**:

```python
# Non-idempotent (raises on retry)
prog.remove(task_id)  # ValueError if not present

# Idempotent (safe for retry)
prog.exclude(task_id)  # Returns False if not present, no error
```

#### 3. Workflow Operations (move/swap/reverse)

**Decision**: Provide dedicated reordering methods instead of expecting users to manipulate `.order` directly

**Rationale**:

- Common workflow patterns deserve first-class support
- Clear intent (move vs manual list manipulation)
- Future optimization potential (move tracking, undo support)

**Alternative Rejected**: Direct list manipulation

```python
# ❌ Unclear intent, error-prone
item = prog.order.pop(2)
prog.order.insert(0, item)

# ✅ Clear intent
prog.move(2, 0)
```

### Trade-offs

**Flexibility vs Constraints**:

- **Chosen**: List semantics (duplicates allowed, ordered)
- **Trade-off**: More memory than set, potential duplicate confusion
- **Mitigation**: Provide `include()` for set-like behavior

**Mutability**:

- **Chosen**: Mutable progression (in-place operations)
- **Trade-off**: Shared reference issues vs ergonomics
- **Mitigation**: Element identity (id/created_at frozen) provides stability

---

## Usage Patterns

### Basic Usage

```python
from lionherd_core.base import Progression
from uuid import uuid4

# Create empty progression
prog = Progression(name="task_queue")

# Add tasks
tasks = [uuid4() for _ in range(5)]
prog.extend(tasks)

# Query
print(f"Pending tasks: {len(prog)}")
print(f"Next task: {prog[0]}")
print(f"Task {tasks[2]} pending: {tasks[2] in prog}")
```

### State Machine Pattern

```python
# Workflow with three states
pending = Progression(name="pending")
active = Progression(name="active")
completed = Progression(name="completed")

# Initialize
tasks = [uuid4() for _ in range(10)]
pending.extend(tasks)

# Start task (pending → active)
task = pending.popleft()
active.append(task)

# Complete task (active → completed)
task = active.pop()
completed.append(task)

# Retry failed task (active → pending)
failed_task = active.pop()
pending.append(failed_task)
```

### Priority Scheduling

```python
# Task priority queue
high_priority = Progression(name="high")
normal_priority = Progression(name="normal")

# Promote task to high priority
task_id = normal_priority.pop(3)
high_priority.append(task_id)

# Process in priority order
while high_priority or normal_priority:
    task = high_priority.popleft() if high_priority else normal_priority.popleft()
    execute(task)
```

### Idempotent Registration

```python
# Safe for retries (idempotent operations)
registered_tasks = Progression(name="registry")

def register_task(task_id: UUID) -> bool:
    """Register task (idempotent - safe to call multiple times)."""
    return registered_tasks.include(task_id)

def deregister_task(task_id: UUID) -> bool:
    """Deregister task (idempotent - safe to call multiple times)."""
    return registered_tasks.exclude(task_id)

# Multiple calls safe
task = uuid4()
register_task(task)  # True (added)
register_task(task)  # False (already present)
```

### Serialization Roundtrip

```python
import json
from uuid import uuid4

# Persist progression state
prog = Progression(order=[uuid4(), uuid4()], name="workflow_steps")

# Serialize to JSON
data = prog.to_dict(mode="json")
json_str = json.dumps(data)

# Restore from JSON
restored = Progression.from_dict(json.loads(json_str))

print(restored.order == prog.order)  # True
print(restored.name == prog.name)    # True
print(restored.id == prog.id)        # True
```

---

## Common Pitfalls

### Pitfall 1: Expecting Set Semantics (No Duplicates)

**Issue**: Assuming Progression prevents duplicates like a set.

```python
prog = Progression()
task_id = uuid4()

prog.append(task_id)
prog.append(task_id)  # Duplicates allowed
print(len(prog))  # 2 (not 1!)
```

**Solution**: Use `include()` for set-like idempotent behavior.

```python
prog = Progression()
task_id = uuid4()

prog.include(task_id)  # Returns True (added)
prog.include(task_id)  # Returns False (already present)
print(len(prog))  # 1
```

### Pitfall 2: Modifying .order Directly

**Issue**: Bypassing Progression methods by modifying `.order` attribute directly.

```python
prog = Progression()
prog.order.append("not-a-uuid")  # ❌ No validation!
```

**Solution**: Use Progression methods (append, extend, etc.) for validation.

```python
prog.append(uuid4())  # ✓ Validated
```

### Pitfall 3: Forgetting Idempotent Operations Exist

**Issue**: Using `remove()` in retry logic, causing errors on second call.

```python
# ❌ Fails on retry
def cleanup_task(task_id: UUID):
    prog.remove(task_id)  # ValueError if already removed
```

**Solution**: Use `exclude()` for retry-safe removal.

```python
# ✓ Safe for retries
def cleanup_task(task_id: UUID):
    return prog.exclude(task_id)  # Returns False if not present
```

---

## See Also

- [Element](element.md) - Base class with identity and serialization
- [Pile](pile.md) - Unordered collection of Elements
- [Event](event.md) - State machine events with progressions
- [Graph](graph.md) - Graph structure with node/edge progressions

---

## Examples

### Example 1: Task Queue (FIFO)

```python
from lionherd_core.base import Progression
from uuid import uuid4

# Create task queue
queue = Progression(name="pending_tasks")

# Add tasks
for _ in range(5):
    queue.append(uuid4())

# Process tasks in order
while queue:
    task_id = queue.popleft()
    print(f"Processing: {task_id}")
    # execute_task(task_id)
```

### Example 2: Workflow State Machine

```python
# Three-state workflow
pending = Progression(name="pending")
in_progress = Progression(name="in_progress")
done = Progression(name="done")

# Create tasks
tasks = [uuid4() for _ in range(10)]
pending.extend(tasks)

# Start task
task = pending.popleft()
in_progress.append(task)
print(f"Started task: {task}")

# Complete task
completed_task = in_progress.pop()
done.append(completed_task)
print(f"Completed task: {completed_task}")

# Check state
print(f"Pending: {len(pending)}, In Progress: {len(in_progress)}, Done: {len(done)}")
# Output: Pending: 9, In Progress: 0, Done: 1
```

### Example 3: Priority Scheduling with move()

```python
# Priority task queue
tasks = Progression(order=[task1, task2, task3, task4], name="priority_queue")

# Urgent task arrives - move to front
urgent_index = tasks.index(task3)
tasks.move(urgent_index, 0)

# Process in priority order
next_task = tasks.popleft()  # Gets task3 (urgent)
```

### Example 4: Idempotent Task Registration

```python
# Task registry with retry-safe operations (idempotent)
registry = Progression(name="task_registry")

def register(task_id: UUID) -> str:
    if registry.include(task_id):
        return f"Registered {task_id}"
    else:
        return f"{task_id} already registered"

# Safe to call multiple times
task = uuid4()
print(register(task))  # "Registered ..."
print(register(task))  # "... already registered"
```

### Example 5: Serialization for Persistence

```python
import json
from lionherd_core.base import Progression
from uuid import uuid4

# Create progression with state
workflow = Progression(
    order=[uuid4() for _ in range(3)],
    name="deployment_steps",
    metadata={"env": "production"}
)

# Serialize to JSON
data = workflow.to_dict(mode="json")
json_str = json.dumps(data, indent=2)

# Persist to file/database
with open("workflow_state.json", "w") as f:
    f.write(json_str)

# Restore from persistence
with open("workflow_state.json") as f:
    restored_data = json.load(f)

restored = Progression.from_dict(restored_data)
print(f"Restored {restored.name} with {len(restored)} steps")
print(f"Environment: {restored.metadata['env']}")
```

### Example 6: Undo Stack with reverse()

```python
# Action history for undo
action_history = Progression(name="undo_stack")

# Record actions
actions = [uuid4() for _ in range(5)]
action_history.extend(actions)

# Undo last action
last_action = action_history.pop()
undo(last_action)

# Reverse to get chronological order
action_history.reverse()
for action in action_history:
    print(f"Chronological: {action}")
```

# Pile

> Thread-safe typed collection with rich query interface

## Overview

`Pile` is a thread-safe collection for managing Element instances with type validation and rich querying capabilities. It combines dict-like keyed access with list-like insertion order preservation and provides a powerful type-dispatched `__getitem__` interface.

**Key Capabilities:**

- **Element Inheritance**: Auto-generated UUID, timestamps, metadata (from Element base class)
- **Thread Safety**: RLock-based synchronization for concurrent access
- **Type Validation**: Flexible constraints with Union support (single type, multiple types, or any Element)
- **Rich Queries**: Type-dispatched `__getitem__` (UUID, index, slice, callable, Progression)
- **Idempotent Operations**: `include`, `exclude` for safe retry (operations that can be called multiple times with the same effect as calling once)
- **Async Support**: Separate async lock for concurrent async operations
- **Serialization**: JSON roundtrip with type preservation

## When to Use Pile

**Use Pile for:**

- Thread-safe collections requiring concurrent access
- Type-validated heterogeneous collections (Union types)
- Collections with rich query requirements (filter, index, progression order)
- Workflow state management with type safety
- Collections requiring both keyed access (by UUID) and ordered iteration

**When NOT to Use Pile:**

- Simple ordered sequences (use `Progression`)
- Content-bearing single entities (use `Node`)
- Untyped collections (use standard `dict` or `list`)
- Single-threaded scenarios where thread safety overhead is unnecessary

See [Element](element.md) for identity-based base class.

## Class Signature

```python
from lionherd_core.base import Pile
from typing import Union

class Pile(Element, Generic[T]):
    """Thread-safe typed collection with rich query interface.

    Type-dispatched __getitem__: pile[uuid], pile[int/slice],
    pile[progression], pile[callable].
    """

    # Constructor signature
    def __init__(
        self,
        items: list[T] | None = None,
        item_type: type[T] | set[type] | list[type] | None = None,
        order: list[UUID] | Progression | None = None,
        strict_type: bool = False,
        # Inherited from Element (keyword-only):
        **kwargs: Any,  # id, created_at, metadata
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**items** : list of Elements, optional

Initial items to add to the pile.

- Type: `list[T]`
- Auto-validation: Items validated against `item_type` if specified
- Default: `None` (empty pile)

**item_type** : type or set of types, optional

Type constraint(s) for validation. Accepts single type, set, list, or Union.

- Type: `type[T] | set[type] | list[type] | None`
- Single type: `Pile(item_type=Task)` - only Task instances
- Union types: `Pile(item_type=Union[Task, Event])` - Task or Event
- Set/list: `Pile(item_type={Task, Event})` - equivalent to Union
- None (default): Any Element subclass allowed
- **Time Complexity**: O(1) type check on each add/update

**order** : list of UUID or Progression, optional

Custom insertion order. If provided, overrides natural insertion order.

- Type: `list[UUID] | Progression | None`
- Validates all UUIDs are present in items
- Default: `None` (insertion order)

**strict_type** : bool, optional

Enforce exact type match (no subclasses).

- Type: `bool`
- `False` (default): Allow subclasses (permissive)
- `True`: Exact type only (strict validation)
- Default: `False`

**id**, **created_at**, **metadata** (inherited from Element)

See [Element](element.md) documentation.

## Attributes

| Attribute     | Type              | Mutable | Inherited | Description                          |
|---------------|-------------------|---------|-----------|--------------------------------------|
| `item_type`   | `set[type] \| None` | Yes     | No        | Type constraints (None = any Element) |
| `strict_type` | `bool`            | Yes     | No        | Exact type enforcement               |
| `id`          | `UUID`            | No      | Yes       | Unique identifier (frozen)           |
| `created_at`  | `datetime`        | No      | Yes       | Creation timestamp (frozen)          |
| `metadata`    | `dict[str, Any]`  | Yes     | Yes       | Additional metadata                  |

**Read-only Properties:**

| Property      | Type                  | Description                              |
|---------------|-----------------------|------------------------------------------|
| `items`       | `MappingProxyType[UUID, T]` | Read-only view of internal dict    |
| `progression` | `Progression`         | Copy of insertion order (prevents mutation) |

## Methods

### Core Operations

#### `add()`

Add item to pile (thread-safe).

**Signature:**

```python
def add(self, item: T) -> None
```

**Parameters:**

- `item` (Element): Item to add

**Raises:**

- `ValueError`: If item already exists (duplicate ID)
- `TypeError`: If item type validation fails

**Returns:** None (modifies in place)

**Example:**

```python
from lionherd_core.base import Pile, Element

pile = Pile()
item = Element()
pile.add(item)
print(len(pile))  # 1
```

**Time Complexity:** O(1) for add operation

**Thread Safety:** Yes - uses RLock synchronization

#### `remove()`

Remove item from pile (thread-safe).

**Signature:**

```python
def remove(self, item_id: UUID | str | Element) -> T
```

**Parameters:**

- `item_id` (UUID | str | Element): Item to remove

**Raises:**

- `ValueError`: If item not found

**Returns:** T - Removed item

**Example:**

```python
item = Element()
pile.add(item)
removed = pile.remove(item.id)
print(removed.id == item.id)  # True
```

**Time Complexity:** O(n) - must update progression order

**Thread Safety:** Yes - uses RLock synchronization

#### `pop()`

Alias for `remove()` - remove and return item.

**Signature:**

```python
def pop(self, item_id: UUID | str | Element) -> T
```

**Time Complexity:** O(n) - same as remove()

#### `get()`

Get item by ID with optional default (thread-safe).

**Signature:**

```python
def get(self, item_id: UUID | str | Element, default: Any = ...) -> T | None
```

**Parameters:**

- `item_id` (UUID | str | Element): Item to retrieve
- `default` (Any, optional): Return value if not found

**Raises:**

- `ValueError`: If item not found and no default provided

**Returns:** T | None - Item or default

**Example:**

```python
item = pile.get(uuid, default=None)
if item is None:
    print("Not found")
```

**Time Complexity:** O(1) - dict lookup

**Thread Safety:** Yes - uses RLock synchronization

#### `update()`

Update existing item (thread-safe).

**Signature:**

```python
def update(self, item: T) -> None
```

**Parameters:**

- `item` (Element): Updated item (must have same ID as existing)

**Raises:**

- `ValueError`: If item not found
- `TypeError`: If type validation fails

**Returns:** None (modifies in place)

**Example:**

```python
item = pile.get(uuid)
item.metadata["updated"] = True
pile.update(item)
```

**Time Complexity:** O(1) - dict update

**Thread Safety:** Yes - uses RLock synchronization

#### `clear()`

Remove all items (thread-safe).

**Signature:**

```python
def clear(self) -> None
```

**Returns:** None (modifies in place)

**Example:**

```python
pile.clear()
print(len(pile))  # 0
```

**Time Complexity:** O(1) - clears dict and progression

**Thread Safety:** Yes - uses RLock synchronization

---

### Idempotent Operations

#### `include()`

Add item if not present (idempotent, safe for retries).

**Signature:**

```python
def include(self, item: T) -> bool
```

**Parameters:**

- `item` (Element): Item to add

**Returns:** bool - `True` if added, `False` if already present

**Example:**

```python
item = Element()
added1 = pile.include(item)  # True (added)
added2 = pile.include(item)  # False (already present)
```

**Use Cases:**

- Retry-safe item registration
- Idempotent event processing
- Task deduplication in workflow systems

**Time Complexity:** O(1) for membership check, O(1) for add if needed

**Note**: Not thread-safe for concurrent calls with same item. Use external synchronization.

#### `exclude()`

Remove item if present (idempotent, safe for retries).

**Signature:**

```python
def exclude(self, item: UUID | str | Element) -> bool
```

**Parameters:**

- `item` (UUID | str | Element): Item to remove

**Returns:** bool - `True` if removed, `False` if not present

**Example:**

```python
removed1 = pile.exclude(item)  # True (removed)
removed2 = pile.exclude(item)  # False (not present)
```

**Time Complexity:** O(n) - must update progression if found

---

### Rich Query Interface

#### `__getitem__()` - Type-Dispatched Queries

Get items by UUID, index, slice, callable, or Progression.

**Signature (overloaded):**

```python
# Get by UUID or string ID
def __getitem__(self, key: UUID | str) -> T: ...

# Get by index (progression order)
def __getitem__(self, key: int) -> T: ...

# Get by slice (progression order)
def __getitem__(self, key: slice) -> list[T]: ...

# Filter by callable predicate
def __getitem__(self, key: Callable[[T], bool]) -> Pile[T]: ...

# Filter by progression
def __getitem__(self, key: Progression) -> Pile[T]: ...
```

**Examples:**

```python
# Query by UUID
item = pile[uuid_obj]

# Query by index (progression order)
first = pile[0]
last = pile[-1]

# Query by slice
middle = pile[1:3]  # Returns list[T]

# Filter by callable (returns new Pile)
high_priority = pile[lambda item: item.priority > 5]

# Filter by progression (custom order)
prog = Progression(order=[uuid1, uuid2])
filtered = pile[prog]  # Returns new Pile with only those items
```

**Time Complexity:**

- UUID/str: O(1) - dict lookup
- int: O(1) - progression index access
- slice: O(k) where k = slice length
- callable: O(n) - must check all items
- Progression: O(m) where m = progression length

**Thread Safety:** Yes for reads - uses RLock synchronization

---

### Collection Methods

#### `__len__()`

Get number of items.

**Returns:** int - Number of items

**Time Complexity:** O(1)

#### `__contains__()`

Check if item exists.

**Signature:**

```python
def __contains__(self, item: UUID | str | Element) -> bool
```

**Returns:** bool - True if present

**Time Complexity:** O(1) - dict membership test

#### `__iter__()`

Iterate in progression order.

**Returns:** Iterator[T]

**Example:**

```python
for item in pile:
    print(item.id)
```

**Time Complexity:** O(1) to start, O(n) to iterate all

#### `keys()`

Get all UUIDs.

**Returns:** Iterator[UUID]

**Time Complexity:** O(1) to start, O(n) for all keys

#### `values()`

Get all items in progression order.

**Returns:** Iterator[T]

**Time Complexity:** O(1) to start, O(n) for all values

#### `to_list()`

Convert to list in progression order.

**Returns:** list[T]

**Time Complexity:** O(n)

#### `size()`

Get number of items (alias for `len()`).

**Returns:** int

**Time Complexity:** O(1)

#### `is_empty()`

Check if pile is empty.

**Returns:** bool

**Time Complexity:** O(1)

---

### Type Operations

#### `filter_by_type()`

Filter items by type (returns new Pile).

**Signature:**

```python
def filter_by_type(self, item_type: type[T]) -> Pile[T]
```

**Parameters:**

- `item_type` (type): Type to filter by

**Returns:** Pile[T] - New pile with only items of specified type

**Example:**

```python
class Task(Element):
    title: str = ""

class Event(Element):
    event_type: str = ""

mixed = Pile()
mixed.add(Task(title="Task 1"))
mixed.add(Event(event_type="alert"))

tasks_only = mixed.filter_by_type(Task)
print(len(tasks_only))  # 1
```

**Time Complexity:** O(n) - must check all items

---

### Async Operations

#### `add_async()`

Add item asynchronously (thread-safe with async lock).

**Signature:**

```python
async def add_async(self, item: T) -> None
```

**Time Complexity:** O(1)

#### `get_async()`

Get item asynchronously.

**Signature:**

```python
async def get_async(self, item_id: UUID | str | Element, default: Any = ...) -> T | None
```

**Time Complexity:** O(1)

#### Async Context Manager

Use pile as async context manager for manual lock control.

**Example:**

```python
async with pile as p:
    # Lock held during context
    items = list(p)
```

---

### Serialization

#### `to_dict()`

Serialize pile to dictionary (inherited from Element).

**Signature:**

```python
def to_dict(
    self,
    mode: Literal["python", "json", "db"] = "python",
    created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
    meta_key: str | None = None,
    item_meta_key: str | None = None,
    item_created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
    **kwargs: Any,
) -> dict[str, Any]
```

**Parameters:**

- `mode`: Serialization mode (python/json/db)
- `created_at_format`: Timestamp format for Pile
- `meta_key`: Rename Pile metadata field
- `item_meta_key`: Pass to each item's to_dict for metadata renaming
- `item_created_at_format`: Pass to each item's to_dict for timestamp format

**Returns:** dict[str, Any]

**Example:**

```python
pile = Pile(items=[Element(), Element()], item_type=Element)
data = pile.to_dict(mode="json")
# Preserves: items (in progression order), item_type, strict_type
```

#### `from_dict()`

Deserialize from dictionary (class method).

**Signature:**

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> Pile
```

**Returns:** Pile - Reconstructed pile

---

## Design Rationale

### Why Pile Exists

**Problem**: Workflow systems need collections that:

- Support concurrent access (thread safety)
- Enforce type constraints (validation)
- Provide rich queries (UUID, index, filter)
- Preserve insertion order
- Enable async operations

**Solution**: Pile combines these requirements in a single primitive.

### Design Decisions

#### 1. Thread Safety with RLock

**Decision**: Use `threading.RLock` for synchronization

**Rationale**:

- Enables reentrant locking (method can call itself)
- Prevents deadlocks in nested operations
- Acceptable overhead for workflow systems

**Alternative Rejected**: Lock-free data structures

- Too complex for workflow use cases
- RLock overhead negligible for typical workloads

#### 2. Type-Dispatched `__getitem__`

**Decision**: Support multiple query modes via type dispatch

**Rationale**:

- Single intuitive interface: `pile[key]`
- Type hints guide correct usage
- Reduces API surface area

**Pattern**:

```python
# Single interface, multiple behaviors
pile[uuid]           # Get by ID
pile[0]              # Get by index
pile[1:3]            # Get by slice
pile[lambda x: ...]  # Filter by predicate
pile[progression]    # Filter by custom order
```

#### 3. Separate Async Lock

**Decision**: Maintain separate `AsyncLock` for async operations

**Rationale**:

- Prevents blocking async event loop with sync lock
- Enables concurrent async operations
- Independent sync and async workflows

**Trade-off**: More complex implementation, but better async performance

### Trade-offs

**Thread Safety vs Performance**:

- **Chosen**: Thread-safe with RLock overhead
- **Trade-off**: ~5-10% slower than unsafe operations
- **Mitigation**: Most workflow systems need thread safety anyway

**Type Validation Overhead**:

- **Chosen**: Validate on add/update
- **Trade-off**: O(1) type check per operation
- **Mitigation**: Negligible for typical Element subclasses

---

## Usage Patterns

### Basic Usage

```python
from lionherd_core.base import Pile, Element

# Create pile
pile = Pile()

# Add items
items = [Element() for _ in range(5)]
for item in items:
    pile.add(item)

# Query
print(f"Total items: {len(pile)}")
print(f"First item: {pile[0]}")
print(f"Item by UUID: {pile[items[2].id]}")
```

### Type-Constrained Collections

```python
class Task(Element):
    title: str = ""
    priority: int = 0

# Single type constraint
tasks = Pile(item_type=Task)
tasks.add(Task(title="Review PR", priority=1))
# tasks.add(Element())  # TypeError: wrong type

# Union types
class Event(Element):
    event_type: str = ""

mixed = Pile(item_type=Union[Task, Event])
mixed.add(Task(title="Deploy"))
mixed.add(Event(event_type="alert"))
```

### Rich Queries

```python
# Filter by callable
high_priority = pile[lambda t: t.priority > 5]

# Custom ordering with Progression
prog = Progression(order=[uuid1, uuid3, uuid2])
ordered = pile[prog]

# Slice operations
first_three = pile[0:3]
last_two = pile[-2:]
```

### Thread-Safe Concurrent Access

```python
import threading

pile = Pile()

def worker(item):
    pile.add(item)

threads = [threading.Thread(target=worker, args=(Element(),)) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Total items: {len(pile)}")  # 10 (thread-safe)
```

### Async Operations

```python
import asyncio
from lionherd_core.libs.concurrency import gather

async def async_workflow():
    pile = Pile()

    # Concurrent async add
    items = [Element() for _ in range(10)]
    await gather(*[pile.add_async(item) for item in items])

    # Concurrent async get
    results = await gather(*[pile.get_async(item.id) for item in items[:5]])

    return pile

pile = await async_workflow()
```

---

## Common Pitfalls

### Pitfall 1: Mutating Read-Only Properties

**Issue**: Attempting to modify `items` or `progression` properties directly.

```python
pile = Pile()
pile.items[uuid] = item  # ❌ TypeError: MappingProxyType is read-only
pile.progression.append(uuid)  # ❌ Modifies copy, not original
```

**Solution**: Use Pile methods for modifications.

```python
pile.add(item)  # ✓ Correct
pile.remove(uuid)  # ✓ Correct
```

### Pitfall 2: Type Validation with Subclasses

**Issue**: Forgetting `strict_type` mode allows subclasses.

```python
class HighPriorityTask(Task):
    urgent: bool = True

tasks = Pile(item_type=Task, strict_type=False)  # Default
tasks.add(HighPriorityTask())  # ✓ Allowed (subclass)
```

**Solution**: Use `strict_type=True` for exact type matching.

```python
strict_tasks = Pile(item_type=Task, strict_type=True)
# strict_tasks.add(HighPriorityTask())  # ❌ TypeError
```

### Pitfall 3: Concurrent `include()` Not Atomic

**Issue**: Two threads calling `include(item)` simultaneously may both add.

```python
# Thread 1 and Thread 2 both call:
pile.include(item)  # Race condition - may add twice
```

**Solution**: Use external lock for concurrent include/exclude.

```python
lock = threading.Lock()
with lock:
    pile.include(item)  # ✓ Atomic
```

---

## See Also

- [Element](element.md) - Base class with identity and serialization
- [Progression](progression.md) - Ordered sequence of UUIDs
- [Node](node.md) - Content-bearing Element with Pile integration
- [Graph](graph.md) - Graph structure using Piles for nodes and edges

---

## Examples

### Example 1: Type-Validated Task Collection

```python
from lionherd_core.base import Pile, Element
from typing import Union

class Task(Element):
    title: str = ""
    priority: int = 0

# Create type-constrained pile
tasks = Pile(item_type=Task)

# Add tasks
tasks.add(Task(title="Review PR #42", priority=2))
tasks.add(Task(title="Fix bug #123", priority=3))

# Filter by priority
high_priority = tasks[lambda t: t.priority >= 3]
print(f"High priority tasks: {len(high_priority)}")
```

### Example 2: Heterogeneous Collection with Union Types

```python
class Task(Element):
    title: str = ""

class Event(Element):
    event_type: str = ""

# Union type pile
mixed = Pile(item_type=Union[Task, Event])
mixed.add(Task(title="Deploy"))
mixed.add(Event(event_type="alert"))
mixed.add(Task(title="Rollback"))

# Filter by type
tasks_only = mixed.filter_by_type(Task)
events_only = mixed.filter_by_type(Event)

print(f"Tasks: {len(tasks_only)}, Events: {len(events_only)}")
```

### Example 3: Thread-Safe Concurrent Workflow

```python
import threading
from lionherd_core.base import Pile, Element

pile = Pile()
results = []

def worker(worker_id):
    for i in range(10):
        item = Element(metadata={"worker": worker_id, "index": i})
        pile.add(item)
        results.append(f"Worker {worker_id} added item {i}")

# Spawn 5 worker threads
threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Total items (should be 50): {len(pile)}")
print(f"All items have unique IDs: {len(pile) == 50}")
```

### Example 4: Async Concurrent Operations

```python
import asyncio
from lionherd_core.base import Pile, Element
from lionherd_core.libs.concurrency import gather

async def async_example():
    pile = Pile()

    # Create items
    items = [Element(metadata={"index": i}) for i in range(20)]

    # Concurrent async add
    await gather(*[pile.add_async(item) for item in items])
    print(f"Added {len(pile)} items concurrently")

    # Concurrent async get
    uuids = [item.id for item in items[:10]]
    results = await gather(*[pile.get_async(uid) for uid in uuids])
    print(f"Retrieved {len(results)} items")

    return pile

pile = await async_example()
```

### Example 5: Serialization with Type Preservation

```python
import json
from lionherd_core.base import Pile, Element

class Task(Element):
    title: str = ""
    priority: int = 0

# Create typed pile
original = Pile(
    items=[
        Task(title="Task A", priority=1),
        Task(title="Task B", priority=2)
    ],
    item_type=Task,
    strict_type=False
)

# Serialize to JSON
data = original.to_dict(mode="json")
json_str = json.dumps(data, indent=2)

# Deserialize
restored = Pile.from_dict(json.loads(json_str))

print(f"Type constraint preserved: {restored.item_type}")
print(f"Strict mode preserved: {restored.strict_type}")
print(f"Items preserved: {len(restored)} items")
print(f"Order preserved: {[t.title for t in restored]}")
```

### Example 6: Custom Ordering with Progression

```python
from lionherd_core.base import Pile, Progression, Element

# Create pile
pile = Pile([Element(metadata={"name": f"Item {i}"}) for i in range(5)])

# Get UUIDs in original order
original_order = list(pile.keys())

# Create custom order (reverse)
custom_order = Progression(order=list(reversed(original_order)))

# Filter by custom order (returns new Pile)
reversed_pile = pile[custom_order]

print("Original order:")
for item in pile:
    print(f"  {item.metadata['name']}")

print("\nReversed order:")
for item in reversed_pile:
    print(f"  {item.metadata['name']}")
```

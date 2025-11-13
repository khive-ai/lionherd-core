# PriorityQueue

> Async priority queue with heapq-based ordering and anyio.Condition synchronization

## Overview

`PriorityQueue` is an async-first priority queue implementation combining Python's `heapq` module with anyio's `Condition` primitive for safe concurrent access. Unlike `asyncio.PriorityQueue`, the nowait methods (`put_nowait`, `get_nowait`) are **async** to maintain proper locking semantics.

**Key Capabilities:**

- **Priority Ordering**: Items retrieved in priority order (lowest value first, heapq min-heap)
- **Async Blocking**: `put()` blocks when full, `get()` blocks when empty
- **Async Nowait**: `put_nowait()` and `get_nowait()` are async (unlike asyncio) for consistent locking
- **Size Limits**: Optional `maxsize` for bounded queues (0 = unlimited)
- **Thread-Safe**: Condition-based locking ensures safe concurrent access
- **Race-Aware Inspection**: `qsize()`, `empty()`, `full()` are unlocked and racy (monitoring only)

**When to Use PriorityQueue:**

- Task scheduling where **priority matters** (critical tasks before low-priority)
- Event processing with **importance-based ordering**
- Rate limiting with **priority lanes** (VIP requests first)
- Work queues where **ordering by value** is required
- Producer-consumer patterns needing **bounded capacity** with backpressure

**When NOT to Use PriorityQueue:**

- **FIFO ordering sufficient**: Use `anyio.Queue` or `asyncio.Queue` (simpler, faster)
- **LIFO needed**: Use stack-based collections
- **No blocking required**: Use `list` with `heapq` directly
- **Thread-based concurrency**: Use `queue.PriorityQueue` from stdlib

**API Compatibility Note:**

This implementation differs from `asyncio.PriorityQueue`:

- `put_nowait()` and `get_nowait()` are **async** (require `await`)
- Provides consistent locking semantics across all methods

## Class Signature

```python
from lionherd_core.libs.concurrency import PriorityQueue

class PriorityQueue(Generic[T]):
    """Async priority queue (heapq + anyio.Condition)."""

    # Constructor signature
    def __init__(self, maxsize: int = 0) -> None: ...
```

## Parameters

### Constructor Parameters

**maxsize** : int, default 0

Maximum queue size. Controls backpressure behavior when queue fills up.

- **0**: Unlimited queue size (no blocking on put)
- **> 0**: Bounded queue, `put()` blocks when full, `put_nowait()` raises `QueueFull`
- Validation: Must be non-negative integer
- Default: `0` (unlimited)

**Priority Item Format:**

Items should be tuples with priority as first element for proper ordering:

```python
# Lower values = higher priority (min-heap)
(priority, data)
(1, "critical_task")   # Higher priority
(5, "normal_task")     # Lower priority
(10, "background")     # Lowest priority
```

For equal priorities, items are ordered by remaining tuple elements (FIFO within priority level).

## Attributes

| Attribute    | Type            | Description                                              |
| ------------ | --------------- | -------------------------------------------------------- |
| `maxsize`    | `int`           | Maximum queue size (0 = unlimited)                       |
| `_queue`     | `list[T]`       | Internal heapq storage (private, do not access directly) |
| `_condition` | `Condition`     | Anyio condition for async locking (private)              |

**Note**: Only `maxsize` is public. Direct access to `_queue` or `_condition` breaks encapsulation and thread-safety guarantees.

## Methods

### Core Operations

#### `put()`

Put item into queue, blocking if full.

**Signature:**

```python
async def put(self, item: T) -> None: ...
```

**Parameters:**

- `item` (T): Item to insert (tuple with priority as first element recommended)

**Returns:**

- None

**Behavior:**

- If queue has space: Inserts immediately and notifies waiting consumers
- If queue is full (maxsize > 0 and len >= maxsize): Blocks until space available
- Uses `heapq.heappush()` to maintain min-heap ordering
- Thread-safe via condition lock

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import PriorityQueue
>>> queue = PriorityQueue[tuple[int, str]](maxsize=2)

# Put items (blocks if full)
>>> await queue.put((1, "high_priority"))
>>> await queue.put((5, "low_priority"))

# Third put blocks until consumer calls get()
>>> await queue.put((3, "medium"))  # Blocks until space available
```

**See Also:**

- `put_nowait()`: Non-blocking variant that raises exception when full

**Notes:**

Always use priority tuples `(priority, data)` for consistent ordering. Lower priority values are retrieved first.

#### `put_nowait()`

Put item without blocking (async, unlike asyncio).

**Signature:**

```python
async def put_nowait(self, item: T) -> None: ...
```

**Parameters:**

- `item` (T): Item to insert (tuple with priority as first element recommended)

**Returns:**

- None

**Raises:**

- `QueueFull`: If queue is at maxsize capacity

**Behavior:**

- If queue has space: Inserts immediately
- If queue is full: Raises `QueueFull` immediately (no blocking)
- Still **async** (requires `await`) for proper locking
- Notifies waiting consumers after insertion

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import PriorityQueue, QueueFull
>>> queue = PriorityQueue[tuple[int, str]](maxsize=2)

# Fill queue
>>> await queue.put_nowait((1, "task1"))
>>> await queue.put_nowait((2, "task2"))

# Third put raises exception
>>> await queue.put_nowait((3, "task3"))
QueueFull: Queue is full
```

**See Also:**

- `put()`: Blocking variant that waits for space

**Notes:**

Unlike `asyncio.PriorityQueue.put_nowait()` (synchronous), this method is **async** to ensure atomic lock acquisition and prevent race conditions.

#### `get()`

Get highest priority item, blocking if empty.

**Signature:**

```python
async def get(self) -> T: ...
```

**Returns:**

- T: Highest priority item (lowest value first, min-heap ordering)

**Behavior:**

- If queue has items: Returns and removes highest priority item immediately
- If queue is empty: Blocks until item available
- Uses `heapq.heappop()` to extract min value
- Notifies waiting producers after removal (frees space)
- Thread-safe via condition lock

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import PriorityQueue
>>> queue = PriorityQueue[tuple[int, str]]()

# Add items out of order
>>> await queue.put((5, "low"))
>>> await queue.put((1, "high"))
>>> await queue.put((3, "medium"))

# Get returns in priority order (lowest first)
>>> await queue.get()
(1, 'high')
>>> await queue.get()
(3, 'medium')
>>> await queue.get()
(5, 'low')

# Next get blocks until put() adds item
>>> await queue.get()  # Blocks until item available
```

**See Also:**

- `get_nowait()`: Non-blocking variant that raises exception when empty

**Notes:**

Priority ordering is **stable within same priority** (FIFO for equal priorities) due to tuple comparison.

#### `get_nowait()`

Get item without blocking (async, unlike asyncio).

**Signature:**

```python
async def get_nowait(self) -> T: ...
```

**Returns:**

- T: Highest priority item

**Raises:**

- `QueueEmpty`: If queue is empty

**Behavior:**

- If queue has items: Returns highest priority item immediately
- If queue is empty: Raises `QueueEmpty` immediately (no blocking)
- Still **async** (requires `await`) for proper locking
- Notifies waiting producers after removal

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import PriorityQueue, QueueEmpty
>>> queue = PriorityQueue[tuple[int, str]]()

# Get from empty queue raises exception
>>> await queue.get_nowait()
QueueEmpty: Queue is empty

# Add and retrieve
>>> await queue.put((1, "task"))
>>> await queue.get_nowait()
(1, 'task')
```

**See Also:**

- `get()`: Blocking variant that waits for items

**Notes:**

Unlike `asyncio.PriorityQueue.get_nowait()` (synchronous), this method is **async** to ensure atomic lock acquisition and prevent race conditions.

### Inspection Methods

#### `qsize()`

Approximate queue size (unlocked, racy).

**Signature:**

```python
def qsize(self) -> int: ...
```

**Returns:**

- int: Number of items in queue

**Thread-Safety:**

- **NOT thread-safe**: Reads `_queue` without lock
- **Racy**: Value may be stale immediately after return
- **Use case**: Monitoring, debugging, approximate metrics only

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import PriorityQueue
>>> queue = PriorityQueue[tuple[int, str]]()

>>> await queue.put((1, "task1"))
>>> await queue.put((2, "task2"))
>>> queue.qsize()
2

# Value can become stale
>>> size = queue.qsize()  # Returns 2
>>> await queue.get()     # Another task removes item
>>> print(size)           # Still 2 (stale)
```

**See Also:**

- `empty()`: Check if queue is empty (also racy)
- `full()`: Check if queue is full (also racy)

**Notes:**

Do **NOT** use for control flow decisions (e.g., `if qsize() > 0: get_nowait()`). Race conditions can cause `QueueEmpty` exceptions. Use blocking `get()` or handle exceptions instead.

#### `empty()`

Check if queue is empty (unlocked, racy).

**Signature:**

```python
def empty(self) -> bool: ...
```

**Returns:**

- bool: True if queue is empty

**Thread-Safety:**

- **NOT thread-safe**: Reads `_queue` without lock
- **Racy**: Value may be stale immediately after return
- **Use case**: Monitoring, debugging, approximate metrics only

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import PriorityQueue
>>> queue = PriorityQueue[tuple[int, str]]()

>>> queue.empty()
True

>>> await queue.put((1, "task"))
>>> queue.empty()
False
```

**See Also:**

- `qsize()`: Get approximate size
- `full()`: Check if queue is full

**Notes:**

**Anti-pattern** (race condition):

```python
# ❌ WRONG: Race between check and get
if not queue.empty():
    item = await queue.get_nowait()  # May raise QueueEmpty!
```

**Correct pattern**:

```python
# ✅ CORRECT: Handle exception
try:
    item = await queue.get_nowait()
except QueueEmpty:
    pass
```

#### `full()`

Check if queue is full (unlocked, racy).

**Signature:**

```python
def full(self) -> bool: ...
```

**Returns:**

- bool: True if queue is at maxsize (always False if maxsize=0)

**Thread-Safety:**

- **NOT thread-safe**: Reads `_queue` and `maxsize` without lock
- **Racy**: Value may be stale immediately after return
- **Use case**: Monitoring, debugging, approximate metrics only

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import PriorityQueue
>>> queue = PriorityQueue[tuple[int, str]](maxsize=2)

>>> queue.full()
False

>>> await queue.put((1, "task1"))
>>> await queue.put((2, "task2"))
>>> queue.full()
True

# Unlimited queue never full
>>> unlimited = PriorityQueue[tuple[int, str]](maxsize=0)
>>> unlimited.full()
False  # Always False
```

**See Also:**

- `qsize()`: Get approximate size
- `empty()`: Check if queue is empty

**Notes:**

**Anti-pattern** (race condition):

```python
# ❌ WRONG: Race between check and put
if not queue.full():
    await queue.put_nowait((1, "task"))  # May raise QueueFull!
```

**Correct pattern**:

```python
# ✅ CORRECT: Handle exception
try:
    await queue.put_nowait((1, "task"))
except QueueFull:
    pass
```

## Exceptions

### `QueueEmpty`

Exception raised when `get_nowait()` is called on empty queue.

**Signature:**

```python
class QueueEmpty(Exception):
    """Exception raised when queue.get_nowait() is called on empty queue."""
```

**Usage:**

```python
from lionherd_core.libs.concurrency import PriorityQueue, QueueEmpty

queue = PriorityQueue[tuple[int, str]]()

try:
    item = await queue.get_nowait()
except QueueEmpty:
    print("Queue is empty")
```

### `QueueFull`

Exception raised when `put_nowait()` is called on full queue.

**Signature:**

```python
class QueueFull(Exception):
    """Exception raised when queue.put_nowait() is called on full queue."""
```

**Usage:**

```python
from lionherd_core.libs.concurrency import PriorityQueue, QueueFull

queue = PriorityQueue[tuple[int, str]](maxsize=10)

try:
    await queue.put_nowait((1, "task"))
except QueueFull:
    print("Queue is full, use blocking put() or retry later")
```

## Usage Patterns

### Basic Producer-Consumer

```python
from lionherd_core.libs.concurrency import (
    PriorityQueue,
    create_task_group,
    sleep,
    run,
)

queue = PriorityQueue[tuple[int, str]](maxsize=10)

async def producer():
    """Add tasks with varying priorities."""
    await queue.put((1, "critical_task"))
    await queue.put((5, "normal_task"))
    await queue.put((10, "background_task"))
    print("Producer done")

async def consumer():
    """Process tasks by priority."""
    while True:
        priority, task = await queue.get()
        print(f"Processing {task} (priority={priority})")
        await sleep(0.1)  # Simulate work

async def main():
    async with create_task_group() as tg:
        tg.start_soon(producer)
        tg.start_soon(consumer)

run(main)
# Output (priority order):
# Processing critical_task (priority=1)
# Processing normal_task (priority=5)
# Processing background_task (priority=10)
```

### Non-Blocking Operations

```python
# noqa:validation
from lionherd_core.libs.concurrency import PriorityQueue, QueueEmpty, QueueFull

queue = PriorityQueue[tuple[int, str]](maxsize=2)

async def try_operations():
    # Try put (may fail if full)
    try:
        await queue.put_nowait((1, "task1"))
        await queue.put_nowait((2, "task2"))
        await queue.put_nowait((3, "task3"))  # Raises QueueFull
    except QueueFull:
        print("Queue full, dropping task or retrying later")

    # Try get (may fail if empty)
    try:
        while True:
            item = await queue.get_nowait()
            print(f"Got: {item}")
    except QueueEmpty:
        print("Queue empty, no more items")

await try_operations()
# Output:
# Queue full, dropping task or retrying later
# Got: (1, 'task1')
# Got: (2, 'task2')
# Queue empty, no more items
```

### Priority-Based Task Scheduling

```python
from lionherd_core.libs.concurrency import (
    PriorityQueue,
    create_task_group,
    sleep,
    run,
)
from enum import IntEnum

class Priority(IntEnum):
    """Task priority levels (lower = higher priority)."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 5
    LOW = 10

queue = PriorityQueue[tuple[Priority, str, callable]]()

async def schedule_task(priority: Priority, name: str, func: callable):
    """Schedule task with priority."""
    await queue.put((priority, name, func))

async def worker():
    """Process tasks by priority."""
    while True:
        priority, name, func = await queue.get()
        print(f"[P{priority}] Executing: {name}")
        await func()

async def main():
    # Schedule tasks (out of order)
    await schedule_task(Priority.NORMAL, "backup", lambda: sleep(0.1))
    await schedule_task(Priority.CRITICAL, "health_check", lambda: sleep(0.05))
    await schedule_task(Priority.LOW, "cleanup", lambda: sleep(0.2))
    await schedule_task(Priority.HIGH, "sync", lambda: sleep(0.1))

    # Worker processes in priority order
    async with create_task_group() as tg:
        tg.start_soon(worker)

run(main)
# Output (priority order):
# [P1] Executing: health_check
# [P2] Executing: sync
# [P5] Executing: backup
# [P10] Executing: cleanup
```

### Bounded Queue with Backpressure

```python
from lionherd_core.libs.concurrency import (
    PriorityQueue,
    create_task_group,
    sleep,
    run,
)

queue = PriorityQueue[tuple[int, str]](maxsize=3)

async def fast_producer():
    """Producer that generates faster than consumer."""
    for i in range(10):
        print(f"Producing task {i}...")
        await queue.put((i, f"task_{i}"))  # Blocks when queue full
        print(f"Task {i} queued")

async def slow_consumer():
    """Slow consumer creates backpressure."""
    while True:
        priority, task = await queue.get()
        print(f"  Consuming {task}...")
        await sleep(0.5)  # Slow processing
        print(f"  Finished {task}")

async def main():
    async with create_task_group() as tg:
        tg.start_soon(fast_producer)
        tg.start_soon(slow_consumer)

run(main)
# Output shows producer blocking when queue fills:
# Producing task 0...
# Task 0 queued
# Producing task 1...
# Task 1 queued
# Producing task 2...
# Task 2 queued
# Producing task 3...
#   Consuming task_0...  # Consumer starts, frees space
# Task 3 queued          # Producer unblocks
# ...
```

### Stable Priority with FIFO Ordering

```python
# noqa:validation
from lionherd_core.libs.concurrency import PriorityQueue, QueueEmpty

queue = PriorityQueue[tuple[int, int, str]]()

async def demonstrate_stable_priority():
    """Items with same priority maintain FIFO order."""
    # Add multiple items with same priority
    # Tuple: (priority, timestamp, data)
    await queue.put((5, 1, "first_normal"))
    await queue.put((5, 2, "second_normal"))
    await queue.put((1, 3, "critical"))
    await queue.put((5, 4, "third_normal"))

    # Retrieve all items (use exception handling, not racy empty() check)
    try:
        while True:
            priority, ts, task = await queue.get_nowait()
            print(f"[P{priority}] {task} (ts={ts})")
    except QueueEmpty:
        pass  # All items processed

await demonstrate_stable_priority()
# Output (stable within priority):
# [P1] critical (ts=3)
# [P5] first_normal (ts=1)   # FIFO within priority 5
# [P5] second_normal (ts=2)
# [P5] third_normal (ts=4)
```

## Design Rationale

### Why Async Nowait Methods?

Unlike `asyncio.PriorityQueue`, this implementation makes `put_nowait()` and `get_nowait()` **async** to ensure proper locking:

**Problem with Sync Nowait (asyncio pattern):**

```python
# asyncio.PriorityQueue (synchronous nowait - potential race)
def put_nowait(self, item):  # Not async
    # No lock acquisition
    if self.full():
        raise QueueFull
    heapq.heappush(self._queue, item)
    # Race condition: Another task could modify queue here
```

**Solution with Async Nowait (lionherd pattern):**

```python
# lionherd PriorityQueue (async nowait - safe)
async def put_nowait(self, item):  # Async
    async with self._condition:  # Atomic lock
        if self.full():
            raise QueueFull
        heapq.heappush(self._queue, item)
        self._condition.notify()
    # Lock ensures atomic check + insert
```

**Benefits:**

1. **No race conditions**: Check-then-insert is atomic
2. **Consistent API**: All mutating methods are async
3. **Proper notification**: Condition.notify() under lock is reliable

**Tradeoff**: Slightly more verbose (`await queue.put_nowait()` vs `queue.put_nowait()`), but eliminates subtle concurrency bugs.

### Why Unlocked Inspection Methods?

`qsize()`, `empty()`, and `full()` are intentionally **unlocked** for performance:

**Rationale:**

1. **Performance**: Lock-free reads avoid contention on hot paths (monitoring/metrics)
2. **Inherent races**: Even locked inspection is racy (value stale after lock release)
3. **Clear semantics**: Documented as "monitoring only" prevents misuse

**Anti-pattern to avoid:**

```python
# ❌ WRONG: Check-then-act pattern (racy even with locks)
if not queue.empty():
    item = await queue.get_nowait()  # May still raise QueueEmpty
```

**Correct patterns:**

```python
# ✅ GOOD: Exception handling
try:
    item = await queue.get_nowait()
except QueueEmpty:
    pass

# ✅ GOOD: Blocking (no races)
item = await queue.get()

# ✅ GOOD: Monitoring (informational only)
print(f"Queue depth: {queue.qsize()}")  # For metrics, not control flow
```

### Why Min-Heap (Lowest First)?

Uses `heapq` min-heap semantics (lowest value = highest priority):

**Rationale:**

1. **Standard pattern**: Matches `queue.PriorityQueue`, `asyncio.PriorityQueue`, Dijkstra's algorithm
2. **Natural for costs**: Lower cost = process first (execution time, resource usage)
3. **Enum compatibility**: `IntEnum` priority levels work naturally (CRITICAL=1, LOW=10)

**Inversion for max-heap:**

```python
# Use negative priorities for highest-first
await queue.put((-10, "highest"))  # -10 processed first
await queue.put((-5, "medium"))
await queue.put((-1, "lowest"))
```

### Why Optional Maxsize?

Supports both **bounded** (backpressure) and **unbounded** (buffer) use cases:

**Unbounded (maxsize=0, default):**

- **Use case**: Event buffering, task accumulation, no memory constraints
- **Behavior**: `put()` never blocks, queue grows indefinitely
- **Risk**: Unbounded memory growth if producers outpace consumers

**Bounded (maxsize > 0):**

- **Use case**: Rate limiting, memory-constrained systems, backpressure needed
- **Behavior**: `put()` blocks when full, throttles producers
- **Benefit**: Prevents memory exhaustion, natural flow control

**Design choice**: Default to unbounded (maxsize=0) for ease of use, require explicit opt-in for backpressure.

## Common Pitfalls

### Pitfall 1: Forgetting to Await Nowait Methods

**Issue**: `put_nowait()` and `get_nowait()` are async, unlike asyncio.

```python
# ❌ WRONG: Missing await
item = queue.get_nowait()  # Returns coroutine, not item!

# ✅ CORRECT: Await the coroutine
item = await queue.get_nowait()
```

**Symptom**: `RuntimeWarning: coroutine was never awaited` or incorrect results.

**Solution**: Always `await` all queue methods (including nowait).

### Pitfall 2: Using Inspection for Control Flow

**Issue**: `qsize()`, `empty()`, `full()` are racy (unlocked).

```python
# ❌ WRONG: Race condition
if not queue.empty():
    item = await queue.get_nowait()  # May raise QueueEmpty!

# ✅ CORRECT: Exception handling
try:
    item = await queue.get_nowait()
except QueueEmpty:
    item = None
```

**Symptom**: Intermittent `QueueEmpty` or `QueueFull` exceptions.

**Solution**: Use exception handling or blocking methods (`get()`, `put()`).

### Pitfall 3: Non-Comparable Items

**Issue**: Items must be comparable for heapq.

```python
# ❌ WRONG: Dicts are not ordered
await queue.put({"priority": 1, "task": "x"})  # TypeError!

# ✅ CORRECT: Tuple with priority first
await queue.put((1, {"task": "x"}))
```

**Symptom**: `TypeError: '<' not supported between instances` during `get()`.

**Solution**: Use tuples with numeric priority as first element.

### Pitfall 4: Priority Inversion

**Issue**: Forgetting min-heap semantics (lower = higher priority).

```python
# ❌ CONFUSING: High number for high priority
await queue.put((10, "critical"))  # Processed LAST (lowest priority)
await queue.put((1, "background"))  # Processed FIRST (highest priority)

# ✅ CLEAR: Use IntEnum with low values for high priority
class Priority(IntEnum):
    CRITICAL = 1  # Highest priority
    NORMAL = 5
    LOW = 10      # Lowest priority

await queue.put((Priority.CRITICAL, "task"))
```

**Symptom**: Tasks processed in reverse of expected order.

**Solution**: Use IntEnum or constants to document priority semantics.

### Pitfall 5: Unbounded Growth

**Issue**: Default `maxsize=0` allows unlimited growth.

```python
# ❌ RISKY: Unbounded queue with fast producer
queue = PriorityQueue()
for i in range(1_000_000):
    await queue.put((i, f"task_{i}"))  # Never blocks, memory grows

# ✅ SAFE: Bounded queue with backpressure
queue = PriorityQueue(maxsize=100)
for i in range(1_000_000):
    await queue.put((i, f"task_{i}"))  # Blocks when full, applies backpressure
```

**Symptom**: Memory exhaustion, OOM errors.

**Solution**: Use `maxsize > 0` for memory-constrained systems or fast producers.

## See Also

- **Related Classes**:
  - `anyio.Queue`: FIFO queue (simpler when priority not needed)
  - `queue.PriorityQueue`: Thread-based priority queue (stdlib)
  - `asyncio.PriorityQueue`: Asyncio priority queue (sync nowait methods)
- **Related Primitives**:
  - `Condition`: Underlying locking primitive (from `_primitives.py`)
  - `heapq`: Min-heap implementation (Python stdlib)
<!-- TODO: Add links to user guides when available (issue #115) -->

## Examples

### Example 1: Event Processing by Severity

```python
from lionherd_core.libs.concurrency import (
    PriorityQueue,
    create_task_group,
    sleep,
    run,
)
from enum import IntEnum

class Severity(IntEnum):
    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4

queue = PriorityQueue[tuple[Severity, str]]()

async def log_event(severity: Severity, message: str):
    """Log event to priority queue."""
    await queue.put((severity, message))

async def process_logs():
    """Process logs by severity."""
    while True:
        severity, message = await queue.get()
        print(f"[{severity.name}] {message}")
        await sleep(0.1)

async def main():
    # Log events out of order
    await log_event(Severity.INFO, "System started")
    await log_event(Severity.CRITICAL, "Database connection lost")
    await log_event(Severity.WARNING, "High memory usage")
    await log_event(Severity.ERROR, "Failed to process request")

    # Process in severity order
    async with create_task_group() as tg:
        tg.start_soon(process_logs)

run(main)
# Output (by severity):
# [CRITICAL] Database connection lost
# [ERROR] Failed to process request
# [WARNING] High memory usage
# [INFO] System started
```

### Example 2: Rate-Limited API with Priority

```python
from lionherd_core.libs.concurrency import (
    PriorityQueue,
    QueueFull,
    create_task_group,
    sleep,
    run,
)

class RequestPriority(IntEnum):
    VIP = 1      # Premium users
    NORMAL = 5   # Regular users
    THROTTLED = 10  # Rate-limited users

queue = PriorityQueue[tuple[RequestPriority, str]](maxsize=100)

async def submit_request(priority: RequestPriority, user_id: str):
    """Submit API request with priority."""
    try:
        await queue.put_nowait((priority, user_id))
        print(f"Queued: {user_id} (P{priority})")
    except QueueFull:
        print(f"Rejected: {user_id} (queue full)")

async def api_worker():
    """Process requests with 10/sec rate limit."""
    while True:
        priority, user_id = await queue.get()
        print(f"Processing: {user_id} (P{priority})")
        await sleep(0.1)  # 10 requests/sec

async def main():
    # Submit mixed priority requests
    await submit_request(RequestPriority.NORMAL, "user_1")
    await submit_request(RequestPriority.VIP, "vip_user")
    await submit_request(RequestPriority.THROTTLED, "slow_user")
    await submit_request(RequestPriority.NORMAL, "user_2")

    async with create_task_group() as tg:
        tg.start_soon(api_worker)

run(main)
# Output (VIP processed first):
# Queued: user_1 (P5)
# Queued: vip_user (P1)
# Queued: slow_user (P10)
# Queued: user_2 (P5)
# Processing: vip_user (P1)
# Processing: user_1 (P5)
# Processing: user_2 (P5)
# Processing: slow_user (P10)
```

### Example 3: Deadline-Based Scheduling

```python
from lionherd_core.libs.concurrency import (
    PriorityQueue,
    create_task_group,
    sleep,
    run,
)
import time

queue = PriorityQueue[tuple[float, str, callable]]()

async def schedule_task(deadline: float, name: str, func: callable):
    """Schedule task by deadline (earliest first)."""
    await queue.put((deadline, name, func))

async def scheduler():
    """Execute tasks by deadline."""
    while True:
        deadline, name, func = await queue.get()
        now = time.time()

        # Wait until deadline
        delay = deadline - now
        if delay > 0:
            print(f"Waiting {delay:.2f}s for {name}...")
            await sleep(delay)

        print(f"Executing {name}")
        await func()

async def main():
    now = time.time()

    # Schedule tasks with deadlines
    await schedule_task(now + 3.0, "task_3s", lambda: sleep(0.1))
    await schedule_task(now + 1.0, "task_1s", lambda: sleep(0.1))
    await schedule_task(now + 2.0, "task_2s", lambda: sleep(0.1))

    async with create_task_group() as tg:
        tg.start_soon(scheduler)

run(main)
# Output (by deadline):
# Waiting 1.00s for task_1s...
# Executing task_1s
# Waiting 1.00s for task_2s...
# Executing task_2s
# Waiting 1.00s for task_3s...
# Executing task_3s
```

### Example 4: Graceful Degradation with Priorities

```python
from lionherd_core.libs.concurrency import (
    PriorityQueue,
    QueueFull,
    create_task_group,
    sleep,
    run,
)

class TaskPriority(IntEnum):
    ESSENTIAL = 1   # Must complete
    IMPORTANT = 5   # Should complete
    OPTIONAL = 10   # Best effort

queue = PriorityQueue[tuple[TaskPriority, str]](maxsize=5)

async def submit_tasks():
    """Submit tasks under load."""
    tasks = [
        (TaskPriority.OPTIONAL, "cache_warmup"),
        (TaskPriority.ESSENTIAL, "process_payment"),
        (TaskPriority.IMPORTANT, "send_notification"),
        (TaskPriority.OPTIONAL, "update_analytics"),
        (TaskPriority.ESSENTIAL, "commit_transaction"),
    ]

    for priority, task in tasks:
        try:
            await queue.put_nowait((priority, task))
            print(f"Queued: {task}")
        except QueueFull:
            if priority == TaskPriority.ESSENTIAL:
                # Block for essential tasks
                print(f"Waiting for space: {task}")
                await queue.put((priority, task))
            else:
                # Drop optional/important tasks under load
                print(f"Dropped: {task} (queue full)")

async def worker():
    """Process tasks by priority."""
    while True:
        priority, task = await queue.get()
        print(f"Processing: {task} (P{priority})")
        await sleep(0.2)

async def main():
    async with create_task_group() as tg:
        tg.start_soon(worker)
        tg.start_soon(submit_tasks)

run(main)
# Output (essential tasks prioritized):
# Queued: cache_warmup
# Queued: process_payment
# Queued: send_notification
# Queued: update_analytics
# Queued: commit_transaction
# Processing: process_payment (P1)
# Processing: commit_transaction (P1)
# Processing: send_notification (P5)
# Processing: cache_warmup (P10)
# Processing: update_analytics (P10)
```

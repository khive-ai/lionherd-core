# Executor

> Flow-based event state tracking with O(1) status queries and background processing.

---

## Overview

**Executor** manages event lifecycle using Flow-based state tracking where each
EventStatus maps 1:1 to a named Progression. This architecture enables O(1) status
queries, explainability, audit trails, and type safety.

**Core Capabilities**:

- **Flow-based state tracking**: Events stored in Flow.items, status tracked via progressions
- **O(1) status queries**: Direct progression lookup vs O(n) pile scanning
- **Background processing**: Creates and manages Processor instance for async execution
- **Type safety**: Optional strict event type enforcement via Flow
- **Explainability**: `inspect_state()` provides human-readable status summary

**When to Use Executor:**

- You need to track event status across lifecycle (pending → processing → completed/failed)
- You want O(1) performance for status queries (not O(n) scanning)
- You need audit trail of all events by status
- You want background event processing with priority queue

**Don't use Executor when:**

- Simple event.invoke() is sufficient → use Event directly
- No need for status tracking → use async functions
- No background processing needed → manual event invocation

---

## Architecture

### Flow-Based State Tracking

Executor uses Flow with progressions mapping 1:1 to EventStatus values:

```python
# EventStatus enum values → Progression names
EventStatus.PENDING     → Progression(name="pending")
EventStatus.PROCESSING  → Progression(name="processing")
EventStatus.COMPLETED   → Progression(name="completed")
EventStatus.FAILED      → Progression(name="failed")
EventStatus.CANCELLED   → Progression(name="cancelled")
EventStatus.SKIPPED     → Progression(name="skipped")
EventStatus.ABORTED     → Progression(name="aborted")
```

**Benefits**:

- **O(1) queries**: `get_events_by_status()` is direct progression lookup (not O(n) scan)
- **Type safety**: EventStatus enum prevents invalid status strings
- **Explainability**: `inspect_state()` shows all status counts at a glance
- **Audit trail**: Flow serialization captures full state history
- **Memory overhead**: ~11% (7 progressions × small metadata)

**Trade-off**: Accepted ~11% memory overhead for O(1) performance and explainability.

### Integration with Processor

Executor creates and manages Processor instance:

```python
executor = MyExecutor()           # Executor.__init__() creates Flow
await executor.start()            # Creates Processor, backfills pending events
await executor.append(event)      # Adds to Flow + enqueues in Processor
# Processor updates progressions via executor._update_progression()
```

**See Also**: [Processor API](processor.md) for processor-specific details.
For integration patterns, see [Processor/Executor Integration](../../user_guide/processor_executor.md#integration-architecture).

---

## Class Signature

```python
class Executor:
    """Event executor with Flow-based state tracking and background processing."""

    processor_type: ClassVar[type[Processor]]  # Processor subclass to create
```

**Generic Parameters**: None (Executor is not generic)

**ClassVar**:

- `processor_type`: Processor subclass this executor creates. Must override in subclass.

---

## Constructor

```python
def __init__(
    self,
    processor_config: dict[str, Any] | None = None,
    strict_event_type: bool = False,
    name: str | None = None,
) -> None:
    """Initialize executor with Flow-based state management.

    Args:
        processor_config: Config dict for creating Processor
        strict_event_type: If True, Flow enforces exact type matching
        name: Optional name for the executor Flow
    """
```

**Parameters**:

- `processor_config` (dict[str, Any] | None): Configuration dict passed to `Processor.create()`.
  Keys: `queue_capacity` (required), `capacity_refresh_time` (required), `concurrency_limit`,
  `max_queue_size`, `max_denial_tracking`. Default: `{}`.
  **Note**: `queue_capacity` and `capacity_refresh_time` are required if you plan to call `start()`.
- `strict_event_type` (bool): If True, Flow enforces exact event type (no subclasses).
  Default: False (allow event subclasses)
- `name` (str | None): Optional name for Flow. Default: "executor_states"

**Example**:

```python
from lionherd_core.base import Event, Processor, Executor

class MyEvent(Event):
    async def _invoke(self):
        return "result"

class MyProcessor(Processor):
    event_type = MyEvent

class MyExecutor(Executor):
    processor_type = MyProcessor

# Create executor with custom config
executor = MyExecutor(
    processor_config={
        "queue_capacity": 100,
        "capacity_refresh_time": 1.0,
        "concurrency_limit": 50,
    },
    strict_event_type=True,  # Enforce exact MyEvent type
    name="my_executor"
)
```

---

## Attributes

### Class Variables

#### `processor_type`

```python
processor_type: ClassVar[type[Processor]]
```

Processor subclass this executor creates. Must be overridden in subclass.

**Type:** ClassVar[type[Processor]]

**Usage:**

```python
class MyExecutor(Executor):
    processor_type = MyProcessor  # Required override
```

### Instance Attributes

#### `states`

```python
states: Flow[Event, Progression]
```

Flow managing event storage and status progressions. Created in `__init__()`.

**Type:** Flow[Event, Progression]

**Structure:**

- `states.items`: Pile[Event] - all events
- `states.progressions`: Pile[Progression] - one per EventStatus value

**Access:** Public (for advanced use cases)

**Thread Safety:** Built-in async locks via Pile

#### `processor`

```python
processor: Processor | None
```

Background processor instance. Created on first `start()` call.

**Type:** Processor | None

**Lifecycle:**

- `None` until first `start()` call
- Created via `await executor._create_processor()`
- Reused across `start()/stop()` cycles

#### `processor_config`

```python
processor_config: dict[str, Any]
```

Configuration dict for creating Processor. Set in `__init__()`.

**Type:** dict[str, Any]

**Keys**: `queue_capacity`, `capacity_refresh_time`, `concurrency_limit`,
`max_queue_size`, `max_denial_tracking`

---

## Properties

### `event_type`

```python
@property
def event_type(self) -> type[Event]:
    """Event subclass handled by processor."""
```

Event subclass this executor processes. Derived from `processor_type.event_type`.

**Returns:** type[Event]

**Example:**

```python
executor = MyExecutor()
print(executor.event_type)  # <class 'MyEvent'>
```

### `strict_event_type`

```python
@property
def strict_event_type(self) -> bool:
    """Whether Flow enforces exact event type matching."""
```

Returns True if Flow enforces exact type (no subclasses).

**Returns:** bool

**Example:**

```python
executor = MyExecutor(strict_event_type=True)
print(executor.strict_event_type)  # True
```

### Status Shortcuts

#### `completed_events`

```python
@property
def completed_events(self) -> list[Event]:
    """All events with COMPLETED status."""
```

Shortcut for `get_events_by_status(EventStatus.COMPLETED)`.

**Returns:** list[Event]

**Performance:** O(1) progression lookup + O(n) event fetching

**Example:**

```python
completed = executor.completed_events
print(f"Completed: {len(completed)}")
```

#### `pending_events`

```python
@property
def pending_events(self) -> list[Event]:
    """All events with PENDING status."""
```

Shortcut for `get_events_by_status(EventStatus.PENDING)`.

#### `failed_events`

```python
@property
def failed_events(self) -> list[Event]:
    """All events with FAILED status."""
```

Shortcut for `get_events_by_status(EventStatus.FAILED)`.

#### `processing_events`

```python
@property
def processing_events(self) -> list[Event]:
    """All events with PROCESSING status."""
```

Shortcut for `get_events_by_status(EventStatus.PROCESSING)`.

---

## Methods

### Event Management

#### `append()`

```python
async def append(self, event: Event, priority: float | None = None) -> None:
    """Add event to Flow and enqueue for processing.

    Args:
        event: Event to add
        priority: Priority value (lower = higher priority). Defaults to event.created_at.
    """
```

Adds event to Flow with PENDING status and enqueues in Processor (if started).

**Parameters:**

- `event` (Event): Event instance to add
- `priority` (float | None): Priority value (lower = higher priority). Default: `event.created_at.timestamp()`

**Async:** Yes (enqueues in Processor)

**Example:**

```python
event = MyEvent()
await executor.append(event)  # Default priority (created_at)

# Custom priority (lower = higher priority)
high_priority_event = MyEvent()
await executor.append(high_priority_event, priority=1.0)
```

**See Also:** [Processor.enqueue()](processor.md#enqueue) for queueing details

#### `cleanup_events()`

```python
async def cleanup_events(self, statuses: list[EventStatus] | None = None) -> int:
    """Remove events with specified statuses from executor.

    This is a manual cleanup method for memory management. Events are removed
    from both progressions and the Flow.items pile. Also cleans up processor
    denial tracking to prevent memory leaks.

    Thread-safe: Uses Pile's built-in async locks to prevent race conditions
    with concurrent _update_progression() calls.

    Args:
        statuses: List of statuses to clean up (default: [COMPLETED, FAILED, ABORTED])

    Returns:
        Number of events removed

    Example:
        # Clean up after events are logged elsewhere
        await executor.cleanup_events([EventStatus.COMPLETED, EventStatus.FAILED])
    """
```

Removes events from Flow and cleans up processor denial tracking.

**Parameters:**

- `statuses` (list[EventStatus] | None): Statuses to clean up. Default: `[COMPLETED, FAILED, ABORTED]`

**Returns:** int (number of events removed)

**Thread Safety:** Acquires locks in consistent order (items first, progressions second) to prevent deadlock

**Async:** Yes (acquires async locks)

**Example:**

```python
# Clean up completed and failed events
removed = await executor.cleanup_events()
print(f"Removed {removed} events")

# Clean up only completed
removed = await executor.cleanup_events([EventStatus.COMPLETED])
```

---

### Status Queries

#### `get_events_by_status()`

```python
def get_events_by_status(self, status: EventStatus | str) -> list[Event]:
    """Get all events with given status.

    Args:
        status: EventStatus enum or status string

    Returns:
        List of events
    """
```

O(1) status query via direct progression lookup.

**Parameters:**

- `status` (EventStatus | str): EventStatus enum or status string (e.g., "pending")

**Returns:** list[Event]

**Performance:** O(1) progression lookup + O(n) event fetching (n = events with status)

**Example:**

```python
from lionherd_core.base import EventStatus

# Using enum (recommended)
pending = executor.get_events_by_status(EventStatus.PENDING)

# Using string (also works)
completed = executor.get_events_by_status("completed")

print(f"Pending: {len(pending)}, Completed: {len(completed)}")
```

**Design Note:** O(1) lookup is core benefit vs O(n) pile scanning.

#### `status_counts()`

```python
def status_counts(self) -> dict[str, int]:
    """Get event count per status."""
```

Returns counts for all statuses (not just non-zero).

**Returns:** dict[str, int] - Mapping of status names to counts

**Example:**

```python
counts = executor.status_counts()
print(counts)
# {'pending': 5, 'processing': 2, 'completed': 10, 'failed': 1, ...}

total = sum(counts.values())
print(f"Total events: {total}")
```

---

### Lifecycle Control

#### `start()`

```python
async def start(self) -> None:
    """Initialize and start processor if not created, backfilling pending events."""
```

Creates Processor on first call and backfills all pending events into queue.

**Async:** Yes (creates processor, enqueues events)

**Idempotent:** Yes (safe to call multiple times)

**Behavior:**

1. If processor doesn't exist: creates via `_create_processor()`
2. Backfills all pending events into processor queue
3. Clears processor stop signal

**Example:**

```python
executor = MyExecutor()
await executor.append(MyEvent())  # Queued but not processed yet
await executor.start()            # Creates processor, starts processing
```

#### `stop()`

```python
async def stop(self) -> None:
    """Stop processor if exists."""
```

Signals processor to stop processing events. Does not destroy processor.

**Async:** Yes (sets stop signal)

**Behavior:**

- Sets processor stop event
- Clears denial tracking (prevent memory leaks)
- Processor can be restarted via `start()`

**Example:**

```python
await executor.stop()   # Stop processing
# ... do maintenance ...
await executor.start()  # Resume processing
```

#### `forward()`

```python
async def forward(self) -> None:
    """Process queued events immediately."""
```

Triggers immediate processing batch (bypasses capacity refresh interval).

**Async:** Yes (calls processor.process())

**Use Case:** Flush queue without waiting for refresh interval

**Example:**

```python
await executor.append(urgent_event)
await executor.forward()  # Process immediately, don't wait
```

---

### Debugging

#### `inspect_state()`

```python
def inspect_state(self) -> str:
    """Debug helper: show counts per status."""
```

Returns human-readable status summary.

**Returns:** str (multi-line status report)

**Example:**

```python
print(executor.inspect_state())
# Output:
# Executor State (my_executor):
#   pending: 5 events
#   processing: 2 events
#   completed: 10 events
#   failed: 1 event
#   cancelled: 0 events
#   skipped: 0 events
#   aborted: 0 events
```

**Use Case:** Quick debugging, monitoring dashboards

---

### Special Methods

#### `__contains__()`

```python
def __contains__(self, event: Event | UUID) -> bool:
    """Check if event is in Flow."""
```

Enables `in` operator for membership testing.

**Parameters:**

- `event` (Event | UUID): Event instance or UUID to check

**Returns:** bool

**Example:**

```python
event = MyEvent()
await executor.append(event)

print(event in executor)        # True
print(event.id in executor)     # True (UUID also works)
print(MyEvent() in executor)    # False (different UUID)
```

#### `__repr__()`

```python
def __repr__(self) -> str:
    """String representation with status counts."""
```

Concise representation showing total and status breakdown.

**Returns:** str

**Example:**

```python
print(repr(executor))
# Executor(total=18, pending=5, processing=2, completed=10, failed=1, cancelled=0, skipped=0, aborted=0)
```

---

## Internal Methods

### `_update_progression()`

```python
async def _update_progression(
    self, event: Event, force_status: EventStatus | None = None
) -> None:
    """Update Flow progression to match event status (thread-safe via Pile lock).

    Raises:
        ConfigurationError: If progression for status doesn't exist
    """
```

**⚠️ Internal Use Only**: Called by Processor during event execution.

**Thread Safety:** Uses Pile async locks to prevent race conditions

**Behavior:**

1. Acquires progression lock
2. Removes event from all progressions (enforce single-ownership)
3. Adds to target progression (based on event.execution.status or force_status)

**Raises:** ConfigurationError if progression missing (shouldn't happen with correct init)

### `_create_processor()`

```python
async def _create_processor(self) -> None:
    """Instantiate processor with config."""
```

**⚠️ Internal Use Only**: Called by `start()` on first invocation.

**Behavior:** Creates processor via `await processor_type.create()` with `processor_config`

---

## Design Benefits

### O(1) Performance

**Problem**: Scanning pile to find events by status is O(n):

```python
# O(n) scan - BAD
completed = [e for e in pile if e.execution.status == EventStatus.COMPLETED]
```

**Solution**: Direct progression lookup is O(1):

```python
# O(1) lookup - GOOD
completed = executor.get_events_by_status(EventStatus.COMPLETED)
```

**Trade-off**: ~11% memory overhead (7 progressions) for O(1) queries.

### Explainability

`inspect_state()` provides instant overview without scanning:

```python
print(executor.inspect_state())
# Shows all 7 statuses with counts - instant debugging
```

### Audit Trail

Flow serialization captures full state history:

```python
state_snapshot = executor.states.to_dict()
# Includes all events and progressions - full audit trail
```

### Type Safety

EventStatus enum prevents invalid status strings:

```python
# ✓ Type-safe
executor.get_events_by_status(EventStatus.PENDING)

# ✓ Also works (runtime validation)
executor.get_events_by_status("pending")

# ✗ Would fail
executor.get_events_by_status("invalid_status")
```

---

## Usage Patterns

### Basic Usage

```python
from lionherd_core.base import Event, Processor, Executor, EventStatus

# 1. Define custom event
class DataProcessingEvent(Event):
    data: dict = {}

    async def _invoke(self):
        # Process data
        return {"processed": self.data}

# 2. Define processor
class DataProcessor(Processor):
    event_type = DataProcessingEvent

# 3. Define executor
class DataExecutor(Executor):
    processor_type = DataProcessor

# 4. Create and start (processor_config required for start())
executor = DataExecutor(
    processor_config={
        "queue_capacity": 100,
        "capacity_refresh_time": 1.0,
    }
)
await executor.start()

# 5. Add events
for i in range(10):
    event = DataProcessingEvent(data={"id": i})
    await executor.append(event)

# 6. Query status
pending = executor.pending_events
print(f"Pending: {len(pending)}")

# 7. Wait and check results
await executor.processor.join()  # Wait for queue to empty
completed = executor.completed_events
print(f"Completed: {len(completed)}")
```

### Status Queries (All Patterns)

```python
# Pattern 1: Property shortcuts
completed = executor.completed_events
failed = executor.failed_events
pending = executor.pending_events
processing = executor.processing_events

# Pattern 2: Generic query
completed = executor.get_events_by_status(EventStatus.COMPLETED)
failed = executor.get_events_by_status("failed")  # String also works

# Pattern 3: Counts only
counts = executor.status_counts()
print(f"Completed: {counts['completed']}")

# Pattern 4: Debugging
print(executor.inspect_state())
```

### State Inspection

```python
# Quick overview
print(executor.inspect_state())

# Detailed counts
counts = executor.status_counts()
total = sum(counts.values())
completion_rate = counts['completed'] / total if total > 0 else 0
print(f"Completion rate: {completion_rate:.1%}")

# Check specific events
event = MyEvent()
await executor.append(event)
print(event in executor)  # True

# Audit trail (serialization)
snapshot = executor.states.to_dict()
# Save snapshot for debugging/monitoring
```

### Memory Management

```python
# Clean up completed/failed/aborted events (default)
removed = await executor.cleanup_events()
print(f"Cleaned up {removed} events")

# Clean up only completed (keep failed for analysis)
removed = await executor.cleanup_events([EventStatus.COMPLETED])

# Manual cleanup before serialization
await executor.cleanup_events()
snapshot = executor.states.to_dict()  # Smaller snapshot
```

---

## Common Pitfalls

### 1. Forgetting to await start()

```python
# ❌ WRONG: Processor not created
executor = MyExecutor()
await executor.append(event)  # Queued but never processed

# ✓ CORRECT: Start processor first
await executor.start()
await executor.append(event)  # Processed in background
```

### 2. Not cleaning up events

```python
# ❌ WRONG: Unbounded memory growth
for i in range(100000):
    await executor.append(MyEvent())
# Memory grows without cleanup

# ✓ CORRECT: Periodic cleanup
for i in range(100000):
    await executor.append(MyEvent())
    if i % 1000 == 0:
        await executor.cleanup_events()
```

### 3. Accessing processor before start()

```python
# ❌ WRONG: Processor is None
executor = MyExecutor()
await executor.processor.join()  # AttributeError

# ✓ CORRECT: Check if processor exists
if executor.processor:
    await executor.processor.join()

# Or ensure started
await executor.start()
await executor.processor.join()  # Safe
```

### 4. Misunderstanding priority values

```python
# ❌ WRONG: Higher number = higher priority
await executor.append(urgent_event, priority=1000)  # Processed LAST

# ✓ CORRECT: Lower number = higher priority
await executor.append(urgent_event, priority=1.0)   # Processed FIRST
await executor.append(normal_event, priority=100.0) # Processed LATER
```

---

## Configuration Guide

### processor_config Structure

```python
executor = MyExecutor(
    processor_config={
        # Required
        "queue_capacity": 100,           # Max events per batch (1-10000)
        "capacity_refresh_time": 1.0,    # Seconds between batches (0.01-3600)

        # Optional (defaults shown)
        "concurrency_limit": 100,        # Max concurrent executions
        "max_queue_size": 1000,          # Max queue size before rejection
        "max_denial_tracking": 10000,    # Max denial entries (FIFO eviction)
    }
)
```

**Tuning Recommendations:**

- **Low latency**: Small `capacity_refresh_time` (0.1-0.5s), high `queue_capacity` (500-1000)
- **High throughput**: Large `queue_capacity` (1000-5000), moderate `capacity_refresh_time` (1-5s)
- **Resource-constrained**: Low `concurrency_limit` (10-50), small `queue_capacity` (50-100)

**See Also:** [Processor Configuration](processor.md#configuration-guide) for detailed tuning

### strict_event_type Usage

```python
# Allow subclasses (default)
executor = MyExecutor(strict_event_type=False)
await executor.append(MyEvent())         # ✓
await executor.append(MyEventSubclass()) # ✓

# Enforce exact type
executor = MyExecutor(strict_event_type=True)
await executor.append(MyEvent())         # ✓
await executor.append(MyEventSubclass()) # ✗ TypeError
```

**When to use `strict_event_type=True`:**

- Type safety critical (e.g., financial transactions)
- Event subclasses have incompatible behavior
- Debugging type-related issues

**Default (`False`)**: More flexible, supports polymorphism

---

## See Also

- [Processor](processor.md) - Background event processing with priority queue
- [Event](event.md) - Async execution with lifecycle tracking
- [Flow](flow.md) - Flow-based state management architecture
- [EventStatus](event.md#eventstatus) - Event lifecycle states
- [Processor/Executor Integration](../../user_guide/processor_executor.md) - Integration patterns and workflows

---

## Examples

### Complete Example: Task Processing System

```python
from lionherd_core.base import Event, Processor, Executor, EventStatus
import asyncio

# 1. Define event for data processing tasks
class TaskEvent(Event):
    task_id: int = 0
    data: str = ""

    async def _invoke(self):
        # Simulate processing
        await asyncio.sleep(0.1)
        return {"task_id": self.task_id, "result": f"Processed: {self.data}"}

# 2. Define processor
class TaskProcessor(Processor):
    event_type = TaskEvent

# 3. Define executor
class TaskExecutor(Executor):
    processor_type = TaskProcessor

# 4. Run system
async def main():
    # Create executor with config
    executor = TaskExecutor(
        processor_config={
            "queue_capacity": 50,
            "capacity_refresh_time": 0.5,
            "concurrency_limit": 20,
        }
    )

    # Start processing
    await executor.start()

    # Add tasks with priorities
    for i in range(100):
        task = TaskEvent(task_id=i, data=f"Task {i}")
        # Urgent tasks (< 10) get higher priority
        priority = 1.0 if i < 10 else 100.0
        await executor.append(task, priority=priority)

    # Monitor progress
    while True:
        counts = executor.status_counts()
        print(f"Progress: {counts}")

        if counts['pending'] == 0 and counts['processing'] == 0:
            break

        await asyncio.sleep(1)

    # Check results
    completed = executor.completed_events
    failed = executor.failed_events
    print(f"✓ Completed: {len(completed)}, ✗ Failed: {len(failed)}")

    # Cleanup
    await executor.cleanup_events()
    await executor.stop()

asyncio.run(main())
```

### Example: Rate-Limited API Calls

```python
# Use custom permission check for rate limiting
class RateLimitedProcessor(Processor):
    event_type = TaskEvent

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_request = 0
        self.min_interval = 0.1  # 100ms between requests

    async def request_permission(self, **kwargs):
        import time
        now = time.time()
        if now - self.last_request < self.min_interval:
            return False  # Deny, will retry with backoff
        self.last_request = now
        return True

class RateLimitedExecutor(Executor):
    processor_type = RateLimitedProcessor

# Usage
executor = RateLimitedExecutor(
    processor_config={
        "queue_capacity": 10,
        "capacity_refresh_time": 0.5,
    }
)
await executor.start()

for i in range(50):
    await executor.append(TaskEvent(task_id=i))

# Events processed with rate limiting (max 10 per second)
```

---

**Version**: 1.0
**Status**: Production Ready
**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

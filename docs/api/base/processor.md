# Processor

> Background event processor with priority queue and capacity-limited batch execution.

---

## Overview

**Processor** is a background event execution engine designed for high-throughput async workflows. It manages a priority queue of events, enforces capacity limits per batch cycle, and provides customizable permission checks for rate limiting and authorization.

**Core Responsibilities:**

- **Priority Queue Execution**: Process events ordered by priority value (default: creation time)
- **Capacity-Limited Batching**: Process up to `queue_capacity` events per `capacity_refresh_time` interval
- **Permission Gating**: Override `request_permission()` for rate limiting, quotas, or auth checks
- **Concurrency Control**: Limit concurrent executions via semaphore (default: 100)
- **Flow Integration**: Updates executor progressions automatically when events transition between states

**When to Use Processor:**

- **High-volume event processing**: Thousands of events with priority-based scheduling
- **Rate-limited operations**: External API calls with quota restrictions
- **Background task execution**: Long-running operations without blocking main workflow
- **Resource-constrained systems**: Control batch size and concurrency to prevent overload

**When Not to Use:**

- Simple sequential workflows → Call `event.invoke()` directly
- Single event execution → No need for queue overhead
- Real-time processing with zero latency tolerance → Queue adds batching delay

**Architecture Integration:** Processor is created by [Executor](executor.md) and shares a reference to `Executor.states.items` (the Flow's Pile containing events). The queue stores event UUIDs only, keeping memory overhead minimal. For integration patterns, see [Processor/Executor Integration](../../user_guide/processor_executor.md).

---

## Class Signature

```python
from lionherd_core.base import Processor

class Processor:
    """Background event processor with priority queue and capacity control.

    Attributes:
        event_type: Event subclass this processor handles
        queue_capacity: Max events per batch
        capacity_refresh_time: Seconds before capacity reset
        concurrency_limit: Max concurrent executions
        pile: Reference to executor's Flow.items
        executor: Reference to executor for progression updates
    """

    event_type: ClassVar[type[Event]]

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event],
        executor: Executor | None = None,
        concurrency_limit: int = 100,
        max_queue_size: int = 1000,
        max_denial_tracking: int = 10000,
    ) -> None: ...
```

---

## Parameters

### `__init__`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queue_capacity` | `int` | *(required)* | Max events per batch (must be in [1, 10000]) |
| `capacity_refresh_time` | `float` | *(required)* | Refresh interval in seconds (must be in [0.01, 3600]) |
| `pile` | `Pile[Event]` | *(required)* | Reference to executor's Flow.items for event storage |
| `executor` | `Executor \| None` | `None` | Reference to executor for progression updates |
| `concurrency_limit` | `int` | `100` | Max concurrent executions (prevents resource exhaustion) |
| `max_queue_size` | `int` | `1000` | Max queue size before rejecting new events |
| `max_denial_tracking` | `int` | `10000` | Max denial entries to track (prevents unbounded growth) |

**Validation Rules:**

- `queue_capacity`: Must be ≥ 1 and ≤ 10000 (prevent unbounded batches)
- `capacity_refresh_time`: Must be ≥ 0.01s (prevent CPU hot loop) and ≤ 3600s (prevent starvation)
- `concurrency_limit`: Must be ≥ 1
- `max_queue_size`: Must be ≥ 1
- `max_denial_tracking`: Must be ≥ 1

**Raises:**

- `ValueError`: If any parameter is out of bounds

---

## Attributes

### Class Variables

| Attribute | Type | Description |
|-----------|------|-------------|
| `event_type` | `ClassVar[type[Event]]` | Event subclass this processor handles (must be set in subclass) |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `queue_capacity` | `int` | Max events per batch cycle |
| `capacity_refresh_time` | `float` | Seconds before capacity resets to `queue_capacity` |
| `max_queue_size` | `int` | Max queue size (prevents unbounded memory growth) |
| `max_denial_tracking` | `int` | Max denial entries tracked (FIFO eviction when exceeded) |
| `pile` | `Pile[Event]` | Reference to executor's event storage (Flow.items) |
| `executor` | `Executor \| None` | Reference to executor for automatic progression updates |
| `queue` | `PriorityQueue[tuple[float, UUID]]` | Priority queue storing `(priority, event_uuid)` tuples (lower priority values processed first) |

**Private Attributes** (implementation details):

- `_available_capacity`: Current capacity remaining in batch cycle
- `_execution_mode`: Whether processor is actively executing (`execute()` loop)
- `_stop_event`: Async event for signaling stop
- `_denial_counts`: Tracks permission denials per event UUID (max 3 attempts before ABORTED)
- `_concurrency_sem`: Semaphore for concurrency control

---

## Properties

### `available_capacity`

```python
@property
def available_capacity(self) -> int:
    """Current capacity available for processing."""
```

**Behavior:**

- Starts at `queue_capacity` when processor initialized or after batch completes
- Decreases by 1 for each event processed
- Resets to `queue_capacity` after `process()` completes a batch
- Read/write property (can be manually adjusted if needed)

---

### `execution_mode`

```python
@property
def execution_mode(self) -> bool:
    """Whether processor is actively executing events."""
```

**Behavior:**

- `True` when `execute()` loop is running
- `False` when stopped or not started
- Set automatically by `execute()` and `stop()`
- Read/write property

---

## Methods

### Queue Management

#### `enqueue()`

```python
async def enqueue(self, event_id: UUID, priority: float | None = None) -> None:
    """Add event UUID to priority queue.

    Args:
        event_id: UUID of event in pile
        priority: Priority value (lower = higher priority).
                 If None, fetches event from pile and uses created_at timestamp.

    Raises:
        QueueFullError: If queue size exceeds max_queue_size
        ValueError: If priority is NaN or Inf
    """
```

**Behavior:**

- **Default Priority**: If `priority=None`, uses `event.created_at.timestamp()` (earlier events have lower priority value → processed first)
- **Queue Size Check**: Raises `QueueFullError` if `queue.qsize() >= max_queue_size`
- **Priority Validation**: Rejects NaN or Inf values (prevents heap corruption)
- **Storage**: Queue stores `(priority, event_id)` tuples, NOT event objects (memory efficient)

**Example:**

```python
# Default priority (creation time)
await processor.enqueue(event.id)

# Custom priority (urgent task)
await processor.enqueue(urgent_event.id, priority=0.0)
```

---

#### `dequeue()`

```python
async def dequeue(self) -> Event:
    """Retrieve highest priority event from queue.

    Returns:
        Event instance fetched from pile (lowest priority value first)
    """
```

**Behavior:**

- **Priority Order**: Returns event with lowest priority value first
- **Pile Lookup**: Fetches event from `pile[event_id]` (O(1) UUID lookup)
- **Blocking**: Awaits if queue is empty

**Example:**

```python
event = await processor.dequeue()
# event is now the highest priority item
```

---

### Execution Control

#### `process()`

```python
async def process(self) -> None:
    """Dequeue and process events up to available capacity."""
```

**Behavior:**

- **Batch Processing**: Processes up to `available_capacity` events
- **Permission Checks**: Calls `request_permission(**event.request)` before processing
- **Denial Handling**: Tracks denials in `_denial_counts` (max 3 attempts → ABORTED)
- **Exponential Backoff**: Requeues denied events with priority adjustment (1st retry +1s, 2nd retry +2s)
- **Missing Events**: Gracefully handles events removed from pile (cleanup race)
- **Progression Updates**: Calls `executor._update_progression()` for state transitions (PROCESSING → COMPLETED/FAILED)
- **Streaming Support**: Detects `event.streaming` flag and consumes via `event.stream()` async generator
- **Concurrency**: Runs event executions concurrently up to `concurrency_limit` (via semaphore)
- **Capacity Reset**: Resets `available_capacity = queue_capacity` after batch completes (only if events were processed)

**Thread Safety**: Uses task group for concurrent execution, safe for multiple `process()` calls.

**Example:**

```python
# Manual batch processing
await processor.process()  # Process up to queue_capacity events
```

---

#### `execute()`

```python
async def execute(self) -> None:
    """Continuously process events until stop() called."""
```

**Behavior:**

- **Continuous Loop**: Calls `process()` repeatedly with `capacity_refresh_time` delay
- **Execution Mode**: Sets `execution_mode = True` at start, `False` at stop
- **Lifecycle**: Calls `start()` at beginning, checks `is_stopped()` each iteration
- **Graceful Shutdown**: Exits when `stop()` called

**Typical Usage:**

```python
# Background processing
import anyio

async with anyio.create_task_group() as tg:
    tg.start_soon(processor.execute)  # Run in background

    # Do other work...

    await processor.stop()  # Signal shutdown
```

---

#### `start()`

```python
async def start(self) -> None:
    """Clear stop signal, allowing processing to resume."""
```

**Behavior:**

- Resets `_stop_event` if currently stopped
- Allows `execute()` loop to continue
- Idempotent (safe to call multiple times)

---

#### `stop()`

```python
async def stop(self) -> None:
    """Signal processor to stop processing events.

    Clears denial tracking to prevent memory leaks across stop/start cycles.
    """
```

**Behavior:**

- Sets `_stop_event` to signal `execute()` loop to exit
- Clears `_denial_counts` (prevents memory leaks across stop/start cycles)
- Non-blocking (does not wait for current batch to finish)

---

#### `join()`

```python
async def join(self) -> None:
    """Block until queue is empty."""
```

**Behavior:**

- Polls `queue.empty()` every 0.1 seconds
- Useful for waiting until all queued events are processed
- Does NOT wait for in-flight events to complete

**Example:**

```python
# Queue many events
for event in events:
    await processor.enqueue(event.id)

# Wait for queue to drain
await processor.join()
```

---

#### `is_stopped()`

```python
def is_stopped(self) -> bool:
    """Check if processor is stopped."""
```

**Behavior:**

- Returns `True` if `stop()` was called
- Returns `False` if running or never started

---

### Permission Control

#### `request_permission()`

```python
async def request_permission(self, **kwargs: Any) -> bool:
    """Override for custom checks (rate limits, permissions, quotas)."""
```

**Default Behavior:** Always returns `True` (all events permitted).

**Override for:**

- **Rate limiting**: Check API quota, enforce requests/second limits
- **Authorization**: Verify user permissions, check access tokens
- **Resource control**: Ensure memory/CPU below thresholds
- **Business logic**: Custom conditions (e.g., only process during business hours)

**Parameters:**

- `**kwargs`: Receives `event.request` dict (set by your Event subclass)

**Return:**

- `True`: Permission granted, event will be processed
- `False`: Permission denied, event will be requeued (max 3 attempts → ABORTED)

**Example:**

```python
class RateLimitedProcessor(Processor):
    async def request_permission(self, **kwargs: Any) -> bool:
        api_key = kwargs.get("api_key")
        # Check quota for this API key
        remaining = await quota_manager.check(api_key)
        return remaining > 0
```

---

### Factory Method

#### `create()` (async classmethod)

```python
@classmethod
async def create(
    cls,
    queue_capacity: int,
    capacity_refresh_time: float,
    pile: Pile[Event],
    executor: Executor | None = None,
    concurrency_limit: int = 100,
    max_queue_size: int = 1000,
    max_denial_tracking: int = 10000,
) -> Self:
    """Asynchronously construct new Processor.

    Args:
        (same as __init__)

    Returns:
        New processor instance
    """
```

**Behavior:**

- Async factory method for subclasses that need async initialization
- Currently just calls `__init__` (placeholder for future async setup)

**Example:**

```python
processor = await MyProcessor.create(
    queue_capacity=50,
    capacity_refresh_time=1.0,
    pile=executor.states.items,
    executor=executor,
)
```

---

## Usage Patterns

### Basic Custom Processor

```python
from lionherd_core.base import Event, Processor, Executor

class MyEvent(Event):
    async def _invoke(self) -> str:
        # Your event logic
        return "result"

class MyProcessor(Processor):
    event_type = MyEvent  # Required ClassVar

# Executor creates processor automatically
class MyExecutor(Executor):
    processor_type = MyProcessor

# Usage
async def main():
    executor = MyExecutor(
        processor_config={
            "queue_capacity": 50,
            "capacity_refresh_time": 1.0,
            "concurrency_limit": 100,
        }
    )

    # Start background processing
    await executor.start()
```

---

### Custom Permission Checks (Rate Limiting)

```python
class RateLimitedProcessor(Processor):
    def __init__(self, *args, requests_per_second: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_requests = requests_per_second
        self.request_count = 0
        self.window_start = time.time()

    async def request_permission(self, **kwargs: Any) -> bool:
        # Reset window every second
        now = time.time()
        if now - self.window_start >= 1.0:
            self.request_count = 0
            self.window_start = now

        # Check quota
        if self.request_count >= self.max_requests:
            return False  # Deny, will requeue with backoff

        self.request_count += 1
        return True
```

---

### Manual vs Background Processing

```python
# Background processing (continuous)
async with anyio.create_task_group() as tg:
    tg.start_soon(processor.execute)

    # Queue events
    for event in events:
        await processor.enqueue(event.id)

    # Stop when done
    await processor.stop()

# Manual processing (one batch)
for event in events:
    await processor.enqueue(event.id)

await processor.process()  # Process single batch
```

---

### Lifecycle Control

```python
# Start processor
await processor.start()

# Queue events
await processor.enqueue(event1.id)
await processor.enqueue(event2.id, priority=0.5)  # Higher priority

# Stop gracefully
await processor.stop()

# Wait for queue to drain before stopping
await processor.join()
await processor.stop()

# Check status
if processor.is_stopped():
    print("Processor stopped")
```

---

## Design Patterns

### Capacity Management

**Purpose:** Control batch size and refresh rate to balance throughput vs latency.

**Pattern:**

- **High Throughput**: Large `queue_capacity` (e.g., 1000), longer `capacity_refresh_time` (e.g., 10s)
- **Low Latency**: Small `queue_capacity` (e.g., 10), short `capacity_refresh_time` (e.g., 0.1s)
- **Balanced**: Medium `queue_capacity` (e.g., 50), moderate `capacity_refresh_time` (e.g., 1.0s)

**Trade-offs:**

- Larger batches → better throughput, higher latency
- Smaller batches → lower latency, more CPU overhead (more `process()` calls)

**Recommendation:** Start with `queue_capacity=50`, `capacity_refresh_time=1.0` and tune based on metrics.

---

### Priority Ordering

**Default Behavior:** Events processed in creation order (`created_at.timestamp()`).

**Custom Priority:**

```python
# Urgent events first
await processor.enqueue(urgent_event.id, priority=0.0)

# Normal events (default: creation time)
await processor.enqueue(normal_event.id)

# Low priority events last
await processor.enqueue(low_priority_event.id, priority=9999.0)
```

**Priority Values:**

- **Lower = Higher Priority**: `priority=0.0` processed before `priority=1.0`
- **Finite Numbers Only**: NaN and Inf raise `ValueError`
- **Float Precision**: Use timestamp or discrete levels (avoid too many distinct values)

---

### Concurrency Control

**Purpose:** Limit concurrent event executions to prevent resource exhaustion.

**Pattern:**

- **CPU-bound**: Set `concurrency_limit` to number of CPU cores (e.g., 8)
- **I/O-bound**: Set `concurrency_limit` to max concurrent connections (e.g., 100)
- **Memory-constrained**: Set `concurrency_limit` to prevent OOM (e.g., 10)

**Implementation:** Uses `anyio.Semaphore` internally. Each `event.invoke()` acquires semaphore before execution.

**Default:** `concurrency_limit=100` (safe for most I/O-bound workloads).

---

### Streaming Support

**Purpose:** Handle events that produce async generators (e.g., LLM streaming responses).

**Pattern:**

```python
class StreamingEvent(Event):
    streaming = True  # Mark as streaming

    async def _invoke(self) -> AsyncGenerator[str, None]:
        for i in range(10):
            yield f"chunk {i}"
```

**Processor Behavior:**

- Detects `event.streaming == True`
- Consumes entire generator via `async for _ in event.stream()`
- Updates progression after stream completes or fails
- Runs concurrently up to `concurrency_limit`

---

## Configuration Guide

### Parameter Tuning Recommendations

| Use Case | `queue_capacity` | `capacity_refresh_time` | `concurrency_limit` |
|----------|------------------|-------------------------|---------------------|
| **High-volume ETL** | 500-1000 | 5-10s | 50-100 |
| **API rate limiting** | 50-100 | 1s | 10-50 |
| **Real-time notifications** | 10-50 | 0.1-0.5s | 100-200 |
| **Resource-constrained** | 10-20 | 1s | 5-10 |
| **CPU-intensive tasks** | 20-50 | 1-5s | num_cpus |

---

### Performance Trade-offs

| Parameter | Increase → Effect | Decrease → Effect |
|-----------|-------------------|-------------------|
| `queue_capacity` | + throughput, + latency, + memory | - latency, - throughput |
| `capacity_refresh_time` | + batch efficiency, + latency | - latency, + CPU overhead |
| `concurrency_limit` | + throughput (I/O), risk OOM | - memory, - throughput |
| `max_queue_size` | + burst capacity, + memory | - memory, risk QueueFullError |

---

### Common Pitfalls

#### 1. Queue Size vs Capacity Confusion

**Issue:** Confusing `queue_capacity` (batch size) with `max_queue_size` (total queue limit).

**Solution:**

- `queue_capacity`: Max events processed *per batch cycle*
- `max_queue_size`: Max events *waiting in queue* before rejection

**Rule of Thumb:** `max_queue_size >= queue_capacity * 10` (buffer for bursts).

---

#### 2. Hot Loop with Low Refresh Time

**Issue:** Setting `capacity_refresh_time < 0.01s` causes CPU hot loop.

**Solution:**

- Minimum: `0.01s` (enforced by validation)
- Recommended: `≥ 0.1s` for most use cases

---

#### 3. Unbounded Denial Tracking

**Issue:** `_denial_counts` grows unbounded if many events are repeatedly denied.

**Solution:**

- Set `max_denial_tracking` to reasonable limit (default: 10000)
- FIFO eviction when limit exceeded (oldest entry removed)
- Cleared on `stop()` to prevent memory leaks

---

#### 4. Missing Executor Reference

**Issue:** `executor=None` prevents automatic progression updates.

**Solution:**

- Always pass `executor=self` when creating processor from Executor
- Manual progression management if `executor=None` (advanced use case)

---

#### 5. Infinite Permission Denials

**Issue:** `request_permission()` always returns `False` → events stuck in retry loop.

**Solution:**

- Processor aborts events after 3 denials (sets status to ABORTED)
- Override `request_permission()` with fallback logic (e.g., allow after backoff period)

---

## See Also

- **[Executor](executor.md)**: Flow-based state tracking for events (creates and manages Processor)
- **[Event](event.md)**: Base event class with `invoke()` and `stream()` methods
- **[Flow](flow.md)**: Dual-pile state machine (used by Executor for event storage)
- **[EventStatus](event.md#eventstatus)**: Enum for event lifecycle states
- **[Processor/Executor Integration](../../user_guide/processor_executor.md)**: Integration patterns and shared concepts
- **[Processor/Executor Examples](../../../notebooks/references/processor_executor.ipynb)**: Copy-paste ready code examples

---

## Examples

### Complete Workflow

```python
import anyio
from lionherd_core.base import Event, Processor, Executor, EventStatus

# 1. Define custom Event
class DataProcessingEvent(Event):
    async def _invoke(self) -> dict:
        # Simulate data processing
        await anyio.sleep(0.1)
        return {"status": "processed", "data": self.request.get("data")}

# 2. Define custom Processor with rate limiting
class DataProcessor(Processor):
    event_type = DataProcessingEvent

    def __init__(self, *args, max_per_second: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_per_second = max_per_second
        self.request_count = 0
        self.window_start = anyio.current_time()

    async def request_permission(self, **kwargs) -> bool:
        # Rate limiting logic
        now = anyio.current_time()
        if now - self.window_start >= 1.0:
            self.request_count = 0
            self.window_start = now

        if self.request_count >= self.max_per_second:
            return False  # Deny, will requeue

        self.request_count += 1
        return True

# 3. Define Executor
class DataExecutor(Executor):
    processor_type = DataProcessor

# 4-9. Usage
async def main():
    # 4. Create executor with processor config
    executor = DataExecutor(
        processor_config={
            "queue_capacity": 50,
            "capacity_refresh_time": 1.0,
            "concurrency_limit": 100,
            "max_per_second": 10,  # Custom param
        }
    )

    # 5. Start background processing
    await executor.start()

    async with anyio.create_task_group() as tg:
        # Run processor in background
        tg.start_soon(executor.processor.execute)

        # 6. Queue events with different priorities
        urgent = DataProcessingEvent(request={"data": "urgent"})
        await executor.append(urgent, priority=0.0)  # Highest priority

        normal = DataProcessingEvent(request={"data": "normal"})
        await executor.append(normal)  # Default priority (creation time)

        # 7. Wait for completion
        await anyio.sleep(2.0)

        # 8. Check results
        print(executor.inspect_state())
        # Output:
        # Executor State (executor_states):
        #   pending: 0 events
        #   processing: 0 events
        #   completed: 2 events
        #   failed: 0 events
        #   cancelled: 0 events
        #   skipped: 0 events
        #   aborted: 0 events

        # 9. Stop processor
        await executor.stop()
```

---

### Manual Batch Processing

```python
# Create processor without executor (manual mode)
from lionherd_core.base import Pile

pile = Pile(item_type=DataProcessingEvent)
processor = DataProcessor(
    queue_capacity=10,
    capacity_refresh_time=1.0,
    pile=pile,
    executor=None,  # No automatic progression updates
)

# Add events to pile and queue
event1 = DataProcessingEvent(request={"data": "test1"})
pile.add(event1)
await processor.enqueue(event1.id)

event2 = DataProcessingEvent(request={"data": "test2"})
pile.add(event2)
await processor.enqueue(event2.id, priority=0.5)  # Higher priority

# Process single batch
await processor.process()

# Check completion
for event in [event1, event2]:
    print(f"{event.id}: {event.execution.status}")
    # Output: completed
```

---

### Streaming Events

```python
from typing import AsyncGenerator

class StreamingEvent(Event):
    streaming = True

    async def _invoke(self) -> AsyncGenerator[str, None]:
        for i in range(5):
            await anyio.sleep(0.1)
            yield f"chunk {i}"

class StreamProcessor(Processor):
    event_type = StreamingEvent

class StreamExecutor(Executor):
    processor_type = StreamProcessor

# Usage
async def main():
    executor = StreamExecutor(processor_config={
        "queue_capacity": 10,
        "capacity_refresh_time": 0.5,
    })

    await executor.start()

    # Queue streaming event
    stream_event = StreamingEvent()
    await executor.append(stream_event)

    # Processor will consume entire stream automatically
    async with anyio.create_task_group() as tg:
        tg.start_soon(executor.processor.execute)
        await anyio.sleep(1.0)
        await executor.stop()

    print(stream_event.execution.status)  # completed
```

---

### Priority Queue Behavior

```python
# Demonstrate priority ordering
processor = DataProcessor(
    queue_capacity=100,
    capacity_refresh_time=1.0,
    pile=pile,
)

# Queue events with explicit priorities
critical = DataProcessingEvent(request={"level": "critical"})
pile.add(critical)
await processor.enqueue(critical.id, priority=0.0)  # Processed first

high = DataProcessingEvent(request={"level": "high"})
pile.add(high)
await processor.enqueue(high.id, priority=10.0)

low = DataProcessingEvent(request={"level": "low"})
pile.add(low)
await processor.enqueue(low.id, priority=100.0)  # Processed last

# Process batch
await processor.process()

# Order: critical → high → low
```

---

## Copyright © 2025 HaiyangLi (Ocean) - Apache 2.0 License**

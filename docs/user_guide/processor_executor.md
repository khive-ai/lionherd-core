# Processor and Executor: Event Processing Architecture

**Purpose**: Comprehensive guide to the Processor/Executor pattern - background event processing with Flow-based state tracking.

**Audience**: Developers building event-driven systems with lionherd-core.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Integration Architecture](#integration-architecture)
3. [Core Patterns](#core-patterns)
4. [Complete Workflows](#complete-workflows)
5. [Advanced Topics](#advanced-topics)
6. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is the Processor/Executor Pattern?

The Processor/Executor pattern provides a complete event processing architecture with two complementary components:

- **Executor**: Flow-based state tracker with O(1) status queries
- **Processor**: Background event processor with priority queue and capacity control

**Key Benefits**:

- **Separation of Concerns**: State tracking (Executor) vs background processing (Processor)
- **O(1) Performance**: Status queries via Flow progressions (not O(n) pile scans)
- **Capacity Control**: Batch processing with configurable limits
- **Priority Support**: Custom priority-based event ordering
- **Type Safety**: Event subclasses with compile-time checks
- **Audit Trail**: Full state serialization via Flow

### When to Use This Pattern

**Use Processor/Executor when**:

- Background event processing required (not blocking user code)
- Status tracking needed (pending → processing → completed)
- Priority-based ordering important
- Rate limiting or permission checks needed
- Audit trail or state serialization required

**Don't use when**:

- Simple sequential processing sufficient (`await event.invoke()`)
- No status tracking needed
- Synchronous execution preferred
- Minimal latency critical (batch processing adds delay)

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│ Executor: State Management                                 │
│                                                             │
│  states: Flow[Event, Progression]                          │
│    ├─ items: Pile[Event] (O(1) storage) ─────┐             │
│    └─ progressions: Pile[Progression]         │             │
│        ├─ pending: [uuid1, uuid2]             │             │
│        ├─ processing: [uuid3]                 │             │
│        ├─ completed: [uuid4, uuid5]           │             │
│        └─ failed: []                          │             │
│                                               │             │
│  ┌────────────────────────────────────────────┼──────────┐  │
│  │ Processor: Background Processing          │          │  │
│  │                                            │          │  │
│  │  pile: Pile[Event] ────────────────────────┘          │  │
│  │  executor: Executor ──────────────┐                   │  │
│  │                                    │                   │  │
│  │  queue: PriorityQueue              │                   │  │
│  │    └─ [(1.0, uuid1), (2.0, uuid2)] │                   │  │
│  │                                    │                   │  │
│  │  execute() loop:                   │                   │  │
│  │    1. Dequeue (priority, uuid)     │                   │  │
│  │    2. Fetch: pile[uuid]            │                   │  │
│  │    3. Check permission             │                   │  │
│  │    4. Invoke event                 │                   │  │
│  │    5. Update progression ──────────┘                   │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Executor owns event storage (Flow.items) and passes it to Processor as a pile reference. Processor queue stores lightweight (priority, UUID) tuples. Status updates flow back via executor reference.

---

## Integration Architecture

### Component Responsibilities

**Executor**:

- Owns event storage (Flow.items)
- Creates Flow progressions (1:1 with EventStatus)
- Manages event lifecycle (append → progression updates)
- Provides O(1) status queries
- Creates and coordinates Processor

**Processor**:

- Receives pile reference (executor's Flow.items)
- Manages priority queue (stores UUIDs only)
- Processes events in background (async loop)
- Enforces capacity limits (batch processing)
- Controls concurrency (semaphore)
- Notifies executor of status changes

### Bidirectional References

```python
# Executor creates Processor with bidirectional refs
class MyExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = MyProcessor

executor = MyExecutor()
await executor.start()  # Creates processor internally

# Result:
executor.processor        # Processor instance
executor.processor.pile   # Same as executor.states.items
executor.processor.executor  # Same as executor
```

**Why Bidirectional?**

- Processor → Executor: For progression updates (`await executor._update_progression()`)
- Executor → Processor: For lifecycle control (`await processor.start()`)
- Optional: Processor can exist standalone (executor=None)

### Data Flow

**Event Submission**:

```text
User code → executor.append(event, priority=1.0)
  ├─ Add to Flow: states.add_item(event, progressions="pending")
  └─ Enqueue: processor.enqueue(event.id, priority=1.0)
```

**Background Processing**:

```text
Processor.execute() loop (every capacity_refresh_time):
  ├─ Dequeue: (priority, uuid) = queue.get()
  ├─ Fetch: event = pile[uuid]
  ├─ Permission: await request_permission(**event.request)
  ├─ Update: executor._update_progression(event, PROCESSING)
  ├─ Invoke: await event.invoke()
  └─ Update: executor._update_progression(event)  # Final status
```

**Status Queries**:

```text
User code → executor.get_events_by_status(EventStatus.COMPLETED)
  ├─ Lookup progression: states.get_progression("completed")
  ├─ Get UUIDs: progression.order
  └─ Fetch events: [states.items[uid] for uid in progression]
```

---

## Core Patterns

This section documents the 12 shared architectural patterns between Processor and Executor. These patterns are the **canonical reference** - all other documentation links here.

### Pattern 1: UUID-Based Event References

**What**: Both classes store/pass UUIDs rather than full Event objects.

**Processor Usage**:

```python
# Queue stores lightweight tuples
self.queue: PriorityQueue[tuple[float, UUID]] = PriorityQueue()

# Enqueue UUID, not event
await self.queue.put((priority, event_id))

# Fetch from pile when needed
event = self.pile[event_id]
```

**Executor Usage**:

```python
# Progressions store UUIDs
progression.order = [uuid1, uuid2, uuid3]

# Fetch events on demand
events = [self.states.items[uid] for uid in progression]
```

**Benefits**:

- **Lightweight**: UUID = 16 bytes vs full Event object (hundreds of bytes)
- **Single Source of Truth**: Events live in Flow.items only
- **No Synchronization**: UUIDs are immutable references
- **Memory Efficient**: Queue of 10,000 events = ~160KB (UUIDs) vs ~2-5MB (objects)

**Example**:

```python
# Event created once, stored in Flow
event = MyEvent()
executor.states.add_item(event, progressions="pending")

# Processor queue stores UUID only
await processor.enqueue(event.id, priority=1.0)

# Fetch when needed (O(1) from pile)
next_event = processor.pile[event.id]
```

**API Reference**: [Processor.enqueue()](../api/base/processor.md#enqueue), [Executor.get_events_by_status()](../api/base/executor.md#get_events_by_status)

---

### Pattern 2: Flow-Based State Tracking

**What**: Executor uses Flow for storage, Processor receives Flow.items as pile reference.

**Flow Structure**:

```python
executor.states: Flow[Event, Progression]
  ├─ items: Pile[Event]           # Event storage (O(1) lookup)
  └─ progressions: Pile[Progression]
      ├─ pending: Progression(order=[uuid1, uuid2])
      ├─ processing: Progression(order=[uuid3])
      ├─ completed: Progression(order=[uuid4, uuid5])
      ├─ failed: Progression(order=[])
      └─ aborted: Progression(order=[])
```

**Processor Integration**:

```python
# Executor creates processor with pile reference
processor = await MyProcessor.create(
    pile=executor.states.items,  # Flow.items Pile
    executor=executor,
    ...
)

# Processor reads from pile, writes via executor
event = processor.pile[event_id]  # Read
await processor.executor._update_progression(event)  # Write
```

**Why Flow, Not Just Pile?**

- **Multiple Progressions**: Track status + custom workflows (e.g., "stage1", "stage2")
- **O(1) Queries**: Status lookup without scanning all events
- **Audit Trail**: Serialize progressions for debugging
- **Type Safety**: Flow enforces item_type=Event

**Example**:

```python
# Create executor (creates Flow internally)
executor = MyExecutor(processor_config={"queue_capacity": 100})

# Flow automatically created with 5 progressions (one per EventStatus)
assert len(executor.states.progressions) == 5
assert executor.states.get_progression("pending").name == "pending"

# Processor shares same pile
await executor.start()
assert executor.processor.pile is executor.states.items  # Same object
```

**API Reference**: [Flow](../api/base/flow.md), [Pile](../api/base/pile.md)

---

### Pattern 3: Event Lifecycle (Status Transitions)

**What**: Both classes coordinate around EventStatus transitions.

**Lifecycle**:

```text
PENDING → PROCESSING → COMPLETED
                     → FAILED
                     → ABORTED (permission denied 3x)
```

**Status Definitions** (from `EventStatus` enum):

- `PENDING`: Event created, not yet processed
- `PROCESSING`: Currently executing
- `COMPLETED`: Executed successfully
- `FAILED`: Execution raised exception
- `ABORTED`: Permission denied 3 times

**Processor Role**:

```python
# Before processing
event.status  # EventStatus.PENDING

# Update to PROCESSING
await executor._update_progression(event, EventStatus.PROCESSING)

# Event.invoke() sets final status
await event.invoke()  # Sets COMPLETED or FAILED

# Update progression to match
await executor._update_progression(event)  # Uses event.execution.status
```

**Executor Role**:

```python
# Maintains 1:1 mapping (EventStatus → Progression)
for status in EventStatus:
    self.states.add_progression(Progression(name=status.value))

# Moves UUIDs between progressions atomically
async def _update_progression(self, event, force_status=None):
    target_status = force_status or event.execution.status

    # Remove from all progressions
    for prog in self.states.progressions:
        if event.id in prog:
            prog.remove(event.id)

    # Add to target progression
    status_prog = self.states.get_progression(target_status.value)
    status_prog.append(event.id)
```

**Invariants**:

- Event exists in **exactly one** status progression (single-ownership)
- Progression name matches EventStatus.value (1:1 mapping)
- Status transitions are atomic (via Flow.progressions async lock)

**Example**:

```python
# Initial state
event = MyEvent()
executor.states.add_item(event, progressions="pending")
assert event.id in executor.states.get_progression("pending")

# Start processing
await executor.start()
await executor.forward()  # Process one batch

# Event moved to processing
assert event.id in executor.states.get_progression("processing")
assert event.id not in executor.states.get_progression("pending")

# After completion
await event.invoke()  # Sets COMPLETED
await executor._update_progression(event)
assert event.id in executor.states.get_progression("completed")
```

**API Reference**: [EventStatus](../api/base/event.md#eventstatus), [Executor._update_progression()](../api/base/executor.md#_update_progression)

---

### Pattern 4: Priority Queue Mechanics

**What**: Processor uses priority queue, Executor provides priority parameter.

**Processor Implementation**:

```python
# Queue stores (priority, uuid) tuples
self.queue: PriorityQueue[tuple[float, UUID]] = PriorityQueue()

# Lower priority values processed first
await self.queue.put((1.0, event1.id))  # High priority
await self.queue.put((10.0, event2.id)) # Low priority

# Dequeue in order
priority, event_id = await self.queue.get()  # Returns (1.0, event1.id)
```

**Executor API**:

```python
# Custom priority
await executor.append(event, priority=1.0)

# Default priority = event.created_at.timestamp()
await executor.append(event)  # Earlier events processed first
```

**Default Behavior**:

- No priority provided → use `event.created_at.timestamp()`
- Earlier events have lower timestamp → processed first (FIFO)
- Custom priorities override created_at

**Priority Strategy Examples**:

```python
# High priority for errors
if event.error_count > 0:
    priority = 0.0  # Process immediately
else:
    priority = event.created_at.timestamp()

# Business priority
priority_map = {"critical": 1.0, "high": 5.0, "normal": 10.0, "low": 20.0}
await executor.append(event, priority=priority_map[event.priority_level])

# Time-sensitive
deadline = event.deadline.timestamp()
now = datetime.now().timestamp()
priority = max(0, deadline - now)  # Earlier deadlines = lower priority value
```

**Validation**:

```python
# Priority must be finite (prevents heap corruption)
await executor.append(event, priority=float('inf'))  # Raises ValueError
await executor.append(event, priority=float('nan'))  # Raises ValueError
```

**API Reference**: [Processor.enqueue()](../api/base/processor.md#enqueue), [Executor.append()](../api/base/executor.md#append)

---

### Pattern 5: Capacity-Limited Batch Processing

**What**: Processor processes events in batches with configurable capacity.

**Configuration**:

```python
processor = await MyProcessor.create(
    queue_capacity=50,           # Max events per batch
    capacity_refresh_time=1.0,   # Seconds between batches
    ...
)
```

**Batch Cycle**:

```python
# execute() loop
while not stopped:
    await process()  # Process up to queue_capacity events
    await sleep(capacity_refresh_time)  # Wait before next batch

# process() implementation
events_processed = 0
while available_capacity > 0 and not queue.empty():
    event = await dequeue()
    if permission_granted:
        await event.invoke()
        events_processed += 1
        available_capacity -= 1

# Reset capacity after batch
if events_processed > 0:
    available_capacity = queue_capacity
```

**Capacity Management**:

- Starts at `queue_capacity`
- Decrements by 1 for each processed event
- Resets to `queue_capacity` after batch
- Permission denials **don't consume capacity** (allows retries)

**Trade-offs**:

| Configuration | Latency | Throughput | CPU Usage |
|---------------|---------|------------|-----------|
| capacity=10, refresh=0.1s | Low (100ms) | Low (100 events/sec max) | High (10 batches/sec) |
| capacity=100, refresh=1.0s | Medium (1s) | Medium (100 events/sec max) | Medium (1 batch/sec) |
| capacity=500, refresh=5.0s | High (5s) | High (100 events/sec max) | Low (0.2 batches/sec) |

**Tuning Guidelines**:

- **Low latency**: Small capacity (10-50), short refresh (0.1-0.5s)
- **High throughput**: Large capacity (100-500), long refresh (1-5s)
- **Balanced**: capacity=100, refresh=1.0s (default recommended)

**Example**:

```python
# Submit 200 events
for i in range(200):
    await executor.append(MyEvent())

# Configure processor for low latency
executor.processor_config = {
    "queue_capacity": 20,          # Small batches
    "capacity_refresh_time": 0.5,  # Fast refresh
}

await executor.start()

# Processing:
# Batch 1 (t=0.0s): Process 20 events
# Batch 2 (t=0.5s): Process 20 events
# ...
# Batch 10 (t=4.5s): Process 20 events
# Total time: ~5s
```

**Validation**:

```python
# Bounds checking in __init__
if queue_capacity < 1:
    raise ValueError("Queue capacity must be > 0")
if capacity_refresh_time < 0.01:
    raise ValueError("Refresh time must be >= 0.01s (prevent CPU hot loop)")
```

**API Reference**: [Processor.**init**()](../api/base/processor.md#__init__), [Processor.process()](../api/base/processor.md#process)

---

### Pattern 6: Concurrency Control (Semaphore)

**What**: Processor limits concurrent event executions via semaphore.

**Implementation**:

```python
# In Processor.__init__
self._concurrency_sem = Semaphore(concurrency_limit)

# In process()
async def _with_semaphore(self, coro):
    async with self._concurrency_sem:
        return await coro

# Usage
tg.start_soon(self._with_semaphore, invoke_and_update(event))
```

**Purpose**:

- Prevents resource exhaustion (e.g., 10,000 concurrent HTTP requests)
- Limits simultaneous async tasks
- Per-processor isolation (multiple executors = independent limits)

**Configuration**:

```python
# Default: 100 (safe for most use cases)
processor = await MyProcessor.create(concurrency_limit=100)

# Low concurrency (rate limiting)
processor = await MyProcessor.create(concurrency_limit=10)

# High concurrency (CPU-bound tasks)
processor = await MyProcessor.create(concurrency_limit=500)
```

**Concurrency vs Capacity**:

- **Capacity**: Max events per batch (sequential processing limit)
- **Concurrency**: Max simultaneous executions (parallel processing limit)

**Example**:

```python
# Capacity=100, Concurrency=10
processor = await MyProcessor.create(
    queue_capacity=100,        # Process 100 events per batch
    concurrency_limit=10,      # But only 10 at a time
)

# Batch behavior:
# - Dequeue 100 events
# - Process in waves of 10 concurrent tasks
# - Wave 1: events 1-10 (concurrent)
# - Wave 2: events 11-20 (concurrent)
# - ...
# - Wave 10: events 91-100 (concurrent)
```

**When to Tune**:

- **Lower limit**: HTTP requests, database connections, file handles
- **Higher limit**: Pure CPU tasks, in-memory processing
- **Default (100)**: Balanced for mixed workloads

**API Reference**: [Processor.**init**()](../api/base/processor.md#__init__)

---

### Pattern 7: Event Lifecycle Management (Start/Stop)

**What**: Coordinated lifecycle control across Executor and Processor.

**Lifecycle Phases**:

```text
Creation → Started → Running → Stopped → Cleanup
```

**Executor API**:

```python
# Phase 1: Creation
executor = MyExecutor(processor_config={...})
assert executor.processor is None  # Processor not created yet

# Phase 2: Start (creates processor + backfills events)
await executor.start()
assert executor.processor is not None
assert not executor.processor.is_stopped()

# Phase 3: Stop (signals processor to stop)
await executor.stop()
assert executor.processor.is_stopped()
```

**Processor API**:

```python
# Start: Clear stop signal
await processor.start()
assert not processor.is_stopped()

# Stop: Set stop event, clear denial tracking
await processor.stop()
assert processor.is_stopped()
assert len(processor._denial_counts) == 0  # Memory leak prevention
```

**Lifecycle Coordination**:

```python
# Executor.start() implementation
async def start(self):
    if not self.processor:
        await self._create_processor()  # Create processor
        for event in self.pending_events:
            await self.processor.enqueue(event.id)  # Backfill pending events
    await self.processor.start()  # Clear stop signal

# Executor.stop() implementation
async def stop(self):
    if self.processor:
        await self.processor.stop()  # Signal stop
```

**Graceful Shutdown**:

```python
# Processor.execute() checks stop signal
async def execute(self):
    self.execution_mode = True
    await self.start()

    while not self.is_stopped():
        await self.process()  # Finish current batch
        await sleep(self.capacity_refresh_time)

    self.execution_mode = False  # Clean exit
```

**Example**:

```python
# Create executor with events
executor = MyExecutor()
for i in range(100):
    await executor.append(MyEvent())

# Start background processing
await executor.start()
print(f"Processing {len(executor.pending_events)} events")

# Run for 10 seconds
await asyncio.sleep(10)

# Graceful shutdown
await executor.stop()
await executor.processor.join()  # Wait for queue to empty

print(f"Completed: {len(executor.completed_events)}")
print(f"Failed: {len(executor.failed_events)}")
```

**Memory Safety**:

```python
# stop() clears denial tracking
await processor.stop()
assert len(processor._denial_counts) == 0

# Prevents memory leak across stop/start cycles
await processor.start()  # Fresh state
```

**API Reference**: [Executor.start()](../api/base/executor.md#start), [Executor.stop()](../api/base/executor.md#stop), [Processor.execute()](../api/base/processor.md#execute)

---

### Pattern 8: O(1) Status Queries

**What**: Executor provides constant-time status lookups via Flow progressions.

**Architecture**:

```python
# Progressions map 1:1 with EventStatus
executor.states.progressions = [
    Progression(name="pending", order=[uuid1, uuid2]),
    Progression(name="processing", order=[uuid3]),
    Progression(name="completed", order=[uuid4, uuid5]),
    Progression(name="failed", order=[]),
    Progression(name="aborted", order=[]),
]
```

**Query API**:

```python
# O(1) status query (progression name lookup + UUID list fetch)
completed = executor.get_events_by_status(EventStatus.COMPLETED)

# Implementation
def get_events_by_status(self, status):
    prog = self.states.get_progression(status.value)  # O(1) dict lookup
    return [self.states.items[uid] for uid in prog]    # O(n) where n = events with status
```

**Comparison: O(1) vs O(n)**:

```python
# ❌ SLOW: O(n) pile scan (n = all events)
completed = [e for e in executor.states.items if e.status == EventStatus.COMPLETED]

# ✅ FAST: O(1) progression lookup + O(m) fetch (m = completed events only)
completed = executor.get_events_by_status(EventStatus.COMPLETED)
```

**Performance Analysis**:

| Total Events | Completed | O(n) Scan Time | O(1) Query Time | Speedup |
|--------------|-----------|----------------|-----------------|---------|
| 1,000        | 100       | ~0.5ms         | ~0.05ms         | 10x     |
| 10,000       | 1,000     | ~5ms           | ~0.5ms          | 10x     |
| 100,000      | 10,000    | ~50ms          | ~5ms            | 10x     |

**Overhead**: ~8% memory overhead (5 progressions × 2% each)

**Convenience Properties**:

```python
# Direct access to common statuses
executor.completed_events  # get_events_by_status(COMPLETED)
executor.pending_events    # get_events_by_status(PENDING)
executor.failed_events     # get_events_by_status(FAILED)
executor.processing_events # get_events_by_status(PROCESSING)

# Status counts (no event fetching)
counts = executor.status_counts()
# {"pending": 10, "processing": 2, "completed": 88, "failed": 0, "aborted": 0}
```

**Example**:

```python
# Submit 10,000 events
for i in range(10_000):
    await executor.append(MyEvent())

# Process for 1 minute
await executor.start()
await asyncio.sleep(60)
await executor.stop()

# Query results (fast!)
completed = executor.completed_events  # ~5ms
failed = executor.failed_events        # ~0.1ms (if few failures)

print(f"Completed: {len(completed)}")
print(f"Failed: {len(failed)}")
```

**API Reference**: [Executor.get_events_by_status()](../api/base/executor.md#get_events_by_status), [Executor.status_counts()](../api/base/executor.md#status_counts)

---

### Pattern 9: Streaming vs Non-Streaming Events

**What**: Processor handles both regular events and streaming events.

**Event Types**:

```python
# Regular event
class MyEvent(Event):
    async def _invoke(self):
        return "result"  # Single return value

# Streaming event
class MyStreamingEvent(Event):
    streaming = True

    async def _invoke(self):
        for i in range(10):
            yield i  # Multiple yielded values
```

**Processor Handling**:

```python
# In process()
if next_event.streaming:
    # Streaming: consume async generator
    async def consume_stream(event):
        async for _ in event.stream():
            pass  # Process each yielded value
        await executor._update_progression(event)

    tg.start_soon(self._with_semaphore, consume_stream(event))
else:
    # Non-streaming: single invoke
    async def invoke_and_update(event):
        await event.invoke()  # Returns single result
        await executor._update_progression(event)

    tg.start_soon(self._with_semaphore, invoke_and_update(event))
```

**Why Different Handling?**

- **Regular**: Call `invoke()` once, get result
- **Streaming**: Call `stream()`, iterate async generator
- **Progression update**: After stream completes (not per-yield)

**Example**:

```python
# Define streaming event
class DataFetcher(Event):
    streaming = True

    async def _invoke(self):
        for page in range(10):
            data = await fetch_page(page)
            yield data  # Yield each page

# Submit to executor
event = DataFetcher()
await executor.append(event)
await executor.start()

# Processor consumes entire stream
# Status only updates after all pages fetched
```

**Capacity Consumption**:

- Regular event: 1 capacity
- Streaming event: 1 capacity (regardless of yield count)
- Rationale: Capacity limits concurrent tasks, not operations per task

**API Reference**: [Event.streaming](../api/base/event.md#streaming), [Event.stream()](../api/base/event.md#stream)

---

### Pattern 10: Permission Checks and Rate Limiting

**What**: Processor supports custom permission checks via `request_permission()` override.

**Base Implementation**:

```python
# In Processor
async def request_permission(self, **kwargs) -> bool:
    """Override for custom checks (rate limits, permissions, quotas)."""
    return True  # Default: always allow
```

**Override Pattern**:

```python
class RateLimitedProcessor(Processor):
    def __init__(self, max_requests_per_second: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.rate_limiter = RateLimiter(max_requests_per_second)

    async def request_permission(self, **kwargs) -> bool:
        return await self.rate_limiter.check()
```

**Permission Denial Handling**:

```python
# In process()
if await self.request_permission(**next_event.request):
    # Permission granted
    await process_event()
else:
    # Permission denied
    denial_count = self._denial_counts.get(event_id, 0) + 1
    self._denial_counts[event_id] = denial_count

    if denial_count >= 3:
        # 3 strikes - abort event
        await executor._update_progression(event, EventStatus.ABORTED)
    else:
        # Requeue with exponential backoff
        backoff = denial_count * 1.0
        await self.queue.put((priority + backoff, event_id))
```

**Backoff Strategy**:

- 1st denial: Requeue with priority + 1.0 (1 second delay)
- 2nd denial: Requeue with priority + 2.0 (2 second delay)
- 3rd denial: Abort event (EventStatus.ABORTED)

**Example - Rate Limiting**:

```python
class RateLimitedProcessor(Processor):
    def __init__(self, max_per_second: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.tokens = max_per_second
        self.max_tokens = max_per_second
        self.last_refill = datetime.now()

    async def request_permission(self, **kwargs) -> bool:
        # Refill tokens
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.max_tokens)
        self.last_refill = now

        # Check tokens
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

**Example - Authorization**:

```python
class AuthProcessor(Processor):
    def __init__(self, auth_service, **kwargs):
        super().__init__(**kwargs)
        self.auth_service = auth_service

    async def request_permission(self, user_id: str, resource: str, **kwargs) -> bool:
        return await self.auth_service.check_permission(user_id, resource)
```

**Capacity Interaction**:

- Permission granted → consumes capacity
- Permission denied → **doesn't consume capacity** (allows retries)
- Aborted events → removed from queue (capacity not consumed)

**API Reference**: [Processor.request_permission()](../api/base/processor.md#request_permission)

---

### Pattern 11: Async Factory Pattern

**What**: Both classes use `create()` classmethod for async initialization.

**Processor**:

```python
@classmethod
async def create(
    cls,
    queue_capacity: int,
    capacity_refresh_time: float,
    pile: Pile[Event],
    executor: Executor | None = None,
    **kwargs,
) -> Self:
    return cls(
        queue_capacity=queue_capacity,
        capacity_refresh_time=capacity_refresh_time,
        pile=pile,
        executor=executor,
        **kwargs,
    )
```

**Executor**:

```python
async def _create_processor(self):
    self.processor = await self.processor_type.create(
        pile=self.states.items,
        executor=self,
        **self.processor_config,
    )
```

**Why Async Factory?**

- **Current**: Body is sync, but signature is async
- **Future**: Enables async initialization (DB connections, resource loading)
- **Consistency**: Uniform API (always `await create()`)

**Pattern Benefits**:

- Future-proof (no breaking changes when async init needed)
- Explicit construction (not hidden in `__init__`)
- Testable (easy to mock async initialization)

**Example**:

```python
# Direct construction (for simple cases)
processor = Processor(
    queue_capacity=100,
    capacity_refresh_time=1.0,
    pile=pile,
)

# Async factory (for consistent API)
processor = await Processor.create(
    queue_capacity=100,
    capacity_refresh_time=1.0,
    pile=pile,
)

# Future async initialization (hypothetical)
@classmethod
async def create(cls, ...) -> Self:
    instance = cls(...)
    await instance._load_rate_limits_from_db()  # Async init
    return instance
```

**API Reference**: [Processor.create()](../api/base/processor.md#create)

---

### Pattern 12: ClassVar Type Annotations

**What**: Both classes use ClassVar for type configuration.

**Processor**:

```python
class MyProcessor(Processor):
    event_type: ClassVar[type[Event]] = MyEvent  # Must override

class Processor:
    event_type: ClassVar[type[Event]]  # Abstract (no default)
```

**Executor**:

```python
class MyExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = MyProcessor  # Must override

class Executor:
    processor_type: ClassVar[type[Processor]]  # Abstract (no default)
```

**Why ClassVar?**

- **Class-level**: Shared across all instances (not per-instance)
- **Type System**: Type checkers understand this is class configuration
- **Abstract**: Subclasses must provide values (no default in base)

**Type Safety**:

```python
# Executor automatically infers event_type from processor_type
class MyExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = MyProcessor

executor = MyExecutor()
assert executor.event_type == MyEvent  # From MyProcessor.event_type
```

**Example**:

```python
# Define custom event
class DataEvent(Event):
    data: str

    async def _invoke(self):
        return f"Processed: {self.data}"

# Define processor
class DataProcessor(Processor):
    event_type: ClassVar[type[Event]] = DataEvent

# Define executor
class DataExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = DataProcessor

# Type inference works
executor = DataExecutor()
assert executor.event_type == DataEvent
assert executor.processor_type == DataProcessor
```

**Validation**:

```python
# Runtime validation in Flow
executor.states = Flow(
    item_type=executor.processor_type.event_type,  # DataEvent
    strict_type=strict_event_type,
)

# Attempt to add wrong event type
await executor.append(OtherEvent())  # Raises ValidationError if strict_type=True
```

**API Reference**: [Processor.event_type](../api/base/processor.md#event_type), [Executor.processor_type](../api/base/executor.md#processor_type)

---

## Complete Workflows

### Workflow 1: Basic Event Processing System

**Goal**: Submit events, process in background, query results.

### Step 1: Define Custom Event**

```python
from lionherd_core import Event

class TaskEvent(Event):
    task_id: str
    data: dict

    async def _invoke(self):
        # Simulate work
        await asyncio.sleep(0.1)
        return {"task_id": self.task_id, "result": "success"}
```

### Step 2: Define Custom Processor**

```python
from lionherd_core import Processor
from typing import ClassVar

class TaskProcessor(Processor):
    event_type: ClassVar[type[Event]] = TaskEvent
```

### Step 3: Define Custom Executor**

```python
from lionherd_core import Executor

class TaskExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = TaskProcessor
```

### Step 4: Create Executor and Configure**

```python
executor = TaskExecutor(
    processor_config={
        "queue_capacity": 50,           # Process 50 events per batch
        "capacity_refresh_time": 1.0,   # 1 second between batches
        "concurrency_limit": 10,        # Max 10 concurrent tasks
    }
)
```

### Step 5: Submit Events**

```python
for i in range(100):
    event = TaskEvent(task_id=f"task-{i}", data={"value": i})
    await executor.append(event)

print(f"Submitted: {len(executor.pending_events)} events")
```

### Step 6: Start Background Processing**

```python
import anyio

# Start executor (creates processor, backfills events)
await executor.start()

# Run execute() in background
async with anyio.create_task_group() as tg:
    tg.start_soon(executor.processor.execute)

    # Let it run for 10 seconds
    await anyio.sleep(10)

    # Stop gracefully
    await executor.stop()
```

### Step 7: Query Results**

```python
print(f"Completed: {len(executor.completed_events)}")
print(f"Failed: {len(executor.failed_events)}")
print(f"Pending: {len(executor.pending_events)}")

# Inspect first completed event
if executor.completed_events:
    event = executor.completed_events[0]
    print(f"Result: {event.execution.output}")
```

**Complete Code**:

```python
import asyncio
import anyio
from typing import ClassVar
from lionherd_core import Event, Processor, Executor

# 1. Define event
class TaskEvent(Event):
    task_id: str
    data: dict

    async def _invoke(self):
        await asyncio.sleep(0.1)
        return {"task_id": self.task_id, "result": "success"}

# 2. Define processor
class TaskProcessor(Processor):
    event_type: ClassVar[type[Event]] = TaskEvent

# 3. Define executor
class TaskExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = TaskProcessor

# 4. Run
async def main():
    # Create executor
    executor = TaskExecutor(
        processor_config={
            "queue_capacity": 50,
            "capacity_refresh_time": 1.0,
            "concurrency_limit": 10,
        }
    )

    # Submit events
    for i in range(100):
        await executor.append(TaskEvent(task_id=f"task-{i}", data={"value": i}))

    # Process
    await executor.start()
    async with anyio.create_task_group() as tg:
        tg.start_soon(executor.processor.execute)
        await anyio.sleep(10)
        await executor.stop()

    # Results
    print(f"Completed: {len(executor.completed_events)}")
    print(f"Failed: {len(executor.failed_events)}")

anyio.run(main)
```

---

### Workflow 2: Rate-Limited API Calls

**Goal**: Process API requests with rate limiting (max 10 requests/second).

### Step 1: Define API Event**

```python
import httpx

class APIRequestEvent(Event):
    url: str
    method: str = "GET"

    async def _invoke(self):
        async with httpx.AsyncClient() as client:
            response = await client.request(self.method, self.url)
            return response.json()
```

### Step 2: Define Rate-Limited Processor**

```python
from datetime import datetime

class RateLimitedProcessor(Processor):
    event_type: ClassVar[type[Event]] = APIRequestEvent

    def __init__(self, max_per_second: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.tokens = max_per_second
        self.max_tokens = max_per_second
        self.last_refill = datetime.now()

    async def request_permission(self, **kwargs) -> bool:
        # Token bucket algorithm
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()

        # Refill tokens
        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.max_tokens
        )
        self.last_refill = now

        # Check and consume token
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

### Step 3: Define Executor**

```python
class APIExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = RateLimitedProcessor
```

### Step 4: Run**

```python
async def main():
    executor = APIExecutor(
        processor_config={
            "queue_capacity": 100,          # Large batch
            "capacity_refresh_time": 0.5,   # Fast refresh
            "concurrency_limit": 5,         # Limit concurrent requests
            "max_per_second": 10,           # Rate limit
        }
    )

    # Submit 100 API requests
    for i in range(100):
        await executor.append(
            APIRequestEvent(url=f"https://api.example.com/item/{i}")
        )

    # Process (will take ~10 seconds due to rate limit)
    await executor.start()
    async with anyio.create_task_group() as tg:
        tg.start_soon(executor.processor.execute)
        await anyio.sleep(15)  # Give time for rate limiting
        await executor.stop()

    print(f"Completed: {len(executor.completed_events)}")
    print(f"Failed: {len(executor.failed_events)}")
    print(f"Aborted: {len(executor.get_events_by_status('aborted'))}")

anyio.run(main)
```

---

### Workflow 3: Priority-Based Processing

**Goal**: Process high-priority events first.

### Step 1: Define Event with Priority**

```python
class PriorityEvent(Event):
    priority_level: str  # "critical", "high", "normal", "low"
    data: dict

    async def _invoke(self):
        await asyncio.sleep(0.1)
        return {"priority": self.priority_level, "data": self.data}
```

### Step 2: Standard Processor and Executor**

```python
class PriorityProcessor(Processor):
    event_type: ClassVar[type[Event]] = PriorityEvent

class PriorityExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = PriorityProcessor
```

### Step 3: Submit with Custom Priorities**

```python
async def main():
    executor = PriorityExecutor(
        processor_config={
            "queue_capacity": 10,
            "capacity_refresh_time": 1.0,
        }
    )

    # Priority mapping
    priority_map = {
        "critical": 1.0,
        "high": 5.0,
        "normal": 10.0,
        "low": 20.0,
    }

    # Submit mixed priorities
    for level in ["low", "normal", "high", "critical"]:
        for i in range(10):
            event = PriorityEvent(priority_level=level, data={"id": i})
            await executor.append(event, priority=priority_map[level])

    # Process
    await executor.start()
    async with anyio.create_task_group() as tg:
        tg.start_soon(executor.processor.execute)
        await anyio.sleep(5)
        await executor.stop()

    # Verify order
    completed = executor.completed_events
    for event in completed[:10]:
        print(f"Priority: {event.priority_level}")  # Should be "critical" first

anyio.run(main)
```

---

### Workflow 4: Monitoring and Debugging

**Goal**: Monitor processing state and debug failures.

### Step 1: Define Event with Simulated Failures**

```python
import random

class UnreliableEvent(Event):
    event_id: str
    fail_rate: float = 0.2  # 20% failure rate

    async def _invoke(self):
        if random.random() < self.fail_rate:
            raise ValueError(f"Simulated failure: {self.event_id}")
        return {"event_id": self.event_id, "status": "success"}
```

### Step 2: Standard Setup**

```python
class UnreliableProcessor(Processor):
    event_type: ClassVar[type[Event]] = UnreliableEvent

class UnreliableExecutor(Executor):
    processor_type: ClassVar[type[Processor]] = UnreliableProcessor
```

### Step 3: Monitor During Processing**

```python
async def main():
    executor = UnreliableExecutor(
        processor_config={
            "queue_capacity": 20,
            "capacity_refresh_time": 1.0,
        }
    )

    # Submit events
    for i in range(100):
        await executor.append(UnreliableEvent(event_id=f"event-{i}"))

    # Start processing
    await executor.start()

    async with anyio.create_task_group() as tg:
        tg.start_soon(executor.processor.execute)

        # Monitor every 2 seconds
        for _ in range(5):
            await anyio.sleep(2)
            print(executor.inspect_state())
            print(f"Capacity: {executor.processor.available_capacity}")
            print("---")

        await executor.stop()

    # Debug failures
    print(f"\nFinal State:")
    print(executor.inspect_state())

    for failed_event in executor.failed_events:
        print(f"Failed: {failed_event.event_id}")
        print(f"Error: {failed_event.execution.error}")

anyio.run(main)
```

**Sample Output**:

```text
Executor State (executor_states):
  pending: 80 events
  processing: 0 events
  completed: 15 events
  failed: 5 events
  aborted: 0 events
Capacity: 20
---
Executor State (executor_states):
  pending: 40 events
  processing: 0 events
  completed: 45 events
  failed: 15 events
  aborted: 0 events
Capacity: 20
---
...
```

---

## Advanced Topics

### Custom Progressions

Beyond the 5 default EventStatus progressions, you can add custom progressions for business logic:

```python
# Add custom progression after executor creation
executor = MyExecutor()
executor.states.add_progression(Progression(name="stage1"))
executor.states.add_progression(Progression(name="stage2"))

# Add events to custom progressions
await executor.append(event)
executor.states.add_item(event, progressions=["pending", "stage1"])

# Query custom progressions
stage1_events = [
    executor.states.items[uid]
    for uid in executor.states.get_progression("stage1")
]
```

**Use Cases**:

- Multi-stage workflows (stage1 → stage2 → stage3)
- Business categories (premium, standard, free)
- Geographic regions (us-east, eu-west)
- Error recovery stages (retry1, retry2, retry3)

---

### State Serialization

Executor state can be serialized for persistence or debugging:

```python
# Serialize entire state
state_dict = executor.states.to_dict()

# Save to file
import json
with open("executor_state.json", "w") as f:
    json.dump(state_dict, f, indent=2)

# Restore from file
with open("executor_state.json", "r") as f:
    state_dict = json.load(f)

from lionherd_core import Flow
restored_states = Flow.from_dict(state_dict)
```

**Serialized Format**:

```json
{
  "items": {
    "uuid-1": {"event_type": "TaskEvent", "task_id": "task-1", ...},
    "uuid-2": {"event_type": "TaskEvent", "task_id": "task-2", ...}
  },
  "progressions": {
    "pending": ["uuid-1"],
    "completed": ["uuid-2"]
  }
}
```

**Use Cases**:

- Checkpoint/restore (long-running processes)
- Audit logs (compliance)
- Debugging (inspect state at failure point)
- Migration (transfer state between systems)

---

### Manual Processing (Standalone Processor)

Processor can be used without Executor for manual control:

```python
from lionherd_core import Pile

# Create pile manually
pile = Pile(item_type=TaskEvent)

# Add events
events = [TaskEvent(task_id=f"task-{i}") for i in range(10)]
for event in events:
    pile.add(event)

# Create processor without executor
processor = await TaskProcessor.create(
    queue_capacity=5,
    capacity_refresh_time=1.0,
    pile=pile,
    executor=None,  # No executor
)

# Enqueue manually
for event in events:
    await processor.enqueue(event.id)

# Process single batch
await processor.process()

# Check results
for event in pile:
    print(f"{event.task_id}: {event.execution.status}")
```

**When to Use**:

- Testing (isolated processor tests)
- Simple use cases (no status tracking needed)
- Custom state management (not Flow-based)

---

### Queue Size Limits

Processor enforces max queue size to prevent unbounded growth:

```python
processor = await MyProcessor.create(
    queue_capacity=100,
    max_queue_size=1000,  # Max 1000 events in queue
    ...
)

# Attempt to exceed limit
for i in range(1001):
    await processor.enqueue(event.id)  # Raises QueueFullError at 1001
```

**Handling QueueFullError**:

```python
from lionherd_core.errors import QueueFullError

try:
    await processor.enqueue(event.id)
except QueueFullError as e:
    # Option 1: Wait and retry
    await asyncio.sleep(1)
    await processor.enqueue(event.id)

    # Option 2: Process immediately
    await processor.process()
    await processor.enqueue(event.id)

    # Option 3: Drop event
    print(f"Queue full, dropping event {event.id}")
```

---

### Denial Tracking Limits

Processor limits denial tracking to prevent unbounded memory growth:

```python
processor = await MyProcessor.create(
    max_denial_tracking=10000,  # Max 10k entries
    ...
)

# When limit exceeded, oldest entries evicted (FIFO)
# Prevents memory leak from denied events accumulating
```

**Denial Lifecycle**:

1. Permission denied → track denial count
2. Denied 3 times → abort event, remove from tracking
3. Permission granted → remove from tracking
4. Exceeds max_denial_tracking → evict oldest entry (FIFO)

---

### Multiple Executors

You can run multiple executors in parallel for workload isolation:

```python
# Executor 1: High priority tasks
executor1 = TaskExecutor(
    name="high_priority",
    processor_config={
        "queue_capacity": 100,
        "capacity_refresh_time": 0.5,  # Fast
        "concurrency_limit": 50,        # High concurrency
    }
)

# Executor 2: Low priority tasks
executor2 = TaskExecutor(
    name="low_priority",
    processor_config={
        "queue_capacity": 20,
        "capacity_refresh_time": 5.0,   # Slow
        "concurrency_limit": 5,          # Low concurrency
    }
)

# Route events by priority
if event.priority == "high":
    await executor1.append(event)
else:
    await executor2.append(event)

# Start both
await executor1.start()
await executor2.start()

async with anyio.create_task_group() as tg:
    tg.start_soon(executor1.processor.execute)
    tg.start_soon(executor2.processor.execute)
    await anyio.sleep(60)
    await executor1.stop()
    await executor2.stop()
```

---

## Troubleshooting

### Issue: Events Not Processing

**Symptoms**: Events stuck in PENDING status.

**Possible Causes**:

1. **Processor not started**:

```python
# ❌ WRONG: Executor created but not started
executor = MyExecutor()
await executor.append(event)
# Events stay pending forever

# ✅ CORRECT: Start executor
await executor.start()
async with anyio.create_task_group() as tg:
    tg.start_soon(executor.processor.execute)
```

1. **Permission always denied**:

```python
# Check denial counts
print(executor.processor._denial_counts)

# Fix: Review request_permission() logic
```

1. **Queue full**:

```python
# Check queue size
print(f"Queue size: {executor.processor.queue.qsize()}")

# Fix: Process events or increase max_queue_size
await executor.processor.process()
```

---

### Issue: High Memory Usage

**Symptoms**: Memory grows unbounded during processing.

**Possible Causes**:

1. **Events not cleaned up**:

```python
# ✅ SOLUTION: Cleanup completed/failed events periodically
await executor.cleanup_events([EventStatus.COMPLETED, EventStatus.FAILED])
```

1. **Denial tracking unbounded**:

```python
# ✅ SOLUTION: Set max_denial_tracking
processor = await MyProcessor.create(
    max_denial_tracking=10000,  # Limit tracking
    ...
)
```

1. **Too many concurrent tasks**:

```python
# ✅ SOLUTION: Lower concurrency_limit
processor = await MyProcessor.create(
    concurrency_limit=50,  # Reduce from default 100
    ...
)
```

---

### Issue: Slow Processing

**Symptoms**: Events taking too long to complete.

**Possible Causes**:

1. **Low capacity**:

```python
# ❌ PROBLEM: Small batches
processor_config = {
    "queue_capacity": 10,           # Too small
    "capacity_refresh_time": 5.0,   # Too slow
}

# ✅ SOLUTION: Increase capacity and refresh rate
processor_config = {
    "queue_capacity": 100,          # Larger batches
    "capacity_refresh_time": 1.0,   # Faster refresh
}
```

1. **Low concurrency**:

```python
# ✅ SOLUTION: Increase concurrency_limit
processor_config = {
    "concurrency_limit": 100,  # More concurrent tasks
}
```

1. **Rate limiting too strict**:

```python
# Check permission denials
denied_count = len(executor.processor._denial_counts)
print(f"Denied events: {denied_count}")

# Fix: Adjust rate limiter
```

---

### Issue: Events Stuck in PROCESSING

**Symptoms**: Events never complete.

**Possible Causes**:

1. **Event never returns**:

```python
# ❌ PROBLEM: Event hangs
class HangingEvent(Event):
    async def _invoke(self):
        await asyncio.sleep(float('inf'))  # Never completes

# ✅ SOLUTION: Add timeout
class TimeoutEvent(Event):
    async def _invoke(self):
        async with asyncio.timeout(10):  # 10 second timeout
            await do_work()
```

1. **Exception not handled**:

```python
# Check failed events
for event in executor.failed_events:
    print(f"Error: {event.execution.error}")
```

---

### Issue: QueueFullError

**Symptoms**: Exception when enqueueing events.

**Solutions**:

```python
# Option 1: Increase max_queue_size
processor_config = {"max_queue_size": 10000}

# Option 2: Process before enqueueing more
await executor.processor.process()
await executor.append(event)

# Option 3: Wait for queue to drain
while executor.processor.queue.qsize() > 900:
    await asyncio.sleep(0.1)
await executor.append(event)
```

---

### Issue: Permission Denials Causing Aborts

**Symptoms**: Events aborted after 3 denials.

**Solutions**:

```python
# Option 1: Review permission logic
class MyProcessor(Processor):
    async def request_permission(self, **kwargs) -> bool:
        # Add logging
        result = await self.check_rate_limit()
        print(f"Permission check: {result}")
        return result

# Option 2: Increase backoff (custom Processor)
class CustomProcessor(Processor):
    async def process(self):
        # Override to use longer backoff
        # Instead of denial_count * 1.0, use exponential:
        backoff = 2 ** denial_count  # 2s, 4s, 8s
```

---

## Best Practices

### 1. Event Design

**DO**:

```python
# Simple, focused events
class FetchUserEvent(Event):
    user_id: str

    async def _invoke(self):
        return await fetch_user(self.user_id)
```

**DON'T**:

```python
# Complex, multi-purpose events
class DoEverythingEvent(Event):
    action: str  # "fetch" | "update" | "delete"
    data: dict

    async def _invoke(self):
        if self.action == "fetch":
            ...
        elif self.action == "update":
            ...
        # Anti-pattern: too much branching
```

---

### 2. Capacity Configuration

**Guidelines**:

- **Development**: Small batches for fast iteration (capacity=10, refresh=0.5s)
- **Production**: Balanced config (capacity=100, refresh=1.0s)
- **High throughput**: Large batches (capacity=500, refresh=5.0s)
- **Low latency**: Small batches with fast refresh (capacity=20, refresh=0.1s)

---

### 3. Error Handling

**DO**:

```python
class RobustEvent(Event):
    async def _invoke(self):
        try:
            return await do_work()
        except SpecificError as e:
            # Handle expected errors
            return {"error": str(e)}
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error: {e}")
            raise  # Re-raise for status tracking
```

**DON'T**:

```python
class QuietFailureEvent(Event):
    async def _invoke(self):
        try:
            return await do_work()
        except Exception:
            return None  # Silent failure - hard to debug
```

---

### 4. Testing

```python
# Test processor in isolation
async def test_processor():
    pile = Pile(item_type=TaskEvent)
    processor = await TaskProcessor.create(
        queue_capacity=10,
        capacity_refresh_time=0.1,
        pile=pile,
    )

    # Add test events
    events = [TaskEvent(task_id=f"test-{i}") for i in range(5)]
    for event in events:
        pile.add(event)
        await processor.enqueue(event.id)

    # Process
    await processor.process()

    # Verify
    for event in events:
        assert event.execution.status == EventStatus.COMPLETED
```

---

## Summary

The Processor/Executor pattern provides:

1. **Separation of Concerns**: State tracking (Executor) vs processing (Processor)
2. **O(1) Performance**: Flow-based status queries
3. **Capacity Control**: Batch processing with configurable limits
4. **Priority Support**: Custom event ordering
5. **Type Safety**: Event subclasses with validation
6. **Extensibility**: Custom permissions, rate limiting, etc.

**12 Shared Patterns** (Canonical Reference):

1. UUID-based event references (lightweight queuing)
2. Flow-based state tracking (O(1) queries)
3. Event lifecycle (status transitions)
4. Priority queue mechanics (custom ordering)
5. Capacity-limited batch processing (throughput control)
6. Concurrency control (semaphore)
7. Event lifecycle management (start/stop)
8. O(1) status queries (progression-based)
9. Streaming vs non-streaming events
10. Permission checks and rate limiting
11. Async factory pattern (future-proof)
12. ClassVar type annotations (type safety)

**Next Steps**:

- Review [Processor API Reference](../api/base/processor.md) for detailed API
- Review [Executor API Reference](../api/base/executor.md) for detailed API
- Try [Reference Notebook](../../notebooks/references/processor_executor.ipynb) for copy-paste examples
- Follow [Event Processing Tutorial](../../notebooks/tutorials/event_processing_system.ipynb) for hands-on learning

---

**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

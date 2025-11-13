# EventBus

> Instance-based pub/sub with topic routing and concurrent handler execution for observability patterns.

---

## Overview

**EventBus** provides instance-based publish-subscribe with topic-based routing and concurrent handler execution. Each EventBus instance maintains its own subscription registry, enabling isolated event channels for different concerns (metrics, tracing, logging).

**Key Features:**

- **Instance-Based**: Each EventBus has independent subscription registry
- **Topic Routing**: Handlers only receive events for subscribed topics
- **Concurrent Execution**: Handlers run concurrently via `gather()`
- **Weakref Cleanup**: Automatic handler removal when garbage collected
- **Exception Isolation**: Handler exceptions suppressed (fire-and-forget)
- **Flexible Signatures**: Handlers receive arbitrary `*args` and `**kwargs`

**Use Cases:**

- Observability (metrics collection, distributed tracing, logging)
- Multi-topic event routing within single component
- Decoupled notification systems requiring topic filtering
- Cross-cutting concerns with isolated event channels

---

## When to Use EventBus

**✅ Use EventBus when:**

- You need topic-based routing (different handlers for different events)
- Handlers should run concurrently for performance
- Per-instance event isolation is required (not global)
- Observability patterns (metrics, tracing, logging aggregation)

**❌ Avoid EventBus when:**

- You need global event channel (use Broadcaster)
- Memory overhead matters more than flexibility (use Broadcaster)
- Sequential execution is required (use Broadcaster)
- Simple single-topic pub/sub suffices (use Broadcaster)

---

## Class Signature

```python
class EventBus:
    """In-process pub/sub with concurrent handler execution."""

    def __init__(self) -> None:
        """Initialize with empty subscription registry."""
        ...
```

---

## Constructor

```python
def __init__(self) -> None
```

Creates new EventBus instance with empty subscription registry.

**Parameters:** None

**Examples:**

```python
from lionherd_core.base.eventbus import EventBus

# Create independent event buses
metrics_bus = EventBus()
trace_bus = EventBus()

# Each has separate subscription registry
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_subs` | `dict[str, list[weakref.ref[Handler]]]` | Subscription registry (topic → weakrefs to handlers) |

**Note**: `_subs` uses `defaultdict(list)` for automatic topic creation on subscription.

---

## Methods

### Category 1: Subscription Management

#### `subscribe(topic, handler)`

Subscribe async handler to topic.

**Signature:**

```python
def subscribe(self, topic: str, handler: Handler) -> None: ...

# where Handler = Callable[..., Awaitable[None]]
```

**Parameters:**

- `topic` (str): Topic identifier (e.g., "request.start", "node.complete")
- `handler` (Handler): Async function accepting `*args` and `**kwargs`

**Behavior:**

- Stores handler as weakref for automatic cleanup
- No deduplication: Same handler can be subscribed multiple times
- Creates topic entry automatically if not exists (defaultdict)

**Examples:**

```python
bus = EventBus()

# Subscribe to different topics
async def log_start(node_id: str):
    print(f"START: {node_id}")

async def log_complete(node_id: str, duration: float):
    print(f"COMPLETE: {node_id} ({duration}s)")

bus.subscribe("node.start", log_start)
bus.subscribe("node.complete", log_complete)
```

#### `unsubscribe(topic, handler)`

Remove handler from topic.

**Signature:**

```python
def unsubscribe(self, topic: str, handler: Handler) -> bool: ...
```

**Parameters:**

- `topic` (str): Topic identifier
- `handler` (Handler): Previously subscribed handler

**Returns:**

- `bool`: True if handler found and removed, False otherwise

**Behavior:**

- Removes first weakref matching handler (identity comparison)
- Returns False if topic doesn't exist or handler not found
- Does NOT clean up dead weakrefs as side effect

**Examples:**

```python
async def temp_handler(**kwargs):
    print("Temporary")

bus.subscribe("test", temp_handler)
removed = bus.unsubscribe("test", temp_handler)
print(removed)  # True
```

#### `clear(topic=None)`

Clear subscriptions for topic (or all topics if None).

**Signature:**

```python
def clear(self, topic: str | None = None) -> None: ...
```

**Parameters:**

- `topic` (str, optional): Topic to clear. If None, clears ALL topics.

**Behavior:**

- `topic=None`: Clears entire subscription registry
- `topic="specific"`: Removes topic entry (no-op if topic doesn't exist)

**Examples:**

```python
bus.clear("node.start")  # Clear specific topic
bus.clear()              # Clear all topics
```

#### `topics()`

Get list of all registered topics.

**Signature:**

```python
def topics(self) -> list[str]: ...
```

**Returns:**

- `list[str]`: List of topic identifiers with subscriptions

**Examples:**

```python
bus.subscribe("node.start", handler1)
bus.subscribe("node.complete", handler2)

print(bus.topics())  # ['node.start', 'node.complete']
```

#### `handler_count(topic)`

Get number of live handlers for topic (excludes garbage-collected handlers).

**Signature:**

```python
def handler_count(self, topic: str) -> int: ...
```

**Parameters:**

- `topic` (str): Topic identifier

**Returns:**

- `int`: Number of live handlers after cleanup (0 if topic doesn't exist)

**Side Effects:**

- **Mutates state**: Cleans up dead weakrefs as side effect
- This ensures count reflects actual live handlers

**Examples:**

```python
count = bus.handler_count("node.start")
print(f"Active handlers: {count}")
```

### Category 2: Event Emission

#### `emit(topic, *args, **kwargs)` (async)

Emit event to all topic subscribers.

**Signature:**

```python
async def emit(self, topic: str, *args: Any, **kwargs: Any) -> None: ...
```

**Parameters:**

- `topic` (str): Topic identifier
- `*args` (Any): Positional arguments passed to handlers
- `**kwargs` (Any): Keyword arguments passed to handlers

**Behavior:**

1. Returns early if topic has no subscriptions
2. Cleans up dead weakrefs (lazy cleanup)
3. Executes all live handlers concurrently via `gather()`
4. Exceptions suppressed via `return_exceptions=True`

**Concurrency:**

- Handlers run concurrently (not sequentially)
- All handlers started together via `gather()`
- No guaranteed execution order

**Examples:**

```python
# Emit with keyword args
await bus.emit("node.start", node_id="n1")

# Emit with mixed args
await bus.emit("node.complete", node_id="n1", duration=0.5)

# No subscribers - no-op
await bus.emit("unknown.topic")
```

### Category 3: Internal (Advanced)

#### `_cleanup_dead_refs(topic)`

Remove garbage-collected handlers and return live handlers.

**Signature:**

```python
def _cleanup_dead_refs(self, topic: str) -> list[Handler]: ...
```

**Parameters:**

- `topic` (str): Topic identifier

**Returns:**

- `list[Handler]`: Live handler callables (weakrefs resolved)

**Side Effects:**

- **Mutates state**: Updates `_subs[topic]` in-place to remove dead weakrefs

**Internal Use Only**: Called automatically by `emit()` and `handler_count()`.

---

## Design Rationale

### Instance-Based vs Singleton

**Why instance-based instead of singleton like Broadcaster?**

1. **Isolation**: Different EventBus instances for different concerns (metrics vs tracing)
2. **Flexibility**: Each component can have its own event channel
3. **Testing**: Easy to mock/isolate in tests (pass custom EventBus)

**Implementation**: No `__new__` override, standard `__init__` for per-instance state.

**Trade-off**: Higher memory overhead than Broadcaster, but more flexible.

### Concurrent Handler Execution

**Why concurrent execution via `gather()` instead of sequential?**

1. **Performance**: Handlers run in parallel (critical for I/O-bound operations)
2. **Latency**: Slow handlers don't block fast handlers
3. **Observability**: Multiple metrics collectors can run concurrently

**Implementation**: `await gather(*(h(*args, **kwargs) for h in handlers), return_exceptions=True)` (uses custom `gather` from `lionherd_core.libs.concurrency`)

**Trade-off**: Unpredictable execution order, but far better performance.

### Topic-Based Routing

**Why topic filtering instead of broadcast-to-all?**

1. **Selective Handling**: Different handlers for different event types
2. **Decoupling**: Metrics handler doesn't receive tracing events
3. **Scalability**: Avoid waking up irrelevant handlers

**Pattern**: Topic strings as namespaced identifiers (e.g., "node.start", "request.complete").

**Trade-off**: Slightly more complex API, but much more flexible.

### Exception Suppression (return_exceptions=True)

**Why suppress exceptions instead of raising?**

1. **Isolation**: One failing handler doesn't break others
2. **Fire-and-Forget**: EventBus doesn't await results or handle errors
3. **Robustness**: Production systems stay operational despite handler bugs

**Implementation**: `gather(..., return_exceptions=True)` catches all exceptions.

**Trade-off**: Less control over error handling, but more robust in production.

---

## Usage Patterns

### Pattern 1: Metrics Collection

```python
# noqa:validation
from lionherd_core.base.eventbus import EventBus

bus = EventBus()
metrics = {"requests": 0, "errors": 0, "total_duration": 0.0}

# Metrics handlers
async def count_request(**kwargs):
    metrics["requests"] += 1

async def count_error(**kwargs):
    metrics["errors"] += 1

async def track_duration(duration: float, **kwargs):
    metrics["total_duration"] += duration

# Subscribe to lifecycle events
bus.subscribe("request.start", count_request)
bus.subscribe("request.complete", track_duration)
bus.subscribe("request.error", count_error)

# Simulate request lifecycle
await bus.emit("request.start", request_id="r1")
await bus.emit("request.complete", request_id="r1", duration=0.15)
await bus.emit("request.start", request_id="r2")
await bus.emit("request.error", request_id="r2", error="timeout")

print(metrics)
# Output: {'requests': 2, 'errors': 1, 'total_duration': 0.15}
```

### Pattern 2: Cross-Cutting Observability

```python
# noqa:validation
# Separate event buses for different concerns
metrics_bus = EventBus()
trace_bus = EventBus()
log_bus = EventBus()

# Metrics collector
async def collect_metrics(event_type: str, **data):
    metrics_store.record(event_type, data)

# Distributed tracer
async def trace_span(event_type: str, span_id: str, **data):
    tracer.record(span_id, event_type, data)

# Logger
async def log_event(level: str, message: str, **context):
    logger.log(level, message, extra=context)

# Subscribe to respective buses
metrics_bus.subscribe("event", collect_metrics)
trace_bus.subscribe("span", trace_span)
log_bus.subscribe("log", log_event)

# Emit to different channels
await metrics_bus.emit("event", event_type="request", latency=0.1)
await trace_bus.emit("span", event_type="start", span_id="s1")
await log_bus.emit("log", level="INFO", message="Request processed")
```

### Pattern 3: Multi-Handler Concurrent Execution

```python
# noqa:validation
from lionherd_core.base.eventbus import EventBus
from lionherd_core.libs.concurrency import sleep

bus = EventBus()
execution_order = []

async def slow_handler():
    execution_order.append("slow_start")
    await sleep(0.02)
    execution_order.append("slow_end")

async def fast_handler():
    execution_order.append("fast")

# Both subscribe to same topic
bus.subscribe("test", slow_handler)
bus.subscribe("test", fast_handler)

# Handlers run concurrently
await bus.emit("test")

print(execution_order)
# Output: ['slow_start', 'fast', 'slow_end']
# Note: 'fast' completes before 'slow_end' (concurrent execution)
```

### Pattern 4: Dynamic Handler Registration

```python
# noqa:validation
bus = EventBus()

# Plugin system with module-level handlers (weakref-compatible)
plugin_registry = {}

async def plugin_handler(plugin_id: str, **data):
    """Module-level function for stable weakref."""
    print(f"{plugin_id}: {data}")

# Register plugins dynamically
for i in range(3):
    plugin_id = f"P{i}"
    plugin_registry[plugin_id] = plugin_id
    # Subscribe module-level function (weakref-stable)
    bus.subscribe("plugin.event", lambda pid=plugin_id, **d: plugin_handler(pid, **d))

# Broadcast to all plugins
await bus.emit("plugin.event", message="test")
# Output:
# P0: {'message': 'test'}
# P1: {'message': 'test'}
# P2: {'message': 'test'}

print(bus.handler_count("plugin.event"))  # 3
```

**Note on Bound Methods**: EventBus uses `weakref.ref()` which creates dead references for bound methods. Use module-level functions or keep strong references to objects. For bound method support, consider using Broadcaster which handles `WeakMethod` internally.

---

## Common Pitfalls

### Pitfall 1: Forgetting Async Handlers

**Issue**: Subscribing sync function (type error at runtime)

```python
def sync_handler(**kwargs):  # Not async!
    print(kwargs)

bus.subscribe("test", sync_handler)
await bus.emit("test", value=42)  # TypeError: object is not awaitable
```

**Solution**: ALL handlers MUST be async:

```python
async def async_handler(**kwargs):  # Correct
    print(kwargs)

bus.subscribe("test", async_handler)
```

### Pitfall 2: Weakref Death Before Emit

**Issue**: Lambda or local function garbage collected before emit

```python
bus.subscribe("test", lambda **kw: print(kw))
# Lambda dies immediately (no strong reference)

await bus.emit("test", value=42)  # No output (weakref dead)
```

**Solution**: Keep strong reference or use module-level function:

```python
# Option 1: Keep reference
handler = lambda **kw: print(kw)
bus.subscribe("test", handler)

# Option 2: Use module function or method
async def log_event(**kw):
    print(kw)

bus.subscribe("test", log_event)
```

### Pitfall 3: Expecting Sequential Execution

**Issue**: Assuming handlers execute in order (they run concurrently)

```python
# noqa:validation
results = []

async def first(**kw):
    results.append("first")

async def second(**kw):
    results.append("second")

bus.subscribe("test", first)
bus.subscribe("test", second)

await bus.emit("test")
print(results)  # Could be ['second', 'first'] or ['first', 'second']
```

**Solution**: Don't rely on handler order. Use explicit orchestration if order matters.

### Pitfall 4: Exception Handling Expectations

**Issue**: Expecting exceptions to propagate (they don't)

```python
async def failing_handler(**kw):
    raise ValueError("Crash")

bus.subscribe("test", failing_handler)

await bus.emit("test")  # No exception raised (suppressed)
```

**Solution**: Understand fire-and-forget semantics. Add logging inside handlers if needed:

```python
async def robust_handler(**kw):
    try:
        # Handler logic
        pass
    except Exception as e:
        logger.error(f"Handler failed: {e}")
```

---

## Protocol Implementations

EventBus does NOT implement lionherd-core protocols (Observable, Serializable, etc.). It is a standalone utility class for pub/sub patterns.

**Rationale**: EventBus manages runtime subscriptions and handlers (not serializable state). It's designed as lightweight infrastructure, not domain objects.

---

## See Also

- [`Broadcaster`](broadcaster.md): Singleton pub/sub for global event channels
- [`Event`](event.md): Base event class for lifecycle tracking
- [Pub/Sub Patterns Notebook](../../../notebooks/references/broadcaster_eventbus.ipynb): Comprehensive examples

---

## Examples

### Example 1: Basic Topic Subscription

```python
# noqa:validation
from lionherd_core.base.eventbus import EventBus

bus = EventBus()
events = []

async def log_start(node_id: str):
    events.append(f"START: {node_id}")

async def log_complete(node_id: str, duration: float):
    events.append(f"COMPLETE: {node_id} ({duration}s)")

# Subscribe to different topics
bus.subscribe("node.start", log_start)
bus.subscribe("node.complete", log_complete)

# Emit events - handlers only receive matching topics
await bus.emit("node.start", node_id="n1")
await bus.emit("node.complete", node_id="n1", duration=0.5)
await bus.emit("node.start", node_id="n2")

print(events)
# Output: ['START: n1', 'COMPLETE: n1 (0.5s)', 'START: n2']
```

### Example 2: Concurrent Handler Execution

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep

bus = EventBus()
execution_log = []

async def slow_handler():
    execution_log.append("slow_start")
    await sleep(0.02)
    execution_log.append("slow_end")

async def fast_handler():
    execution_log.append("fast")

bus.subscribe("test", slow_handler)
bus.subscribe("test", fast_handler)

await bus.emit("test")

print(execution_log)
# Output: ['slow_start', 'fast', 'slow_end']
# 'fast' completes before 'slow_end' (concurrent execution)
```

### Example 3: Exception Isolation

```python
# noqa:validation
bus = EventBus()
results = []

async def failing_handler(value: int):
    raise ValueError(f"Failed on {value}")

async def working_handler(value: int):
    results.append(f"Success: {value}")

bus.subscribe("test", failing_handler)
bus.subscribe("test", working_handler)

# Emit doesn't raise - exceptions suppressed
await bus.emit("test", value=42)

print(results)
# Output: ['Success: 42']
# failing_handler exception suppressed (fire-and-forget)
```

### Example 4: Multi-Topic Event Routing

```python
# noqa:validation
bus = EventBus()
logs = []

async def log_all(event_type: str, **kwargs):
    logs.append(f"[{event_type}] {kwargs}")

# Subscribe to multiple topics
bus.subscribe("request.start", log_all)
bus.subscribe("request.complete", log_all)
bus.subscribe("request.error", log_all)

# Emit to different topics
await bus.emit("request.start", event_type="start", request_id="r1")
await bus.emit("request.complete", event_type="complete", request_id="r1", duration=0.15)
await bus.emit("request.error", event_type="error", request_id="r2", error="timeout")

for log in logs:
    print(log)
# Output:
# [start] {'request_id': 'r1'}
# [complete] {'request_id': 'r1', 'duration': 0.15}
# [error] {'request_id': 'r2', 'error': 'timeout'}
```

---

## Summary

EventBus provides **instance-based pub/sub with topic routing** for observability:

1. **Instance-Based**: Each EventBus has independent subscription registry
2. **Topic Routing**: Handlers only receive events for subscribed topics
3. **Concurrent Execution**: Handlers run concurrently via `gather()`
4. **Weakref Cleanup**: Automatic handler removal when garbage collected
5. **Exception Isolation**: Handler exceptions suppressed (fire-and-forget)

**Trade-offs:**

- ✅ Flexible topic routing (selective handler execution)
- ✅ Concurrent handler execution (performance for I/O-bound)
- ✅ Per-instance isolation (independent event channels)
- ✅ Weakref auto-cleanup (prevents memory leaks)
- ❌ Higher memory overhead than Broadcaster (per-instance state)
- ❌ Unpredictable handler order (concurrent execution)
- ❌ Requires async handlers (no sync callback support)

**When to use**: Observability patterns (metrics, tracing, logging) where topic filtering and concurrent handler execution are critical.

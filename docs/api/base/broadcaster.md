# Broadcaster

> Singleton pub/sub pattern for O(1) memory overhead event broadcasting with automatic weakref cleanup.

---

## Overview

**Broadcaster** provides a singleton-based publish-subscribe pattern optimized for application-wide event notifications. All instances of a Broadcaster subclass share the same subscription registry at the class level, resulting in O(1) memory overhead regardless of instance count.

**Key Features:**

- **Singleton Pattern**: One subscriber registry per subclass (class-level state)
- **Weakref Cleanup**: Automatic subscriber removal when callback objects are garbage collected
- **Mixed Async/Sync**: Supports both sync and async callback functions
- **Exception Isolation**: Callback exceptions logged, not propagated (fire-and-forget)
- **Type Safety**: Enforces event type validation at broadcast time

**Use Cases:**

- Application-wide notifications (shutdown, config reload, system events)
- Cross-cutting concerns with minimal memory footprint
- Scenarios where all components need to observe same event stream
- Long-running services with dynamic tenant/agent lifecycles

---

## When to Use Broadcaster

**✅ Use Broadcaster when:**

- You need a global event channel shared across all instances
- Memory overhead must be minimal (O(1) per subclass)
- All subscribers should receive same event stream (no topic filtering)
- Automatic cleanup of dead subscribers is critical (weakref)

**❌ Avoid Broadcaster when:**

- You need per-instance event channels (use EventBus)
- Topic-based routing is required (use EventBus)
- You need to await handler completion (Broadcaster is fire-and-forget)
- Multiple independent event streams needed (use multiple EventBus instances)

---

## Class Signature

```python
class Broadcaster:
    """Singleton pub/sub for O(1) memory event broadcasting."""

    # ClassVars (shared across all instances)
    _instance: ClassVar[Broadcaster | None] = None
    _subscribers: ClassVar[list[weakref.ref[Callable]]] = []
    _event_type: ClassVar[type]

    def __new__(cls) -> Broadcaster:
        """Return singleton instance (creates once per subclass)."""
        ...
```

---

## ClassVars (Shared State)

| ClassVar | Type | Description |
|----------|------|-------------|
| `_instance` | `Broadcaster \| None` | Singleton instance (one per subclass) |
| `_subscribers` | `list[weakref.ref[Callable]]` | Weak references to subscriber callbacks |
| `_event_type` | `type` | Expected event type for validation |

**Important**: ClassVars are shared at the **subclass level**, not across all Broadcaster subclasses. Each subclass has its own singleton instance and subscriber registry.

---

## Methods

### Category 1: Subscription Management

#### `subscribe(callback)`

Add subscriber callback to class-level registry.

**Signature:**

```python
@classmethod
def subscribe(
    cls,
    callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]
) -> None: ...
```

**Parameters:**

- `callback` (Callable): Sync or async function accepting event as single argument

**Behavior:**

- Stores callback as weakref for automatic cleanup
- Uses `WeakMethod` for bound methods, `weakref.ref` for regular callables
- Deduplicates: If callback already subscribed, no-op (identity comparison)
- Thread-safe: No explicit locking (relies on GIL for list operations)

**Examples:**

```python
from lionherd_core.base.broadcaster import Broadcaster
from lionherd_core.base.event import Event

class ShutdownEvent(Event):
    reason: str = ""

class ShutdownBroadcaster(Broadcaster):
    _event_type = ShutdownEvent
    _subscribers = []  # Must redeclare for subclass isolation
    _instance = None   # Must redeclare for subclass singleton

# Sync callback
def on_shutdown(event):
    print(f"Shutting down: {event.reason}")

ShutdownBroadcaster.subscribe(on_shutdown)

# Async callback
async def async_shutdown(event):
    await cleanup_resources()
    print(f"Async cleanup for: {event.reason}")

ShutdownBroadcaster.subscribe(async_shutdown)
```

#### `unsubscribe(callback)`

Remove subscriber callback from registry.

**Signature:**

```python
@classmethod
def unsubscribe(
    cls,
    callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]
) -> None: ...
```

**Parameters:**

- `callback` (Callable): Previously subscribed callback to remove

**Behavior:**

- Removes weakref pointing to this callback (identity comparison)
- No-op if callback not found
- Does NOT clean up dead weakrefs as side effect

**Examples:**

```python
def temp_handler(event):
    print("Temporary handler")

ShutdownBroadcaster.subscribe(temp_handler)
# ... later ...
ShutdownBroadcaster.unsubscribe(temp_handler)
```

#### `get_subscriber_count()`

Get number of live subscribers (excludes garbage-collected callbacks).

**Signature:**

```python
@classmethod
def get_subscriber_count(cls) -> int: ...
```

**Returns:**

- `int`: Count of live subscribers after cleanup

**Side Effects:**

- **Mutates state**: Cleans up dead weakrefs as side effect
- This is intentional: ensures count reflects actual live subscribers

**Examples:**

```python
count = ShutdownBroadcaster.get_subscriber_count()
print(f"Active subscribers: {count}")
```

### Category 2: Event Broadcasting

#### `broadcast(event)` (async)

Broadcast event to all live subscribers.

**Signature:**

```python
@classmethod
async def broadcast(cls, event: Any) -> None: ...
```

**Parameters:**

- `event` (Any): Event instance matching `_event_type`

**Raises:**

- `ValueError`: If event type doesn't match `_event_type`

**Behavior:**

1. Validates event type against `_event_type`
2. Cleans up dead weakrefs (lazy cleanup)
3. Executes each live callback:
   - Async callbacks: awaited
   - Sync callbacks: executed directly
4. Exceptions caught and logged (fire-and-forget semantics)

**Concurrency:**

- Sequential execution (not concurrent)
- Each callback waits for previous to complete
- Use EventBus for concurrent handler execution

**Examples:**

```python
# Broadcast event
event = ShutdownEvent(reason="maintenance")
await ShutdownBroadcaster.broadcast(event)

# Type validation
class OtherEvent(Event):
    pass

await ShutdownBroadcaster.broadcast(OtherEvent())  # Raises ValueError
```

### Category 3: Internal (Advanced)

#### `_cleanup_dead_refs()`

Remove garbage-collected callbacks and return live callbacks.

**Signature:**

```python
@classmethod
def _cleanup_dead_refs(
    cls
) -> list[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]: ...
```

**Returns:**

- `list[Callable]`: Live callback callables (weakrefs resolved)

**Side Effects:**

- **Mutates ClassVar**: Updates `_subscribers` in-place via slice assignment
- Uses `cls._subscribers[:] = alive_refs` to maintain ClassVar identity

**Internal Use Only**: Called automatically by `broadcast()` and `get_subscriber_count()`.

---

## Design Rationale

### Singleton Pattern for O(1) Memory

**Why singleton per subclass?**

1. **Shared Subscription Registry**: All instances observe same event stream
2. **O(1) Memory**: Subscriber list independent of instance count
3. **Predictable Behavior**: No confusion about which instance receives events

**Implementation**: `__new__` returns cached instance per subclass via `cls._instance`.

**Trade-off**: Less flexible than instance-based (EventBus), but far more memory-efficient.

### Weakref for Automatic Cleanup

**Why weakref instead of strong references?**

1. **Memory Leaks Prevention**: Long-running services with dynamic agents/tenants
2. **Implicit Cleanup**: No manual unsubscribe needed when object dies
3. **Lazy Collection**: Dead refs cleaned during normal operations (broadcast/count)

**Pattern**:

- Bound methods: `weakref.WeakMethod` (special handling for `__self__`)
- Regular callables: `weakref.ref`

**Trade-off**: Slight complexity increase, but prevents production memory leaks.

### Exception Isolation (Fire-and-Forget)

**Why log exceptions instead of raising?**

1. **Decoupling**: One failing subscriber doesn't break others
2. **Observability**: Exceptions logged with full traceback for debugging
3. **Fire-and-Forget**: Broadcaster doesn't await results or handle errors

**Implementation**: `try/except` around each callback in `broadcast()`.

**Trade-off**: Less control over error handling, but more robust in production.

### ClassVar In-Place Update

**Why `cls._subscribers[:] = alive_refs` instead of `cls._subscribers = alive_refs`?**

1. **Identity Preservation**: Maintains ClassVar identity across subclasses
2. **Subclass Safety**: Each subclass keeps its own subscription list
3. **Reference Stability**: External code holding reference to list stays in sync

**Technical Detail**: Slice assignment mutates list in-place without rebinding name.

---

## Usage Patterns

### Pattern 1: Application-Wide Shutdown

```python
# noqa:validation
from lionherd_core.base.broadcaster import Broadcaster
from lionherd_core.base.event import Event

class ShutdownEvent(Event):
    reason: str = ""
    grace_period: float = 5.0

class ShutdownBroadcaster(Broadcaster):
    _event_type = ShutdownEvent
    _subscribers = []
    _instance = None

# Service components subscribe
class DatabaseService:
    def __init__(self):
        ShutdownBroadcaster.subscribe(self.on_shutdown)

    async def on_shutdown(self, event):
        await self.close_connections()
        print(f"DB closed: {event.reason}")

class CacheService:
    def __init__(self):
        ShutdownBroadcaster.subscribe(self.on_shutdown)

    async def on_shutdown(self, event):
        await self.flush()
        print(f"Cache flushed: {event.reason}")

# Application broadcasts shutdown
db = DatabaseService()
cache = CacheService()

await ShutdownBroadcaster.broadcast(
    ShutdownEvent(reason="maintenance", grace_period=10.0)
)
```

### Pattern 2: Config Reload Propagation

```python
# noqa:validation
class ConfigChangeEvent(Event):
    config_key: str = ""
    new_value: object = None

class ConfigBroadcaster(Broadcaster):
    _event_type = ConfigChangeEvent
    _subscribers = []
    _instance = None

# Multiple components react to config changes
def update_rate_limit(event):
    if event.config_key == "api.rate_limit":
        set_rate_limit(event.new_value)

async def update_cache_ttl(event):
    if event.config_key == "cache.ttl":
        await reconfigure_cache(event.new_value)

ConfigBroadcaster.subscribe(update_rate_limit)
ConfigBroadcaster.subscribe(update_cache_ttl)

# Config service broadcasts changes
await ConfigBroadcaster.broadcast(
    ConfigChangeEvent(config_key="api.rate_limit", new_value=1000)
)
```

### Pattern 3: Weakref Auto-Cleanup

```python
class Handler:
    def __init__(self, name):
        self.name = name
        # Subscribe bound method (weakref auto-cleanup)
        ShutdownBroadcaster.subscribe(self.on_event)

    def on_event(self, event):
        print(f"{self.name} received: {event.reason}")

# Create and use handlers
h1 = Handler("Service1")
h2 = Handler("Service2")

print(ShutdownBroadcaster.get_subscriber_count())  # 2

# Delete handler - weakref auto-cleanup on next operation
del h1
import gc
gc.collect()

# Next operation cleans up dead weakrefs
print(ShutdownBroadcaster.get_subscriber_count())  # 1 (h1 auto-removed)
```

### Pattern 4: Mixed Sync/Async Handlers

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep

# Sync handler (immediate execution)
def sync_log(event):
    print(f"SYNC: {event.reason}")

# Async handler (awaited execution)
async def async_cleanup(event):
    await sleep(0.1)  # Simulate async work
    print(f"ASYNC: {event.reason}")

ShutdownBroadcaster.subscribe(sync_log)
ShutdownBroadcaster.subscribe(async_cleanup)

# Both execute correctly (async awaited, sync executed directly)
await ShutdownBroadcaster.broadcast(ShutdownEvent(reason="update"))
```

---

## Common Pitfalls

### Pitfall 1: Expecting Instance-Level Isolation

**Issue**: Subscribing to one instance affects all instances (singleton pattern)

```python
b1 = ShutdownBroadcaster()
b2 = ShutdownBroadcaster()

b1.subscribe(handler1)
# b2 also has handler1 subscribed (same instance!)
```

**Solution**: Understand that `b1 is b2` (singleton). Use EventBus for instance-level isolation.

### Pitfall 2: Weakref Death Before Broadcast

**Issue**: Lambda or local function garbage collected before broadcast

```python
ShutdownBroadcaster.subscribe(lambda e: print(e.reason))
# Lambda dies immediately (no strong reference)

await ShutdownBroadcaster.broadcast(event)  # No output (weakref dead)
```

**Solution**: Keep strong reference to callback or use module-level function:

```python
# Option 1: Keep reference
handler = lambda e: print(e.reason)
ShutdownBroadcaster.subscribe(handler)

# Option 2: Use module function
def log_event(e):
    print(e.reason)

ShutdownBroadcaster.subscribe(log_event)
```

### Pitfall 3: Assuming Concurrent Execution

**Issue**: Expecting handlers to run concurrently (they execute sequentially)

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep

async def slow_handler(event):
    await sleep(1.0)  # Blocks next handler

async def fast_handler(event):
    print("Fast")  # Waits for slow_handler to complete

await ShutdownBroadcaster.broadcast(event)  # Sequential execution
```

**Solution**: Use EventBus for concurrent handler execution via lionherd's `gather()`.

### Pitfall 4: Relying on Execution Order

**Issue**: Subscriber execution order is insertion order, but not guaranteed stable

```python
ShutdownBroadcaster.subscribe(handler1)
ShutdownBroadcaster.subscribe(handler2)

# handler1 executes before handler2 (insertion order)
# BUT: Dead weakref cleanup can reorder list
```

**Solution**: Don't rely on execution order. Use explicit orchestration if order matters.

---

## Protocol Implementations

Broadcaster does NOT implement lionherd-core protocols (Observable, Serializable, etc.). It is a standalone utility class for pub/sub patterns.

**Rationale**: Broadcaster manages class-level state (ClassVars), which doesn't fit Element-based protocol patterns designed for instance state.

---

## See Also

- [`EventBus`](eventbus.md): Instance-based pub/sub with topic routing and concurrent handlers
- [`Event`](event.md): Base event class for lifecycle tracking
- [Pub/Sub Patterns Notebook](../../../notebooks/references/broadcaster_eventbus.ipynb): Comprehensive examples

---

## Examples

### Example 1: Basic Subscription & Broadcasting

```python
# noqa:validation
from lionherd_core.base.broadcaster import Broadcaster
from lionherd_core.base.event import Event
from lionherd_core.libs.concurrency import sleep

# Define event type
class SystemEvent(Event):
    message: str = ""

# Define broadcaster subclass
class SystemBroadcaster(Broadcaster):
    _event_type = SystemEvent
    _subscribers = []
    _instance = None

# Subscribe handlers
received = []

def handler1(event):
    received.append(f"H1: {event.message}")

async def handler2(event):
    await sleep(0.01)
    received.append(f"H2: {event.message}")

SystemBroadcaster.subscribe(handler1)
SystemBroadcaster.subscribe(handler2)

# Broadcast
await SystemBroadcaster.broadcast(SystemEvent(message="test"))

print(received)
# Output: ['H1: test', 'H2: test']
```

### Example 2: Automatic Weakref Cleanup

```python
class Service:
    def __init__(self, name):
        self.name = name
        SystemBroadcaster.subscribe(self.handle)

    def handle(self, event):
        print(f"{self.name}: {event.message}")

# Create services
s1 = Service("A")
s2 = Service("B")

print(SystemBroadcaster.get_subscriber_count())  # 2

# Delete service - auto-cleanup
del s1
import gc
gc.collect()

print(SystemBroadcaster.get_subscriber_count())  # 1 (s1 auto-removed)
```

### Example 3: Exception Isolation

```python
def failing_handler(event):
    raise RuntimeError("Crash!")

def working_handler(event):
    print(f"Success: {event.message}")

SystemBroadcaster.subscribe(failing_handler)
SystemBroadcaster.subscribe(working_handler)

# Broadcast continues despite exception (logged, not raised)
await SystemBroadcaster.broadcast(SystemEvent(message="robust"))
# Output: "Success: robust" (failing_handler logged as error)
```

---

## Summary

Broadcaster provides **singleton pub/sub** with automatic memory management:

1. **Singleton**: One subscriber registry per subclass (O(1) memory)
2. **Weakref**: Automatic cleanup when callback objects garbage collected
3. **Mixed Sync/Async**: Supports both callback types
4. **Fire-and-Forget**: Exceptions logged, not propagated
5. **Type Safety**: Event type validation at broadcast time

**Trade-offs:**

- ✅ Minimal memory overhead (class-level state)
- ✅ Automatic cleanup (weakref prevents leaks)
- ✅ Simple API (subscribe/broadcast/unsubscribe)
- ❌ No topic filtering (use EventBus)
- ❌ Sequential execution (use EventBus for concurrency)
- ❌ No instance isolation (singleton per subclass)

**When to use**: Application-wide notifications where all components need same event stream and memory efficiency is critical.

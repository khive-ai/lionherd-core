# Event

> Async execution with observable lifecycle tracking, idempotency, and retry support.

---

## Overview

**Event** is the foundation for trackable async operations. Events manage
execution lifecycle, capture responses/errors, measure duration, and enable
retry patterns.

**Core Components**:

- **EventStatus**: Enum with 7 lifecycle states (PENDING → terminal)
- **Execution**: Dataclass tracking status, duration, response, error, retryable
- **Event**: Base class with `_invoke()` for custom logic, `invoke()` for
  execution

**Thread Safety**: All `invoke()` calls are synchronized via async lock.

---

## When to Use Event

**Use Event when you need:**

- **Observable execution**: Track status transitions, duration, response/error
- **Idempotency**: Multiple concurrent invocations return same cached result
- **Retry logic**: `as_fresh_event()` creates fresh instance for retries
- **Timeout handling**: Optional timeout with automatic cancellation
- **Error capture**: Exceptions stored in execution state (no propagation)
- **Serializable state**: Full execution history for logging/monitoring

**Don't use Event when:**

- Simple function call is sufficient → use async function directly
- No need for state tracking → use coroutine
- Synchronous operation → use regular function

---

## Class Signatures

### EventStatus

```python
class EventStatus(Enum):
    """Event execution status states."""

    PENDING = "pending"       # Not yet started
    PROCESSING = "processing" # Currently executing
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Execution failed with error
    CANCELLED = "cancelled"   # Interrupted by timeout or cancellation
    SKIPPED = "skipped"       # Bypassed due to condition
    ABORTED = "aborted"       # Pre-validation rejected, never started
```

**Terminal States**: COMPLETED, FAILED, CANCELLED, SKIPPED, ABORTED

**Active States**: PENDING, PROCESSING

---

### Execution

```python
@dataclass(slots=True)
class Execution:
    """Execution state (status, duration, response, error, retryable)."""

    status: EventStatus = EventStatus.PENDING
    duration: MaybeUnset[float] = Unset
    response: MaybeSentinel[Any] = Unset
    error: MaybeUnset[BaseException] | None = Unset
    retryable: MaybeUnset[bool] = Unset
```

**Sentinel Semantics**:

- `Unset`: Value not available (event pending/failed without response)
- `None`: Legitimate null value (event completed successfully with null result)

---

### Event

```python
class Event(Element):
    """Base event with lifecycle tracking and execution state."""
```

**Generic Parameters**: None (Event is not generic)

**Inheritance**: Extends [Element](element.md)

---

## Constructor

```python
def __init__(
    self,
    timeout: float | None = None,
    **data: Any,  # id, created_at, metadata
) -> None: ...
```

**Parameters**:

- `timeout` (float | None): Optional execution timeout in seconds. Must be
  positive and finite. Default: None (no timeout)
- `**data`: Additional Element fields (id, created_at, metadata)

**Example**:

```python
from lionherd_core.base import Event

class APICall(Event):
    url: str = ""

    async def _invoke(self):
        # Custom logic here
        return {"status": "ok"}

# Create with 5-second timeout
event = APICall(url="https://api.example.com", timeout=5.0)
```

---

## Attributes

### Execution State

#### `execution`

```python
execution: Execution
```

Execution state tracking status, duration, response, error, and retryable flag.

**Type:** Execution dataclass

**Access:** Pydantic Field (mutable)

**Default:** `Execution()` (PENDING status, all fields Unset)

#### `timeout`

```python
timeout: float | None
```

Optional execution timeout in seconds. If exceeded, raises
`LionherdTimeoutError` and sets status to CANCELLED.

**Type:** float | None

**Validation**: Must be positive and finite (raises ValueError if not)

**Serialization**: Excluded from serialization (`exclude=True`)

### Read-Only Properties

#### `status`

```python
@property
def status(self) -> EventStatus: ...

@status.setter
def status(self, val: EventStatus | str) -> None: ...
```

Get/set execution status. Setter accepts EventStatus enum or string value.

**Type:** EventStatus

**Access:** Property with setter

**Example:**

```python
print(event.status)  # EventStatus.PENDING
event.status = "processing"  # Accepts string
event.status = EventStatus.COMPLETED  # Accepts enum
```

#### `response`

```python
@property
def response(self) -> Any: ...
```

Get execution response (read-only property).

**Type:** Any (MaybeSentinel[Any] internally)

**Access:** Read-only property (no setter)

**Sentinel Handling**:

- Returns `Unset` if event pending/failed without response
- Returns `None` if event completed with null value
- Returns result value if event completed successfully

#### `request`

```python
@property
def request(self) -> dict: ...
```

Get request info (override in subclasses for custom request data).

**Type:** dict

**Default:** `{}`

---

## Methods

### Execution

#### `_invoke()`

Abstract method - implement in subclasses

```python
async def _invoke(self) -> Any
```

Execute event logic. Subclasses implement custom execution here.

**Returns:** Any - Execution result

**Raises**: NotImplementedError if not overridden

**Example**:

```python
class DataFetch(Event):
    source: str = "db"

    async def _invoke(self):
        # Custom implementation
        if self.source == "db":
            return await fetch_from_db()
        else:
            return await fetch_from_api()
```

**Design Note**: This is the extension point. `invoke()` handles lifecycle, you
implement business logic here.

---

#### `invoke()`

Execute with lifecycle management (idempotent)

```python
@final
@async_synchronized
async def invoke(self) -> Any
```

Execute event with status tracking, timing, and error capture. Multiple
concurrent calls execute `_invoke()` exactly once and return the same cached
result.

**Returns:** Any - Execution result (cached after first call)

**Idempotency**: Once executed, subsequent calls return cached result without
re-execution. Status must be PENDING for execution to occur.

**Timeout Handling**: If timeout specified and exceeded:

- Raises `LionherdTimeoutError` internally
- Sets status to CANCELLED
- Sets retryable to True
- Returns None

**Error Handling**: If `_invoke()` raises exception:

- Captures error in `execution.error`
- Sets status to FAILED
- Sets retryable based on error type (LionherdError.retryable or True)
- Returns None

**ExceptionGroup Support**: If multiple errors occur:

- Captures all in ExceptionGroup
- Retryable = True only if ALL exceptions retryable

**Thread Safety**: Synchronized with `@async_synchronized` decorator (async
lock)

**Time Complexity:** O(1) after first execution (cached result lookup)

**Example**:

```python
from lionherd_core.libs.concurrency import gather

# First call - executes _invoke()
result1 = await event.invoke()

# Subsequent calls - return cached result
result2 = await event.invoke()
result3 = await event.invoke()

assert result1 is result2 is result3  # Same cached result

# Concurrent calls also return same result
results = await gather(
    event.invoke(),
    event.invoke(),
    event.invoke(),
)
assert results[0] is results[1] is results[2]
```

**Retry Pattern**: To retry after FAILED status:

```python
if event.execution.retryable:
    fresh = event.as_fresh_event()
    result = await fresh.invoke()  # Fresh execution
```

---

#### `stream()`

Streaming execution (override if supported)

```python
async def stream(self) -> Any
```

Stream execution results. Override in subclasses that support streaming.

**Returns:** Any - Stream result

**Raises**: NotImplementedError if not overridden

**Design Note**: For events that produce incremental results (e.g., LLM
streaming responses).

---

#### `as_fresh_event()`

Clone with reset execution state

```python
def as_fresh_event(self, copy_meta: bool = False) -> Event
```

Create fresh event instance with new ID, PENDING status, and reset execution
state. Enables retry pattern after FAILED/CANCELLED execution.

**Parameters**:

- `copy_meta` (bool): If True, copy metadata from original. Default: False

**Returns:** Event - New instance with fresh execution state

**Behavior**:

- Generates new UUID for fresh event
- Resets execution to PENDING status
- Preserves timeout if set
- Copies metadata if `copy_meta=True`
- Adds `metadata["original"]` with original event's id and created_at

**Time Complexity:** O(1) for field copying

**Example**:

```python
# Original event fails
original = APICall(url="...", timeout=5.0)
result = await original.invoke()
print(original.status)  # FAILED

# Create fresh event for retry
if original.execution.retryable:
    fresh = original.as_fresh_event(copy_meta=True)

    print(fresh.id != original.id)  # True - new UUID
    print(fresh.status)  # PENDING - reset status
    print(fresh.timeout)  # 5.0 - preserved
    print(fresh.metadata["original"]["id"])  # original.id

    # Retry execution
    result = await fresh.invoke()
```

**Pattern**: Use for retry strategies:

1. Check `event.execution.retryable`
2. Create fresh event: `fresh = event.as_fresh_event()`
3. Retry: `await fresh.invoke()`

---

### Serialization

#### `to_dict()`

Serialize event to dictionary with full execution state (inherited from
Element).

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

**Parameters**: See [Element.to_dict()](element.md#to_dict)

**Returns:** dict[str, Any] with execution state serialized

**Execution Serialization**:

- `status`: Enum value as string
- `duration`: Float in seconds or None
- `response`: Serialized result or None (Unset → None)
- `error`: Dict with error/message/exceptions or None
- `retryable`: Bool or None (Unset → None)

**Example**:

```python
data = event.to_dict(mode="json")
# {
#   "id": "uuid-string",
#   "created_at": "2025-11-09T03:00:00.000000Z",
#   "metadata": {},
#   "execution": {
#     "status": "completed",
#     "duration": 0.123,
#     "response": {"result": "value"},
#     "error": null,
#     "retryable": false
#   }
# }
```

**Note**: `timeout` is excluded from serialization (`exclude=True` field
config).

---

## Design Rationale

### Idempotency via Status Check

**Why check status before execution?**

1. **Concurrent Safety**: Multiple concurrent `invoke()` calls execute
   `_invoke()` exactly once
2. **Deterministic**: Same event always returns same result (cached)
3. **Performance**: Avoid re-execution overhead after first call
4. **State Consistency**: Terminal states (COMPLETED/FAILED/CANCELLED) are
   immutable

**Implementation**: Status check at beginning of `invoke()`:

```python
if self.execution.status != EventStatus.PENDING:
    return self.execution.response  # Cached result
```

### Sentinel Semantics (Unset vs None)

**Why distinguish Unset from None?**

1. **Precision**: Unset = "no value available", None = "null value"
2. **Debugging**: Know if event never executed vs completed with null
3. **Retry Logic**: Check if response exists before deciding retry
4. **Serialization**: Both serialize to None, but status provides context

**Pattern**:

- PENDING event: `response is Unset` (never executed)
- FAILED event: `response is Unset` (no successful result)
- COMPLETED event: `response is None` (legitimate null) or `response = value`

### Error Capture Instead of Propagation

**Why store errors instead of raising?**

1. **Observability**: Full execution history in one object
2. **Retry Logic**: Check `execution.error` and `retryable` flag
3. **Serialization**: Include error details in logs/monitoring
4. **Idempotency**: Return None consistently for failures

**Exception**: `BaseException` (like CancelledError) is raised to preserve
cancellation semantics.

### Timeout → CANCELLED Status

**Why CANCELLED instead of FAILED for timeouts?**

1. **Semantic Clarity**: Timeout is cancellation, not logic error
2. **Retry Semantics**: Timeouts are always retryable
3. **Monitoring**: Distinguish timeout from application errors
4. **Consistency**: Matches anyio cancellation model

### ExceptionGroup Conservative Retryability

**Why require ALL exceptions retryable?**

1. **Safety**: One non-retryable error makes retry unsafe
2. **Parallel Failures**: ExceptionGroup from concurrent operations
3. **Conservative**: Avoid retry loops on persistent failures

**Pattern**: Loop through `eg.exceptions`, set `retryable = False` if any
`LionherdError` has `retryable=False`.

---

## Protocol Implementations

Event implements three core protocols:

### Invocable

**Method**: `invoke()` (async)

Execute the event and manage lifecycle transitions. This is the primary protocol implementation that makes Event a first-class async operation.

**Features**:

- Idempotent execution (repeated calls return cached result)
- Automatic status transitions (PENDING → PROCESSING → terminal)
- Timeout support with cancellation
- Error capture and retryable flag management

**See**: [`invoke()` method](#invoke) for full signature and behavior.

### Serializable

**Methods** (inherited from Element):

- `to_dict(mode='python'|'json'|'db', **kwargs)`: Dictionary serialization with three modes
- `to_json(pretty=False, sort_keys=False, decode=True, **kwargs)`: JSON string serialization

**Event-specific serialization**:

- Execution state included in serialization (status, duration, response, error, retryable)
- Timeout excluded from serialization (transient field marked with `exclude=True`)
- EventStatus serialized as string value (`"pending"`, `"completed"`, etc.)

**Polymorphism**: Automatically injects `lion_class` in metadata for subclass reconstruction.

### Observable

**Method** (inherited from Element): `id` property

UUID-based identity for tracking events across distributed systems.

**Usage**: Use `event.id` for correlation IDs in logs, distributed tracing, and debugging across async workflows.

---

## Usage Patterns

### Pattern 1: Simple Async Operation

```python
# noqa:validation
class DatabaseQuery(Event):
    query: str = ""

    async def _invoke(self):
        return await db.execute(self.query)

# Execute
event = DatabaseQuery(query="SELECT * FROM users", timeout=10.0)
result = await event.invoke()

# Check outcome
if event.status == EventStatus.COMPLETED:
    print(f"Rows: {len(result)}")
elif event.status == EventStatus.FAILED:
    print(f"Error: {event.execution.error}")
elif event.status == EventStatus.CANCELLED:
    print("Query timed out")
```

---

### Pattern 2: Retry Strategy with Backoff

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep

async def retry_with_backoff(event: Event, max_retries: int = 3) -> Any:
    """Retry event with exponential backoff."""
    current = event

    for attempt in range(max_retries):
        result = await current.invoke()

        if current.status == EventStatus.COMPLETED:
            return result

        if not current.execution.retryable:
            print(f"Non-retryable error: {current.execution.error}")
            return None

        if attempt < max_retries - 1:
            # Exponential backoff
            delay = 2 ** attempt
            print(f"Retry {attempt + 1}/{max_retries} after {delay}s")
            await sleep(delay)

            # Create fresh event for retry
            current = current.as_fresh_event(copy_meta=True)
            current.metadata["retry_attempt"] = attempt + 1

    print(f"Failed after {max_retries} retries")
    return None

# Usage
event = APICall(url="https://api.example.com", timeout=5.0)
result = await retry_with_backoff(event, max_retries=3)
```

---

### Pattern 3: Execution Monitoring

```python
import logging

async def execute_with_monitoring(event: Event) -> Any:
    """Execute event with comprehensive monitoring."""
    logger = logging.getLogger(__name__)

    # Pre-execution
    logger.info(f"Starting event {event.id}")
    start = time.time()

    # Execute
    result = await event.invoke()

    # Post-execution monitoring
    exec_data = event.execution.to_dict()
    logger.info(
        "Event completed",
        extra={
            "event_id": str(event.id),
            "status": exec_data["status"],
            "duration": exec_data["duration"],
            "retryable": exec_data["retryable"],
            "error": exec_data["error"],
        }
    )

    # Metrics
    metrics.record("event.duration", exec_data["duration"])
    metrics.increment(f"event.status.{exec_data['status']}")

    return result
```

---

### Pattern 4: Parallel Execution with Error Aggregation

```python
# noqa:validation
from lionherd_core.libs.concurrency import gather

async def execute_parallel(events: list[Event]) -> dict:
    """Execute events in parallel, aggregate results."""
    # Execute all in parallel
    await gather(*[e.invoke() for e in events])

    # Aggregate results
    results = {
        "completed": [],
        "failed": [],
        "cancelled": [],
    }

    for event in events:
        if event.status == EventStatus.COMPLETED:
            results["completed"].append({
                "id": str(event.id),
                "response": event.response,
                "duration": event.execution.duration,
            })
        elif event.status == EventStatus.FAILED:
            results["failed"].append({
                "id": str(event.id),
                "error": str(event.execution.error),
                "retryable": event.execution.retryable,
            })
        elif event.status == EventStatus.CANCELLED:
            results["cancelled"].append({
                "id": str(event.id),
                "timeout": event.timeout,
            })

    return results

# Usage
events = [
    APICall(url="https://api1.com", timeout=5.0),
    APICall(url="https://api2.com", timeout=5.0),
    APICall(url="https://api3.com", timeout=5.0),
]

results = await execute_parallel(events)
print(f"Completed: {len(results['completed'])}/{len(events)}")
```

---

## Common Pitfalls

### Pitfall 1: Calling invoke() on Completed Event

**Issue**: Expecting re-execution after event already completed.

```python
event = SimpleEvent(return_value=42)
result1 = await event.invoke()  # Executes _invoke(), returns 42

# ❌ WRONG: Expecting re-execution
result2 = await event.invoke()  # Returns cached 42 (no re-execution)
```

**Solution**: Use `as_fresh_event()` to create new instance for re-execution.

```python
# ✓ CORRECT: Fresh event for re-execution
fresh = event.as_fresh_event()
result2 = await fresh.invoke()  # Executes _invoke() again
```

---

### Pitfall 2: Confusing Unset with None

**Issue**: Checking `response is None` when should check `is Unset`.

```python
event = FailingEvent()
await event.invoke()

# ❌ WRONG: None check doesn't distinguish failure from null result
if event.response is None:
    print("Failed")  # Triggered for both FAILED and COMPLETED(null)
```

**Solution**: Check status or use `is Unset` for precision.

```python
# ✓ CORRECT: Check status
if event.status == EventStatus.FAILED:
    print(f"Failed: {event.execution.error}")

# ✓ CORRECT: Unset check for "no result"
from lionherd_core.types import Unset
if event.response is Unset:
    print("No result available")
```

---

### Pitfall 3: Not Checking Retryable Flag

**Issue**: Retrying non-retryable errors wastes resources.

```python
# ❌ WRONG: Blind retry
for _ in range(3):
    event = event.as_fresh_event()
    await event.invoke()
    if event.status == EventStatus.COMPLETED:
        break  # May retry non-retryable errors
```

**Solution**: Check `execution.retryable` before retry.

```python
# ✓ CORRECT: Check retryable
if event.execution.retryable:
    fresh = event.as_fresh_event()
    await fresh.invoke()
else:
    print(f"Non-retryable: {event.execution.error}")
```

---

### Pitfall 4: Modifying Execution State Directly

**Issue**: Changing `execution.status` manually breaks lifecycle contract.

```python
# ❌ WRONG: Manual status manipulation
event.execution.status = EventStatus.COMPLETED
event.execution.response = "fake_result"
```

**Solution**: Use `invoke()` for execution, `as_fresh_event()` for reset.

```python
# ✓ CORRECT: Let invoke() manage lifecycle
result = await event.invoke()  # Proper status transitions

# ✓ CORRECT: Reset via fresh event
fresh = event.as_fresh_event()  # New instance with PENDING status
```

---

## Examples

<!-- TODO: Add link to event.ipynb notebook when available (issue #115) -->

Comprehensive examples demonstrating:

1. Construction + Basic Invocation
2. Status Lifecycle Transitions
3. Success vs Failure Handling
4. Timeout Handling
5. Idempotency - Cached Results
6. Retry Pattern with `as_fresh_event()`
7. ExceptionGroup Support
8. Sentinel Handling (Unset vs None)

---

## Summary

**Event** provides production-grade async execution with:

**Key Features**:

- 7-state lifecycle (PENDING → PROCESSING → terminal)
- Idempotent execution (cached results after first call)
- Observable state (status, duration, response, error, retryable)
- Timeout support with automatic cancellation
- Error capture without propagation
- Retry pattern via `as_fresh_event()`
- ExceptionGroup support with conservative retryability
- Full serialization for logging/monitoring

**Common Patterns**:

- `await event.invoke()` - Execute with lifecycle management
- `if event.execution.retryable: fresh = event.as_fresh_event()` - Retry pattern
- `event.execution.to_dict()` - Serialize for monitoring
- `await gather(*[e.invoke() for e in events])` - Parallel execution (import from `lionherd_core.libs.concurrency`)

**Performance**:

- O(1) invoke after first call (cached result)
- Thread-safe concurrent invocations
- Minimal overhead for status tracking

**Design Principles**:

- Idempotency prevents duplicate execution
- Sentinel semantics distinguish "no value" from "null value"
- Error capture enables observability without disrupting control flow
- Status-based lifecycle provides clear execution contract

See `src/lionherd_core/base/event.py` for full implementation details.

---

## See Also

- [Element](element.md) - Base class with identity and serialization
- [Execution](#execution) - Execution state dataclass
- [EventStatus](#eventstatus) - Status enum

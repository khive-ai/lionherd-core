# Resource Tracker

> Synchronous resource leak detection and tracking system for debugging

## Overview

The resource tracker module provides **synchronous leak detection** for debugging resource management issues in lionherd applications. It uses weak references and finalizers to track live objects and detect when they're not being properly cleaned up.

**Key Capabilities:**

- **Weak Reference Tracking**: Automatically detects when tracked objects are garbage collected
- **Leak Detection**: Identifies resources that remain alive longer than expected
- **Thread-Safe Operations**: Uses `threading.Lock` for safe concurrent access
- **Metadata Capture**: Stores creation timestamp and optional classification
- **Global Singleton**: Centralized tracker accessible via module-level functions

**When to Use Resource Tracker:**

- **Debugging resource leaks**: Identify objects that aren't being cleaned up properly
- **Development/testing**: Track resource lifecycle during development
- **Memory profiling**: Understand which resources are still live
- **Cleanup verification**: Ensure cleanup logic properly releases resources

**When NOT to Use Resource Tracker:**

- **Production hot paths**: Uses blocking locks, not suitable for performance-critical code
- **Async code**: Blocking operations will block the event loop
- **High-frequency tracking**: Overhead of tracking every object creation may impact performance
- **Thread pool contexts**: Lock contention may degrade performance

**Warning:**

This tracker uses `threading.Lock` and is designed for **SYNC contexts only**. Do NOT call `track()`/`untrack()` from async code as it may block the event loop. This is intentional: `weakref.finalize` callbacks must be synchronous.

For debugging resource leaks, this overhead is acceptable as it's not used in hot paths.

## Module Contents

```python
from lionherd_core.libs.concurrency import (
    LeakInfo,           # Dataclass storing tracked object metadata
    LeakTracker,        # Class for tracking live objects
    track_resource,     # Track an object for leak detection
    untrack_resource,   # Remove object from tracking
)
```

## LeakInfo

Immutable dataclass storing metadata about a tracked object.

### Class Signature

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LeakInfo:
    """Metadata for a tracked object."""

    name: str           # Human-readable identifier
    kind: str | None    # Optional resource classification
    created_at: float   # Unix timestamp when tracking started
```

### Attributes

| Attribute    | Type         | Description                                    |
| ------------ | ------------ | ---------------------------------------------- |
| `name`       | `str`        | Human-readable identifier (auto-generated if not provided) |
| `kind`       | `str \| None` | Optional resource classification (e.g., "connection", "file") |
| `created_at` | `float`      | Unix timestamp when tracking started (via `time.time()`) |

### Examples

```python
from lionherd_core.libs.concurrency import LeakInfo
import time

# Create leak info manually
info = LeakInfo(
    name="database_connection_1",
    kind="connection",
    created_at=time.time()
)

print(info.name)        # "database_connection_1"
print(info.kind)        # "connection"
print(info.created_at)  # 1699438200.123456

# Frozen dataclass (immutable)
# info.name = "new_name"  # FrozenInstanceError
```

## LeakTracker

Thread-safe tracker for detecting resource leaks via weak references.

### Class Signature

```python
class LeakTracker:
    """Track live objects for leak detection.

    Warning:
        This tracker uses threading.Lock and is designed for SYNC contexts only.
        Do NOT call track()/untrack() from async code as it may block the event loop.
        This is intentional: weakref.finalize callbacks must be synchronous.

        For debugging resource leaks, this is acceptable as it's not used in hot paths.
    """

    def __init__(self) -> None: ...
```

### Attributes

| Attribute | Type                       | Description                                |
| --------- | -------------------------- | ------------------------------------------ |
| `_live`   | `dict[int, LeakInfo]`      | Internal registry mapping object IDs to metadata |
| `_lock`   | `threading.Lock`           | Thread synchronization lock                |

### Methods

#### `track()`

Track an object for leak detection using weak references.

**Signature:**

```python
def track(
    self,
    obj: object,
    *,
    name: str | None,
    kind: str | None,
) -> None: ...
```

**Parameters:**

- `obj` (object): Object to track for leaks
- `name` (str, optional): Human-readable identifier. If None, auto-generated as `f"obj-{id(obj)}"`
- `kind` (str, optional): Resource classification (e.g., "connection", "file", "lock")

**Returns:**

- None

**Examples:**

```python
from lionherd_core.libs.concurrency import LeakTracker

tracker = LeakTracker()

# Track with custom name
class DatabaseConnection:
    pass

conn = DatabaseConnection()
tracker.track(conn, name="db_conn_1", kind="connection")

# Track with auto-generated name
obj = object()
tracker.track(obj, name=None, kind="temp_object")
# Internally: name="obj-140234567890123"

# Check tracked objects
live_objects = tracker.live()
print(len(live_objects))  # 2
print(live_objects[0].name)  # "db_conn_1"
```

**Notes:**

- Automatically unracks object when garbage collected via `weakref.finalize`
- Thread-safe: uses internal lock for concurrent access
- Blocking operation: do NOT call from async code
- Creation timestamp captured via `time.time()`

#### `untrack()`

Manually remove an object from tracking.

**Signature:**

```python
def untrack(self, obj: object) -> None: ...
```

**Parameters:**

- `obj` (object): Object to stop tracking

**Returns:**

- None

**Examples:**

```python
from lionherd_core.libs.concurrency import LeakTracker

tracker = LeakTracker()

obj = object()
tracker.track(obj, name="temp", kind="test")

# Verify tracked
assert len(tracker.live()) == 1

# Manually untrack
tracker.untrack(obj)

# Verify removed
assert len(tracker.live()) == 0
```

**Notes:**

- Safe to call on untracked objects (no-op)
- Thread-safe operation
- Useful for explicit cleanup before garbage collection

#### `live()`

Get list of currently tracked objects.

**Signature:**

```python
def live(self) -> list[LeakInfo]: ...
```

**Returns:**

- list[LeakInfo]: List of metadata for all currently tracked objects

**Examples:**

```python
from lionherd_core.libs.concurrency import LeakTracker
import time

tracker = LeakTracker()

# Track multiple objects
obj1 = object()
obj2 = object()
tracker.track(obj1, name="obj1", kind="test")
time.sleep(0.1)  # Different timestamps
tracker.track(obj2, name="obj2", kind="test")

# Get live objects
live = tracker.live()
print(len(live))  # 2

for info in live:
    print(f"{info.name} ({info.kind}): created at {info.created_at}")
# obj1 (test): created at 1699438200.123456
# obj2 (test): created at 1699438200.223456

# Objects automatically removed when garbage collected
del obj1
import gc
gc.collect()

live = tracker.live()
print(len(live))  # 1 (obj1 removed)
```

**Notes:**

- Returns snapshot of current state (safe to modify returned list)
- Thread-safe operation
- Useful for leak detection reports and debugging

#### `clear()`

Remove all tracked objects from the registry.

**Signature:**

```python
def clear(self) -> None: ...
```

**Returns:**

- None

**Examples:**

```python
from lionherd_core.libs.concurrency import LeakTracker

tracker = LeakTracker()

# Track objects
tracker.track(object(), name="obj1", kind="test")
tracker.track(object(), name="obj2", kind="test")
assert len(tracker.live()) == 2

# Clear all
tracker.clear()
assert len(tracker.live()) == 0
```

**Notes:**

- Thread-safe operation
- Does NOT affect the tracked objects themselves (only removes from registry)
- Useful for test cleanup and resetting tracker state

## Module-Level Functions

### `track_resource()`

Track an object for leak detection using the global singleton tracker.

**Signature:**

```python
def track_resource(
    obj: object,
    name: str | None = None,
    kind: str | None = None,
) -> None: ...
```

**Parameters:**

- `obj` (object): Object to track for leaks
- `name` (str, optional): Human-readable identifier. Default: `None` (auto-generated)
- `kind` (str, optional): Resource classification. Default: `None`

**Returns:**

- None

**Examples:**

```python
from lionherd_core.libs.concurrency import track_resource

class FileHandle:
    def __init__(self, path: str):
        self.path = path
        track_resource(self, name=f"file:{path}", kind="file")

    def close(self):
        # File cleanup logic
        pass

# Create tracked resource
handle = FileHandle("/tmp/data.txt")

# Resource automatically untracked when garbage collected
del handle
```

**Warning:**

Do not call from async code - uses blocking `threading.Lock`. For leak detection/debugging only, not production hot paths.

**See Also:**

- `untrack_resource()`: Remove object from tracking
- `LeakTracker.track()`: Underlying implementation

### `untrack_resource()`

Remove an object from the global tracker.

**Signature:**

```python
def untrack_resource(obj: object) -> None: ...
```

**Parameters:**

- `obj` (object): Object to stop tracking

**Returns:**

- None

**Examples:**

```python
from lionherd_core.libs.concurrency import track_resource, untrack_resource

class ManagedResource:
    def __init__(self, name: str):
        self.name = name
        track_resource(self, name=name, kind="resource")

    def cleanup(self):
        # Explicit cleanup
        untrack_resource(self)
        # Resource cleanup logic...

res = ManagedResource("important_resource")
# ... use resource ...
res.cleanup()  # Explicitly untrack before garbage collection
```

**See Also:**

- `track_resource()`: Add object to tracking
- `LeakTracker.untrack()`: Underlying implementation

## Usage Patterns

### Basic Leak Detection

```python
from lionherd_core.libs.concurrency import track_resource

class Connection:
    def __init__(self, host: str):
        self.host = host
        track_resource(self, name=f"conn:{host}", kind="connection")

    def close(self):
        # Cleanup logic
        pass

# Create connections
conn1 = Connection("db.example.com")
conn2 = Connection("api.example.com")

# Connections are tracked automatically
# When they're deleted and garbage collected, they're automatically untracked

# Close and cleanup
conn1.close()
del conn1
import gc
gc.collect()

# Objects are automatically removed when garbage collected
# due to weakref.finalize callbacks
```

### Context Manager Integration

```python
from lionherd_core.libs.concurrency import track_resource, untrack_resource

class TrackedFile:
    """File handle with leak tracking."""

    def __init__(self, path: str):
        self.path = path
        self.file = open(path, 'r')
        track_resource(self, name=f"file:{path}", kind="file")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if hasattr(self, 'file'):
            self.file.close()
            untrack_resource(self)  # Explicit untrack on cleanup

# Usage with context manager
with TrackedFile("/tmp/data.txt") as f:
    data = f.read()
# Automatically untracked on exit
```

### Test Cleanup Verification

For testing, create isolated `LeakTracker` instances rather than using the global tracker:

```python
from lionherd_core.libs.concurrency import LeakTracker, track_resource
import pytest
import gc

@pytest.fixture
def leak_tracker():
    """Isolated leak tracker for testing."""
    tracker = LeakTracker()
    yield tracker
    # Auto-cleanup after test
    tracker.clear()

def test_resource_cleanup(leak_tracker):
    """Test that properly cleans up resources."""
    obj = object()
    leak_tracker.track(obj, name="test_obj", kind="test")

    # Verify tracked
    assert len(leak_tracker.live()) == 1

    # Cleanup
    del obj
    gc.collect()

    # Verify cleanup
    assert len(leak_tracker.live()) == 0
```

### Custom Leak Reporting

To implement custom leak reporting, create a helper function that wraps a `LeakTracker`:

```python
from lionherd_core.libs.concurrency import LeakTracker
import time

class TrackerWithReporting:
    """Wrapper for LeakTracker with reporting capabilities."""

    def __init__(self):
        self.tracker = LeakTracker()

    def track(self, obj, name=None, kind=None):
        """Track an object."""
        self.tracker.track(obj, name=name, kind=kind)

    def report(self) -> str:
        """Generate human-readable leak report."""
        leaks = self.tracker.live()

        if not leaks:
            return "No resource leaks detected."

        lines = [f"Resource Leaks Detected: {len(leaks)}"]

        # Sort by age (oldest first)
        sorted_leaks = sorted(leaks, key=lambda x: x.created_at)

        for info in sorted_leaks:
            age = time.time() - info.created_at
            kind_str = f" ({info.kind})" if info.kind else ""
            lines.append(f"  - {info.name}{kind_str}: {age:.2f}s old")

        return "\n".join(lines)

# Usage
reporter = TrackerWithReporting()
print(reporter.report())
# Resource Leaks Detected: 0
```

## Design Rationale

### Why Synchronous Only?

The tracker uses `threading.Lock` instead of async locks because:

1. **Finalizer Constraint**: `weakref.finalize` callbacks must be synchronous (no async support)
2. **Debugging Context**: Leak detection is a debugging tool, not production infrastructure
3. **Simplicity**: Synchronous locks avoid async context management complexity
4. **Thread Safety**: Works correctly in multi-threaded environments

For async code, track resources in synchronous context (e.g., `__init__`) and untrack in cleanup methods called from synchronous finalizers.

### Why Weak References?

Using `weakref.finalize` instead of manual tracking:

1. **Automatic Cleanup**: No need to manually untrack when objects are deleted
2. **Leak Detection**: Finalizers detect when objects are garbage collected
3. **No Circular References**: Weak references don't prevent garbage collection
4. **Correct Timing**: Finalizers run after object becomes unreachable

This makes the tracker reliable for detecting actual leaks (objects kept alive unintentionally).

### Why Global Singleton?

The module-level `_TRACKER` singleton provides:

1. **Centralized Tracking**: All resources tracked in one place
2. **Simple API**: Module-level functions don't require tracker instance management
3. **Test Inspection**: Easy to check global state for leak detection
4. **Consistent State**: Single source of truth for all tracked resources

Projects needing isolated tracking can create separate `LeakTracker` instances.

### Why Unix Timestamps?

Using `time.time()` (float Unix timestamp) instead of `datetime`:

1. **Performance**: Float comparison faster than datetime arithmetic
2. **Simplicity**: No timezone handling needed
3. **Age Calculation**: Easy to compute age via `time.time() - created_at`
4. **Serialization**: Float timestamps serialize trivially

For human-readable output, convert to datetime in reporting logic.

## Common Pitfalls

### Pitfall 1: Blocking Async Event Loop

**Issue**: Calling tracking functions from async code blocks the event loop.

```python
import asyncio
from lionherd_core.libs.concurrency import track_resource

async def bad_example():
    obj = object()
    # WRONG: Blocks event loop with threading.Lock
    track_resource(obj, name="async_obj", kind="test")
```

**Solution**: Track resources in synchronous context only (e.g., `__init__` methods, synchronous factory functions).

### Pitfall 2: Forgetting Garbage Collection

**Issue**: Objects remain tracked until garbage collection, not deletion.

```python
from lionherd_core.libs.concurrency import LeakTracker, track_resource
import gc

# Using isolated tracker for testing
tracker = LeakTracker()

obj = object()
tracker.track(obj, name="obj", kind="test")

# Object still tracked after del (GC hasn't run)
del obj
print(len(tracker.live()))  # May still be 1

# Force garbage collection
gc.collect()
print(len(tracker.live()))  # Now 0 (after GC)
```

**Solution**: Always call `gc.collect()` in tests before checking tracker state, as Python's garbage collector is not deterministic.

### Pitfall 3: Circular References Prevent Cleanup

**Issue**: Circular references prevent garbage collection and leak detection.

```python
from lionherd_core.libs.concurrency import track_resource

class Node:
    def __init__(self, name: str):
        self.name = name
        self.next = None
        track_resource(self, name=name, kind="node")

# Create circular reference
n1 = Node("node1")
n2 = Node("node2")
n1.next = n2
n2.next = n1  # Circular reference

del n1, n2
import gc
gc.collect()
# Both objects remain tracked (circular reference prevents GC)
```

**Solution**: Break circular references explicitly or use weak references for cycles.

### Pitfall 4: Performance Impact in Hot Paths

**Issue**: Tracking every object creation in performance-critical code.

```python
from lionherd_core.libs.concurrency import track_resource

def hot_path_function():
    for i in range(1_000_000):
        obj = object()
        track_resource(obj, name=f"obj-{i}", kind="temp")  # Expensive!
```

**Solution**: Only track long-lived resources or use tracking conditionally in debug builds.

## See Also

- **Related Modules**:
  - [Concurrency Primitives](primitives.md): Async primitives (Semaphore, Lock, Queue)
  - [Concurrency Patterns](patterns.md): High-level concurrency patterns
  - [Cancel Scopes](cancel.md): Structured cancellation for async tasks
<!-- TODO: Add links to user guides when available (issue #115) -->

## Examples

The resource tracker is best learned through the **Usage Patterns** section above. For comprehensive production examples, see the planned tutorials:

- **Tutorial: Database Connection Pool Leak Detection** - Connection pool with automatic leak tracking and reporting
- **Tutorial: File Handle Tracking** - File resource management with leak detection at exit
- **Tutorial: Lock Acquisition Debugging** - Detecting deadlocks and lock contention with acquisition tracking

These tutorials will be created as separate guides with detailed explanations of production patterns.

### Quick Testing Example

For testing scenarios, use isolated tracker instances as shown in the **Test Cleanup Verification** usage pattern above.

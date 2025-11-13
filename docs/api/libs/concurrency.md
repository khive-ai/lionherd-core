# Concurrency

> Async-first concurrency utilities for safe, structured concurrent operations

## Overview

The `lionherd_core.libs.concurrency` module provides a comprehensive suite of **async concurrency utilities** built on AnyIO for cross-backend compatibility. It offers primitives, patterns, and safety mechanisms for building reliable concurrent applications with structured concurrency principles.

**Design Philosophy:**

- **Structured Concurrency**: Task groups ensure all spawned tasks complete before scope exit
- **Cross-Backend Compatibility**: Works with asyncio, trio, and other AnyIO-supported backends
- **Resource Safety**: Built-in leak tracking and automatic cleanup mechanisms
- **Type Safety**: Full type hints with generics for compile-time safety
- **Zero-Overhead Wrappers**: Thin abstractions over AnyIO primitives

**Key Capabilities:**

- **Primitives**: Lock, Semaphore, CapacityLimiter, Queue, Event, Condition for coordination
- **Cancellation**: Timeout and deadline management with flexible strategies
- **Patterns**: gather, race, bounded_map, retry for common workflows
- **Task Management**: Structured task groups with automatic cancellation
- **Resource Tracking**: Debug resource leaks with automatic tracking
- **Priority Queues**: Priority-based async work scheduling
- **Error Handling**: Concurrency-aware exception utilities and shields

**When to Use:**

- Concurrent I/O operations (network requests, file operations)
- Rate-limited bulk processing with controlled parallelism
- Timeout-aware operations for reliability
- Resource coordination between concurrent tasks
- Async workflow orchestration

**When NOT to Use:**

- CPU-bound tasks (use process pools via `ProcessPoolExecutor`)
- Synchronous code (use threading primitives instead)
- Single sequential operations (adds unnecessary overhead)

## Module Organization

The concurrency module is organized into focused submodules:

### Core Primitives

**[primitives](concurrency/primitives.md)** - Async synchronization primitives

Provides Lock, Semaphore, CapacityLimiter, Queue, Event, and Condition for coordinating concurrent tasks. These are AnyIO wrappers offering cross-backend compatibility.

**[task](concurrency/task.md)** - Task group management

Structured concurrency via TaskGroup for spawning and managing concurrent tasks with automatic cancellation on scope exit.

### Cancellation & Timeouts

**[cancel](concurrency/cancel.md)** - Timeout and deadline utilities

Relative timeouts (`fail_after`, `move_on_after`) and absolute deadlines (`fail_at`, `move_on_at`) with flexible cancellation strategies.

**[errors](concurrency/errors.md)** - Concurrency exception utilities

Exception group filtering, cancellation detection, and shielding utilities for robust error handling in concurrent code.

### High-Level Patterns

**[patterns](concurrency/patterns.md)** - Common concurrency workflows

Production-ready patterns: `gather()` for parallel execution, `race()` for first-wins, `bounded_map()` for rate-limited processing, `retry()` for resilience, and `CompletionStream` for streaming results.

### Specialized Utilities

**[priority_queue](concurrency/priority_queue.md)** - Priority-based async queues

Priority queue implementation with async put/get operations for priority-based task scheduling.

**[resource_tracker](concurrency/resource_tracker.md)** - Resource leak detection

Automatic tracking of unclosed async resources (streams, locks) with leak detection and reporting.

**[utils](concurrency/utils.md)** - Helper utilities

Time utilities (`current_time`, `sleep`), sync/async bridges (`run_sync`, `run_async`), and coroutine inspection (`is_coro_func`).

## Common Patterns

### Safe Concurrent Execution

Use task groups for automatic cleanup:

```python
# noqa:validation
from lionherd_core.libs.concurrency import create_task_group, sleep

async def worker(name: str):
    await sleep(0.1)
    return f"Worker {name} done"

async with create_task_group() as tg:
    tg.start_soon(worker, "A")
    tg.start_soon(worker, "B")
    # Both tasks complete or get cancelled before exit
```

### Timeout Management

Enforce time limits with flexible error handling:

```python
from lionherd_core.libs.concurrency import fail_after, move_on_after

# Raise TimeoutError if operation takes >5s
async with fail_after(5.0):
    result = await long_running_operation()

# Silent cancellation if operation takes >5s
async with move_on_after(5.0):
    await optional_background_task()
```

### Parallel Processing with Limits

Control concurrency for rate-limited APIs:

```python
# noqa:validation
from lionherd_core.libs.concurrency import bounded_map

async def fetch(url: str):
    # Make HTTP request
    return {"url": url, "data": "..."}

urls = [f"https://api.example.com/{i}" for i in range(100)]

# Process max 10 URLs concurrently
results = await bounded_map(fetch, urls, limit=10)
```

### Resource Coordination

Use primitives for safe shared access:

```python
from lionherd_core.libs.concurrency import Lock

lock = Lock()

async def critical_section():
    async with lock:
        # Only one task at a time
        await update_shared_resource()
```

### Resilient Operations

Retry with exponential backoff:

```python
# noqa:validation
from lionherd_core.libs.concurrency import retry, fail_after

async def flaky_operation():
    # Operation that may fail temporarily
    ...

# Retry up to 3 times with exponential backoff, max 10s total
async with fail_after(10.0):
    result = await retry(
        flaky_operation,
        attempts=3,
        base_delay=0.1,
        max_delay=2.0,
        jitter=0.1,
    )
```

### First-Wins Racing

Get first successful result:

```python
from lionherd_core.libs.concurrency import race

# Query multiple mirrors, use first response
result = await race(
    fetch_from_mirror_1(),
    fetch_from_mirror_2(),
    fetch_from_mirror_3(),
)
```

## Quick Reference

### Import All Primitives

```python
from lionherd_core.libs.concurrency import (
    # Primitives
    Lock,
    Semaphore,
    CapacityLimiter,
    Queue,
    Event,
    Condition,
    # Task Management
    TaskGroup,
    create_task_group,
    # Cancellation
    CancelScope,
    fail_after,
    move_on_after,
    fail_at,
    move_on_at,
    effective_deadline,
    # Patterns
    gather,
    race,
    bounded_map,
    retry,
    CompletionStream,
    # Priority Queue
    PriorityQueue,
    QueueEmpty,
    QueueFull,
    # Resource Tracking
    LeakTracker,
    track_resource,
    untrack_resource,
    # Error Utilities
    shield,
    is_cancelled,
    non_cancel_subgroup,
    # Utils
    sleep,
    current_time,
    run_async,
    run_sync,
    is_coro_func,
)
```

## Submodule Reference

Detailed documentation for each submodule:

- **[primitives](concurrency/primitives.md)** - Lock, Semaphore, Queue, Event, Condition, CapacityLimiter
- **[task](concurrency/task.md)** - TaskGroup, create_task_group
- **[cancel](concurrency/cancel.md)** - fail_after, move_on_after, fail_at, move_on_at, effective_deadline
- **[errors](concurrency/errors.md)** - shield, is_cancelled, non_cancel_subgroup, get_cancelled_exc_class
- **[patterns](concurrency/patterns.md)** - gather, race, bounded_map, retry, CompletionStream
- **[priority_queue](concurrency/priority_queue.md)** - PriorityQueue, QueueEmpty, QueueFull
- **[resource_tracker](concurrency/resource_tracker.md)** - LeakTracker, track_resource, untrack_resource, LeakInfo
- **[utils](concurrency/utils.md)** - sleep, current_time, run_async, run_sync, is_coro_func

## Design Notes

### Why AnyIO?

lionherd-core uses AnyIO as the async backend abstraction layer, enabling code to work across asyncio, trio, and other async frameworks. This provides:

- **Future-proofing**: Switch async backends without code changes
- **Library compatibility**: Works in any AnyIO-compatible environment
- **Testing flexibility**: Use trio for better structured concurrency testing
- **Ecosystem integration**: Compatible with FastAPI, Starlette, and other AnyIO-based frameworks

### Structured Concurrency Principles

All concurrency utilities follow structured concurrency:

1. **Bounded Lifetimes**: Tasks live within well-defined scopes
2. **Automatic Cleanup**: Resources released when scope exits
3. **Error Propagation**: Exceptions properly bubble up the task tree
4. **Cancellation Safety**: Cancellation handled at scope boundaries

This prevents resource leaks and makes concurrent code easier to reason about.

### Performance Considerations

- **Primitive Overhead**: ~1-5μs per lock acquisition (negligible for I/O)
- **Task Spawning**: ~10-20μs per task (dominated by I/O wait time)
- **Queue Operations**: O(log n) for priority queue, O(1) for FIFO queue
- **Leak Tracking**: Minimal overhead when disabled, ~1-2% when enabled

For CPU-bound work, use `ProcessPoolExecutor` instead of async concurrency.

## Related Resources

- **[AnyIO Documentation](https://anyio.readthedocs.io/)** - Underlying async abstraction
- **[Trio Documentation](https://trio.readthedocs.io/)** - Structured concurrency inspiration
- **Base Module** (`../base/`) - Integration with Element, Node, Pile for async workflows
- **Notebooks** (`../../notebooks/`) - Practical examples and tutorials

## See Also

- **libs.schema_handlers** - Async schema validation utilities (see `../libs/schema_handlers/`)
- **libs.string_handlers** - Async string processing utilities (see `../libs/string_handlers/`)
- **protocols** - Protocol-based composition patterns (documentation coming soon)

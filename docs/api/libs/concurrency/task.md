# Task Groups

> Structured concurrency primitives for managing concurrent task lifecycles

## Overview

The `lionherd_core.libs.concurrency._task` module provides **structured concurrency** primitives for managing groups of concurrent tasks. Based on AnyIO's task group abstraction, these primitives ensure that tasks are properly coordinated and cleaned up.

**Key Capabilities:**

- **Structured Concurrency**: Tasks in a group cannot outlive the group's context
- **Cross-Backend Compatibility**: Works with asyncio, trio, and other AnyIO-supported backends
- **Automatic Cleanup**: Tasks are cancelled when group exits
- **Cancellation Propagation**: Cancelling the group cancels all child tasks
- **Exception Handling**: Exceptions from child tasks propagate to group context
- **Initialization Support**: Start tasks and wait for them to initialize before proceeding

**Core Components:**

- **TaskGroup**: Wrapper around AnyIO task groups for structured concurrency
- **create_task_group()**: Context manager for creating and managing task groups

## Import

```python
from lionherd_core.libs.concurrency import (
    TaskGroup,
    create_task_group,
    sleep,
    current_time,
)
```

---

## TaskGroup

Structured concurrency task group wrapper.

### Class Signature

```python
class TaskGroup:
    """Structured concurrency task group."""

    def __init__(self, tg: anyio.abc.TaskGroup) -> None: ...

    @property
    def cancel_scope(self) -> anyio.CancelScope: ...

    def start_soon(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        name: str | None = None,
    ) -> None: ...

    async def start(
        self,
        func: Callable[..., Awaitable[R]],
        *args: Any,
        name: str | None = None,
    ) -> R: ...
```

### Parameters

#### Constructor Parameters

**tg** : anyio.abc.TaskGroup

AnyIO task group instance to wrap.

**Notes:**

Typically not instantiated directly. Use `create_task_group()` instead.

### Properties

#### `cancel_scope`

Get cancel scope for controlling task group lifetime.

**Returns:**

- anyio.CancelScope: Cancel scope for the task group

**Examples:**

```python
async with create_task_group() as tg:
    # Set timeout for all tasks in group
    tg.cancel_scope.deadline = current_time() + 30.0

    tg.start_soon(long_running_task)
    tg.start_soon(another_task)
    # All tasks cancelled after 30 seconds
```

### Methods

#### `start_soon()`

Start task without waiting for initialization.

**Signature:**

```python
def start_soon(
    self,
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    name: str | None = None,
) -> None: ...
```

**Parameters:**

- `func` (Callable[..., Awaitable[Any]]): Async function to run as task
- `*args` (Any): Positional arguments to pass to function
- `name` (str, optional): Name for the task (useful for debugging)

**Returns:**

- None

**Notes:**

- Task starts immediately but `start_soon()` returns without waiting
- Useful for fire-and-forget tasks
- All tasks must complete before group context exits

**Examples:**

```python
# noqa:validation
async def worker(task_id: int):
    print(f"Worker {task_id} starting")
    await sleep(1)
    print(f"Worker {task_id} done")

async with create_task_group() as tg:
    for i in range(5):
        tg.start_soon(worker, i, name=f"worker-{i}")
    # Returns immediately, tasks run concurrently
# Waits here until all workers complete
```

#### `start()`

Start task and wait for initialization.

**Signature:**

```python
async def start(
    self,
    func: Callable[..., Awaitable[R]],
    *args: Any,
    name: str | None = None,
) -> R: ...
```

**Parameters:**

- `func` (Callable[..., Awaitable[R]]): Async function to run as task
- `*args` (Any): Positional arguments to pass to function
- `name` (str, optional): Name for the task

**Returns:**

- R: Initialization result from the task

**Notes:**

- Waits for task to call `task_status.started()` before returning
- Used for tasks that need to signal when ready
- Requires task function to accept `task_status` parameter

**Examples:**

```python
# noqa:validation
from anyio.abc import TaskStatus
import anyio
from lionherd_core.libs.concurrency import create_task_group, sleep

async def server_task(
    port: int,
    *,
    task_status: TaskStatus[str] = anyio.TASK_STATUS_IGNORED,
):
    # Initialize server (simulated)
    await sleep(0.1)  # Simulate server startup
    server_url = f"http://localhost:{port}"

    # Signal ready and return server URL
    task_status.started(server_url)

    # Continue running (simulated work)
    for _ in range(10):
        await sleep(0.5)

async def client_task(url: str):
    """Simulate client making requests."""
    await sleep(0.1)
    print(f"Client connected to {url}")

async with create_task_group() as tg:
    # Wait for server to be ready
    url = await tg.start(server_task, 8000, name="server")
    print(f"Server ready at {url}")

    # Now safe to start client tasks
    tg.start_soon(client_task, url)
```

---

## create_task_group()

Create task group context manager for structured concurrency.

### Function Signature

```python
@asynccontextmanager
async def create_task_group() -> AsyncIterator[TaskGroup]: ...
```

**Returns:**

- AsyncIterator[TaskGroup]: Async context manager yielding TaskGroup instance

**Notes:**

- All tasks must complete (or be cancelled) before context exits
- Exceptions from any task propagate to context
- Cancelling the context cancels all tasks

**Examples:**

```python
async with create_task_group() as tg:
    tg.start_soon(task1)
    tg.start_soon(task2)
    tg.start_soon(task3)
# All tasks guaranteed to be complete here
```

### Usage Patterns

#### Basic Concurrent Execution

```python
# noqa:validation
from lionherd_core.libs.concurrency import create_task_group, sleep

async def fetch_data(url: str) -> dict:
    """Simulate fetching data from URL."""
    await sleep(0.1)  # Simulate network latency
    return {"url": url, "data": f"result_{url}"}

urls = [
    "https://api.example.com/user/1",
    "https://api.example.com/user/2",
    "https://api.example.com/user/3",
]

results = []

async def collect_result(url: str):
    data = await fetch_data(url)
    results.append(data)

async with create_task_group() as tg:
    for url in urls:
        tg.start_soon(collect_result, url)
# All fetches complete, results populated
```

#### Exception Handling

```python
# noqa:validation
async def failing_task():
    await sleep(1)
    raise ValueError("Task failed!")

async def normal_task():
    await sleep(2)
    print("Normal task complete")

try:
    async with create_task_group() as tg:
        tg.start_soon(failing_task)
        tg.start_soon(normal_task)
except ValueError as e:
    # Exception from failing_task propagates here
    # normal_task is automatically cancelled
    print(f"Task group failed: {e}")
```

#### Timeout with Cancel Scope

```python
# noqa:validation
from lionherd_core.libs.concurrency import create_task_group, sleep, current_time

async def slow_task():
    await sleep(100)
    return "Done"

try:
    async with create_task_group() as tg:
        # Set 5 second timeout for all tasks
        tg.cancel_scope.deadline = current_time() + 5.0

        tg.start_soon(slow_task)
        tg.start_soon(slow_task)
except TimeoutError:
    print("Tasks timed out after 5 seconds")
```

#### Server Initialization Pattern

```python
# noqa:validation
from anyio.abc import TaskStatus
import anyio
from lionherd_core.libs.concurrency import create_task_group, Queue, sleep

async def background_worker(
    queue: Queue,
    *,
    task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
):
    # Initialize resources
    await sleep(0.1)  # Simulate connection setup
    connection_id = "conn_123"

    # Signal ready
    task_status.started()

    # Run worker loop
    try:
        while True:
            item = await queue.get()
            await sleep(0.1)  # Simulate processing
            print(f"Processed {item} with {connection_id}")
    except Exception:
        pass  # Queue closed, exit

async with create_task_group() as tg:
    # Create work queue
    work_queue = Queue()

    # Wait for worker to initialize
    await tg.start(background_worker, work_queue)

    # Safe to add work now
    await work_queue.put("task_1")
    await work_queue.put("task_2")
    await sleep(0.5)  # Let worker process
    await work_queue.close()
```

---

## Design Rationale

### Why Wrap AnyIO TaskGroup?

The module provides a thin wrapper around AnyIO's task group for several reasons:

1. **Consistent API**: Standardized interface across lionherd-core codebase
2. **Type Safety**: Explicit type hints for better IDE support
3. **Documentation**: Centralized usage patterns for team
4. **Future Flexibility**: Can add lionherd-specific features without breaking API

### Why Structured Concurrency?

Structured concurrency (via task groups) provides critical guarantees:

1. **No Orphaned Tasks**: Tasks cannot outlive their parent context
2. **Automatic Cleanup**: Tasks are cancelled when context exits
3. **Exception Propagation**: Errors don't get lost in background tasks
4. **Deterministic Lifetime**: Clear start and end points for concurrent operations

This prevents common async bugs like:

- Background tasks running after main function exits
- Exceptions silently swallowed in fire-and-forget tasks
- Resource leaks from uncancelled tasks

### start() vs start_soon()

Two patterns for different use cases:

**start_soon()**: Fire and forget

```python
# Tasks run independently, no coordination needed
async with create_task_group() as tg:
    tg.start_soon(log_event, "starting")
    tg.start_soon(send_metric, "count")
    tg.start_soon(update_status, "active")
```

**start()**: Coordinated initialization

```python
# Wait for server before starting clients
async with create_task_group() as tg:
    url = await tg.start(server_task, 8000)  # Wait for ready
    tg.start_soon(client_task, url)          # Can use URL now
```

---

## Common Pitfalls

### Pitfall 1: Tasks Outliving Context

**Issue**: Attempting to create tasks that continue after group exits.

```python
# noqa:validation
# ❌ WRONG: Cannot do this
async def background_loop():
    while True:
        await sleep(1)
        print("Still running...")

async with create_task_group() as tg:
    tg.start_soon(background_loop)
    # Task never ends, context blocks forever
```

**Solution**: Either add termination condition or use cancel scope.

```python
# noqa:validation
# ✅ Correct: Add termination signal
shutdown = Event()

async def background_loop():
    while not shutdown.is_set():
        await sleep(1)
        print("Still running...")

async with create_task_group() as tg:
    tg.start_soon(background_loop)
    await sleep(5)
    shutdown.set()  # Signal termination
# Context exits cleanly

# ✅ Correct: Use timeout
async with create_task_group() as tg:
    tg.cancel_scope.deadline = current_time() + 10.0
    tg.start_soon(background_loop)
# Tasks cancelled after 10 seconds
```

### Pitfall 2: Ignoring Exceptions

**Issue**: Exceptions from child tasks crash the entire task group.

```python
# noqa:validation
# ❌ WRONG: Unhandled exception crashes all tasks
async def risky_task():
    await sleep(1)
    raise ValueError("Oops!")

async def important_task():
    # This gets cancelled when risky_task fails
    await sleep(10)
    await save_critical_data()

async with create_task_group() as tg:
    tg.start_soon(risky_task)
    tg.start_soon(important_task)
    # ValueError propagates, important_task cancelled!
```

**Solution**: Handle exceptions within tasks or at group level.

```python
# noqa:validation
# ✅ Correct: Handle within task
import logging

logger = logging.getLogger(__name__)

async def risky_task():
    try:
        await sleep(1)
        raise ValueError("Oops!")
    except ValueError as e:
        logger.error(f"Task failed: {e}")
        # Doesn't propagate to group

# ✅ Correct: Handle at group level
try:
    async with create_task_group() as tg:
        tg.start_soon(risky_task)
        tg.start_soon(important_task)
except ValueError:
    # Handle group failure
    await cleanup()
```

### Pitfall 3: Forgetting task_status for start()

**Issue**: Using `start()` with task that doesn't call `task_status.started()`.

```python
# noqa:validation
from lionherd_core.libs.concurrency import create_task_group, sleep

# ❌ WRONG: Task doesn't signal ready
async def server_task(port: int):
    await sleep(0.1)  # Simulate server startup
    # Simulated server operation
    for _ in range(100):
        await sleep(0.1)
    # Never calls task_status.started()!

async with create_task_group() as tg:
    await tg.start(server_task, 8000)  # Hangs forever!
```

**Solution**: Always call `task_status.started()` in tasks used with `start()`.

```python
# noqa:validation
# ✅ Correct: Signal when ready
from anyio.abc import TaskStatus
import anyio
from lionherd_core.libs.concurrency import create_task_group, sleep

async def server_task(
    port: int,
    *,
    task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
):
    await sleep(0.1)  # Simulate server startup
    task_status.started()  # Signal ready!
    # Continue running simulated server
    for _ in range(100):
        await sleep(0.1)

async with create_task_group() as tg:
    await tg.start(server_task, 8000)  # Returns when ready
```

### Pitfall 4: Not Using Context Manager

**Issue**: Creating task group without context manager.

```python
# ❌ WRONG: No structured concurrency guarantees
tg = create_task_group()  # Returns async context manager, not TaskGroup!
# Can't use it properly
```

**Solution**: Always use `async with`.

```python
# ✅ Correct: Context manager
async with create_task_group() as tg:
    tg.start_soon(task1)
    tg.start_soon(task2)
# Guaranteed cleanup
```

---

## See Also

- **AnyIO Documentation**: [https://anyio.readthedocs.io/](https://anyio.readthedocs.io/)
- **Related Modules**:
  - `lionherd_core.libs.concurrency.primitives`: Lock, Semaphore, Queue, etc.
  - `lionherd_core.libs.concurrency.patterns`: High-level concurrency patterns
  - `lionherd_core.libs.concurrency.cancel`: Cancellation utilities

---

## Examples

### Example 1: Parallel Data Processing

```python
# noqa:validation
from lionherd_core.libs.concurrency import create_task_group, sleep

async def process_chunk(chunk_id: int, data: list) -> dict:
    """Process data chunk and return results."""
    result = {"chunk_id": chunk_id, "items": []}

    for item in data:
        await sleep(0.01)  # Simulate expensive computation
        processed = item * 2  # Simple transformation
        result["items"].append(processed)

    return result

async def parallel_process(data: list, chunk_size: int = 100):
    """Process data in parallel chunks."""
    # Split data into chunks
    chunks = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

    results = []

    async def collect_result(chunk_id: int, chunk: list):
        result = await process_chunk(chunk_id, chunk)
        results.append(result)

    # Process all chunks concurrently
    async with create_task_group() as tg:
        for chunk_id, chunk in enumerate(chunks):
            tg.start_soon(
                collect_result,
                chunk_id,
                chunk,
                name=f"chunk-{chunk_id}"
            )

    # Sort by chunk_id to maintain order
    results.sort(key=lambda r: r["chunk_id"])
    return results

# Usage
data = list(range(100))
results = await parallel_process(data, chunk_size=25)
print(f"Processed {sum(len(r['items']) for r in results)} items")
```

### Example 2: Timeout and Retry Pattern

```python
# noqa:validation
from lionherd_core.libs.concurrency import create_task_group, sleep, current_time

async def with_timeout_and_retry(
    func: Callable,
    *args,
    timeout: float = 30.0,
    max_retries: int = 3,
    **kwargs,
):
    """Execute function with timeout and retry logic."""

    for attempt in range(max_retries):
        try:
            async with create_task_group() as tg:
                # Set timeout for this attempt
                tg.cancel_scope.deadline = current_time() + timeout

                # Create result container
                result_container = []

                async def execute():
                    result = await func(*args, **kwargs)
                    result_container.append(result)

                tg.start_soon(execute)

            # Success - return result
            if result_container:
                return result_container[0]

        except TimeoutError:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} timed out, retrying...")
                await sleep(2 ** attempt)  # Exponential backoff
            else:
                raise TimeoutError(f"Failed after {max_retries} attempts")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                await sleep(2 ** attempt)
            else:
                raise

# Usage
async def flaky_api_call(endpoint: str):
    """Simulate an API call that may fail."""
    await sleep(0.2)  # Simulate network latency
    return {"data": f"response_from_{endpoint}"}

try:
    data = await with_timeout_and_retry(
        flaky_api_call,
        "https://api.example.com/data",
        timeout=10.0,
        max_retries=3,
    )
    print(f"Got data: {data}")
except TimeoutError:
    print("API call failed after retries")
```

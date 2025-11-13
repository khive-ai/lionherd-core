# Concurrency Patterns

> High-level concurrency utilities built on structured concurrency primitives

## Overview

The `concurrency.patterns` module provides production-ready concurrency patterns for common async workflows. Built on lionherd's structured concurrency primitives (task groups, cancel scopes), these utilities offer safe, composable abstractions for parallel execution, racing, retries, and streaming completions.

**Key Capabilities:**

- **gather**: Execute multiple awaitables concurrently with exception handling
- **race**: Return first completion among competing awaitables
- **bounded_map**: Apply async function to items with concurrency limits
- **CompletionStream**: Stream results as they complete with structured lifecycle
- **retry**: Deadline-aware exponential backoff with jitter

**When to Use:**

- Parallel execution of independent async operations (API calls, I/O)
- Rate-limited bulk processing (bounded concurrency)
- First-wins scenarios (timeouts, redundant requests)
- Streaming results as they complete (progress tracking)
- Resilient operations with automatic retry logic

**When NOT to Use:**

- Single async operations (use `await` directly)
- CPU-bound tasks (use process pools instead)
- Sequential operations with dependencies (use normal `await` chains)

## Functions

### gather

Execute multiple awaitables concurrently and collect results.

**Signature:**

```python
async def gather(
    *aws: Awaitable[T],
    return_exceptions: bool = False
) -> list[T | BaseException]: ...
```

**Parameters:**

- `*aws` (Awaitable[T]): Awaitables to execute concurrently
- `return_exceptions` (bool, default False): If True, return exceptions as results instead of raising

**Returns:**

- list[T | BaseException]: Results in same order as input awaitables. If `return_exceptions=True`, failed awaitables return exception objects; otherwise exceptions propagate.

**Raises:**

- ExceptionGroup: If `return_exceptions=False` and any awaitable raises (cancellations filtered out)

**Examples:**

```python
# noqa:validation
from lionherd_core.libs.concurrency import gather, sleep

# Basic concurrent execution
async def fetch(url):
    """Simulate fetching from URL."""
    await sleep(0.1)
    return {"url": url, "data": f"result_from_{url}"}

results = await gather(
    fetch("https://api.example.com/1"),
    fetch("https://api.example.com/2"),
    fetch("https://api.example.com/3"),
)
# [{"url": "...", "data": "..."}, ...]

# With exception handling
results = await gather(
    fetch("https://api.example.com/valid"),
    sleep(0.1),  # Dummy operation
    return_exceptions=True,
)
# [result_dict, None]
```

**Notes:**

- Uses structured concurrency via `create_task_group()` - all tasks cancelled on exit
- Results maintain input order regardless of completion order
- Cancellations are separated from real failures in exception handling
- Empty input returns empty list immediately

**See Also:**

- `bounded_map()`: Gather with concurrency limits
- `race()`: First completion instead of all completions

---

### race

Return the first completed awaitable, cancelling the rest.

**Signature:**

```python
async def race(*aws: Awaitable[T]) -> T: ...
```

**Parameters:**

- `*aws` (Awaitable[T]): Awaitables to race (must provide at least one)

**Returns:**

- T: Result of first completed awaitable

**Raises:**

- ValueError: If called with no awaitables
- Exception: If first completed awaitable raises, exception propagates outside ExceptionGroup

**Examples:**

```python
# noqa:validation
from lionherd_core.libs.concurrency import race, sleep

# Timeout pattern
async def fetch_with_timeout(url):
    async def fetch(url):
        """Simulate fetch operation."""
        await sleep(10.0)  # Slow operation
        return {"url": url, "data": "result"}

    return await race(
        fetch(url),
        sleep(5.0),  # 5-second timeout wins
    )

# Redundant requests (fastest server wins)
async def fetch_from_region(region):
    """Simulate fetching from different regions."""
    delays = {"us-east": 0.2, "eu-west": 0.1, "ap-south": 0.3}
    await sleep(delays[region])
    return {"region": region, "data": "result"}

result = await race(
    fetch_from_region("us-east"),
    fetch_from_region("eu-west"),  # Fastest (0.1s)
    fetch_from_region("ap-south"),
)
```

**Notes:**

- Immediately cancels remaining awaitables when first completes
- If first completion raises, exception is re-raised outside task group context (avoids ExceptionGroup wrapping)
- Uses memory stream for communication between tasks
- All tasks run concurrently until first completion

**See Also:**

- `gather()`: Collect all results instead of first
- `effective_deadline()`: Ambient deadline handling

---

### bounded_map

Apply async function to items with concurrency limit.

**Signature:**

```python
async def bounded_map(
    func: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    limit: int,
    return_exceptions: bool = False,
) -> list[R | BaseException]: ...
```

**Parameters:**

- `func` (Callable[[T], Awaitable[R]]): Async function to apply to each item
- `items` (Iterable[T]): Items to process
- `limit` (int): Maximum concurrent executions (must be >= 1)
- `return_exceptions` (bool, default False): If True, return exceptions as results instead of raising

**Returns:**

- list[R | BaseException]: Results in same order as input items. If `return_exceptions=True`, failed items return exception objects.

**Raises:**

- ValueError: If `limit <= 0`
- ExceptionGroup: If `return_exceptions=False` and any execution raises (cancellations filtered out)

**Examples:**

```python
# noqa:validation
from lionherd_core.libs.concurrency import bounded_map, sleep

# Rate-limited API calls (max 5 concurrent)
async def fetch_user(user_id):
    """Simulate API call."""
    await sleep(0.1)
    return {"id": user_id, "name": f"User{user_id}"}

user_ids = range(1, 101)
users = await bounded_map(
    fetch_user,
    user_ids,
    limit=5,
)
# Processes 100 users with max 5 concurrent requests

# With exception handling
async def process_item(item):
    """Simulate processing that may fail."""
    await sleep(0.05)
    if item % 10 == 0:
        raise ValueError(f"Failed to process {item}")
    return {"item": item, "processed": True}

items = range(1, 21)
results = await bounded_map(
    process_item,
    items,
    limit=10,
    return_exceptions=True,
)
successful = [r for r in results if not isinstance(r, Exception)]
failed = [r for r in results if isinstance(r, Exception)]
```

**Notes:**

- Uses `CapacityLimiter` for concurrency control
- Results maintain input order regardless of completion order
- Starts all tasks immediately but limits concurrent execution via semaphore
- Empty input returns empty list immediately
- Concurrency limit applies to function execution, not task creation

**See Also:**

- `gather()`: Unlimited concurrency
- `CompletionStream()`: Stream results as they complete
- `CapacityLimiter`: Underlying concurrency control primitive

---

### retry

Execute function with deadline-aware exponential backoff retry.

**Signature:**

```python
async def retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.1,
) -> T: ...
```

**Parameters:**

- `fn` (Callable[[], Awaitable[T]]): Async function to retry (must be nullary - no arguments)
- `attempts` (int, default 3): Maximum retry attempts (original + retries)
- `base_delay` (float, default 0.1): Initial delay in seconds
- `max_delay` (float, default 2.0): Maximum delay cap in seconds
- `retry_on` (tuple[type[BaseException], ...], default (Exception,)): Exception types to retry
- `jitter` (float, default 0.1): Random jitter factor (0.0 = no jitter, 0.1 = ±10% variance)

**Returns:**

- T: Result of successful execution

**Raises:**

- Exception: Last caught exception if all retry attempts exhausted or deadline exceeded
- Other exceptions: Exceptions not in `retry_on` propagate immediately

**Examples:**

```python
# noqa:validation
from lionherd_core.libs.concurrency import retry, move_on_after, sleep

# Retry transient failures
attempt_count = 0

async def fetch_data():
    """Simulate flaky API call."""
    global attempt_count
    attempt_count += 1
    await sleep(0.1)
    if attempt_count < 3:
        raise ConnectionError("Network error")
    return {"data": "success"}

result = await retry(
    fetch_data,
    attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    retry_on=(ConnectionError, TimeoutError),
)

# With ambient deadline (retry won't exceed deadline)
async def expensive_operation():
    """Simulate expensive operation."""
    await sleep(2.0)
    return {"result": "computed"}

with move_on_after(30.0):  # 30-second total timeout
    result = await retry(
        expensive_operation,
        attempts=10,
        base_delay=1.0,
    )
    # Retries respect 30s deadline even if attempts not exhausted
```

**Retry Schedule:**

Delay calculation: `min(max_delay, base_delay * 2^(attempt-1)) * (1 + random() * jitter)`

| Attempt | Base Delay | No Jitter | With 10% Jitter |
|---------|------------|-----------|-----------------|
| 1       | 0.1s       | 0.1s      | 0.1-0.11s       |
| 2       | 0.1s       | 0.2s      | 0.2-0.22s       |
| 3       | 0.1s       | 0.4s      | 0.4-0.44s       |
| 4       | 0.1s       | 0.8s      | 0.8-0.88s       |
| 5       | 0.1s       | 1.6s      | 1.6-1.76s       |
| 6       | 0.1s       | 2.0s      | 2.0-2.2s (capped)|

**Notes:**

- **Deadline-aware**: Respects ambient deadlines from `move_on_at`/`move_on_after`
- Avoids TOCTOU race between deadline check and sleep via `move_on_at` guard
- Jitter prevents thundering herd when multiple clients retry simultaneously
- Re-raises last exception if deadline exceeded mid-retry
- Does not retry if `retry_on` doesn't match exception type

**See Also:**

- `effective_deadline()`: Get current ambient deadline
- `move_on_at()`: Set deadline for retry boundary

---

## Classes

### CompletionStream

Async completion stream with structured concurrency and explicit lifecycle.

**Signature:**

```python
class CompletionStream:
    def __init__(
        self,
        aws: Sequence[Awaitable[T]],
        *,
        limit: int | None = None
    ): ...
```

**Parameters:**

- `aws` (Sequence[Awaitable[T]]): Awaitables to execute concurrently
- `limit` (int, optional): Maximum concurrent executions. If None, no limit (all start immediately)

**Attributes:**

| Attribute           | Type                                    | Description                                      |
|---------------------|-----------------------------------------|--------------------------------------------------|
| `aws`               | `Sequence[Awaitable[T]]`                | Input awaitables                                 |
| `limit`             | `int \| None`                           | Concurrency limit (None = unlimited)             |
| `_completed_count`  | `int`                                   | Number of results yielded                        |
| `_total_count`      | `int`                                   | Total awaitables to process                      |

**Methods:**

#### Lifecycle Management

```python
async def __aenter__(self) -> CompletionStream: ...
async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool: ...
```

Must be used as async context manager. Entry starts all tasks, exit cleans up resources and cancels pending tasks.

#### Iteration

```python
def __aiter__(self) -> CompletionStream: ...
async def __anext__(self) -> tuple[int, T]: ...
```

Yields results as `(index, result)` tuples in **completion order** (not input order).

**Raises:**

- RuntimeError: If used outside async context manager
- StopAsyncIteration: When all awaitables completed

**Examples:**

```python
# noqa:validation
from lionherd_core.libs.concurrency import CompletionStream, sleep

# Stream results as they complete
async def fetch(url):
    """Simulate fetch operation."""
    await sleep(0.1)
    return {"url": url, "data": f"result_{url}"}

urls = [f"https://api.example.com/{i}" for i in range(10)]
tasks = [fetch(url) for url in urls]

async with CompletionStream(tasks, limit=5) as stream:
    async for idx, result in stream:
        print(f"Task {idx} completed: {result}")
        # Process results incrementally as they arrive

# Early termination (remaining tasks cancelled)
async with CompletionStream(tasks) as stream:
    async for idx, result in stream:
        if result.get("data", "").endswith("_5"):  # Found what we need
            break  # Exit context - cancels pending tasks

# Collect results with completion order
results = []
async with CompletionStream(tasks, limit=10) as stream:
    async for idx, result in stream:
        results.append((idx, result))
# results = [(2, res2), (0, res0), (1, res1)] - ordered by completion time
```

**Notes:**

- **Structured concurrency**: All tasks cancelled on context exit (normal or exception)
- **Completion order**: Results yielded as they complete, not in input order
- **Early termination**: Breaking from iteration cleanly cancels pending tasks
- **Index tracking**: Tuple `(index, result)` maps results back to input position
- **Resource safety**: Automatically closes streams and task group on exit
- Swallows `ClosedResourceError` gracefully if stream closed during send

**Usage Patterns:**

```python
# Progress tracking
total = len(tasks)
async with CompletionStream(tasks, limit=5) as stream:
    async for idx, result in stream:
        completed = stream._completed_count
        print(f"Progress: {completed}/{total}")

# Timeout handling
from lionherd_core.libs.concurrency import move_on_after

with move_on_after(30.0):  # 30-second deadline
    async with CompletionStream(tasks) as stream:
        async for idx, result in stream:
            process(result)
# Partial results collected if timeout exceeded
```

**See Also:**

- `bounded_map()`: Collect all results in input order
- `gather()`: Wait for all completions
- `race()`: First completion only

---

## Common Pitfalls

### Pitfall 1: Forgetting return_exceptions

**Issue**: Uncaught exceptions in `gather()` or `bounded_map()` propagate as ExceptionGroup.

```python
# First failure cancels all tasks
results = await gather(task1, task2, task3)
# If task2 fails → ExceptionGroup raised, task3 cancelled
```

**Solution**: Use `return_exceptions=True` for failure-tolerant processing.

```python
results = await gather(task1, task2, task3, return_exceptions=True)
successful = [r for r in results if not isinstance(r, BaseException)]
```

---

### Pitfall 2: Unbounded Concurrency

**Issue**: Using `gather()` with thousands of tasks overwhelms resources.

```python
# BAD: Creates 10,000 concurrent HTTP connections
results = await gather(*[fetch(url) for url in urls])  # len(urls) = 10,000
```

**Solution**: Use `bounded_map()` with reasonable limit.

```python
# GOOD: Max 50 concurrent requests
results = await bounded_map(fetch, urls, limit=50)
```

---

### Pitfall 3: Ignoring Completion Order in CompletionStream

**Issue**: Assuming results arrive in input order.

```python
async with CompletionStream(tasks) as stream:
    async for idx, result in stream:
        # idx is NOT sequential (0, 1, 2...)
        # idx reflects input position, but arrives in completion order
```

**Solution**: Use index to map back to input position if order matters.

```python
results = [None] * len(tasks)
async with CompletionStream(tasks) as stream:
    async for idx, result in stream:
        results[idx] = result  # Restore input order
```

---

### Pitfall 4: Race Without Cleanup

**Issue**: Forgetting that `race()` cancels losing awaitables.

```python
# Loser tasks are cancelled - ensure they handle cancellation gracefully
result = await race(
    acquire_lock(),  # If this loses, lock might not be released
    timeout_task(),
)
```

**Solution**: Ensure tasks clean up resources in finally blocks or use structured concurrency primitives.

---

### Pitfall 5: Retry Without Deadline

**Issue**: Unbounded retry time when ambient deadline not set.

```python
# Retries for potentially minutes if attempts=10 and max_delay=60
await retry(flaky_operation, attempts=10, max_delay=60.0)
```

**Solution**: Set ambient deadline with `move_on_after()`.

```python
with move_on_after(120.0):  # 2-minute hard timeout
    await retry(flaky_operation, attempts=10, max_delay=60.0)
```

---

## Design Rationale

### Why Structured Concurrency?

All patterns use `create_task_group()` for **structured concurrency**, ensuring:

1. **Automatic cleanup**: Tasks cancelled on scope exit (exception or normal)
2. **No task leaks**: No background tasks outliving their scope
3. **Clear lifetime**: Task lifecycle tied to explicit context boundaries

This contrasts with `asyncio.create_task()` where tasks can leak if not awaited.

---

### Why gather() Filters Cancellations?

`gather()` separates cancellations from real failures via `non_cancel_subgroup()`:

- **Cancellations**: Propagated as-is (expected control flow)
- **Real exceptions**: Re-raised without cancellation noise

This prevents masking actual errors when tasks are cancelled due to timeout or explicit cancellation.

---

### Why Bounded Concurrency via Semaphore?

`bounded_map()` uses `CapacityLimiter` (semaphore) instead of batching:

- **Better resource utilization**: New task starts immediately when slot frees
- **Predictable memory**: Max `limit` tasks in flight, not `limit * batch_size`
- **Simpler code**: No batching logic or result merging

---

### Why CompletionStream Over AsyncIterator?

`CompletionStream` requires explicit context manager usage instead of simple async iteration because:

1. **Resource safety**: Forces explicit cleanup of task group and streams
2. **Structured concurrency**: Clear lifetime boundaries for concurrent tasks
3. **Cancellation safety**: Early break from iteration cleanly cancels pending tasks

---

### Why Deadline-Aware Retry?

`retry()` respects ambient deadlines from `move_on_at()` to prevent:

1. **Deadline overruns**: Retry doesn't exceed outer timeout
2. **TOCTOU races**: Avoids race between deadline check and sleep
3. **Predictable behavior**: Total operation time bounded by deadline

---

## See Also

- **Related Modules**:
  - [Task Management](task.md): Task groups and structured concurrency primitives
  - [Cancel Scopes](cancel.md): Deadline and cancellation scope management
  - [Primitives](primitives.md): Low-level concurrency primitives (locks, semaphores)
- **Related Patterns**:
  - `create_task_group()`: Underlying structured concurrency primitive
  - `CapacityLimiter`: Semaphore for bounded concurrency
  - `effective_deadline()`: Ambient deadline detection

---

## Examples

### Example 1: Parallel API Fetching with Error Handling

```python
from lionherd_core.libs.concurrency import gather, sleep

async def fetch_user_data():
    # Simulate async API calls
    async def fetch(user_id):
        await sleep(0.1)  # Simulate network delay
        if user_id == 5:  # Simulate one failure
            raise ValueError(f"User {user_id} not found")
        return {"id": user_id, "name": f"User{user_id}"}

    user_ids = range(1, 11)
    results = await gather(
        *[fetch(uid) for uid in user_ids],
        return_exceptions=True,
    )

    # Separate successes and failures
    users = []
    errors = []
    for uid, result in zip(user_ids, results):
        if isinstance(result, Exception):
            errors.append((uid, result))
        else:
            users.append(result)

    print(f"Fetched {len(users)} users, {len(errors)} errors")
    return users, errors
```

---

### Example 2: Rate-Limited Bulk Processing

```python
# noqa:validation
from lionherd_core.libs.concurrency import bounded_map, sleep

async def process_documents(doc_ids):
    # Simulate expensive API call
    async def analyze(doc_id):
        await sleep(0.2)  # Simulate processing time
        return {"doc_id": doc_id, "analysis": f"result_{doc_id}"}

    # Process with max 10 concurrent operations
    results = await bounded_map(
        analyze,
        doc_ids,
        limit=10,
        return_exceptions=True,
    )

    # Filter failures
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]

    print(f"Processed {len(successful)}, failed {len(failed)}")
    return successful

# Usage: Process 100 documents with concurrency limit
await process_documents(range(100))
```

---

### Example 3: Timeout with Race

```python
# noqa:validation
from lionherd_core.libs.concurrency import race, sleep

async def fetch_with_timeout():
    # Primary operation
    async def primary_fetch():
        await sleep(0.5)  # Simulate work
        return {"source": "primary", "data": [1, 2, 3]}

    # Timeout
    async def timeout():
        await sleep(1.0)
        raise TimeoutError("Request timed out")

    # Race primary operation vs timeout
    try:
        result = await race(primary_fetch(), timeout())
        print(f"Success: {result}")
        return result
    except TimeoutError:
        print("Timed out, using fallback")
        return {"source": "fallback", "data": []}

# Usage
await fetch_with_timeout()
```

---

### Example 4: Progress Tracking with CompletionStream

```python
# noqa:validation
from lionherd_core.libs.concurrency import CompletionStream, sleep

async def process_with_progress(item_count):
    # Create tasks that complete at different times
    async def process_item(item_id):
        await sleep(0.1 + (item_id % 3) * 0.1)  # Variable duration
        return {"id": item_id, "result": f"processed_{item_id}"}

    tasks = [process_item(i) for i in range(item_count)]
    completed = 0

    async with CompletionStream(tasks, limit=5) as stream:
        async for idx, result in stream:
            completed += 1
            progress = (completed / len(tasks)) * 100
            print(f"Progress: {progress:.1f}% - Task {idx} completed")

    print(f"All {completed} tasks completed")

# Usage: Process 20 items with progress tracking
await process_with_progress(20)
```

---

### Example 5: Resilient Operation with Retry

```python
# noqa:validation
from lionherd_core.libs.concurrency import retry, move_on_after, sleep
import random

async def fetch_important_data():
    # Simulate flaky operation
    attempt_count = 0

    async def flaky_fetch():
        nonlocal attempt_count
        attempt_count += 1
        await sleep(0.1)  # Simulate network delay

        if attempt_count < 3:  # Fail first 2 attempts
            raise ConnectionError(f"Attempt {attempt_count} failed")

        return {"data": "success", "attempts": attempt_count}

    # Retry up to 5 times with exponential backoff
    # Total timeout: 30 seconds
    with move_on_after(30.0):
        result = await retry(
            flaky_fetch,
            attempts=5,
            base_delay=0.5,
            max_delay=5.0,
            retry_on=(ConnectionError,),
            jitter=0.2,
        )
        print(f"Success after {result['attempts']} attempts")
        return result

# Usage
await fetch_important_data()
```

---

### Example 6: Mixed Gather and Race Pattern

```python
# noqa:validation
from lionherd_core.libs.concurrency import gather, race, sleep

async def fetch_from_multiple_sources():
    # Race to get data from fastest source
    async def source_a():
        await sleep(0.2)  # Slower source
        return {"source": "A", "value": 100}

    async def source_b():
        await sleep(0.1)  # Faster source (wins)
        return {"source": "B", "value": 200}

    # Gather supplementary data in parallel
    async def metadata():
        await sleep(0.15)
        return {"version": "1.0"}

    async def config():
        await sleep(0.1)
        return {"timeout": 30}

    # Race primary sources, gather supplementary data concurrently
    primary, (meta, cfg) = await gather(
        race(source_a(), source_b()),
        gather(metadata(), config()),
    )

    return {"data": primary, "metadata": meta, "config": cfg}

# Usage
result = await fetch_from_multiple_sources()
# result["data"]["source"] == "B" (faster source won the race)
```

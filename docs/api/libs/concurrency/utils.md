# Concurrency Utilities

> Async utility functions for time, sleep, sync/async detection, and thread pool execution

## Overview

The `lionherd_core.libs.concurrency` utilities module provides **essential async utilities** for common concurrency operations. These functions handle time management, async/sync function detection, and seamless execution of synchronous code in async contexts.

**Key Capabilities:**

- **Monotonic Time**: Clock-independent time measurement for timeouts and benchmarks
- **Coroutine Detection**: Cached runtime detection of async functions (handles partials)
- **Sync-to-Async Bridge**: Run blocking sync functions in thread pool without blocking event loop
- **Non-Blocking Sleep**: Sleep with async/await without blocking other tasks
- **Type-Safe**: ParamSpec-based generics preserve function signatures

**Available Utilities:**

- **current_time()**: Get monotonic clock time in seconds
- **is_coro_func()**: Check if callable is a coroutine function
- **run_sync()**: Execute synchronous function in thread pool (async)
- **sleep()**: Non-blocking async sleep

## Import

```python
from lionherd_core.libs.concurrency import (
    current_time,
    is_coro_func,
    run_sync,
    sleep,
)
```

---

## current_time()

Get current time in seconds using monotonic clock.

### Signature

```python
def current_time() -> float: ...
```

### Returns

**float**: Current time in seconds from monotonic clock (not wall clock)

### Examples

### Helper Functions for Examples

```python
# Mock functions used in examples below
from lionherd_core.libs.concurrency import sleep
import httpx

async def do_work():
    """Simulate doing work."""
    await sleep(0.1)
    return {"work": "done"}

async def check_condition():
    """Simulate checking a condition."""
    import random
    await sleep(0.05)
    return random.choice([True, False])

async def fetch(url):
    """Simulate HTTP fetch."""
    await sleep(0.1)
    return {"url": url, "data": "response"}

async def process_data(data):
    """Simulate processing data."""
    await sleep(0.1)
    return {"processed": data}

def compute(x):
    """Simulate CPU-bound computation."""
    import time
    time.sleep(0.1)
    return x ** 2

async def process_item(item):
    """Simulate processing an item."""
    await sleep(0.1)
    return f"processed_{item}"

async def check_status():
    """Simulate checking status."""
    await sleep(0.05)
    return "ready"

async def fetch_api(url):
    """Simulate API fetch."""
    await sleep(0.1)
    return {"api_response": "data"}

async def expensive_computation():
    """Simulate expensive computation."""
    await sleep(0.5)
    return {"result": "expensive"}

def get_system_load():
    """Simulate getting system load."""
    import random
    return random.uniform(0, 1)

def blocking_operation(n):
    """Simulate blocking operation."""
    import time
    time.sleep(n)
    return n * 2

async def fetch_raw_data():
    """Simulate fetching raw data."""
    await sleep(0.1)
    return [{"id": i} for i in range(10)]

async def process_background_jobs():
    """Simulate processing background jobs."""
    await sleep(1.0)

async def check_health():
    """Simulate health check."""
    await sleep(0.5)

async def process_task_with_timeout(task, timeout):
    """Simulate processing task with timeout."""
    await sleep(min(0.1, timeout))
    return {"task": task, "status": "complete"}
```

```python
from lionherd_core.libs.concurrency import current_time

# Measure elapsed time
start = current_time()
await do_work()
elapsed = current_time() - start
print(f"Operation took {elapsed:.2f} seconds")

# Timeout calculation
deadline = current_time() + 30.0  # 30 second timeout
while current_time() < deadline:
    if await check_condition():
        break
    await sleep(0.1)
```

### Notes

Uses `anyio.current_time()` which provides a **monotonic clock** guaranteed to never go backwards (immune to system clock adjustments). This is critical for:

1. **Accurate Timeouts**: System clock changes don't affect timeout calculations
2. **Reliable Benchmarks**: Measures actual elapsed time, not wall clock time
3. **Event Loop Integration**: Compatible with AnyIO's time-based operations

**Monotonic vs Wall Clock:**

```python
# ❌ Wall clock (can go backwards with system time changes)
import time
start = time.time()

# ✅ Monotonic clock (guaranteed monotonically increasing)
from lionherd_core.libs.concurrency import current_time
start = current_time()
```

### See Also

- `sleep()`: Non-blocking async sleep
- `anyio.current_time()`: Underlying AnyIO implementation

---

## is_coro_func()

Check if callable is a coroutine function.

### Signature

```python
def is_coro_func(func: Callable[..., Any]) -> bool: ...
```

### Parameters

**func** : Callable

Function or callable to check. Handles regular functions, coroutine functions, and partials.

### Returns

**bool**: True if `func` is a coroutine function (async def), False otherwise

### Examples

```python
from lionherd_core.libs.concurrency import is_coro_func
from functools import partial

# Regular function
def sync_func():
    return 42

# Async function
async def async_func():
    return 42

# Partial of async function
partial_async = partial(async_func)

print(is_coro_func(sync_func))      # False
print(is_coro_func(async_func))     # True
print(is_coro_func(partial_async))  # True (unwraps partial)
```

### Usage Patterns

#### Runtime Dispatch Based on Function Type

```python
from lionherd_core.libs.concurrency import is_coro_func

async def call_function(func: Callable[[], Any]) -> Any:
    """Call function correctly based on its type."""
    if is_coro_func(func):
        return await func()
    else:
        # Run sync function in thread pool
        return await run_sync(func)
```

#### Validation in Configuration

```python
from lionherd_core.libs.concurrency import is_coro_func

class WorkflowStep:
    def __init__(self, handler: Callable):
        if not is_coro_func(handler):
            raise ValueError(f"Handler must be async function, got {handler}")
        self.handler = handler
```

### Notes

**Caching**: Results are cached using `@cache` decorator for performance. Repeated checks on the same function are instant (no runtime overhead).

**Partial Unwrapping**: Automatically unwraps `functools.partial` objects to check the underlying function:

```python
from functools import partial

async def async_func(x, y):
    return x + y

partial_func = partial(async_func, 1)
is_coro_func(partial_func)  # True (unwraps to async_func)
```

**Implementation**: Uses `inspect.iscoroutinefunction()` under the hood, which checks for `async def` functions (not generators or awaitables).

### See Also

- `run_sync()`: Execute sync functions in async context
- `inspect.iscoroutinefunction()`: Standard library coroutine detection

---

## run_sync()

Run synchronous function in thread pool without blocking event loop.

### Signature

```python
async def run_sync(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R: ...
```

### Parameters

**func** : Callable[P, R]

Synchronous function to execute in thread pool. Must NOT be a coroutine function.

**\*args** : P.args

Positional arguments forwarded to `func`.

**\*\*kwargs** : P.kwargs

Keyword arguments forwarded to `func`.

### Returns

**R**: Result of calling `func(*args, **kwargs)` in thread pool

### Examples

```python
from lionherd_core.libs.concurrency import run_sync
import time

# Blocking sync function
def blocking_operation(n: int) -> int:
    time.sleep(n)  # Blocks thread, not event loop
    return n * 2

# Run without blocking event loop
result = await run_sync(blocking_operation, 5)
print(result)  # 10 (after 5 seconds)

# With keyword arguments
def complex_operation(x: int, multiplier: int = 2) -> int:
    time.sleep(1)
    return x * multiplier

result = await run_sync(complex_operation, 10, multiplier=3)
print(result)  # 30
```

### Usage Patterns

#### Integrating Sync Libraries

```python
# noqa:validation
from lionherd_core.libs.concurrency import run_sync, gather, sleep

async def fetch_sync_api(url: str) -> dict:
    """Simulate sync API call in async context."""
    async def fetch():
        await sleep(0.1)  # Simulate network call
        return {"url": url, "data": "response"}

    return await fetch()

# Multiple concurrent requests (event loop not blocked)
results = await gather(
    fetch_sync_api("https://api.example.com/1"),
    fetch_sync_api("https://api.example.com/2"),
    fetch_sync_api("https://api.example.com/3"),
)
```

#### File I/O Without Blocking

```python
from lionherd_core.libs.concurrency import run_sync

async def read_file_async(path: str) -> str:
    """Read file without blocking event loop."""
    def read_file():
        with open(path) as f:
            return f.read()

    return await run_sync(read_file)

async def write_file_async(path: str, content: str) -> None:
    """Write file without blocking event loop."""
    def write_file():
        with open(path, 'w') as f:
            f.write(content)

    await run_sync(write_file)
```

#### CPU-Bound Operations

```python
# noqa:validation
from lionherd_core.libs.concurrency import run_sync, gather
import hashlib

async def compute_hash(data: bytes) -> str:
    """Compute hash in thread pool (CPU-bound)."""
    def hash_data():
        return hashlib.sha256(data).hexdigest()

    return await run_sync(hash_data)

# Process multiple hashes concurrently
hashes = await gather(
    compute_hash(b"data1"),
    compute_hash(b"data2"),
    compute_hash(b"data3"),
)
```

### Notes

**Thread Pool**: Uses `anyio.to_thread.run_sync()` which executes functions in AnyIO's worker thread pool. This prevents blocking the event loop but doesn't bypass the GIL for CPU-bound work.

**Keyword Arguments**: The function handles kwargs specially due to AnyIO API limitations:

```python
# Internally uses partial when kwargs present
if kwargs:
    func_with_kwargs = partial(func, **kwargs)
    return await anyio.to_thread.run_sync(func_with_kwargs, *args)
```

**Type Safety**: Uses `ParamSpec` (P) and `TypeVar` (R) to preserve function signatures:

```python
def sync_func(x: int, y: str) -> bool: ...

# Type checker knows: run_sync(sync_func, 1, "hello") -> bool
result: bool = await run_sync(sync_func, 1, "hello")
```

**Performance Considerations**:

- **Thread Pool Overhead**: ~1-5ms per call (thread switching cost)
- **GIL Limitation**: CPU-bound Python code won't parallelize (use `multiprocessing` instead)
- **I/O-Bound**: Excellent for blocking I/O (file ops, sync HTTP libs, database drivers)

**When NOT to Use**:

```python
# noqa:validation
# ❌ Don't use for async functions
async def async_func():
    return 42

# await run_sync(async_func)  # Wrong! Just await it directly
result = await async_func()  # ✅ Correct

# ❌ Don't use for CPU-intensive pure Python (use ProcessPoolExecutor)
def cpu_bound():
    return sum(i*i for i in range(10_000_000))

# await run_sync(cpu_bound)  # GIL-limited, won't parallelize
```

### See Also

- `is_coro_func()`: Detect if function needs `run_sync()`
- `anyio.to_thread.run_sync()`: Underlying implementation
- `asyncio.to_thread()`: Standard library equivalent (Python 3.9+)

---

## sleep()

Sleep without blocking event loop.

### Signature

```python
async def sleep(seconds: float) -> None: ...
```

### Parameters

**seconds** : float

Duration to sleep in seconds. Can be fractional (e.g., 0.1 for 100ms).

### Returns

- None

### Examples

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep

# Sleep for 1 second
await sleep(1.0)

# Sleep for 100 milliseconds
await sleep(0.1)

# Polling with sleep
while not condition_met():
    await sleep(0.5)  # Check every 500ms

# Timeout pattern (using anyio directly - advanced feature)
async def with_timeout(coro, timeout_sec: float):
    import anyio
    with anyio.fail_after(timeout_sec):
        return await coro
```

### Usage Patterns

#### Rate Limiting

```python
from lionherd_core.libs.concurrency import sleep

async def rate_limited_requests(urls: list[str], delay: float = 0.5):
    """Fetch URLs with delay between requests."""
    results = []
    for url in urls:
        result = await fetch(url)
        results.append(result)
        await sleep(delay)  # 500ms between requests
    return results
```

#### Retry with Exponential Backoff

```python
from lionherd_core.libs.concurrency import sleep

async def retry_with_backoff(func, max_retries: int = 3):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            backoff = 2 ** attempt  # 1s, 2s, 4s
            print(f"Retry {attempt + 1}/{max_retries} after {backoff}s")
            await sleep(backoff)
```

#### Polling Loop

```python
from lionherd_core.libs.concurrency import sleep, current_time

async def wait_for_condition(check_func, timeout: float = 30.0):
    """Poll until condition met or timeout."""
    deadline = current_time() + timeout
    while current_time() < deadline:
        if await check_func():
            return True
        await sleep(0.5)  # Poll every 500ms
    raise TimeoutError(f"Condition not met within {timeout}s")
```

#### Simulated Work in Tests

```python
from lionherd_core.libs.concurrency import sleep, gather

async def test_concurrent_operations():
    """Test multiple operations running concurrently."""
    async def worker(worker_id: int):
        await sleep(0.1)  # Simulate work
        return f"Worker {worker_id} done"

    results = await gather(
        worker(1), worker(2), worker(3)
    )
    # All workers run concurrently, total time ~100ms (not 300ms)
```

### Notes

**Non-Blocking**: Uses `anyio.sleep()` which yields control to the event loop, allowing other tasks to run during the sleep period.

**Comparison with time.sleep()**:

```python
import time
from lionherd_core.libs.concurrency import sleep

# ❌ BLOCKS event loop (entire process frozen)
async def bad_sleep():
    time.sleep(1.0)  # Nothing else can run during this

# ✅ YIELDS to event loop (other tasks continue)
async def good_sleep():
    await sleep(1.0)  # Other tasks run during this
```

**Precision**: Sleep duration is approximate. Actual sleep time may be slightly longer due to:

- Event loop scheduling overhead (~1-10ms)
- System timer resolution (typically 1-15ms on different platforms)
- CPU load and context switching

**Cancellation**: Sleep is cancellation-safe. If the task is cancelled during sleep, it raises `asyncio.CancelledError` immediately.

### See Also

- `current_time()`: Get monotonic time for timeout calculations
- `anyio.sleep()`: Underlying implementation
- `asyncio.sleep()`: Standard library equivalent

---

## Design Rationale

### Why Wrap AnyIO Utilities?

These utilities wrap AnyIO functions to provide:

1. **Consistent Import Path**: Single import for all concurrency utilities
2. **Future Flexibility**: Abstraction layer allows backend changes without API breaks
3. **Enhanced Type Safety**: Explicit type hints (e.g., ParamSpec for `run_sync()`)
4. **Team Patterns**: Centralized utilities establish standard practices

### Why Monotonic Clock?

`current_time()` uses monotonic clock instead of wall clock because:

1. **Timeout Accuracy**: System clock changes (NTP sync, DST, manual adjustment) don't affect timeouts
2. **Benchmark Reliability**: Measures actual elapsed time, not calendar time
3. **Event Loop Compatibility**: Integrates with AnyIO's time-based operations

Example of wall clock issues:

```python
import time

# ❌ Wall clock can go backwards
start = time.time()  # 1699438200.0
# System clock adjusted backwards 10 minutes
elapsed = time.time() - start  # -600.0 (negative!)

# ✅ Monotonic clock never decreases
from lionherd_core.libs.concurrency import current_time
start = current_time()
# System clock adjusted (doesn't matter)
elapsed = current_time() - start  # Always >= 0
```

### Why Cache is_coro_func()?

Caching coroutine detection provides:

1. **Zero Overhead**: Repeated checks are instant (O(1) dict lookup)
2. **Framework Performance**: High-frequency runtime dispatch has no penalty
3. **Partial Unwrapping**: Expensive partial unwrapping only happens once

```python
# First call: unwraps partial, checks coroutine
is_coro_func(partial_func)  # ~10 microseconds

# Subsequent calls: cached lookup
is_coro_func(partial_func)  # ~0.1 microseconds (100× faster)
```

### Why ParamSpec for run_sync()?

`ParamSpec` (PEP 612) preserves function signatures for better type inference:

```python
# Without ParamSpec (old approach)
async def run_sync(func: Callable, *args, **kwargs) -> Any: ...
result = await run_sync(typed_func, 1, "hello")
# Type checker: result is Any (lost type information)

# With ParamSpec (current approach)
async def run_sync(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R: ...
result = await run_sync(typed_func, 1, "hello")
# Type checker: result has correct type (e.g., bool)
```

This enables IDE autocomplete and static type checking for `run_sync()` calls.

## Common Pitfalls

### Pitfall 1: Using time.sleep() Instead of sleep()

**Issue**: `time.sleep()` blocks the entire event loop, freezing all tasks.

```python
# ❌ Blocks event loop (200ms total - sequential)
from lionherd_core.libs.concurrency import gather

async def bad_concurrent_sleep():
    async def worker():
        time.sleep(0.1)  # BLOCKS everything
        return "done"

    results = await gather(worker(), worker())
    # Takes 200ms (workers run sequentially due to blocking)
```

**Solution**: Always use async `sleep()` in async functions.

```python
# ✅ Yields to event loop (100ms total - concurrent)
from lionherd_core.libs.concurrency import sleep, gather

async def good_concurrent_sleep():
    async def worker():
        await sleep(0.1)  # Yields control
        return "done"

    results = await gather(worker(), worker())
    # Takes 100ms (workers run concurrently)
```

### Pitfall 2: run_sync() on Async Functions

**Issue**: Passing async function to `run_sync()` causes runtime error.

```python
async def async_func():
    return 42

# ❌ TypeError: coroutine can't be used in thread pool
result = await run_sync(async_func)
```

**Solution**: Check function type first or just await it directly.

```python
# ✅ Correct: await async functions directly
result = await async_func()

# ✅ Or dispatch based on type
from lionherd_core.libs.concurrency import is_coro_func, run_sync

if is_coro_func(func):
    result = await func()
else:
    result = await run_sync(func)
```

### Pitfall 3: Assuming run_sync() Bypasses GIL

**Issue**: Expecting CPU-bound Python code to parallelize in thread pool.

```python
from lionherd_core.libs.concurrency import run_sync, gather

def cpu_intensive():
    return sum(i*i for i in range(10_000_000))

# ❌ Runs in thread, but GIL prevents parallelization
results = await gather(
    run_sync(cpu_intensive),
    run_sync(cpu_intensive),
    run_sync(cpu_intensive),
)
# Not faster than running sequentially (GIL-limited)
```

**Solution**: Use `ProcessPoolExecutor` for CPU-bound work.

```python
from concurrent.futures import ProcessPoolExecutor
from lionherd_core.libs.concurrency import gather
import asyncio

executor = ProcessPoolExecutor()

# ✅ Runs in separate processes (true parallelism)
loop = asyncio.get_event_loop()
results = await gather(
    loop.run_in_executor(executor, cpu_intensive),
    loop.run_in_executor(executor, cpu_intensive),
    loop.run_in_executor(executor, cpu_intensive),
)
# 3× faster on multi-core CPU
```

### Pitfall 4: Expecting Precise Sleep Timing

**Issue**: Assuming `sleep(0.01)` sleeps exactly 10ms.

```python
from lionherd_core.libs.concurrency import current_time, sleep

start = current_time()
await sleep(0.01)  # Request 10ms sleep
elapsed = current_time() - start
# elapsed might be 12-15ms due to event loop overhead
```

**Solution**: Accept sleep is approximate, add tolerance for critical timing.

```python
# ✅ Accept variance for non-critical timing
await sleep(0.5)  # ~500ms, variance doesn't matter

# ✅ For critical timing, measure and adjust
target_interval = 0.01
while True:
    start = current_time()
    await do_work()
    elapsed = current_time() - start
    sleep_time = max(0, target_interval - elapsed)
    await sleep(sleep_time)
```

## See Also

- **Related Modules**:
  - [Primitives](primitives.md): Lock, Semaphore, Queue, Event
  - [Patterns](patterns.md): TaskGroup, ExitStack, exception handlers
  - [Cancel](cancel.md): Cancellation scopes and timeout utilities
- **AnyIO Documentation**: [https://anyio.readthedocs.io/](https://anyio.readthedocs.io/)
- **Python asyncio**: [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)

## Examples

### Example 1: Retry with Timeout and Backoff

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep, current_time

async def retry_with_timeout(
    func,
    max_retries: int = 3,
    timeout: float = 30.0,
    base_backoff: float = 1.0,
):
    """Retry function with exponential backoff and total timeout."""
    deadline = current_time() + timeout

    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            if current_time() >= deadline:
                raise TimeoutError(f"Operation timed out after {timeout}s")

            backoff = base_backoff * (2 ** attempt)  # 1s, 2s, 4s
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
            await sleep(backoff)

# Usage
result = await retry_with_timeout(
    lambda: fetch_api("https://api.example.com/data"),
    max_retries=3,
    timeout=30.0,
)
```

### Example 2: Adaptive Polling

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep, current_time

async def wait_for_condition_adaptive(
    check_func,
    timeout: float = 60.0,
    initial_interval: float = 0.1,
    max_interval: float = 5.0,
):
    """Poll with adaptive interval (starts fast, slows down)."""
    deadline = current_time() + timeout
    interval = initial_interval

    while current_time() < deadline:
        if await check_func():
            return True

        await sleep(interval)

        # Exponential backoff up to max_interval
        interval = min(interval * 1.5, max_interval)

    raise TimeoutError(f"Condition not met within {timeout}s")

# Usage: Check frequently at first, then slow down
await wait_for_condition_adaptive(
    lambda: check_status(),
    timeout=60.0,
    initial_interval=0.1,  # Start with 100ms
    max_interval=5.0,      # Max 5s between checks
)
```

### Example 3: Mixed Sync/Async Pipeline

```python
# noqa:validation
from lionherd_core.libs.concurrency import is_coro_func, run_sync

class Pipeline:
    """Execute pipeline of mixed sync/async functions."""

    def __init__(self, steps: list[Callable]):
        self.steps = steps

    async def execute(self, data: Any) -> Any:
        """Run all steps, handling sync/async automatically."""
        result = data
        for step in self.steps:
            if is_coro_func(step):
                result = await step(result)
            else:
                result = await run_sync(step, result)
        return result

# Mix of sync and async transformations
def sync_transform(data: dict) -> dict:
    return {k: v.upper() for k, v in data.items()}

async def async_transform(data: dict) -> dict:
    await sleep(0.1)  # Simulate async I/O
    return {k: len(v) for k, v in data.items()}

def sync_filter(data: dict) -> dict:
    return {k: v for k, v in data.items() if v > 3}

pipeline = Pipeline([
    sync_transform,
    async_transform,
    sync_filter,
])

result = await pipeline.execute({"a": "hello", "b": "world"})
# {'a': 5, 'b': 5} (both > 3)
```

# Cancellation Utilities

> Timeout and deadline management utilities with flexible cancellation strategies

## Overview

The `lionherd_core.libs.concurrency.cancel` module provides a unified interface for timeout and deadline management built on top of AnyIO's cancellation scopes. It offers both **relative timeouts** (seconds from now) and **absolute deadlines** (specific time points), with two cancellation strategies: **fail** (raise TimeoutError) and **move on** (silent cancellation).

**Key Capabilities:**

- **Relative Timeouts**: Time-limited operations with `fail_after()` and `move_on_after()`
- **Absolute Deadlines**: Deadline-based cancellation with `fail_at()` and `move_on_at()`
- **Flexible Cancellation**: Choose between error-raising or silent cancellation
- **Null Timeout Support**: `None` timeout creates cancellable scope without deadline
- **Deadline Inspection**: Query effective deadline with `effective_deadline()`
- **AnyIO Integration**: Built on `anyio.CancelScope` for cross-platform async support

**When to Use Cancel Utilities:**

- **API Rate Limiting**: Timeout network requests to prevent indefinite blocking
- **Resource Cleanup**: Ensure operations complete within time budget
- **Graceful Degradation**: Silent cancellation for optional background tasks
- **Test Timeouts**: Prevent test suites from hanging on async operations
- **User Experience**: Enforce response time guarantees for interactive systems

**When NOT to Use:**

- **CPU-Bound Tasks**: Cancellation only works in async code, not blocking CPU work
- **Critical Operations**: Don't silence errors for operations that must complete
- **Fine-Grained Control**: For complex cancellation logic, use `CancelScope` directly

## Module Contents

```python
from lionherd_core.libs.concurrency.cancel import (
    CancelScope,           # Re-exported from anyio
    fail_after,            # Timeout context (raises on timeout)
    move_on_after,         # Timeout context (silent on timeout)
    fail_at,               # Deadline context (raises on timeout)
    move_on_at,            # Deadline context (silent on timeout)
    effective_deadline,    # Query ambient deadline
)
```

## API Reference

### CancelScope

Re-exported from `anyio.CancelScope`. See [AnyIO documentation](https://anyio.readthedocs.io/en/stable/cancellation.html) for detailed API.

**Type:** `type[anyio.CancelScope]`

**Usage:**

```python
from lionherd_core.libs.concurrency.cancel import CancelScope

async def example():
    with CancelScope() as scope:
        # Manual cancellation control
        scope.cancel()
```

---

### fail_after()

Create a context manager with relative timeout that raises `TimeoutError` when exceeded.

**Signature:**

```python
@contextmanager
def fail_after(seconds: float | None) -> Iterator[CancelScope]: ...
```

**Parameters:**

- `seconds` (float or None): Maximum execution time in seconds
  - If `None`, creates cancellable scope without timeout (still cancellable by outer scopes)
  - If float, raises `TimeoutError` after specified seconds
  - Must be non-negative if provided

**Yields:**

- CancelScope: Scope object for inspecting/controlling cancellation state

**Raises:**

- TimeoutError: When operation exceeds specified timeout

**Examples:**

```python
from lionherd_core.libs.concurrency.cancel import fail_after
import anyio

async def fetch_data():
    await anyio.sleep(2.0)
    return "data"

async def example():
    try:
        # Timeout after 1 second - raises TimeoutError
        async with fail_after(1.0) as scope:
            result = await fetch_data()
    except TimeoutError:
        print("Operation timed out!")

    # No timeout - still cancellable by outer scopes
    async with fail_after(None) as scope:
        result = await fetch_data()  # Will complete normally

    # Check if cancelled
    async with fail_after(0.5) as scope:
        try:
            await anyio.sleep(1.0)
        except TimeoutError:
            print(f"Cancelled: {scope.cancel_called}")  # True
```

**See Also:**

- `move_on_after()`: Silent cancellation variant
- `fail_at()`: Absolute deadline variant

**Notes:**

Internally delegates to `anyio.fail_after()` for non-None timeouts. The `None` timeout case creates a plain `CancelScope()` that can still be cancelled by outer scopes or manual `scope.cancel()` calls.

---

### move_on_after()

Create a context manager with relative timeout that silently cancels when exceeded.

**Signature:**

```python
@contextmanager
def move_on_after(seconds: float | None) -> Iterator[CancelScope]: ...
```

**Parameters:**

- `seconds` (float or None): Maximum execution time in seconds
  - If `None`, creates cancellable scope without timeout
  - If float, silently cancels after specified seconds (no exception raised)
  - Must be non-negative if provided

**Yields:**

- CancelScope: Scope object for inspecting/controlling cancellation state

**Examples:**

```python
from lionherd_core.libs.concurrency.cancel import move_on_after
import anyio

async def fetch_optional_data():
    await anyio.sleep(2.0)
    return "optional_data"

async def example():
    # Timeout after 1 second - no exception, result is None
    async with move_on_after(1.0) as scope:
        result = await fetch_optional_data()

    if scope.cancel_called:
        print("Timed out, using default")
        result = "default_data"

    # Pattern: optional enrichment with fallback
    user_data = {"id": 123}
    async with move_on_after(0.5) as scope:
        user_data["avatar"] = await fetch_avatar()  # Optional

    # Continue regardless of timeout
    return user_data
```

**See Also:**

- `fail_after()`: Error-raising variant
- `move_on_at()`: Absolute deadline variant

**Notes:**

Use for **optional operations** where timeout is acceptable. The operation is silently cancelled, allowing code to continue with fallback values or degraded functionality.

---

### fail_at()

Create a context manager that raises `TimeoutError` at an absolute deadline.

**Signature:**

```python
@contextmanager
def fail_at(deadline: float | None) -> Iterator[CancelScope]: ...
```

**Parameters:**

- `deadline` (float or None): Absolute deadline as monotonic timestamp (from `time.monotonic()`)
  - If `None`, creates cancellable scope without deadline
  - If float, raises `TimeoutError` when current time reaches deadline
  - If deadline is in the past, raises immediately

**Yields:**

- CancelScope: Scope object for inspecting/controlling cancellation state

**Raises:**

- TimeoutError: When current time reaches specified deadline

**Examples:**

```python
from lionherd_core.libs.concurrency.cancel import fail_at
from lionherd_core.libs.concurrency import current_time
import anyio

async def batch_process(items):
    # Process items until deadline
    for item in items:
        await process_item(item)

async def example():
    # Absolute deadline: 5 seconds from now
    deadline = current_time() + 5.0

    try:
        async with fail_at(deadline) as scope:
            await batch_process(large_dataset)
    except TimeoutError:
        print("Deadline exceeded")

    # Shared deadline across multiple operations
    end_time = current_time() + 10.0

    async with fail_at(end_time):
        await operation_1()  # Uses remaining time

    async with fail_at(end_time):
        await operation_2()  # Uses further reduced remaining time
```

**See Also:**

- `fail_after()`: Relative timeout variant
- `move_on_at()`: Silent cancellation variant
- `current_time()`: Get current monotonic timestamp

**Notes:**

Converts absolute deadline to relative timeout by subtracting current time. If deadline is in the past, `max(0.0, deadline - now)` ensures immediate timeout rather than negative duration.

**Use Case:** Coordinating multiple operations under a shared deadline where each operation should respect the remaining time budget.

---

### move_on_at()

Create a context manager that silently cancels at an absolute deadline.

**Signature:**

```python
@contextmanager
def move_on_at(deadline: float | None) -> Iterator[CancelScope]: ...
```

**Parameters:**

- `deadline` (float or None): Absolute deadline as monotonic timestamp (from `time.monotonic()`)
  - If `None`, creates cancellable scope without deadline
  - If float, silently cancels when current time reaches deadline
  - If deadline is in the past, cancels immediately

**Yields:**

- CancelScope: Scope object for inspecting/controlling cancellation state

**Examples:**

```python
from lionherd_core.libs.concurrency.cancel import move_on_at
from lionherd_core.libs.concurrency import current_time
import anyio

async def gather_metrics():
    metrics = []
    async for metric in metric_stream():
        metrics.append(metric)
    return metrics

async def example():
    # Gather metrics until deadline
    deadline = current_time() + 3.0

    async with move_on_at(deadline) as scope:
        metrics = await gather_metrics()

    # Continue with partial results if timed out
    if scope.cancel_called:
        print(f"Gathered {len(metrics)} metrics before deadline")

    # Pattern: best-effort data collection
    return metrics or []
```

**See Also:**

- `move_on_after()`: Relative timeout variant
- `fail_at()`: Error-raising variant
- `current_time()`: Get current monotonic timestamp

**Notes:**

Use for **best-effort operations** where partial results are acceptable. Silently stops at deadline, allowing code to proceed with whatever was collected.

---

### effective_deadline()

Return the ambient effective deadline from enclosing cancel scopes, or None if unlimited.

**Signature:**

```python
def effective_deadline() -> float | None: ...
```

**Returns:**

- float or None: Effective deadline as monotonic timestamp, or None if no deadline is set
  - Aggregates all enclosing `CancelScope` deadlines (returns earliest)
  - Returns `None` if all scopes are unlimited

**Examples:**

```python
from lionherd_core.libs.concurrency.cancel import (
    fail_after,
    move_on_after,
    effective_deadline,
)
import anyio

async def adaptive_operation():
    deadline = effective_deadline()

    if deadline is None:
        # No deadline - use thorough algorithm
        return await thorough_processing()
    else:
        # Limited time - use fast approximation
        remaining = deadline - current_time()
        if remaining < 1.0:
            return await quick_approximation()
        else:
            return await balanced_processing()

async def example():
    # No deadline
    async with fail_after(None):
        print(effective_deadline())  # None

    # Single deadline
    async with fail_after(5.0):
        print(effective_deadline())  # ~current_time() + 5.0

    # Nested deadlines - returns earliest
    async with fail_after(10.0):
        async with move_on_after(3.0):
            print(effective_deadline())  # ~current_time() + 3.0 (inner scope)
```

**See Also:**

- `current_time()`: Get current monotonic timestamp for remaining time calculation

**Notes:**

**AnyIO Conversion:** AnyIO uses `+inf` to represent "no deadline". This function converts that to `None` for consistency with lionherd's `None`-based unlimited timeout convention.

**Use Case:** Adaptive algorithms that adjust behavior based on available time. For example, caching strategies that skip expensive validation when deadline is tight.

---

## Usage Patterns

### Pattern 1: API Request Timeout

```python
from lionherd_core.libs.concurrency.cancel import fail_after
import httpx

async def fetch_with_timeout(url: str, timeout: float = 5.0):
    """Fetch URL with timeout, raising on failure."""
    async with fail_after(timeout):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return response.json()

# Raises TimeoutError if request exceeds 5 seconds
data = await fetch_with_timeout("https://api.example.com/data")
```

### Pattern 2: Optional Background Enrichment

```python
from lionherd_core.libs.concurrency.cancel import move_on_after

async def enrich_user_profile(user_id: int):
    """Enrich user profile with optional data sources."""
    profile = {"id": user_id}

    # Best-effort avatar fetch (max 0.5s)
    async with move_on_after(0.5) as scope:
        profile["avatar_url"] = await fetch_avatar(user_id)

    # Best-effort activity history (max 1.0s)
    async with move_on_after(1.0) as scope:
        profile["recent_activity"] = await fetch_activity(user_id)

    return profile  # Returns with partial data if timeouts occur
```

### Pattern 3: Shared Deadline Across Operations

```python
from lionherd_core.libs.concurrency.cancel import fail_at
from lionherd_core.libs.concurrency import current_time

async def multi_stage_pipeline(data, total_timeout: float = 10.0):
    """Process data through multiple stages under shared deadline."""
    deadline = current_time() + total_timeout

    try:
        async with fail_at(deadline):
            validated = await validate_data(data)

        async with fail_at(deadline):
            transformed = await transform_data(validated)

        async with fail_at(deadline):
            result = await store_data(transformed)

        return result

    except TimeoutError:
        logger.error(f"Pipeline exceeded {total_timeout}s deadline")
        raise
```

### Pattern 4: Adaptive Algorithm Selection

```python
from lionherd_core.libs.concurrency.cancel import (
    fail_after,
    effective_deadline,
)
from lionherd_core.libs.concurrency import current_time

async def search_database(query: str):
    """Adaptive search based on available time."""
    deadline = effective_deadline()

    if deadline is None:
        # No deadline - use comprehensive search
        return await full_text_search(query)

    remaining = deadline - current_time()

    if remaining < 0.5:
        # Very tight deadline - use index-only search
        return await quick_index_search(query)
    elif remaining < 2.0:
        # Moderate time - use optimized search
        return await optimized_search(query)
    else:
        # Ample time - use comprehensive search
        return await full_text_search(query)

# Usage with timeout
async with fail_after(5.0):
    results = await search_database("machine learning")
```

### Pattern 5: Graceful Degradation

```python
from lionherd_core.libs.concurrency.cancel import move_on_after

async def fetch_dashboard_data(user_id: int):
    """Fetch dashboard with graceful degradation."""
    dashboard = {
        "user_id": user_id,
        "core_data": None,      # Required
        "analytics": None,       # Optional
        "recommendations": None, # Optional
    }

    # Core data - required, higher timeout
    try:
        async with fail_after(3.0):
            dashboard["core_data"] = await fetch_core_data(user_id)
    except TimeoutError:
        raise ValueError("Core data fetch failed")

    # Analytics - optional, moderate timeout
    async with move_on_after(1.0) as scope:
        dashboard["analytics"] = await fetch_analytics(user_id)

    # Recommendations - optional, low timeout
    async with move_on_after(0.5) as scope:
        dashboard["recommendations"] = await fetch_recommendations(user_id)

    return dashboard  # Returns with available optional data
```

## Design Rationale

### Why Both Relative and Absolute Timeouts?

**Relative Timeouts** (`fail_after`, `move_on_after`):

- **Use Case**: Single operation with time budget
- **Advantage**: Simple to reason about (max 5 seconds for this task)
- **Example**: API request timeout, user interaction limit

**Absolute Deadlines** (`fail_at`, `move_on_at`):

- **Use Case**: Multiple operations under shared deadline
- **Advantage**: Automatically reduces remaining time budget for each stage
- **Example**: Request pipeline with total response time SLA

### Why Two Cancellation Strategies?

**Fail Strategy** (raises `TimeoutError`):

- **Use Case**: Operation must succeed or be explicitly handled
- **Advantage**: Caller forced to handle timeout case
- **Example**: Critical API calls, transactions, data validation

**Move On Strategy** (silent cancellation):

- **Use Case**: Optional operations where timeout is acceptable
- **Advantage**: Cleaner code for best-effort scenarios
- **Example**: Optional enrichment, metrics collection, caching

### Why Support None Timeout?

`None` timeout creates a cancellable scope without deadline:

- **Conditional Timeouts**: Apply timeout only in certain conditions
- **Outer Scope Cancellation**: Still respects parent scope deadlines
- **Manual Cancellation**: Enable `scope.cancel()` control
- **API Consistency**: Uniform interface for timeout and no-timeout cases

**Example:**

```python
async def operation(timeout: float | None):
    # Works with or without timeout - no special case logic
    async with fail_after(timeout):
        await process()
```

### Why Convert +inf to None?

AnyIO uses `+inf` (infinity) for "no deadline", but lionherd uses `None`:

- **Type Safety**: `None` clearly signals "no value", `+inf` can cause arithmetic bugs
- **API Consistency**: Matches lionherd's `timeout: float | None` convention
- **Readability**: `if deadline is None` clearer than `if isinf(deadline)`

`effective_deadline()` performs this conversion for consistent API surface.

## Common Pitfalls

### Pitfall 1: Using Timeouts in Sync Code

**Issue:** Cancellation only works in async contexts.

```python
# ❌ WRONG: Timeout has no effect
async with fail_after(1.0):
    time.sleep(10.0)  # Blocks entire async loop, timeout ineffective
```

**Solution:** Use async equivalents or run sync code in thread executor.

```python
# ✅ CORRECT: Async sleep
async with fail_after(1.0):
    await anyio.sleep(10.0)  # Properly cancelled

# ✅ CORRECT: Sync code in thread with timeout
async with fail_after(1.0):
    await anyio.to_thread.run_sync(blocking_function)
```

### Pitfall 2: Silencing Critical Errors

**Issue:** Using `move_on_after()` for operations that must complete.

```python
# ❌ WRONG: Database write might fail silently
async with move_on_after(1.0):
    await db.commit_transaction()
```

**Solution:** Use `fail_after()` for critical operations.

```python
# ✅ CORRECT: Fail loudly if transaction times out
async with fail_after(5.0):
    await db.commit_transaction()
```

### Pitfall 3: Nested Timeouts Shorter Than Outer

**Issue:** Inner timeout longer than outer has no effect.

```python
async with fail_after(1.0):      # Outer: 1 second
    async with fail_after(5.0):  # Inner: 5 seconds (ineffective)
        await long_operation()   # Cancelled after 1 second
```

**Solution:** Aware of effective deadline (innermost shortest timeout wins).

```python
# Check effective deadline
async with fail_after(1.0):
    remaining = effective_deadline() - current_time()
    print(f"Only {remaining}s available")  # ~1.0s, not 5.0s
```

### Pitfall 4: Forgetting to Check Cancellation

**Issue:** Not inspecting `scope.cancel_called` after silent cancellation.

```python
# ❌ WRONG: Using potentially None result
async with move_on_after(1.0):
    result = await fetch_data()

return result  # Could be None if timed out
```

**Solution:** Check cancellation state and provide fallback.

```python
# ✅ CORRECT: Handle timeout case
async with move_on_after(1.0) as scope:
    result = await fetch_data()

if scope.cancel_called:
    result = default_value

return result
```

### Pitfall 5: Using Wall Clock Time for Deadlines

**Issue:** Using `time.time()` (wall clock) instead of `time.monotonic()`.

```python
# ❌ WRONG: Wall clock affected by time adjustments
import time
deadline = time.time() + 5.0  # System time changes break this
```

**Solution:** Use `current_time()` which uses monotonic clock.

```python
# ✅ CORRECT: Monotonic time unaffected by clock adjustments
from lionherd_core.libs.concurrency import current_time
deadline = current_time() + 5.0
```

## See Also

- **Related Modules**:
  - [Concurrency Utils](../concurrency/utils.md): `current_time()` for monotonic timestamps
  - [Async Utilities](../concurrency/async_utils.md): Other async helpers
- **External Documentation**:
  - [AnyIO Cancellation](https://anyio.readthedocs.io/en/stable/cancellation.html): Underlying cancellation scope API
  - [Trio Timeouts](https://trio.readthedocs.io/en/stable/reference-core.html#cancellation-and-timeouts): Inspiration for timeout patterns
- **Related Concepts**:
  - [Task Orchestration](../../user_guide/orchestration.md): Using timeouts in workflows
  - [Error Handling](../../user_guide/error_handling.md): Timeout error patterns

## Examples

### Example 1: Retry with Timeout

```python
from lionherd_core.libs.concurrency.cancel import fail_after
import anyio

async def retry_with_timeout(
    operation,
    max_attempts: int = 3,
    timeout: float = 5.0,
    retry_delay: float = 1.0,
):
    """Retry operation with per-attempt timeout."""
    for attempt in range(1, max_attempts + 1):
        try:
            async with fail_after(timeout):
                return await operation()
        except TimeoutError:
            if attempt == max_attempts:
                raise
            print(f"Attempt {attempt} timed out, retrying...")
            await anyio.sleep(retry_delay)

# Usage
async def flaky_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

result = await retry_with_timeout(flaky_api_call, max_attempts=3, timeout=2.0)
```

### Example 2: Parallel Operations with Individual Timeouts

```python
from lionherd_core.libs.concurrency.cancel import move_on_after
import anyio

async def fetch_all_sources(sources: list[str], timeout: float = 2.0):
    """Fetch from multiple sources with individual timeouts."""

    async def fetch_with_timeout(source: str):
        async with move_on_after(timeout) as scope:
            data = await fetch_from_source(source)

        if scope.cancel_called:
            return None
        return data

    async with anyio.create_task_group() as tg:
        results = []
        for source in sources:
            results.append(await tg.start(fetch_with_timeout, source))

    # Filter out None (timed out sources)
    return [r for r in results if r is not None]

# Usage - get results from sources that respond within 2s
data = await fetch_all_sources(
    ["source1", "source2", "source3"],
    timeout=2.0
)
```

### Example 3: Deadline-Aware Task Queue

```python
from lionherd_core.libs.concurrency.cancel import fail_at, effective_deadline
from lionherd_core.libs.concurrency import current_time
import anyio

async def process_queue(tasks, total_timeout: float = 30.0):
    """Process tasks until queue empty or deadline reached."""
    deadline = current_time() + total_timeout
    results = []

    async with fail_at(deadline):
        for task in tasks:
            # Check remaining time before starting task
            remaining = effective_deadline() - current_time()

            if remaining < 1.0:
                print(f"Insufficient time ({remaining:.2f}s), stopping")
                break

            try:
                result = await process_task(task)
                results.append(result)
            except TimeoutError:
                print(f"Deadline reached after {len(results)} tasks")
                break

    return results

# Usage
tasks = generate_task_queue(100)
completed = await process_queue(tasks, total_timeout=30.0)
print(f"Completed {len(completed)}/100 tasks within deadline")
```

### Example 4: Conditional Timeout Based on Environment

```python
from lionherd_core.libs.concurrency.cancel import fail_after
import os

async def environment_aware_operation():
    """Apply strict timeout in production, relaxed in development."""

    # Production: strict 5s timeout
    # Development: no timeout (easier debugging)
    timeout = 5.0 if os.getenv("ENV") == "production" else None

    async with fail_after(timeout):
        result = await complex_operation()

    return result

# Production: raises TimeoutError if exceeds 5s
# Development: runs until completion
result = await environment_aware_operation()
```

### Example 5: Circuit Breaker with Timeout

```python
from lionherd_core.libs.concurrency.cancel import fail_after
import anyio

class CircuitBreaker:
    """Circuit breaker with timeout enforcement."""

    def __init__(self, timeout: float = 5.0, failure_threshold: int = 3):
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.failure_count = 0
        self.state = "closed"  # closed, open, half_open

    async def call(self, operation):
        """Execute operation with timeout and circuit breaking."""

        if self.state == "open":
            raise Exception("Circuit breaker open")

        try:
            async with fail_after(self.timeout):
                result = await operation()

            # Success - reset failure count
            self.failure_count = 0
            if self.state == "half_open":
                self.state = "closed"

            return result

        except (TimeoutError, Exception) as e:
            # Failure - increment count
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print("Circuit breaker opened")

            raise

# Usage
breaker = CircuitBreaker(timeout=2.0, failure_threshold=3)

for i in range(10):
    try:
        result = await breaker.call(lambda: api_request())
        print(f"Request {i}: success")
    except Exception as e:
        print(f"Request {i}: failed - {e}")

    await anyio.sleep(1.0)
```

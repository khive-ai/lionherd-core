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
from lionherd_core.libs.concurrency import (
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
from lionherd_core.libs.concurrency import CancelScope

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
from lionherd_core.libs.concurrency import sleep

async def example():
    try:
        # Timeout after 1 second - raises TimeoutError
        async with fail_after(1.0):
            await sleep(2.0)  # Takes 2s, times out at 1s
    except TimeoutError:
        print("Operation timed out!")

    # No timeout - still cancellable by outer scopes
    async with fail_after(None):
        await sleep(0.5)  # Completes normally

    # Check if cancelled
    async with fail_after(0.5) as scope:
        try:
            await sleep(1.0)
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
from lionherd_core.libs.concurrency import sleep

async def example():
    result = None

    # Timeout after 1 second - no exception raised
    async with move_on_after(1.0) as scope:
        await sleep(2.0)  # Takes 2s, cancelled at 1s
        result = "optional_data"  # Never reached

    if scope.cancel_called:
        print("Timed out, using default")
        result = "default_data"

    # Pattern: optional enrichment with fallback
    user_data = {"id": 123}
    async with move_on_after(0.5) as scope:
        await sleep(0.2)  # Simulate avatar fetch
        user_data["avatar"] = "https://example.com/avatar.png"

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
from lionherd_core.libs.concurrency import current_time, sleep

async def example():
    # Absolute deadline: 2 seconds from now
    deadline = current_time() + 2.0

    try:
        async with fail_at(deadline):
            await sleep(3.0)  # Takes 3s, times out at 2s
    except TimeoutError:
        print("Deadline exceeded")

    # Shared deadline across multiple operations
    end_time = current_time() + 1.0

    try:
        async with fail_at(end_time):
            await sleep(0.4)  # Uses 0.4s, succeeds

        async with fail_at(end_time):
            await sleep(0.4)  # Uses 0.4s, ~0.2s remaining, succeeds

        async with fail_at(end_time):
            await sleep(0.5)  # Would need 0.5s, times out
    except TimeoutError:
        print("Final operation exceeded shared deadline")
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
from lionherd_core.libs.concurrency import current_time, sleep

async def example():
    metrics = []

    # Gather metrics until deadline
    deadline = current_time() + 1.0

    async with move_on_at(deadline) as scope:
        # Simulate collecting metrics
        for i in range(100):
            await sleep(0.1)  # Each metric takes 0.1s
            metrics.append({"id": i, "value": i * 10})
            # After ~10 iterations, deadline reached

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
from lionherd_core.libs.concurrency import current_time, sleep

async def adaptive_operation():
    deadline = effective_deadline()

    if deadline is None:
        # No deadline - use thorough approach
        await sleep(1.0)
        return {"method": "thorough", "accuracy": 0.99}
    else:
        # Limited time - adapt based on remaining time
        remaining = deadline - current_time()
        if remaining < 1.0:
            await sleep(0.1)
            return {"method": "quick", "accuracy": 0.85}
        else:
            await sleep(0.5)
            return {"method": "balanced", "accuracy": 0.92}

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
# noqa:validation
import httpx

from lionherd_core.libs.concurrency.cancel import fail_after

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
from lionherd_core.libs.concurrency import sleep

async def enrich_user_profile(user_id: int):
    """Enrich user profile with optional data sources."""
    profile = {"id": user_id}

    # Best-effort avatar fetch (max 0.5s)
    async with move_on_after(0.5) as scope:
        await sleep(0.2)  # Simulate avatar fetch
        if not scope.cancel_called:
            profile["avatar_url"] = f"https://example.com/avatar/{user_id}.png"

    # Best-effort activity history (max 1.0s)
    async with move_on_after(1.0) as scope:
        await sleep(0.4)  # Simulate activity fetch
        if not scope.cancel_called:
            profile["recent_activity"] = [{"action": "login", "ts": "2025-01-01"}]

    return profile  # Returns with partial data if timeouts occur
```

### Pattern 3: Shared Deadline Across Operations

```python
from lionherd_core.libs.concurrency.cancel import fail_at
from lionherd_core.libs.concurrency import current_time, sleep

async def multi_stage_pipeline(data, total_timeout: float = 10.0):
    """Process data through multiple stages under shared deadline."""
    deadline = current_time() + total_timeout

    try:
        # Stage 1: Validation
        async with fail_at(deadline):
            await sleep(0.1)
            validated = {**data, "validated": True}

        # Stage 2: Transformation
        async with fail_at(deadline):
            await sleep(0.1)
            transformed = {**validated, "transformed": True}

        # Stage 3: Storage
        async with fail_at(deadline):
            await sleep(0.1)
            result = {"stored": True, "data": transformed}

        return result

    except TimeoutError:
        print(f"Pipeline exceeded {total_timeout}s deadline")
        raise
```

### Pattern 4: Adaptive Algorithm Selection

```python
# noqa:validation
from lionherd_core.libs.concurrency.cancel import (
    fail_after,
    effective_deadline,
)
from lionherd_core.libs.concurrency import current_time, sleep

async def search_database(query: str):
    """Adaptive search based on available time."""
    deadline = effective_deadline()

    if deadline is None:
        # No deadline - use comprehensive search
        await sleep(0.3)
        return [{"result": query, "score": 0.95, "method": "full"}]

    remaining = deadline - current_time()

    if remaining < 0.5:
        # Very tight deadline - use index-only search
        await sleep(0.05)
        return [{"result": query, "score": 0.75, "method": "quick"}]
    elif remaining < 2.0:
        # Moderate time - use optimized search
        await sleep(0.15)
        return [{"result": query, "score": 0.85, "method": "optimized"}]
    else:
        # Ample time - use comprehensive search
        await sleep(0.3)
        return [{"result": query, "score": 0.95, "method": "full"}]

# Usage with timeout
async with fail_after(5.0):
    results = await search_database("machine learning")
```

### Pattern 5: Graceful Degradation

```python
from lionherd_core.libs.concurrency.cancel import move_on_after, fail_after
from lionherd_core.libs.concurrency import sleep

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
            await sleep(0.2)
            dashboard["core_data"] = {"name": "User", "email": "user@example.com"}
    except TimeoutError:
        raise ValueError("Core data fetch failed")

    # Analytics - optional, moderate timeout
    async with move_on_after(1.0) as scope:
        await sleep(0.5)
        if not scope.cancel_called:
            dashboard["analytics"] = {"views": 100, "clicks": 50}

    # Recommendations - optional, low timeout
    async with move_on_after(0.5) as scope:
        await sleep(0.3)
        if not scope.cancel_called:
            dashboard["recommendations"] = ["item1", "item2", "item3"]

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
from lionherd_core.libs.concurrency import sleep, run_sync

# ✅ CORRECT: Async sleep
async with fail_after(1.0):
    await sleep(10.0)  # Properly cancelled

# ✅ CORRECT: Sync code in thread with timeout
async with fail_after(1.0):
    await run_sync(blocking_function)
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
  - [Concurrency Utils](../concurrency/utils.md): `current_time()` for monotonic timestamps and other async helpers
- **External Documentation**:
  - [AnyIO Cancellation](https://anyio.readthedocs.io/en/stable/cancellation.html): Underlying cancellation scope API
  - [Trio Timeouts](https://trio.readthedocs.io/en/stable/reference-core.html#cancellation-and-timeouts): Inspiration for timeout patterns
<!-- TODO: Add links to user guides when available (issue #115) -->

## Examples

> **Note:** For production patterns and complex use cases, see the tutorials section. These examples focus on demonstrating the API surface.

### Example 1: Retry with Timeout

```python
# noqa:validation
import httpx

from lionherd_core.libs.concurrency.cancel import fail_after
from lionherd_core.libs.concurrency import sleep

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
            await sleep(retry_delay)

# Usage
async def flaky_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

result = await retry_with_timeout(flaky_api_call, max_attempts=3, timeout=2.0)
```

### Example 2: Conditional Timeout Based on Environment

```python
# noqa:validation
import os

from lionherd_core.libs.concurrency.cancel import fail_after
from lionherd_core.libs.concurrency import sleep

async def environment_aware_operation():
    """Apply strict timeout in production, relaxed in development."""

    # Production: strict 5s timeout
    # Development: no timeout (easier debugging)
    timeout = 5.0 if os.getenv("ENV") == "production" else None

    async with fail_after(timeout):
        # Simulate complex operation
        await sleep(1.0)
        result = {"result": "complex", "env": os.getenv("ENV", "dev")}

    return result

# Production: raises TimeoutError if exceeds 5s
# Development: runs until completion
result = await environment_aware_operation()
```

> **Note:** For more complex patterns like parallel operations with timeouts, deadline-aware task queues, and circuit breakers, see the concurrency tutorials (issues [#64](https://github.com/khive-ai/lionherd-core/issues/64), [#65](https://github.com/khive-ai/lionherd-core/issues/65), [#66](https://github.com/khive-ai/lionherd-core/issues/66)).

# Concurrency Errors

> Backend-agnostic cancellation handling utilities for async operations

## Overview

The `errors` module provides **backend-agnostic utilities** for handling cancellation exceptions and ExceptionGroup filtering in async workflows. Built on top of `anyio`, these functions abstract away async backend differences (asyncio, trio) and provide consistent cancellation semantics.

**Key Capabilities:**

- **Backend Detection**: Dynamically retrieve the native cancellation exception class
- **Cancellation Testing**: Check if exceptions are cancellation-related
- **Shielding**: Run critical operations immune to outer cancellation
- **ExceptionGroup Filtering**: Split and filter ExceptionGroups by cancellation type (Python 3.11+)

**When to Use:**

- **Graceful Shutdown**: Distinguish cancellation from errors during cleanup
- **Critical Sections**: Shield database commits, file writes, or API calls from cancellation
- **Error Reporting**: Filter out cancellations to report only actionable errors
- **Multi-Task Coordination**: Process ExceptionGroups from task groups, separating cancellations from failures

**When NOT to Use:**

- **Simple async/await**: Basic async code doesn't need explicit cancellation handling
- **Synchronous Code**: These utilities are async-specific (use standard exception handling)
- **Direct anyio Usage**: If you're already using anyio directly, use its APIs instead

## Module Exports

```python
from lionherd_core.libs.concurrency import (
    get_cancelled_exc_class,
    is_cancelled,
    shield,
    non_cancel_subgroup,
)
```

## Functions

### `get_cancelled_exc_class()`

Return the backend-native cancellation exception class.

**Signature:**

```python
def get_cancelled_exc_class() -> type[BaseException]: ...
```

**Returns:**

- **type[BaseException]**: Cancellation exception class for current async backend
  - `asyncio.CancelledError` for asyncio backend
  - `trio.Cancelled` for trio backend

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import get_cancelled_exc_class

# Get the cancellation exception class for current backend
>>> cancel_exc = get_cancelled_exc_class()
>>> cancel_exc.__name__
'CancelledError'  # asyncio backend

# Use in exception handling
>>> from lionherd_core.libs.concurrency import sleep
>>> try:
...     await sleep(1.0)  # Some async operation
... except get_cancelled_exc_class():
...     print("Operation was cancelled")
```

**Notes:**

Wraps `anyio.get_cancelled_exc_class()` for consistent backend-agnostic cancellation detection. The returned class varies by backend but behaves equivalently.

**See Also:**

- `is_cancelled()`: Test if an exception instance is a cancellation

---

### `is_cancelled()`

Check if an exception is the backend-native cancellation exception.

**Signature:**

```python
def is_cancelled(exc: BaseException) -> bool: ...
```

**Parameters:**

- **exc** (BaseException): Exception instance to test

**Returns:**

- **bool**: True if `exc` is a cancellation exception for the current backend, False otherwise

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import is_cancelled, get_cancelled_exc_class

# Test cancellation exception
>>> cancel_exc = get_cancelled_exc_class()()
>>> is_cancelled(cancel_exc)
True

# Test regular exception
>>> value_err = ValueError("bad value")
>>> is_cancelled(value_err)
False

# Use in multi-exception handling
>>> from lionherd_core.libs.concurrency import sleep
>>> try:
...     await sleep(1.0)  # Some operation
... except Exception as e:
...     if is_cancelled(e):
...         print("Cancelled - cleanup and exit")
...     else:
...         print(f"Error: {e}")
...         raise
```

**Notes:**

Uses `isinstance()` check against `anyio.get_cancelled_exc_class()`, ensuring correct detection across async backends without hardcoding exception types.

**See Also:**

- `get_cancelled_exc_class()`: Get the cancellation exception class
- `non_cancel_subgroup()`: Extract non-cancellation exceptions from ExceptionGroups

---

### `shield()`

Run an async function immune to outer cancellation scope.

**Signature:**

```python
async def shield(
    func: Callable[P, Awaitable[T]],
    *args: P.args,
    **kwargs: P.kwargs
) -> T: ...
```

**Parameters:**

- **func** (Callable[P, Awaitable[T]]): Async function to execute with cancellation shielding
- **\*args** (P.args): Positional arguments passed to `func`
- **\*\*kwargs** (P.kwargs): Keyword arguments passed to `func`

**Returns:**

- **T**: Result of `func(*args, **kwargs)`

**Raises:**

- Any exception raised by `func` is propagated normally
- Outer cancellation is **blocked** while `func` executes

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import shield, get_cancelled_exc_class
>>> from lionherd_core.libs.concurrency import sleep
>>> import anyio

# Shield critical cleanup operation
>>> async def critical_cleanup():
...     """Critical cleanup that must complete."""
...     await sleep(0.1)
...     print("Critical cleanup done")

>>> async def workflow():
...     try:
...         async with anyio.create_task_group() as tg:
...             # This cleanup won't be interrupted by task group cancellation
...             await shield(critical_cleanup)
...     except get_cancelled_exc_class():
...         print("Workflow cancelled, but cleanup completed")

# Shield multiple cleanup operations
>>> async def cleanup_resources():
...     async def close_connections():
...         await sleep(0.05)
...         print("Connections closed")
...
...     async def flush_logs():
...         await sleep(0.05)
...         print("Logs flushed")
...
...     await shield(close_connections)
...     await shield(flush_logs)
```

**Notes:**

Uses `anyio.CancelScope(shield=True)` to create a cancellation barrier. The shielded function completes normally even if the outer scope is cancelled. Cancellation is **deferred** until the shield exits.

**Warning:**

Overuse of shielding can prevent graceful shutdown. Only shield **critical operations** that must complete (commits, cleanup, resource release). Long-running operations should remain cancellable.

**See Also:**

- `anyio.CancelScope`: Underlying cancellation scope primitive

---

### `non_cancel_subgroup()`

Extract non-cancellation exceptions from an ExceptionGroup, discarding cancellations.

**Signature:**

```python
def non_cancel_subgroup(eg: BaseExceptionGroup) -> BaseExceptionGroup | None: ...
```

**Parameters:**

- **eg** (BaseExceptionGroup): Exception group to filter

**Returns:**

- **BaseExceptionGroup | None**: Subgroup containing only non-cancellation exceptions, or None if all were cancellations

**Examples:**

```python
>>> from lionherd_core.libs.concurrency import non_cancel_subgroup
>>> import anyio

# Simplify error reporting - ignore cancellations
>>> async def run_workflow():
...     try:
...         async with anyio.create_task_group() as tg:
...             tg.start_soon(task1)
...             tg.start_soon(task2)
...             tg.start_soon(task3)
...     except BaseExceptionGroup as eg:
...         errors = non_cancel_subgroup(eg)
...         if errors:
...             # Only log/raise actual errors, not cancellations
...             await error_reporter.log(errors)
...             raise errors
...         else:
...             # All exceptions were cancellations - graceful shutdown
...             print("All tasks cancelled successfully")

# Chain with other error handling
>>> async def robust_workflow():
...     try:
...         await run_parallel_tasks()
...     except BaseExceptionGroup as eg:
...         errors = non_cancel_subgroup(eg)
...         if errors:
...             # Retry only failed tasks, not cancelled ones
...             await retry_failed_tasks(errors)
```

**Notes:**

Uses `BaseExceptionGroup.split()` internally to filter out cancellation exceptions, returning only the subgroup with actionable errors. Returns `None` if all exceptions were cancellations.

**Use Case:**

Simplifies error handling when you only care about actionable errors and want to ignore cancellations. Common pattern: graceful shutdown on cancellation, log/retry on errors.

---

## Usage Patterns

### Pattern 1: Graceful Shutdown with Error Reporting

```python
from lionherd_core.libs.concurrency import non_cancel_subgroup
import anyio

async def run_services():
    """Run multiple services, report errors but ignore cancellations."""
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(api_server)
            tg.start_soon(background_worker)
            tg.start_soon(metrics_collector)
    except BaseExceptionGroup as eg:
        errors = non_cancel_subgroup(eg)
        if errors:
            # Actionable errors - log and alert
            logger.error(f"Services failed: {errors}")
            await alert_operations_team(errors)
            raise errors
        else:
            # Clean cancellation - graceful shutdown
            logger.info("Services stopped gracefully")
```

### Pattern 2: Critical Section Shielding

```python
from lionherd_core.libs.concurrency import shield
import anyio

async def process_transaction(data):
    """Process transaction with guaranteed commit/rollback."""
    async with database.transaction() as txn:
        # Normal operations - cancellable
        validated_data = await validate(data)
        await txn.insert(validated_data)

        # Critical section - must complete
        await shield(txn.commit)
        # or on error: await shield(txn.rollback)

async def cleanup_on_shutdown():
    """Ensure cleanup completes even during cancellation."""
    try:
        await long_running_operation()
    finally:
        # Shield all cleanup - must finish
        await shield(close_database_connections)
        await shield(flush_pending_logs)
        await shield(save_state_to_disk)
```

### Pattern 3: Distinguish Cancellation from Errors

```python
from lionherd_core.libs.concurrency import is_cancelled, get_cancelled_exc_class
from lionherd_core.libs.concurrency import sleep

async def retry_on_error(operation, max_retries=3):
    """Retry on errors, but propagate cancellations immediately."""
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            if is_cancelled(e):
                # Cancellation - don't retry, propagate immediately
                raise
            elif attempt < max_retries - 1:
                # Error - retry with backoff
                await sleep(2 ** attempt)
            else:
                # Max retries exceeded
                raise

async def log_non_cancellation_errors(operation):
    """Log errors but not cancellations."""
    try:
        await operation()
    except get_cancelled_exc_class():
        # Expected cancellation - don't log
        raise
    except Exception as e:
        # Unexpected error - log for investigation
        logger.exception(f"Operation failed: {e}")
        raise
```

### Pattern 4: Parallel Task Error Aggregation

```python
from lionherd_core.libs.concurrency import non_cancel_subgroup
import anyio
import logging

logger = logging.getLogger(__name__)

async def run_parallel_with_partial_failure(tasks):
    """Run tasks in parallel, report failures, succeed on partial completion."""
    results = {}

    try:
        async with anyio.create_task_group() as tg:
            for task_id, task in tasks.items():
                tg.start_soon(task, results, task_id)
    except BaseExceptionGroup as eg:
        # Extract actionable errors (filter out cancellations)
        errors = non_cancel_subgroup(eg)

        # Handle errors (actionable)
        if errors:
            logger.error(f"{len(errors.exceptions)} tasks failed")
            for exc in errors.exceptions:
                logger.error(f"Task error: {exc}")

            # Decide: raise if critical, continue if optional
            if len(errors.exceptions) > len(tasks) / 2:
                # Majority failed - abort
                raise errors
            else:
                # Partial failure acceptable
                logger.warning("Continuing with partial results")
        else:
            # All exceptions were cancellations - graceful shutdown
            logger.info("All tasks cancelled gracefully")

    return results
```

## Common Pitfalls

### Pitfall 1: Over-Shielding Operations

**Issue**: Shielding long-running operations prevents graceful shutdown.

```python
# ❌ BAD: Shields entire workflow
async def process_batch(items):
    await shield(process_all_items, items)  # Can't cancel, even if takes hours
```

**Solution**: Only shield **critical finalization** steps, not entire workflows.

```python
# ✅ GOOD: Shield only commit
async def process_batch(items):
    results = await process_all_items(items)  # Cancellable
    await shield(save_results, results)  # Only shield the commit
```

### Pitfall 2: Not Preserving Exception Context

**Issue**: Catching and discarding exception groups loses debugging context.

```python
# ❌ BAD: Loses original exceptions
try:
    await run_tasks()
except BaseExceptionGroup as eg:
    errors = non_cancel_subgroup(eg)
    if errors:
        raise RuntimeError("Tasks failed")  # Original errors lost!
```

**Solution**: Re-raise the filtered exception group to preserve tracebacks.

```python
# ✅ GOOD: Preserves exception context
try:
    await run_tasks()
except BaseExceptionGroup as eg:
    errors = non_cancel_subgroup(eg)
    if errors:
        logger.error(f"Tasks failed: {errors}")
        raise errors  # Preserves tracebacks, causes, notes
```

### Pitfall 3: Assuming Single Exception Type

**Issue**: ExceptionGroups can contain multiple exception types.

```python
# ❌ BAD: Only handles first exception
try:
    await run_tasks()
except BaseExceptionGroup as eg:
    errors = non_cancel_subgroup(eg)
    if errors:
        # Only processes first exception
        raise errors.exceptions[0]
```

**Solution**: Handle the full exception group or iterate all exceptions.

```python
# ✅ GOOD: Handles all exceptions
try:
    await run_tasks()
except BaseExceptionGroup as eg:
    errors = non_cancel_subgroup(eg)
    if errors:
        # Log all errors
        for exc in errors.exceptions:
            logger.error(f"Task failed: {exc}")
        # Re-raise group
        raise errors
```

### Pitfall 4: Forgetting Backend Differences

**Issue**: Hardcoding backend-specific cancellation exceptions breaks on other backends.

```python
# ❌ BAD: Backend-specific (hardcoding asyncio)
try:
    await operation()
except asyncio.CancelledError:  # Fails on trio!
    handle_cancellation()
```

**Solution**: Use `get_cancelled_exc_class()` or `is_cancelled()` for portability.

```python
# ✅ GOOD: Backend-agnostic
from lionherd_core.libs.concurrency import get_cancelled_exc_class, is_cancelled

# Option 1: Exception class
try:
    await operation()
except get_cancelled_exc_class():
    handle_cancellation()

# Option 2: Test helper
try:
    await operation()
except Exception as e:
    if is_cancelled(e):
        handle_cancellation()
    else:
        raise
```

## Design Rationale

### Why Backend Abstraction?

Python has multiple async backends (asyncio, trio, curio) with different cancellation exception types:

- **asyncio**: `asyncio.CancelledError`
- **trio**: `trio.Cancelled`
- **curio**: `curio.CancelledError`

Hardcoding exception types breaks portability. These utilities use `anyio`'s backend detection to provide **backend-agnostic** cancellation handling, enabling library code to work across async frameworks.

### Why Shield Critical Sections?

Some operations **must complete atomically**:

1. **Database Transactions**: Commits/rollbacks must finish to maintain consistency
2. **Resource Cleanup**: File closes, connection releases prevent leaks
3. **State Persistence**: Saving checkpoints ensures recovery after crashes

Shielding these operations from cancellation prevents partial completion and data corruption. However, **minimal shielding** is critical - over-shielding prevents graceful shutdown.

### Why ExceptionGroup Filtering?

Python 3.11+ task groups raise `ExceptionGroup` when multiple tasks fail. Common scenarios:

1. **Graceful Shutdown**: Some tasks cancelled (expected), some failed (errors)
2. **Partial Failure**: Want to continue if only some tasks fail
3. **Error Reporting**: Log failures, ignore cancellations

Filtering cancellations from errors enables **selective error handling** - retry/log/alert on failures, gracefully shut down on cancellations.

### Why Preserve Exception Metadata?

ExceptionGroups carry rich context:

- **Nested Structure**: Groups can contain groups (hierarchical task structure)
- **Tracebacks**: Each exception has full stack trace
- **Cause/Context**: `__cause__` and `__context__` chains show error origins
- **Notes**: `__notes__` provide additional debugging info

Using `ExceptionGroup.split()` preserves all metadata, unlike manual filtering which loses context. This is **critical for debugging** complex multi-task failures.

## See Also

- **Related Modules**:
  - [anyio documentation](https://anyio.readthedocs.io/): Underlying async backend abstraction
- **Related Concepts**:
  - [Task Groups](https://anyio.readthedocs.io/en/stable/tasks.html): anyio's structured concurrency primitive
  - [ExceptionGroup PEP 654](https://peps.python.org/pep-0654/): Python 3.11 exception groups specification
  - [Cancellation Scopes](https://anyio.readthedocs.io/en/stable/cancellation.html): anyio's cancellation primitive

## Examples

> **Note:** For API reference, see function documentation above. For complex production patterns, see the concurrency tutorials.

### Example: Retry with Cancellation Awareness

```python
from lionherd_core.libs.concurrency import is_cancelled, sleep

async def retry_with_backoff(
    operation,
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
):
    """Retry operation with exponential backoff, respecting cancellation."""
    for attempt in range(max_retries):
        try:
            return await operation()

        except Exception as e:
            # Check if cancellation - propagate immediately
            if is_cancelled(e):
                print(f"Operation cancelled on attempt {attempt + 1}")
                raise

            # Regular error - retry if attempts remain
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await sleep(delay)
            else:
                print(f"All {max_retries} attempts failed")
                raise

# Usage
async def flaky_api_call():
    """Simulate flaky API that sometimes fails."""
    import random
    if random.random() < 0.3:
        raise ConnectionError("Network timeout")
    return {"status": "success"}

async def main():
    try:
        result = await retry_with_backoff(flaky_api_call)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")

anyio.run(main)
```

> **Note:** For complex patterns like graceful service shutdown, batch processing with partial failures, and database transactions with shielding, see the concurrency tutorials (issues [#67](https://github.com/khive-ai/lionherd-core/issues/67), [#68](https://github.com/khive-ai/lionherd-core/issues/68), [#69](https://github.com/khive-ai/lionherd-core/issues/69)).

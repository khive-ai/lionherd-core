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
from lionherd_core.libs.concurrency._errors import (
    get_cancelled_exc_class,
    is_cancelled,
    shield,
    split_cancellation,
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
>>> from lionherd_core.libs.concurrency._errors import get_cancelled_exc_class

# Get the cancellation exception class for current backend
>>> cancel_exc = get_cancelled_exc_class()
>>> cancel_exc.__name__
'CancelledError'  # asyncio backend

# Use in exception handling
>>> try:
...     await some_async_operation()
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
>>> from lionherd_core.libs.concurrency._errors import is_cancelled, get_cancelled_exc_class

# Test cancellation exception
>>> cancel_exc = get_cancelled_exc_class()()
>>> is_cancelled(cancel_exc)
True

# Test regular exception
>>> value_err = ValueError("bad value")
>>> is_cancelled(value_err)
False

# Use in multi-exception handling
>>> try:
...     await some_operation()
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
- `split_cancellation()`: Split ExceptionGroups by cancellation status

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
>>> from lionherd_core.libs.concurrency._errors import shield, get_cancelled_exc_class
>>> import anyio

# Shield critical database commit
>>> async def save_critical_data(data):
...     async with database.transaction():
...         await database.insert(data)
...         await database.commit()  # Must complete even if parent cancelled

>>> async def workflow():
...     try:
...         async with anyio.create_task_group() as tg:
...             # This commit won't be interrupted by task group cancellation
...             await shield(save_critical_data, {"key": "value"})
...             tg.start_soon(other_task)
...     except get_cancelled_exc_class():
...         print("Workflow cancelled, but data was saved")

# Shield cleanup operations
>>> async def cleanup_resources():
...     await shield(close_connections)
...     await shield(flush_logs)
...     await shield(send_shutdown_signal)
```

**Notes:**

Uses `anyio.CancelScope(shield=True)` to create a cancellation barrier. The shielded function completes normally even if the outer scope is cancelled. Cancellation is **deferred** until the shield exits.

**Warning:**

Overuse of shielding can prevent graceful shutdown. Only shield **critical operations** that must complete (commits, cleanup, resource release). Long-running operations should remain cancellable.

**See Also:**

- `anyio.CancelScope`: Underlying cancellation scope primitive

---

### `split_cancellation()`

Split an ExceptionGroup into cancellation and non-cancellation subgroups.

**Signature:**

```python
def split_cancellation(
    eg: BaseExceptionGroup,
) -> tuple[BaseExceptionGroup | None, BaseExceptionGroup | None]: ...
```

**Parameters:**

- **eg** (BaseExceptionGroup): Exception group to split (Python 3.11+ ExceptionGroup)

**Returns:**

- **tuple[BaseExceptionGroup | None, BaseExceptionGroup | None]**:
  - First element: Subgroup containing **only cancellation exceptions** (or None if empty)
  - Second element: Subgroup containing **only non-cancellation exceptions** (or None if empty)

**Examples:**

```python
>>> from lionherd_core.libs.concurrency._errors import split_cancellation
>>> import anyio

# Split mixed exception group
>>> async def main():
...     try:
...         async with anyio.create_task_group() as tg:
...             tg.start_soon(task_that_gets_cancelled)
...             tg.start_soon(task_that_raises_value_error)
...             tg.start_soon(task_that_raises_type_error)
...     except BaseExceptionGroup as eg:
...         cancels, errors = split_cancellation(eg)
...
...         if cancels:
...             print(f"Cancelled tasks: {len(cancels.exceptions)}")
...
...         if errors:
...             print(f"Failed tasks: {len(errors.exceptions)}")
...             for exc in errors.exceptions:
...                 print(f"  - {type(exc).__name__}: {exc}")
...             raise errors  # Re-raise only non-cancellation errors

# Filter for error reporting
>>> async def run_workflow():
...     try:
...         await run_task_group()
...     except BaseExceptionGroup as eg:
...         _, errors = split_cancellation(eg)
...         if errors:
...             await log_errors(errors)
...             raise errors
```

**Notes:**

Uses Python 3.11+ `ExceptionGroup.split()` to preserve exception structure, tracebacks, `__cause__`, `__context__`, and `__notes__`. The split is based on `anyio.get_cancelled_exc_class()`, ensuring backend compatibility.

**Behavior:**

- Nested exception groups are preserved (split recurses)
- Original exception metadata is maintained
- Either return value may be None if that category is empty
- Both may be None if input exception group is empty

**See Also:**

- `non_cancel_subgroup()`: Convenience function to get only non-cancellation errors
- `is_cancelled()`: Check individual exceptions

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
>>> from lionherd_core.libs.concurrency._errors import non_cancel_subgroup
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

Convenience wrapper around `split_cancellation()` that returns only the non-cancellation subgroup (second element of tuple). Equivalent to:

```python
_, errors = split_cancellation(eg)
return errors
```

**Use Case:**

Simplifies error handling when you only care about actionable errors and want to ignore cancellations. Common pattern: graceful shutdown on cancellation, log/retry on errors.

**See Also:**

- `split_cancellation()`: Get both cancellation and non-cancellation subgroups

---

## Usage Patterns

### Pattern 1: Graceful Shutdown with Error Reporting

```python
from lionherd_core.libs.concurrency._errors import non_cancel_subgroup
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
from lionherd_core.libs.concurrency._errors import shield
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
from lionherd_core.libs.concurrency._errors import is_cancelled, get_cancelled_exc_class
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
from lionherd_core.libs.concurrency._errors import split_cancellation
import anyio

async def run_parallel_with_partial_failure(tasks):
    """Run tasks in parallel, report failures, succeed on partial completion."""
    results = {}

    try:
        async with anyio.create_task_group() as tg:
            for task_id, task in tasks.items():
                tg.start_soon(task, results, task_id)
    except BaseExceptionGroup as eg:
        cancels, errors = split_cancellation(eg)

        # Log cancellations (informational)
        if cancels:
            logger.info(f"{len(cancels.exceptions)} tasks cancelled")

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
from lionherd_core.libs.concurrency._errors import get_cancelled_exc_class, is_cancelled

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
  - [concurrency](../concurrency.md): High-level concurrency module overview
  - [anyio documentation](https://anyio.readthedocs.io/): Underlying async backend abstraction
- **Related Concepts**:
  - [Task Groups](https://anyio.readthedocs.io/en/stable/tasks.html): anyio's structured concurrency primitive
  - [ExceptionGroup PEP 654](https://peps.python.org/pep-0654/): Python 3.11 exception groups specification
  - [Cancellation Scopes](https://anyio.readthedocs.io/en/stable/cancellation.html): anyio's cancellation primitive

## Examples

### Example 1: Graceful Service Shutdown

```python
from lionherd_core.libs.concurrency._errors import non_cancel_subgroup, shield
from lionherd_core.libs.concurrency import sleep
import anyio
import signal

class ServiceManager:
    def __init__(self):
        self.running = True

    async def run(self):
        """Run services with graceful shutdown."""
        # Set up signal handler for graceful shutdown
        def handle_signal(signum, frame):
            self.running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self.api_server)
                tg.start_soon(self.worker_pool)
                tg.start_soon(self.metrics_exporter)

                # Wait for shutdown signal
                while self.running:
                    await sleep(1)

                # Cancel all tasks
                tg.cancel_scope.cancel()

        except BaseExceptionGroup as eg:
            # Filter out expected cancellations
            errors = non_cancel_subgroup(eg)
            if errors:
                # Unexpected failures - log and alert
                print(f"Services failed with errors: {errors}")
                raise errors
            else:
                # Clean shutdown - all cancellations expected
                print("Services stopped gracefully")

        finally:
            # Shield cleanup - must complete
            await shield(self.cleanup_connections)
            await shield(self.flush_logs)
            print("Cleanup complete")

    async def api_server(self):
        # API server implementation
        pass

    async def worker_pool(self):
        # Worker pool implementation
        pass

    async def metrics_exporter(self):
        # Metrics implementation
        pass

    async def cleanup_connections(self):
        # Close database, redis, etc.
        pass

    async def flush_logs(self):
        # Ensure logs are written
        pass

# Run services
anyio.run(ServiceManager().run)
```

### Example 2: Retry with Cancellation Awareness

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

### Example 3: Parallel Processing with Error Aggregation

```python
from lionherd_core.libs.concurrency import non_cancel_subgroup, sleep
import anyio

async def process_batch_with_partial_failure(items, min_success_rate=0.8):
    """Process items in parallel, tolerate partial failures."""
    results = {}
    errors = {}

    async def process_item(item_id, item):
        try:
            result = await process_single_item(item)
            results[item_id] = result
        except Exception as e:
            errors[item_id] = e
            raise

    try:
        async with anyio.create_task_group() as tg:
            for item_id, item in items.items():
                tg.start_soon(process_item, item_id, item)

    except BaseExceptionGroup as eg:
        # Extract non-cancellation errors
        failures = non_cancel_subgroup(eg)

        # Analyze failures
        if failures:
            failure_count = len(failures.exceptions)
            total_count = len(items)
            success_count = total_count - failure_count
            success_rate = success_count / total_count

            print(f"Batch processing: {success_count}/{total_count} succeeded")

            # Log each failure
            for exc in failures.exceptions:
                print(f"  Error: {exc}")

            # Check if success rate acceptable
            if success_rate < min_success_rate:
                print(f"Success rate {success_rate:.1%} below threshold {min_success_rate:.1%}")
                raise failures
            else:
                print(f"Partial failure acceptable ({success_rate:.1%} success)")
        else:
            # All exceptions were cancellations
            print("Processing cancelled - all tasks stopped gracefully")

    return results

async def process_single_item(item):
    """Process a single item (implementation)."""
    await sleep(0.1)  # Simulate work
    if item.get("should_fail"):
        raise ValueError(f"Item processing failed: {item}")
    return {"processed": item}

# Usage
async def main():
    items = {
        "item1": {"data": "value1"},
        "item2": {"data": "value2", "should_fail": True},
        "item3": {"data": "value3"},
        "item4": {"data": "value4"},
    }

    results = await process_batch_with_partial_failure(items, min_success_rate=0.5)
    print(f"Final results: {results}")

anyio.run(main)
```

### Example 4: Database Transaction with Guaranteed Commit

```python
from lionherd_core.libs.concurrency._errors import shield
from lionherd_core.libs.concurrency import sleep

class Database:
    """Simplified database with async transactions."""

    async def transaction(self):
        return DatabaseTransaction()

class DatabaseTransaction:
    def __init__(self):
        self.operations = []
        self.committed = False

    async def insert(self, data):
        self.operations.append(("insert", data))
        await sleep(0.01)  # Simulate I/O

    async def commit(self):
        print(f"Committing {len(self.operations)} operations...")
        await sleep(0.1)  # Simulate commit I/O
        self.committed = True
        print("Transaction committed successfully")

    async def rollback(self):
        print("Rolling back transaction...")
        await sleep(0.05)  # Simulate rollback I/O
        self.operations.clear()
        print("Transaction rolled back")

async def save_user_data(db, user_data):
    """Save user data with guaranteed commit/rollback."""
    txn = await db.transaction()

    try:
        # Regular operations - cancellable
        await txn.insert({"type": "user", "data": user_data})
        await txn.insert({"type": "audit", "action": "user_created"})

        # Shield commit - must complete atomically
        await shield(txn.commit)
        print("User data saved successfully")

    except Exception as e:
        # Shield rollback - must complete to maintain consistency
        await shield(txn.rollback)
        print(f"Transaction failed: {e}")
        raise

    finally:
        # Verify transaction completed one way or another
        if not txn.committed and txn.operations:
            print("WARNING: Transaction neither committed nor rolled back!")

# Usage with cancellation
async def main():
    db = Database()

    async def run_with_timeout():
        async with anyio.fail_after(5.0):  # 5 second timeout
            await save_user_data(db, {"name": "Alice", "email": "alice@example.com"})

    try:
        await run_with_timeout()
    except TimeoutError:
        print("Operation timed out, but transaction was handled safely")

anyio.run(main)
```

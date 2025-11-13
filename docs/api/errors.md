# Error Handling

Lionherd provides a comprehensive exception hierarchy with structured error context and retry semantics.

## Overview

The error module provides:

- **Semantic exceptions**: `NotFoundError` and `ExistsError` replace generic `ValueError`
- **Structured context**: `.details` dict for debugging information
- **Retry semantics**: `.retryable` flag for retry strategies
- **Exception chaining**: `.__cause__` preservation for root cause analysis
- **Serialization**: `.to_dict()` for logging and monitoring

### Exception Hierarchy

```text
Exception
└── LionherdError (base)
    ├── NotFoundError (semantic)
    ├── ExistsError (semantic)
    ├── ValidationError
    ├── ConfigurationError
    ├── ExecutionError
    ├── ConnectionError
    └── TimeoutError
```

### Design Philosophy

Lionherd uses **semantic exceptions** that convey intent, not just symptoms:

- `NotFoundError`: Item doesn't exist (not generic `ValueError`)
- `ExistsError`: Item already exists (not generic `ValueError`)

This enables:

1. **Precise error handling**: Catch specific conditions
2. **Retry logic**: Use `.retryable` flag to decide retry strategy
3. **Structured debugging**: Access `.details` for context
4. **Root cause tracking**: Inspect `.__cause__` chain

---

## Exception Classes

### LionherdError

Base exception for all lionherd errors.

**Attributes:**

- `message: str` - Human-readable error message
- `details: dict[str, Any]` - Additional structured context
- `retryable: bool` - Whether this error can be retried

**Constructor:**

```python
LionherdError(
    message: str | None = None,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool | None = None,
    cause: Exception | None = None
)
```

**Parameters:**

- `message`: Error message (uses `default_message` if None)
- `details`: Additional context dict for debugging
- `retryable`: Whether error can be retried (uses `default_retryable` if None)
- `cause`: Original exception that caused this error (preserved as `__cause__`)

**Methods:**

#### to_dict()

Serialize error to dict for logging/debugging.

```python
def to_dict(self) -> dict[str, Any]
```

**Returns:**

- Dict with `error`, `message`, `retryable`, and optionally `details`

**Example:**

```python
from lionherd_core.errors import LionherdError

error = LionherdError(
    "Operation failed",
    details={"operation": "fetch", "resource": "user/123"},
    retryable=True
)

# Structured error data for logging
error_data = error.to_dict()
# {
#     "error": "LionherdError",
#     "message": "Operation failed",
#     "retryable": True,
#     "details": {"operation": "fetch", "resource": "user/123"}
# }
```

---

### NotFoundError

Item not found. **Not retryable** by default.

**Use when:** An expected item is missing (collection access, file lookup, database query).

**Attributes:**

- `default_message = "Item not found"`
- `default_retryable = False` (missing items won't appear on retry)

**Example:**

```python
from lionherd_core.errors import NotFoundError

# Basic usage
raise NotFoundError("User not found")

# With context
raise NotFoundError(
    "Node missing from graph",
    details={"node_id": str(node_id), "graph": "workflow"}
)

# With cause chain
try:
    item = pile[uuid]  # KeyError from dict
except KeyError as e:
    raise NotFoundError(
        f"Item {uuid} not found in pile",
        details={"uuid": str(uuid)}
    ) from e

# Progression: pop() raises NotFoundError (not IndexError)
from lionherd_core.base import Progression

prog = Progression(order=[uuid4(), uuid4()])

try:
    task = prog.pop(10)  # Index out of range
except NotFoundError as e:
    print(f"Index not found: {e}")
    # Error details: {"index": 10, "length": 2}

# Safe fallback with default parameter
task = prog.pop(10, default=None)  # Returns None, no exception
```

**Migration from ValueError:**

```python
# Before (v0.x):
if uuid not in pile:
    raise ValueError(f"Item {uuid} not found")

# After (v1.0.0+):
try:
    item = pile[uuid]
except KeyError as e:
    raise NotFoundError(
        f"Item {uuid} not found",
        details={"uuid": str(uuid)}
    ) from e
```

**Migration from IndexError (Progression):**

```python
# Before (v1.0.0-alpha3 and earlier):
try:
    task = progression.pop(index)
except IndexError:  # Standard Python exception
    handle_missing()

# After (v1.0.0-alpha4+):
try:
    task = progression.pop(index)
except NotFoundError:  # Semantic exception
    handle_missing()

# Or use safe fallback (recommended):
task = progression.pop(index, default=None)
if task is None:
    handle_missing()
```

---

### ExistsError

Item already exists. **Not retryable** by default.

**Use when:** Attempting to create an item that already exists (duplicate insertion, unique constraint violation).

**Attributes:**

- `default_message = "Item already exists"`
- `default_retryable = False` (duplicate items won't resolve on retry)

**Example:**

```python
from lionherd_core.errors import ExistsError

# Basic usage
raise ExistsError("User already registered")

# With context
raise ExistsError(
    "Progression name already exists",
    details={"name": "pending", "flow_id": str(flow.id)}
)

# Preventing duplicates
if name in self._progression_names:
    raise ExistsError(
        f"Progression '{name}' already exists",
        details={"name": name, "existing_id": str(self._progression_names[name])}
    )
```

**Migration from ValueError:**

```python
# Before (v0.x):
if item.id in pile:
    raise ValueError("Item already exists")

# After (v1.0.0+):
if item.id in pile:
    raise ExistsError(
        "Item already exists in pile",
        details={"item_id": str(item.id)}
    )
```

---

### ValidationError

Validation failure. **Not retryable**.

**Use when:** Input validation fails (schema mismatch, type error, constraint violation).

**Attributes:**

- `default_message = "Validation failed"`
- `default_retryable = False` (validation errors won't fix themselves)

**Example:**

```python
from lionherd_core.errors import ValidationError

# Type validation
if not isinstance(item, Node):
    raise ValidationError(
        "Item must be a Node",
        details={"type": type(item).__name__, "expected": "Node"}
    )

# Schema validation
if not all(field in data.keys() for field in required_fields):
    raise ValidationError(
        "Missing required fields",
        details={"missing": set(required_fields) - set(data.keys())}
    )
```

---

### ConfigurationError

Configuration error. **Not retryable**.

**Use when:** Invalid configuration (missing environment variable, invalid settings, incompatible options).

**Attributes:**

- `default_message = "Configuration error"`
- `default_retryable = False` (config errors need manual fixes)

**Example:**

```python
from lionherd_core.errors import ConfigurationError

# Missing configuration
if not api_key:
    raise ConfigurationError(
        "API key not configured",
        details={"env_var": "API_KEY"}
    )

# Invalid configuration
if max_workers <= 0:
    raise ConfigurationError(
        "max_workers must be positive",
        details={"max_workers": max_workers}
    )
```

---

### ExecutionError

Event/Calling execution failure. **Retryable** by default.

**Use when:** Runtime execution fails (function call error, async task failure, operation timeout).

**Attributes:**

- `default_message = "Execution failed"`
- `default_retryable = True` (most execution failures are transient)

**Example:**

```python
from lionherd_core.errors import ExecutionError

# Async task failure
try:
    result = await task()
except Exception as e:
    raise ExecutionError(
        "Task execution failed",
        details={"task": task.__name__},
        cause=e
    )

# Function call error
try:
    output = func(*args, **kwargs)
except Exception as e:
    raise ExecutionError(
        f"Function {func.__name__} failed",
        details={"args": args, "kwargs": kwargs},
        cause=e
    )
```

---

### ConnectionError

Connection/network failure. **Retryable** by default.

**Use when:** Network operations fail (API call timeout, database connection lost, socket error).

**Attributes:**

- `default_message = "Connection error"`
- `default_retryable = True` (network issues are often transient)

**Example:**

```python
from lionherd_core.errors import ConnectionError

# API call failure
try:
    response = await api_client.get(url)
except aiohttp.ClientError as e:
    raise ConnectionError(
        "API request failed",
        details={"url": url, "method": "GET"},
        cause=e
    )

# Database connection
try:
    conn = await db.connect()
except Exception as e:
    raise ConnectionError(
        "Database connection failed",
        details={"host": db.host, "port": db.port},
        cause=e
    )
```

---

### TimeoutError

Operation timeout. **Retryable** by default.

**Use when:** Operations exceed time limit (async timeout, request timeout, lock timeout).

**Attributes:**

- `default_message = "Operation timed out"`
- `default_retryable = True` (timeouts might succeed with more time)

**Example:**

```python
from lionherd_core.errors import TimeoutError
from lionherd_core import concurrency

# Async timeout
try:
    async with concurrency.fail_after(30):
        result = await long_operation()
except concurrency.get_cancelled_exc_class() as e:
    raise TimeoutError(
        "Operation timed out after 30s",
        details={"timeout": 30, "operation": "long_operation"}
    ) from e

# Lock timeout
try:
    async with concurrency.fail_after(5):
        async with lock:
            await critical_section()
except concurrency.get_cancelled_exc_class() as e:
    raise TimeoutError(
        "Failed to acquire lock",
        details={"timeout": 5, "lock": "critical_section"}
    ) from e
```

---

## Exception Metadata

### .retryable Flag

The `.retryable` flag indicates whether an operation can be retried after this error.

**Retryable exceptions** (default `True`):

- `ExecutionError` - Transient execution failures
- `ConnectionError` - Network issues
- `TimeoutError` - Time limits exceeded

**Non-retryable exceptions** (default `False`):

- `NotFoundError` - Missing items won't appear
- `ExistsError` - Duplicates won't resolve
- `ValidationError` - Invalid input won't change
- `ConfigurationError` - Config errors need manual fixes

**Usage with retry logic:**

```python
# noqa:validation
from lionherd_core.errors import LionherdError, ExecutionError
from lionherd_core import concurrency

async def retry_operation(operation, max_attempts=3):
    """Retry operation if error is retryable."""
    for attempt in range(max_attempts):
        try:
            return await operation()
        except LionherdError as e:
            if not e.retryable or attempt == max_attempts - 1:
                raise

            # Exponential backoff for retryable errors
            delay = 2 ** attempt
            print(f"Attempt {attempt + 1} failed (retryable), retrying in {delay}s...")
            await concurrency.sleep(delay)

# Usage
try:
    result = await retry_operation(fetch_data)
except LionherdError as e:
    if e.retryable:
        print("Operation failed after retries")
    else:
        print("Operation failed (non-retryable)")
```

**Override retryable flag:**

```python
from lionherd_core.errors import ExecutionError, NotFoundError

# Force non-retryable
raise ExecutionError(
    "Critical failure - do not retry",
    retryable=False  # Override default True
)

# Force retryable (rare)
raise NotFoundError(
    "Item temporarily unavailable",
    retryable=True  # Override default False
)
```

---

### .details Dictionary

The `.details` dict provides structured context for debugging and logging.

**Best practices:**

1. **Include identifiers**: UUIDs, names, IDs
2. **Add operation context**: What was being attempted
3. **Provide relevant state**: Configuration, parameters
4. **Use JSON-serializable types**: Strings, numbers, booleans, lists, dicts

**Example:**

```python
from lionherd_core.errors import NotFoundError

# Good: Structured context
raise NotFoundError(
    "Node not found in graph",
    details={
        "node_id": str(node_id),
        "graph_id": str(graph.id),
        "graph_size": len(graph),
        "operation": "get_node"
    }
)

# Avoid: Non-serializable objects
raise NotFoundError(
    "Node not found",
    details={"node": node}  # ❌ Node object not JSON-serializable
)

# Better: Extract serializable data
raise NotFoundError(
    "Node not found",
    details={"node_id": str(node.id), "node_content": str(node.content)}
)
```

**Logging with details:**

```python
from lionherd_core.errors import NotFoundError
import logging

try:
    item = pile[uuid]
except NotFoundError as e:
    # Structured logging
    logging.error(
        "Item not found",
        extra=e.to_dict()  # {"error": "NotFoundError", "message": ..., "details": {...}}
    )
```

---

### .**cause** Chaining

The `__cause__` attribute preserves the original exception that triggered this error.

**Pattern:**

```python
from lionherd_core.errors import NotFoundError

try:
    # Low-level operation
    data = dict_lookup[key]
except KeyError as e:
    # Raise domain-specific exception with cause
    raise NotFoundError(
        f"Item {key} not found",
        details={"key": key}
    ) from e  # Preserves KeyError as __cause__
```

**Benefits:**

1. **Full stack trace**: See both exceptions
2. **Root cause analysis**: Inspect original error
3. **Debugging workflow**: Trace through exception chain

**Example:**

```python
from lionherd_core.errors import NotFoundError, ExecutionError

# Multi-layer exception chain
async def fetch_user(user_id):
    try:
        # Database layer (KeyError)
        user_data = db_cache[user_id]
    except KeyError as e:
        # Domain layer (NotFoundError)
        raise NotFoundError(
            f"User {user_id} not in cache",
            details={"user_id": user_id}
        ) from e

async def process_user(user_id):
    try:
        # Application layer
        user = await fetch_user(user_id)
    except NotFoundError as e:
        # Service layer (ExecutionError)
        raise ExecutionError(
            "User processing failed",
            details={"user_id": user_id, "reason": "not_found"}
        ) from e

# Exception chain:
# ExecutionError
#   ↳ NotFoundError (.__cause__)
#       ↳ KeyError (.__cause__)
```

**Inspecting exception chain:**

```python
from lionherd_core.errors import ExecutionError

try:
    await process_user(123)
except ExecutionError as e:
    print(f"Top-level error: {e}")
    print(f"Caused by: {e.__cause__}")  # NotFoundError
    print(f"Root cause: {e.__cause__.__cause__}")  # KeyError

    # Walk the chain
    current = e
    while current:
        print(f"  - {type(current).__name__}: {current}")
        current = current.__cause__
```

---

## Error Handling Patterns

### Pattern 1: Single-Lookup with Exception Transformation

Prefer single dictionary access with exception transformation over double lookup.

**Before (double lookup):**

```python
# ❌ Inefficient: Two dict lookups
if uuid not in pile:
    raise ValueError("Item not found")
item = pile[uuid]
```

**After (single lookup):**

```python
# ✅ Efficient: One dict lookup
try:
    item = pile[uuid]
except KeyError as e:
    raise NotFoundError(
        f"Item {uuid} not found",
        details={"uuid": str(uuid)}
    ) from e
```

**Benefits:**

- 2x fewer lookups (performance)
- Exception chain preserved (`.__cause__`)
- Better error messages (domain context)

---

### Pattern 2: Retry Strategy with .retryable Flag

Use `.retryable` flag to decide retry strategy.

```python
# noqa:validation
from lionherd_core.errors import LionherdError
from lionherd_core import concurrency

async def resilient_operation(operation, max_attempts=3):
    """Execute operation with retry logic for transient failures."""
    last_error = None

    for attempt in range(max_attempts):
        try:
            return await operation()
        except LionherdError as e:
            last_error = e

            # Check if error is retryable
            if not e.retryable:
                # Non-retryable error - fail immediately
                raise

            # Don't retry on last attempt
            if attempt == max_attempts - 1:
                raise

            # Exponential backoff for retryable errors
            delay = 2 ** attempt
            print(f"Attempt {attempt + 1} failed: {e.message}")
            print(f"Error is retryable, waiting {delay}s before retry...")
            await concurrency.sleep(delay)

    # Should never reach here
    raise last_error

# Usage
from lionherd_core.errors import ConnectionError, NotFoundError

async def fetch_data():
    # Simulate transient network error
    raise ConnectionError("Network timeout", retryable=True)

try:
    # Will retry up to 3 times
    result = await resilient_operation(fetch_data)
except ConnectionError as e:
    print(f"Failed after retries: {e}")

async def get_item():
    # Non-retryable error
    raise NotFoundError("Item missing", retryable=False)

try:
    # Will fail immediately (not retryable)
    result = await resilient_operation(get_item)
except NotFoundError as e:
    print(f"Failed immediately (non-retryable): {e}")
```

---

### Pattern 3: Exception Aggregation with ExceptionGroup

Collect multiple errors from batch operations using `ExceptionGroup` (Python 3.11+).

```python
from lionherd_core.errors import ExistsError, NotFoundError

def batch_add_items(flow, items, progression_id):
    """Add multiple items to flow, collecting all errors."""
    errors = []

    for item in items:
        try:
            # Validate item doesn't exist
            if item.id in flow.items:
                raise ExistsError(
                    f"Item {item.id} already exists",
                    details={"item_id": str(item.id)}
                )

            # Validate progression exists
            if progression_id not in flow.progressions:
                raise NotFoundError(
                    f"Progression {progression_id} not found",
                    details={"progression_id": progression_id}
                )

            # Add item
            flow.add_item(item, progression_ids=[progression_id])

        except (ExistsError, NotFoundError) as e:
            # Collect error, continue processing
            errors.append(e)

    # Raise all errors together
    if errors:
        raise ExceptionGroup("Batch operation failed", errors)

# Usage
try:
    batch_add_items(flow, items, "active")
except ExceptionGroup as eg:
    print(f"Batch failed with {len(eg.exceptions)} errors:")
    for exc in eg.exceptions:
        print(f"  - {type(exc).__name__}: {exc.message}")
        if exc.details:
            print(f"    Details: {exc.details}")

        # Selective handling
        if isinstance(exc, ExistsError):
            print("    → Item already exists, skip")
        elif isinstance(exc, NotFoundError):
            print("    → Progression missing, create it?")
```

---

### Pattern 4: Structured Logging with .to_dict()

Use `.to_dict()` for structured logging and monitoring.

```python
# noqa:validation
import logging
import json
from lionherd_core.errors import LionherdError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)

async def monitored_operation(operation_name, **kwargs):
    """Execute operation with structured error logging."""
    try:
        result = await perform_operation(**kwargs)

        # Success logging
        logger.info(json.dumps({
            "event": "operation_success",
            "operation": operation_name,
            "params": kwargs
        }))

        return result

    except LionherdError as e:
        # Structured error logging
        error_data = e.to_dict()
        error_data["event"] = "operation_error"
        error_data["operation"] = operation_name
        error_data["params"] = kwargs

        logger.error(json.dumps(error_data))

        # Re-raise for caller
        raise

# Usage
try:
    await monitored_operation("fetch_user", user_id=123)
except LionherdError as e:
    # Already logged with full context
    pass

# Log output:
# {
#   "event": "operation_error",
#   "operation": "fetch_user",
#   "params": {"user_id": 123},
#   "error": "NotFoundError",
#   "message": "User not found",
#   "retryable": false,
#   "details": {"user_id": 123}
# }
```

---

## Migration Guide

### From ValueError to Semantic Exceptions

Version 1.0.0-alpha4 introduced semantic exceptions to replace generic `ValueError`.

#### Before (v0.x)

```python
# Collection access
if uuid not in pile:
    raise ValueError(f"Item {uuid} not found")

# Duplicate prevention
if item.id in pile:
    raise ValueError("Item already exists")

# Graph operations
if node_id not in graph:
    raise ValueError("Node not found")
```

#### After (v1.0.0+)

```python
from lionherd_core.errors import NotFoundError, ExistsError

# Collection access
try:
    item = pile[uuid]
except KeyError as e:
    raise NotFoundError(
        f"Item {uuid} not found",
        details={"uuid": str(uuid)}
    ) from e

# Duplicate prevention
if item.id in pile:
    raise ExistsError(
        "Item already exists",
        details={"item_id": str(item.id)}
    )

# Graph operations
try:
    node = graph.nodes[node_id]
except KeyError as e:
    raise NotFoundError(
        "Node not found",
        details={"node_id": str(node_id)}
    ) from e
```

### Migration Checklist

1. **Find all `raise ValueError`**: Search codebase for `ValueError` raises
2. **Classify error type**:
   - Item not found → `NotFoundError`
   - Item already exists → `ExistsError`
   - Invalid input → `ValidationError`
   - Invalid config → `ConfigurationError`
3. **Add structured context**: Replace string message with `details` dict
4. **Preserve exception chain**: Use `from e` to preserve `.__cause__`
5. **Update error handling**: Catch specific exception types instead of `ValueError`

### Breaking Changes

**Removed:**

- Generic `ValueError` for missing/duplicate items

**Added:**

- `NotFoundError` for missing items
- `ExistsError` for duplicates
- `.retryable` flag for retry logic
- `.details` dict for structured context
- `.to_dict()` for serialization

**Migration impact:**

```python
from lionherd_core.errors import NotFoundError

# Old code will break
try:
    item = pile[uuid]
except ValueError:  # ❌ No longer raised
    handle_missing()

# Update to new exceptions
try:
    item = pile[uuid]
except NotFoundError:  # ✅ Semantic exception
    handle_missing()
```

---

## See Also

- **Protocols**: `Serializable` protocol implemented by `LionherdError`
- **Base Collections**: `Pile`, `Graph`, `Flow` raise semantic exceptions
- **Concurrency Patterns**: `retry()` uses `.retryable` flag
- **Testing**: Error handling in test suite (`tests/test_errors.py`)

---

**Version**: 1.0.0-alpha4+
**Breaking Changes**: Yes (ValueError → NotFoundError/ExistsError)
**Protocol Support**: `Serializable` (`.to_dict()`)

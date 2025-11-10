# Concurrency Tutorials

Master async/await patterns using lionherd-core's `libs.concurrency` module. Learn production-ready patterns for timeout management, graceful shutdown, resource leak detection, and error handling in async Python applications.

## Overview

These tutorials teach you to build resilient async systems using:
- **AnyIO TaskGroups**: Structured concurrency with automatic cleanup
- **Timeout utilities**: `move_on_after`, `fail_at`, deadline propagation
- **Resource management**: `shield()`, `LeakTracker` for connections and handles
- **Production patterns**: Circuit breakers, rate limiting, graceful shutdown

## Prerequisites

- Python 3.11+
- Basic understanding of async/await
- Familiarity with `asyncio` or `trio` (helpful but not required)

## Quick Start

```bash
pip install lionherd-core
jupyter notebook parallel_timeout.ipynb
```

## Tutorials (13)

### Beginner-Friendly (Start Here)

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [**Parallel Timeout**](./parallel_timeout.ipynb) | 20-30min | Run multiple async ops with individual timeouts using `move_on_after` |
| [**Deadline Task Queue**](./deadline_task_queue.ipynb) | 20-30min | Process queues with hard deadlines using `fail_at` |
| [**Batch Partial Failure**](./batch_partial_failure.ipynb) | 20-30min | Handle partial failures in batch processing without stopping remaining tasks |

### Intermediate Patterns

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [**Circuit Breaker**](./circuit_breaker_timeout.ipynb) | 20-30min | Isolate failing services with timeout-based circuit breaker pattern |
| [**DB Transactions Shielded**](./db_transactions_shielded.ipynb) | 20-30min | Protect commit/rollback operations from cancellation using `shield()` |
| [**File Handle Tracking**](./file_handle_tracking.ipynb) | 20-25min | Detect file handle leaks using `LeakTracker` |
| [**Rate-Limited Batch**](./rate_limited_batch.ipynb) | 20-30min | Implement token bucket rate limiting for batch processing |

### Advanced Production Patterns

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [**Graceful Shutdown**](./graceful_shutdown.ipynb) | 20-30min | Handle shutdown signals and cleanup with error tolerance |
| [**Connection Pool Leak**](./connection_pool_leak.ipynb) | 25-35min | Build leak-proof connection pools with `LeakTracker` |
| [**Lock Debugging**](./lock_debugging.ipynb) | 25-35min | Debug deadlocks and track lock acquisition patterns |
| [**Service Lifecycle**](./service_lifecycle.ipynb) | 25-35min | Manage service startup, health checks, and shutdown |
| [**Fan-Out/Fan-In**](./fan_out_fan_in.ipynb) | 25-35min | Distribute work across worker pools and aggregate results |
| [**Deadline Worker Pool**](./deadline_worker_pool.ipynb) | 25-35min | Build worker pools with deadline propagation |

## Learning Path

### Path 1: Essential Timeout Patterns (1-2 hours)
1. **Parallel Timeout** - Learn `move_on_after` for soft timeouts
2. **Deadline Task Queue** - Learn `fail_at` for hard deadlines
3. **Circuit Breaker** - Combine timeouts with failure isolation

**Outcome**: Handle timeouts gracefully in production async code

### Path 2: Resource Management (2-3 hours)
1. **File Handle Tracking** - Understand `LeakTracker` basics
2. **DB Transactions Shielded** - Learn when to use `shield()`
3. **Connection Pool Leak** - Build production-grade resource pools

**Outcome**: Prevent resource leaks in long-running services

### Path 3: Production Service Patterns (3-4 hours)
1. **Service Lifecycle** - Service management fundamentals
2. **Graceful Shutdown** - Handle cleanup and signals
3. **Fan-Out/Fan-In** - Distribute and aggregate work
4. **Rate-Limited Batch** - Control request rates

**Outcome**: Build production-ready async services

## Key Concepts

### Structured Concurrency (AnyIO TaskGroups)

All tutorials use AnyIO's TaskGroup API for structured concurrency:

```python
from lionherd_core.libs.concurrency import create_task_group

async with create_task_group() as tg:
    tg.start_soon(task1)
    tg.start_soon(task2)
# Automatic wait + cleanup + error propagation
```

**Benefits**:
- Automatic task cleanup (no orphaned tasks)
- Exception propagation from children to parent
- Guaranteed resource cleanup on exit

### Timeout Strategies

| Pattern | Use When | Tutorial |
|---------|----------|----------|
| `move_on_after` | Soft timeout (continue on timeout) | [Parallel Timeout](./parallel_timeout.ipynb) |
| `fail_at` | Hard deadline (raise on timeout) | [Deadline Task Queue](./deadline_task_queue.ipynb) |
| `shield` | Protect critical sections from cancellation | [DB Transactions](./db_transactions_shielded.ipynb) |

### Resource Leak Detection

`LeakTracker` helps detect resource leaks (connections, file handles, locks):

```python
from lionherd_core.libs.concurrency import LeakTracker

tracker = LeakTracker()

async with tracker.track(resource_id):
    # Use resource
# Warns if not properly cleaned up
```

See [Connection Pool Leak](./connection_pool_leak.ipynb) for comprehensive examples.

## Common Patterns

### Pattern 1: Parallel with Timeouts
Execute multiple async operations with individual timeouts (won't block each other):

```python
from lionherd_core.libs.concurrency import move_on_after, create_task_group

async with create_task_group() as tg:
    for item in batch:
        async with move_on_after(5.0):  # 5s per item
            await tg.start_soon(process, item)
```

**Tutorial**: [Parallel Timeout](./parallel_timeout.ipynb)

### Pattern 2: Circuit Breaker
Isolate failing services to prevent cascade failures:

```python
class CircuitBreaker:
    def __init__(self, timeout: float, threshold: int):
        self.timeout = timeout
        self.failure_count = 0
        self.threshold = threshold
        self.open = False

    async def call(self, func, *args):
        if self.open:
            raise CircuitOpenError()

        try:
            async with move_on_after(self.timeout):
                return await func(*args)
        except TimeoutError:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.open = True
            raise
```

**Tutorial**: [Circuit Breaker](./circuit_breaker_timeout.ipynb)

### Pattern 3: Graceful Shutdown
Handle shutdown signals and cleanup with error tolerance:

```python
import signal
from lionherd_core.libs.concurrency import create_task_group

shutdown_event = anyio.Event()

def handle_signal(signum, frame):
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_signal)

async with create_task_group() as tg:
    tg.start_soon(service.run)
    await shutdown_event.wait()
    tg.cancel_scope.cancel()  # Graceful cancel
# TaskGroup handles cleanup
```

**Tutorial**: [Graceful Shutdown](./graceful_shutdown.ipynb)

## Production Considerations

### Error Handling

All tutorials demonstrate production-grade error handling:
- Partial failure tolerance (don't stop on single task failure)
- Retry with exponential backoff
- Error aggregation and reporting
- Graceful degradation

### Performance

- **TaskGroup overhead**: <1ms per task (AnyIO optimization)
- **LeakTracker overhead**: <0.1ms per track() call
- **Timeout precision**: ±10ms (OS scheduler dependent)

### Testing

Use `anyio.from_thread.run()` for testing async code in sync tests:

```python
import anyio

def test_async_function():
    result = anyio.from_thread.run(async_function, arg1, arg2)
    assert result == expected
```

## Troubleshooting

### Common Issues

**Issue**: Tasks don't cancel when TaskGroup is cancelled
**Solution**: Ensure tasks check cancellation (`await checkpoint()`) and don't suppress `CancelledError`

**Issue**: Resource leaks despite using `LeakTracker`
**Solution**: Verify `async with tracker.track()` is used (context manager required)

**Issue**: Deadlocks in multi-lock code
**Solution**: Use [Lock Debugging](./lock_debugging.ipynb) tutorial patterns to track acquisition order

## Related Resources

- **API Reference**: [libs/concurrency](../../../docs/api/libs/concurrency/)
- **AnyIO Documentation**: https://anyio.readthedocs.io/
- **Structured Concurrency**: https://en.wikipedia.org/wiki/Structured_concurrency

## Contributing

Found issues or have suggestions? Open an issue at [lionherd-core GitHub](https://github.com/khive-ai/lionherd-core/issues).

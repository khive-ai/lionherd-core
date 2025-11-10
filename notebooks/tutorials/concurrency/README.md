# Concurrency Tutorials

Master async/await patterns using lionherd-core's `libs.concurrency` module. Learn production-ready patterns for deadline management, task coordination, and service lifecycle management in async Python applications.

## Overview

These consolidated tutorials teach you to build resilient async systems using:

- **TaskGroups**: Structured concurrency with automatic cleanup
- **Timeout utilities**: `move_on_at`, `fail_at`, deadline propagation
- **Resource management**: `shield()`, graceful shutdown patterns
- **Production patterns**: Worker pools, service coordination, event signaling

## Prerequisites

- Python 3.11+
- Basic understanding of async/await
- Familiarity with `asyncio` concepts (helpful but not required)

## Quick Start

```bash
pip install lionherd-core
jupyter notebook deadline_patterns.ipynb
```

## Tutorials (3)

> **Note**: These tutorials consolidate 7+ original notebooks into 3 focused, comprehensive guides. Each tutorial combines multiple related patterns for efficient learning.

### 1. Deadline-Aware Processing Patterns

**File**: [`deadline_patterns.ipynb`](./deadline_patterns.ipynb)
**Time**: 20 minutes
**Difficulty**: Intermediate

Learn to process work within fixed time budgets:

- Sequential deadline-aware processing (check time before each task)
- Parallel worker pool pattern (multiple workers, shared queue)
- Sentinel pattern for graceful shutdown
- Production-ready copypaste code

**Use Cases**: ETL pipelines, batch notifications, background jobs, API handlers with SLAs

**Key APIs**: `move_on_at()`, `effective_deadline()`, `current_time()`, `Queue.with_maxsize()`

---

### 2. Task Coordination Patterns

**File**: [`task_coordination.ipynb`](./task_coordination.ipynb)
**Time**: 20-30 minutes
**Difficulty**: Advanced

Master coordinating concurrent workers with proper lifecycle management:

- Fan-out/fan-in pattern (distribute work, collect results)
- Graceful shutdown with `shield()` (cleanup despite cancellation)
- Worker pool with two-phase shutdown
- Error vs cancellation distinction (`non_cancel_subgroup()`)

**Use Cases**: Worker pools, batch processing pipelines, multi-tenant systems, data processing

**Key APIs**: `TaskGroup`, `Queue`, `shield()`, `non_cancel_subgroup()`

---

### 3. Service Lifecycle Management

**File**: [`service_lifecycle.ipynb`](./service_lifecycle.ipynb)
**Time**: 25-35 minutes
**Difficulty**: Advanced

Build production-ready multi-component service managers:

- Multi-service coordination with dependencies (Database → Cache → API)
- Initialization protocol (`task_status.started()`)
- Health monitoring with Event coordination
- Background workers with queue processing
- Coordinated graceful shutdown

**Use Cases**: HTTP APIs, microservices, long-running services, production systems

**Key APIs**: `create_task_group()`, `TaskGroup.start()`, `Event`, `Queue`, `get_cancelled_exc_class()`

---

## Learning Paths

### Path 1: Essential Patterns (1-2 hours)

1. **Deadline Patterns** - Time-bounded processing basics
2. **Task Coordination** - Worker pools and shutdown
3. **Service Lifecycle** - Production service management

**Outcome**: Build production-ready async services with proper lifecycle management

### Path 2: Focused Skills (pick what you need)

- **Need deadline management?** → Start with Deadline Patterns
- **Need worker pools?** → Start with Task Coordination
- **Need multi-service coordination?** → Start with Service Lifecycle

## Key Concepts

### Structured Concurrency (TaskGroups)

All tutorials use lionherd-core's TaskGroup API:

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
| `move_on_at` | Soft timeout (return partial results) | Deadline Patterns |
| `effective_deadline` | Query remaining time | Deadline Patterns |
| `shield` | Protect cleanup from cancellation | Task Coordination |

### Worker Pool Patterns

| Pattern | Use When | Tutorial |
|---------|----------|----------|
| Sequential processing | Dependencies, rate limits | Deadline Patterns |
| Parallel worker pool | Independent I/O tasks | Deadline Patterns |
| Fan-out/fan-in | Distribute + aggregate | Task Coordination |
| Service lifecycle | Multi-component coordination | Service Lifecycle |

## Common Patterns

### Pattern 1: Deadline-Aware Processing

Process as much as possible before deadline:

```python
from lionherd_core.libs.concurrency import move_on_at, current_time

deadline = current_time() + 30.0
with move_on_at(deadline) as scope:
    for task in tasks:
        if (deadline - current_time()) < 0.01:
            break  # Not enough time
        result = await task()
```

**Tutorial**: [Deadline Patterns](./deadline_patterns.ipynb)

### Pattern 2: Worker Pool with Graceful Shutdown

Distribute work across workers with clean shutdown:

```python
from lionherd_core.libs.concurrency import Queue, create_task_group

async with create_task_group() as tg:
    queue = Queue.with_maxsize(100)

    # Spawn workers
    for i in range(num_workers):
        tg.start_soon(worker, i, queue)

    # Feed work
    for task in tasks:
        await queue.put(task)

    # Signal shutdown
    await queue.put(SENTINEL)
```

**Tutorial**: [Task Coordination](./task_coordination.ipynb)

### Pattern 3: Multi-Service Coordination

Start services in dependency order:

```python
async with create_task_group() as tg:
    # Sequential dependency chain
    db_status = await tg.start(database_service)
    cache_status = await tg.start(cache_service, db_status)
    api_status = await tg.start(api_service, db_status, cache_status)

    # All dependencies ready, start workers
    tg.start_soon(background_worker, queue)
```

**Tutorial**: [Service Lifecycle](./service_lifecycle.ipynb)

## Production Considerations

### Error Handling

All tutorials demonstrate:

- Partial failure tolerance (don't stop on single task failure)
- Error vs cancellation distinction
- Graceful degradation
- Production-ready error reporting

### Performance

- **TaskGroup overhead**: <1ms per task
- **Queue operations**: <0.1ms per put/get
- **Deadline checking**: <0.01ms per check

### Testing

All code cells are executable - run notebooks top-to-bottom to verify patterns work.

## Troubleshooting

### Common Issues

**Issue**: Tasks don't cancel when TaskGroup is cancelled
**Solution**: Ensure tasks don't suppress `CancelledError` / `get_cancelled_exc_class()`

**Issue**: Workers hang on shutdown
**Solution**: Use sentinel pattern or `queue.close()` + `anyio.EndOfStream` handling

**Issue**: Deadline reached but tasks continue
**Solution**: Check `effective_deadline()` before starting tasks, not during

## Related Resources

- **API Reference**: [libs/concurrency](../../../docs/api/libs/concurrency/)
- **lionherd-core Examples**: [GitHub Examples](https://github.com/khive-ai/lionherd-core/tree/main/examples)
- **Structured Concurrency**: [Nathaniel Smith's Blog](https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/)

## Contributing

Found issues or have suggestions? Open an issue at [lionherd-core GitHub](https://github.com/khive-ai/lionherd-core/issues).

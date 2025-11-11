# Concurrency Tutorials

Production-ready async/await patterns using lionherd-core's `libs.concurrency` module: deadline management, task coordination, resource leak detection, and service lifecycle.

## Overview

Build resilient async systems with:

- **TaskGroups**: Structured concurrency with automatic cleanup
- **Timeout utilities**: `move_on_at`, `fail_at`, deadline propagation
- **Resource management**: `LeakTracker`, `shield()`, graceful shutdown
- **Production patterns**: Worker pools, service coordination, transaction handling

## Prerequisites

- Python 3.11+
- Basic understanding of async/await
- Familiarity with `asyncio` (helpful but not required)

## Quick Start

```bash
pip install lionherd-core
jupyter notebook task_coordination.ipynb
```

## Tutorials (6)

### 1. Task Coordination Patterns

**File**: [`task_coordination.ipynb`](./task_coordination.ipynb)
**Time**: 20-30 min | **Difficulty**: 🔵 Intermediate

Master coordinating concurrent workers with proper lifecycle management:

- Fan-out/fan-in pattern (distribute work, collect results)
- Worker pool with graceful shutdown
- Error vs cancellation distinction
- Result aggregation with error handling

**Use Cases**: Worker pools, batch processing, multi-tenant systems, data pipelines

**Key APIs**: `TaskGroup`, `Queue`, `shield()`, `non_cancel_subgroup()`

---

### 2. Deadline Patterns

**File**: [`deadline_patterns.ipynb`](./deadline_patterns.ipynb)
**Time**: 20-30 min | **Difficulty**: 🔵 Intermediate

Process work within fixed time budgets:

- Sequential deadline-aware processing
- Parallel worker pool with deadlines
- Deadline propagation across tasks
- Sentinel pattern for graceful shutdown

**Use Cases**: ETL pipelines, batch notifications, API handlers with SLAs, background jobs

**Key APIs**: `move_on_at()`, `fail_at()`, `effective_deadline()`, `current_time()`

---

### 3. Error Handling & Timeouts

**File**: [`error_handling_timeouts.ipynb`](./error_handling_timeouts.ipynb)
**Time**: 20-30 min | **Difficulty**: 🔵 Intermediate

Robust timeout and error recovery:

- Soft timeouts with `move_on_after` (partial results)
- Hard timeouts with `fail_at` (failure on exceed)
- Timeout recovery strategies
- Partial failure handling in batch operations

**Use Cases**: API retry logic, batch processing with timeouts, resilient services

**Key APIs**: `move_on_after()`, `fail_at()`, exception handling patterns

---

### 4. Transaction Shielding

**File**: [`transaction_shielding.ipynb`](./transaction_shielding.ipynb)
**Time**: 20-30 min | **Difficulty**: 🔵 Intermediate

Protect critical operations from cancellation:

- `shield()` for commit/rollback patterns
- Transaction cleanup despite cancellation
- Database transaction handling
- Cleanup guarantee patterns

**Use Cases**: Database transactions, file operations, cleanup handlers, critical sections

**Key APIs**: `shield()`, context managers, cleanup protocols

---

### 5. Resource Leak Detection

**File**: [`resource_leak_detection.ipynb`](./resource_leak_detection.ipynb)
**Time**: 25-35 min | **Difficulty**: 🟠 Advanced

Track and detect resource leaks in production:

- `LeakTracker` for connection pools
- File handle leak detection
- Resource lifecycle tracking
- Leak reporting and debugging

**Use Cases**: Connection pool management, file handle tracking, debugging resource leaks

**Key APIs**: `LeakTracker`, resource context managers, leak detection patterns

---

### 6. Service Lifecycle Management

**File**: [`service_lifecycle.ipynb`](./service_lifecycle.ipynb)
**Time**: 25-35 min | **Difficulty**: 🟠 Advanced

Build production-ready multi-component services:

- Multi-service coordination with dependencies (Database → Cache → API)
- Initialization protocol and health monitoring
- Background workers with queue processing
- Coordinated graceful shutdown

**Use Cases**: HTTP APIs, microservices, long-running services, production systems

**Key APIs**: `create_task_group()`, `TaskGroup.start()`, `Event`, `Queue`

---

## Learning Paths

### New to Concurrency

1. **Task Coordination** - Basic worker pool patterns
2. **Error Handling & Timeouts** - Resilience basics
3. **Deadline Patterns** - Time-bounded processing

### Production Deployment

1. **Service Lifecycle** - Multi-component management
2. **Resource Leak Detection** - Debug production issues
3. **Transaction Shielding** - Critical operation safety

### Specific Needs

- **Need deadline management?** → Deadline Patterns
- **Need worker pools?** → Task Coordination
- **Need transaction safety?** → Transaction Shielding
- **Need leak detection?** → Resource Leak Detection

## Key Concepts

### Structured Concurrency (TaskGroups)

```python
from lionherd_core.libs.concurrency import create_task_group

async with create_task_group() as tg:
    tg.start_soon(task1)
    tg.start_soon(task2)
# Automatic wait + cleanup + error propagation
```

**Benefits**: Automatic cleanup, exception propagation, guaranteed resource cleanup

### Timeout Strategies

| Pattern | Use When | Tutorial |
|---------|----------|----------|
| `move_on_at` | Soft timeout (return partial results) | Deadline Patterns, Error Handling |
| `fail_at` | Hard timeout (fail on exceed) | Deadline Patterns, Error Handling |
| `effective_deadline` | Query remaining time | Deadline Patterns |
| `shield` | Protect cleanup from cancellation | Transaction Shielding |

### Resource Management

| Pattern | Use When | Tutorial |
|---------|----------|----------|
| `LeakTracker` | Track connection/file leaks | Resource Leak Detection |
| `shield` | Protect critical operations | Transaction Shielding |
| Graceful shutdown | Service cleanup | Service Lifecycle |

## Common Patterns

### Pattern 1: Deadline-Aware Processing

```python
from lionherd_core.libs.concurrency import move_on_at, current_time

deadline = current_time() + 30.0  # 30 second budget
async with move_on_at(deadline):
    for item in work_queue:
        await process(item)
```

### Pattern 2: Worker Pool

```python
from lionherd_core.libs.concurrency import create_task_group, Queue

async with create_task_group() as tg:
    queue = Queue.with_maxsize(100)
    for _ in range(5):  # 5 workers
        tg.start_soon(worker, queue)
    await producer(queue)
```

### Pattern 3: Resource Leak Detection

```python
from lionherd_core.libs.concurrency import LeakTracker

tracker = LeakTracker()
conn = await pool.acquire()
tracker.acquire("connection", conn)
# ... use connection ...
tracker.release("connection", conn)
tracker.check_leaks()  # Detect unreleased resources
```

## API Documentation

- [libs.concurrency](../../../docs/api/libs/concurrency/) - Complete API reference
- [TaskGroup](../../../docs/api/libs/concurrency/task.md) - TaskGroup API
- [Timeout utilities](../../../docs/api/libs/concurrency/patterns.md) - Timeout patterns
- [LeakTracker](../../../docs/api/libs/concurrency/resource_tracker.md) - Resource tracking

## Related

- [ln Utilities](../ln_utilities/) - Data transformation patterns
- [String Handlers](../string_handlers/) - String similarity algorithms

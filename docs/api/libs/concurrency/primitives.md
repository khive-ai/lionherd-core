# Concurrency Primitives

> Async concurrency primitives built on AnyIO for cross-backend async coordination

## Overview

The `lionherd_core.libs.concurrency` module provides **async-first concurrency primitives** that work across different async backends (asyncio, trio) through AnyIO. These primitives enable safe coordination between concurrent tasks with familiar synchronization patterns.

**Key Capabilities:**

- **Cross-Backend Compatibility**: Works with asyncio, trio, and other AnyIO-supported backends
- **Async/Await Native**: Designed for modern async Python patterns
- **Context Manager Support**: All primitives support `async with` for automatic cleanup
- **Type-Safe**: Full type hints with generic support where applicable
- **Zero-Overhead Wrappers**: Thin wrappers around AnyIO primitives for consistency

**Available Primitives:**

- **Lock**: Mutual exclusion for protecting critical sections
- **Semaphore**: Limited resource access with configurable capacity
- **CapacityLimiter**: Token-based resource limiting with borrower tracking
- **Queue**: Async FIFO queue for inter-task communication
- **Event**: Boolean flag for task signaling and coordination
- **Condition**: Advanced synchronization with wait/notify patterns

## Import

```python
from lionherd_core.libs.concurrency import (
    Lock,
    Semaphore,
    CapacityLimiter,
    Queue,
    Event,
    Condition,
)
```

---

## Lock

Async mutex lock for mutual exclusion.

### Class Signature

```python
class Lock:
    """Async mutex lock."""

    def __init__(self) -> None: ...
    async def acquire(self) -> None: ...
    def release(self) -> None: ...
```

### Methods

#### `acquire()`

Acquire lock, blocking until available.

**Signature:**

```python
async def acquire(self) -> None: ...
```

**Examples:**

```python
lock = Lock()
await lock.acquire()
try:
    # Critical section
    pass
finally:
    lock.release()
```

#### `release()`

Release lock, allowing waiting tasks to proceed.

**Signature:**

```python
def release(self) -> None: ...
```

**Notes:**

Must be called from the same task that acquired the lock.

### Context Manager

Supports `async with` for automatic lock management.

```python
lock = Lock()
async with lock:
    # Lock acquired automatically
    # Critical section
    pass
# Lock released automatically
```

### Usage Patterns

#### Basic Mutual Exclusion

```python
from lionherd_core.libs.concurrency import Lock, sleep

lock = Lock()
shared_resource = []

async def worker(task_id: int):
    async with lock:
        # Only one task can execute this block at a time
        shared_resource.append(task_id)
        await sleep(0.1)
        print(f"Task {task_id}: {shared_resource}")
```

---

## Semaphore

Async semaphore for limiting concurrent access to resources.

### Class Signature

```python
class Semaphore:
    """Async semaphore."""

    def __init__(self, initial_value: int) -> None: ...
    async def acquire(self) -> None: ...
    def release(self) -> None: ...
```

### Parameters

#### Constructor Parameters

**initial_value** : int

Number of available slots. Must be >= 0.

**Raises:**

- ValueError: If `initial_value < 0`

### Methods

#### `acquire()`

Acquire semaphore slot, blocking until available.

**Signature:**

```python
async def acquire(self) -> None: ...
```

**Examples:**

```python
sem = Semaphore(3)
await sem.acquire()  # Blocks if 3 tasks already acquired
try:
    # Resource access
    pass
finally:
    sem.release()
```

#### `release()`

Release semaphore slot, incrementing available count.

**Signature:**

```python
def release(self) -> None: ...
```

### Context Manager

```python
sem = Semaphore(3)
async with sem:
    # Slot acquired automatically
    # Resource access (max 3 concurrent tasks)
    pass
# Slot released automatically
```

### Usage Patterns

#### Rate Limiting Concurrent Operations

```python
# noqa:validation
from lionherd_core.libs.concurrency import Semaphore, gather, sleep

# Allow max 5 concurrent API calls
api_limiter = Semaphore(5)

async def fetch_data(url: str):
    """Simulate API call."""
    async with api_limiter:
        # Only 5 tasks can execute this block concurrently
        await sleep(0.1)  # Simulate network delay
        return {"url": url, "data": f"result_from_{url}"}

# Launch 100 tasks, but only 5 run concurrently
tasks = [fetch_data(f"https://api.example.com/data/{i}") for i in range(100)]
results = await gather(*tasks)
```

---

## CapacityLimiter

Async capacity limiter with token-based resource management.

### Class Signature

```python
class CapacityLimiter:
    """Async capacity limiter."""

    def __init__(self, total_tokens: float) -> None: ...
    async def acquire(self) -> None: ...
    def release(self) -> None: ...
    async def acquire_on_behalf_of(self, borrower: object) -> None: ...
    def release_on_behalf_of(self, borrower: object) -> None: ...
```

### Parameters

#### Constructor Parameters

**total_tokens** : float

Total capacity available. Must be > 0.

**Raises:**

- ValueError: If `total_tokens <= 0`

### Attributes

| Attribute           | Type    | Description                                        |
| ------------------- | ------- | -------------------------------------------------- |
| `total_tokens`      | `float` | Total capacity limit (get/set property)           |
| `available_tokens`  | `float` | Currently available capacity (read-only)          |
| `borrowed_tokens`   | `float` | Currently borrowed capacity (read-only)           |
| `remaining_tokens`  | `float` | Available capacity (deprecated, use `available_tokens`) |

### Methods

#### `acquire()`

Acquire 1 token of capacity, blocking until available.

**Signature:**

```python
async def acquire(self) -> None: ...
```

#### `release()`

Release 1 token of capacity.

**Signature:**

```python
def release(self) -> None: ...
```

#### `acquire_on_behalf_of()`

Acquire capacity for a specific borrower, enabling per-borrower tracking.

**Signature:**

```python
async def acquire_on_behalf_of(self, borrower: object) -> None: ...
```

**Parameters:**

- `borrower` (object): Object borrowing capacity (tracked by identity)

**Examples:**

```python
limiter = CapacityLimiter(10.0)
task_obj = object()
await limiter.acquire_on_behalf_of(task_obj)
# ... work ...
limiter.release_on_behalf_of(task_obj)
```

#### `release_on_behalf_of()`

Release capacity for a specific borrower.

**Signature:**

```python
def release_on_behalf_of(self, borrower: object) -> None: ...
```

**Parameters:**

- `borrower` (object): Object releasing capacity

### Properties

#### `total_tokens` (get/set)

Get or set total capacity limit.

**Setter Raises:**

- ValueError: If value <= 0

**Examples:**

```python
limiter = CapacityLimiter(10.0)
print(limiter.total_tokens)  # 10.0

limiter.total_tokens = 20.0  # Increase capacity
```

#### `available_tokens`

Get currently available capacity.

**Examples:**

```python
limiter = CapacityLimiter(10.0)
await limiter.acquire()
print(limiter.available_tokens)  # 9.0
```

#### `borrowed_tokens`

Get currently borrowed capacity.

**Examples:**

```python
limiter = CapacityLimiter(10.0)
await limiter.acquire()
print(limiter.borrowed_tokens)  # 1.0
```

### Context Manager

```python
limiter = CapacityLimiter(10.0)
async with limiter:
    # 1 token acquired automatically
    # Work...
    pass
# Token released automatically
```

### Usage Patterns

#### Managing Limited Resources

```python
from lionherd_core.libs.concurrency import CapacityLimiter, sleep

# Limit total memory usage (100 MB = 100 tokens)
memory_limiter = CapacityLimiter(100.0)

async def process_batch(data: list):
    """Process batch with memory-based capacity limiting."""
    # Acquire tokens based on estimated memory usage
    batch_size = len(data)  # 1 token per MB
    for _ in range(batch_size):
        await memory_limiter.acquire()

    try:
        # Process data
        await sleep(0.2)  # Simulate computation
        result = {"batch_size": len(data), "processed": True}
        return result
    finally:
        for _ in range(batch_size):
            memory_limiter.release()
```

#### Dynamic Capacity Adjustment

```python
from lionherd_core.libs.concurrency import sleep, CapacityLimiter
import random

limiter = CapacityLimiter(10.0)

# Monitor and adjust capacity based on system load
async def adjust_capacity():
    """Dynamically adjust capacity based on simulated load."""
    while True:
        # Simulate getting system load
        await sleep(1.0)
        system_load = random.uniform(0.0, 1.0)

        if system_load > 0.8:
            limiter.total_tokens = 5.0  # Reduce capacity
        else:
            limiter.total_tokens = 20.0  # Increase capacity
        await sleep(60)
```

---

## Queue

Async FIFO queue for inter-task communication.

### Class Signature

```python
@dataclass
class Queue(Generic[T]):
    """Async FIFO queue."""

    @classmethod
    def with_maxsize(cls, maxsize: int) -> Queue[T]: ...
    async def put(self, item: T) -> None: ...
    def put_nowait(self, item: T) -> None: ...
    async def get(self) -> T: ...
    def get_nowait(self) -> T: ...
    async def close(self) -> None: ...
```

### Class Methods

#### `with_maxsize()`

Create queue with maximum size limit.

**Signature:**

```python
@classmethod
def with_maxsize(cls, maxsize: int) -> Queue[T]: ...
```

**Parameters:**

- `maxsize` (int): Maximum number of items in queue

**Returns:**

- Queue[T]: New queue instance

**Examples:**

```python
# Create queue that holds max 100 items
queue = Queue.with_maxsize(100)
```

### Methods

#### `put()`

Put item in queue, blocking if queue is full.

**Signature:**

```python
async def put(self, item: T) -> None: ...
```

**Parameters:**

- `item` (T): Item to add to queue

#### `put_nowait()`

Put item without blocking.

**Signature:**

```python
def put_nowait(self, item: T) -> None: ...
```

**Parameters:**

- `item` (T): Item to add to queue

**Raises:**

- anyio.WouldBlock: If queue is full

#### `get()`

Get item from queue, blocking if queue is empty.

**Signature:**

```python
async def get(self) -> T: ...
```

**Returns:**

- T: Item from queue (FIFO order)

#### `get_nowait()`

Get item without blocking.

**Signature:**

```python
def get_nowait(self) -> T: ...
```

**Returns:**

- T: Item from queue

**Raises:**

- anyio.WouldBlock: If queue is empty

#### `close()`

Close send and receive streams.

**Signature:**

```python
async def close(self) -> None: ...
```

**Notes:**

Should be called when queue is no longer needed to clean up resources.

### Properties

#### `sender`

Get underlying send stream for advanced usage.

**Returns:**

- anyio.abc.ObjectSendStream[T]: Send stream

#### `receiver`

Get underlying receive stream for advanced usage.

**Returns:**

- anyio.abc.ObjectReceiveStream[T]: Receive stream

### Context Manager

```python
queue = Queue.with_maxsize(100)
async with queue:
    await queue.put(item)
    item = await queue.get()
    # Streams closed automatically on exit
```

### Usage Patterns

#### Producer-Consumer Pattern

```python
# noqa:validation
from lionherd_core.libs.concurrency import Queue, gather, sleep

queue = Queue.with_maxsize(50)

async def producer():
    for i in range(100):
        await queue.put(i)
        print(f"Produced: {i}")
    await queue.put(None)  # Sentinel value

async def consumer():
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        # Process item
        await sleep(0.05)  # Simulate processing

async with queue:
    await gather(producer(), consumer())
```

#### Work Distribution

```python
from lionherd_core.libs.concurrency import gather, sleep, Queue

# Distribute work across multiple workers
queue = Queue.with_maxsize(100)

async def worker(worker_id: int):
    while True:
        task = await queue.get()
        if task is None:
            break
        # Process task
        await sleep(0.1)  # Simulate processing
        result = {"task_id": task, "status": "completed"}
        print(f"Worker {worker_id} processed {task}: {result}")

async def distribute_work(tasks: list):
    async with queue:
        # Add tasks
        for task in tasks:
            await queue.put(task)

        # Add sentinels for workers
        for _ in range(5):
            await queue.put(None)

        # Launch workers
        workers = [worker(i) for i in range(5)]
        await gather(*workers)
```

---

## Event

Async event for task signaling.

### Class Signature

```python
class Event:
    """Async event for task signaling."""

    def __init__(self) -> None: ...
    def set(self) -> None: ...
    def is_set(self) -> bool: ...
    async def wait(self) -> None: ...
    def statistics(self) -> anyio.EventStatistics: ...
```

### Methods

#### `set()`

Set event flag, waking all waiting tasks.

**Signature:**

```python
def set(self) -> None: ...
```

**Examples:**

```python
event = Event()
event.set()  # Wake all tasks waiting on this event
```

#### `is_set()`

Check if event flag is set.

**Signature:**

```python
def is_set(self) -> bool: ...
```

**Returns:**

- bool: True if event is set, False otherwise

**Examples:**

```python
event = Event()
print(event.is_set())  # False
event.set()
print(event.is_set())  # True
```

#### `wait()`

Wait until event flag is set.

**Signature:**

```python
async def wait(self) -> None: ...
```

**Examples:**

```python
event = Event()

async def waiter():
    print("Waiting...")
    await event.wait()  # Blocks until set() is called
    print("Event occurred!")
```

#### `statistics()`

Get event statistics.

**Signature:**

```python
def statistics(self) -> anyio.EventStatistics: ...
```

**Returns:**

- anyio.EventStatistics: Statistics about waiting tasks

### Usage Patterns

#### Coordinating Task Startup

```python
# noqa:validation
from lionherd_core.libs.concurrency import Event, gather, sleep

ready = Event()

async def setup():
    # Perform initialization
    await sleep(0.5)  # Simulate resource initialization
    print("Setup complete")
    ready.set()  # Signal that setup is complete

async def worker():
    await ready.wait()  # Wait for setup to complete
    # Start processing
    await sleep(0.2)  # Simulate work
    print("Worker processing")

await gather(setup(), worker(), worker(), worker())
```

#### One-Time Notification

```python
# noqa:validation
from lionherd_core.libs.concurrency import sleep, gather, Event

shutdown_event = Event()

async def monitor():
    while not shutdown_event.is_set():
        # Check health
        await sleep(0.5)  # Simulate health check
        print("Health check passed")
        await sleep(1)
    print("Shutdown initiated")

async def shutdown_handler():
    await sleep(10)
    shutdown_event.set()  # Trigger shutdown

await gather(monitor(), shutdown_handler())
```

---

## Condition

Async condition variable for advanced coordination.

### Class Signature

```python
class Condition:
    """Async condition variable."""

    def __init__(self, lock: Lock | None = None) -> None: ...
    async def acquire(self) -> None: ...
    def release(self) -> None: ...
    async def wait(self) -> None: ...
    def notify(self, n: int = 1) -> None: ...
    def notify_all(self) -> None: ...
    def statistics(self) -> anyio.ConditionStatistics: ...
```

### Parameters

#### Constructor Parameters

**lock** : Lock, optional

Lock to use for synchronization. If not provided, creates internal lock.

### Methods

#### `acquire()`

Acquire underlying lock.

**Signature:**

```python
async def acquire(self) -> None: ...
```

**Notes:**

Must be held when calling `wait()`, `notify()`, or `notify_all()`.

#### `release()`

Release underlying lock.

**Signature:**

```python
def release(self) -> None: ...
```

#### `wait()`

Wait until notified, atomically releasing lock.

**Signature:**

```python
async def wait(self) -> None: ...
```

**Notes:**

- Must hold lock before calling
- Lock is released while waiting
- Lock is reacquired before returning

**Examples:**

```python
condition = Condition()

async def waiter():
    async with condition:
        await condition.wait()  # Atomically releases and reacquires lock
        # Lock held here
```

#### `notify()`

Wake up to n waiting tasks.

**Signature:**

```python
def notify(self, n: int = 1) -> None: ...
```

**Parameters:**

- `n` (int, default 1): Number of tasks to wake

**Notes:**

Must hold lock when calling.

#### `notify_all()`

Wake up all waiting tasks.

**Signature:**

```python
def notify_all(self) -> None: ...
```

**Notes:**

Must hold lock when calling.

#### `statistics()`

Get condition statistics.

**Signature:**

```python
def statistics(self) -> anyio.ConditionStatistics: ...
```

**Returns:**

- anyio.ConditionStatistics: Statistics about waiting tasks

### Context Manager

```python
condition = Condition()
async with condition:
    # Lock acquired automatically
    await condition.wait()
    # Lock reacquired after wait
# Lock released automatically
```

### Usage Patterns

#### Producer-Consumer with Condition

```python
# noqa:validation
from lionherd_core.libs.concurrency import Condition, sleep, gather

condition = Condition()
queue = []

async def producer():
    for i in range(10):
        async with condition:
            queue.append(i)
            condition.notify()  # Wake one consumer
        await sleep(0.5)

async def consumer(consumer_id: int):
    while True:
        async with condition:
            while not queue:
                await condition.wait()  # Wait for items
            item = queue.pop(0)
        print(f"Consumer {consumer_id} got {item}")
        if item == 9:
            break

await gather(
    producer(),
    consumer(1),
    consumer(2),
)
```

#### Bounded Buffer

```python
condition = Condition()
buffer = []
MAX_SIZE = 10

async def producer():
    for i in range(100):
        async with condition:
            while len(buffer) >= MAX_SIZE:
                await condition.wait()  # Wait for space
            buffer.append(i)
            condition.notify()  # Wake consumers

async def consumer():
    count = 0
    while count < 100:
        async with condition:
            while not buffer:
                await condition.wait()  # Wait for items
            item = buffer.pop(0)
            condition.notify()  # Wake producers
        # Process item
        await sleep(0.05)  # Simulate processing
        count += 1
```

---

## Design Rationale

### Why Wrap AnyIO Primitives?

The module provides thin wrappers around AnyIO primitives for several reasons:

1. **Consistent API Surface**: Standardized interface across lionherd-core codebase
2. **Type Safety**: Explicit type hints for better IDE support and static analysis
3. **Future Flexibility**: Abstraction layer allows implementation changes without API breaks
4. **Documentation**: Centralized documentation for team patterns

### Why Generic Queue?

`Queue[T]` uses generics to provide type safety for queue items:

```python
# Type checker knows items are strings
string_queue: Queue[str] = Queue.with_maxsize(10)
await string_queue.put("hello")
item: str = await string_queue.get()  # Type-safe
```

This prevents runtime errors from incorrect item types.

### Why Separate acquire_on_behalf_of?

`CapacityLimiter.acquire_on_behalf_of()` enables per-borrower tracking, useful for:

1. **Debugging**: Identify which tasks hold capacity
2. **Fairness**: Prevent single borrower from monopolizing resources
3. **Observability**: Monitor capacity distribution across borrowers

### Why Both Blocking and Non-Blocking Queue Operations?

`Queue` provides both `put()`/`get()` (blocking) and `put_nowait()`/`get_nowait()` (non-blocking) for different use cases:

- **Blocking**: Natural backpressure when producer/consumer speed differs
- **Non-blocking**: Error handling when queue state matters (e.g., poll pattern)

## Common Pitfalls

### Pitfall 1: Forgetting to Release Locks

**Issue**: Acquiring lock without releasing causes deadlock.

```python
lock = Lock()
await lock.acquire()
# ... exception raised ...
# lock.release() never called → deadlock
```

**Solution**: Always use context manager or try/finally.

```python
# ✅ Correct: Context manager
async with lock:
    # Critical section
    pass

# ✅ Correct: try/finally
await lock.acquire()
try:
    # Critical section
    pass
finally:
    lock.release()
```

### Pitfall 2: notify() Without Lock

**Issue**: Calling `notify()` without holding lock causes errors.

```python
condition = Condition()

async def notifier():
    condition.notify()  # ❌ RuntimeError: lock not held
```

**Solution**: Hold lock when notifying.

```python
async def notifier():
    async with condition:
        condition.notify()  # ✅ Correct
```

### Pitfall 3: Forgetting to Close Queue

**Issue**: Not closing queue leaks resources.

```python
queue = Queue.with_maxsize(10)
# ... use queue ...
# Never closed → resource leak
```

**Solution**: Use context manager or explicitly close.

```python
# ✅ Correct: Context manager
async with Queue.with_maxsize(10) as queue:
    # Use queue
    pass

# ✅ Correct: Explicit close
queue = Queue.with_maxsize(10)
try:
    # Use queue
    pass
finally:
    await queue.close()
```

### Pitfall 4: Semaphore Initial Value Zero

**Issue**: `Semaphore(0)` blocks all acquires until release called.

```python
sem = Semaphore(0)
await sem.acquire()  # Blocks forever (no releases yet)
```

**Solution**: Use Event for 0/1 signaling, Semaphore for resource counting.

```python
# ✅ For signaling: Use Event
event = Event()
await event.wait()  # Blocks until set()
event.set()

# ✅ For resources: Use Semaphore with initial_value > 0
sem = Semaphore(5)
```

## See Also

- **AnyIO Documentation**: [https://anyio.readthedocs.io/](https://anyio.readthedocs.io/)
- **Related Modules**:
  - `lionherd_core.libs.concurrency.throttle`: Rate limiting utilities
  - `lionherd_core.base.broadcaster`: Event broadcasting with async primitives
  - `lionherd_core.base.eventbus`: Pub/sub messaging

## Examples

### Example 1: Rate-Limited Batch Processing

```python
# noqa:validation
from lionherd_core.libs.concurrency import Semaphore, Queue, gather, sleep

# Limit concurrent operations to 5
rate_limiter = Semaphore(5)
results_queue = Queue.with_maxsize(100)

async def fetch_and_process(item_id: int):
    async with rate_limiter:
        # Simulate API call
        await sleep(0.1)
        result = {"id": item_id, "data": f"processed_{item_id}"}
        await results_queue.put(result)

async def collect_results(count: int):
    results = []
    for _ in range(count):
        result = await results_queue.get()
        results.append(result)
    return results

# Process 100 items with max 5 concurrent
item_ids = range(100)

async with results_queue:
    fetch_tasks = [fetch_and_process(i) for i in item_ids]

    # Run all tasks and collect results
    _, results = await gather(
        gather(*fetch_tasks),
        collect_results(len(item_ids))
    )
    print(f"Processed {len(results)} items")
```

### Example 2: Graceful Shutdown Coordination

```python
# noqa:validation
from lionherd_core.libs.concurrency import Event, Lock, sleep, gather

shutdown_event = Event()
active_tasks = 0
active_lock = Lock()

async def worker(worker_id: int):
    global active_tasks

    async with active_lock:
        active_tasks += 1

    try:
        while not shutdown_event.is_set():
            # Simulate work
            await sleep(1)
            print(f"Worker {worker_id} processing...")
    finally:
        async with active_lock:
            active_tasks -= 1

async def shutdown_handler():
    # Wait then initiate shutdown
    await sleep(10)
    shutdown_event.set()

    # Wait for all workers to finish
    while True:
        async with active_lock:
            if active_tasks == 0:
                break
        await sleep(0.1)

    print("All workers stopped cleanly")

# Run 5 workers with graceful shutdown
workers = [worker(i) for i in range(5)]
await gather(*workers, shutdown_handler())
```

### Example 3: Dynamic Resource Pool

```python
# noqa:validation
from lionherd_core.libs.concurrency import CapacityLimiter, sleep, create_task_group
import random

# Resource pool with dynamic sizing
pool = CapacityLimiter(10.0)

class ResourceManager:
    def __init__(self):
        self.pool = pool
        self.monitoring = True

    async def adjust_capacity(self):
        while self.monitoring:
            # Simulate system load
            load = random.uniform(0, 1)

            if load > 0.9:
                # Reduce capacity under heavy load
                new_capacity = max(5.0, self.pool.total_tokens * 0.8)
            elif load < 0.3:
                # Increase capacity when idle
                new_capacity = min(20.0, self.pool.total_tokens * 1.2)
            else:
                new_capacity = self.pool.total_tokens

            self.pool.total_tokens = new_capacity
            print(f"Capacity: {new_capacity:.1f} (load: {load:.2f})")

            await sleep(2)

    async def use_resource(self, task_id: int):
        async with self.pool:
            print(f"Task {task_id} using resource "
                  f"({self.pool.borrowed_tokens}/{self.pool.total_tokens})")
            await sleep(0.5)  # Simulate work

manager = ResourceManager()

# Run with structured concurrency
async with create_task_group() as tg:
    tg.start_soon(manager.adjust_capacity)
    for i in range(20):
        tg.start_soon(manager.use_resource, i)
```

### Example 4: Multi-Stage Pipeline

```python
# noqa:validation
from lionherd_core.libs.concurrency import Queue, Event, gather, sleep

stage1_queue = Queue.with_maxsize(10)
stage2_queue = Queue.with_maxsize(10)
done_event = Event()

async def stage1_worker():
    """Fetch raw data"""
    for i in range(20):
        await sleep(0.1)  # Simulate fetch
        data = {"id": i, "value": f"raw_{i}"}
        await stage1_queue.put(data)
    await stage1_queue.put(None)  # Sentinel

async def stage2_worker():
    """Process data"""
    while True:
        data = await stage1_queue.get()
        if data is None:
            await stage2_queue.put(None)
            break
        # Transform data
        processed = {"id": data["id"], "result": data["value"].upper()}
        await stage2_queue.put(processed)

async def stage3_worker():
    """Save results"""
    results = []
    while True:
        result = await stage2_queue.get()
        if result is None:
            break
        results.append(result)
    print(f"Pipeline complete: {len(results)} items processed")
    done_event.set()

async with stage1_queue, stage2_queue:
    await gather(
        stage1_worker(),
        stage2_worker(),
        stage3_worker(),
    )
    await done_event.wait()
```

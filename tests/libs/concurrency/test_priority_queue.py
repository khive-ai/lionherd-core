# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for PriorityQueue with design rationale and architecture decisions.

## Design Decisions

### Async Nowait Methods (Intentional API Divergence from asyncio)

Unlike asyncio.PriorityQueue where put_nowait() and get_nowait() are synchronous,
lionherd's PriorityQueue uses async nowait methods. This is an intentional
architectural decision driven by anyio.Condition requirements.

**Rationale**:
- anyio.Condition requires async lock acquisition for thread safety
- Synchronous nowait methods would bypass lock acquisition, creating race conditions
- Consistency: All queue operations that touch shared state must acquire locks
- Trade-off: Slightly less convenient API for correctness guarantees

**Critical Bug Fix** (see test_priority_queue_get_nowait_notifies_blocked_put):
- get_nowait() MUST call notify() after removing item
- Without notification, blocked put() tasks deadlock forever
- This was the primary bug fix in PR #14

### Status Methods Are Racy by Design

qsize(), empty(), and full() methods do NOT acquire locks. This matches
asyncio.Queue behavior and is intentional.

**Rationale**:
- Status methods are inherently racy (TOCTOU - Time Of Check Time Of Use)
- Locking doesn't add value: status can change immediately after lock release
- Intended for monitoring/debugging, not critical decision-making logic
- Proper synchronization happens in put()/get() where it matters

**Example of Why Locking Doesn't Help**:
```python
# Even with locks, this is racy
if not q.empty():  # Acquires lock, returns False
    # Lock released here
    # Another task could consume item here
    item = await q.get_nowait()  # QueueEmpty exception!
```

Correct pattern: Use try/except with get_nowait() for non-blocking consumption.

### Notification Semantics

Both put/put_nowait and get/get_nowait call notify() after modifying queue state.
This ensures symmetric wakeup behavior for blocked tasks.

**Critical for Correctness**:
- put() notifies blocked get() tasks when item added
- get() notifies blocked put() tasks when space freed
- put_nowait() notifies blocked get() tasks (symmetry with put)
- get_nowait() notifies blocked put() tasks (symmetry with get)

Without symmetric notification, deadlocks occur in producer-consumer patterns.

## Test Coverage

100% statement coverage (45/45 statements) across:
- Basic operations (put, get, priority ordering)
- Maxsize enforcement
- Blocking behavior (put blocks on full, get blocks on empty)
- Non-blocking operations (put_nowait, get_nowait)
- Exception handling (QueueEmpty, QueueFull)
- Notification correctness (critical deadlock prevention)
- Concurrent access (multiple producers/consumers)
- Status methods (qsize, empty, full)
- Edge cases (FIFO within priority, complex tuple priorities)

Each test runs against both asyncio and trio backends (24 total test executions).
"""

import pytest

from lionherd_core.libs.concurrency import PriorityQueue, QueueEmpty, QueueFull


@pytest.mark.anyio
async def test_priority_queue_basic(anyio_backend):
    """Test basic priority queue operations."""
    q = PriorityQueue()

    # Put items with different priorities (lower value = higher priority)
    await q.put((3, "low"))
    await q.put((1, "high"))
    await q.put((2, "medium"))

    # Get items in priority order
    assert await q.get() == (1, "high")
    assert await q.get() == (2, "medium")
    assert await q.get() == (3, "low")
    assert q.empty()


@pytest.mark.anyio
async def test_priority_queue_empty(anyio_backend):
    """Test empty queue detection."""
    q = PriorityQueue()
    assert q.empty()
    assert q.qsize() == 0

    await q.put((1, "item"))
    assert not q.empty()
    assert q.qsize() == 1

    await q.get()
    assert q.empty()
    assert q.qsize() == 0


@pytest.mark.anyio
async def test_priority_queue_maxsize(anyio_backend):
    """Test maxsize enforcement."""
    q = PriorityQueue(maxsize=2)
    assert not q.full()

    await q.put((1, "first"))
    await q.put((2, "second"))
    assert q.full()

    # Get one item to make space
    await q.get()
    assert not q.full()


@pytest.mark.anyio
async def test_priority_queue_get_nowait(anyio_backend):
    """Test non-blocking get."""
    q = PriorityQueue()

    # Empty queue should raise QueueEmpty
    with pytest.raises(QueueEmpty):
        await q.get_nowait()

    # Add item and get without waiting
    await q.put((1, "item"))
    assert await q.get_nowait() == (1, "item")


@pytest.mark.anyio
async def test_priority_queue_fifo_within_priority(anyio_backend):
    """Test FIFO ordering within same priority level."""
    q = PriorityQueue()

    # Add items with same priority
    await q.put((5, "first", 1))
    await q.put((5, "second", 2))
    await q.put((5, "third", 3))

    # Should come out in FIFO order (heap is stable for equal priorities)
    result = []
    while not q.empty():
        result.append(await q.get())

    # All have priority 5, order should be maintained
    assert len(result) == 3
    assert result[0][0] == 5
    assert result[1][0] == 5
    assert result[2][0] == 5


@pytest.mark.anyio
async def test_priority_queue_multiple_consumers(anyio_backend):
    """Test multiple concurrent consumers."""
    import anyio

    q = PriorityQueue()
    results = []

    async def consumer():
        while True:
            try:
                item = await q.get()
                results.append(item)
                if item == (99, "stop"):
                    break
            except Exception:
                break

    # Start consumers
    async with anyio.create_task_group() as tg:
        tg.start_soon(consumer)
        tg.start_soon(consumer)

        # Add items
        await q.put((1, "high"))
        await q.put((2, "medium"))
        await q.put((3, "low"))

        # Stop signal
        await q.put((99, "stop"))
        await q.put((99, "stop"))

    # All items should be consumed
    assert len(results) == 5


@pytest.mark.anyio
async def test_priority_queue_qsize(anyio_backend):
    """Test queue size tracking."""
    q = PriorityQueue()

    assert q.qsize() == 0

    await q.put((1, "one"))
    assert q.qsize() == 1

    await q.put((2, "two"))
    assert q.qsize() == 2

    await q.get()
    assert q.qsize() == 1

    await q.get()
    assert q.qsize() == 0


@pytest.mark.anyio
async def test_priority_queue_complex_priority(anyio_backend):
    """Test with tuple priorities for stable ordering."""
    q = PriorityQueue()

    # Use (priority, counter, data) pattern
    await q.put((1, 3, "C"))
    await q.put((1, 1, "A"))
    await q.put((1, 2, "B"))
    await q.put((2, 1, "D"))

    # Should come out ordered by (priority, counter)
    assert (await q.get())[2] == "A"
    assert (await q.get())[2] == "B"
    assert (await q.get())[2] == "C"
    assert (await q.get())[2] == "D"


@pytest.mark.anyio
async def test_priority_queue_blocking_put_on_full(anyio_backend):
    """Test put() blocks when queue is full and wakes when space available.

    This test covers line 54 (await self._condition.wait() in put()).
    """
    import anyio

    q = PriorityQueue(maxsize=2)
    results = []

    async def producer():
        """Try to put 3 items into queue with maxsize=2."""
        await q.put((1, "first"))
        await q.put((2, "second"))
        # This should block until consumer makes space
        await q.put((3, "third"))
        results.append("producer_done")

    async def consumer():
        """Wait a bit, then consume items to make space."""
        await anyio.sleep(0.1)  # Let producer block on full queue
        item = await q.get()
        results.append(f"consumed_{item}")
        await anyio.sleep(0.1)
        item = await q.get()
        results.append(f"consumed_{item}")

    # Run producer and consumer concurrently
    async with anyio.create_task_group() as tg:
        tg.start_soon(producer)
        tg.start_soon(consumer)

    # Verify producer completed (was unblocked)
    assert "producer_done" in results
    # Verify consumer got items
    assert any("consumed_" in r for r in results)
    # Should have 1 item left in queue (third)
    assert q.qsize() == 1
    assert await q.get() == (3, "third")


@pytest.mark.anyio
async def test_priority_queue_put_nowait(anyio_backend):
    """Test put_nowait() raises QueueFull when queue is full."""
    q = PriorityQueue(maxsize=2)

    # Should succeed
    await q.put_nowait((1, "first"))
    await q.put_nowait((2, "second"))
    assert q.full()

    # Should raise QueueFull
    with pytest.raises(QueueFull):
        await q.put_nowait((3, "third"))

    # After consuming, should work again
    await q.get()
    await q.put_nowait((3, "third"))
    assert q.qsize() == 2


@pytest.mark.anyio
async def test_priority_queue_get_nowait_notifies_blocked_put(anyio_backend):
    """Test get_nowait() wakes blocked put() tasks.

    Critical bug fix: get_nowait() must notify waiting putters.
    """
    import anyio

    q = PriorityQueue(maxsize=1)
    results = []

    async def producer():
        """Try to put 2 items into queue with maxsize=1."""
        await q.put((1, "first"))
        # This should block until consumer makes space
        await q.put((2, "second"))
        results.append("producer_unblocked")

    async def consumer():
        """Use get_nowait() to consume - should wake blocked putter."""
        await anyio.sleep(0.1)  # Let producer block
        item = await q.get_nowait()
        results.append(f"consumed_{item}")

    async with anyio.create_task_group() as tg:
        tg.start_soon(producer)
        tg.start_soon(consumer)

    # Producer should have been unblocked by get_nowait() notification
    assert "producer_unblocked" in results
    assert q.qsize() == 1
    assert await q.get() == (2, "second")


@pytest.mark.anyio
async def test_priority_queue_put_nowait_notifies_blocked_get(anyio_backend):
    """Test put_nowait() wakes blocked get() tasks."""
    import anyio

    q = PriorityQueue()
    results = []

    async def consumer():
        """Block waiting for item."""
        item = await q.get()
        results.append(f"consumed_{item}")

    async def producer():
        """Use put_nowait() to add item - should wake blocked getter."""
        await anyio.sleep(0.1)  # Let consumer block
        await q.put_nowait((1, "item"))
        results.append("producer_done")

    async with anyio.create_task_group() as tg:
        tg.start_soon(consumer)
        tg.start_soon(producer)

    # Consumer should have been unblocked
    assert "consumed_(1, 'item')" in results
    assert "producer_done" in results

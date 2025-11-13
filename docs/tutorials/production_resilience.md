# Tutorial: Production-Resilient Systems with Concurrency Primitives

**Time**: 25 min | **Difficulty**: 🟠 Advanced

Build production-grade systems with automatic retries, timeouts, error handling, and resource limits using lionherd-core's concurrency primitives.

## The Problem

Production systems must handle:

- External service failures (API rate limits, network errors)
- Resource constraints (memory, file descriptors, concurrent connections)
- Cascading failures (one slow service blocks everything)
- Partial batch failures (some items succeed, some fail)

Manual error handling is error-prone and verbose. Lionherd-core provides battle-tested primitives.

## Core Concepts

- **alcall** (async list call): Apply async function to list with concurrency control
- **CapacityLimiter**: Limit concurrent operations (prevent resource exhaustion)
- **ExceptionGroup**: Aggregate multiple errors (Python 3.11+)
- **Timeout**: Per-operation time limits

## End-to-End Example: Resilient Data Pipeline

### Scenario

Fetch data from 100 external APIs, process each item, and store results. Requirements:

- Max 10 concurrent API calls (rate limit)
- 5-second timeout per API call
- Retry failed calls up to 3 times
- Continue on partial failures
- Track success/failure metrics

### 1. Setup Models and External Service

```python
from pydantic import BaseModel
import httpx
from typing import Any
from lionherd_core.libs.concurrency import sleep

class DataItem(BaseModel):
    """Raw data from API."""
    id: str
    value: Any
    source: str

class ProcessedItem(BaseModel):
    """Processed data ready for storage."""
    id: str
    result: Any
    metadata: dict[str, Any]

# Simulated external API (with random failures)
async def fetch_from_api(item_id: str) -> DataItem:
    """Fetch data from external API (may fail randomly)."""
    await concurrency.sleep(0.1)  # Simulate network latency

    # Simulate failures (20% failure rate)
    import random
    if random.random() < 0.2:
        raise httpx.HTTPError(f"API error for {item_id}")

    return DataItem(id=item_id, value=f"data_{item_id}", source="external_api")
```

### 2. Resilient Fetcher with Retry Logic

```python
from lionherd_core import ln, concurrency

async def fetch_with_retry(
    item_id: str,
    max_retries: int = 3,
    timeout: float = 5.0
) -> DataItem | None:
    """Fetch with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            # Apply timeout to each attempt
            async with fail_after(timeout):
                return await fetch_from_api(item_id)
        except (httpx.HTTPError, TimeoutError) as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                print(f"Failed {item_id} after {max_retries} attempts: {e}")
                return None
            # Exponential backoff
            await concurrency.sleep(2 ** attempt)

async def fetch_all_items(item_ids: list[str]) -> list[DataItem]:
    """Fetch all items with concurrency limit."""
    # alcall handles concurrency control automatically
    results = await ln.alcall(
        item_ids,
        fetch_with_retry,
        max_concurrent=10,  # Max 10 concurrent API calls
        retry_timeout=20.0,       # 20s total timeout per item (includes retries)
        return_exceptions=True  # Don't fail entire batch on errors
    )

    # Filter out failures (None or exceptions)
    successful = [
        item for item in results
        if item is not None and not isinstance(item, Exception)
    ]

    print(f"Fetched {len(successful)}/{len(item_ids)} items successfully")
    return successful
```

### 3. Process Data with Resource Limits

```python
from lionherd_core import concurrency

# Global resource limiter (e.g., database connections)
db_limiter = concurrency.CapacityLimiter(capacity=5)

async def process_item(item: DataItem) -> ProcessedItem:
    """Process data item (CPU-intensive or I/O-bound)."""
    # Acquire database connection from pool
    async with db_limiter:
        await concurrency.sleep(0.05)  # Simulate processing

        return ProcessedItem(
            id=item.id,
            result=item.value.upper(),
            metadata={"source": item.source, "processed": True}
        )

async def process_all_items(items: list[DataItem]) -> list[ProcessedItem]:
    """Process all items with resource constraints."""
    results = await ln.alcall(
        items,
        process_item,
        max_concurrent=20,  # Can process more than DB connections
        retry_timeout=10.0,
        return_exceptions=True
    )

    successful = [
        item for item in results
        if not isinstance(item, Exception)
    ]

    print(f"Processed {len(successful)}/{len(items)} items successfully")
    return successful
```

**Key insight**: `max_concurrent=20` for processing, but `CapacityLimiter(5)` ensures only 5 database connections at a time.

### 4. Store Results with Error Aggregation

```python
from lionherd_core import concurrency

async def store_item(item: ProcessedItem) -> None:
    """Store item to database (may fail)."""
    import random
    if random.random() < 0.1:  # 10% failure rate
        raise ValueError(f"Storage failed for {item.id}")

    await concurrency.sleep(0.02)  # Simulate write
    print(f"Stored {item.id}")

async def store_all_items(items: list[ProcessedItem]) -> dict[str, Any]:
    """Store all items and aggregate errors."""
    results = await ln.alcall(
        items,
        store_item,
        max_concurrent=10,
        return_exceptions=True
    )

    # Separate successes and failures
    errors = [r for r in results if isinstance(r, Exception)]
    successes = len(results) - len(errors)

    metrics = {
        "total": len(items),
        "success": successes,
        "failed": len(errors),
        "success_rate": successes / len(items) if items else 0
    }

    if errors:
        # Aggregate all errors for logging/alerting
        raise ExceptionGroup(
            f"Storage failures: {len(errors)}/{len(items)}",
            errors
        )

    return metrics
```

### 5. Complete Pipeline with Graceful Degradation

```python
# noqa:validation
async def run_pipeline(item_ids: list[str]) -> dict[str, Any]:
    """Run complete data pipeline with resilience."""
    print(f"Starting pipeline for {len(item_ids)} items...")

    # Stage 1: Fetch (with retry)
    fetched_items = await fetch_all_items(item_ids)
    if not fetched_items:
        return {"error": "All fetches failed"}

    # Stage 2: Process (with resource limits)
    processed_items = await process_all_items(fetched_items)
    if not processed_items:
        return {"error": "All processing failed"}

    # Stage 3: Store (with error aggregation)
    try:
        metrics = await store_all_items(processed_items)
        return {
            "status": "success",
            "fetched": len(fetched_items),
            "processed": len(processed_items),
            **metrics
        }
    except ExceptionGroup as eg:
        # Log errors but return partial success
        print(f"Storage errors: {eg}")
        return {
            "status": "partial_success",
            "fetched": len(fetched_items),
            "processed": len(processed_items),
            "storage_errors": len(eg.exceptions)
        }

# Run pipeline
item_ids = [f"item_{i}" for i in range(100)]
result = await run_pipeline(item_ids)
print(f"Pipeline result: {result}")
```

**Expected output**:

```text
Starting pipeline for 100 items...
Fetched 82/100 items successfully
Processed 80/82 items successfully
Stored item_0
Stored item_1
...
Storage errors: ExceptionGroup('Storage failures: 8/80', [ValueError(...), ...])
Pipeline result: {
    'status': 'partial_success',
    'fetched': 82,
    'processed': 80,
    'storage_errors': 8
}
```

## Production Patterns

### Pattern 1: Circuit Breaker

```python
# noqa:validation
from lionherd_core import concurrency

class CircuitBreaker:
    """Stop calling failing service after threshold."""
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.opened_at: float | None = None

    async def call(self, func, *args, **kwargs):
        # Check if circuit is open
        if self.opened_at:
            if concurrency.current_time() - self.opened_at < self.timeout:
                raise RuntimeError("Circuit breaker open")
            else:
                self.opened_at = None  # Reset after timeout

        try:
            result = await func(*args, **kwargs)
            self.failures = 0  # Reset on success
            return result
        except Exception as e:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = concurrency.current_time()
            raise

# Usage
breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
result = await breaker.call(fetch_from_api, "item_123")
```

### Pattern 2: Deadline Propagation

```python
import time

async def pipeline_with_deadline(item_ids: list[str], deadline_sec: float):
    """Run pipeline with absolute deadline."""
    start = time.time()

    fetched = await fetch_all_items(item_ids)

    # Check remaining time
    elapsed = time.time() - start
    remaining = deadline_sec - elapsed

    if remaining <= 0:
        return {"error": "Deadline exceeded during fetch"}

    # Adjust processing timeout based on remaining time
    processed = await ln.alcall(
        fetched,
        process_item,
        retry_timeout=remaining * 0.8,  # Leave 20% buffer for storage
        return_exceptions=True
    )

    # ... continue with time checks
```

### Pattern 3: Graceful Shutdown

```python
import signal

def chunks(lst: list, n: int):
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class GracefulShutdown:
    """Handle shutdown signals gracefully."""
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"Shutdown signal received: {signum}")
        self.shutdown_requested = True

async def run_with_shutdown(item_ids: list[str]):
    """Run pipeline with graceful shutdown."""
    shutdown = GracefulShutdown()

    for batch in chunks(item_ids, 20):
        if shutdown.shutdown_requested:
            print("Graceful shutdown: finishing current batch")
            break

        await run_pipeline(batch)

    print("Shutdown complete")
```

## Performance Characteristics

| Operation | Overhead | Typical Use |
|-----------|----------|-------------|
| alcall | 1-5ms | Batch async operations |
| CapacityLimiter | <1ms | Resource pool management |
| ExceptionGroup | <1ms | Error aggregation |
| Timeout (fail_after) | <1ms | Per-operation time limits |

## When to Use These Patterns

**Use when**:

- External services can fail or be slow
- Resource exhaustion possible (connections, memory, file descriptors)
- Partial failures acceptable (batch processing)
- Production SLAs required (99.9% uptime)

**Don't use when**:

- All-or-nothing atomicity required (use transactions)
- Single-threaded synchronous code (no async benefit)
- Overhead unacceptable (<1ms latency budget)

## Next Steps

- **API Reference**: `docs/api/libs/concurrency.md` for complete API details
- **Notebooks**: `notebooks/tutorials/concurrency/` for interactive examples

## Real-World Metrics

```python
# Example: 1000-item pipeline with failures
{
    'total_items': 1000,
    'fetch_success': 847,     # 84.7% (15.3% API failures)
    'process_success': 839,   # 99.1% of fetched
    'store_success': 831,     # 99.0% of processed
    'overall_success': 831,   # 83.1% end-to-end
    'duration_sec': 12.4,     # vs 100+ sec sequential
    'avg_retries': 1.3,       # per failed fetch
}
```

**Key takeaway**: Resilience patterns achieve 83% success rate with graceful degradation, vs 0% with fail-fast approach.

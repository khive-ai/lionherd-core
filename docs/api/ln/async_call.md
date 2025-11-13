# Async Call Utilities

> Asynchronous list processing with retry, concurrency control, and batch operations

## Overview

The `async_call` module provides high-performance utilities for applying functions to list elements asynchronously with comprehensive control over concurrency, retries, throttling, and batch processing.

**Key Capabilities:**

- **Async List Processing**: Apply sync or async functions to list items concurrently
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Concurrency Control**: Semaphore-based concurrency limits and throttling
- **Timeout Management**: Per-call timeout with cancellation support
- **Input/Output Processing**: Flexible flattening, dropna, and deduplication
- **Batch Operations**: Stream-based batch processing via async generator
- **Order Preservation**: Results maintain input order regardless of completion sequence
- **Parameter Objects**: Reusable dataclass-based parameter configurations

**When to Use:**

- Processing large lists with async API calls (LLM batches, HTTP requests)
- Parallel execution with rate limiting and retry requirements
- Batch processing workflows where results stream incrementally
- Workflows requiring fine-grained concurrency and timeout control
- Operations where input/output preprocessing is needed (flatten, dropna, unique)

**When NOT to Use:**

- Single-item operations (use direct `await func(item)`)
- Simple list comprehensions without concurrency needs
- Operations requiring complex inter-item dependencies (use task graphs)
- Real-time streaming where order doesn't matter (use lionherd's `gather` directly from `lionherd_core.libs.concurrency`)

## Functions

### `alcall()`

Apply function to each list element asynchronously with retry and concurrency control.

**Signature:**

```python
async def alcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    *,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    delay_before_start: float = 0,
    retry_initial_delay: float = 0,
    retry_backoff: float = 1,
    retry_default: Any = Unset,
    retry_timeout: float | None = None,
    retry_attempts: int = 0,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[T | BaseException]:
    ...
```

**Parameters:**

**input_** : list of Any (positional-only)

List of items to process. Accepts any iterable; automatically converted to list. Pydantic models are wrapped in single-item lists.

- Type coercion: Tuples/iterables converted to list, single items wrapped in `[item]`
- Preprocessing: Apply `input_flatten`, `input_dropna`, `input_unique` before processing
- Required: Yes

**func** : Callable returning T (positional-only)

Callable to apply to each item (sync or async). Automatically detected and wrapped.

- Type detection: Uses `inspect.iscoroutinefunction()` to identify async functions
- Sync functions: Executed via `run_sync()` for async compatibility
- Validation: Must be callable or iterable containing exactly one callable
- Required: Yes

**Input Processing:**

**input_flatten** : bool, default False

Flatten nested input structures before processing.

**input_dropna** : bool, default False

Remove `None` and `Unset` values from input.

**input_unique** : bool, default False

Remove duplicate inputs (requires `input_flatten=True`).

**input_flatten_tuple_set** : bool, default False

Include tuples and sets in flattening operation.

**Output Processing:**

**output_flatten** : bool, default False

Flatten nested output structures after processing.

**output_dropna** : bool, default False

Remove `None` and `Unset` values from output.

**output_unique** : bool, default False

Remove duplicate outputs (requires `output_flatten=True`).

**output_flatten_tuple_set** : bool, default False

Include tuples and sets in output flattening.

**Retry and Timeout:**

**delay_before_start** : float, default 0

Initial delay before processing starts (seconds). Useful for rate limit alignment.

**retry_initial_delay** : float, default 0

Initial retry delay (seconds). First retry waits this duration.

**retry_backoff** : float, default 1

Backoff multiplier for retry delays. Each subsequent retry waits `delay * backoff`.

- Example: `initial_delay=1, backoff=2` → delays: 1s, 2s, 4s, 8s

**retry_default** : Any, default Unset

Default value returned on retry exhaustion. If `Unset`, exception is raised.

- `Unset`: Raise exception after max retries (default)
- Any value: Return this value instead of raising

**retry_timeout** : float or None, default None

Timeout per function call (seconds). Raises `TimeoutError` if exceeded.

- Uses `move_on_after()` for cancellation-safe timeouts
- `None`: No timeout

**retry_attempts** : int, default 0

Maximum retry attempts (0 = no retry). Total attempts = `retry_attempts + 1`.

- `0`: Execute once, no retries
- `N`: Execute up to N+1 times total

**Concurrency and Throttling:**

**max_concurrent** : int or None, default None

Max concurrent executions. Uses semaphore for concurrency control.

- `None`: Unlimited concurrency (all tasks start immediately)
- `N`: At most N tasks execute concurrently

**throttle_period** : float or None, default None

Delay between starting tasks (seconds). Prevents request bursts.

- `None`: No throttling
- `N`: Wait N seconds between starting each task

**return_exceptions** : bool, default False

Return exceptions in results instead of raising.

- `False`: Raise `ExceptionGroup` on task failures (default)
- `True`: Return exceptions in result list at corresponding indices

**\*\*kwargs** : Any

Additional arguments passed to `func(item, **kwargs)` for each call.

**Returns:**

- **list[T | BaseException]**: Results list preserving input order. May include exceptions if `return_exceptions=True`.
  - Order: Results at index `i` correspond to `input_[i]`
  - Exceptions: Included if `return_exceptions=True`, otherwise raised as `ExceptionGroup`

**Raises:**

### ValueError

If `func` is not callable or is iterable containing multiple/non-callable items.

### TimeoutError

If `retry_timeout` exceeded during function call.

### ExceptionGroup

If `return_exceptions=False` and one or more tasks raise exceptions. Contains all task exceptions with preserved tracebacks.

**Examples:**

```python
# noqa:validation
import httpx
from lionherd_core.ln import alcall

# Basic usage - parallel API calls
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

urls = ["https://api.example.com/1", "https://api.example.com/2"]
results = await alcall(urls, fetch_data)
# [{'data': ...}, {'data': ...}]

# Retry with exponential backoff
results = await alcall(
    urls,
    fetch_data,
    retry_attempts=3,
    retry_initial_delay=1,
    retry_backoff=2,  # Delays: 1s, 2s, 4s
    retry_timeout=10,  # 10s timeout per attempt
)

# Concurrency control + throttling
results = await alcall(
    urls,
    fetch_data,
    max_concurrent=5,  # Max 5 concurrent requests
    throttle_period=0.1,  # 100ms between starting requests
)

# Input preprocessing
nested_data = [[1, 2], [3, [4, 5]], None, [6]]
results = await alcall(
    nested_data,
    lambda x: x * 2,
    input_flatten=True,  # [1, 2, 3, 4, 5, None, 6]
    input_dropna=True,   # [1, 2, 3, 4, 5, 6]
    input_unique=True,   # Remove duplicates
)
# [2, 4, 6, 8, 10, 12]

# Output preprocessing
async def fetch_items(id: str) -> list[dict]:
    return [{'id': id, 'item': 1}, {'id': id, 'item': 2}]

results = await alcall(
    ["a", "b"],
    fetch_items,
    output_flatten=True,  # Flatten nested lists
    output_unique=True,   # Remove duplicate dicts
)
# [{'id': 'a', 'item': 1}, {'id': 'a', 'item': 2}, ...]

# Default values on failure
results = await alcall(
    [1, 2, "invalid", 4],
    lambda x: x * 2,
    retry_default=None,  # Return None on failure
)
# [2, 4, None, 8]

# Return exceptions instead of raising
results = await alcall(
    [1, 2, "invalid", 4],
    lambda x: x * 2,
    return_exceptions=True,
)
# [2, 4, TypeError(...), 8]

# Sync function (auto-wrapped)
def sync_process(item: int) -> int:
    return item ** 2

results = await alcall([1, 2, 3], sync_process)
# [1, 4, 9]

# Pass kwargs to function
results = await alcall(
    [1, 2, 3],
    lambda x, multiplier: x * multiplier,
    multiplier=10,  # Passed as kwarg
)
# [10, 20, 30]
```

**Implementation Notes:**

- **Order Preservation**: Preallocates result list and fills by index (no sorting overhead)
- **Semaphore**: Uses `Semaphore` for concurrency control (non-blocking async)
- **Retry Logic**: Cancellation-aware retry loop (respects `CancelledError`)
- **Task Group**: Uses `create_task_group()` for structured concurrency
- **Exception Handling**: Non-cancel subgroup extraction for clean error propagation
- **Lazy Pydantic Import**: Thread-safe lazy import to avoid circular dependencies

**See Also:**

- `bcall()`: Batch processing wrapper using alcall
- `AlcallParams`: Reusable parameter configuration object
- `to_list()`: Input/output preprocessing utility

---

### `bcall()`

Process input in batches using `alcall()`. Yields results batch by batch.

**Signature:**

```python
async def bcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    batch_size: int,
    **kwargs: Any,
) -> AsyncGenerator[list[T | BaseException], None]:
    ...
```

**Parameters:**

**input_** : list of Any (positional-only)

Items to process in batches. Automatically flattened and cleaned (dropna).

- Preprocessing: Always applies `flatten=True, dropna=True`
- Type coercion: Same as `alcall()`

**func** : Callable returning T (positional-only)

Callable to apply to each item. Same semantics as `alcall()`.

**batch_size** : int (positional-only)

Number of items per batch. Determines yield frequency.

- Must be positive integer
- Last batch may be smaller if `len(input_) % batch_size != 0`

**\*\*kwargs** : Any

Arguments passed to `alcall()` for each batch. See `alcall()` parameters.

**Yields:**

- **list[T | BaseException]**: Results for each batch (batch_size items, except possibly last batch).

**Examples:**

```python
# noqa:validation
from lionherd_core.ln import bcall
from lionherd_core.libs.concurrency import sleep

# Batch process with streaming results
async def process_item(item: int) -> int:
    await sleep(0.1)
    return item * 2

items = list(range(20))

async for batch_results in bcall(items, process_item, batch_size=5):
    print(f"Batch completed: {batch_results}")
    # Batch completed: [0, 2, 4, 6, 8]
    # Batch completed: [10, 12, 14, 16, 18]
    # ... (4 batches total)

# Batch with retry and concurrency
async for batch_results in bcall(
    urls,
    fetch_data,
    batch_size=10,
    max_concurrent=3,
    retry_attempts=2,
):
    # Process results as they arrive
    save_to_db(batch_results)

# Collect all batches
all_results = [
    result
    async for batch in bcall(items, process_item, batch_size=5)
    for result in batch
]
# Flattens batches back to single list
```

**Use Cases:**

- **Memory Efficiency**: Process large datasets without loading all results at once
- **Progressive Updates**: Update UI or database as batches complete
- **Failure Isolation**: Batch failures don't block remaining batches
- **Rate Limit Management**: Batch size controls burst rate

**See Also:**

- `alcall()`: Underlying function for batch processing
- `BcallParams`: Reusable parameter configuration for bcall

---

## Parameter Classes

### `AlcallParams`

Dataclass for reusable `alcall()` parameter configurations.

**Signature:**

```python
from lionherd_core.ln import AlcallParams

@dataclass(slots=True, init=False, frozen=True)
class AlcallParams(Params):
    # Input processing
    input_flatten: bool
    input_dropna: bool
    input_unique: bool
    input_flatten_tuple_set: bool

    # Output processing
    output_flatten: bool
    output_dropna: bool
    output_unique: bool
    output_flatten_tuple_set: bool

    # Retry and timeout
    delay_before_start: float
    retry_initial_delay: float
    retry_backoff: float
    retry_default: Any
    retry_timeout: float
    retry_attempts: int

    # Concurrency and throttling
    max_concurrent: int
    throttle_period: float

    # Additional kwargs
    kw: dict[str, Any] = Unset

    async def __call__(
        self,
        input_: list[Any],
        func: Callable[..., T],
        **kw: Any,
    ) -> list[T]:
        ...
```

**Attributes:**

All fields correspond to `alcall()` parameters. See `alcall()` documentation for detailed parameter descriptions.

**Methods:**

#### `__call__()`

Execute `alcall()` with configured parameters.

**Parameters:**

- `input_` (list[Any]): Items to process
- `func` (Callable): Function to apply
- `**kw` (Any): Override parameters (merged with configured parameters)

**Returns:**

- list[T]: Results from `alcall()`

**Examples:**

```python
from lionherd_core.ln import AlcallParams
from lionherd_core.types import Unset

# Create reusable configuration
api_config = AlcallParams(
    max_concurrent=5,
    retry_attempts=3,
    retry_initial_delay=1,
    retry_backoff=2,
    retry_timeout=10,
    throttle_period=0.2,
    return_exceptions=True,
    # Unset for unused parameters
    input_flatten=Unset,
    input_dropna=Unset,
    # ... other fields
)

# Use configuration multiple times
results1 = await api_config(urls1, fetch_data)
results2 = await api_config(urls2, fetch_data)

# Override parameters per call
results3 = await api_config(
    urls3,
    fetch_data,
    max_concurrent=10,  # Override default
)

# Access default kwargs
default_kw = api_config.default_kw()
# {'max_concurrent': 5, 'retry_attempts': 3, ...}
```

**Design Notes:**

- **Frozen**: Immutable after creation (prevents accidental modification)
- **Slots**: Memory-efficient attribute storage
- **None Sentinel**: Uses `none_as_sentinel=True` for proper `None` vs `Unset` handling
- **Callable**: Implements `__call__()` for function-like usage

---

### `BcallParams`

Dataclass for reusable `bcall()` parameter configurations. Extends `AlcallParams` with `batch_size`.

**Signature:**

```python
from lionherd_core.ln import BcallParams

@dataclass(slots=True, init=False, frozen=True)
class BcallParams(AlcallParams):
    batch_size: int

    async def __call__(
        self,
        input_: list[Any],
        func: Callable[..., T],
        **kw: Any,
    ) -> AsyncGenerator[list[T], None]:
        ...
```

**Additional Attributes:**

**batch_size** : int

Number of items per batch. Must be positive integer.

**Methods:**

#### `__call__()`

Execute `bcall()` with configured parameters.

**Parameters:**

- `input_` (list[Any]): Items to process
- `func` (Callable): Function to apply
- `**kw` (Any): Override parameters

**Returns:**

- AsyncGenerator[list[T], None]: Batch results generator

**Examples:**

```python
from lionherd_core.ln import BcallParams

# Create batch configuration
batch_config = BcallParams(
    batch_size=10,
    max_concurrent=3,
    retry_attempts=2,
    # ... other AlcallParams fields
)

# Stream batch results
async for batch in batch_config(urls, fetch_data):
    process_batch(batch)

# Override batch size per call
async for batch in batch_config(urls, fetch_data, batch_size=20):
    process_batch(batch)
```

**See Also:**

- `AlcallParams`: Parent class with shared parameters
- `bcall()`: Underlying batch processing function

---

## Usage Patterns

### Basic Parallel Processing

```python
# noqa:validation
from lionherd_core.ln import alcall
from lionherd_core.libs.concurrency import sleep

# Process list items in parallel
items = [1, 2, 3, 4, 5]
results = await alcall(items, lambda x: x ** 2)
# [1, 4, 9, 16, 25]

# Async function
async def async_process(item: int) -> int:
    await sleep(0.1)
    return item * 2

results = await alcall(items, async_process)
# [2, 4, 6, 8, 10]
```

### Retry with Exponential Backoff

```python
# noqa:validation
import httpx
from lionherd_core.ln import alcall

async def flaky_api_call(id: str) -> dict:
    """API that may fail transiently."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/api/{id}")
        return response.json()

# Retry up to 3 times with exponential backoff
item_ids = ["item1", "item2", "item3"]
results = await alcall(
    item_ids,
    flaky_api_call,
    retry_attempts=3,
    retry_initial_delay=1,
    retry_backoff=2,  # Waits: 1s, 2s, 4s
    retry_timeout=5,  # 5s timeout per attempt
)
```

### Concurrency and Rate Limiting

```python
from lionherd_core.ln import alcall

# Limit concurrent requests and throttle start rate
results = await alcall(
    urls,
    fetch_data,
    max_concurrent=10,  # At most 10 concurrent requests
    throttle_period=0.1,  # 100ms between starting each request
)

# Useful for respecting API rate limits:
# - max_concurrent: Total concurrent connections
# - throttle_period: Requests per second limit
```

### Input Preprocessing

```python
from lionherd_core.ln import alcall

# Flatten nested structures
nested = [[1, 2], [3, [4, 5]], [6]]
results = await alcall(
    nested,
    lambda x: x * 2,
    input_flatten=True,
)
# [2, 4, 6, 8, 10, 12]

# Remove None values
data_with_none = [1, 2, None, 4, None, 6]
results = await alcall(
    data_with_none,
    lambda x: x * 2,
    input_dropna=True,
)
# [2, 4, 8, 12]

# Deduplicate inputs
duplicates = [1, 2, 2, 3, 1, 4]
results = await alcall(
    duplicates,
    lambda x: x ** 2,
    input_flatten=True,  # Required for unique
    input_unique=True,
)
# [1, 4, 9, 16] (unique inputs only)
```

### Exception Handling

```python
from lionherd_core.ln import alcall

# Return exceptions in results
results = await alcall(
    [1, 2, "invalid", 4],
    lambda x: x * 2,
    return_exceptions=True,
)
# [2, 4, TypeError(...), 8]

# Filter exceptions
valid_results = [r for r in results if not isinstance(r, Exception)]

# Use default value on failure
results = await alcall(
    [1, 2, "invalid", 4],
    lambda x: x * 2,
    retry_default=0,  # Return 0 on failure
)
# [2, 4, 0, 8]

# Raise ExceptionGroup on failure (default)
try:
    results = await alcall(items, failing_func)
except ExceptionGroup as eg:
    for exc in eg.exceptions:
        print(f"Task failed: {exc}")
```

### Batch Processing

```python
# noqa:validation
from lionherd_core.ln import bcall

# User-defined async function (replace with actual implementation)
async def save_to_db(batch: list) -> None:
    """Save batch to database."""
    pass  # Placeholder for actual database logic

# Stream results in batches
async for batch_results in bcall(
    items,
    process_item,
    batch_size=100,
    max_concurrent=10,
):
    # Process batch immediately
    await save_to_db(batch_results)
    print(f"Saved batch of {len(batch_results)} items")

# Collect with progress tracking
total = len(items)
processed = 0

async for batch in bcall(items, process_item, batch_size=50):
    processed += len(batch)
    print(f"Progress: {processed}/{total} ({processed/total*100:.1f}%)")
```

### Reusable Configurations

```python
from lionherd_core.ln import AlcallParams, BcallParams
from lionherd_core.types import Unset

# Define configuration once
api_params = AlcallParams(
    max_concurrent=5,
    retry_attempts=3,
    retry_initial_delay=1,
    retry_backoff=2,
    throttle_period=0.2,
    return_exceptions=True,
    input_flatten=Unset,  # Unset unused params
    # ... set all required fields
)

# Reuse across multiple calls
results1 = await api_params(urls1, fetch_data)
results2 = await api_params(urls2, fetch_data)
results3 = await api_params(urls3, fetch_data, max_concurrent=10)  # Override

# Batch configuration
batch_params = BcallParams(
    batch_size=50,
    max_concurrent=5,
    retry_attempts=2,
    # ... other params
)

async for batch in batch_params(items, process_item):
    handle_batch(batch)
```

### Common Pitfalls

#### Pitfall 1: Forgetting input_flatten for input_unique

**Issue**: `input_unique=True` without `input_flatten=True` has no effect.

```python
# Wrong - unique has no effect
results = await alcall(
    [[1, 2], [1, 2]],
    lambda x: x,
    input_unique=True,  # Ignored without flatten
)
# [[1, 2], [1, 2]]

# Correct - flatten first
results = await alcall(
    [[1, 2], [1, 2]],
    lambda x: x,
    input_flatten=True,
    input_unique=True,
)
# [1, 2]
```

**Solution**: Always use `input_flatten=True` with `input_unique=True`.

#### Pitfall 2: Mixing retry_default with return_exceptions

**Issue**: Both handle failures differently, causing confusion.

```python
# Unclear behavior - which failure handling applies?
results = await alcall(
    items,
    failing_func,
    retry_default=None,
    return_exceptions=True,
)
```

**Solution**: Choose one failure strategy:

- Use `retry_default` for value substitution
- Use `return_exceptions` for explicit exception handling

#### Pitfall 3: Over-throttling with Large Batches

**Issue**: `throttle_period` applies between ALL task starts, not batches.

```python
# This takes 100 seconds to START all tasks!
results = await alcall(
    list(range(1000)),
    process_item,
    throttle_period=0.1,  # 100ms between each of 1000 tasks
)
```

**Solution**: Use `bcall()` for large datasets to throttle batch-level, not item-level.

#### Pitfall 4: Timeout vs Retry Confusion

**Issue**: `retry_timeout` is per-attempt, not total timeout.

```python
# Total timeout = retry_timeout * (retry_attempts + 1)
# 10s * 4 attempts = up to 40 seconds!
results = await alcall(
    items,
    slow_func,
    retry_timeout=10,
    retry_attempts=3,
)
```

**Solution**: Account for total timeout = `retry_timeout * (retry_attempts + 1)`.

---

## Design Rationale

### Why Order Preservation?

Results maintain input order regardless of completion sequence because:

1. **Predictability**: Index mapping simplifies result lookup (`results[i]` ↔ `input_[i]`)
2. **No Sorting Overhead**: Preallocated list with index-based filling avoids post-processing
3. **Debugging**: Easier to correlate inputs with outputs during troubleshooting
4. **API Compatibility**: Matches common async patterns (lionherd's `gather()` preserves order like `asyncio.gather()`)

### Why Separate alcall and bcall?

`alcall()` processes entire list, `bcall()` streams batches because:

1. **Memory Control**: Large datasets need incremental result processing
2. **Progress Visibility**: Streaming enables real-time progress tracking
3. **Failure Isolation**: Batch-level retry doesn't reprocess successful batches
4. **Different Use Cases**: `alcall()` for bounded datasets, `bcall()` for large/unbounded

### Why Parameter Dataclasses?

`AlcallParams` and `BcallParams` provide reusable configurations because:

1. **DRY Principle**: Define once, use many times
2. **Type Safety**: Dataclass validation ensures parameter correctness
3. **Immutability**: Frozen dataclasses prevent accidental modification
4. **Override Flexibility**: `__call__(**kw)` allows per-call overrides
5. **Documentation**: Explicit fields improve IDE autocomplete and type hints

### Why Support Both Sync and Async Functions?

Automatic sync/async detection via `is_coro_func()` enables:

1. **Flexibility**: Use sync functions without manual wrapping
2. **Migration Path**: Incrementally convert sync→async without API changes
3. **Library Integration**: Many libraries provide sync-only APIs
4. **Performance**: Sync functions run via `run_sync()` without blocking event loop

---

## See Also

- **Related Functions**:
  - `to_list()`: Input/output preprocessing utility
  - `gather()` from lionherd: Enhanced gathering with retry/concurrency control (use this instead of `asyncio.gather()`)
  - `create_task_group()`: Structured concurrency primitive
- **Related Classes**:
  - `Params`: Base class for parameter dataclasses
  - `Semaphore`: Concurrency control primitive
<!-- TODO: Add links to user guides when available (issue #115) -->

---

## Examples

```python
# Standard imports for ln.async_call examples
from lionherd_core.ln import alcall, bcall, AlcallParams, BcallParams
from lionherd_core.types import Unset
```

### Example 1: LLM Batch Processing

```python
# noqa:validation
from openai import AsyncOpenAI
from lionherd_core.ln import bcall

client = AsyncOpenAI()

# User-defined async function (replace with actual database logic)
async def save_completions(batch: list[str]) -> None:
    """Save completions to database."""
    pass  # Placeholder for actual database logic

async def generate_completion(prompt: str) -> str:
    """Generate completion with retry."""
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Process 1000 prompts in batches with rate limiting
prompts = [f"Prompt {i}" for i in range(1000)]

async for batch_results in bcall(
    prompts,
    generate_completion,
    batch_size=50,
    max_concurrent=10,  # OpenAI rate limit
    throttle_period=0.1,  # 10 req/sec
    retry_attempts=3,
    retry_initial_delay=1,
    retry_backoff=2,
):
    # Save batch immediately
    await save_completions(batch_results)
    print(f"Saved {len(batch_results)} completions")
```

### Example 2: API Scraping with Error Recovery

```python
# noqa:validation
import httpx
from lionherd_core.ln import alcall

# User-defined function (replace with actual HTML parser like BeautifulSoup)
def parse_html(html: str) -> dict:
    """Parse HTML and extract data."""
    return {"title": "Example", "content": html[:100]}

# User-defined function (replace with actual file reading logic)
def load_urls_from_file(filename: str) -> list[str]:
    """Load URLs from file."""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]

async def scrape_page(url: str) -> dict:
    """Scrape page with timeout."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return parse_html(response.text)

urls = load_urls_from_file("urls.txt")

# Scrape with error recovery
results = await alcall(
    urls,
    scrape_page,
    max_concurrent=20,
    retry_attempts=5,
    retry_timeout=30,  # 30s per attempt
    retry_default={"error": "failed"},  # Return placeholder on failure
    throttle_period=0.05,  # Respect robots.txt
)

# Filter successful results
successful = [r for r in results if "error" not in r]
failed = [r for r in results if "error" in r]

print(f"Scraped: {len(successful)}, Failed: {len(failed)}")
```

### Example 3: Database Bulk Insert with Preprocessing

```python
# noqa:validation
from lionherd_core.ln import alcall

# User-defined async function (replace with actual database client like asyncpg)
async def db_insert(table: str, data: dict) -> str:
    """Insert record into database."""
    # Placeholder: would use asyncpg, motor, etc.
    return f"id_{hash(str(data))}"

async def insert_record(data: dict) -> str:
    """Insert record and return ID."""
    return await db_insert("users", data)

# Nested data with duplicates and None values
raw_data = [
    [{"name": "Alice"}, {"name": "Bob"}],
    None,
    [{"name": "Alice"}],  # Duplicate
    {"name": "Charlie"},
]

# Clean and insert
inserted_ids = await alcall(
    raw_data,
    insert_record,
    input_flatten=True,  # Flatten nested lists
    input_dropna=True,   # Remove None
    input_unique=True,   # Deduplicate
    max_concurrent=50,   # Batch inserts
)
# ['id1', 'id2', 'id3'] (Alice, Bob, Charlie)
```

### Example 4: Parallel File Processing

```python
# noqa:validation
import glob
import aiofiles
from lionherd_core.ln import alcall

# User-defined function (replace with actual text analysis logic)
def analyze_text(content: str) -> dict:
    """Analyze text content."""
    return {
        "word_count": len(content.split()),
        "char_count": len(content),
        "lines": len(content.split('\n'))
    }

async def process_file(filepath: str) -> dict:
    """Read and process file."""
    async with aiofiles.open(filepath, 'r') as f:
        content = await f.read()
    return analyze_text(content)

# Process all .txt files in directory
files = glob.glob("data/**/*.txt", recursive=True)

results = await alcall(
    files,
    process_file,
    max_concurrent=100,  # I/O bound, high concurrency OK
    retry_attempts=2,
    return_exceptions=True,  # Don't fail entire batch on single file error
)

# Separate successful and failed
successful = [r for r in results if not isinstance(r, Exception)]
failed_files = [
    files[i]
    for i, r in enumerate(results)
    if isinstance(r, Exception)
]
```

### Example 5: Reusable Configuration for Different Environments

```python
from lionherd_core.ln import AlcallParams
from lionherd_core.types import Unset

# Production config - strict retry and rate limits
prod_config = AlcallParams(
    max_concurrent=5,
    retry_attempts=5,
    retry_initial_delay=2,
    retry_backoff=2,
    retry_timeout=30,
    throttle_period=0.2,
    return_exceptions=False,  # Fail fast
    input_flatten=Unset,
    # ... set all fields
)

# Development config - fast iteration
dev_config = AlcallParams(
    max_concurrent=50,
    retry_attempts=1,
    retry_timeout=10,
    throttle_period=0,
    return_exceptions=True,  # Continue on errors
    # ... set all fields
)

# Select config based on environment
import os
config = prod_config if os.getenv("ENV") == "prod" else dev_config

# Use same code with different configs
results = await config(items, process_item)
```

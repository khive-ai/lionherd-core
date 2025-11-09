# ln Module

> Core utilities for async operations, data processing, fuzzy matching, and JSON serialization

## Overview

The `ln` module provides essential utility functions for the lionherd-core ecosystem, focusing on high-performance data processing, flexible type conversions, and robust async operations. These utilities power the framework's data handling, serialization, and execution patterns.

**Key Capabilities:**

- **Async Execution**: Parallel function application with retry, throttling, and concurrency control
- **Fuzzy Matching**: String similarity-based key matching and validation for robust data handling
- **JSON Serialization**: High-performance orjson-based serialization with custom type support
- **List Processing**: Flexible list transformations with flattening, deduplication, and filtering
- **Type Conversion**: Universal converters for dict/list with support for nested structures
- **Utility Functions**: Path creation, module importing, datetime helpers, and binning

**When to Use This Module:**

- Processing LLM outputs with inconsistent key names (fuzzy matching)
- Parallel execution of I/O operations with retry logic (alcall/bcall)
- High-performance JSON serialization with custom types (json_dumps)
- Converting arbitrary data structures to dicts/lists (to_dict/to_list)
- Generating stable hashes for complex data structures (hash_dict)

## Module Exports

```python
from lionherd_core.ln import (
    # Async call utilities
    alcall,
    bcall,
    AlcallParams,
    BcallParams,

    # Fuzzy matching and validation
    fuzzy_match_keys,
    fuzzy_validate_mapping,
    fuzzy_validate_pydantic,
    FuzzyMatchKeysParams,

    # Hashing
    hash_dict,

    # JSON utilities
    get_orjson_default,
    json_dumpb,
    json_dumps,
    json_lines_iter,
    json_dict,
    make_options,

    # List processing
    lcall,
    to_list,

    # Dictionary conversion
    to_dict,

    # General utilities
    acreate_path,
    get_bins,
    import_module,
    is_import_installed,
    now_utc,
)
```

## Async Call Utilities

### `alcall()`

Apply function to each list element asynchronously with retry and concurrency control.

**Signature:**

```python
async def alcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    *,
    # Input processing
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_flatten_tuple_set: bool = False,
    # Output processing
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    # Retry and timeout
    delay_before_start: float = 0,
    retry_initial_delay: float = 0,
    retry_backoff: float = 1,
    retry_default: Any = Unset,
    retry_timeout: float | None = None,
    retry_attempts: int = 0,
    # Concurrency and throttling
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[T | BaseException]: ...
```

**Parameters:**

- `input_` (list[Any]): Items to process (auto-converted from iterables)
- `func` (Callable[..., T]): Function to apply (sync or async)
- `input_flatten` (bool, default False): Flatten nested input structures before processing
- `input_dropna` (bool, default False): Remove None/Unset from input
- `input_unique` (bool, default False): Remove duplicate inputs (requires flatten)
- `input_flatten_tuple_set` (bool, default False): Include tuples/sets in input flattening
- `output_flatten` (bool, default False): Flatten nested output structures
- `output_dropna` (bool, default False): Remove None/Unset from output
- `output_unique` (bool, default False): Remove duplicate outputs (requires flatten)
- `output_flatten_tuple_set` (bool, default False): Include tuples/sets in output flattening
- `delay_before_start` (float, default 0): Initial delay before processing (seconds)
- `retry_initial_delay` (float, default 0): Initial retry delay (seconds)
- `retry_backoff` (float, default 1): Backoff multiplier for retry delays
- `retry_default` (Any, default Unset): Default value on retry exhaustion (Unset = raise)
- `retry_timeout` (float | None, default None): Timeout per function call (seconds)
- `retry_attempts` (int, default 0): Maximum retry attempts (0 = no retry)
- `max_concurrent` (int | None, default None): Max concurrent executions (None = unlimited)
- `throttle_period` (float | None, default None): Delay between starting tasks (seconds)
- `return_exceptions` (bool, default False): Return exceptions instead of raising
- `**kwargs` (Any): Additional arguments passed to func

**Returns:**

- list[T | BaseException]: Results in input order (may include exceptions if return_exceptions=True)

**Raises:**

- ValueError: If func is not callable
- TimeoutError: If retry_timeout exceeded
- ExceptionGroup: If return_exceptions=False and tasks raise

**Examples:**

```python
from lionherd_core.ln import alcall

# Basic parallel execution
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

urls = ["https://api.example.com/1", "https://api.example.com/2"]
results = await alcall(urls, fetch_data)

# With retry and concurrency limits
results = await alcall(
    urls,
    fetch_data,
    retry_attempts=3,
    retry_initial_delay=1.0,
    retry_backoff=2.0,
    max_concurrent=5,
    retry_timeout=10.0,
)

# With input/output processing
nested_data = [[1, 2], [3, 4], [5, 6]]
results = await alcall(
    nested_data,
    lambda x: x * 2,
    input_flatten=True,  # Flatten to [1, 2, 3, 4, 5, 6]
    output_unique=True,  # Remove duplicates from output
)

# With throttling for rate-limited APIs
results = await alcall(
    urls,
    fetch_data,
    throttle_period=0.5,  # 500ms delay between starts
    max_concurrent=3,     # Max 3 concurrent requests
)

# Error handling with default values
results = await alcall(
    urls,
    fetch_data,
    retry_attempts=2,
    retry_default={"error": "failed"},  # Return default on failure
)
```

**Notes:**

- Automatically handles both sync and async functions
- Preserves input order in output (no sorting needed)
- Uses task groups for structured concurrency
- Supports exponential backoff for retries
- Throttling applies between task starts, not completions

### `bcall()`

Process input in batches using alcall. Yields results batch by batch.

**Signature:**

```python
async def bcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    batch_size: int,
    **kwargs: Any,
) -> AsyncGenerator[list[T | BaseException], None]: ...
```

**Parameters:**

- `input_` (list[Any]): Items to process
- `func` (Callable[..., T]): Function to apply
- `batch_size` (int): Number of items per batch
- `**kwargs` (Any): Arguments passed to alcall (see alcall for details)

**Yields:**

- list[T | BaseException]: Results for each batch

**Examples:**

```python
from lionherd_core.ln import bcall

# Process large dataset in batches
async def process_item(item: dict) -> dict:
    # Some processing logic
    return {"id": item["id"], "processed": True}

large_dataset = [{"id": i} for i in range(10000)]

async for batch_results in bcall(
    large_dataset,
    process_item,
    batch_size=100,
    max_concurrent=10,
):
    # Process each batch of 100 results as they complete
    print(f"Processed {len(batch_results)} items")
    await save_to_database(batch_results)
```

**Notes:**

- Yields batches as they complete (streaming behavior)
- Automatically flattens and drops None from input
- Useful for memory-efficient processing of large datasets

### `AlcallParams`

Parameter dataclass for alcall with callable interface.

**Signature:**

```python
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

    kw: dict[str, Any] = Unset

    async def __call__(
        self,
        input_: list[Any],
        func: Callable[..., T],
        **kw: Any,
    ) -> list[T]: ...
```

**Examples:**

```python
from lionherd_core.ln import AlcallParams

# Create reusable parameter set
api_params = AlcallParams(
    retry_attempts=3,
    retry_backoff=2.0,
    max_concurrent=5,
    retry_timeout=10.0,
)

# Use multiple times
results1 = await api_params(urls1, fetch_data)
results2 = await api_params(urls2, fetch_data, retry_attempts=5)  # Override
```

### `BcallParams`

Parameter dataclass for bcall (extends AlcallParams).

**Signature:**

```python
@dataclass(slots=True, init=False, frozen=True)
class BcallParams(AlcallParams):
    batch_size: int

    async def __call__(
        self,
        input_: list[Any],
        func: Callable[..., T],
        **kw: Any,
    ) -> list[T]: ...
```

**Examples:**

```python
from lionherd_core.ln import BcallParams

# Create reusable batch processing config
batch_params = BcallParams(
    batch_size=100,
    max_concurrent=10,
    retry_attempts=2,
)

async for batch in batch_params(large_dataset, process_item):
    print(f"Batch completed: {len(batch)} items")
```

## Fuzzy Matching and Validation

### `fuzzy_match_keys()`

Validate and correct dict keys using fuzzy string matching.

**Signature:**

```python
def fuzzy_match_keys(
    d_: dict[str, Any],
    keys: KeysLike,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | SimilarityAlgo | SimilarityFunc = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = Unset,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]: ...
```

**Parameters:**

- `d_` (dict[str, Any]): Input dictionary to validate
- `keys` (KeysLike): Expected keys (list or dict-like with .keys())
- `similarity_algo` (SIMILARITY_TYPE | SimilarityAlgo | SimilarityFunc, default "jaro_winkler"): String similarity algorithm
- `similarity_threshold` (float, default 0.85): Minimum similarity score (0.0-1.0)
- `fuzzy_match` (bool, default True): Enable fuzzy matching for unmatched keys
- `handle_unmatched` ({'ignore', 'raise', 'remove', 'fill', 'force'}, default 'ignore'): How to handle unmatched keys
  - `'ignore'`: Keep unmatched keys as-is
  - `'raise'`: Raise ValueError on unmatched keys
  - `'remove'`: Remove unmatched keys
  - `'fill'`: Fill missing expected keys with fill_value
  - `'force'`: Fill missing keys, remove unmatched
- `fill_value` (Any, default Unset): Default value for missing keys when filling
- `fill_mapping` (dict[str, Any] | None, default None): Custom values for specific missing keys
- `strict` (bool, default False): Raise if expected keys are missing

**Returns:**

- dict[str, Any]: Dictionary with corrected keys

**Raises:**

- TypeError: If d_ is not a dict or keys is None
- ValueError: If similarity_threshold out of range or unmatched keys found in raise mode

**Examples:**

```python
from lionherd_core.ln import fuzzy_match_keys
from pydantic import BaseModel

class UserModel(BaseModel):
    user_name: str
    email_address: str
    age: int

# LLM output with typos/variations
llm_output = {
    "username": "alice",      # Close to user_name
    "emailAddress": "a@b.com", # Close to email_address
    "age": 30,                 # Exact match
    "extra_field": "ignore"    # Not in model
}

# Fuzzy match to Pydantic model fields
corrected = fuzzy_match_keys(
    llm_output,
    UserModel.model_fields,
    handle_unmatched="remove",  # Remove extra_field
)
# Result: {"user_name": "alice", "email_address": "a@b.com", "age": 30}

user = UserModel.model_validate(corrected)

# Strict mode - raise on missing fields
try:
    fuzzy_match_keys(
        {"username": "alice"},
        UserModel.model_fields,
        strict=True,
    )
except ValueError as e:
    print(f"Missing fields: {e}")

# Fill missing fields with defaults
corrected = fuzzy_match_keys(
    {"username": "alice"},
    UserModel.model_fields,
    handle_unmatched="fill",
    fill_value=None,
    fill_mapping={"age": 0},  # Custom default for age
)
# Result: {"user_name": "alice", "email_address": None, "age": 0}
```

**Notes:**

- Default jaro_winkler algorithm works well for typos and case variations
- Threshold 0.85 balances flexibility vs false positives
- Use `handle_unmatched="remove"` for clean Pydantic validation
- First tries exact matches, then fuzzy matches (fast path optimization)

### `fuzzy_validate_pydantic()`

Validate and parse text/dict into Pydantic model with fuzzy parsing.

**Signature:**

```python
def fuzzy_validate_pydantic(
    text,
    /,
    model_type: type[BaseModel],
    fuzzy_parse: bool = True,
    fuzzy_match: bool = False,
    fuzzy_match_params: FuzzyMatchKeysParams | dict | None = None,
) -> BaseModel: ...
```

**Parameters:**

- `text` (BaseModel | dict | str): Input data (model instance, dict, or JSON string)
- `model_type` (type[BaseModel]): Target Pydantic model class
- `fuzzy_parse` (bool, default True): Enable fuzzy JSON extraction from text (handles markdown, code blocks)
- `fuzzy_match` (bool, default False): Enable fuzzy key matching for field names
- `fuzzy_match_params` (FuzzyMatchKeysParams | dict | None, default None): Parameters for fuzzy matching

**Returns:**

- BaseModel: Validated Pydantic model instance

**Raises:**

- ValidationError: If JSON extraction or model validation fails
- TypeError: If fuzzy_match_params is invalid type

**Examples:**

````python
from lionherd_core.ln import fuzzy_validate_pydantic
from pydantic import BaseModel

class Task(BaseModel):
    task_name: str
    priority: int
    status: str

# LLM response with markdown formatting
llm_response = """
Here's the task:
```json
{
  "taskName": "Fix bug",
  "priority": 1,
  "status": "pending"
}
```

"""

# Fuzzy parse and validate

task = fuzzy_validate_pydantic(
    llm_response,
    Task,
    fuzzy_parse=True,   # Extract JSON from markdown
    fuzzy_match=True,   # Match taskName -> task_name
)
print(task)  # Task(task_name="Fix bug", priority=1, status="pending")

# Already valid instance (no-op)

existing_task = Task(task_name="test", priority=2, status="done")
result = fuzzy_validate_pydantic(existing_task, Task)
assert result is existing_task  # Same object

# Custom fuzzy matching parameters

from lionherd_core.ln import FuzzyMatchKeysParams

task = fuzzy_validate_pydantic(
    llm_response,
    Task,
    fuzzy_match=True,
    fuzzy_match_params=FuzzyMatchKeysParams(
        similarity_threshold=0.9,
        handle_unmatched="remove",
    ),
)

````

**Notes:**

- Handles common LLM output formats (markdown code blocks, plain JSON)
- Pass-through for already-valid model instances (zero overhead)
- Default fuzzy_match_params uses `handle_unmatched="remove"` when None

### `fuzzy_validate_mapping()`

Validate any input into dict with expected keys and fuzzy matching.

**Signature:**

```python
def fuzzy_validate_mapping(
    d: Any,
    keys: KeysLike,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
) -> dict[str, Any]: ...
```

**Parameters:**

- `d` (Any): Input to convert and validate (dict, JSON string, XML, object, etc.)
- `keys` (KeysLike): Expected keys (list or dict-like)
- `similarity_algo` (SIMILARITY_TYPE | Callable, default "jaro_winkler"): String similarity algorithm
- `similarity_threshold` (float, default 0.85): Minimum similarity score (0.0-1.0)
- `fuzzy_match` (bool, default True): Enable fuzzy key matching
- `handle_unmatched` ({'ignore', 'raise', 'remove', 'fill', 'force'}, default 'ignore'): How to handle unmatched keys
- `fill_value` (Any, default None): Default value for missing keys
- `fill_mapping` (dict[str, Any] | None, default None): Custom values for specific keys
- `strict` (bool, default False): Raise if expected keys are missing
- `suppress_conversion_errors` (bool, default False): Return empty dict on conversion failure

**Returns:**

- dict[str, Any]: Validated dictionary with corrected keys

**Raises:**

- TypeError: If d is None
- ValueError: If conversion fails and suppress_conversion_errors is False

**Examples:**

```python
from lionherd_core.ln import fuzzy_validate_mapping

# Expected structure
expected_keys = ["user_id", "username", "email"]

# Various input formats
json_str = '{"userId": 1, "userName": "alice", "email": "a@b.com"}'
xml_obj = SomeXMLObject()  # Has to_dict() method
pydantic_model = SomeModel(user_id=1, username="alice", email="a@b.com")

# All convert and validate consistently
result1 = fuzzy_validate_mapping(json_str, expected_keys, fuzzy_match=True)
result2 = fuzzy_validate_mapping(xml_obj, expected_keys, fuzzy_match=True)
result3 = fuzzy_validate_mapping(pydantic_model, expected_keys, fuzzy_match=True)
# All produce: {"user_id": 1, "username": "alice", "email": "a@b.com"}

# Graceful error handling
result = fuzzy_validate_mapping(
    "invalid data",
    expected_keys,
    suppress_conversion_errors=True,
)
# Returns: {}
```

**Notes:**

- Universal converter - handles JSON, XML, Pydantic, dataclasses, etc.
- First converts to dict via `to_dict()`, then applies fuzzy matching
- Use `suppress_conversion_errors=True` for optional data sources

### `FuzzyMatchKeysParams`

Parameter dataclass for fuzzy_match_keys with callable interface.

**Signature:**

```python
@dataclass(slots=True, init=False, frozen=True)
class FuzzyMatchKeysParams(Params):
    similarity_algo: SIMILARITY_TYPE | SimilarityAlgo | SimilarityFunc = "jaro_winkler"
    similarity_threshold: float = 0.85
    fuzzy_match: bool = True
    handle_unmatched: HandleUnmatched = "ignore"
    fill_value: Any = Unset
    fill_mapping: dict[str, Any] | Any = Unset
    strict: bool = False

    def __call__(self, d_: dict[str, Any], keys: KeysLike) -> dict[str, Any]: ...
```

**Examples:**

```python
from lionherd_core.ln import FuzzyMatchKeysParams

# Create reusable fuzzy matcher
strict_matcher = FuzzyMatchKeysParams(
    similarity_threshold=0.9,
    handle_unmatched="remove",
    strict=True,
)

# Use for multiple validations
corrected1 = strict_matcher(data1, expected_keys)
corrected2 = strict_matcher(data2, expected_keys)
```

## Hashing

### `hash_dict()`

Generate stable hash for any data structure including dicts, lists, and Pydantic models.

**Signature:**

```python
def hash_dict(data: Any, strict: bool = False) -> int: ...
```

**Parameters:**

- `data` (Any): Data to hash (dict, list, BaseModel, or any object)
- `strict` (bool, default False): If True, deepcopy data before hashing to prevent mutation side effects

**Returns:**

- int: Integer hash value (stable across equivalent structures)

**Raises:**

- TypeError: If generated representation is not hashable

**Examples:**

```python
from lionherd_core.ln import hash_dict

# Dict hashing (order-independent)
d1 = {"a": 1, "b": 2}
d2 = {"b": 2, "a": 1}
assert hash_dict(d1) == hash_dict(d2)  # Same hash

# Nested structures
nested = {"users": [{"id": 1}, {"id": 2}], "count": 2}
h = hash_dict(nested)

# Pydantic model hashing
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user1 = User(name="alice", age=30)
user2 = User(name="alice", age=30)
assert hash_dict(user1) == hash_dict(user2)  # Content-based

# Set hashing (order-independent)
s1 = {3, 1, 2}
s2 = {1, 2, 3}
assert hash_dict(s1) == hash_dict(s2)

# Cache key for complex objects
cache = {}
key = hash_dict({"query": "test", "params": {"limit": 10}})
cache[key] = "result"

# Strict mode (prevent mutation affecting hash)
mutable_data = {"key": [1, 2, 3]}
h1 = hash_dict(mutable_data, strict=True)
mutable_data["key"].append(4)  # Mutation doesn't affect original hash
h2 = hash_dict(mutable_data, strict=True)
assert h1 != h2  # Different hashes
```

**Notes:**

- Produces stable hashes across Python sessions
- Order-independent for dicts and sets
- Handles mixed-type sets and dicts
- Uses sorted representation for deterministic ordering
- Pydantic models hashed via model_dump()

## JSON Utilities

### `json_dumps()`

Serialize to JSON string with high performance and custom type support.

**Signature:**

```python
def json_dumps(
    obj: Any,
    /,
    *,
    decode: bool = True,
    pretty: bool = False,
    sort_keys: bool = False,
    naive_utc: bool = False,
    utc_z: bool = False,
    append_newline: bool = False,
    allow_non_str_keys: bool = False,
    deterministic_sets: bool = False,
    decimal_as_float: bool = False,
    enum_as_name: bool = False,
    passthrough_datetime: bool = False,
    safe_fallback: bool = False,
    fallback_clip: int = 2048,
    default: Callable[[Any], Any] | None = None,
    options: int | None = None,
) -> str | bytes: ...
```

**Parameters:**

- `obj` (Any): Object to serialize
- `decode` (bool, default True): Return str (True) or bytes (False)
- `pretty` (bool, default False): Indent output for human readability
- `sort_keys` (bool, default False): Sort dictionary keys alphabetically
- `naive_utc` (bool, default False): Assume naive datetimes are UTC
- `utc_z` (bool, default False): Use 'Z' suffix for UTC timestamps
- `append_newline` (bool, default False): Append newline to output
- `allow_non_str_keys` (bool, default False): Allow non-string dict keys
- `deterministic_sets` (bool, default False): Sort sets deterministically (slower)
- `decimal_as_float` (bool, default False): Serialize Decimal as float (precision loss)
- `enum_as_name` (bool, default False): Serialize Enum as .name (else .value)
- `passthrough_datetime` (bool, default False): Custom datetime handling
- `safe_fallback` (bool, default False): Never raise on unknown types (for logging)
- `fallback_clip` (int, default 2048): Max length for fallback repr
- `default` (Callable | None, default None): Custom type serializer
- `options` (int | None, default None): Raw orjson option flags

**Returns:**

- str | bytes: JSON representation (str if decode=True, bytes if decode=False)

**Examples:**

```python
from lionherd_core.ln import json_dumps
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Basic serialization
data = {"name": "alice", "age": 30}
json_str = json_dumps(data)
# '{"name":"alice","age":30}'

# Pretty printing
json_str = json_dumps(data, pretty=True)
# {
#   "name": "alice",
#   "age": 30
# }

# Custom types (auto-handled)
data = {
    "path": Path("/tmp/file.txt"),
    "timestamp": datetime.now(),
    "amount": Decimal("123.45"),
}
json_str = json_dumps(data)
# Path -> str, datetime -> ISO8601, Decimal -> str

# Decimal as float (smaller, faster, precision loss)
json_str = json_dumps(data, decimal_as_float=True)

# Pydantic models (auto-handled)
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str

user = User(name="alice", email="a@b.com")
json_str = json_dumps(user)
# Calls user.model_dump() automatically

# Deterministic output (for comparison/hashing)
json_str = json_dumps(data, sort_keys=True, deterministic_sets=True)

# Safe fallback for logging
json_str = json_dumps(
    {"obj": some_weird_object},
    safe_fallback=True,  # Never raises
)

# Return bytes for network transmission
json_bytes = json_dumps(data, decode=False)
assert isinstance(json_bytes, bytes)
```

**Notes:**

- Uses orjson for high performance (2-3x faster than stdlib json)
- Automatically handles Pydantic models, dataclasses, Path, UUID, Decimal, Enum
- `safe_fallback=True` is recommended for logging only (loses type safety)
- `deterministic_sets=True` adds overhead - use only when needed

### `json_dumpb()`

Serialize to bytes (fast path). Prefer this in hot code.

**Signature:**

```python
def json_dumpb(
    obj: Any,
    *,
    # Same parameters as json_dumps except no decode
    pretty: bool = False,
    sort_keys: bool = False,
    naive_utc: bool = False,
    utc_z: bool = False,
    append_newline: bool = False,
    allow_non_str_keys: bool = False,
    deterministic_sets: bool = False,
    decimal_as_float: bool = False,
    enum_as_name: bool = False,
    passthrough_datetime: bool = False,
    safe_fallback: bool = False,
    fallback_clip: int = 2048,
    default: Callable[[Any], Any] | None = None,
    options: int | None = None,
) -> bytes: ...
```

**Returns:**

- bytes: JSON representation as bytes

**Examples:**

```python
from lionherd_core.ln import json_dumpb

# Direct bytes output (no decode step)
data = {"key": "value"}
json_bytes = json_dumpb(data)

# Write to file
with open("data.json", "wb") as f:
    f.write(json_dumpb(data, pretty=True))

# Network transmission
socket.sendall(json_dumpb(data))
```

**Notes:**

- Slightly faster than json_dumps (no UTF-8 decode step)
- Use when you need bytes output (files, network, etc.)

### `json_lines_iter()`

Stream an iterable as NDJSON (one JSON object per line) in bytes.

**Signature:**

```python
def json_lines_iter(
    it: Iterable[Any],
    *,
    # default() configuration
    deterministic_sets: bool = False,
    decimal_as_float: bool = False,
    enum_as_name: bool = False,
    passthrough_datetime: bool = False,
    safe_fallback: bool = False,
    fallback_clip: int = 2048,
    # options
    naive_utc: bool = False,
    utc_z: bool = False,
    allow_non_str_keys: bool = False,
    # advanced
    default: Callable[[Any], Any] | None = None,
    options: int | None = None,
) -> Iterable[bytes]: ...
```

**Yields:**

- bytes: JSON line with trailing newline

**Examples:**

```python
from lionherd_core.ln import json_lines_iter

# Stream large dataset to file
users = [{"id": i, "name": f"user{i}"} for i in range(100000)]

with open("users.jsonl", "wb") as f:
    for line in json_lines_iter(users):
        f.write(line)

# Stream to HTTP response
async def stream_response():
    for line in json_lines_iter(data):
        yield line

# Memory-efficient processing
for line in json_lines_iter(large_dataset, safe_fallback=True):
    process(line)
```

**Notes:**

- Always appends newline (OPT_APPEND_NEWLINE)
- Memory-efficient for large datasets (lazy evaluation)
- Compatible with NDJSON parsers

### `json_dict()`

Round-trip serialize to dict (useful for type coercion).

**Signature:**

```python
def json_dict(obj: Any, /, **kwargs: Any) -> dict: ...
```

**Parameters:**

- `obj` (Any): Object to serialize
- `**kwargs` (Any): Arguments passed to json_dumpb

**Returns:**

- dict: Deserialized dictionary

**Examples:**

```python
from lionherd_core.ln import json_dict
from datetime import datetime

# Type coercion via serialization
data = {
    "timestamp": datetime.now(),
    "nested": SomeCustomObject(),
}

# Serialize to JSON and deserialize back (coerces types)
clean_dict = json_dict(data)
# timestamp -> ISO8601 string, nested -> dict
```

**Notes:**

- Useful for normalizing types through JSON serialization
- Equivalent to `orjson.loads(json_dumpb(obj, **kwargs))`

### `get_orjson_default()`

Build a fast, extensible default= callable for orjson.dumps.

**Signature:**

```python
def get_orjson_default(
    *,
    order: list[type] | None = None,
    additional: Mapping[type, Callable[[Any], Any]] | None = None,
    extend_default: bool = True,
    deterministic_sets: bool = False,
    decimal_as_float: bool = False,
    enum_as_name: bool = False,
    passthrough_datetime: bool = False,
    safe_fallback: bool = False,
    fallback_clip: int = 2048,
) -> Callable[[Any], Any]: ...
```

**Parameters:**

- `order` (list[type] | None, default None): Type resolution order
- `additional` (Mapping[type, Callable] | None, default None): Custom type serializers
- `extend_default` (bool, default True): Extend base types or replace
- `deterministic_sets` (bool, default False): Sort sets deterministically
- `decimal_as_float` (bool, default False): Serialize Decimal as float
- `enum_as_name` (bool, default False): Serialize Enum as .name
- `passthrough_datetime` (bool, default False): Custom datetime handling
- `safe_fallback` (bool, default False): Never raise on unknown types
- `fallback_clip` (int, default 2048): Max length for fallback repr

**Returns:**

- Callable[[Any], Any]: Serializer function for orjson

**Examples:**

```python
from lionherd_core.ln import get_orjson_default
import orjson
from typing import NamedTuple

# Custom type serializer
class Point(NamedTuple):
    x: float
    y: float

def serialize_point(p: Point) -> dict:
    return {"x": p.x, "y": p.y}

# Build custom serializer
default_fn = get_orjson_default(
    additional={Point: serialize_point},
)

# Use with orjson
data = {"location": Point(1.0, 2.0)}
json_bytes = orjson.dumps(data, default=default_fn)
# b'{"location":{"x":1.0,"y":2.0}}'

# Safe fallback for logging
log_serializer = get_orjson_default(
    safe_fallback=True,
    fallback_clip=500,
)
```

**Notes:**

- Caches type lookups for performance
- Handles Pydantic models via duck-typed model_dump()
- Use `additional` for project-specific types
- `safe_fallback=True` is for logging only

### `make_options()`

Compose orjson option bit flags succinctly.

**Signature:**

```python
def make_options(
    *,
    pretty: bool = False,
    sort_keys: bool = False,
    naive_utc: bool = False,
    utc_z: bool = False,
    append_newline: bool = False,
    passthrough_datetime: bool = False,
    allow_non_str_keys: bool = False,
) -> int: ...
```

**Parameters:**

- `pretty` (bool, default False): Indent with 2 spaces
- `sort_keys` (bool, default False): Sort dictionary keys
- `naive_utc` (bool, default False): Assume naive datetimes are UTC
- `utc_z` (bool, default False): Use 'Z' suffix for UTC
- `append_newline` (bool, default False): Append newline
- `passthrough_datetime` (bool, default False): Custom datetime handling
- `allow_non_str_keys` (bool, default False): Allow non-string keys

**Returns:**

- int: Bitwise OR of orjson option flags

**Examples:**

```python
from lionherd_core.ln import make_options
import orjson

# Build option flags
opts = make_options(pretty=True, sort_keys=True)

# Use with orjson directly
json_bytes = orjson.dumps(data, option=opts)
```

**Notes:**

- Helper for working with orjson directly
- Most users should use json_dumps/json_dumpb instead

## List Processing

### `to_list()`

Convert input to list with optional transformations.

**Signature:**

```python
def to_list(
    input_: Any,
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    use_values: bool = False,
    flatten_tuple_set: bool = False,
) -> list: ...
```

**Parameters:**

- `input_` (Any): Value to convert
- `flatten` (bool, default False): Recursively flatten nested iterables
- `dropna` (bool, default False): Remove None and undefined values
- `unique` (bool, default False): Remove duplicates (requires flatten=True)
- `use_values` (bool, default False): Extract values from enums/mappings
- `flatten_tuple_set` (bool, default False): Include tuples/sets in flattening

**Returns:**

- list: Processed list

**Raises:**

- ValueError: If unique=True without flatten=True

**Examples:**

```python
from lionherd_core.ln import to_list

# Basic conversion
result = to_list(5)
# [5]

result = to_list((1, 2, 3))
# [1, 2, 3]

# Flatten nested structures
nested = [[1, 2], [3, [4, 5]], 6]
result = to_list(nested, flatten=True)
# [1, 2, 3, 4, 5, 6]

# Drop None values
data = [1, None, 2, None, 3]
result = to_list(data, dropna=True)
# [1, 2, 3]

# Remove duplicates (requires flatten)
data = [[1, 2], [2, 3], [3, 4]]
result = to_list(data, flatten=True, unique=True)
# [1, 2, 3, 4]

# Extract values from enum
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    DONE = "done"

result = to_list(Status, use_values=True)
# ["pending", "done"]

# Extract values from dict
data = {"a": 1, "b": 2, "c": 3}
result = to_list(data, use_values=True)
# [1, 2, 3]

# Flatten tuples and sets
data = [(1, 2), {3, 4}, [5, 6]]
result = to_list(data, flatten=True, flatten_tuple_set=True)
# [1, 2, 3, 4, 5, 6]

# LLM output processing
from pydantic import BaseModel

class TaskList(BaseModel):
    tasks: list[str]

llm_output = [
    TaskList(tasks=["task1", "task2"]),
    TaskList(tasks=["task3"]),
    None,  # Invalid output
]

tasks = to_list(
    llm_output,
    use_values=True,  # Extract tasks field
    flatten=True,     # Flatten nested lists
    dropna=True,      # Remove None
)
# ["task1", "task2", "task3"]
```

**Notes:**

- Pydantic Undefined/Unset treated as None for dropna
- Strings/bytes not flattened (unless use_values=True)
- Unique uses hash_dict() for unhashable types
- Mappings converted to [mapping] unless use_values=True

### `lcall()`

Apply function to each element synchronously with optional input/output processing.

**Signature:**

```python
def lcall(
    input_: Iterable[T] | T,
    func: Callable[[T], R] | Iterable[Callable[[T], R]],
    /,
    *args: Any,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_use_values: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    **kwargs: Any,
) -> list[R]: ...
```

**Parameters:**

- `input_` (Iterable[T] | T): Items to process
- `func` (Callable[[T], R] | Iterable[Callable]): Function to apply
- `*args` (Any): Positional arguments passed to func
- `input_flatten` (bool, default False): Flatten input structures
- `input_dropna` (bool, default False): Remove None/undefined from input
- `input_unique` (bool, default False): Remove duplicate inputs
- `input_use_values` (bool, default False): Extract values from enums/mappings
- `input_flatten_tuple_set` (bool, default False): Include tuples/sets in input flattening
- `output_flatten` (bool, default False): Flatten output structures
- `output_dropna` (bool, default False): Remove None/undefined from output
- `output_unique` (bool, default False): Remove duplicate outputs
- `output_flatten_tuple_set` (bool, default False): Include tuples/sets in output flattening
- `**kwargs` (Any): Keyword arguments passed to func

**Returns:**

- list[R]: Results

**Raises:**

- ValueError: If func is not callable or output_unique without flatten/dropna
- TypeError: If func or input processing fails

**Examples:**

```python
from lionherd_core.ln import lcall

# Basic list mapping
numbers = [1, 2, 3, 4, 5]
result = lcall(numbers, lambda x: x * 2)
# [2, 4, 6, 8, 10]

# With input processing
nested = [[1, 2], [3, 4], [5, 6]]
result = lcall(
    nested,
    lambda x: x * 2,
    input_flatten=True,  # Flatten to [1, 2, 3, 4, 5, 6]
)
# [2, 4, 6, 8, 10, 12]

# With output processing
result = lcall(
    numbers,
    lambda x: [x, x * 2],
    output_flatten=True,  # Flatten nested results
)
# [1, 2, 2, 4, 3, 6, 4, 8, 5, 10]

# Drop None from output
def maybe_process(x: int) -> int | None:
    return x * 2 if x > 2 else None

result = lcall([1, 2, 3, 4], maybe_process, output_dropna=True)
# [6, 8]

# Unique output
result = lcall(
    [1, 2, 3, 2, 1],
    lambda x: x % 2,
    output_flatten=True,
    output_unique=True,
)
# [1, 0]

# Additional arguments
def add(x: int, y: int, z: int = 0) -> int:
    return x + y + z

result = lcall([1, 2, 3], add, 10, z=5)
# [16, 17, 18]
```

**Notes:**

- Synchronous version of alcall (no async overhead)
- Supports InterruptedError for graceful cancellation
- Use for CPU-bound transformations

## Dictionary Conversion

### `to_dict()`

Convert various input types to dictionary with optional recursive processing.

**Signature:**

```python
def to_dict(
    input_: Any,
    /,
    *,
    prioritize_model_dump: bool = False,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    parser: Callable[[str], Any] | None = None,
    recursive: bool = False,
    max_recursive_depth: int | None = None,
    recursive_python_only: bool = True,
    use_enum_values: bool = False,
    **kwargs: Any,
) -> dict[str | int, Any]: ...
```

**Parameters:**

- `input_` (Any): Object to convert
- `prioritize_model_dump` (bool, default False): Prefer .model_dump() for Pydantic models
- `fuzzy_parse` (bool, default False): Use fuzzy JSON parsing for strings
- `suppress` (bool, default False): Return {} on errors instead of raising
- `parser` (Callable | None, default None): Custom parser for string inputs
- `recursive` (bool, default False): Recursively process nested structures
- `max_recursive_depth` (int | None, default None): Maximum recursion depth (default 5, max 10)
- `recursive_python_only` (bool, default True): Only recurse into Python builtins
- `use_enum_values` (bool, default False): Use .value for Enum members
- `**kwargs` (Any): Additional kwargs passed to parser

**Returns:**

- dict[str | int, Any]: Dictionary representation

**Examples:**

````python
from lionherd_core.ln import to_dict
from pydantic import BaseModel

# Pydantic model
class User(BaseModel):
    name: str
    email: str

user = User(name="alice", email="a@b.com")
result = to_dict(user)
# {"name": "alice", "email": "a@b.com"}

# JSON string
json_str = '{"key": "value"}'
result = to_dict(json_str)
# {"key": "value"}

# Fuzzy JSON parsing (markdown code blocks)
markdown = """
Here's the data:
```json
{"key": "value"}
```

"""
result = to_dict(markdown, fuzzy_parse=True)

# {"key": "value"}

# Dataclass

from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

point = Point(1.0, 2.0)
result = to_dict(point)

# {"x": 1.0, "y": 2.0}

# Iterable (converted to enumerated dict)

result = to_dict([10, 20, 30])

# {0: 10, 1: 20, 2: 30}

# Set

result = to_dict({1, 2, 3})

# {1: 1, 2: 2, 3: 3}

# Enum class

from enum import Enum

class Status(Enum):
    PENDING = "pending"
    DONE = "done"

result = to_dict(Status, use_enum_values=True)

# {"PENDING": "pending", "DONE": "done"}

# Recursive conversion

nested = {
    "user": '{"name": "alice"}',  # JSON string
    "items": [(1, 2), (3, 4)],
}
result = to_dict(nested, recursive=True, fuzzy_parse=True)

# {

# "user": {"name": "alice"},  # Parsed

# "items": {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}  # Enumerated

# }

# Suppress errors

result = to_dict("invalid data", suppress=True)

# {}

````

**Notes:**

- Tries multiple conversion strategies (model_dump, to_dict, dict, **dict**)
- JSON strings automatically parsed (via orjson)
- None/Undefined converted to {}
- Recursive mode processes nested JSON strings and objects

## General Utilities

### `now_utc()`

Get current UTC datetime.

**Signature:**

```python
def now_utc() -> datetime: ...
```

**Returns:**

- datetime: Current UTC datetime with timezone info

**Examples:**

```python
from lionherd_core.ln import now_utc

timestamp = now_utc()
# datetime.datetime(2025, 11, 9, 14, 30, 0, tzinfo=UTC)

# Use in serialization
data = {"created_at": now_utc()}
```

**Notes:**

- Always returns timezone-aware datetime (UTC)
- Equivalent to `datetime.now(UTC)`

### `acreate_path()`

Generate file path asynchronously with optional timeout.

**Signature:**

```python
async def acreate_path(
    directory: StdPath | AsyncPath | str,
    filename: str,
    extension: str | None = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
    timeout: float | None = None,
) -> AsyncPath: ...
```

**Parameters:**

- `directory` (Path | AsyncPath | str): Base directory path
- `filename` (str): Target filename (may contain subdirectory with /)
- `extension` (str | None, default None): File extension (if filename doesn't have one)
- `timestamp` (bool, default False): Add timestamp to filename
- `dir_exist_ok` (bool, default True): Allow existing directories
- `file_exist_ok` (bool, default False): Allow existing files
- `time_prefix` (bool, default False): Put timestamp before filename
- `timestamp_format` (str | None, default None): Custom strftime format
- `random_hash_digits` (int, default 0): Add random hash suffix (0 = disabled)
- `timeout` (float | None, default None): Maximum time for async I/O (seconds)

**Returns:**

- AsyncPath: Created/validated file path

**Raises:**

- ValueError: If filename contains backslash
- FileExistsError: If file exists and file_exist_ok is False
- TimeoutError: If timeout is exceeded

**Examples:**

```python
from lionherd_core.ln import acreate_path

# Basic usage
path = await acreate_path("/tmp", "output.txt")
# AsyncPath("/tmp/output.txt")

# With subdirectory in filename
path = await acreate_path("/tmp", "subdir/output.txt")
# AsyncPath("/tmp/subdir/output.txt")

# Add timestamp
path = await acreate_path(
    "/tmp",
    "report",
    extension="pdf",
    timestamp=True,
)
# AsyncPath("/tmp/report_20251109143000.pdf")

# Time prefix
path = await acreate_path(
    "/tmp",
    "log",
    extension="txt",
    timestamp=True,
    time_prefix=True,
)
# AsyncPath("/tmp/20251109143000_log.txt")

# Custom timestamp format
path = await acreate_path(
    "/tmp",
    "backup",
    extension="tar.gz",
    timestamp=True,
    timestamp_format="%Y-%m-%d",
)
# AsyncPath("/tmp/backup_2025-11-09.tar.gz")

# Random suffix for uniqueness
path = await acreate_path(
    "/tmp",
    "temp",
    extension="json",
    random_hash_digits=8,
)
# AsyncPath("/tmp/temp-a3f8b2c1.json")

# With timeout
path = await acreate_path(
    "/tmp",
    "output.txt",
    timeout=5.0,  # Max 5 seconds
)

# Fail if file exists
path = await acreate_path(
    "/tmp",
    "critical.txt",
    file_exist_ok=False,  # Raise if exists
)
```

**Notes:**

- Creates parent directories automatically
- Uses AsyncPath for async I/O operations
- Timeout applies to mkdir and exists checks

### `get_bins()`

Organize indices into bins by cumulative length.

**Signature:**

```python
def get_bins(input_: list[str], upper: int) -> list[list[int]]: ...
```

**Parameters:**

- `input_` (list[str]): List of strings to bin
- `upper` (int): Maximum cumulative length per bin

**Returns:**

- list[list[int]]: List of bins (each bin is list of indices)

**Examples:**

```python
from lionherd_core.ln import get_bins

# Bin strings by total length
strings = ["abc", "de", "f", "ghij", "kl"]
bins = get_bins(strings, upper=5)
# [[0, 1], [2, 3], [4]]
# Bin 1: "abc" (3) + "de" (2) = 5
# Bin 2: "f" (1) + "ghij" (4) = 5
# Bin 3: "kl" (2)

# Use for batching by token count
texts = ["This is a long text...", "Short", "Another long one..."]
token_counts = [len(t) for t in texts]
bins = get_bins(token_counts, upper=100)
```

**Notes:**

- Greedy binning algorithm (fills current bin before starting new)
- Useful for batching by size constraints (tokens, bytes, etc.)

### `import_module()`

Import module by path with optional name extraction.

**Signature:**

```python
def import_module(
    package_name: str,
    module_name: str | None = None,
    import_name: str | list | None = None,
) -> Any: ...
```

**Parameters:**

- `package_name` (str): Package name
- `module_name` (str | None, default None): Module name within package
- `import_name` (str | list | None, default None): Specific names to import from module

**Returns:**

- Any: Imported module or name(s)

**Raises:**

- ImportError: If import fails

**Examples:**

```python
from lionherd_core.ln import import_module

# Import package
json = import_module("json")

# Import module from package
dumps = import_module("json", module_name="encoder", import_name="JSONEncoder")

# Import multiple names
BaseModel, Field = import_module(
    "pydantic",
    import_name=["BaseModel", "Field"],
)

# Dynamic import
module_name = "numpy"
if is_import_installed(module_name):
    np = import_module(module_name)
```

**Notes:**

- Uses `__import__` for dynamic imports
- Returns single object for single import_name, list for multiple

### `is_import_installed()`

Check if package is installed.

**Signature:**

```python
def is_import_installed(package_name: str) -> bool: ...
```

**Parameters:**

- `package_name` (str): Package name to check

**Returns:**

- bool: True if package is installed

**Examples:**

```python
from lionherd_core.ln import is_import_installed

# Check optional dependencies
if is_import_installed("numpy"):
    import numpy as np
    # Use numpy
else:
    # Fallback implementation
    pass

# Conditional imports
HAS_TORCH = is_import_installed("torch")
HAS_TF = is_import_installed("tensorflow")

if not HAS_TORCH:
    raise ImportError("PyTorch required for this feature")
```

**Notes:**

- Uses `importlib.util.find_spec()` for fast checking
- Does not import the package (no side effects)

## Usage Patterns

### Pattern 1: LLM Output Processing

````python
from lionherd_core.ln import fuzzy_validate_pydantic, to_list
from pydantic import BaseModel

class Task(BaseModel):
    task_name: str
    priority: int
    assignee: str

# LLM returns markdown with JSON
llm_response = """
Here are the tasks:
```json
{
  "taskName": "Fix bug #123",
  "priority": 1,
  "assignee": "alice"
}
```

"""

# Parse and validate

task = fuzzy_validate_pydantic(
    llm_response,
    Task,
    fuzzy_parse=True,   # Extract JSON from markdown
    fuzzy_match=True,   # Match taskName -> task_name
)

# Process list outputs

tasks_list = [task1, task2, None, task3]
clean_tasks = to_list(tasks_list, dropna=True, unique=True)

````

### Pattern 2: Parallel API Calls with Retry

```python
from lionherd_core.ln import alcall, bcall

# Fetch data from multiple endpoints
urls = ["https://api.example.com/users/1", ...]

results = await alcall(
    urls,
    fetch_data,
    retry_attempts=3,
    retry_backoff=2.0,
    max_concurrent=10,
    retry_timeout=30.0,
    return_exceptions=True,  # Don't fail on single error
)

# Process large dataset in batches
async for batch in bcall(
    large_dataset,
    process_item,
    batch_size=100,
    max_concurrent=5,
):
    await save_results(batch)
```

### Pattern 3: High-Performance JSON Serialization

```python
from lionherd_core.ln import json_dumps, json_lines_iter

# Serialize complex data
data = {
    "users": [User(name="alice"), User(name="bob")],
    "timestamp": datetime.now(),
    "config": {"path": Path("/tmp/data")},
}

# JSON string (auto-handles Pydantic, Path, datetime)
json_str = json_dumps(data, pretty=True)

# Stream large dataset to file
with open("users.jsonl", "wb") as f:
    for line in json_lines_iter(users):
        f.write(line)
```

### Pattern 4: Content-Based Hashing

```python
from lionherd_core.ln import hash_dict

# Cache LLM responses by input hash
cache = {}

def get_llm_response(prompt: dict) -> str:
    cache_key = hash_dict(prompt)
    if cache_key in cache:
        return cache[cache_key]

    response = call_llm(prompt)
    cache[cache_key] = response
    return response

# Deduplicate configs
configs = [{"model": "gpt-4"}, {"model": "gpt-4"}, {"model": "claude"}]
unique_configs = list({hash_dict(c): c for c in configs}.values())
```

### Pattern 5: Universal Data Conversion

```python
from lionherd_core.ln import to_dict, to_list

# Convert various formats to dict
sources = [
    '{"key": "value"}',           # JSON string
    User(name="alice"),           # Pydantic model
    {"nested": [1, 2, 3]},        # Dict
    Point(x=1, y=2),              # Dataclass
]

dicts = [to_dict(s, fuzzy_parse=True, recursive=True) for s in sources]

# Convert to lists with transformations
nested = [[1, 2], [3, [4, 5]], None]
flat_unique = to_list(nested, flatten=True, dropna=True, unique=True)
# [1, 2, 3, 4, 5]
```

## See Also

- **Related Modules**:
  - [base](../base/element.md): Element class with serialization
  - [types](../types/model.md): Type definitions and utilities
  - [libs](../libs/overview.md): Low-level library functions
- **Related Guides**:
  - [Async Operations](../../user_guide/async.md): Async execution patterns
  - [Data Processing](../../user_guide/data_processing.md): Data transformation workflows
  - [Serialization](../../user_guide/serialization.md): JSON serialization guide

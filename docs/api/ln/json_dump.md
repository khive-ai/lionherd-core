# JSON Serialization (ln._json_dump)

> Fast, extensible JSON serialization with orjson backend and custom type support

## Overview

The `ln._json_dump` module provides high-performance JSON serialization powered by orjson with extensive customization for handling Python types beyond standard JSON. It offers fast default serializers for common types (Path, Decimal, Enum, datetime), safe fallback modes for logging, and streaming support for large datasets.

**Key Functions:**

- **Serialization**: `json_dumpb()`, `json_dumps()` for bytes/string output
- **Configuration**: `get_orjson_default()` for custom type handlers, `make_options()` for orjson flags
- **Streaming**: `json_lines_iter()` for NDJSON output of large iterables

**Common Use Cases:**

- Serializing API responses with Path, UUID, datetime, Decimal types
- Logging structured data with safe fallback (no serialization errors)
- Streaming large datasets as NDJSON (newline-delimited JSON)
- Building custom serializers with deterministic output (sorted sets, consistent datetime formatting)
- Converting Python objects to JSON dictionaries via orjson round-trip

**When to Use:**

- Production code requiring fast JSON serialization (orjson is 2-5× faster than stdlib json)
- Handling types beyond JSON spec (Path, Decimal, Enum, UUID, datetime)
- Logging pipelines where serialization errors must not crash (safe_fallback mode)
- Streaming APIs with large response bodies (json_lines_iter)

**When NOT to Use:**

- Simple JSON with only str/int/float/bool/list/dict (stdlib json sufficient)
- When orjson dependency is unacceptable (use stdlib json)
- Pretty-printing for human reading (stdlib json.dumps indent is fine)

## Functions

### Default Handler Factory

#### `get_orjson_default()`

Build a fast, extensible `default=` callable for orjson.dumps with custom type handling.

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
) -> Callable[[Any], Any]:
```

**Parameters:**

**order** : list of type, optional

Custom type resolution order for serializers.

- Controls which types are checked first during serialization
- Default order: `[Path, Decimal, set, frozenset]` (with Enum/datetime prepended if enabled)
- If `extend_default=True`, provided types are appended to default order
- If `extend_default=False`, replaces default order entirely
- Types already in orjson's native fast path (datetime/date/time/UUID) are automatically excluded unless `passthrough_datetime=True`

**additional** : Mapping[type, Callable], optional

Custom serializers for additional types.

- Maps type to serialization function: `{MyClass: lambda obj: obj.to_dict()}`
- Merged with built-in serializers (Path, Decimal, etc.)
- Overrides built-in serializers if same type provided
- Default: `None` (no custom serializers)

**extend_default** : bool, default True

Whether to extend built-in type order or replace it.

- `True`: Provided `order` types appended after built-ins
- `False`: Provided `order` replaces built-in order entirely
- Only relevant when `order` parameter is provided

**deterministic_sets** : bool, default False

Sort sets/frozensets deterministically for reproducible output.

- `False`: Sets serialized to list with arbitrary order (fast)
- `True`: Sets sorted by `(class_name, normalized_str)` for determinism (slower)
- Normalized sorting removes memory addresses from repr output
- Useful for testing, caching, or content-based hashing

**decimal_as_float** : bool, default False

Serialize Decimal as float instead of string.

- `False`: `Decimal("3.14")` → `"3.14"` (precise, no loss)
- `True`: `Decimal("3.14")` → `3.14` (faster, smaller, precision loss possible)
- Use `True` for performance when precision loss acceptable

**enum_as_name** : bool, default False

Serialize Enum as `.name` instead of `.value`.

- `False`: `Color.RED` → `1` (uses `.value`, orjson default)
- `True`: `Color.RED` → `"RED"` (uses `.name`)
- Useful when enum names are more meaningful than values

**passthrough_datetime** : bool, default False

Enable passthrough mode for datetime serialization.

- `False`: orjson handles datetime natively (ISO 8601 format)
- `True`: datetime routed through default handler (calls `.isoformat()`)
- Must also set `OPT_PASSTHROUGH_DATETIME` in options via `make_options(passthrough_datetime=True)`
- Rarely needed unless custom datetime formatting required

**safe_fallback** : bool, default False

Never raise errors for unknown types (for logging only).

- `False`: Raises `TypeError` for non-serializable types
- `True`: Fallback behavior for unknown types:
  - `Exception` objects → `{"type": "ExceptionName", "message": "error message"}`
  - Other objects → `repr(obj)` clipped to `fallback_clip` length
- **WARNING**: Only use for logging. Produces lossy output.

**fallback_clip** : int, default 2048

Maximum repr length when `safe_fallback=True`.

- Clips repr output beyond this length: `"...(+N chars)"`
- Default 2048 characters balances detail vs. log bloat
- Only relevant when `safe_fallback=True`

**Returns:**

- Callable[[Any], Any]: Default handler function for orjson.dumps

**Examples:**

```python
>>> from lionherd_core.ln._json_dump import get_orjson_default
>>> from pathlib import Path
>>> from decimal import Decimal
>>> import orjson

# Basic usage with built-in types
>>> default = get_orjson_default()
>>> orjson.dumps({"path": Path("/tmp")}, default=default)
b'{"path":"/tmp"}'

# Decimal as float for smaller output
>>> default = get_orjson_default(decimal_as_float=True)
>>> orjson.dumps({"price": Decimal("19.99")}, default=default)
b'{"price":19.99}'

# Enum as name instead of value
>>> from enum import Enum
>>> class Status(Enum):
...     ACTIVE = 1
...     INACTIVE = 2
>>> default = get_orjson_default(enum_as_name=True)
>>> orjson.dumps({"status": Status.ACTIVE}, default=default)
b'{"status":"ACTIVE"}'

# Deterministic set ordering
>>> default = get_orjson_default(deterministic_sets=True)
>>> data = {"tags": {3, 1, 2}}
>>> orjson.dumps(data, default=default)
b'{"tags":[1,2,3]}'  # Always same order

# Custom serializer for your class
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
>>> default = get_orjson_default(
...     additional={Point: lambda p: {"x": p.x, "y": p.y}}
... )
>>> orjson.dumps({"pos": Point(10, 20)}, default=default)
b'{"pos":{"x":10,"y":20}}'

# Safe fallback for logging (never crashes)
>>> default = get_orjson_default(safe_fallback=True)
>>> class Custom: pass
>>> orjson.dumps({"obj": Custom()}, default=default)
b'{"obj":"<__main__.Custom object at 0x?>"}'
```

**Usage Patterns:**

```python
# Pattern 1: Reusable default handler
from lionherd_core.ln._json_dump import get_orjson_default
import orjson

# Create once, reuse for all serializations
default = get_orjson_default(
    deterministic_sets=True,
    decimal_as_float=True,
    enum_as_name=True,
)

def serialize_response(data: dict) -> bytes:
    return orjson.dumps(data, default=default)

# Pattern 2: Custom type support
from datetime import datetime

class CustomTimestamp:
    def __init__(self, dt: datetime):
        self.dt = dt

    def to_iso(self) -> str:
        return self.dt.isoformat()

default = get_orjson_default(
    additional={CustomTimestamp: lambda ts: ts.to_iso()}
)

# Pattern 3: Duck-typed model support (automatic)
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

default = get_orjson_default()
user = User(name="Alice", age=30)
orjson.dumps(user, default=default)
# Automatically calls user.model_dump()

# Pattern 4: Safe logging
import logging

log_default = get_orjson_default(
    safe_fallback=True,  # Never crash on unknown types
    fallback_clip=500,   # Shorter clips for logs
)

def log_event(event: dict):
    # Safe even if event contains non-serializable objects
    json_str = orjson.dumps(event, default=log_default).decode()
    logging.info(json_str)
```

**Notes:**

- **Caching**: Default handlers are cached via `_cached_default()` (LRU cache size 128)
- **Duck Typing**: Automatically handles objects with `.model_dump()` (Pydantic) or `.dict()` methods
- **Performance**: Type resolution uses isinstance checks with caching (first lookup slow, subsequent fast)
- **Native Types**: orjson natively serializes str/int/float/bool/None/list/dict/datetime/date/time/UUID at C speed
- **Passthrough**: `passthrough_datetime=True` requires `OPT_PASSTHROUGH_DATETIME` flag in options (use `make_options(passthrough_datetime=True)`)

**See Also:**

- `json_dumpb()`: Convenience wrapper that uses cached default handlers
- `make_options()`: Compose orjson option flags
- [orjson documentation](https://github.com/ijl/orjson): Underlying library

---

### Options Builder

#### `make_options()`

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
) -> int:
```

**Parameters:**

**pretty** : bool, default False

Enable pretty-printing with 2-space indentation.

- `True`: Adds `OPT_INDENT_2` (human-readable JSON)
- `False`: Compact output (machine-readable)

**sort_keys** : bool, default False

Sort dictionary keys alphabetically.

- `True`: Adds `OPT_SORT_KEYS` (deterministic output)
- `False`: Keys in insertion order (Python 3.7+)

**naive_utc** : bool, default False

Serialize naive datetime objects as UTC.

- `True`: Assumes naive datetimes are UTC, serializes as `"YYYY-MM-DDTHH:MM:SSZ"`
- `False`: Naive datetimes serialized without timezone suffix

**utc_z** : bool, default False

Use "Z" suffix instead of "+00:00" for UTC.

- `True`: UTC datetimes → `"2025-11-09T14:30:45Z"`
- `False`: UTC datetimes → `"2025-11-09T14:30:45+00:00"`

**append_newline** : bool, default False

Append `\n` to output (for NDJSON/log files).

- `True`: Adds `OPT_APPEND_NEWLINE`
- `False`: No trailing newline

**passthrough_datetime** : bool, default False

Route datetime serialization through default handler.

- `True`: Adds `OPT_PASSTHROUGH_DATETIME`
- `False`: orjson handles datetime natively
- Must also implement datetime serializer in default handler

**allow_non_str_keys** : bool, default False

Allow non-string dictionary keys (int, float, etc.).

- `True`: Adds `OPT_NON_STR_KEYS`
- `False`: Only string keys allowed (standard JSON)

**Returns:**

- int: Combined option flags (bitwise OR of selected flags)

**Examples:**

```python
>>> from lionherd_core.ln._json_dump import make_options
>>> import orjson

# No options (compact output)
>>> opts = make_options()
>>> orjson.dumps({"a": 1}, option=opts)
b'{"a":1}'

# Pretty-printing
>>> opts = make_options(pretty=True)
>>> orjson.dumps({"a": 1}, option=opts)
b'{\n  "a": 1\n}'

# Sorted keys for determinism
>>> opts = make_options(sort_keys=True)
>>> orjson.dumps({"z": 1, "a": 2}, option=opts)
b'{"a":2,"z":1}'

# UTC datetime formatting
>>> from datetime import datetime, UTC
>>> opts = make_options(utc_z=True)
>>> orjson.dumps({"ts": datetime(2025, 11, 9, 14, 30, tzinfo=UTC)}, option=opts)
b'{"ts":"2025-11-09T14:30:00Z"}'

# NDJSON (newline-delimited)
>>> opts = make_options(append_newline=True)
>>> orjson.dumps({"id": 1}, option=opts)
b'{"id":1}\n'

# Combined options
>>> opts = make_options(pretty=True, sort_keys=True)
>>> orjson.dumps({"z": 1, "a": 2}, option=opts)
b'{\n  "a": 2,\n  "z": 1\n}'
```

**Notes:**

- **Bitwise OR**: Function combines flags via `|=` operator
- **Performance**: Pretty-printing and key sorting add overhead
- **NDJSON**: Use `append_newline=True` for streaming log formats
- **Determinism**: `sort_keys=True` ensures reproducible output for caching/hashing

**See Also:**

- [orjson options](https://github.com/ijl/orjson#option): Full list of orjson flags
- `json_dumpb()`: Uses make_options internally

---

### Serialization Functions

#### `json_dumpb()`

Serialize to **bytes** (fast path). Prefer this in hot code.

**Signature:**

```python
def json_dumpb(
    obj: Any,
    *,
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
) -> bytes:
```

**Parameters:**

**obj** : Any

Object to serialize (dict, list, primitives, custom types).

**pretty** : bool, default False
**sort_keys** : bool, default False
**naive_utc** : bool, default False
**utc_z** : bool, default False
**append_newline** : bool, default False
**allow_non_str_keys** : bool, default False

Passed to `make_options()`. See `make_options()` documentation.

**deterministic_sets** : bool, default False
**decimal_as_float** : bool, default False
**enum_as_name** : bool, default False
**passthrough_datetime** : bool, default False
**safe_fallback** : bool, default False
**fallback_clip** : int, default 2048

Passed to `get_orjson_default()`. See `get_orjson_default()` documentation.

**default** : Callable[[Any], Any], optional

Custom default handler for non-JSON types.

- If `None`: Uses cached default handler from `_cached_default()`
- If provided: Overrides automatic default handler
- Default: `None` (use cached handler)

**options** : int, optional

Pre-composed orjson option flags.

- If `None`: Composes options via `make_options()` from individual flags
- If provided: Uses directly (overrides individual flag parameters)
- Default: `None` (compose from flags)

**Returns:**

- bytes: UTF-8 encoded JSON bytes

**Raises:**

- TypeError: If obj contains non-serializable type and `safe_fallback=False`

**Examples:**

```python
>>> from lionherd_core.ln._json_dump import json_dumpb
>>> from pathlib import Path
>>> from decimal import Decimal

# Basic serialization
>>> json_dumpb({"key": "value"})
b'{"key":"value"}'

# With custom types (automatic handling)
>>> json_dumpb({"path": Path("/tmp"), "price": Decimal("19.99")})
b'{"path":"/tmp","price":"19.99"}'

# Decimal as float
>>> json_dumpb({"price": Decimal("19.99")}, decimal_as_float=True)
b'{"price":19.99}'

# Pretty-printing
>>> json_dumpb({"a": 1, "b": 2}, pretty=True)
b'{\n  "a": 1,\n  "b": 2\n}'

# Sorted keys
>>> json_dumpb({"z": 1, "a": 2}, sort_keys=True)
b'{"a":2,"z":1}'

# Safe fallback for logging
>>> class Custom: pass
>>> json_dumpb({"obj": Custom()}, safe_fallback=True)
b'{"obj":"<__main__.Custom object at 0x?>"}'

# NDJSON line
>>> json_dumpb({"id": 1}, append_newline=True)
b'{"id":1}\n'
```

**Notes:**

- **Prefer bytes**: Direct bytes output is fastest (no decode overhead)
- **Caching**: Default handlers cached via `_cached_default()` (LRU size 128)
- **passthrough_datetime**: If `True`, must set both `passthrough_datetime=True` in params AND in options
- **safe_fallback**: Only use for logging. Produces lossy output for unknown types.

**See Also:**

- `json_dumps()`: String variant (decodes bytes to str)
- `get_orjson_default()`: Default handler configuration
- `make_options()`: Option flag configuration

---

#### `json_dumps()`

Serialize to str by default (decode=True), or bytes if decode=False.

**Signature:**

```python
def json_dumps(
    obj: Any,
    /,
    *,
    decode: bool = True,
    **kwargs: Any,
) -> str | bytes:
```

**Parameters:**

**obj** : Any (positional-only)

Object to serialize.

**decode** : bool, default True

Whether to decode bytes to string.

- `True`: Returns UTF-8 decoded string
- `False`: Returns bytes (equivalent to `json_dumpb()`)

**kwargs** : Any

All other parameters passed to `json_dumpb()` (see `json_dumpb()` documentation).

**Returns:**

- str | bytes: UTF-8 JSON string (if decode=True) or bytes (if decode=False)

**Examples:**

```python
>>> from lionherd_core.ln._json_dump import json_dumps

# String output (default)
>>> json_dumps({"key": "value"})
'{"key":"value"}'

# Bytes output
>>> json_dumps({"key": "value"}, decode=False)
b'{"key":"value"}'

# All json_dumpb parameters work
>>> json_dumps({"a": 1}, pretty=True, sort_keys=True)
'{\n  "a": 1\n}'
```

**Notes:**

- **Convenience Wrapper**: Calls `json_dumpb()` then optionally decodes
- **Performance**: Adds decode overhead. Use `json_dumpb()` if bytes acceptable.
- **Compatibility**: Matches stdlib `json.dumps()` signature (returns str)

**See Also:**

- `json_dumpb()`: Bytes variant (faster)

---

### Streaming

#### `json_lines_iter()`

Stream an iterable as **NDJSON** (newline-delimited JSON) in **bytes**.

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
) -> Iterable[bytes]:
```

**Parameters:**

**it** : Iterable of Any

Iterable of objects to serialize (list, generator, etc.).

**deterministic_sets** : bool, default False
**decimal_as_float** : bool, default False
**enum_as_name** : bool, default False
**passthrough_datetime** : bool, default False
**safe_fallback** : bool, default False
**fallback_clip** : int, default 2048

Passed to `get_orjson_default()`. See `get_orjson_default()` documentation.

**naive_utc** : bool, default False
**utc_z** : bool, default False
**allow_non_str_keys** : bool, default False

Passed to `make_options()`. See `make_options()` documentation.

**default** : Callable[[Any], Any], optional

Custom default handler. If `None`, uses cached handler.

**options** : int, optional

Pre-composed option flags. If `None`, composes from individual flags.

**Returns:**

- Iterable[bytes]: Generator yielding JSON bytes with trailing newline per item

**Examples:**

```python
>>> from lionherd_core.ln._json_dump import json_lines_iter

# Stream list to NDJSON
>>> items = [{"id": 1}, {"id": 2}, {"id": 3}]
>>> for line in json_lines_iter(items):
...     print(line)
b'{"id":1}\n'
b'{"id":2}\n'
b'{"id":3}\n'

# Write to file
>>> with open("output.ndjson", "wb") as f:
...     for line in json_lines_iter(items):
...         f.write(line)

# Stream generator (memory-efficient)
>>> def generate_data():
...     for i in range(1000000):
...         yield {"id": i, "value": i * 2}
>>> for line in json_lines_iter(generate_data()):
...     process_line(line)  # Process without loading all in memory

# With custom types
>>> from pathlib import Path
>>> items = [{"path": Path(f"/tmp/{i}")} for i in range(3)]
>>> list(json_lines_iter(items))
[b'{"path":"/tmp/0"}\n', b'{"path":"/tmp/1"}\n', b'{"path":"/tmp/2"}\n']
```

**Usage Patterns:**

```python
# Pattern 1: Streaming API response
from lionherd_core.ln._json_dump import json_lines_iter
from starlette.responses import StreamingResponse

async def stream_large_dataset(query: str):
    """Stream database results as NDJSON."""
    results = db.query(query)  # Generator
    return StreamingResponse(
        json_lines_iter(results),
        media_type="application/x-ndjson"
    )

# Pattern 2: Memory-efficient file processing
def process_large_file(input_path: str, output_path: str):
    """Convert large dataset to NDJSON without loading all in memory."""
    def read_data():
        # Generator reading chunks
        with open(input_path) as f:
            for line in f:
                yield parse_line(line)

    with open(output_path, "wb") as out:
        for json_line in json_lines_iter(read_data()):
            out.write(json_line)

# Pattern 3: Log streaming
import logging

def stream_logs_as_json(log_records: Iterable[logging.LogRecord]):
    """Stream log records as NDJSON."""
    def format_record(record):
        return {
            "timestamp": record.created,
            "level": record.levelname,
            "message": record.getMessage(),
        }

    records = (format_record(r) for r in log_records)
    return json_lines_iter(records, safe_fallback=True)
```

**Notes:**

- **Newline Enforcement**: Always appends `\n` via `OPT_APPEND_NEWLINE` (NDJSON spec)
- **No Pretty-Print**: NDJSON is always compact (one object per line)
- **Memory Efficient**: Processes items one at a time (generator pattern)
- **Streaming**: Suitable for HTTP streaming responses, file writing, log tailing

**See Also:**

- [NDJSON Specification](http://ndjson.org/): Newline-delimited JSON format
- `json_dumpb()`: Single-object serialization

---

### Utility

#### `json_dict()`

Serialize object to JSON bytes then parse back to dict (orjson round-trip).

**Signature:**

```python
def json_dict(
    obj: Any,
    /,
    **kwargs: Any,
) -> dict:
```

**Parameters:**

**obj** : Any (positional-only)

Object to convert to dictionary via JSON round-trip.

**kwargs** : Any

Parameters passed to `json_dumpb()` (see `json_dumpb()` documentation).

**Returns:**

- dict: Python dictionary parsed from JSON

**Examples:**

```python
>>> from lionherd_core.ln._json_dump import json_dict
>>> from pathlib import Path
>>> from pydantic import BaseModel

# Convert object to JSON-compatible dict
>>> class User(BaseModel):
...     name: str
...     age: int
>>> user = User(name="Alice", age=30)
>>> json_dict(user)
{'name': 'Alice', 'age': 30}

# Path serialization
>>> json_dict({"path": Path("/tmp")})
{'path': '/tmp'}

# Handles custom types via default handler
>>> from decimal import Decimal
>>> json_dict({"price": Decimal("19.99")})
{'price': '19.99'}
```

**Notes:**

- **Round-Trip**: `orjson.loads(orjson.dumps(obj, default=default))`
- **Type Coercion**: Converts non-JSON types via default handler, then back to primitives
- **Use Case**: Normalize objects to JSON-compatible dicts for storage/transmission

**See Also:**

- `json_dumpb()`: Underlying serialization
- `orjson.loads()`: Deserialization

---

## Common Patterns

### API Response Serialization

```python
from lionherd_core.ln._json_dump import json_dumpb
from pathlib import Path
from decimal import Decimal

def serialize_api_response(data: dict) -> bytes:
    """Serialize API response with custom types."""
    return json_dumpb(
        data,
        decimal_as_float=True,  # Smaller output
        sort_keys=True,         # Deterministic
        utc_z=True,             # Standard UTC format
    )

# Usage
response = {
    "user_id": 123,
    "balance": Decimal("1234.56"),
    "data_dir": Path("/var/data"),
}
json_bytes = serialize_api_response(response)
```

### Safe Logging

```python
from lionherd_core.ln._json_dump import json_dumps
import logging

# Create reusable safe serializer
def log_json(data: dict, level: int = logging.INFO):
    """Log dict as JSON, never crash on serialization errors."""
    json_str = json_dumps(
        data,
        safe_fallback=True,   # Never raise
        fallback_clip=500,    # Short clips for logs
        sort_keys=True,       # Consistent log format
    )
    logging.log(level, json_str)

# Safe even with non-serializable objects
log_json({"event": "error", "exception": Exception("Failed")})
# {"event": "error", "exception": {"type": "Exception", "message": "Failed"}}
```

### NDJSON Streaming

```python
from lionherd_core.ln._json_dump import json_lines_iter

# Stream iterable as NDJSON
data = [{"id": 1}, {"id": 2}, {"id": 3}]

for json_line in json_lines_iter(data):
    print(json_line)  # b'{"id":1}\n', b'{"id":2}\n', ...

# Usage with FastAPI
from starlette.responses import StreamingResponse

@app.get("/export")
async def export_data():
    results = db.execute("SELECT * FROM large_table")
    return StreamingResponse(
        json_lines_iter(results),
        media_type="application/x-ndjson"
    )
```

### Deterministic Serialization

```python
from lionherd_core.ln._json_dump import json_dumpb

# Ensure consistent output regardless of input order
data = {"tags": {3, 1, 2}, "name": "test"}

json_output = json_dumpb(
    data,
    sort_keys=True,           # Alphabetical key order
    deterministic_sets=True,  # Sorted set elements
)

# Same data, different order → identical output
data2 = {"name": "test", "tags": {1, 2, 3}}
json_output2 = json_dumpb(data2, sort_keys=True, deterministic_sets=True)
assert json_output == json_output2  # ✓ Deterministic
```

<!-- TODO: Add link to content-based caching tutorial when available (issue #115) -->

### Custom Type Handlers

```python
from lionherd_core.ln._json_dump import get_orjson_default, json_dumpb
from datetime import datetime

# Map custom types to serialization functions
custom_default = get_orjson_default(
    additional={
        datetime: lambda dt: dt.timestamp(),  # datetime → Unix timestamp
    }
)

# Use custom serializer
data = {"created": datetime.now()}
result = json_dumpb(data, default=custom_default)
# datetime objects converted via custom handler
```

<!-- TODO: Add link to advanced type serialization tutorial when available (issue #115) -->

## Design Rationale

### Why orjson Backend?

orjson provides 2-5× faster serialization than stdlib json with additional benefits:

1. **Speed**: Written in Rust, compiles to C extensions
2. **Type Support**: Native datetime/UUID/dataclass support
3. **Correctness**: Handles edge cases (infinity, NaN) correctly
4. **Small Output**: No trailing whitespace, efficient encoding

### Why Separate dumpb/dumps?

`json_dumpb()` returns bytes to match orjson's native API:

1. **Performance**: Avoids decode overhead when bytes acceptable (HTTP responses, file writes)
2. **Explicitness**: Caller controls decode timing and error handling
3. **Compatibility**: `json_dumps()` wrapper provides familiar str-returning API

### Why Cached Default Handlers?

Default handler creation is relatively expensive (builds serializer dict, processes order list). Caching provides:

1. **Performance**: Amortizes setup cost across calls
2. **Convenience**: No need to manually cache handlers
3. **Safety**: LRU bound (128 entries) prevents memory leaks

### Why safe_fallback Mode?

Production logging must never crash:

1. **Resilience**: Unknown types produce `repr()` instead of raising
2. **Debugging**: Exception objects serialized with type + message
3. **Trade-off**: Lossy output acceptable for logs, unacceptable for data interchange

### Why Deterministic Sets?

Set ordering varies across runs (memory addresses), breaking:

1. **Testing**: Assertions fail on set-containing JSON
2. **Caching**: Content-based cache keys require stable output
3. **Signatures**: Content hashing requires deterministic serialization

Cost: Sorting overhead (~2-3× slower for large sets).

### Why Separate passthrough_datetime?

orjson handles datetime natively at C speed. Passthrough mode only needed for:

1. **Custom Formatting**: Non-ISO 8601 datetime strings
2. **Millisecond Precision**: Truncate microseconds
3. **Timezone Conversion**: Non-UTC output

Rare use case, so opt-in to avoid default handler overhead.

## See Also

- **Related Functions**:
  - [to_dict()](to_dict.md): Dictionary conversion utilities
  - [hash_dict()](hash.md): Content-based dict hashing
- **External Libraries**:
  - [orjson](https://github.com/ijl/orjson): Underlying serialization library
  - [json](https://docs.python.org/3/library/json.html): Stdlib JSON module

## Examples

```python
# Standard imports for ln.json_dump examples
from lionherd_core.ln import (
    json_dumps,
    json_dumpb,
    json_lines_iter,
    get_orjson_default,
    make_options
)
```

### Helper Functions for Examples

```python
# Mock database interface (simplified for demonstration)
class MockDB:
    """Mock database for example code."""
    def execute(self, query: str):
        """Mock query execution - yields sample records."""
        for i in range(10):
            yield {
                "id": i,
                "data": f"record_{i}",
                "created_at": "2025-11-09T14:30:00Z",
            }

    def query(self, query: str):
        """Alias for execute."""
        return self.execute(query)

# Helper functions used in examples
def parse_line(line: str) -> dict:
    """Parse a line of input data (mock implementation)."""
    import json
    return json.loads(line)

def process_record(record: dict) -> dict:
    """Transform database record for export."""
    return {
        "id": record["id"],
        "data": record["data"],
        "created": record["created_at"],
    }

# Create mock database instance for examples
db = MockDB()
```

### Example 1: Production API Serialization

```python
from pathlib import Path
from decimal import Decimal
from uuid import UUID
from datetime import datetime, UTC

# Configure default handler once
api_default = get_orjson_default(
    decimal_as_float=True,  # Smaller payloads
    enum_as_name=True,      # Human-readable enums
)

def serialize_response(data: dict) -> bytes:
    """Standard API response serialization."""
    return json_dumpb(
        data,
        default=api_default,
        sort_keys=True,  # Consistent output
        utc_z=True,      # Standard UTC format
    )

# Usage
response = {
    "user_id": UUID("550e8400-e29b-41d4-a716-446655440000"),
    "balance": Decimal("1234.56"),
    "data_dir": Path("/var/data"),
    "created_at": datetime(2025, 11, 9, 14, 30, tzinfo=UTC),
}
json_bytes = serialize_response(response)
# b'{"balance":1234.56,"created_at":"2025-11-09T14:30:00Z",...}'
```

### Example 2: Structured Logging

```python
import logging
from lionherd_core.ln._json_dump import json_dumps, get_orjson_default

# Configure safe logging serializer (never crashes on unknown types)
log_default = get_orjson_default(
    safe_fallback=True,   # Never raise TypeError
    fallback_clip=500,    # Keep log lines manageable
)

def setup_json_logging(logger_name: str):
    """Simple JSON logger setup."""
    logger = logging.getLogger(logger_name)

    def json_format(record: logging.LogRecord) -> str:
        data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": record.created,
        }
        return json_dumps(data, default=log_default, sort_keys=True)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# Usage
logger = setup_json_logging("myapp")
logger.info("User action")  # Safely logs even with non-serializable objects
```

<!-- TODO: Add link to JSON logging integration tutorial when available (issue #115) -->

### Example 3: Large Dataset Streaming

```python
from lionherd_core.ln._json_dump import json_lines_iter
from pathlib import Path

# Stream large dataset to NDJSON file
def export_results(output_path: Path):
    """Stream database results to NDJSON file."""
    results = db.execute("SELECT * FROM large_table")

    with output_path.open("wb") as f:
        for json_line in json_lines_iter(
            (process_record(r) for r in results),
            safe_fallback=True  # Tolerant of data anomalies
        ):
            f.write(json_line)

# Usage
export_results(Path("export.ndjson"))
```

<!-- TODO: Add link to large-scale data export tutorial when available (issue #115) -->

### Example 4: Deterministic Hashing

```python
from lionherd_core.ln._json_dump import json_dumpb
import hashlib

# Generate content-based hash (deterministic across runs)
def hash_dict(data: dict) -> str:
    """Hash dict content regardless of key order or set order."""
    json_bytes = json_dumpb(
        data,
        sort_keys=True,
        deterministic_sets=True,
    )
    return hashlib.sha256(json_bytes).hexdigest()

# Usage - same content = same hash
params1 = {"model": "gpt-4", "tags": {1, 2, 3}}
params2 = {"tags": {3, 1, 2}, "model": "gpt-4"}  # Different order
hash1 = hash_dict(params1)
hash2 = hash_dict(params2)
assert hash1 == hash2  # ✓ Content-based, order-independent
```

<!-- TODO: Add link to content-based caching tutorial when available (issue #115) -->

### Example 5: Custom Type Integration

```python
from lionherd_core.ln._json_dump import get_orjson_default, json_dumpb
from datetime import datetime

# Define custom serializers for domain types
custom_default = get_orjson_default(
    additional={
        # Custom type → serialization function
        dict: lambda obj: {
            "id": obj.get("id"),
            "data": obj.get("data"),
        }
    },
    enum_as_name=True,  # Enums serialize as names, not values
    utc_z=True,         # UTC datetimes use "Z" suffix
)

# Usage
event = {
    "id": 123,
    "data": {"name": "Conference", "timestamp": datetime.now()},
}

json_output = json_dumpb(event, default=custom_default)
# Handles custom types via additional serializers
```

<!-- TODO: Add link to advanced type serialization tutorial when available (issue #115) -->

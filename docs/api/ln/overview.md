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

## Quick Reference

| Function | Category | Purpose |
|----------|----------|---------|
| [`alcall`](#alcall) | Async | Apply function to each list element asynchronously with retry/concurrency |
| [`bcall`](#bcall) | Async | Process input in batches using alcall |
| [`AlcallParams`](#alcallparams) | Async | Reusable parameter configurations for alcall |
| [`BcallParams`](#bcallparams) | Async | Reusable parameter configurations for bcall |
| [`fuzzy_match_keys`](#fuzzy_match_keys) | Fuzzy | Validate and correct dict keys using fuzzy matching |
| [`fuzzy_validate_pydantic`](#fuzzy_validate_pydantic) | Fuzzy | Parse text/dict into Pydantic model with fuzzy parsing |
| [`fuzzy_validate_mapping`](#fuzzy_validate_mapping) | Fuzzy | Validate any input into dict with expected keys |
| [`FuzzyMatchKeysParams`](#fuzzymatchkeysparams) | Fuzzy | Reusable fuzzy matching configurations |
| [`hash_dict`](#hash_dict) | Hashing | Generate stable hash for any data structure |
| [`json_dumps`](#json_dumps) | JSON | Serialize to JSON string with custom type support |
| [`json_dumpb`](#json_dumpb) | JSON | Serialize to bytes (fast path) |
| [`json_lines_iter`](#json_lines_iter) | JSON | Stream iterable as NDJSON |
| [`json_dict`](#json_dict) | JSON | Round-trip serialize to dict |
| [`get_orjson_default`](#get_orjson_default) | JSON | Build custom serializer for orjson |
| [`make_options`](#make_options) | JSON | Compose orjson option bit flags |
| [`to_list`](#to_list) | List | Convert input to list with transformations |
| [`lcall`](#lcall) | List | Apply function to each element synchronously |
| [`to_dict`](#to_dict) | Dict | Convert various input types to dictionary |
| [`now_utc`](#now_utc) | Utility | Get current UTC datetime |
| [`acreate_path`](#acreate_path) | Utility | Generate file path asynchronously |
| [`get_bins`](#get_bins) | Utility | Organize indices into bins by cumulative length |
| [`import_module`](#import_module) | Utility | Import module by path dynamically |
| [`is_import_installed`](#is_import_installed) | Utility | Check if package is installed |

## Async Call Utilities

### `alcall()`

Apply function to each list element asynchronously with retry and concurrency control.

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
) -> list[T | BaseException]: ...
```

**Returns**: `list[T | BaseException]` - Results in input order

**Raises**: `ValueError`, `TimeoutError`, `ExceptionGroup`

**See**: [async_call.md](async_call.md) for detailed API reference

### `bcall()`

Process input in batches using alcall. Yields results batch by batch.

```python
async def bcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    batch_size: int,
    **kwargs: Any,
) -> AsyncGenerator[list[T | BaseException], None]: ...
```

**Yields**: `list[T | BaseException]` - Results for each batch

**See**: [async_call.md](async_call.md) for detailed API reference

### `AlcallParams`

Reusable parameter dataclass for alcall with callable interface.

```python
@dataclass(slots=True, init=False, frozen=True)
class AlcallParams(Params):
    async def __call__(
        self,
        input_: list[Any],
        func: Callable[..., T],
        **kw: Any,
    ) -> list[T]: ...
```

**See**: [async_call.md](async_call.md) for detailed API reference

### `BcallParams`

Reusable parameter dataclass for bcall (extends AlcallParams).

```python
@dataclass(slots=True, init=False, frozen=True)
class BcallParams(AlcallParams):
    batch_size: int
```

**See**: [async_call.md](async_call.md) for detailed API reference

## Fuzzy Matching and Validation

### `fuzzy_match_keys()`

Validate and correct dict keys using fuzzy string matching.

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

**Returns**: `dict[str, Any]` - Dictionary with corrected keys

**Raises**: `TypeError`, `ValueError`

**See**: [fuzzy_match.md](fuzzy_match.md) and [fuzzy_validate.md](fuzzy_validate.md) for detailed API reference

### `fuzzy_validate_pydantic()`

Validate and parse text/dict into Pydantic model with fuzzy parsing.

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

**Returns**: `BaseModel` - Validated Pydantic model instance

**Raises**: `ValidationError`, `TypeError`

**See**: [fuzzy_match.md](fuzzy_match.md) and [fuzzy_validate.md](fuzzy_validate.md) for detailed API reference

### `fuzzy_validate_mapping()`

Validate any input into dict with expected keys and fuzzy matching.

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

**Returns**: `dict[str, Any]` - Validated dictionary with corrected keys

**Raises**: `TypeError`, `ValueError`

**See**: [fuzzy_match.md](fuzzy_match.md) and [fuzzy_validate.md](fuzzy_validate.md) for detailed API reference

### `FuzzyMatchKeysParams`

Reusable parameter dataclass for fuzzy_match_keys with callable interface.

```python
@dataclass(slots=True, init=False, frozen=True)
class FuzzyMatchKeysParams(Params):
    def __call__(self, d_: dict[str, Any], keys: KeysLike) -> dict[str, Any]: ...
```

**See**: [fuzzy_match.md](fuzzy_match.md) and [fuzzy_validate.md](fuzzy_validate.md) for detailed API reference

## Hashing

### `hash_dict()`

Generate stable hash for any data structure including dicts, lists, and Pydantic models.

```python
def hash_dict(data: Any, strict: bool = False) -> int: ...
```

**Returns**: `int` - Integer hash value (stable across equivalent structures)

**Raises**: `TypeError`

**See**: [to_dict.md](to_dict.md), [to_list.md](to_list.md), and [hash.md](hash.md) for detailed API reference

## JSON Utilities

### `json_dumps()`

Serialize to JSON string with high performance and custom type support.

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

**Returns**: `str | bytes` - JSON representation

**See**: [json_dump.md](json_dump.md) for detailed API reference

### `json_dumpb()`

Serialize to bytes (fast path for network/file I/O).

```python
def json_dumpb(
    obj: Any,
    *,
    pretty: bool = False,
    sort_keys: bool = False,
    # ... same parameters as json_dumps except no decode
) -> bytes: ...
```

**Returns**: `bytes` - JSON representation as bytes

**See**: [json_dump.md](json_dump.md) for detailed API reference

### `json_lines_iter()`

Stream an iterable as NDJSON (newline-delimited JSON) in bytes.

```python
def json_lines_iter(
    it: Iterable[Any],
    *,
    deterministic_sets: bool = False,
    decimal_as_float: bool = False,
    enum_as_name: bool = False,
    passthrough_datetime: bool = False,
    safe_fallback: bool = False,
    fallback_clip: int = 2048,
    naive_utc: bool = False,
    utc_z: bool = False,
    allow_non_str_keys: bool = False,
    default: Callable[[Any], Any] | None = None,
    options: int | None = None,
) -> Iterable[bytes]: ...
```

**Yields**: `bytes` - JSON line with trailing newline

**See**: [json_dump.md](json_dump.md) for detailed API reference

### `json_dict()`

Round-trip serialize to dict (useful for type coercion).

```python
def json_dict(obj: Any, /, **kwargs: Any) -> dict: ...
```

**Returns**: `dict` - Deserialized dictionary

**See**: [json_dump.md](json_dump.md) for detailed API reference

### `get_orjson_default()`

Build a fast, extensible default= callable for orjson.dumps.

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

**Returns**: `Callable[[Any], Any]` - Serializer function for orjson

**See**: [json_dump.md](json_dump.md) for detailed API reference

### `make_options()`

Compose orjson option bit flags succinctly.

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

**Returns**: `int` - Bitwise OR of orjson option flags

**See**: [json_dump.md](json_dump.md) for detailed API reference

## List Processing

### `to_list()`

Convert input to list with optional transformations.

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

**Returns**: `list` - Processed list

**Raises**: `ValueError`

**See**: [to_dict.md](to_dict.md), [to_list.md](to_list.md), and [hash.md](hash.md) for detailed API reference

### `lcall()`

Apply function to each element synchronously with optional input/output processing.

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

**Returns**: `list[R]` - Results

**Raises**: `ValueError`, `TypeError`

**See**: [to_dict.md](to_dict.md), [to_list.md](to_list.md), and [hash.md](hash.md) for detailed API reference

## Dictionary Conversion

### `to_dict()`

Convert various input types to dictionary with optional recursive processing.

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

**Returns**: `dict[str | int, Any]` - Dictionary representation

**See**: [to_dict.md](to_dict.md), [to_list.md](to_list.md), and [hash.md](hash.md) for detailed API reference

## General Utilities

### `now_utc()`

Get current UTC datetime.

```python
def now_utc() -> datetime: ...
```

**Returns**: `datetime` - Current UTC datetime with timezone info

**See**: [utils.md](utils.md) for detailed API reference

### `acreate_path()`

Generate file path asynchronously with optional timeout.

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

**Returns**: `AsyncPath` - Created/validated file path

**Raises**: `ValueError`, `FileExistsError`, `TimeoutError`

**See**: [utils.md](utils.md) for detailed API reference

### `get_bins()`

Organize indices into bins by cumulative length.

```python
def get_bins(input_: list[str], upper: int) -> list[list[int]]: ...
```

**Returns**: `list[list[int]]` - List of bins (each bin is list of indices)

**See**: [utils.md](utils.md) for detailed API reference

### `import_module()`

Import module by path with optional name extraction.

```python
def import_module(
    package_name: str,
    module_name: str | None = None,
    import_name: str | list | None = None,
) -> Any: ...
```

**Returns**: `Any` - Imported module or name(s)

**Raises**: `ImportError`

**See**: [utils.md](utils.md) for detailed API reference

### `is_import_installed()`

Check if package is installed.

```python
def is_import_installed(package_name: str) -> bool: ...
```

**Returns**: `bool` - True if package is installed

**See**: [utils.md](utils.md) for detailed API reference

## Discussion

**Design & Performance:**

- [Design Decisions](discussion/ln_design_decisions.md): Rationale behind key design choices
- [Performance](discussion/ln_performance.md): Benchmarks and optimization strategies

## See Also

- **Related Modules**:
  - [base](../base/element.md): Element class with serialization
  - types: Type definitions and utilities (documentation pending)
  - libs: Low-level library functions (documentation pending)

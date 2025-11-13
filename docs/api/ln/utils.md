# Utilities (ln._utils)

> Core utility functions for datetime, path creation, import management, and binning operations

## Overview

The `ln._utils` module provides essential utility functions used throughout lionherd-core for common operations that don't belong to specific classes. These functions handle timezone-aware datetime operations, async path creation with timeout support, module importing with validation, and list partitioning.

**Key Functions:**

- **Datetime**: `now_utc()` for consistent UTC timestamp generation
- **Async I/O**: `acreate_path()` for timeout-aware async path creation
- **Import Management**: `import_module()`, `is_import_installed()` for dynamic imports
- **Data Partitioning**: `get_bins()` for length-based list binning

**Common Use Cases:**

- Generating UTC timestamps for Element creation
- Creating file paths asynchronously with directory/filename validation
- Dynamically importing modules at runtime
- Checking optional dependency availability
- Partitioning text/data by cumulative length constraints

## Functions

### Datetime Operations

#### `now_utc()`

Get current UTC datetime with timezone awareness.

**Signature:**

```python
def now_utc() -> datetime: ...
```

**Returns:**

- datetime: Current UTC time with timezone info (timezone-aware)

**Examples:**

```python
>>> from lionherd_core.ln._utils import now_utc
>>> timestamp = now_utc()
>>> timestamp
datetime.datetime(2025, 11, 9, 14, 30, 45, 123456, tzinfo=datetime.UTC)

>>> timestamp.tzinfo
datetime.UTC

# Use for Element creation timestamps
>>> from lionherd_core import Element
>>> elem = Element(created_at=now_utc())
```

**Notes:**

Uses `datetime.now(UTC)` to ensure timezone-aware datetime objects. All timestamps in lionherd-core should use UTC for consistency across distributed systems.

**See Also:**

- [Element.created_at](../base/element.md#attributes): Default timestamp generation uses this function

---

### Async Path Operations

#### `acreate_path()`

Generate file path asynchronously with validation, timestamps, and timeout support.

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

**directory** : Path or AsyncPath or str

Base directory path for file creation. Automatically converted to `AsyncPath`.

- Supports nested subdirectories via `filename` parameter (e.g., `filename="subdir/file.txt"`)
- Created recursively if doesn't exist (when `dir_exist_ok=True`)

**filename** : str

Target filename with optional subdirectory path using forward slashes.

- Can include subdirectories: `"logs/2025/output.txt"` splits into directory extension
- Extension auto-detected if present (e.g., `"file.txt"` → name="file", ext=".txt")
- Backslashes not allowed (raises `ValueError`)

**extension** : str, optional

File extension to append if `filename` doesn't have one.

- Leading dot optional: `"txt"` and `".txt"` both work
- Ignored if `filename` already has extension
- Default: `None` (no extension added)

**timestamp** : bool, default False

Add timestamp to filename for uniqueness.

- Uses `timestamp_format` or default `"%Y%m%d%H%M%S"` format
- Position controlled by `time_prefix` parameter
- Example: `"output.txt"` → `"output_20251109143045.txt"` or `"20251109143045_output.txt"`

**dir_exist_ok** : bool, default True

Allow existing directories during creation.

- `True`: Silently succeeds if directory exists
- `False`: Raises `FileExistsError` if directory exists

**file_exist_ok** : bool, default False

Allow existing files at target path.

- `True`: Returns path even if file exists
- `False`: Raises `FileExistsError` if file exists
- Note: Does **not** create the file, only validates path availability

**time_prefix** : bool, default False

Put timestamp before filename instead of after.

- `False`: `"file_20251109143045.txt"` (timestamp suffix)
- `True`: `"20251109143045_file.txt"` (timestamp prefix)
- Only applies when `timestamp=True`

**timestamp_format** : str, optional

Custom `strftime` format for timestamp.

- Default: `"%Y%m%d%H%M%S"` (e.g., `"20251109143045"`)
- Examples: `"%Y-%m-%d"` → `"2025-11-09"`, `"%Y%m%d_%H%M"` → `"20251109_1430"`
- See [strftime documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)

**random_hash_digits** : int, default 0

Add random hash suffix for uniqueness.

- `0`: No hash added
- `>0`: Appends first N digits of UUID4 hex (max 32)
- Example: `random_hash_digits=8` → `"file-a1b2c3d4.txt"`
- Combined with timestamp: `"file_20251109143045-a1b2c3d4.txt"`

**timeout** : float, optional

Maximum time in seconds for async I/O operations.

- `None`: No timeout (waits indefinitely)
- `>0`: Raises `TimeoutError` if operations exceed timeout
- Covers directory creation and file existence checks
- Default: `None`

**Returns:**

- AsyncPath: Validated path to target file (directory created, file availability checked)

**Raises:**

- ValueError: If `filename` contains backslash `\` (use forward slash `/` for subdirectories)
- FileExistsError: If file exists and `file_exist_ok=False`, or directory exists and `dir_exist_ok=False`
- TimeoutError: If `timeout` exceeded during I/O operations

**Examples:**

```python
>>> from lionherd_core.ln._utils import acreate_path
>>> from anyio import Path as AsyncPath

# Basic usage
>>> path = await acreate_path("/tmp", "output.txt")
>>> path
AsyncPath('/tmp/output.txt')

# Subdirectory in filename
>>> path = await acreate_path("/logs", "2025/11/output.txt")
>>> path
AsyncPath('/logs/2025/11/output.txt')

# Extension auto-detection
>>> path = await acreate_path("/data", "results")  # No extension
>>> path
AsyncPath('/data/results')

>>> path = await acreate_path("/data", "results", extension="json")
>>> path
AsyncPath('/data/results.json')

# Timestamp suffix
>>> path = await acreate_path(
...     "/logs",
...     "app.log",
...     timestamp=True,
...     timestamp_format="%Y%m%d_%H%M%S"
... )
>>> path
AsyncPath('/logs/app_20251109_143045.log')

# Timestamp prefix
>>> path = await acreate_path(
...     "/logs",
...     "app.log",
...     timestamp=True,
...     time_prefix=True
... )
>>> path
AsyncPath('/logs/20251109143045_app.log')

# Random hash for uniqueness
>>> path = await acreate_path(
...     "/tmp",
...     "temp.txt",
...     random_hash_digits=8
... )
>>> path
AsyncPath('/tmp/temp-a1b2c3d4.txt')

# Combined timestamp + hash
>>> path = await acreate_path(
...     "/outputs",
...     "result.json",
...     timestamp=True,
...     random_hash_digits=6
... )
>>> path
AsyncPath('/outputs/result_20251109143045-a1b2c3.json')

# Timeout protection
>>> path = await acreate_path(
...     "/slow-mount",
...     "file.txt",
...     timeout=5.0  # Fail if I/O takes >5s
... )
# TimeoutError: acreate_path timed out after 5.0s

# File existence validation
>>> path = await acreate_path("/tmp", "existing.txt", file_exist_ok=False)
# FileExistsError: File /tmp/existing.txt already exists...
```

**Usage Patterns:**

```python
# Pattern 1: Timestamped log files
async def create_log_file(log_dir: str, name: str) -> AsyncPath:
    return await acreate_path(
        log_dir,
        name,
        extension="log",
        timestamp=True,
        timestamp_format="%Y%m%d_%H%M%S",
        file_exist_ok=True  # Allow overwrite
    )

# Pattern 2: Unique temporary files
async def create_temp_file(temp_dir: str) -> AsyncPath:
    return await acreate_path(
        temp_dir,
        "temp",
        extension="tmp",
        random_hash_digits=12,  # High uniqueness
        file_exist_ok=False  # Must be new
    )

# Pattern 3: Nested output directories
async def create_output_path(base: str, category: str, name: str) -> AsyncPath:
    return await acreate_path(
        base,
        f"{category}/{name}",  # Subdirectory in filename
        extension="json",
        dir_exist_ok=True  # Allow nested dirs
    )

# Pattern 4: Timeout-protected slow storage
async def create_network_path(mount: str, filename: str) -> AsyncPath:
    try:
        return await acreate_path(
            mount,
            filename,
            timeout=10.0  # Fail fast on network issues
        )
    except TimeoutError:
        # Fallback to local storage
        return await acreate_path("/tmp", filename)
```

**Notes:**

- **Async I/O**: Uses `anyio.Path` for async directory creation and file checks
- **Path Creation**: Creates parent directories recursively (like `mkdir -p`)
- **No File Creation**: Function only validates path and creates directories, does **not** create the file itself
- **Timeout Scope**: `timeout` parameter uses `move_on_after` from `lionherd_core.libs.concurrency`
- **Subdirectory Handling**: Forward slashes in `filename` split into directory components (e.g., `"a/b/c.txt"` → directory: `base/a/b`, filename: `c.txt`)

**See Also:**

- `anyio.Path`: Async path operations
- `lionherd_core.libs.concurrency.move_on_after`: Timeout implementation

---

### Data Partitioning

#### `get_bins()`

Organize list indices into bins by cumulative length constraint.

**Signature:**

```python
def get_bins(input_: list[str], upper: int) -> list[list[int]]: ...
```

**Parameters:**

**input_** : list of str

List of strings to partition into bins.

- Length measured per string: `len(item)`
- Order preserved in output bins

**upper** : int

Maximum cumulative length per bin.

- Bin closed when `current_length + next_item_length >= upper`
- Individual items longer than `upper` placed in own bin

**Returns:**

- list of list of int: List of bins, where each bin contains indices into `input_`

**Examples:**

```python
>>> from lionherd_core.ln._utils import get_bins

# Basic partitioning
>>> items = ["short", "tiny", "medium length", "x"]
>>> bins = get_bins(items, upper=15)
>>> bins
[[0, 1], [2], [3]]
# Bin 0: "short" (5) + "tiny" (4) = 9 < 15 ✓
# Bin 1: "medium length" (13) alone (adding would exceed 15)
# Bin 2: "x" (1) alone

# Access items by bin
>>> for bin_indices in bins:
...     bin_items = [items[i] for i in bin_indices]
...     print(bin_items)
['short', 'tiny']
['medium length']
['x']

# Item exceeds upper limit (gets own bin)
>>> items = ["short", "this is a very long string that exceeds limit", "end"]
>>> bins = get_bins(items, upper=20)
>>> bins
[[0], [1], [2]]
# Each item in separate bin (item[1] is 46 chars > 20)

# Empty input
>>> get_bins([], upper=100)
[]

# Single item
>>> get_bins(["hello"], upper=10)
[[0]]
```

**Usage Patterns:**

```python
# Pattern 1: Batch API requests by token limit
def batch_by_tokens(messages: list[str], max_tokens: int) -> list[list[str]]:
    """Partition messages into batches under token limit."""
    bins = get_bins(messages, upper=max_tokens)
    return [[messages[i] for i in bin_indices] for bin_indices in bins]

# Example usage
messages = ["Message 1", "Message 2 is longer", "Msg 3", "Message 4"]
batches = batch_by_tokens(messages, max_tokens=25)
# batches = [['Message 1', 'Msg 3'], ['Message 2 is longer'], ['Message 4']]

# Pattern 2: Split text for processing
def split_paragraphs(paragraphs: list[str], chunk_size: int) -> list[list[int]]:
    """Group paragraphs into chunks under size limit."""
    return get_bins(paragraphs, upper=chunk_size)

# Pattern 3: Load balancing by content size
def distribute_tasks(tasks: list[str], max_size: int) -> list[list[str]]:
    """Distribute tasks across workers by cumulative size."""
    bins = get_bins(tasks, upper=max_size)
    return [[tasks[i] for i in bin_indices] for bin_indices in bins]
```

**Notes:**

- **Greedy Algorithm**: Uses greedy bin-packing (first-fit strategy)
- **Order Preserved**: Output bins maintain input order
- **Index-Based**: Returns indices, not values, for memory efficiency with large items
- **No Splitting**: Individual items never split across bins (atomic units)
- **Length Metric**: Uses `len()` for length measurement (works for strings, lists, bytes)

**Algorithm Behavior:**

```python
# Pseudocode
current_bin = []
current_length = 0

for idx, item in enumerate(input_):
    if current_length + len(item) < upper:
        # Add to current bin
        current_bin.append(idx)
        current_length += len(item)
    else:
        # Start new bin
        bins.append(current_bin)
        current_bin = [idx]
        current_length = len(item)

# Append final bin if non-empty
if current_bin:
    bins.append(current_bin)
```

---

### Import Management

#### `import_module()`

Dynamically import module or specific attributes from module path.

**Signature:**

```python
def import_module(
    package_name: str,
    module_name: str | None = None,
    import_name: str | list | None = None,
) -> Any: ...
```

**Parameters:**

**package_name** : str

Top-level package name (e.g., `"lionherd_core"`, `"numpy"`).

- Used as base for module path construction
- If `module_name` is None, imports this package directly

**module_name** : str, optional

Submodule path within package (e.g., `"base.element"`).

- Joined with `package_name` via dot: `f"{package_name}.{module_name}"`
- Default: `None` (import package only)

**import_name** : str or list of str, optional

Specific attribute(s) to import from module.

- `None`: Import entire module
- `str`: Import single attribute (e.g., `"Element"`)
- `list`: Import multiple attributes (e.g., `["Element", "Node"]`)
- Default: `None`

**Returns:**

- Any:
  - If `import_name=None`: Module object
  - If `import_name=str`: Single imported attribute
  - If `import_name=list`: List of imported attributes (same order as input)

**Raises:**

- ImportError: If module/attribute not found, with detailed error message

**Examples:**

```python
>>> from lionherd_core.ln._utils import import_module

# Import entire package
>>> module = import_module("lionherd_core")
>>> module
<module 'lionherd_core' from '...'>

# Import submodule
>>> element_module = import_module("lionherd_core", "base.element")
>>> element_module
<module 'lionherd_core.base.element' from '...'>

# Import single class
>>> Element = import_module("lionherd_core", "base.element", "Element")
>>> Element
<class 'lionherd_core.base.element.Element'>

# Import multiple classes
>>> classes = import_module(
...     "lionherd_core",
...     "base",
...     ["Element", "Node", "Event"]
... )
>>> classes
[<class 'Element'>, <class 'Node'>, <class 'Event'>]

# Import from top-level package
>>> datetime = import_module("datetime", import_name="datetime")
>>> datetime
<class 'datetime.datetime'>

# Error handling
>>> import_module("nonexistent_package")
# ImportError: Failed to import module nonexistent_package: No module named 'nonexistent_package'

>>> import_module("lionherd_core", "base.element", "NonExistent")
# ImportError: Failed to import module lionherd_core.base.element: ...
```

**Usage Patterns:**

```python
from typing import Callable

# Pattern 1: Optional dependency with fallback
def get_parser() -> Callable:
    """Get parser with optional orjson fallback."""
    try:
        orjson = import_module("orjson")
        return orjson.loads
    except ImportError:
        json = import_module("json")
        return json.loads

# Pattern 2: Plugin system
def load_plugin(plugin_path: str, class_name: str) -> type:
    """Load plugin class dynamically."""
    package, module = plugin_path.split(":", 1)
    return import_module(package, module, class_name)

# Usage
Plugin = load_plugin("myplugins.custom:handlers", "CustomHandler")

# Pattern 3: Lazy imports
class LazyLoader:
    def __init__(self, package: str, module: str, name: str):
        self._package = package
        self._module = module
        self._name = name
        self._obj = None

    def __call__(self, *args, **kwargs):
        if self._obj is None:
            self._obj = import_module(self._package, self._module, self._name)
        return self._obj(*args, **kwargs)

# Usage (delays import until first call)
create_element = LazyLoader("lionherd_core", "base.element", "Element")
```

**Notes:**

- **Dynamic Import**: Uses `__import__()` for runtime module loading
- **Error Context**: Wraps `ImportError` with full module path for debugging
- **Fromlist Handling**: Uses `fromlist` parameter for attribute imports
- **Return Type**: Returns single object for `str` import_name, list for `list` import_name

**See Also:**

- `is_import_installed()`: Check package availability before importing
- `importlib`: Standard library import utilities

---

#### `is_import_installed()`

Check if package is installed and importable.

**Signature:**

```python
def is_import_installed(package_name: str) -> bool: ...
```

**Parameters:**

**package_name** : str

Package name to check (e.g., `"numpy"`, `"lionherd_core"`).

- Must be top-level package name (not submodules)
- Case-sensitive

**Returns:**

- bool: `True` if package is installed and importable, `False` otherwise

**Examples:**

```python
>>> from lionherd_core.ln._utils import is_import_installed

# Check standard library
>>> is_import_installed("datetime")
True

# Check installed package
>>> is_import_installed("lionherd_core")
True

# Check missing package
>>> is_import_installed("nonexistent_package")
False

# Use for optional dependencies
>>> if is_import_installed("orjson"):
...     import orjson
...     loads = orjson.loads
... else:
...     import json
...     loads = json.loads
```

**Usage Patterns:**

```python
# Pattern 1: Optional dependency guard
def get_fast_parser() -> any:
    """Get fastest available JSON parser."""
    if is_import_installed("orjson"):
        from lionherd_core.ln._utils import import_module
        return import_module("orjson")
    elif is_import_installed("ujson"):
        from lionherd_core.ln._utils import import_module
        return import_module("ujson")
    else:
        import json
        return json

# Pattern 2: Feature flags
FEATURES = {
    "async": is_import_installed("anyio"),
    "fast_json": is_import_installed("orjson"),
    "numpy_support": is_import_installed("numpy"),
}

def check_feature(name: str) -> bool:
    """Check if feature is available."""
    return FEATURES.get(name, False)

# Pattern 3: Dependency validation
def validate_requirements(packages: list[str]) -> dict[str, bool]:
    """Check which required packages are installed."""
    return {pkg: is_import_installed(pkg) for pkg in packages}

# Usage
requirements = validate_requirements(["numpy", "pandas", "scipy"])
missing = [pkg for pkg, installed in requirements.items() if not installed]
if missing:
    raise ImportError(f"Missing required packages: {missing}")
```

**Notes:**

- **Fast Check**: Uses `importlib.util.find_spec()` (doesn't actually import)
- **No Side Effects**: Doesn't load package into memory
- **Top-Level Only**: Checks package availability, not submodules
- **Import vs Install**: Returns `False` for both "not installed" and "not importable" cases

**See Also:**

- `import_module()`: Dynamic import after availability check
- `importlib.util.find_spec()`: Underlying implementation

---

## Common Patterns

### Timestamped File Creation

```python
# noqa:validation
from lionherd_core.ln._utils import acreate_path, now_utc

async def create_log(log_dir: str, event: str) -> AsyncPath:
    """Create timestamped log file for event."""
    return await acreate_path(
        log_dir,
        f"{event}.log",
        timestamp=True,
        timestamp_format="%Y%m%d_%H%M%S",
        file_exist_ok=True
    )

# Usage
log_path = await create_log("/var/logs", "api_error")
# /var/logs/api_error_20251109_143045.log
```

### Conditional Imports

```python
from lionherd_core.ln._utils import is_import_installed, import_module

# Pattern: Fast path with fallback
if is_import_installed("orjson"):
    json_lib = import_module("orjson")
else:
    json_lib = import_module("json")

loads = json_lib.loads
dumps = json_lib.dumps
```

### Batching by Size

```python
from lionherd_core.ln._utils import get_bins

def batch_requests(items: list[str], max_size: int) -> list[list[str]]:
    """Batch items into requests under size limit."""
    bins = get_bins(items, upper=max_size)
    return [[items[i] for i in bin_indices] for bin_indices in bins]

# Process batches
messages = ["msg1", "longer message 2", "msg3", "message 4"]
for batch in batch_requests(messages, max_size=20):
    await send_batch(batch)
```

### Plugin Loading with Validation

```python
from lionherd_core.ln._utils import is_import_installed, import_module

def load_optional_plugin(plugin_spec: str):
    """Load plugin if available, return None otherwise."""
    package, rest = plugin_spec.split(":", 1)

    if not is_import_installed(package):
        return None

    module_path, class_name = rest.rsplit(".", 1)
    return import_module(package, module_path, class_name)

# Usage
Handler = load_optional_plugin("myplugins.handlers:CustomHandler")
if Handler:
    handler = Handler()
else:
    handler = DefaultHandler()
```

## Design Rationale

### Why Separate Utility Module?

The `_utils` module collects frequently-needed operations that:

1. **Don't Belong to Classes**: Generic operations used across multiple classes
2. **Avoid Circular Imports**: Low-level functions that classes depend on
3. **Single Responsibility**: Each function does one thing well
4. **Reusable**: Used throughout lionherd-core and by external code

### Why AsyncPath for acreate_path?

`acreate_path` uses `anyio.Path` (AsyncPath) for true async I/O:

1. **Non-Blocking**: Directory creation and file checks don't block event loop
2. **Timeout Support**: Async context enables timeout via `move_on_after`
3. **Concurrent Safe**: Multiple coroutines can create paths concurrently
4. **Network Mounts**: Handles slow storage (NFS, cloud mounts) without blocking

### Why Index-Based Binning?

`get_bins()` returns indices rather than values because:

1. **Memory Efficiency**: No copying of potentially large items
2. **Flexibility**: Caller decides whether to extract values or process indices
3. **Multi-Use**: Same bin indices can index multiple parallel lists
4. **Preserves Order**: Index lists maintain original ordering

### Why Import Utilities?

`import_module()` and `is_import_installed()` provide controlled dynamic imports:

1. **Plugin Systems**: Load extensions without hardcoding imports
2. **Optional Dependencies**: Check availability before importing
3. **Error Handling**: Wrapped imports provide better error messages
4. **Lazy Loading**: Defer imports until actually needed

## See Also

- **Related Modules**:
  - [Element](../base/element.md): Uses `now_utc()` for timestamp generation
- **External Libraries**:
  - [anyio](https://anyio.readthedocs.io/): Async path operations
  - [importlib](https://docs.python.org/3/library/importlib.html): Import utilities

## Examples

```python
# Standard imports for ln.utils examples
from lionherd_core.ln import (
    now_utc,
    acreate_path,
    get_bins,
    import_module,
    is_import_installed
)
```

### Example 1: Complete Log File Workflow

```python
# noqa:validation
from anyio import Path as AsyncPath

async def write_log(log_dir: str, message: str):
    """Create timestamped log file and write message."""
    # Create unique log path
    log_path = await acreate_path(
        log_dir,
        "app.log",
        timestamp=True,
        random_hash_digits=6,  # Avoid collisions
        file_exist_ok=False     # Must be new file
    )

    # Write log entry
    timestamp = now_utc().isoformat()
    entry = f"[{timestamp}] {message}\n"

    # Use AsyncPath for async write
    await log_path.write_text(entry)

    return log_path

# Usage
log_path = await write_log("/var/logs", "Application started")
# /var/logs/app_20251109143045-a1b2c3.log
```

### Example 2: Conditional Feature Import

```python
from lionherd_core.ln._utils import is_import_installed, import_module

class DataProcessor:
    """Processor with optional numpy acceleration."""

    def __init__(self):
        self.has_numpy = is_import_installed("numpy")
        if self.has_numpy:
            self.np = import_module("numpy")

    def process(self, data: list[float]) -> float:
        """Calculate mean with numpy if available."""
        if self.has_numpy:
            return float(self.np.mean(data))
        else:
            return sum(data) / len(data)

# Usage
processor = DataProcessor()
result = processor.process([1.0, 2.0, 3.0, 4.0, 5.0])
# Uses numpy.mean if available, else built-in sum/len
```

### Example 3: Batched API Requests

```python
# noqa:validation
from lionherd_core.ln._utils import get_bins

async def send_batched_messages(
    messages: list[str],
    max_batch_size: int = 1000
):
    """Send messages in batches under size limit."""
    # Partition by cumulative length
    bins = get_bins(messages, upper=max_batch_size)

    results = []
    for bin_indices in bins:
        batch = [messages[i] for i in bin_indices]

        # Calculate batch stats
        batch_size = sum(len(msg) for msg in batch)
        print(f"Sending batch: {len(batch)} messages, {batch_size} chars")

        # Process batch (example: could send to API, write to file, etc.)
        result = {"batch_size": batch_size, "messages": len(batch), "data": batch}
        results.append(result)

    return results

# Usage
messages = ["short", "medium length message", "x", "another message"]
responses = await send_batched_messages(messages, max_batch_size=30)
# Sending batch: 2 messages, 23 chars
# Sending batch: 1 messages, 21 chars
# Sending batch: 1 messages, 15 chars
```

### Example 4: Plugin System with Validation

```python
from lionherd_core.ln._utils import is_import_installed, import_module

class PluginManager:
    """Manage optional plugins with validation."""

    def __init__(self):
        self.plugins = {}

    def load_plugin(
        self,
        name: str,
        package: str,
        module: str,
        class_name: str
    ):
        """Load plugin if package installed."""
        if not is_import_installed(package):
            print(f"Plugin '{name}' not available (missing {package})")
            return False

        try:
            plugin_class = import_module(package, module, class_name)
            self.plugins[name] = plugin_class()
            print(f"Plugin '{name}' loaded successfully")
            return True
        except ImportError as e:
            print(f"Plugin '{name}' failed to load: {e}")
            return False

    def get_plugin(self, name: str):
        """Get plugin instance or None."""
        return self.plugins.get(name)

# Usage
manager = PluginManager()
manager.load_plugin("fast_json", "orjson", "", "loads")
manager.load_plugin("numpy_ops", "numpy", "", "array")

# Use loaded plugins
if fast_json := manager.get_plugin("fast_json"):
    data = fast_json('{"key": "value"}')
```

### Example 5: Timeout-Protected File Creation

```python
# noqa:validation
from lionherd_core.ln._utils import acreate_path

async def create_output_safely(
    output_dir: str,
    filename: str,
    timeout: float = 5.0
) -> AsyncPath | None:
    """Create output path with timeout fallback."""
    try:
        # Try primary location with timeout
        return await acreate_path(
            output_dir,
            filename,
            extension="json",
            timeout=timeout
        )
    except TimeoutError:
        print(f"Primary storage timed out, using local fallback")
        # Fallback to local temp directory (no timeout)
        return await acreate_path(
            "/tmp",
            filename,
            extension="json"
        )
    except Exception as e:
        print(f"Failed to create path: {e}")
        return None

# Usage
output_path = await create_output_safely("/mnt/slow-network", "results")
if output_path:
    await output_path.write_text('{"status": "complete"}')
```

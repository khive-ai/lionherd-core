# lionherd-core Documentation Standard

> Pandas-level API documentation standard for lionherd-core

Version: 1.0
Date: 2025-11-08
Status: Active

---

## Philosophy

lionherd-core documentation follows **Google-style docstrings** with
lionherd-specific adaptations for protocol-based architecture and
async-first design.

**Core Principles:**

1. **Clarity over cleverness** - Explain "why" not just "what"
2. **Examples are executable** - All code must run successfully
3. **Progressive disclosure** - Basic → Advanced usage patterns
4. **Protocol awareness** - Document protocol implementations explicitly
5. **Type precision** - Leverage Pydantic for accurate type documentation

---

## Documentation Layers

### Layer 1: Inline Docstrings (Google Style)

**Location**: Source code
**Audience**: IDE users, API consumers
**Format**: Google-style docstring convention (matching existing
lionherd-core codebase)

**Required Sections:**

```python
"""Short summary (single sentence, infinitive verb, period).

Extended summary with context, use cases, and design rationale.
Explain protocol implementations and architectural decisions.

Args:
    param_name (type): Description of parameter. Use "optional" for
        None-default params. For multiple types: "int or str"
        (use "or" for final pairing). For discrete values:
        "{'low', 'medium', 'high'}". For defaults: "int, default 0"
        or "str, optional"
    another_param (str, optional): Optional parameter description.

Returns:
    return_type: Description of return value. Include state changes
        if applicable.

Raises:
    ExceptionType: When and why this exception is raised.

See Also:
    related_function: Brief description
    related_class: Related functionality

Note:
    Technical implementation details, caveats, performance considerations.
    Reference protocols implemented: Observable, Serializable, etc.

Examples:
    >>> from lionherd_core import Element
    >>> elem = Element(metadata={"key": "value"})
    >>> elem.id
    UUID('...')
"""
```

**Style Rules:**

- Use three double-quotes (`"""`)
- No blank lines before/after docstring
- Text starts on line after opening quotes
- Closing quotes on separate line
- Inline code uses backticks: `` `param_name` ``
- Cross-references use markdown links: `` [`Element`](../api/base/element.md) ``
- Auto-generated API docs via mkdocstrings support docstring cross-refs

**Type Documentation:**

```python
# Simple types
int, float, str, bool, None

# Collections
list of int
dict of {str : int}
tuple of (str, int, float)

# Multiple types (use "or" for final pairing)
int or str
int, float, or str

# Discrete values
{'auto', 'manual', 'hybrid'}

# Default values
int, default 0
str, optional  # when None means "not provided"

# Async
Coroutine[None, None, str]  # async function returning str

# Protocols
Serializable  # protocol interface
Element  # concrete class
```

**Examples Section Standards:**

- **Imports**: numpy and pandas assumed pre-imported; explicitly import all else
- **Naming**: Consistent variable names: `elem` for Element,
  `pile` for Pile, `df` for DataFrame
- **Data size**: ~4 rows of data (concise examples)
- **Meaningful data**: Prefer semantic data over random matrices
- **Keyword args**: Use `head(n=3)` not `head(3)`
- **Random seeds**: Fix seeds when randomness needed: `random.seed(42)`
- **Continuations**: Use `...` for multi-line code
- **Doctest format**: `>>>` for code, expected output without prefix

### Layer 2: API Reference (Markdown)

**Location**: `docs/api/{module}.md`
**Audience**: Documentation site visitors, comprehensive reference
**Format**: Markdown with structured sections

**Required Structure:**

```markdown
# Class/Function Name

> One-line summary

## Overview

2-3 paragraphs explaining:
- What it is and when to use it
- Key capabilities and design philosophy
- How it fits in lionherd-core architecture

## Class Signature

\```python
class ClassName(BaseClass):
    """Docstring first line."""

    # Constructor signature
    def __init__(
        self,
        param1: Type1,
        param2: Type2 = default,
        **kwargs: Any,
    ) -> None: ...
\```

## Parameters

Detailed parameter documentation from docstring.

## Attributes

Table format:

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Unique identifier (auto-generated, frozen) |
| `created_at` | `datetime` | UTC timestamp (auto-generated, frozen) |

## Methods

### Category 1: Core Operations

#### `method_name()`

Brief description.

**Signature:**
\```python
def method_name(self, param: Type) -> ReturnType: ...
\```

**Parameters:**
- `param` (Type): Description

**Returns:**
- ReturnType: Description

**Examples:**
\```python
>>> instance.method_name(param)
result
\```

### Category 2: Serialization

(Organize methods by functionality)

## Protocol Implementations

Explicitly document which protocols are implemented
(use `@implements()` decorator):

- **Observable**: `id` property (UUID identifier for object identity)
- **Serializable**: `to_dict()`, `to_json()` for serialization
- **Deserializable**: `from_dict()`, `from_json()` for deserialization
- **Hashable**: `__hash__()` based on ID (identity-based hashing)
- **Adaptable**: `adapt_to()`, `adapt_from()` for sync format
  conversion (TOML/YAML/JSON/SQL)
- **AsyncAdaptable**: `adapt_to_async()`, `adapt_from_async()` for
  async I/O format conversion
- **Invocable**: `invoke()` for async execution
- **Containable**: `__contains__()` for membership testing (`in` operator)
- **Allowable**: `allowed()` for defining allowed values/keys

## Usage Patterns

### Basic Usage

\```python
# Simplest use case
\```

### Advanced Usage

\```python
# Complex patterns, composition with other classes
\```

### Common Pitfalls

- **Issue**: What goes wrong
  - **Solution**: How to fix it

## Design Rationale

Explain architectural decisions:
- Why this design?
- What trade-offs were made?
- How does it support the framework's goals?

## See Also

- Related classes/functions with links
- Relevant user guide sections
- External references

## Examples

Comprehensive, real-world examples demonstrating:
1. Basic instantiation
2. Common workflows
3. Advanced patterns
4. Integration with other lionherd-core components

All examples must be executable and produce shown output.
```

### Layer 3: User Guide (Tutorial/Conceptual)

**Location**: `docs/user_guide/{topic}.md`
**Audience**: New users, conceptual learners
**Format**: Narrative markdown with worked examples

**Structure:**

1. **Motivation** - Problem being solved
2. **Concepts** - Core ideas and terminology
3. **Tutorial** - Step-by-step walkthrough
4. **Patterns** - Common usage patterns
5. **Best Practices** - Do's and don'ts
6. **Next Steps** - Related topics

---

## lionherd-core Specifics

### Protocol Documentation

Always document protocol implementations
(use `@implements()` decorator in class):

```markdown
## Protocol Implementations

This class implements the following protocols
(declared via `@implements()`):

- **Observable**: UUID identifier via `id` property
- **Serializable**: Supports `to_dict(mode='python'|'json'|'db')`
  and `to_json()`
- **Deserializable**: Supports `from_dict()` and `from_json()` with
  polymorphic reconstruction
- **Hashable**: ID-based hashing via `__hash__()` (identity equality)
- **Adaptable**: Sync format conversion via `adapt_to()`, `adapt_from()`
  (if applicable)
- **AsyncAdaptable**: Async format conversion via `adapt_to_async()`,
  `adapt_from_async()` (if applicable)
- **Invocable**: Async execution via `invoke()` (if applicable)
- **Containable**: Membership testing via `__contains__()` (if applicable)
- **Allowable**: Allowed values via `allowed()` (if applicable)

<!-- TODO: Add link to Protocols Guide when available (issue #115) -->
```

### Async Documentation

For async methods, document:

1. **Concurrency safety**: Thread-safe? Reentrant?
2. **Cancellation**: How does it handle cancellation?
3. **Timeouts**: Built-in timeout support?
4. **Error handling**: Exception behavior in async context

Example:

```markdown
#### `invoke()` (async)

**Signature:**
\```python
async def invoke(self, **kwargs: Any) -> Any: ...
\```

**Concurrency**: Thread-safe, uses internal lock for state transitions
**Cancellation**: Gracefully handles asyncio.CancelledError
**Timeout**: Configurable via `timeout` parameter (seconds)
```

### Serialization Modes

Document all serialization modes (`python`, `json`, `db`) with examples:

```markdown
### Serialization Modes

#### Python Mode (`mode='python'`)

Native Python types, suitable for in-memory operations.

\```python
>>> elem.to_dict(mode='python')
{'id': UUID('...'), 'created_at': datetime(...), 'metadata': {...}}
\```

#### JSON Mode (`mode='json'`)

JSON-safe types (UUIDs → str, datetime → ISO8601).

\```python
>>> elem.to_dict(mode='json')
{'id': '123e4567-...', 'created_at': '2025-11-08T10:30:00Z', ...}
\```

#### Database Mode (`mode='db'`)

Database adapter format via pydapter.

\```python
>>> elem.to_dict(mode='db')
# Adapter-specific format
\```
```

### Type Annotations

lionherd-core uses Pydantic V2. Document field types precisely:

```markdown
## Attributes

| Attribute | Type | Validation | Description |
|-----------|------|------------|-------------|
| `id` | `UUID` | Auto-generated, frozen | Unique identifier |
| `created_at` | `datetime` | Auto-coerced to UTC, frozen | Creation timestamp |
| `metadata` | `dict[str, Any]` | Auto-converted to dict | Arbitrary metadata |
```

---

## Quality Checklist

Before publishing documentation:

- [ ] All code examples are executable and produce shown output
- [ ] All cross-references resolve correctly
- [ ] Protocol implementations are documented
- [ ] Async behavior is clearly explained (if applicable)
- [ ] Serialization modes are demonstrated with examples
- [ ] Type annotations match actual implementation
- [ ] Design rationale explains "why" not just "what"
- [ ] Common pitfalls are addressed
- [ ] Links to related documentation work
- [ ] Spelling and grammar checked
- [ ] Examples use consistent naming conventions
- [ ] No orphaned sections (all referenced content exists)

---

## File Organization

```text
docs/
├── api/
│   ├── index.md           # API reference landing page
│   ├── base/
│   │   ├── element.md     # Element class
│   │   ├── node.md        # Node class
│   │   ├── event.md       # Event class
│   │   └── ...
│   ├── graph/
│   │   ├── pile.md        # Pile class
│   │   ├── flow.md        # Flow class
│   │   └── ...
│   └── ...
├── user_guide/
│   ├── quickstart.md
│   ├── serialization.md
│   ├── async_patterns.md
│   └── ...
└── development/
    ├── contributing.md
    ├── testing.md
    └── ...
```

---

## Examples

### Good Example (Element.to_dict)

```python
def to_dict(
    self,
    mode: Literal["python", "json", "db"] = "python",
    meta_key: str | None = None,
    created_at_format: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Serialize Element to dictionary with polymorphic reconstruction.

    Injects `lion_class` into metadata for polymorphic deserialization
    via `from_dict()`. Supports three serialization modes for different
    use cases.

    Args:
        mode ({'python', 'json', 'db'}, optional): Serialization mode.
            Defaults to 'python'.
            - 'python': Native Python types (UUID, datetime objects)
            - 'json': JSON-safe types (str representations)
            - 'db': Database adapter format (via pydapter)
        meta_key (str, optional): Custom metadata key. If None,
            uses 'metadata'.
        created_at_format (str, optional): strftime format for created_at.
            Applies to ALL modes. If None, uses default format based
            on mode.
        **kwargs: Passed to pydantic's model_dump(). Common: include,
            exclude, by_alias.

    Returns:
        dict[str, Any]: Serialized dictionary with lion_class injected
            in metadata for polymorphism.

    See Also:
        from_dict: Deserialize from dictionary with polymorphic reconstruction
        to_json: Serialize to JSON string

    Note:
        The `lion_class` key (injected into metadata dict) enables
        polymorphic deserialization - `from_dict()` uses it to
        reconstruct the correct subclass. This is critical for
        deserialization in multi-class workflows.

        Mode selection:
        - Use 'python' for in-memory operations and local serialization
        - Use 'json' for API responses and JSON persistence
        - Use 'db' for database storage via pydapter adapters

    Examples:
    >>> from lionherd_core import Element
    >>> elem = Element(metadata={"key": "value"})

    Python mode (native types):

    >>> elem.to_dict(mode='python')
    {
        'id': UUID('...'),
        'created_at': datetime.datetime(...),
        'metadata': {
            'key': 'value',
            'lion_class': 'lionherd_core.base.element.Element'
        }
    }

    JSON mode (JSON-safe types):

    >>> elem.to_dict(mode='json')
    {
        'id': '123e4567-...',
        'created_at': '2025-11-08T10:30:00.123456+00:00',
        'metadata': {
            'key': 'value',
            'lion_class': 'lionherd_core.base.element.Element'
        }
    }

    Custom created_at format (applies to all modes):

    >>> elem.to_dict(mode='json', created_at_format='%Y-%m-%d')
    {
        'id': '123e4567-...',
        'created_at': '2025-11-08',
        'metadata': {
            'key': 'value',
            'lion_class': 'lionherd_core.base.element.Element'
        }
    }
    """
```

### Bad Example (Missing Context)

```python
def to_dict(self, mode="python", **kwargs):
    """Convert to dict.

    Args:
        mode: The mode
        **kwargs: Other args

    Returns:
        dict: A dictionary
    """
```

**Problems:**

- No explanation of polymorphism support
- Vague parameter descriptions ("The mode")
- Missing See Also, Notes, Examples
- No design rationale
- Doesn't explain **kwargs forwarding

---

## Maintenance

**Review Cycle**: Documentation reviewed during PR process

**Update Triggers**:

- API changes (signature, parameters, return types)
- New protocol implementations
- Behavior changes (even if signature unchanged)
- Bug fixes affecting documented behavior

**Deprecation**:

- Document deprecated APIs with migration path
- Add warning to docstring and API reference
- Link to replacement functionality

Example deprecation notice:

```python
"""
.. deprecated:: 0.5.0
    `old_method()` is deprecated and will be removed in 1.0.0.
    Use :meth:`new_method` instead.
"""
```

---

## mkdocs Configuration

lionherd-core uses **mkdocs** with **mkdocstrings** for documentation
generation.

**Required `mkdocs.yml` configuration:**

```yaml
site_name: lionherd-core Documentation
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
    - search.suggest

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
            show_root_full_path: false
            merge_init_into_class: true
            show_signature_annotations: true

nav:
  - Home: index.md
  - User Guide:
      - Quickstart: user_guide/quickstart.md
      - Serialization: user_guide/serialization.md
  - API Reference:
      - Base: api/base/element.md
      - Graph: api/graph/pile.md
```

**Auto-generating API reference pages:**

mkdocstrings automatically extracts docstrings. Create minimal markdown files:

```markdown
# Element

::: lionherd_core.base.Element
```

This renders the full API documentation from source code docstrings.

---

## References

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)
- [mkdocs Documentation](https://www.mkdocs.org/)
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)

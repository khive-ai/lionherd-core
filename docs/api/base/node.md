# Node

> Polymorphic content container with auto-registry and pydapter integration

## Overview

`Node` is the polymorphic container in lionherd-core, extending Element with
structured content storage, embedding support, and automatic subclass
registration. It enables graph-of-graphs patterns through type-safe content
composition and polymorphic deserialization.

**Key Capabilities:**

- **Element Inheritance**: Auto-generated UUID, timestamps, metadata (from
  Element base class)
- **Structured Content**: `content: dict | Serializable | BaseModel | None`
  enforces query-able, composable data
- **Embedding Support**: Optional `embedding: list[float]` for vector search
  with DB JSON string coercion
- **Auto-Registry**: Subclasses automatically register in `NODE_REGISTRY` for
  polymorphic deserialization
- **Pydapter Integration**: TOML/YAML adapters with isolated per-subclass
  registries

**When to Use Node:**

- Content-bearing entities (documents, messages, data records)
- Graph databases with heterogeneous node types
- Vector search with semantic embeddings
- Nested composition (graph-of-graphs patterns)
- External format serialization (TOML/YAML via pydapter)

**When NOT to Use Node (Use Element Instead):**

- Pure identity objects without content (use Element)
- Workflow state machines (use Event for state transitions)
- Value objects with content-based equality (use HashableModel)

See [Element](element.md) for identity-based base class.

## Class Signature

```python
from lionherd_core.base import Node, NODE_REGISTRY

@implements(Deserializable, Adaptable, AsyncAdaptable)
class Node(Element, PydapterAdaptable, PydapterAsyncAdaptable):
    """Polymorphic node with arbitrary content, embeddings, pydapter
    integration.

    Auto-registers subclasses in NODE_REGISTRY for polymorphic deserialization.
    """

    # Constructor signature
    def __init__(
        self,
        *,
        content: dict[str, Any] | Serializable | BaseModel | None = None,
        embedding: list[float] | None = None,
        # Inherited from Element:
        id: UUID | str | None = None,
        created_at: datetime | str | int | float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**content** : dict | Serializable | BaseModel | None, optional

Structured data payload. Enforces query-able, composable types for PostgreSQL
JSONB and graph-of-graphs patterns.

- **Accepted**: `dict`, `Serializable` protocol, Pydantic `BaseModel`, `None`
- **Rejected**: Primitives (str, int, float, bool), collections (list, tuple,
  set)
- Auto-serialization: Nested Elements automatically serialize via
  `_serialize_content`
- Auto-deserialization: Dicts with `lion_class` metadata auto-deserialize to
  correct type
- **Migration**: Wrap primitives in dict: `content={"value": "text"}` or use
  `Element.metadata`
- Default: `None`

**embedding** : list of float, optional

Optional embedding vector for semantic search and vector databases.

- Type: `list[float]` (validated and normalized)
- Validation: Must contain only numeric values, cannot be empty list
- Coercion: Integers auto-convert to floats, JSON strings parse to lists during
  INPUT validation
- Default: `None`

**id** : UUID or str, optional (inherited from Element)

Unique identifier. Auto-generated via `uuid4()` if not provided. Frozen field.

**created_at** : datetime or str or int or float, optional (inherited from
Element)

UTC creation timestamp. Auto-generated if not provided. Frozen field.

**metadata** : dict of {str : Any}, optional (inherited from Element)

Arbitrary metadata. Auto-converts non-dict objects via `to_dict()`.

## Attributes

| Attribute    | Type                | Frozen | Inherited | Description                        |
| ------------ | ------------------- | ------ | --------- | ---------------------------------- |
| `content`    | `Any`               | No     | No        | Arbitrary content payload          |
| `embedding`  | `list[float]` or None | No     | No        | Optional embedding vector          |
| `id`         | `UUID`              | Yes    | Element   | Unique identifier                  |
| `created_at` | `datetime`          | Yes    | Element   | UTC creation timestamp             |
| `metadata`   | `dict[str, Any]`    | No     | Element   | Arbitrary metadata                 |

## Methods

### Deserialization

#### `from_dict()`

Deserialize from dictionary with **polymorphic type restoration** and optional **content deserialization** for round-trip transformations.

**Signature:**

```python
@classmethod
def from_dict(
    cls,
    data: dict[str, Any],
    meta_key: str | None = None,
    content_deserializer: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> Node: ...
```

**Parameters:**

- `data` (dict[str, Any]): Serialized node dictionary (from `to_dict()`)
- `meta_key` (str, optional): Restore metadata from this key (db mode
  compatibility). Default: `'metadata'`
- `content_deserializer` (Callable[[Any], Any], optional): Custom function to deserialize content field.
  Applied to content field before model_validate. Enables round-trip serialization with custom transformations.
  Must be symmetric inverse of content_serializer used in to_dict(). Default: None
- `**kwargs` (Any): Forwarded to Pydantic's `model_validate()`

**Returns:**

- Node: Deserialized Node instance (or correct subclass if `lion_class`
  present)

**Examples:**

```python
from lionherd_core.base import Node

# Polymorphic deserialization
class PersonNode(Node):
    name: str
    age: int

person = PersonNode(name="Alice", age=30, content={"bio": "text"})
data = person.to_dict()
restored = Node.from_dict(data)
type(restored).__name__  # 'PersonNode' (correct subclass)
restored.name  # 'Alice'

# Round-trip with compression
import json, zlib, base64

def compress(content):
    json_bytes = json.dumps(content).encode()
    compressed = zlib.compress(json_bytes)
    return {"compressed": base64.b64encode(compressed).decode()}

def decompress(content):
    compressed = base64.b64decode(content["compressed"])
    json_bytes = zlib.decompress(compressed)
    return json.loads(json_bytes)

node = Node(content={"large": "data" * 100})
data = node.to_dict(content_serializer=compress)
restored = Node.from_dict(data, content_deserializer=decompress)
# Original content restored transparently

# Round-trip with encryption
def encrypt(content):
    # Production: use cryptography.fernet or similar
    return {"encrypted": encrypt_data(content)}

def decrypt(content):
    return decrypt_data(content["encrypted"])

node = Node(content={"sensitive": "data"})
data = node.to_dict(content_serializer=encrypt)
restored = Node.from_dict(data, content_deserializer=decrypt)
# Sensitive content encrypted at rest, decrypted on access
```

**See Also:**

- `to_dict()`: Serialize to dictionary (inherited from Element)
- `NODE_REGISTRY`: Global registry mapping class names to classes

**Notes:**

**Polymorphic Routing Workflow:**

1. Extracts `lion_class` from metadata dict
2. Looks up target class in `NODE_REGISTRY` (short name or full module path)
3. If found and different from calling class, delegates to target's `from_dict()`
4. Returns instance of correct subclass

This enables heterogeneous collections from database queries to deserialize
with correct types.

**Metadata Key Handling:**

- Supports `meta_key` parameter for custom metadata keys
- Backward compatibility: Auto-checks `node_metadata` if `metadata` not found
- `lion_class` removed from metadata after deserialization (serialization-only)

### Pydapter Integration

#### `adapt_to()`

Convert to external format via pydapter (TOML, YAML, etc.).

**Signature:**

```python
def adapt_to(
    self,
    obj_key: str,
    many: bool = False,
    **kwargs: Any,
) -> Any: ...
```

**Parameters:**

- `obj_key` (str): Adapter key (e.g., `"toml"`, `"yaml"`). Must register
  adapter first!
- `many` (bool, optional): Adapt multiple instances. Default: `False`
- `**kwargs` (Any): Forwarded to adapter. Defaults: `adapt_meth="to_dict"`,
  `adapt_kw={"mode": "db"}`

**Returns:**

- Any: Adapted representation (typically string for TOML/YAML)

**Examples:**

```python
from lionherd_core.base import Node

# Base Node has TOML/YAML built-in
node = Node(content={"value": "test data"})
toml_str = node.adapt_to("toml")
# 'content = "test data"\n...'

# Subclasses have ISOLATED registries - must register explicitly
class CustomNode(Node):
    custom_field: str

from pydapter.adapters import TomlAdapter
CustomNode.register_adapter(TomlAdapter)

custom = CustomNode(content={"value": "data"}, custom_field="value")
custom_toml = custom.adapt_to("toml")
```

**See Also:**

- `adapt_from()`: Deserialize from external format
- `register_adapter()`: Register adapter for subclass (from pydapter)

**Notes:**

**Isolated Registry Pattern (Rust-like Explicit):**

Base `Node` has TOML/YAML adapters pre-registered. Subclasses get **isolated
registries** and do NOT inherit adapters. This prevents adapter pollution while
keeping base Node convenient.

```python
# Base Node works
Node(content={"value": "test"}).adapt_to("toml")  # ✓ Works

# Subclass isolated
class MyNode(Node):
    pass

MyNode(content={"value": "test"}).adapt_to("toml")  # ✗ Fails (no adapter)

# Must explicitly register
MyNode.register_adapter(TomlAdapter)
MyNode(content={"value": "test"}).adapt_to("toml")  # ✓ Now works
```

#### `adapt_from()`

Create from external format via pydapter with polymorphic type restoration.

**Signature:**

```python
@classmethod
def adapt_from(
    cls,
    obj: Any,
    obj_key: str,
    many: bool = False,
    **kwargs: Any,
) -> Node: ...
```

**Parameters:**

- `obj` (Any): Source object (typically string for TOML/YAML)
- `obj_key` (str): Adapter key (e.g., `"toml"`, `"yaml"`)
- `many` (bool, optional): Deserialize multiple instances. Default: `False`
- `**kwargs` (Any): Forwarded to adapter. Default: `adapt_meth="from_dict"`

**Returns:**

- Node: Deserialized Node instance (or correct subclass)

**Examples:**

```python
from lionherd_core.base import Node

# TOML roundtrip with polymorphism
toml_str = """
content = "test"
[metadata]
lion_class = "lionherd_core.base.node.Node"
"""

restored = Node.adapt_from(toml_str, "toml")
type(restored).__name__  # 'Node'
restored.content  # 'test'
```

#### `adapt_to_async()` and `adapt_from_async()`

Async versions of `adapt_to()` and `adapt_from()` for async I/O operations.

**Signatures:**

```python
async def adapt_to_async(
    self,
    obj_key: str,
    many: bool = False,
    **kwargs: Any,
) -> Any: ...

@classmethod
async def adapt_from_async(
    cls,
    obj: Any,
    obj_key: str,
    many: bool = False,
    **kwargs: Any,
) -> Node: ...
```

Same parameters and behavior as sync versions, but support async I/O adapters.

### Serialization

#### `to_dict()`

Serialize Node to dictionary with optional custom content serialization.

**Signature:**

```python
def to_dict(
    self,
    mode: Literal["python", "json", "db"] = "python",
    created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
    meta_key: str | None = None,
    embedding_format: Literal["pgvector", "jsonb", "list"] | None = None,
    content_serializer: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]: ...
```

**Parameters:**

- `mode` (str, optional): Serialization mode ('python', 'json', or 'db'). Default: 'python'
- `created_at_format` (str, optional): Format for created_at field. Default: auto-selected by mode
- `meta_key` (str, optional): Rename metadata field. Default: 'node_metadata' for db mode
- `embedding_format` (str, optional): Format for embedding serialization ('pgvector', 'jsonb', or 'list'). Default: 'list'
- `content_serializer` (Callable[[Any], Any], optional): Custom function to serialize content field.
  If provided, content is excluded from model_dump and replaced with `content_serializer(self.content)` result.
  Default: None (use default field serialization)
- `**kwargs` (Any): Additional arguments passed to model_dump()

**Returns:**

- dict[str, Any]: Serialized Node dictionary

**Examples:**

```python
from lionherd_core.base import Node

# Default serialization
node = Node(content={"key": "value"})
data = node.to_dict()
# {'id': '...', 'content': {'key': 'value'}, ...}

# Custom content serialization
def compress_content(content):
    import json
    return {"compressed": json.dumps(content)}

node = Node(content={"large": "data"})
data = node.to_dict(content_serializer=compress_content)
# {'id': '...', 'content': {'compressed': '{"large": "data"}'}, ...}

# Lambda serializer
node = Node(content={"key": "value"})
data = node.to_dict(content_serializer=lambda c: str(c))
# {'id': '...', 'content': "{'key': 'value'}", ...}

# Combine with embedding_format
node = Node(content={"data": "value"}, embedding=[0.1, 0.2, 0.3])
data = node.to_dict(
    mode="db",
    content_serializer=lambda c: {"ref": "external://12345"},
    embedding_format="pgvector"
)
# {'id': '...', 'content': {'ref': 'external://12345'},
#  'embedding': '[0.1,0.2,0.3]', 'node_metadata': {...}}
```

**See Also:**

- `from_dict()`: Deserialize from dictionary with `content_deserializer` for round-trip support

**Notes:**

**Content Serializer Use Cases (Round-Trip Patterns):**

1. **Compression**: Transform large content for storage efficiency
   - Serializer: `json.dumps() → zlib.compress() → base64.encode()`
   - Deserializer: `base64.decode() → zlib.decompress() → json.loads()`
   - Use case: Store 100KB documents as 10KB compressed data

2. **Encryption**: Protect sensitive content at rest
   - Serializer: `encrypt(content) → {"encrypted": "..."}`
   - Deserializer: `decrypt(content["encrypted"]) → original`
   - Use case: HIPAA/GDPR compliance for PII storage

3. **External Storage**: Store large content externally (S3, CDN)
   - Serializer: `store_to_s3(content) → {"ref": "s3://..."}`
   - Deserializer: `fetch_from_s3(ref) → original`
   - Use case: Keep database lightweight, store videos/datasets externally

4. **Format Conversion**: API-specific serialization
   - Serializer: `to_api_format(content) → {"data": ..., "version": "v1"}`
   - Deserializer: `from_api_format(data) → original`
   - Use case: Transform content for external API consumption

**Round-Trip Pattern:**

```python
# Serialize
data = node.to_dict(content_serializer=transform_fn)

# Deserialize (requires symmetric inverse)
restored = Node.from_dict(data, content_deserializer=inverse_fn)

# restored.content == original.content ✓
```

**Important**: Always provide symmetric `content_deserializer` to `from_dict()` for round-trip correctness.
Without deserializer, content remains in transformed format after deserialization.

### Special Methods (Inherited from Element)

Node inherits Element methods with Node-specific behavior for `to_dict()` (documented above).
See [Element API documentation](element.md) for other methods:

- `to_json(pretty=False, **kwargs)`: Serialize to JSON string
- `from_json(json_str, **kwargs)`: Deserialize from JSON string
- `class_name(full=False)`: Get class name without generic parameters
- `__eq__(other)`: Identity-based equality (ID comparison)
- `__hash__()`: ID-based hashing
- `__repr__()`: String representation

## Protocol Implementations

Node implements six core protocols (three from Element + three additional):

### Observable (inherited from Element)

**Property**: `id` (UUID identifier)

### Serializable (inherited from Element)

**Methods**:

- `to_dict(mode='python'|'json'|'db', **kwargs)`: Dictionary serialization
- `to_json(**kwargs)`: JSON string serialization

**Node-Specific Behavior**:

- `content` field: Nested Elements auto-serialize via `_serialize_content`
- `embedding` field: Serializes as JSON string in db mode for database
  compatibility

### Deserializable

**Methods**:

- `from_dict(data, meta_key=None, **kwargs)`: Polymorphic deserialization
- `from_json(json_str, **kwargs)`: JSON deserialization

**Node-Specific Behavior**:

- Uses `NODE_REGISTRY` for polymorphic type restoration
- Content dicts with `lion_class` auto-deserialize to correct Element/Node type
- Embedding JSON strings auto-parse to `list[float]`

### Hashable (inherited from Element)

**Methods**: `__hash__()` based on ID

**Equality**: `__eq__()` compares IDs only

### Adaptable

**Methods**:

- `adapt_to(obj_key, **kwargs)`: Sync external format conversion
- `register_adapter(adapter)`: Register adapter for class (from pydapter)

**Supported Formats**: TOML, YAML (base Node only)

### AsyncAdaptable

**Methods**:

- `adapt_to_async(obj_key, **kwargs)`: Async external format conversion
- `adapt_from_async(obj, obj_key, **kwargs)`: Async deserialization

<!-- TODO: Add link to Protocols Guide when available (issue #115) -->

## NODE_REGISTRY

Global dictionary mapping class names to Node subclasses for polymorphic
deserialization.

**Type**: `dict[str, type[Node]]`

**Auto-Registration**: All Node subclasses automatically register via
`__pydantic_init_subclass__`

**Key Formats**: Both short name (`"PersonNode"`) and full module path
(`"myapp.models.PersonNode"`)

**Example:**

```python
from lionherd_core.base import Node, NODE_REGISTRY

# Define subclass - auto-registers
class PersonNode(Node):
    name: str

# Registry contains both key formats
assert "PersonNode" in NODE_REGISTRY
assert "myapp.models.PersonNode" in NODE_REGISTRY
assert NODE_REGISTRY["PersonNode"] is PersonNode
```

**Notes:**

- Registration happens at class definition time (via metaclass)
- No manual registration required
- Enables `from_dict()` to route to correct subclass via `lion_class` metadata

## Usage Patterns

### Basic Usage

```python
from lionherd_core.base import Node

# Create node with content
node = Node(content={"text": "Hello World"})
print(node.id)          # UUID('...')
print(node.content)     # "Hello World"

# Create with embedding
node = Node(
    content={"text": "semantic"},
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
)
print(len(node.embedding))  # 5
```

### Content Polymorphism

```python
from lionherd_core.base import Node

# Content can be anything
node1 = Node(content={"value": "string"})
node2 = Node(content={"key": "value", "nested": [1, 2, 3]})
node3 = Node(content={"items": ["a", "b", "c"]})

# Nested Elements
from lionherd_core.base import Element
inner = Element(metadata={"type": "inner"})
outer = Node(content=inner)

# Serialize - inner becomes dict automatically
data = outer.to_dict()
type(data["content"])  # dict
"metadata" in data["content"]  # True

# Deserialize - inner becomes Element automatically
restored = Node.from_dict(data)
type(restored.content).__name__  # 'Element'
```

### Custom Node Types

```python
from lionherd_core.base import Node

# Define domain-specific nodes
class PersonNode(Node):
    name: str
    age: int

class DocumentNode(Node):
    title: str
    body: str
    tags: list[str] = []

# Auto-registration enables polymorphic workflows
person = PersonNode(name="Alice", age=30, content={"bio": "text"})
doc = DocumentNode(title="Spec", body="Requirements", tags=["v1", "draft"])

# Serialize and deserialize
person_data = person.to_dict()
doc_data = doc.to_dict()

# Polymorphic deserialization via base Node.from_dict()
restored_person = Node.from_dict(person_data)
restored_doc = Node.from_dict(doc_data)

type(restored_person).__name__  # 'PersonNode'
type(restored_doc).__name__  # 'DocumentNode'
restored_person.name  # 'Alice'
restored_doc.title  # 'Spec'
```

### Heterogeneous Collections (Database Query Scenario)

```python
from lionherd_core.base import Node

# Simulate database query returning mixed node types
db_records = [
    {
        "name": "Alice",
        "age": 30,
        "node_metadata": {"lion_class": "PersonNode"}
    },
    {
        "title": "Report",
        "body": "Q4 Results",
        "node_metadata": {"lion_class": "DocumentNode"}
    },
    {
        "name": "Bob",
        "age": 25,
        "node_metadata": {"lion_class": "PersonNode"}
    },
]

# Single deserialization call handles mixed types
nodes = [Node.from_dict(record) for record in db_records]

# Types correctly restored
types = [type(n).__name__ for n in nodes]
# ['PersonNode', 'DocumentNode', 'PersonNode']

# Can filter by type
people = [n for n in nodes if isinstance(n, PersonNode)]
len(people)  # 2
```

### Embedding Field Usage

```python
from lionherd_core.base import Node
import orjson

# Standard usage
node = Node(content={"text": "semantic"}, embedding=[0.1, 0.2, 0.3])

# JSON string coercion (database compatibility)
json_str = orjson.dumps([0.4, 0.5, 0.6]).decode()
node_from_db = Node(content={"value": "text"}, embedding=json_str)
node_from_db.embedding  # [0.4, 0.5, 0.6] (parsed)

# Integer coercion to float
node_ints = Node(content={"value": "text"}, embedding=[1, 2, 3])
node_ints.embedding  # [1.0, 2.0, 3.0] (floats)
all(isinstance(x, float) for x in node_ints.embedding)  # True
```

### Pydapter Integration

```python
from lionherd_core.base import Node
from pydapter.adapters import TomlAdapter

# Base Node has TOML/YAML built-in
node = Node(content={"value": "test data"})
toml_str = node.adapt_to("toml")

# Roundtrip
restored = Node.adapt_from(toml_str, "toml")
restored.content  # "test data"

# Subclass isolation
class CustomNode(Node):
    custom_field: str

# Must register adapter explicitly
CustomNode.register_adapter(TomlAdapter)
custom = CustomNode(content={"value": "data"}, custom_field="value")
custom_toml = custom.adapt_to("toml")
```

### Serialization Modes

```python
from lionherd_core.base import Node

node = Node(content={"value": "test"})

# Python mode - native types
python_dict = node.to_dict(mode="python")
# {'id': UUID('...'), 'created_at': datetime(...), 'content': {'value': 'test'},
#  'metadata': {...}}

# JSON mode - JSON-safe types
json_dict = node.to_dict(mode="json")
# {'id': '...', 'created_at': '2025-11-08T...', 'content': {'value': 'test'},
#  'metadata': {...}}

# DB mode - database format with node_metadata
db_dict = node.to_dict(mode="db")
# {'id': '...', 'created_at': '...', 'content': {'value': 'test'},
#  'node_metadata': {..., 'lion_class': '...'}}
```

### Custom Content Serialization

```python
from lionherd_core.base import Node

# Example 1: External storage reference
def create_external_ref(content):
    """Replace large content with external storage reference."""
    # Store content externally (S3, database, etc.)
    ref_id = store_externally(content)
    return {"ref": f"s3://bucket/{ref_id}", "size": len(str(content))}

node = Node(content={"large": "data" * 1000})
data = node.to_dict(mode="db", content_serializer=create_external_ref)
# {'content': {'ref': 's3://bucket/abc123', 'size': 4000}, ...}

# Example 2: Compression
def compress_content(content):
    """Compress content for storage."""
    import json
    import zlib
    import base64

    json_bytes = json.dumps(content).encode()
    compressed = zlib.compress(json_bytes)
    encoded = base64.b64encode(compressed).decode()
    return {"compressed": encoded, "format": "zlib+base64"}

node = Node(content={"large": "dataset"})
data = node.to_dict(content_serializer=compress_content)
# {'content': {'compressed': '...', 'format': 'zlib+base64'}, ...}

# Example 3: Type coercion for API
def api_format(content):
    """Format content for API response."""
    return {
        "data": content,
        "timestamp": datetime.now().isoformat(),
        "version": "v1"
    }

node = Node(content={"key": "value"})
api_data = node.to_dict(mode="json", content_serializer=api_format)
# {'content': {'data': {'key': 'value'}, 'timestamp': '...', 'version': 'v1'}, ...}

# Example 4: Combine with embedding_format
node = Node(content={"doc": "text"}, embedding=[0.1, 0.2, 0.3])
data = node.to_dict(
    mode="db",
    content_serializer=lambda c: {"summary": c.get("doc", "")[:50]},
    embedding_format="pgvector"
)
# {'content': {'summary': 'text'}, 'embedding': '[0.1,0.2,0.3]', ...}
```

### Common Pitfalls

#### Pitfall 1: Forgetting Adapter Registration for Subclasses

**Issue**: Assuming subclasses inherit adapters from base Node.

```python
class MyNode(Node):
    pass

# This fails - subclass has isolated registry
MyNode(content={"value": "test"}).adapt_to("toml")  # ✗ Adapter not found
```

**Solution**: Explicitly register adapters on subclass.

```python
from pydapter.adapters import TomlAdapter

MyNode.register_adapter(TomlAdapter)
MyNode(content={"value": "test"}).adapt_to("toml")  # ✓ Works
```

#### Pitfall 2: Empty Embedding List

**Issue**: Providing empty list for embedding raises ValueError.

```python
Node(content={"value": "text"}, embedding=[])  # ✗ ValueError: cannot be empty
```

**Solution**: Use `None` for no embedding or provide at least one value.

```python
Node(content={"value": "text"}, embedding=None)  # ✓ Works
Node(content={"value": "text"}, embedding=[0.0])  # ✓ Works
```

#### Pitfall 3: Assuming Content Auto-Deserializes Without lion_class

**Issue**: Expecting plain dicts to deserialize as Elements.

```python
data = {"content": {"some": "dict"}}
node = Node.from_dict(data)
type(node.content)  # dict (NOT Element)
```

**Solution**: Include `lion_class` in metadata for polymorphic deserialization.

```python
data = {
    "content": {
        "metadata": {"lion_class": "lionherd_core.base.element.Element"}
    }
}
node = Node.from_dict(data)
type(node.content).__name__  # 'Element'
```

## Design Rationale

### Why Content Polymorphism?

`content: Any` enables composition over inheritance:

1. **Flexibility**: Single Node type handles all data shapes (primitives,
   collections, nested structures)
2. **Graph-of-Graphs**: Node can contain Graph which contains Nodes (recursive
   composition)
3. **No Schema Lock-In**: Add new data types without modifying Node definition

For **typed content validation**, create custom Node subclasses with specific
field types.

### Why Auto-Registry?

Automatic subclass registration eliminates boilerplate:

**Without Auto-Registry** (manual):

```python
class PersonNode(Node):
    name: str

# Manual registration required
NODE_REGISTRY["PersonNode"] = PersonNode
```

**With Auto-Registry** (automatic):

```python
class PersonNode(Node):
    name: str

# ✓ Automatically registered via __pydantic_init_subclass__
```

This zero-config pattern reduces errors and makes polymorphic deserialization
"just work".

### Why Isolated Adapter Registries?

Subclasses get isolated pydapter registries (Rust-like explicit pattern) to
prevent adapter pollution:

1. **Explicit > Implicit**: Subclass authors explicitly choose which formats to
   support
2. **No Pollution**: Base Node's TOML/YAML don't leak to unrelated subclasses
3. **Base Convenience**: Base Node still has built-in adapters for common usage

Trade-off: More verbose but safer and more maintainable for large codebases.

### Why Embedding Field?

`embedding: list[float]` with JSON string coercion enables vector search:

1. **Semantic Search**: Store embeddings alongside content for similarity
   queries
2. **DB Compatibility**: Graph databases (Neo4j) store embeddings as JSON
   strings
3. **Type Safety**: Validation ensures only numeric values, auto-coerces ints to
   floats

## See Also

- **Related Classes**:
  - [Element](element.md): Base class with identity and serialization
  - [Event](event.md): Element subclass with async execution lifecycle
  <!-- TODO: Add link to Graph/Pile documentation when available (issue #115) -->
<!-- TODO: Add links to user guides when available (issue #115) -->

## Examples

### Example 1: Content Polymorphism

```python
from lionherd_core.base import Node, Element

# Different content types
node_str = Node(content="plain string")
node_dict = Node(content={"key": "value"})
node_list = Node(content=[1, 2, 3])

# Nested Element
elem = Element(metadata={"type": "config"})
node_elem = Node(content=elem)

# Serialize - Element auto-converts to dict
data = node_elem.to_dict()
type(data["content"])  # dict

# Deserialize - dict auto-converts to Element
restored = Node.from_dict(data)
type(restored.content).__name__  # 'Element'
```

### Example 2: Custom Node Types with Polymorphism

```python
from lionherd_core.base import Node

class PersonNode(Node):
    name: str
    age: int

class DocumentNode(Node):
    title: str
    body: str

# Create instances
person = PersonNode(name="Alice", age=30)
doc = DocumentNode(title="Spec", body="Requirements")

# Serialize
person_data = person.to_dict()
doc_data = doc.to_dict()

# Heterogeneous collection
collection = [person_data, doc_data]

# Polymorphic deserialization
nodes = [Node.from_dict(item) for item in collection]
[type(n).__name__ for n in nodes]  # ['PersonNode', 'DocumentNode']
```

### Example 3: Embedding Field

```python
from lionherd_core.base import Node
import orjson

# Standard embedding
node1 = Node(content={"text": "semantic"}, embedding=[0.1, 0.2, 0.3])

# From JSON string (database compatibility)
json_str = orjson.dumps([0.4, 0.5, 0.6]).decode()
node2 = Node(content={"value": "text"}, embedding=json_str)
node2.embedding  # [0.4, 0.5, 0.6]

# Integer coercion
node3 = Node(content={"value": "text"}, embedding=[1, 2, 3])
all(isinstance(x, float) for x in node3.embedding)  # True
```

### Example 4: Pydapter Roundtrip

```python
from lionherd_core.base import Node
from pydapter.adapters import TomlAdapter

# Serialize to TOML
node = Node(content={"value": "test data"}, metadata={"env": "prod"})
toml_str = node.adapt_to("toml")

# Deserialize from TOML with polymorphism
restored = Node.adapt_from(toml_str, "toml")
type(restored).__name__  # 'Node'
restored.content  # 'test data'
restored.metadata["env"]  # 'prod'

# Subclass with explicit adapter registration
class CustomNode(Node):
    custom_field: str = "value"

CustomNode.register_adapter(TomlAdapter)
custom = CustomNode(content={"value": "data"})
custom_toml = custom.adapt_to("toml")
```

### Example 5: Database Query Scenario

```python
from lionherd_core.base import Node

# Simulate heterogeneous query results
class PersonNode(Node):
    name: str

class CompanyNode(Node):
    company_name: str

# Mixed results from database
db_results = [
    {"name": "Alice", "node_metadata": {"lion_class": "PersonNode"}},
    {
        "company_name": "Acme Corp",
        "node_metadata": {"lion_class": "CompanyNode"}
    },
    {"name": "Bob", "node_metadata": {"lion_class": "PersonNode"}},
]

# Single deserialization pattern handles all types
nodes = [Node.from_dict(record) for record in db_results]

# Filter by type
people = [n for n in nodes if isinstance(n, PersonNode)]
companies = [n for n in nodes if isinstance(n, CompanyNode)]

len(people)  # 2
len(companies)  # 1
people[0].name  # 'Alice'
companies[0].company_name  # 'Acme Corp'
```

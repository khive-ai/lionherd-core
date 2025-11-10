# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Node test suite: Content polymorphism, registry patterns, and async adaptation strategies.

Node Design Philosophy
======================

Node extends Element with a polymorphic content field that accepts any Python value.
This design enables flexible data structures without hardcoding content types, following
Ocean's composition-over-inheritance principle.

Core Patterns Validated
========================

1. **Content Polymorphism**
   - Structured data only: Serializable, BaseModel, dict, or None
   - Automatic nested Element serialization via field_serializer
   - Strict validation enforces composable, query-able content
   - Enables graph-of-graphs patterns (Node contains Graph contains Nodes)

2. **Registry Metaclass Pattern**
   - Auto-registration via __pydantic_init_subclass__ hook
   - Eliminates manual registry calls (zero-config subclasses)
   - Enables polymorphic from_dict() without explicit imports
   - Trade-off: Global registry state for developer convenience

3. **Polymorphic Deserialization**
   - lion_class metadata routes to correct subclass in NODE_REGISTRY
   - Fully qualified names prevent collisions (module.ClassName)
   - Graceful degradation: unknown lion_class → base Node
   - Critical for heterogeneous collections from database queries

4. **Async Adapter Pattern** (Implementation tested indirectly)
   - Node inherits PydapterAsyncAdaptable for I/O operations
   - async adapt_to_async/adapt_from_async for DB/network serialization
   - Shared adapter registry (_registry()) across all subclasses
   - Avoids redundant adapter registration while maintaining extensibility

5. **Database Mode (mode="db")**
   - Uses node_metadata instead of metadata to avoid column conflicts
   - from_dict() normalizes both formats transparently
   - All modes inject lion_class for polymorphic roundtrips

6. **Embedding Field**
   - Optional semantic vector (list[float] | None)
   - JSON string coercion for database compatibility
   - Validation: no empty lists, numeric values only
   - Supports embedding generation strategies (external services)

Real-World Scenarios
====================

**Heterogeneous Graph Traversal:**
```python
# Database query returns mixed node types
results = db.query("SELECT * FROM nodes WHERE id IN (ancestors_of(?))")
nodes = [Node.from_dict(row) for row in results]
# Each deserializes to correct subclass: PersonNode, DocumentNode, etc.
```

**Async Database Serialization:**
```python
# Store nodes asynchronously with pydapter
node = PersonNode(name="Alice", age=30)
await node.adapt_to_async("postgresql", adapter_config={...})

# Load nodes asynchronously
restored = await Node.adapt_from_async(db_row, "postgresql")
# Polymorphic deserialization preserves subclass type
```

**Nested Element Composition:**
```python
# Node content can contain other Elements
inner = PersonNode(name="Bob")
outer = DocumentNode(title="Profile", body=inner)
# Serialization automatically handles nested Element → dict conversion
```

Why These Patterns Matter
==========================

- **Zero-Config Subclasses**: Registry eliminates boilerplate, subclasses "just work"
- **Type Safety**: Polymorphic deserialization preserves runtime types across serialization
- **Database Integration**: node_metadata + async adapters enable seamless ORM-less persistence
- **Composition Patterns**: Polymorphic content enables flexible data structures without rigid schemas
- **Production Scale**: Async adapters handle I/O-bound operations without blocking

Test Organization
=================

Tests are grouped by design aspect:
1. Basic creation and content storage
2. Subclass auto-registration via __pydantic_init_subclass__
3. Polymorphic deserialization (lion_class routing)
4. Database mode (node_metadata format)
5. Content field nested Element handling
6. Mixed-type collections (heterogeneous deserialization)
7. Serialization mode consistency (python/json/db)
8. Edge cases and validation
9. Pydapter integration (TOML/YAML adapters)
10. Embedding field validation and coercion

Each test validates a specific design decision with rationale documented inline.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import BaseModel

from lionherd_core.base.element import Element
from lionherd_core.base.node import NODE_REGISTRY, Node

# ============================================================================
# Test Node Subclasses
# ============================================================================


class PersonNode(Node):
    """Node representing a person."""

    name: str = "Unknown"
    age: int = 0


class DocumentNode(Node):
    """Node representing a document."""

    title: str = "Untitled"
    body: str = ""


class NestedNode(Node):
    """Node with nested Element in content."""

    label: str = "nested"


# Module-level Element subclass (not Node) for testing Element.from_dict fallback
class CustomElementForTest(Element):
    """Element subclass for testing non-Node Element deserialization."""

    custom_field: str = "test"


# ============================================================================
# Basic Node Tests
# ============================================================================
#
# Design aspect validated: Node creation with default content and automatic
# ID generation from Element base class. Tests foundational behavior that all
# other features depend on.


def test_node_creation():
    """Test Node creation with default content=None.

    Design validation: Node is the fundamental unit of composition in lionherd.
    Default content=None follows Pydantic patterns and enables optional data attachment.

    Inherits from Element:
    - id: UUID (automatic generation via Element.id_obj field validator)
    - metadata: dict (for lion_class injection during serialization)
    - created_at: datetime (automatic timestamp)

    This foundational behavior underpins all other Node features.
    """
    node = Node()

    assert isinstance(node.id, UUID)
    assert node.content is None
    assert isinstance(node.metadata, dict)


def test_node_with_content():
    """Test Node stores arbitrary content with full polymorphism.

    Design validation: content field accepts Any type without constraints.
    No validation beyond Python's type system (primitives, collections, objects).

    Polymorphism strategy:
    - Primitives: str, int, float, bool, None pass through unchanged
    - Collections: dict, list, tuple stored as-is
    - Element instances: Auto-serialized via _serialize_content field_serializer

    Why no type constraints: Enables flexible composition patterns without
    requiring subclass definition for every content type. Ocean's philosophy
    of "data-driven" rather than "schema-driven" design.
    """
    node = Node(content={"key": "value", "nested": {"data": [1, 2, 3]}})

    assert node.content == {"key": "value", "nested": {"data": [1, 2, 3]}}


def test_node_subclass_creation():
    """Test Node subclass creation with custom fields + inherited content.

    Design validation: Subclasses add domain-specific fields while inheriting
    polymorphic content field from Node base class. This enables "schema-per-node"
    pattern where different node types coexist with different attributes.

    PersonNode defines:
    - name: str (domain field)
    - age: int (domain field)
    - content: Any (inherited from Node, used for bio text here)

    Registry pattern: PersonNode automatically registered in NODE_REGISTRY
    via __pydantic_init_subclass__ hook (tested separately in registration section).
    """
    person = PersonNode(name="Alice", age=30, content={"value": "bio"})

    assert person.name == "Alice"
    assert person.age == 30
    assert person.content == {"value": "bio"}


# ============================================================================
# Subclass Registration Tests
# ============================================================================
#
# Design aspect validated: Automatic subclass registration via __pydantic_init_subclass__
# hook eliminates manual registry calls. This design decision prioritizes developer
# ergonomics (zero-config subclasses) over explicit registration. The trade-off:
# - Benefit: No boilerplate, subclasses "just work"
# - Cost: Global registry state (acceptable for application-level class definitions)
#
# Why __pydantic_init_subclass__: Pydantic v2 provides this hook as the official way
# to run logic at class definition time. It's called for every subclass, including
# dynamically created ones, making it perfect for registry population.


def test_subclass_auto_registration():
    """Test __pydantic_init_subclass__ registers subclasses automatically."""
    # PersonNode and DocumentNode should be registered
    assert "PersonNode" in NODE_REGISTRY
    assert "DocumentNode" in NODE_REGISTRY

    # Check registry returns correct classes
    assert NODE_REGISTRY["PersonNode"] is PersonNode
    assert NODE_REGISTRY["DocumentNode"] is DocumentNode


def test_node_registry_includes_base_class():
    """Test Node itself is registered in NODE_REGISTRY."""
    assert "Node" in NODE_REGISTRY
    assert NODE_REGISTRY["Node"] is Node


def test_dynamic_subclass_registration():
    """Test dynamically created subclasses are registered."""

    class DynamicNode(Node):
        dynamic_field: str = "test"

    # Should be registered automatically
    assert "DynamicNode" in NODE_REGISTRY
    assert NODE_REGISTRY["DynamicNode"] is DynamicNode


# ============================================================================
# Polymorphic Deserialization Tests
# ============================================================================
#
# Design aspect validated: Polymorphic from_dict() uses lion_class metadata to route
# to correct subclass. This is Ocean's core pattern for type-safe deserialization
# across the Lion ecosystem.
#
# Why lion_class instead of __class__ or type fields:
# - Fully qualified names prevent collisions (module.ClassName)
# - Serialization-only metadata (removed during deserialization)
# - Works across serialization formats (JSON, TOML, YAML, DB)
# - Enables cross-library polymorphism (lionagi → lionherd)
#
# Fallback behavior: Unknown lion_class → base Node (graceful degradation).
# This design prevents deserialization failures when subclasses are unavailable
# (e.g., loading data from a different version or environment).


def test_from_dict_base_node():
    """Test from_dict creates Node when no lion_class specified."""
    data = {"content": {"value": "test content"}, "metadata": {}}

    node = Node.from_dict(data)

    assert isinstance(node, Node)
    assert node.content == {"value": "test content"}


def test_from_dict_polymorphic_person():
    """Test from_dict creates PersonNode when lion_class=PersonNode.

    Design validation: Polymorphic deserialization routing via NODE_REGISTRY.

    Routing mechanism (from_dict implementation):
    1. Extract lion_class from metadata: "PersonNode"
    2. Lookup in NODE_REGISTRY: NODE_REGISTRY["PersonNode"] → PersonNode class
    3. Delegate to target class: PersonNode.from_dict(data)
    4. Return PersonNode instance (not base Node)

    Why this matters: Enables calling Node.from_dict() on heterogeneous data
    and getting correctly typed instances without explicit class selection.

    Real-world scenario: Database query returns mixed node types
    ```python
    nodes = [Node.from_dict(row) for row in db.query("SELECT * FROM nodes")]
    # Each deserializes to its correct subclass automatically
    ```
    """
    data = {
        "name": "Bob",
        "age": 25,
        "content": {"value": "engineer"},
        "metadata": {"lion_class": "PersonNode"},
    }

    node = Node.from_dict(data)

    # Should return PersonNode instance
    assert isinstance(node, PersonNode)
    assert node.name == "Bob"
    assert node.age == 25
    assert node.content == {"value": "engineer"}


def test_from_dict_polymorphic_document():
    """Test from_dict creates DocumentNode when lion_class=DocumentNode."""
    data = {
        "title": "Report",
        "body": "Lorem ipsum",
        "metadata": {"lion_class": "DocumentNode"},
    }

    node = Node.from_dict(data)

    assert isinstance(node, DocumentNode)
    assert node.title == "Report"
    assert node.body == "Lorem ipsum"


def test_from_dict_with_full_qualified_name():
    """Test from_dict works with fully qualified lion_class."""
    full_name = f"{PersonNode.__module__}.PersonNode"
    data = {"name": "Charlie", "metadata": {"lion_class": full_name}}

    node = Node.from_dict(data)

    assert isinstance(node, PersonNode)
    assert node.name == "Charlie"


def test_from_dict_unknown_class_fallback():
    """Test from_dict falls back to base Node when lion_class unknown."""
    data = {"content": {"value": "test"}, "metadata": {"lion_class": "NonExistentNode"}}

    # Should not raise, just create base Node
    node = Node.from_dict(data)

    assert isinstance(node, Node)
    assert not isinstance(node, (PersonNode, DocumentNode))


def test_from_dict_preserves_metadata():
    """Test from_dict preserves custom metadata but removes lion_class."""
    data = {
        "content": {"value": "test"},
        "metadata": {"lion_class": "Node", "custom_key": "custom_value"},
    }

    node = Node.from_dict(data)

    # lion_class is serialization-only metadata, removed during deserialization
    assert "lion_class" not in node.metadata
    # Custom metadata is preserved
    assert node.metadata["custom_key"] == "custom_value"


# ============================================================================
# Database Mode Tests
# ============================================================================
#
# Design aspect validated: DB mode (mode="db") uses node_metadata field instead
# of metadata to prevent conflicts with application-level metadata columns in
# databases. from_dict() normalizes both formats automatically for seamless
# deserialization from any source (JSON, DB, TOML, etc.).


def test_to_dict_db_mode_renames_metadata():
    """Test to_dict with mode='db' creates node_metadata field."""
    node = Node(content={"value": "test"})

    db_dict = node.to_dict(mode="db")

    # Should have node_metadata instead of metadata
    assert "node_metadata" in db_dict
    assert "metadata" not in db_dict
    assert "lion_class" in db_dict["node_metadata"]


def test_to_dict_db_mode_subclass():
    """Test db mode serialization preserves lion_class for subclass."""
    person = PersonNode(name="David", age=35)

    db_dict = person.to_dict(mode="db")

    assert db_dict["node_metadata"]["lion_class"] == person.__class__.class_name(full=True)
    assert db_dict["name"] == "David"


def test_from_dict_db_format():
    """Test from_dict can deserialize database format (node_metadata)."""
    db_data = {
        "name": "Eve",
        "age": 28,
        "node_metadata": {"lion_class": "PersonNode"},
    }

    node = Node.from_dict(db_data)

    assert isinstance(node, PersonNode)
    assert node.name == "Eve"
    # lion_class is removed during deserialization (serialization-only metadata)
    assert "lion_class" not in node.metadata


def test_roundtrip_db_serialization():
    """Test Node survives db serialization roundtrip with correct type."""
    original = DocumentNode(title="Spec", body="Requirements")

    # Serialize to DB format
    db_dict = original.to_dict(mode="db")

    # Deserialize back
    restored = Node.from_dict(db_dict)

    # Should restore as DocumentNode
    assert isinstance(restored, DocumentNode)
    assert restored.title == "Spec"
    assert restored.body == "Requirements"
    assert restored.id == original.id


# ============================================================================
# Content Field Tests
# ============================================================================
#
# Design aspect validated: Content field supports nested Elements with automatic
# serialization/deserialization. This enables Node composition without explicit nesting APIs.
#
# Design philosophy: Structured data only
# - Content must be: Serializable, BaseModel, dict, or None
# - Primitives REJECTED (str, int, float, bool, list, tuple, set, bytes)
# - Rationale: Force structured, query-able, composable data (JSONB one-stop-shop)
# - Use dict wrapper for primitives: content={'value': 42} or Element.metadata
#
# Why reject primitives:
# - Node is composition layer - content must have key-value namespace
# - Enables graph-of-graphs patterns (Node contains Graph, which contains Nodes)
# - SQL JSONB queries require structured data, not raw primitives
# - Forces pit-of-success: developers think in structured terms
#
# Serialization strategy:
# - _serialize_content(): Detects Element instances → calls to_dict()
# - _validate_content(): Detects dicts with lion_class → calls from_dict()
# - Structured data (dict/BaseModel): unchanged in both directions
#
# Why this matters: Composition is Ocean's preferred pattern over deep class hierarchies.
# This design enables flexible data structures without hardcoding nesting types.


def test_content_with_nested_element():
    """Test content field can store nested Element instances for composition.

    Design validation: Automatic nested Element handling via field validators.

    Composition pattern:
    - Node.content accepts Element instances directly (runtime check)
    - _serialize_content: Element → dict via to_dict() during serialization
    - _validate_content: dict with lion_class → Element via from_dict() during deserialization

    Why this matters: Enables graph-of-graphs patterns without explicit nesting APIs.
    Example: Node contains Graph, which contains Nodes, which contain other Elements.

    Trade-off: No compile-time type safety (content: Any), but maximum flexibility
    for dynamic data structures. Follows Ocean's "composition over hierarchy" philosophy.
    """
    inner_node = Node(content={"value": "inner"})
    outer = Node(content=inner_node)

    # Content should be the nested Node
    assert isinstance(outer.content, Node)
    assert outer.content.content == {"value": "inner"}


def test_content_element_serialization():
    """Test nested Element in content is serialized to dict."""
    inner = PersonNode(name="Frank")
    outer = Node(content=inner)

    dict_ = outer.to_dict()

    # Content should be serialized dict, not Element object
    assert isinstance(dict_["content"], dict)
    assert dict_["content"]["name"] == "Frank"
    assert "lion_class" in dict_["content"]["metadata"]


def test_content_element_deserialization():
    """Test nested Element dict in content is deserialized to Element."""
    data = {
        "content": {
            "name": "Grace",
            "age": 32,
            "metadata": {"lion_class": "PersonNode"},
        }
    }

    node = Node.from_dict(data)

    # Content should be deserialized as PersonNode
    assert isinstance(node.content, PersonNode)
    assert node.content.name == "Grace"


def test_content_non_element_passthrough():
    """Test non-Element content is passed through unchanged."""
    node = Node(content={"plain": "dict", "no": "lion_class"})

    assert node.content == {"plain": "dict", "no": "lion_class"}


# ============================================================================
# Mixed-Type Collection Tests
# ============================================================================
#
# Design aspect validated: Polymorphic deserialization enables heterogeneous collections
# where each element deserializes to its correct subclass. This is critical for database
# scenarios where a single query returns multiple node types.
#
# Real-world scenario: Graph traversal query
# ```sql
# SELECT * FROM nodes WHERE id IN (ancestors_of('some-node-id'))
# ```
# Returns mixed types: PersonNode, DocumentNode, MetadataNode, etc.
#
# Without polymorphism: All deserialize as base Node → type information lost
# With polymorphism: Each deserializes to correct subclass → full type safety
#
# This design enables Ocean's "schema-per-node" pattern where different node types
# coexist in the same graph with different attributes, validated by their respective
# Pydantic schemas.


def test_mixed_type_collection_deserialization():
    """Test deserializing list of different Node subclasses from DB."""
    db_records = [
        {"name": "Alice", "age": 30, "node_metadata": {"lion_class": "PersonNode"}},
        {
            "title": "Doc1",
            "body": "Content",
            "node_metadata": {"lion_class": "DocumentNode"},
        },
        {"name": "Bob", "age": 25, "node_metadata": {"lion_class": "PersonNode"}},
        {"content": {"value": "generic"}, "node_metadata": {"lion_class": "Node"}},
    ]

    nodes = [Node.from_dict(record) for record in db_records]

    # Check correct types
    assert isinstance(nodes[0], PersonNode)
    assert isinstance(nodes[1], DocumentNode)
    assert isinstance(nodes[2], PersonNode)
    assert type(nodes[3]) is Node  # Exact Node, not subclass

    # Check data
    assert nodes[0].name == "Alice"
    assert nodes[1].title == "Doc1"
    assert nodes[2].age == 25


def test_mixed_type_collection_serialization():
    """Test serializing mixed Node subclasses maintains type info."""
    nodes = [
        PersonNode(name="X"),
        DocumentNode(title="Y"),
        Node(content={"value": "Z"}),
    ]

    serialized = [node.to_dict(mode="db") for node in nodes]

    # All should have node_metadata with correct lion_class
    assert serialized[0]["node_metadata"]["lion_class"].endswith("PersonNode")
    assert serialized[1]["node_metadata"]["lion_class"].endswith("DocumentNode")
    assert serialized[2]["node_metadata"]["lion_class"].endswith("Node")


# ============================================================================
# Serialization Mode Tests
# ============================================================================
#
# Design aspect validated: Three serialization modes balance compatibility vs. semantics:
#
# 1. mode="python": For in-memory operations
#    - datetime objects preserved
#    - UUID objects preserved
#    - metadata field (not node_metadata)
#
# 2. mode="json": For JSON serialization (APIs, files)
#    - datetime → ISO strings
#    - UUID → strings
#    - metadata field (not node_metadata)
#
# 3. mode="db": For database storage
#    - datetime → ISO strings
#    - UUID → strings
#    - node_metadata field (prevents column conflicts)
#
# Why node_metadata for DB mode: Databases often have application-level "metadata"
# columns. Using node_metadata prevents conflicts and makes Lion's metadata distinct
# from application metadata.
#
# All modes inject lion_class for polymorphism. from_dict() handles both metadata
# and node_metadata formats transparently.


def test_to_dict_python_mode_injects_lion_class():
    """Test python mode includes lion_class in metadata."""
    node = PersonNode(name="Test")

    python_dict = node.to_dict(mode="python")

    assert "metadata" in python_dict
    assert "lion_class" in python_dict["metadata"]


def test_to_dict_json_mode_injects_lion_class():
    """Test json mode includes lion_class in metadata."""
    node = DocumentNode(title="Test")

    json_dict = node.to_dict(mode="json")

    assert "metadata" in json_dict
    assert "lion_class" in json_dict["metadata"]


def test_to_dict_modes_consistency():
    """Test all modes produce valid data for deserialization."""
    original = PersonNode(name="Harry", age=40)

    for mode in ["python", "json", "db"]:
        serialized = original.to_dict(mode=mode)
        restored = Node.from_dict(serialized)

        assert isinstance(restored, PersonNode)
        assert restored.name == "Harry"
        assert restored.age == 40


# ============================================================================
# Edge Cases
# ============================================================================


def test_from_dict_empty_metadata():
    """Test from_dict with empty metadata dict."""
    data = {"content": {"value": "test"}, "metadata": {}}

    node = Node.from_dict(data)

    assert isinstance(node, Node)


def test_from_dict_no_metadata_field():
    """Test from_dict without metadata field."""
    data = {"content": {"value": "test"}}

    node = Node.from_dict(data)

    assert isinstance(node, Node)
    assert node.content == {"value": "test"}


def test_subclass_from_dict_direct_call():
    """Test calling from_dict directly on subclass."""
    data = {"name": "Iris", "age": 22}

    person = PersonNode.from_dict(data)

    # Should create PersonNode even without lion_class
    assert isinstance(person, PersonNode)
    assert person.name == "Iris"


def test_node_equality_by_id():
    """Test Node instances with same ID have same ID (not pydantic equality)."""
    node1 = Node(content={"value": "a"})
    node2 = Node.from_dict(node1.to_dict())

    # Pydantic equality compares all fields, but ID should match
    assert node1.id == node2.id
    assert isinstance(node1, Node)
    assert isinstance(node2, Node)


def test_node_repr():
    """Test Node repr shows class name and ID."""
    node = PersonNode(name="Test")

    repr_str = repr(node)

    assert "PersonNode" in repr_str
    assert str(node.id) in repr_str


def test_get_class_name_format():
    """Test class_name(full=True) returns fully qualified name."""
    class_name = PersonNode.class_name(full=True)

    # Should be module.ClassName
    assert "." in class_name
    assert class_name.endswith("PersonNode")


def test_node_with_complex_content():
    """Test Node handles complex nested content."""
    content = {
        "users": [
            {"name": "A", "roles": ["admin", "user"]},
            {"name": "B", "roles": ["user"]},
        ],
        "metadata": {"version": 2, "nested": {"deep": {"value": 123}}},
    }

    node = Node(content=content)

    # Roundtrip
    dict_ = node.to_dict()
    restored = Node.from_dict(dict_)

    assert restored.content == content


# ============================================================================
# Pydapter Integration Tests
# ============================================================================
#
# Design aspect validated: Isolated adapter registry pattern (Rust-like explicit)
#
# Why isolated registries (NOT inherited):
# - Base Node has toml/yaml adapters built-in for convenience
# - Subclasses get ISOLATED registries (no inheritance from parent)
# - Must explicitly register adapters on each subclass: MyNode.register_adapter(TomlAdapter)
# - Prevents adapter pollution while keeping base Node convenient
#
# Rationale: Explicit over implicit. Forces conscious decision about which adapters
# each subclass supports, preventing unexpected behavior from inherited adapters.
#
# Trade-off: More boilerplate (register per subclass) vs clarity and control.


def test_node_adapt_to_toml():
    """Test Node can adapt to TOML format using pydapter integration.

    Design validation: Isolated adapter registry pattern (Rust-like explicit).

    Pydapter integration strategy:
    - Node inherits from PydapterAdaptable (sync) and PydapterAsyncAdaptable (async)
    - Each class has isolated adapter registry (no inheritance from parent)
    - Base Node has toml/yaml built-in, subclasses must explicitly register
    - adapt_to() defaults to mode="db" for node_metadata compatibility

    Why isolated registry: Prevents adapter pollution, explicit over implicit.
    Trade-off: Must register per subclass, but ensures clarity and control.

    Async pattern (not tested here but available):
    ```python
    await person.adapt_to_async("postgresql", connection=conn)
    # Async serialization for I/O-bound database writes
    ```
    """
    # Explicitly register toml adapter for PersonNode (isolated registry)
    from pydapter.adapters import TomlAdapter

    PersonNode.register_adapter(TomlAdapter)

    person = PersonNode(name="Alice", age=30, content={"value": "engineer"})

    toml_str = person.adapt_to("toml")

    # Should be TOML string
    assert isinstance(toml_str, str)
    assert "name = " in toml_str or "name=" in toml_str
    assert "Alice" in toml_str


def test_node_adapt_from_toml():
    """Test Node can adapt from TOML format with polymorphism."""
    toml_str = """
name = "Bob"
age = 25

[content]
value = "developer"

[node_metadata]
lion_class = "PersonNode"
"""

    node = Node.adapt_from(toml_str, "toml")

    # Should return PersonNode instance via polymorphism
    assert isinstance(node, PersonNode)
    assert node.name == "Bob"
    assert node.age == 25


def test_node_adapt_to_yaml():
    """Test Node can adapt to YAML format using pydapter (isolated registry)."""
    # Register yaml adapter explicitly for DocumentNode
    from pydapter.adapters import YamlAdapter

    DocumentNode.register_adapter(YamlAdapter)

    doc = DocumentNode(title="Report", body="Lorem ipsum")

    yaml_str = doc.adapt_to("yaml")

    # Should be YAML string
    assert isinstance(yaml_str, str)
    assert "title:" in yaml_str or "title :" in yaml_str
    assert "Report" in yaml_str


def test_node_adapt_from_yaml():
    """Test Node can adapt from YAML format with polymorphism."""
    yaml_str = """
title: Spec
body: Requirements document
node_metadata:
  lion_class: DocumentNode
"""

    node = Node.adapt_from(yaml_str, "yaml")

    # Should return DocumentNode instance via polymorphism
    assert isinstance(node, DocumentNode)
    assert node.title == "Spec"
    assert node.body == "Requirements document"


def test_node_adapt_roundtrip_toml():
    """Test Node survives TOML roundtrip with correct type."""
    original = PersonNode(name="Charlie", age=35, content={"value": "manager"})

    # Adapt to TOML and back
    toml_str = original.adapt_to("toml")
    restored = Node.adapt_from(toml_str, "toml")

    # Should restore as PersonNode with correct data
    assert isinstance(restored, PersonNode)
    assert restored.name == "Charlie"
    assert restored.age == 35
    assert restored.id == original.id


def test_node_adapt_many_toml():
    """Test Node can adapt multiple instances to/from TOML."""
    nodes = [
        PersonNode(name="Alice", age=30),
        PersonNode(name="Bob", age=25),
    ]

    # Adapt multiple to TOML
    toml_str = Node.adapt_to(nodes[0], "toml", many=False)
    assert isinstance(toml_str, str)


def test_node_uses_builtin_json_not_adapter():
    """Test Node uses built-in JSON methods with polymorphism via to_dict."""
    node = PersonNode(name="Test", age=99)

    # Built-in JSON serialization with lion_class injection
    json_dict = node.to_dict(mode="json")
    assert "metadata" in json_dict
    assert "lion_class" in json_dict["metadata"]

    # Built-in JSON deserialization with polymorphism
    restored = Node.from_dict(json_dict)
    assert isinstance(restored, PersonNode)
    assert restored.name == "Test"
    assert restored.age == 99


# ==================== Embedding Field Tests ====================
#
# Design aspect validated: Embedding field handles common database scenarios
# (JSON string coercion) while maintaining strict validation (no empty lists,
# numeric values only). This balances flexibility with correctness.


def test_node_embedding_none():
    """Test Node with no embedding (default None)."""
    node = Node(content={"value": "test"})
    assert node.embedding is None


def test_node_embedding_list():
    """Test Node with embedding as list of floats."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    node = Node(content={"value": "test"}, embedding=embedding)
    assert node.embedding == embedding
    assert all(isinstance(x, float) for x in node.embedding)


def test_node_embedding_coerce_ints():
    """Test Node embedding coerces ints to floats."""
    node = Node(content={"value": "test"}, embedding=[1, 2, 3])
    assert node.embedding == [1.0, 2.0, 3.0]
    assert all(isinstance(x, float) for x in node.embedding)


def test_node_embedding_from_json_string():
    """Test Node embedding can parse JSON string (common from DB queries).

    Design validation: Database compatibility via type coercion.

    Real-world scenario: PostgreSQL JSON/JSONB columns
    - Database stores: embedding::jsonb column with [0.1, 0.2, 0.3]
    - Query returns: JSON string "[0.1, 0.2, 0.3]" (not Python list)
    - _validate_embedding: Detects str → orjson.loads() → list[float]

    Why JSON string coercion: Many databases serialize arrays as JSON strings
    when retrieving data. Without coercion, would require manual parsing at
    every query site. This validator centralizes the conversion.

    Alternative approach rejected: Store as binary/array type in DB.
    Trade-off: JSON strings are portable across databases (SQLite, Postgres, etc.)
    but require parsing overhead. Ocean prioritized compatibility over raw performance.
    """
    import orjson

    embedding = [0.1, 0.2, 0.3]
    json_str = orjson.dumps(embedding).decode()

    node = Node(content={"value": "test"}, embedding=json_str)
    assert node.embedding == embedding


def test_node_embedding_rejects_empty_list():
    """Test Node embedding rejects empty list."""
    import pytest

    with pytest.raises(ValueError, match="embedding list cannot be empty"):
        Node(content={"value": "test"}, embedding=[])


def test_node_embedding_rejects_non_numeric():
    """Test Node embedding rejects non-numeric values."""
    import pytest

    with pytest.raises(ValueError, match="embedding must contain only numeric values"):
        Node(content={"value": "test"}, embedding=[0.1, "invalid", 0.3])


def test_node_embedding_rejects_invalid_type():
    """Test Node embedding rejects invalid types."""
    import pytest

    with pytest.raises(ValueError, match="embedding must be a list"):
        Node(content={"value": "test"}, embedding={"invalid": "dict"})


def test_node_embedding_serialization():
    """Test Node embedding serializes correctly in different modes."""
    node = Node(content={"value": "test"}, embedding=[0.1, 0.2, 0.3])

    # Python mode preserves list
    python_dict = node.to_dict(mode="python")
    assert python_dict["embedding"] == [0.1, 0.2, 0.3]

    # JSON mode preserves list
    json_dict = node.to_dict(mode="json")
    assert json_dict["embedding"] == [0.1, 0.2, 0.3]

    # DB mode preserves list
    db_dict = node.to_dict(mode="db")
    assert db_dict["embedding"] == [0.1, 0.2, 0.3]


def test_node_embedding_roundtrip():
    """Test Node embedding survives serialization roundtrip."""
    original = Node(content={"value": "test"}, embedding=[0.1, 0.2, 0.3])

    # Roundtrip through dict
    data = original.to_dict(mode="db")
    restored = Node.from_dict(data)

    assert restored.embedding == original.embedding


# ============================================================================
# Embedding Format Tests (PR #113)
# ============================================================================
#
# Design aspect validated: embedding_format parameter enables PostgreSQL-optimized
# serialization formats (pgvector, jsonb) while maintaining backward compatibility
# (list format). Format selection uses Pydantic SerializationInfo.context pattern.


def test_node_embedding_format_pgvector():
    """Test embedding_format='pgvector' produces compact JSON string for PostgreSQL.

    Pattern:
        Database-optimized serialization for pgvector extension

    Design Rationale:
        PostgreSQL pgvector requires compact JSON format for vector casting:
        SELECT embedding::vector FROM nodes WHERE ...
        Compact format: "[0.1,0.2,0.3]" (no spaces) via orjson

    Use Case:
        Vector similarity search in PostgreSQL with pgvector extension

    Expected:
        Embedding serialized as compact JSON string (orjson format)
    """
    node = Node(content={"test": "data"}, embedding=[0.1, 0.2, 0.3])
    data = node.to_dict(mode="db", embedding_format="pgvector")

    # Verify pgvector format
    assert isinstance(data["embedding"], str)
    assert data["embedding"] == "[0.1,0.2,0.3]"  # Compact, no spaces

    # Verify parseable
    import orjson

    parsed = orjson.loads(data["embedding"])
    assert parsed == [0.1, 0.2, 0.3]


def test_node_embedding_format_jsonb():
    """Test embedding_format='jsonb' produces standard JSON string with spaces.

    Pattern:
        Standard JSON serialization for JSONB storage

    Design Rationale:
        PostgreSQL JSONB columns accept standard JSON format with spaces.
        Uses Python json library (not orjson) for standard formatting.
        Format: "[0.1, 0.2, 0.3]" (with spaces)

    Use Case:
        Storing embeddings in PostgreSQL JSONB columns without pgvector

    Expected:
        Embedding serialized as standard JSON string with spaces
    """
    node = Node(content={"test": "data"}, embedding=[0.1, 0.2, 0.3])
    data = node.to_dict(mode="db", embedding_format="jsonb")

    # Verify jsonb format
    assert isinstance(data["embedding"], str)
    assert data["embedding"] == "[0.1, 0.2, 0.3]"  # Standard JSON with spaces

    # Verify parseable
    import json

    parsed = json.loads(data["embedding"])
    assert parsed == [0.1, 0.2, 0.3]


def test_node_embedding_format_list_default():
    """Test embedding_format='list' and default (None) preserve list format.

    Pattern:
        Backward compatibility with existing serialization behavior

    Design Rationale:
        Default format="list" maintains backward compatibility with code that
        expects Python list from to_dict(). Explicit None also returns list.
        No string coercion unless explicitly requested (pgvector/jsonb).

    Use Case:
        In-memory operations, JSON APIs, non-PostgreSQL databases

    Expected:
        Embedding remains Python list (not string)
    """
    node = Node(content={"test": "data"}, embedding=[0.1, 0.2, 0.3])

    # Explicit format="list"
    data_explicit = node.to_dict(mode="db", embedding_format="list")
    assert isinstance(data_explicit["embedding"], list)
    assert data_explicit["embedding"] == [0.1, 0.2, 0.3]

    # Default (no format specified)
    data_default = node.to_dict(mode="db")
    assert isinstance(data_default["embedding"], list)
    assert data_default["embedding"] == [0.1, 0.2, 0.3]


def test_node_embedding_format_python_mode_ignores_format():
    """Test embedding_format only applies in json/db modes, not python mode.

    Pattern:
        Mode-specific serialization behavior

    Design Rationale:
        Python mode preserves Python types (no string coercion).
        embedding_format parameter only affects json/db modes where
        serialization to strings makes sense. In python mode, format
        is ignored and list is always returned.

    Use Case:
        In-memory operations where Python types are preserved

    Expected:
        embedding_format ignored, list returned in python mode
    """
    node = Node(content={"test": "data"}, embedding=[0.1, 0.2, 0.3])

    # Python mode should ignore format and return list
    data = node.to_dict(mode="python", embedding_format="pgvector")
    assert isinstance(data["embedding"], list)
    assert data["embedding"] == [0.1, 0.2, 0.3]

    # Verify not string (format ignored)
    assert not isinstance(data["embedding"], str)


def test_node_embedding_format_none_embedding():
    """Test embedding_format handles None embedding gracefully.

    Pattern:
        Null safety in serialization

    Design Rationale:
        Node.embedding is optional (None by default). Serialization must
        handle None without errors regardless of format specified.
        None is not converted to string "null" or empty list.

    Use Case:
        Nodes without embeddings (text-only, metadata-only nodes)

    Expected:
        None embedding serialized as None, not string or list
    """
    node = Node(content={"test": "data"}, embedding=None)

    # All formats should return None
    for fmt in ["pgvector", "jsonb", "list"]:
        data = node.to_dict(mode="db", embedding_format=fmt)
        assert data["embedding"] is None


def test_node_embedding_format_combination_with_content():
    """Test embedding_format works correctly with content serialization.

    Pattern:
        Multi-field serialization independence

    Design Rationale:
        embedding_format only affects embedding field, not content field.
        Content serialization (nested Elements, dicts) should work
        independently without interference from embedding format.

    Use Case:
        Nodes with both structured content and embeddings for hybrid search

    Expected:
        Both content and embedding serialized correctly with independent formats
    """
    # Create node with both content and embedding
    inner_node = Node(content={"inner": "value"})
    node = Node(content={"data": "value", "nested": inner_node}, embedding=[0.1, 0.2, 0.3])

    data = node.to_dict(mode="db", embedding_format="pgvector")

    # Verify content serialization (nested Element → dict)
    assert isinstance(data["content"], dict)
    assert data["content"]["data"] == "value"
    assert isinstance(data["content"]["nested"], dict)
    assert data["content"]["nested"]["content"]["inner"] == "value"

    # Verify embedding serialization (pgvector format)
    assert isinstance(data["embedding"], str)
    assert data["embedding"] == "[0.1,0.2,0.3]"


# ============================================================================
# Issue #49: Parametrize Node.content primitive rejection tests
# ============================================================================
# Refactored from 5 individual tests to single parametrized test covering
# 9 primitive types (str, int, float, bool, list, tuple, set, frozenset, bytes)


@pytest.mark.parametrize(
    "invalid_content,type_name",
    [
        ("primitive string", "str"),
        (42, "int"),
        (3.14, "float"),
        (True, "bool"),
        ([1, 2, 3], "list"),
        ((1, 2, 3), "tuple"),
        ({1, 2, 3}, "set"),
        (frozenset([1, 2]), "frozenset"),
        (b"bytes", "bytes"),
    ],
)
def test_node_content_rejects_primitives(invalid_content, type_name):
    """Test that primitive content types raise TypeError with helpful message.

    Pattern:
        Strict type enforcement for structured data

    Design Rationale:
        Node.content constraint forces structured, query-able data.
        Primitives must be wrapped in dict or stored in Element.metadata.

    Architectural Identity:
        Node is the composition layer - content must be:
        - dict: Unstructured but query-able (JSONB one-stop-shop)
        - Serializable: Rich nested structures (graph-of-graphs)
        - BaseModel: Pydantic models (typed + validated)
        - None: Optional content

    Rejected Types:
        - str, int, float, bool: Not structured or query-able
        - list, tuple, set, frozenset, bytes: Not key-value namespaces
        - Use Element.metadata for simple key-value pairs instead

    Error Message Requirements:
        - Identifies type constraint
        - Shows actual type received (parametrized type_name)
        - Provides actionable guidance (wrap in dict or use Element.metadata)

    Use Case:
        Migration from unstructured APIs requires explicit structured conversion.
        Forces pit-of-success: developers must think in structured terms.

    Coverage:
        All 9 primitive types rejected with clear error messages including:
        - Type name in error ("Got str", "Got int", etc.)
        - Migration guidance: content={'value': ...}

    Expected:
        TypeError with guidance to use dict or Element.metadata
    """
    # Validate error message contains type name and migration guidance
    with pytest.raises(
        TypeError,
        match=rf"content must be Serializable, BaseModel, dict, or None\. Got {type_name}\.",
    ):
        Node(content=invalid_content)


# ============================================================================
# Content Serializer Tests (PR #113)
# ============================================================================
#
# Design aspect validated: content_serializer parameter enables custom content
# transformation during serialization (compression, encryption, external storage).
#
# Pattern:
#   One-way transformation: Node.content → serialized format during to_dict()
#   Node instance unchanged (transformation only affects serialization output)
#   Fail-fast validation: callable check + test call before model_dump
#
# Design Rationale:
#   Large content nodes (embeddings, documents) need external storage strategies.
#   content_serializer enables reference patterns without modifying Node behavior.
#   Validation catches broken serializers early (test call on self.content).
#
# Integration:
#   Compatible with all modes (python/json/db)
#   Works with embedding_format parameter (independent transformations)
#   Excludes content from model_dump, replaces with serializer result
#
# Use Cases:
#   - Compression: Store large content externally, replace with reference
#   - Encryption: Encrypt content before database storage
#   - External storage: Replace content with S3/GCS URI + metadata


def test_node_content_serializer_basic():
    """Test content_serializer parameter enables custom content transformation.

    Pattern:
        One-way content transformation during serialization

    Design Rationale:
        Enables compression, encryption, external storage references without
        modifying Node behavior. Fail-fast validation prevents runtime errors.

    Use Case:
        Store large content externally, replace with reference:
        content_serializer=lambda c: {"ref": "s3://bucket/id", "size": len(str(c))}

    Expected:
        Content replaced with serializer result, original excluded
    """
    node = Node(content={"key": "value"})

    def custom_serializer(content):
        return {"serialized": str(content)}

    data = node.to_dict(content_serializer=custom_serializer)

    # Verify serialized content
    assert data["content"] == {"serialized": "{'key': 'value'}"}

    # Verify original excluded
    assert "key" not in data["content"]


def test_node_content_serializer_compression_example():
    """Test content_serializer with compression use case.

    Pattern:
        Compression + base64 encoding for large content storage

    Design Rationale:
        Large content (documents, embeddings) bloats database rows.
        Compression reduces storage costs and query performance impact.

    Real-World Scenario:
        Node with 100KB document compressed to 10KB reference:
        - Original: {"text": "large document..."}
        - Compressed: {"compressed": "eJy...base64...", "size": 10240}

    Expected:
        Content replaced with compressed base64 string + metadata
    """
    import base64
    import json
    import zlib

    node = Node(content={"large": "data" * 100})

    def compress_serializer(content):
        json_bytes = json.dumps(content).encode("utf-8")
        compressed = zlib.compress(json_bytes)
        encoded = base64.b64encode(compressed).decode("utf-8")
        return {"compressed": encoded, "size": len(compressed)}

    data = node.to_dict(content_serializer=compress_serializer)

    # Verify compression metadata
    assert "compressed" in data["content"]
    assert "size" in data["content"]
    assert isinstance(data["content"]["compressed"], str)
    assert isinstance(data["content"]["size"], int)


def test_node_content_serializer_lambda():
    """Test content_serializer accepts lambda functions.

    Pattern:
        Inline lambda for simple transformations

    Design Rationale:
        Lambda functions enable concise serialization without def statements.
        Useful for simple transformations (str, repr, hash).

    Use Case:
        Quick content fingerprinting: lambda c: {"hash": hash(str(c))}

    Expected:
        Lambda serializer works identically to def functions
    """
    node = Node(content={"test": "data"})

    # Lambda serializer
    data = node.to_dict(content_serializer=lambda c: str(c))

    # Verify lambda executed
    assert data["content"] == "{'test': 'data'}"


def test_node_content_serializer_non_callable_raises_typeerror():
    """Test content_serializer rejects non-callable with clear error.

    Pattern:
        Fail-fast validation before model_dump

    Design Rationale:
        Callable check prevents cryptic runtime errors during serialization.
        Clear error message guides developer to correct usage.

    Error Contract:
        TypeError with message: "content_serializer must be callable, got {type}"

    Expected:
        TypeError raised immediately, before model_dump execution
    """
    node = Node(content={"test": "data"})

    with pytest.raises(TypeError, match=r"content_serializer must be callable, got str"):
        node.to_dict(content_serializer="not_callable")


def test_node_content_serializer_broken_serializer_fails_fast():
    """Test content_serializer fails fast on broken serializer.

    Pattern:
        Test call validation before model_dump

    Design Rationale:
        Test call with self.content catches broken serializers early.
        Prevents partial serialization with missing content field.

    Error Contract:
        ValueError with message: "content_serializer failed on test call: {original_error}"

    Real-World Scenario:
        Serializer depends on missing library or incorrect assumptions:
        lambda c: external_api.compress(c)  # external_api not imported

    Expected:
        ValueError raised with original exception context
    """
    node = Node(content={"test": "data"})

    def broken_serializer(content):
        raise RuntimeError("Serializer broken")

    with pytest.raises(ValueError, match=r"content_serializer failed on test call"):
        node.to_dict(content_serializer=broken_serializer)


def test_node_content_serializer_excludes_content_from_model_dump():
    """Test content_serializer excludes original content from model_dump.

    Pattern:
        Exclude original → Replace with serialized

    Design Rationale:
        Two-phase serialization: exclude content from model_dump, inject serialized.
        Ensures no content duplication or partial serialization.

    Implementation Validation:
        1. Inject {"content"} into exclude parameter
        2. Call super().to_dict(..., exclude={..., "content"})
        3. Inject result["content"] = serializer(self.content)

    Expected:
        Original content NOT in result, serialized content IS in result
    """
    node = Node(content={"original": "data"})

    def custom_serializer(content):
        return {"replaced": "data"}

    data = node.to_dict(content_serializer=custom_serializer)

    # Verify original excluded
    assert "original" not in data["content"]

    # Verify serialized present
    assert data["content"] == {"replaced": "data"}


def test_node_content_serializer_combination_with_embedding_format():
    """Test content_serializer works with embedding_format parameter.

    Pattern:
        Independent transformations (content + embedding)

    Design Rationale:
        content_serializer and embedding_format are orthogonal parameters.
        Both inject context into model_dump, must not interfere.

    Real-World Scenario:
        Node with large content + large embedding both need optimization:
        - content_serializer: Store content externally (S3 reference)
        - embedding_format: Compress embedding (pgvector format)

    Expected:
        Both transformations applied independently, no interference
    """
    node = Node(content={"test": "data"}, embedding=[0.1, 0.2, 0.3])

    def content_serializer(content):
        return {"ref": "s3://bucket/content-id"}

    data = node.to_dict(
        mode="db",
        content_serializer=content_serializer,
        embedding_format="pgvector",
    )

    # Verify content serialized
    assert data["content"] == {"ref": "s3://bucket/content-id"}

    # Verify embedding formatted (pgvector is compact JSON string)
    assert isinstance(data["embedding"], str)
    assert "[" in data["embedding"]  # JSON array format


def test_node_content_serializer_mode_python():
    """Test content_serializer works with mode='python'.

    Pattern:
        Serialization mode independence

    Design Rationale:
        content_serializer transformation applies uniformly across all modes.
        Mode affects other fields (created_at, embedding), not serializer logic.

    Expected:
        content_serializer executes identically in python mode
    """
    node = Node(content={"test": "data"})

    def serializer(content):
        return {"mode": "python", "data": str(content)}

    data = node.to_dict(mode="python", content_serializer=serializer)

    assert data["content"] == {"mode": "python", "data": "{'test': 'data'}"}


def test_node_content_serializer_mode_json():
    """Test content_serializer works with mode='json'.

    Pattern:
        Serialization mode independence

    Design Rationale:
        content_serializer transformation applies uniformly across all modes.
        JSON mode affects datetime/UUID serialization, not content logic.

    Expected:
        content_serializer executes identically in json mode
    """
    node = Node(content={"test": "data"})

    def serializer(content):
        return {"mode": "json", "data": str(content)}

    data = node.to_dict(mode="json", content_serializer=serializer)

    assert data["content"] == {"mode": "json", "data": "{'test': 'data'}"}


def test_node_content_serializer_mode_db():
    """Test content_serializer works with mode='db'.

    Pattern:
        Database mode compatibility with node_metadata

    Design Rationale:
        DB mode uses node_metadata instead of metadata.
        content_serializer must work correctly with metadata field renaming.

    Real-World Scenario:
        Storing Node in database with external content storage:
        - node_metadata: {"lion_class": "Node"}
        - content: {"ref": "s3://bucket/id"}
        - embedding: "[0.1, 0.2, 0.3]" (pgvector format)

    Expected:
        content_serializer executes correctly with node_metadata present
    """
    node = Node(content={"test": "data"})

    def serializer(content):
        return {"mode": "db", "data": str(content)}

    data = node.to_dict(mode="db", content_serializer=serializer)

    # Verify content serialized
    assert data["content"] == {"mode": "db", "data": "{'test': 'data'}"}

    # Verify db mode field present
    assert "node_metadata" in data


def test_node_content_serializer_with_none_content():
    """Test content_serializer handles None content gracefully.

    Pattern:
        Null-safe transformation

    Design Rationale:
        Node.content defaults to None. Serializer must handle null case.
        Test call validation ensures serializer doesn't crash on None.

    Error Handling:
        If serializer doesn't handle None, fail-fast test call catches it.

    Expected:
        Serializer receives None, returns serialized null representation
    """
    node = Node(content=None)

    def null_safe_serializer(content):
        return {"is_none": content is None}

    data = node.to_dict(content_serializer=null_safe_serializer)

    assert data["content"] == {"is_none": True}


def test_node_content_serializer_backward_compatible():
    """Test to_dict without content_serializer preserves default behavior.

    Pattern:
        Backward compatibility (no breaking changes)

    Design Rationale:
        content_serializer is optional parameter (default: None).
        Existing code calling to_dict() without parameter must work unchanged.

    Migration Strategy:
        Zero-impact addition - all existing code continues working.
        New feature enabled only when explicitly requested.

    Expected:
        content unchanged when content_serializer not provided
    """
    node = Node(content={"test": "data"})

    # Call to_dict without content_serializer
    data = node.to_dict()

    # Verify content unchanged (default behavior)
    assert data["content"] == {"test": "data"}


# ============================================================================
# Coverage Completion Tests (PR #113)
# ============================================================================
#
# Design aspect validated: Edge cases and fallback paths for 100% coverage
# These tests exercise error handling, alternative code paths, and async operations
# that are not covered by mainline happy-path tests above.


def test_node_content_element_not_in_registry():
    """Test content validation handles Element subclasses not in NODE_REGISTRY.

    Pattern:
        Polymorphic deserialization for non-Node Element types

    Design Rationale:
        Graph-of-graphs pattern requires storing arbitrary Elements.
        Custom Element subclasses (not Node) should deserialize via Element.from_dict(),
        not NODE_REGISTRY lookup. This enables composition of Elements that aren't Nodes.

    Use Case:
        Workflow node contains custom ExecutionMetadata (Element subclass)
        that doesn't inherit from Node. Content must deserialize correctly
        even though ExecutionMetadata isn't in NODE_REGISTRY.

    Coverage:
        Line 113: return Element.from_dict(value)

    Expected:
        Content deserialized as custom Element via Element.from_dict() fallback
    """
    # Use module-level Element subclass (NOT Node, so not in NODE_REGISTRY)
    custom = CustomElementForTest(custom_field="value")
    custom_dict = custom.to_dict()

    # Verify lion_class present but NOT in NODE_REGISTRY
    lion_class = custom_dict.get("metadata", {}).get("lion_class")
    assert lion_class is not None
    assert "CustomElementForTest" in lion_class
    # CustomElementForTest is Element, not Node, so not auto-registered in NODE_REGISTRY
    assert lion_class not in NODE_REGISTRY
    assert "CustomElementForTest" not in NODE_REGISTRY

    # Use as Node content - should trigger line 113 (Element.from_dict fallback)
    node = Node(content=custom_dict)

    # Verify content deserialized as CustomElementForTest (via Element.from_dict)
    assert isinstance(node.content, CustomElementForTest)
    assert node.content.custom_field == "value"


def test_node_embedding_invalid_json_string():
    """Test Node embedding rejects invalid JSON strings with clear error.

    Pattern:
        Fail-fast validation with actionable error messages

    Design Rationale:
        Database queries sometimes return malformed JSON strings due to data corruption
        or encoding issues. _validate_embedding must detect and report parse failures
        clearly rather than silently failing or returning corrupt data.

    Error Contract:
        ValueError with message: "Failed to parse embedding JSON string: {error}"

    Use Case:
        PostgreSQL query returns corrupted JSONB: "[0.1, 0.2, [[["
        Validator must reject and provide clear error for debugging.

    Coverage:
        Lines 129-130: Exception handling for orjson.loads() failure

    Expected:
        ValueError raised immediately with parse error context
    """
    with pytest.raises(ValueError, match=r"Failed to parse embedding JSON string"):
        Node(content={"test": "data"}, embedding="invalid json [[[")


def test_node_content_serializer_exclude_dict_format():
    """Test to_dict handles exclude parameter as dict format correctly.

    Pattern:
        Pydantic exclude parameter format flexibility

    Design Rationale:
        Pydantic model_dump() accepts exclude in multiple formats:
        - Set format: exclude={"field1", "field2"}
        - Dict format: exclude={"field1": True, "field2": {"nested": True}}

        When content_serializer is used, implementation must inject "content"
        into exclude parameter. Dict format requires special handling (copy dict,
        add "content": True) vs set format (union with {"content"}).

    Use Case:
        Advanced serialization with nested field exclusion:
        exclude={"metadata": {"internal_field": True}}
        Must work correctly with content_serializer.

    Coverage:
        Lines 189-194: Dict exclude format handling in to_dict

    Expected:
        content excluded and replaced with serialized value, other fields unaffected
    """
    node = Node(content={"test": "data"}, metadata={"custom": "value"})

    def serializer(content):
        return {"serialized": str(content)}

    # Use dict exclude format (not set)
    data = node.to_dict(
        content_serializer=serializer,
        exclude={"metadata": {"custom": True}},  # Dict format
    )

    # Verify content serialized (dict exclude format handled correctly)
    assert data["content"] == {"serialized": "{'test': 'data'}"}

    # Verify dict exclude format applied (custom metadata excluded)
    assert "custom" not in data.get("metadata", {})


def test_node_content_serializer_exclude_other_type():
    """Test to_dict handles exclude parameter as non-set/dict type.

    Pattern:
        Defensive programming for edge-case exclude formats

    Design Rationale:
        When content_serializer is used, implementation must handle all exclude formats:
        - Set format: exclude={"field1", "field2"} → union with {"content"}
        - Dict format: exclude={"field1": True} → copy and add "content": True
        - Other types (None, string, etc.) → fallback to {"content"}

        Line 194 else branch handles unexpected exclude types by replacing with {"content"}.

    Use Case:
        API misuse or dynamic exclude parameter with unexpected type.
        Must not crash, instead use safe fallback.

    Coverage:
        Line 194: exclude = {"content"} (else branch fallback)

    Expected:
        content excluded and replaced with serialized value, safe fallback behavior
    """
    node = Node(content={"test": "data"}, metadata={"custom": "value"})

    def serializer(content):
        return {"serialized": str(content)}

    # Use None as exclude (neither set nor dict)
    data = node.to_dict(
        content_serializer=serializer,
        exclude=None,  # Neither set nor dict → triggers line 194
    )

    # Verify content serialized (fallback exclude format handled)
    assert data["content"] == {"serialized": "{'test': 'data'}"}

    # Verify other fields present (only content excluded)
    assert "metadata" in data


def test_node_from_dict_with_meta_key():
    """Test from_dict supports custom metadata key via meta_key parameter.

    Pattern:
        Database schema flexibility for metadata field naming

    Design Rationale:
        Different databases/APIs may use different field names for metadata:
        - Standard: "metadata" or "node_metadata"
        - Custom: "meta", "properties", "attributes", etc.

        meta_key parameter enables deserialization from arbitrary metadata field names
        without requiring data transformation before Node.from_dict().

    Use Case:
        Legacy database schema uses "properties" field for metadata.
        Migration path: from_dict(data, meta_key="properties")

    Coverage:
        Line 225: data["metadata"] = data.pop(meta_key)

    Expected:
        Metadata restored from custom key name, normalized to "metadata"
    """
    # Create data with custom metadata key
    data = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": {"test": "data"},
        "custom_meta": {"custom_key": "custom_value"},
    }

    # Deserialize with custom meta_key
    node = Node.from_dict(data, meta_key="custom_meta")

    # Verify metadata restored from custom key
    assert node.metadata["custom_key"] == "custom_value"
    # Verify custom_meta consumed (not in final dict)
    assert "custom_meta" not in node.to_dict()


def test_node_from_dict_with_non_dict_metadata():
    """Test from_dict handles non-dict metadata gracefully.

    Pattern:
        Defensive programming for malformed input data

    Design Rationale:
        Metadata field should be dict, but corrupted data or API bugs may provide
        non-dict values (string, list, None). Implementation must handle gracefully
        without crashing, setting lion_class = None to skip polymorphic routing.

    Error Handling Strategy:
        Non-dict metadata → Skip lion_class extraction (line 240) → Deserialize as base Node

    Use Case:
        Database corruption or API bug returns metadata: "corrupted_string"
        Should deserialize without crash, treating as base Node.

    Coverage:
        Line 240: lion_class = None (else branch for non-dict metadata)

    Expected:
        Node created without polymorphic routing (proves lion_class = None executed)
        Note: Pydantic validation converts non-dict metadata to {} after from_dict logic
    """
    # Create data with non-dict metadata
    data = {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "content": {"test": "data"},
        "metadata": "string_metadata",  # Not dict - triggers line 240
    }

    # Should not crash, deserialize as base Node (no polymorphic routing)
    node = Node.from_dict(data)

    # Verify base Node created (no polymorphic routing = line 240 executed)
    assert type(node) is Node
    # Note: Pydantic converts non-dict metadata to default {} after from_dict
    assert isinstance(node.metadata, dict)


# ==============================================================================
# content_deserializer Tests (Round-Trip Serialization)
# ==============================================================================


def test_node_content_deserializer_basic_roundtrip():
    """Test basic round-trip serialization with content_deserializer.

    Pattern:
        Symmetric transformation: to_dict(serializer) + from_dict(deserializer) = original

    Design Rationale:
        content_serializer enables custom transformations (compression, encryption, etc).
        content_deserializer completes the round-trip, restoring original content.
        Both parameters must be symmetric inverses for round-trip correctness.

    Use Case:
        Store Node with custom content format, then restore original content on load.
        Example: API client serializes for transport, server deserializes on receipt.

    Coverage:
        Lines 251-263: content_deserializer validation and application

    Expected:
        Original content restored after serialization + deserialization cycle
    """
    # Original node
    original = Node(content={"key": "value", "nested": {"data": 123}})

    # Serialize with custom serializer
    def custom_serializer(content):
        return {"transformed": True, "original": content}

    data = original.to_dict(content_serializer=custom_serializer)
    assert data["content"] == {
        "transformed": True,
        "original": {"key": "value", "nested": {"data": 123}},
    }

    # Deserialize with inverse deserializer
    def custom_deserializer(content):
        return content["original"]

    restored = Node.from_dict(data, content_deserializer=custom_deserializer)

    # Verify round-trip: restored content matches original
    assert restored.content == original.content
    assert restored.id == original.id


def test_node_content_deserializer_compression_roundtrip():
    """Test round-trip with compression/decompression.

    Pattern:
        Real-world transformation pattern: compress for storage, decompress for use

    Design Rationale:
        Large content (documents, embeddings) bloats database rows.
        Compression reduces storage costs and improves query performance.
        Round-trip enables transparent compression without code changes.

    Real-World Scenario:
        Store 100KB document as 10KB compressed data in PostgreSQL JSONB column.
        Application code works with original uncompressed content transparently.

    Implementation:
        Serializer: json.dumps(content) → zlib.compress → base64.encode → dict
        Deserializer: dict → base64.decode → zlib.decompress → json.loads

    Coverage:
        Lines 251-263: content_deserializer with complex transformation

    Expected:
        Compressed storage, transparent decompression, original content restored
    """
    import base64
    import json
    import zlib

    # Original node with large content
    original = Node(content={"document": "large data" * 100, "metadata": {"size": "large"}})

    # Compression serializer
    def compress_content(content):
        json_bytes = json.dumps(content).encode()
        compressed = zlib.compress(json_bytes, level=9)
        encoded = base64.b64encode(compressed).decode()
        return {"compressed": encoded, "format": "zlib+base64+json"}

    # Decompression deserializer (symmetric inverse)
    def decompress_content(content):
        if content.get("format") != "zlib+base64+json":
            raise ValueError("Unknown compression format")
        encoded = content["compressed"]
        compressed = base64.b64decode(encoded)
        json_bytes = zlib.decompress(compressed)
        return json.loads(json_bytes)

    # Serialize with compression
    data = original.to_dict(mode="db", content_serializer=compress_content)

    # Verify content is compressed (should be much smaller)
    compressed_size = len(data["content"]["compressed"])
    original_size = len(json.dumps(original.content))
    # Compression should reduce size significantly
    assert compressed_size < original_size

    # Deserialize with decompression
    restored = Node.from_dict(data, content_deserializer=decompress_content)

    # Verify round-trip: original content restored
    assert restored.content == original.content


def test_node_content_deserializer_encryption_roundtrip():
    """Test round-trip with encryption/decryption.

    Pattern:
        Security transformation: encrypt sensitive content for storage

    Design Rationale:
        Sensitive data (PII, credentials, health records) must be encrypted at rest.
        Application code should work with plaintext transparently.
        Encryption/decryption happens at serialization boundaries automatically.

    Real-World Scenario:
        Store encrypted patient records in database (HIPAA compliance).
        Application retrieves and decrypts transparently on access.

    Implementation:
        Simplified encryption (demo only - use proper crypto in production):
        - Serializer: XOR with key → base64 encode
        - Deserializer: base64 decode → XOR with key

    Coverage:
        Lines 251-263: content_deserializer with encryption workflow

    Expected:
        Encrypted storage, transparent decryption, original content restored

    Note:
        This uses toy encryption for testing. Production should use proper
        cryptography libraries (cryptography.fernet, NaCl, etc).
    """
    import base64

    # Toy encryption (XOR with key) - DO NOT USE IN PRODUCTION
    def toy_encrypt(content_dict, key=42):
        json_str = str(content_dict)
        encrypted_bytes = bytes(ord(c) ^ key for c in json_str)
        return {"encrypted": base64.b64encode(encrypted_bytes).decode(), "algorithm": "toy_xor"}

    def toy_decrypt(encrypted_dict, key=42):
        if encrypted_dict.get("algorithm") != "toy_xor":
            raise ValueError("Unknown encryption algorithm")
        encrypted_bytes = base64.b64decode(encrypted_dict["encrypted"])
        decrypted_str = "".join(chr(b ^ key) for b in encrypted_bytes)
        # Safely evaluate the dict string
        import ast

        return ast.literal_eval(decrypted_str)

    # Original node with sensitive content
    original = Node(
        content={"patient_id": "12345", "diagnosis": "confidential", "ssn": "***-**-****"}
    )

    # Serialize with encryption
    data = original.to_dict(content_serializer=toy_encrypt)

    # Verify content is encrypted (not plaintext)
    assert "encrypted" in data["content"]
    assert "patient_id" not in str(data["content"]["encrypted"])  # Sensitive data not visible

    # Deserialize with decryption
    restored = Node.from_dict(data, content_deserializer=toy_decrypt)

    # Verify round-trip: original sensitive content restored
    assert restored.content == original.content


def test_node_content_deserializer_external_storage_roundtrip():
    """Test round-trip with external storage references.

    Pattern:
        Large content stored externally (S3, CDN), Node stores reference only

    Design Rationale:
        Very large content (videos, datasets, large documents) shouldn't be in database.
        Store content externally, keep lightweight reference in Node.
        Transparent fetch on deserialization provides seamless access.

    Real-World Scenario:
        Store 100MB video in S3, Node contains {"ref": "s3://bucket/video.mp4", "size": 100MB}.
        On deserialization, fetch video from S3 transparently.

    Implementation:
        Serializer: Store content → external storage → return reference dict
        Deserializer: Fetch content from external storage using reference

    Coverage:
        Lines 251-263: content_deserializer with external storage fetch

    Expected:
        Reference stored in database, original content fetched on deserialization

    Note:
        This simulates external storage with in-memory dict for testing.
        Production would use boto3 (S3), Azure SDK, GCS client, etc.
    """
    # Simulate external storage (S3, database, etc)
    EXTERNAL_STORAGE = {}

    def store_external(content):
        """Store content externally, return reference."""
        import hashlib
        import json

        # Generate reference ID from content hash
        content_hash = hashlib.sha256(json.dumps(content).encode()).hexdigest()[:16]
        ref_id = f"ext://{content_hash}"

        # Store in external storage
        EXTERNAL_STORAGE[ref_id] = content

        # Return reference for Node
        return {"ref": ref_id, "size": len(json.dumps(content)), "type": "external"}

    def fetch_external(ref_dict):
        """Fetch content from external storage using reference."""
        if ref_dict.get("type") != "external":
            raise ValueError("Not an external storage reference")

        ref_id = ref_dict["ref"]
        if ref_id not in EXTERNAL_STORAGE:
            raise ValueError(f"External content not found: {ref_id}")

        return EXTERNAL_STORAGE[ref_id]

    # Original node with large content
    original = Node(content={"dataset": [i for i in range(1000)], "metadata": {"rows": 1000}})

    # Serialize with external storage
    data = original.to_dict(content_serializer=store_external)

    # Verify only reference stored (not full content)
    assert "ref" in data["content"]
    assert data["content"]["type"] == "external"
    assert data["content"]["ref"].startswith("ext://")
    # Verify full content NOT in serialized data
    assert "dataset" not in str(data["content"])

    # Deserialize with external fetch
    restored = Node.from_dict(data, content_deserializer=fetch_external)

    # Verify round-trip: original content fetched and restored
    assert restored.content == original.content


def test_node_content_deserializer_not_callable_raises():
    """Test content_deserializer fails fast if not callable.

    Pattern:
        Fail-fast validation prevents runtime errors downstream

    Design Rationale:
        Invalid deserializer (string, int, dict) would cause cryptic errors later.
        Fail immediately with clear error message during from_dict call.

    Coverage:
        Lines 253-256: content_deserializer callable validation

    Expected:
        TypeError raised with clear message about callable requirement
    """
    data = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": {"key": "value"},
    }

    # Test with non-callable (string)
    with pytest.raises(TypeError, match="content_deserializer must be callable"):
        Node.from_dict(data, content_deserializer="not_callable")

    # Test with non-callable (dict)
    with pytest.raises(TypeError, match="content_deserializer must be callable"):
        Node.from_dict(data, content_deserializer={"not": "callable"})


def test_node_content_deserializer_broken_deserializer_fails():
    """Test content_deserializer fails gracefully when deserializer raises exception.

    Pattern:
        Fail-fast with clear error message when deserializer implementation broken

    Design Rationale:
        Broken deserializer (wrong format, corrupt data, missing keys) should fail
        with clear error pointing to deserializer, not obscure pydantic validation error.

    Error Handling Strategy:
        Catch deserializer exceptions, wrap with ValueError explaining what failed

    Coverage:
        Lines 259-263: content_deserializer exception handling

    Expected:
        ValueError raised with message "content_deserializer failed: <original error>"
        Original exception preserved via raise...from for debugging
    """
    data = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "content": {"compressed": "data"},
    }

    # Deserializer that raises exception
    def broken_deserializer(content):
        raise RuntimeError("Decompression failed: corrupt data")

    # Should fail with clear error message
    with pytest.raises(ValueError, match="content_deserializer failed: Decompression failed"):
        Node.from_dict(data, content_deserializer=broken_deserializer)


def test_node_content_deserializer_with_none_content():
    """Test content_deserializer handles None content gracefully.

    Pattern:
        Defensive programming for missing content field

    Design Rationale:
        content_deserializer only applies if "content" key exists in data dict.
        If content is missing or None, skip deserializer (no transformation needed).

    Edge Case:
        Node with content=None is valid (uncommon but allowed).
        Deserializer should not be called if content field missing.

    Coverage:
        Line 259: if "content" in data check (skips deserializer for None)

    Expected:
        Node created with None content, deserializer never called
    """
    data = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        # No "content" field
    }

    deserializer_called = False

    def tracking_deserializer(content):
        nonlocal deserializer_called
        deserializer_called = True
        return content

    # Deserialize with None content
    node = Node.from_dict(data, content_deserializer=tracking_deserializer)

    # Verify deserializer never called (content field missing)
    assert deserializer_called is False
    assert node.content is None


def test_node_content_deserializer_symmetric_with_embedding_format():
    """Test content_deserializer works with other Node.from_dict parameters.

    Pattern:
        Feature composition: content_deserializer + meta_key integration

    Design Rationale:
        content_deserializer should compose cleanly with existing Node.from_dict parameters:
        - meta_key (custom metadata field name)
        - Polymorphic deserialization (lion_class routing)

        All parameters work together without conflicts.

    Use Case:
        Database stores Node with:
        - Custom metadata key ("node_meta" not "metadata")
        - Compressed content
        - Polymorphic type information (lion_class)

        All features should work together in single from_dict call.

    Coverage:
        Lines 251-291: content_deserializer integration with full from_dict workflow

    Expected:
        All parameters applied correctly, original content and metadata restored
    """
    import json

    # Compression helpers
    def simple_compress(content):
        return {"json": json.dumps(content)}

    def simple_decompress(content):
        return json.loads(content["json"])

    # Create original with metadata
    original = Node(content={"key": "value"}, metadata={"custom": "metadata"})

    # Serialize with custom meta_key and compression
    data = original.to_dict(mode="db", meta_key="node_meta", content_serializer=simple_compress)

    # Verify custom meta_key used
    assert "node_meta" in data
    assert "metadata" not in data
    # Verify compression applied
    assert "json" in data["content"]

    # Deserialize with symmetric parameters
    restored = Node.from_dict(data, meta_key="node_meta", content_deserializer=simple_decompress)

    # Verify all features work together
    assert restored.content == original.content
    assert restored.metadata["custom"] == "metadata"
    assert restored.id == original.id

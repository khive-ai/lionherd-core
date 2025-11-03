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
   - Any Python value: primitives, collections, Element instances
   - Automatic nested Element serialization via field_serializer
   - Lenient validation with type coercion (list→content, dict→content)
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
    person = PersonNode(name="Alice", age=30, content="bio")

    assert person.name == "Alice"
    assert person.age == 30
    assert person.content == "bio"


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
    data = {"content": "test content", "metadata": {}}

    node = Node.from_dict(data)

    assert isinstance(node, Node)
    assert node.content == "test content"


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
        "content": "engineer",
        "metadata": {"lion_class": "PersonNode"},
    }

    node = Node.from_dict(data)

    # Should return PersonNode instance
    assert isinstance(node, PersonNode)
    assert node.name == "Bob"
    assert node.age == 25
    assert node.content == "engineer"


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
    data = {"content": "test", "metadata": {"lion_class": "NonExistentNode"}}

    # Should not raise, just create base Node
    node = Node.from_dict(data)

    assert isinstance(node, Node)
    assert not isinstance(node, (PersonNode, DocumentNode))


def test_from_dict_preserves_metadata():
    """Test from_dict preserves custom metadata but removes lion_class."""
    data = {
        "content": "test",
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
    node = Node(content="test")

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
# Design philosophy: Maximum flexibility
# - No type constraints on content (Any type)
# - Nested Elements/Nodes handled automatically via field serializer/validator
# - Plain data passes through unchanged
# - Enables graph-of-graphs patterns (Node contains Graph, which contains Nodes)
#
# Serialization strategy:
# - _serialize_content(): Detects Element instances → calls to_dict()
# - _validate_content(): Detects dicts with lion_class → calls from_dict()
# - Non-Element data: unchanged in both directions
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
    inner_node = Node(content="inner")
    outer = Node(content=inner_node)

    # Content should be the nested Node
    assert isinstance(outer.content, Node)
    assert outer.content.content == "inner"


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
        {"content": "generic", "node_metadata": {"lion_class": "Node"}},
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
        Node(content="Z"),
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
    data = {"content": "test", "metadata": {}}

    node = Node.from_dict(data)

    assert isinstance(node, Node)


def test_from_dict_no_metadata_field():
    """Test from_dict without metadata field."""
    data = {"content": "test"}

    node = Node.from_dict(data)

    assert isinstance(node, Node)
    assert node.content == "test"


def test_subclass_from_dict_direct_call():
    """Test calling from_dict directly on subclass."""
    data = {"name": "Iris", "age": 22}

    person = PersonNode.from_dict(data)

    # Should create PersonNode even without lion_class
    assert isinstance(person, PersonNode)
    assert person.name == "Iris"


def test_node_equality_by_id():
    """Test Node instances with same ID have same ID (not pydantic equality)."""
    node1 = Node(content="a")
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
# Design aspect validated: Node uses shared adapter registry (_registry() override)
# so adapters registered once on Node are available on all subclasses. This design
# avoids redundant adapter registration while maintaining pydapter's extensibility.


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

    person = PersonNode(name="Alice", age=30, content="engineer")

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
content = "developer"

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
    original = PersonNode(name="Charlie", age=35, content="manager")

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
    node = Node(content="test")
    assert node.embedding is None


def test_node_embedding_list():
    """Test Node with embedding as list of floats."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    node = Node(content="test", embedding=embedding)
    assert node.embedding == embedding
    assert all(isinstance(x, float) for x in node.embedding)


def test_node_embedding_coerce_ints():
    """Test Node embedding coerces ints to floats."""
    node = Node(content="test", embedding=[1, 2, 3])
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

    node = Node(content="test", embedding=json_str)
    assert node.embedding == embedding


def test_node_embedding_rejects_empty_list():
    """Test Node embedding rejects empty list."""
    import pytest

    with pytest.raises(ValueError, match="embedding list cannot be empty"):
        Node(content="test", embedding=[])


def test_node_embedding_rejects_non_numeric():
    """Test Node embedding rejects non-numeric values."""
    import pytest

    with pytest.raises(ValueError, match="embedding must contain only numeric values"):
        Node(content="test", embedding=[0.1, "invalid", 0.3])


def test_node_embedding_rejects_invalid_type():
    """Test Node embedding rejects invalid types."""
    import pytest

    with pytest.raises(ValueError, match="embedding must be a list"):
        Node(content="test", embedding={"invalid": "dict"})


def test_node_embedding_serialization():
    """Test Node embedding serializes correctly in different modes."""
    node = Node(content="test", embedding=[0.1, 0.2, 0.3])

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
    original = Node(content="test", embedding=[0.1, 0.2, 0.3])

    # Roundtrip through dict
    data = original.to_dict(mode="db")
    restored = Node.from_dict(data)

    assert restored.embedding == original.embedding


# ==================== created_at_format Tests ====================


def test_node_created_at_format_datetime():
    """Test Node to_dict with created_at_format='datetime' (default)."""
    import datetime as dt

    node = Node(content="test")
    data = node.to_dict(mode="python", created_at_format="datetime")

    # Default: keep as datetime object
    assert isinstance(data["created_at"], dt.datetime)
    assert data["created_at"] == node.created_at


def test_node_created_at_format_isoformat():
    """Test Node to_dict with created_at_format='isoformat'."""
    node = Node(content="test")
    data = node.to_dict(mode="python", created_at_format="isoformat")

    # ISO format: string
    assert isinstance(data["created_at"], str)
    assert data["created_at"] == node.created_at.isoformat()


def test_node_created_at_format_timestamp():
    """Test Node to_dict with created_at_format='timestamp' (legacy)."""
    node = Node(content="test")
    data = node.to_dict(mode="python", created_at_format="timestamp")

    # Timestamp: float
    assert isinstance(data["created_at"], float)
    assert data["created_at"] == node.created_at.timestamp()


def test_node_created_at_format_db_mode():
    """Test Node to_dict with created_at_format in db mode."""
    node = Node(content="test")

    # DB mode with isoformat
    data = node.to_dict(mode="db", created_at_format="isoformat")
    assert isinstance(data["created_at"], str)
    assert "node_metadata" in data  # DB mode uses node_metadata


def test_node_created_at_roundtrip_isoformat():
    """Test Node created_at survives roundtrip via isoformat."""
    original = Node(content="test")

    # Serialize with isoformat
    data = original.to_dict(mode="db", created_at_format="isoformat")
    assert isinstance(data["created_at"], str)

    # Deserialize - Element validator handles ISO strings
    restored = Node.from_dict(data)
    assert restored.created_at == original.created_at


# ============================================================================
# Content Validation Edge Cases
# ============================================================================
#
# Design aspect validated: Content field polymorphism edge cases, embedding
# validation corner cases, and metadata handling robustness.
#
# These tests cover scenarios where content is not a standard Node, embedding
# data comes from databases (JSON strings), and metadata has unexpected types.
# Critical for production resilience when deserializing from external sources.


class CustomElement(Element):
    """Custom Element that's not a Node - for testing content polymorphism."""

    value: int = 0


class TestNodeContentValidationEdgeCases:
    """
    Test Node content validation edge cases and error handling.

    Edge Cases:
        - Content with non-Node Elements: Polymorphic content handling
        - Embedding JSON parsing errors: Invalid vector data from databases
        - Metadata edge cases: Complex nested structures and type mismatches
        - Embedding type coercion: Integer vectors from external sources

    Scenarios:
        - Element in Node.content: Polymorphic serialization of non-Node Elements
        - Invalid embedding JSON: Graceful error handling with clear messages
        - Metadata with non-dict types: Robustness against malformed data
        - JSON string embeddings: Database compatibility (PostgreSQL JSONB)

    Invariants Tested:
        - Content polymorphism: Any Element subclass as content
        - Embedding validation: Type coercion and format parsing
        - Metadata robustness: Handle edge case values gracefully
        - Error messages: Clear diagnostic information for failures

    Design Rationale:
        These edge cases ensure Node handles real-world scenarios:
        - Legacy data without proper metadata structure
        - Database queries returning JSON-serialized vectors
        - Custom Element types in graph-of-graphs patterns
        - Malformed input from external systems (APIs, migrations)

        Trade-offs:
        - Strictness vs Resilience: Reject clearly invalid data (empty embeddings)
          but coerce recoverable data (int→float, JSON string→list)
        - Performance vs Safety: JSON parsing overhead for database compatibility
        - Type Safety vs Flexibility: Allow Any content but validate on access
    """

    def test_validate_content_with_element_not_node(self):
        """Test content validator deserializes non-Node Element subclasses via lion_class.

        Pattern:
            Polymorphic content deserialization for Element subtypes

        Edge Case:
            Content is an Element subclass that's not a Node (CustomElement)

        Design Rationale:
            Graph-of-graphs patterns require storing arbitrary Elements in Node.content.
            Example: Workflow node contains metadata Element with execution history.

        Validation Strategy:
            1. Detect dict with lion_class metadata
            2. Route to Element.from_dict() (not Node-specific)
            3. Deserialize to correct Element subclass via polymorphism

        Use Case:
            Heterogeneous graph where nodes contain different Element types:
            - DocumentNode.content = TextElement
            - WorkflowNode.content = ExecutionMetadata (Element subclass)
            - TaskNode.content = nested Node

        Expected:
            Content deserializes as CustomElement, not base Element or Node
        """
        # Create a non-Node Element with lion_class metadata
        elem = CustomElement(value=42)
        elem_dict = elem.to_dict()

        # Content is an Element dict with lion_class but NOT a Node
        # Validator should use Element.from_dict to restore correct type
        node = Node(content=elem_dict)

        # Content should be deserialized as CustomElement (not Node)
        assert isinstance(node.content, CustomElement)
        assert isinstance(node.content, Element)
        assert not isinstance(node.content, Node)
        assert node.content.value == 42

    def test_embedding_invalid_json_string(self):
        """Test malformed embedding JSON raises clear ValueError.

        Pattern:
            Defensive validation with actionable error messages

        Edge Case:
            Database returns corrupted JSON string for embedding field

        Design Rationale:
            JSON string parsing is common failure mode when deserializing from databases.
            Clear error messages enable fast debugging in production.

        Error Message Requirements:
            - Identifies field: "embedding"
            - Identifies format: "JSON string"
            - Identifies action: "Failed to parse"

        Use Case:
            PostgreSQL JSONB column corrupted during migration or manual edit

        Expected:
            ValueError with "Failed to parse embedding JSON string" message
        """
        # Pass invalid JSON string as embedding
        with pytest.raises(ValueError, match="Failed to parse embedding JSON string"):
            Node(content="test", embedding="not valid json [[[")

    def test_embedding_malformed_json_string(self):
        """Test incomplete embedding JSON raises clear ValueError.

        Pattern:
            Defensive validation for truncated data

        Edge Case:
            JSON string truncated during network transfer or database query limit

        Expected:
            ValueError with "Failed to parse embedding JSON string" message
        """
        with pytest.raises(ValueError, match="Failed to parse embedding JSON string"):
            Node(content="test", embedding='{"incomplete": ')

    def test_from_dict_with_non_dict_metadata(self):
        """Test non-dict metadata disables polymorphism but doesn't crash.

        Pattern:
            Graceful degradation for malformed input

        Edge Case:
            Metadata field is string/number instead of dict (legacy data or bug)

        Design Rationale:
            Prioritize resilience over strict validation. Node creation should succeed
            even with malformed metadata, just without polymorphic deserialization.

        Trade-off:
            - Safety: Don't crash on unexpected data
            - Capability Loss: No polymorphism without dict metadata (lion_class unavailable)

        Use Case:
            Importing nodes from external systems that don't follow Lion conventions

        Expected:
            Node created successfully, metadata stored as-is, no polymorphism
        """
        # Create data with metadata as string (not dict)
        data = {"content": "test", "metadata": "not a dict"}

        # Should handle non-dict metadata (lion_class = None)
        node = Node.from_dict(data)

        # Should still create node, just no polymorphism
        assert isinstance(node, Node)
        assert node.content == "test"

    def test_from_dict_with_metadata_as_list(self):
        """Test metadata as list raises validation error.

        Pattern:
            Strict validation for clearly wrong types

        Edge Case:
            Metadata field is list instead of dict (data corruption or API mismatch)

        Design Rationale:
            Lists as metadata have no recovery path (can't extract lion_class).
            Better to fail fast with clear error than silently create invalid state.

        Contrast with test_from_dict_with_non_dict_metadata:
            - String metadata: Degrades gracefully (just stored as-is)
            - List metadata: No graceful degradation (reject immediately)

        Expected:
            ValidationError or TypeError from Pydantic
        """
        data = {"content": "test", "metadata": ["list", "not", "dict"]}

        # Pydantic will reject list as metadata
        with pytest.raises((ValueError, TypeError)):
            Node.from_dict(data)

    def test_adapt_async_kwargs_defaults_skipped(self):
        """Test adapter kwargs.setdefault is tested via integration tests.

        Pattern:
            Skip tests that duplicate coverage in other test suites

        Rationale:
            Adapter kwargs.setdefault() is pydapter internals tested by pydapter's
            own test suite. Integration tests in this file validate end-to-end
            adapter behavior (test_node_adapt_to_toml, test_node_adapt_from_yaml).

        Expected:
            Test skipped with clear reason
        """
        pytest.skip("Adapter kwargs.setdefault tested by integration tests")

    def test_embedding_json_string_with_ints(self):
        """Test embedding JSON string with integers coerces to floats.

        Pattern:
            Type coercion for database compatibility

        Edge Case:
            Database stores embedding as JSON "[1, 2, 3]" (integer array)

        Design Rationale:
            Embeddings are mathematically float vectors, but databases/APIs may
            serialize as integers for compactness. Coercion enables seamless
            integration without manual type conversion at every query site.

        Use Case:
            PostgreSQL JSONB storing [1, 2, 3] retrieves as JSON string,
            validator parses to [1.0, 2.0, 3.0] automatically

        Complexity:
            O(n) coercion where n = embedding dimension (typically 768-4096)

        Expected:
            All values coerced to float, original ints preserved as 1.0, 2.0, 3.0
        """
        import orjson

        embedding = [1, 2, 3]
        json_str = orjson.dumps(embedding).decode()

        node = Node(content="test", embedding=json_str)

        # Should parse and coerce to floats
        assert node.embedding == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in node.embedding)

    def test_embedding_json_string_with_mixed_numbers(self):
        """Test embedding JSON string with mixed int/float values.

        Pattern:
            Type coercion with mixed numeric types

        Edge Case:
            Embedding vector has both integers and floats (sparse vector encoding)

        Expected:
            All values parsed correctly, integers coerced to floats
        """
        import orjson

        embedding = [1, 2.5, 3, 4.7]
        json_str = orjson.dumps(embedding).decode()

        node = Node(content="test", embedding=json_str)

        # Should parse correctly
        assert node.embedding == [1.0, 2.5, 3.0, 4.7]

    def test_content_with_element_dict_no_lion_class(self):
        """Test content dict without lion_class remains as plain dict.

        Pattern:
            Graceful degradation when polymorphic metadata absent

        Edge Case:
            Content is dict that looks like Element but lacks lion_class
            (legacy data, external systems, manual construction)

        Design Rationale:
            Without lion_class, cannot determine target type for deserialization.
            Better to preserve as plain dict than guess or crash.

        Trade-off:
            - Safety: Don't crash, don't guess types
            - Type Loss: Lose Element identity (acceptable for legacy data)

        Use Case:
            Migrating nodes from non-Lion systems where content is arbitrary JSON

        Expected:
            Content remains as dict, not converted to Element
        """
        # Dict with metadata but no lion_class
        content_dict = {"key": "value", "metadata": {"custom": "data"}}

        node = Node(content=content_dict)

        # Should remain as dict (not converted to Element)
        assert isinstance(node.content, dict)
        assert node.content == content_dict

    def test_content_with_nested_node_in_node_registry(self):
        """Test content with custom Node subclass deserializes correctly.

        Pattern:
            Nested polymorphic deserialization via registry

        Edge Case:
            Custom Node subclass (not predefined) in content field

        Design Rationale:
            Registry pattern enables zero-config subclass polymorphism.
            Custom Node subclasses automatically registered via __pydantic_init_subclass__.

        Validation Flow:
            1. CustomNode defined → auto-registered in NODE_REGISTRY
            2. CustomNode.to_dict() → includes lion_class="CustomNode"
            3. Node(content=custom_dict) → content validator detects lion_class
            4. Element.from_dict(custom_dict) → routes to CustomNode
            5. Result: content is CustomNode instance, not base Node

        Use Case:
            Domain-specific Node subclasses (PersonNode, DocumentNode) nested
            in generic workflow nodes for type-safe deserialization

        Expected:
            Content deserializes as CustomNode with custom_field preserved
        """

        class CustomNode(Node):
            """Custom Node subclass."""

            custom_field: str = "test"

        # Create custom node
        custom = CustomNode(custom_field="value")
        custom_dict = custom.to_dict()

        # Create outer node with custom node as content
        outer = Node(content=custom_dict)

        # Content should be deserialized as CustomNode
        assert isinstance(outer.content, CustomNode)
        assert outer.content.custom_field == "value"

# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for meta_key parameter in serialization/deserialization.

Tests: to_dict/from_dict meta_key renaming, item_meta_key for nested items, backward compatibility.
Classes: Element, Node, Pile, Graph
"""

import pytest

from lionherd_core.base import Edge, Element, Graph, Node, Pile


class TestElementMetaKey:
    """Test meta_key parameter for Element class."""

    def test_element_to_dict_db_mode_default_meta_key(self):
        """Test Element.to_dict with mode='db' uses default 'node_metadata' key."""
        elem = Element(metadata={"custom": "value", "nested": {"key": "val"}})

        data = elem.to_dict(mode="db")

        # Default meta_key is "node_metadata"
        assert "node_metadata" in data
        assert "metadata" not in data
        assert data["node_metadata"]["custom"] == "value"
        assert data["node_metadata"]["nested"]["key"] == "val"

    def test_element_to_dict_db_mode_custom_meta_key(self):
        """Test Element.to_dict with mode='db' and custom meta_key."""
        elem = Element(metadata={"foo": "bar"})

        data = elem.to_dict(mode="db", meta_key="custom_meta")

        # Custom meta_key should be used
        assert "custom_meta" in data
        assert "metadata" not in data
        assert "node_metadata" not in data
        assert data["custom_meta"]["foo"] == "bar"

    def test_element_from_dict_default_meta_key(self):
        """Test Element.from_dict with default meta_key restores metadata."""
        elem = Element(metadata={"test": "data"})
        data = elem.to_dict(mode="db")  # Creates node_metadata

        restored = Element.from_dict(data, meta_key="node_metadata")

        assert restored.metadata["test"] == "data"
        assert "lion_class" not in restored.metadata  # Should be removed during from_dict

    def test_element_from_dict_custom_meta_key(self):
        """Test Element.from_dict with custom meta_key restores metadata."""
        elem = Element(metadata={"key": "value"})
        data = elem.to_dict(mode="db", meta_key="my_metadata")

        restored = Element.from_dict(data, meta_key="my_metadata")

        assert restored.metadata["key"] == "value"

    def test_element_from_dict_backward_compatibility_node_metadata(self):
        """Test Element.from_dict handles legacy 'node_metadata' key without meta_key parameter."""
        # Simulate legacy data with node_metadata key
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-01-01T00:00:00Z",
            "node_metadata": {"legacy": "data", "lion_class": "lionherd_core.base.element.Element"},
        }

        # Should automatically handle node_metadata → metadata
        restored = Element.from_dict(data)

        assert restored.metadata["legacy"] == "data"
        assert "lion_class" not in restored.metadata  # Removed during deserialization

    def test_element_from_dict_meta_key_takes_precedence_over_node_metadata(self):
        """Test that explicit meta_key takes precedence over legacy node_metadata."""
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-01-01T00:00:00Z",
            "custom_meta": {"priority": "high"},
            "node_metadata": {"priority": "low"},
        }

        # meta_key should take precedence
        restored = Element.from_dict(data, meta_key="custom_meta")

        assert restored.metadata["priority"] == "high"


class TestNodeMetaKey:
    """Test meta_key parameter for Node class."""

    def test_node_to_dict_db_mode_custom_meta_key(self):
        """Test Node.to_dict with mode='db' and custom meta_key."""
        node = Node(content={"value": "test node"}, metadata={"node_data": "value"})

        data = node.to_dict(mode="db", meta_key="custom_node_meta")

        assert "custom_node_meta" in data
        assert "metadata" not in data
        assert data["custom_node_meta"]["node_data"] == "value"
        assert data["content"] == {"value": "test node"}

    def test_node_from_dict_custom_meta_key(self):
        """Test Node.from_dict with custom meta_key."""
        node = Node(content={"value": "original"}, metadata={"info": "test"})
        data = node.to_dict(mode="db", meta_key="my_meta")

        restored = Node.from_dict(data, meta_key="my_meta")

        assert restored.content == {"value": "original"}
        assert restored.metadata["info"] == "test"

    def test_node_from_dict_backward_compatibility(self):
        """Test Node.from_dict handles legacy node_metadata automatically."""
        # Element.from_dict now handles this, but verify Node still works
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-01-01T00:00:00Z",
            "content": {"value": "legacy node"},
            "node_metadata": {"legacy": "node_data"},
        }

        restored = Node.from_dict(data)

        assert restored.content == {"value": "legacy node"}
        assert restored.metadata["legacy"] == "node_data"


class TestPileMetaKey:
    """Test meta_key and item_meta_key parameters for Pile class."""

    def test_pile_to_dict_db_mode_custom_meta_key(self):
        """Test Pile.to_dict with mode='db' and custom meta_key."""
        pile = Pile([Element(), Element()], metadata={"pile_info": "data"})

        data = pile.to_dict(mode="db", meta_key="pile_meta")

        assert "pile_meta" in data
        assert "metadata" not in data
        assert data["pile_meta"]["pile_info"] == "data"

    def test_pile_from_dict_custom_meta_key(self):
        """Test Pile.from_dict with custom meta_key restores pile metadata."""
        pile = Pile([Element()], metadata={"test": "pile"})
        data = pile.to_dict(mode="db", meta_key="custom_pile")

        restored = Pile.from_dict(data, meta_key="custom_pile")

        assert restored.metadata["test"] == "pile"
        assert len(restored) == 1

    def test_pile_from_dict_with_item_meta_key(self):
        """Test Pile.from_dict with item_meta_key passes to Element.from_dict."""
        # Create pile with items that have metadata
        elem1 = Element(metadata={"item": "one"})
        elem2 = Element(metadata={"item": "two"})
        pile = Pile([elem1, elem2])

        # Serialize to db mode (items will have node_metadata)
        data = pile.to_dict(mode="db")

        # Deserialize with item_meta_key
        restored = Pile.from_dict(data, meta_key="node_metadata", item_meta_key="node_metadata")

        assert len(restored) == 2
        items = list(restored)
        # Note: order might not be preserved in dict, but IDs should match
        assert items[0].metadata["item"] in ["one", "two"]
        assert items[1].metadata["item"] in ["one", "two"]

    def test_pile_roundtrip_with_custom_meta_keys(self):
        """Test full roundtrip with custom meta_key and item_meta_key."""
        elem1 = Element(metadata={"data": "A"})
        elem2 = Element(metadata={"data": "B"})
        pile = Pile([elem1, elem2], metadata={"pile": "test"})

        # Serialize with custom meta_key
        data = pile.to_dict(mode="db", meta_key="custom_pile_meta")

        # Pile metadata should be renamed to custom_pile_meta
        assert "custom_pile_meta" in data
        assert data["custom_pile_meta"]["pile"] == "test"
        # Items are serialized with mode="json", so they have "metadata" not renamed
        assert data["items"][0]["metadata"]["data"] in ["A", "B"]

        # Deserialize with matching keys
        restored = Pile.from_dict(data, meta_key="custom_pile_meta")

        assert restored.metadata["pile"] == "test"
        assert len(restored) == 2


class TestGraphMetaKey:
    """Test meta_key and item_meta_key parameters for Graph class."""

    def test_graph_to_dict_db_mode_custom_meta_key(self):
        """Test Graph.to_dict with mode='db' and custom meta_key."""
        graph = Graph(metadata={"graph_data": "test"})
        n1 = Node(content={"value": "A"})
        graph.add_node(n1)

        data = graph.to_dict(mode="db", meta_key="graph_meta")

        assert "graph_meta" in data
        assert "metadata" not in data
        assert data["graph_meta"]["graph_data"] == "test"

    def test_graph_from_dict_custom_meta_key(self):
        """Test Graph.from_dict with custom meta_key."""
        graph = Graph(metadata={"info": "graph"})
        n1 = Node(content={"value": "Node1"})
        graph.add_node(n1)

        data = graph.to_dict(mode="db", meta_key="my_graph_meta")

        restored = Graph.from_dict(data, meta_key="my_graph_meta", item_meta_key="node_metadata")

        assert restored.metadata["info"] == "graph"
        assert len(restored.nodes) == 1

    def test_graph_from_dict_with_item_meta_key_for_nodes(self):
        """Test Graph.from_dict with item_meta_key restores node metadata correctly."""
        graph = Graph()
        n1 = Node(content={"value": "A"}, metadata={"node": "one"})
        n2 = Node(content={"value": "B"}, metadata={"node": "two"})
        graph.add_node(n1)
        graph.add_node(n2)

        data = graph.to_dict(mode="db")

        restored = Graph.from_dict(data, meta_key="node_metadata", item_meta_key="node_metadata")

        assert len(restored.nodes) == 2
        nodes = list(restored.nodes)
        assert nodes[0].metadata["node"] in ["one", "two"]
        assert nodes[1].metadata["node"] in ["one", "two"]

    def test_graph_roundtrip_with_edges_and_custom_meta_keys(self):
        """Test full graph roundtrip with nodes and edges using custom meta_keys."""
        graph = Graph(metadata={"graph": "roundtrip"})
        n1 = Node(content={"value": "Source"}, metadata={"role": "source"})
        n2 = Node(content={"value": "Target"}, metadata={"role": "target"})
        graph.add_node(n1)
        graph.add_node(n2)
        edge = Edge(head=n1.id, tail=n2.id)
        graph.add_edge(edge)

        data = graph.to_dict(mode="db", meta_key="custom_graph")

        restored = Graph.from_dict(data, meta_key="custom_graph", item_meta_key="node_metadata")

        assert restored.metadata["graph"] == "roundtrip"
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1


class TestItemMetaKeyInToDict:
    """Test item_meta_key parameter in Pile and Graph to_dict methods."""

    def test_pile_to_dict_with_item_meta_key(self):
        """Test Pile.to_dict passes item_meta_key to items."""
        elem1 = Element(metadata={"data": "A"})
        elem2 = Element(metadata={"data": "B"})
        pile = Pile([elem1, elem2])

        # Serialize with item_meta_key
        data = pile.to_dict(mode="json", item_meta_key="custom_item_meta")

        # Items should have their metadata renamed
        assert data["items"][0]["custom_item_meta"]["data"] in ["A", "B"]
        assert "metadata" not in data["items"][0]

    def test_pile_to_dict_with_item_created_at_format(self):
        """Test Pile.to_dict passes item_created_at_format to items."""
        elem = Element()
        pile = Pile([elem])

        # Serialize with item_created_at_format
        data = pile.to_dict(mode="python", item_created_at_format="timestamp")

        # Item's created_at should be timestamp format
        assert isinstance(data["items"][0]["created_at"], float)

    def test_graph_to_dict_with_item_meta_key_for_nodes(self):
        """Test Graph.to_dict passes item_meta_key to node items."""
        graph = Graph()
        n1 = Node(content={"value": "Test"}, metadata={"node": "data"})
        graph.add_node(n1)

        # Serialize with item_meta_key
        data = graph.to_dict(mode="json", item_meta_key="custom_node_meta")

        # Node items should have their metadata renamed
        assert data["nodes"]["items"][0]["custom_node_meta"]["node"] == "data"
        assert "metadata" not in data["nodes"]["items"][0]

    def test_graph_to_dict_with_item_created_at_format_for_edges(self):
        """Test Graph.to_dict passes item_created_at_format to edge items."""
        graph = Graph()
        n1 = Node(content={"value": "A"})
        n2 = Node(content={"value": "B"})
        graph.add_node(n1)
        graph.add_node(n2)
        edge = Edge(head=n1.id, tail=n2.id)
        graph.add_edge(edge)

        # Serialize with item_created_at_format
        data = graph.to_dict(mode="python", item_created_at_format="timestamp")

        # Edge items should have timestamp format
        assert isinstance(data["edges"]["items"][0]["created_at"], float)

    def test_pile_full_control_with_all_parameters(self):
        """Test Pile.to_dict with meta_key, item_meta_key, and item_created_at_format together."""
        elem = Element(metadata={"item": "value"})
        pile = Pile([elem], metadata={"pile": "data"})

        # Use mode="python" so items can use timestamp format (json mode always uses isoformat)
        data = pile.to_dict(
            mode="python",
            meta_key="pile_meta",
            item_meta_key="item_meta",
            item_created_at_format="timestamp",
        )

        # Pile metadata should be renamed
        assert "pile_meta" in data
        assert data["pile_meta"]["pile"] == "data"

        # Item metadata should be renamed and timestamp format used
        assert data["items"][0]["item_meta"]["item"] == "value"
        assert isinstance(data["items"][0]["created_at"], float)


class TestMetaKeyEdgeCases:
    """Test edge cases and error conditions for meta_key parameter."""

    def test_from_dict_with_nonexistent_meta_key(self):
        """Test from_dict with meta_key that doesn't exist in data (should use default metadata)."""
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-01-01T00:00:00Z",
            "metadata": {"default": "data"},
        }

        # meta_key doesn't exist, should use regular metadata
        restored = Element.from_dict(data, meta_key="nonexistent_key")

        assert restored.metadata["default"] == "data"

    def test_from_dict_with_both_meta_key_and_metadata(self):
        """Test from_dict when both custom meta_key and metadata exist (meta_key takes precedence)."""
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2025-01-01T00:00:00Z",
            "custom_meta": {"priority": "high"},
            "metadata": {"priority": "low"},
        }

        restored = Element.from_dict(data, meta_key="custom_meta")

        # meta_key data should be used
        assert restored.metadata["priority"] == "high"

    def test_pile_from_dict_without_item_meta_key_uses_default(self):
        """Test Pile.from_dict without item_meta_key parameter uses default behavior."""
        elem = Element(metadata={"item": "data"})
        pile = Pile([elem])

        # Serialize to python mode (keeps "metadata" key)
        data = pile.to_dict(mode="python")

        # Deserialize without item_meta_key (should work with default "metadata")
        restored = Pile.from_dict(data)

        assert len(restored) == 1
        items = list(restored)
        assert items[0].metadata["item"] == "data"

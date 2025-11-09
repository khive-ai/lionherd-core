# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""HashableModel test suite: content-based hashing, serialization, protocol implementation.

Design Philosophy:
    HashableModel provides content-based hashing as alternative to Element's ID-based
    hashing. Two instances with identical field values have the same hash, enabling
    use as cache keys, set deduplication, and value-based equality.

Test Architecture:
    - Construction: Basic instantiation, nested models, optional fields
    - Hashing: Content-based equality, nested structures, determinism
    - Serialization: python/json modes, roundtrips
    - Protocols: Serializable, Deserializable, Hashable implementation
    - Edge Cases: Mutability, nested HashableModels, mixed BaseModel nesting

Key Differences from Element:
    - Element: ID-based hash (same ID = same hash, content ignored)
    - HashableModel: Content-based hash (same fields = same hash)
"""

import pytest
from pydantic import BaseModel, ValidationError

from lionherd_core.types import HashableModel


class SimpleConfig(HashableModel):
    """Test model with basic fields."""

    name: str
    value: int


class NestedConfig(HashableModel):
    """Test model with nested HashableModel."""

    config: SimpleConfig
    enabled: bool = True


class ConfigWithOptional(HashableModel):
    """Test model with optional fields."""

    required: str
    optional: str | None = None


class TestHashableModelConstruction:
    """HashableModel instantiation and field handling."""

    def test_basic_construction(self):
        """Basic instantiation with required fields."""
        config = SimpleConfig(name="test", value=42)
        assert config.name == "test"
        assert config.value == 42

    def test_nested_construction(self):
        """Nested HashableModel construction."""
        inner = SimpleConfig(name="inner", value=1)
        outer = NestedConfig(config=inner, enabled=False)
        assert outer.config.name == "inner"
        assert outer.enabled is False

    def test_optional_fields(self):
        """Optional fields and None values."""
        config = ConfigWithOptional(required="test")
        assert config.required == "test"
        assert config.optional is None


class TestHashableModelHashing:
    """Content-based hashing behavior."""

    def test_identical_content_same_hash(self):
        """Identical field values produce same hash."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)
        assert hash(c1) == hash(c2)

    def test_different_content_different_hash(self):
        """Different field values produce different hashes."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=99)
        assert hash(c1) != hash(c2)

    def test_nested_content_hash(self):
        """Nested models contribute to hash."""
        inner1 = SimpleConfig(name="test", value=1)
        inner2 = SimpleConfig(name="test", value=1)
        inner3 = SimpleConfig(name="test", value=2)

        outer1 = NestedConfig(config=inner1)
        outer2 = NestedConfig(config=inner2)
        outer3 = NestedConfig(config=inner3)

        assert hash(outer1) == hash(outer2)
        assert hash(outer1) != hash(outer3)

    def test_hash_determinism(self):
        """Hash is deterministic across multiple calls."""
        config = SimpleConfig(name="test", value=42)
        hash1 = hash(config)
        hash2 = hash(config)
        hash3 = hash(config)
        assert hash1 == hash2 == hash3

    def test_use_in_set(self):
        """Content-based deduplication in sets."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)
        c3 = SimpleConfig(name="different", value=42)

        s = {c1, c2, c3}
        assert len(s) == 2  # c1 and c2 deduplicated

    def test_use_as_dict_key(self):
        """Can use as dict key (content-based)."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)

        d = {c1: "value1"}
        d[c2] = "value2"  # Overwrites c1 (same content)

        assert len(d) == 1
        assert d[c1] == "value2"


class TestHashableModelSerialization:
    """Serialization in python and json modes."""

    def test_to_dict_python_mode(self):
        """Python mode preserves native types."""
        config = SimpleConfig(name="test", value=42)
        data = config.to_dict(mode="python")

        assert data == {"name": "test", "value": 42}
        assert isinstance(data["value"], int)

    def test_to_dict_json_mode(self):
        """JSON mode returns JSON-safe types."""
        config = SimpleConfig(name="test", value=42)
        data = config.to_dict(mode="json")

        assert data == {"name": "test", "value": 42}
        # All values JSON-serializable
        import orjson

        orjson.dumps(data)  # Should not raise

    def test_to_json_deterministic(self):
        """JSON output is deterministic (sorted keys)."""
        config = SimpleConfig(name="test", value=42)
        json1 = config.to_json()
        json2 = config.to_json()

        assert json1 == json2
        assert "name" in json1
        assert "value" in json1

    def test_to_json_bytes(self):
        """to_json with decode=False returns bytes."""
        config = SimpleConfig(name="test", value=42)
        json_bytes = config.to_json(decode=False)

        assert isinstance(json_bytes, bytes)
        import orjson

        parsed = orjson.loads(json_bytes)
        assert parsed["name"] == "test"


class TestHashableModelDeserialization:
    """Deserialization from dict and JSON."""

    def test_from_dict_python_mode(self):
        """Python mode deserialization."""
        data = {"name": "test", "value": 42}
        config = SimpleConfig.from_dict(data, mode="python")

        assert config.name == "test"
        assert config.value == 42

    def test_from_dict_json_mode(self):
        """JSON mode deserialization."""
        data = {"name": "test", "value": 42}
        config = SimpleConfig.from_dict(data, mode="json")

        assert config.name == "test"
        assert config.value == 42

    def test_from_json_string(self):
        """from_json with string input."""
        json_str = '{"name": "test", "value": 42}'
        config = SimpleConfig.from_json(json_str)

        assert config.name == "test"
        assert config.value == 42

    def test_from_json_bytes(self):
        """from_json with bytes input."""
        json_bytes = b'{"name": "test", "value": 42}'
        config = SimpleConfig.from_json(json_bytes)

        assert config.name == "test"
        assert config.value == 42

    def test_roundtrip_python_mode(self):
        """Python mode roundtrip preserves content."""
        original = SimpleConfig(name="test", value=42)
        data = original.to_dict(mode="python")
        restored = SimpleConfig.from_dict(data, mode="python")

        assert hash(original) == hash(restored)
        assert original.name == restored.name
        assert original.value == restored.value

    def test_roundtrip_json_mode(self):
        """JSON mode roundtrip preserves content."""
        original = SimpleConfig(name="test", value=42)
        data = original.to_dict(mode="json")
        restored = SimpleConfig.from_dict(data, mode="json")

        assert hash(original) == hash(restored)

    def test_roundtrip_json_string(self):
        """JSON string roundtrip."""
        original = SimpleConfig(name="test", value=42)
        json_str = original.to_json()
        restored = SimpleConfig.from_json(json_str)

        assert hash(original) == hash(restored)

    def test_nested_model_roundtrip(self):
        """Nested HashableModel roundtrip."""
        inner = SimpleConfig(name="inner", value=1)
        original = NestedConfig(config=inner, enabled=True)

        json_str = original.to_json()
        restored = NestedConfig.from_json(json_str)

        assert hash(original) == hash(restored)
        assert restored.config.name == "inner"
        assert restored.config.value == 1


class TestHashableModelEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_model(self):
        """Model with no required fields."""

        class EmptyModel(HashableModel):
            pass

        m1 = EmptyModel()
        m2 = EmptyModel()

        assert hash(m1) == hash(m2)

    def test_deeply_nested_models(self):
        """Multiple levels of nesting."""

        class Level3(HashableModel):
            value: int

        class Level2(HashableModel):
            inner: Level3

        class Level1(HashableModel):
            inner: Level2

        deep1 = Level1(inner=Level2(inner=Level3(value=42)))
        deep2 = Level1(inner=Level2(inner=Level3(value=42)))

        assert hash(deep1) == hash(deep2)

    def test_frozen_by_default(self):
        """HashableModel is frozen by default to prevent hash corruption."""
        config = SimpleConfig(name="test", value=42)

        with pytest.raises(ValidationError):
            config.value = 99

    def test_mutation_changes_hash_when_unfrozen(self):
        """When explicitly unfrozen, mutating fields changes hash (edge case)."""

        class MutableConfig(HashableModel):
            value: int

            model_config = {**HashableModel.model_config, "frozen": False}

        config = MutableConfig(value=42)
        hash1 = hash(config)

        config.value = 99
        hash2 = hash(config)

        assert hash1 != hash2

    def test_pydantic_equality(self):
        """Pydantic's value equality (independent of hash)."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)

        # Pydantic models are equal by value
        assert c1 == c2
        # And have same hash (content-based)
        assert hash(c1) == hash(c2)

    def test_mixed_basemodel_nesting(self):
        """HashableModel with regular BaseModel nested."""

        class RegularModel(BaseModel):
            data: str

        class MixedModel(HashableModel):
            regular: RegularModel
            value: int

        m1 = MixedModel(regular=RegularModel(data="test"), value=1)
        m2 = MixedModel(regular=RegularModel(data="test"), value=1)

        # Should be hashable and equal
        assert hash(m1) == hash(m2)

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        config = SimpleConfig(name="test", value=42)

        with pytest.raises(ValueError, match="Invalid mode"):
            config.to_dict(mode="invalid")  # type: ignore

        with pytest.raises(ValueError, match="Invalid mode"):
            SimpleConfig.from_dict({"name": "test", "value": 1}, mode="invalid")  # type: ignore


class TestHashableModelProtocols:
    """Protocol implementation verification."""

    def test_implements_serializable(self):
        """HashableModel implements Serializable protocol."""
        config = SimpleConfig(name="test", value=42)

        # Has required methods
        assert hasattr(config, "to_dict")
        assert hasattr(config, "to_json")
        assert callable(config.to_dict)
        assert callable(config.to_json)

    def test_implements_deserializable(self):
        """HashableModel implements Deserializable protocol."""
        # Has required class methods
        assert hasattr(SimpleConfig, "from_dict")
        assert hasattr(SimpleConfig, "from_json")
        assert callable(SimpleConfig.from_dict)
        assert callable(SimpleConfig.from_json)

    def test_implements_hashable(self):
        """HashableModel implements Hashable protocol."""
        config = SimpleConfig(name="test", value=42)

        # Has __hash__
        assert hasattr(config, "__hash__")
        assert callable(config.__hash__)

        # Actually hashable
        hash(config)  # Should not raise

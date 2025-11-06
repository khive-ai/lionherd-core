# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamic Pydantic model generation from JSON schemas."""

import pytest
from pydantic import BaseModel, ValidationError

from lionherd_core import ln

# Check if optional dependency is available
_HAS_SCHEMA_GEN = ln.is_import_installed("datamodel_code_generator")

pytestmark = pytest.mark.skipif(
    not _HAS_SCHEMA_GEN,
    reason="datamodel-code-generator not installed (optional dependency)",
)


class TestLoadPydanticModelFromSchema:
    """Test load_pydantic_model_from_schema function."""

    def test_simple_schema_dict(self):
        """Load model from simple dict schema."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "User",
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }

        UserModel = load_pydantic_model_from_schema(schema)

        assert issubclass(UserModel, BaseModel)
        assert UserModel.__name__ == "User"

        user = UserModel(name="Alice", age=30)
        assert user.name == "Alice"
        assert user.age == 30

        with pytest.raises(ValidationError):
            UserModel(age=25)  # missing required name

    def test_simple_schema_string(self):
        """Load model from JSON string schema."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = """{
            "title": "Product",
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "price": {"type": "number"}
            },
            "required": ["id", "name"]
        }"""

        ProductModel = load_pydantic_model_from_schema(schema)

        assert ProductModel.__name__ == "Product"

        product = ProductModel(id=1, name="Widget", price=9.99)
        assert product.id == 1
        assert product.name == "Widget"
        assert product.price == 9.99

    def test_nested_schema(self):
        """Load model with nested objects."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Company",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "string"},
                    },
                },
            },
        }

        CompanyModel = load_pydantic_model_from_schema(schema)

        company = CompanyModel(
            name="Acme Corp", address={"street": "123 Main St", "city": "Springfield"}
        )
        assert company.name == "Acme Corp"
        assert company.address.street == "123 Main St"
        assert company.address.city == "Springfield"

    def test_schema_with_array(self):
        """Load model with array fields."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Team",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "members": {"type": "array", "items": {"type": "string"}},
            },
        }

        TeamModel = load_pydantic_model_from_schema(schema)

        team = TeamModel(name="Engineering", members=["Alice", "Bob", "Charlie"])
        assert team.name == "Engineering"
        assert len(team.members) == 3
        assert team.members[0] == "Alice"

    def test_schema_with_enum(self):
        """Load model with enum constraints."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Task",
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            },
        }

        TaskModel = load_pydantic_model_from_schema(schema)

        task = TaskModel(title="Write tests", status="in_progress")
        # Generator creates Enum, check value
        assert task.status.value == "in_progress"  # type: ignore[attr-defined]

        with pytest.raises(ValidationError):
            TaskModel(title="Invalid", status="invalid_status")

    def test_custom_model_name(self):
        """Generator falls back to 'Model' when schema has no title."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {"type": "object", "properties": {"value": {"type": "string"}}}

        Model = load_pydantic_model_from_schema(schema, "CustomModel")

        # Generator ignores model_name param, uses 'Model' fallback
        assert Model.__name__ == "Model"
        instance = Model(value="test")
        assert instance.value == "test"

    def test_schema_title_with_spaces(self):
        """Handle schema titles with spaces."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "My Custom Model",
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }

        Model = load_pydantic_model_from_schema(schema)

        assert Model.__name__ == "MyCustomModel"

    def test_schema_title_with_special_chars(self):
        """Sanitize schema titles with special characters."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "User@Profile#123",
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        Model = load_pydantic_model_from_schema(schema)

        # Special chars should be stripped
        assert Model.__name__ == "UserProfile123"

    def test_invalid_schema_dict(self):
        """Raise on invalid schema dict."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        # Non-serializable dict
        invalid_schema = {"properties": {"func": lambda x: x}}

        with pytest.raises(ValueError, match="Invalid dictionary"):
            load_pydantic_model_from_schema(invalid_schema)

    def test_invalid_schema_string(self):
        """Raise on invalid JSON string."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        invalid_json = "{invalid json"

        with pytest.raises(ValueError, match="Invalid JSON schema"):
            load_pydantic_model_from_schema(invalid_json)

    def test_invalid_schema_type(self):
        """Raise on invalid schema type."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        with pytest.raises(TypeError, match="Schema must be"):
            load_pydantic_model_from_schema(123)  # type: ignore[arg-type]

    def test_model_rebuild_with_forward_refs(self):
        """Ensure model_rebuild resolves forward references."""
        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {
            "title": "Node",
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
                "children": {"type": "array", "items": {"$ref": "#"}},
            },
        }

        NodeModel = load_pydantic_model_from_schema(schema)

        # Should handle recursive structure
        node = NodeModel(value=1, children=[{"value": 2, "children": []}])
        assert node.value == 1
        assert len(node.children) == 1
        assert node.children[0].value == 2


class TestImportErrorHandling:
    """Test behavior when optional dependency is missing."""

    def test_import_error_message(self, monkeypatch):
        """Verify helpful error when datamodel-code-generator not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "datamodel_code_generator":
                raise ImportError("No module named 'datamodel_code_generator'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from lionherd_core.libs.schema_handlers import load_pydantic_model_from_schema

        schema = {"title": "Test", "type": "object"}

        with pytest.raises(ImportError, match="datamodel-code-generator not installed"):
            load_pydantic_model_from_schema(schema)


class TestLazyImport:
    """Test lazy import via __getattr__."""

    def test_lazy_import_works(self):
        """Verify function is accessible via lazy import."""
        from lionherd_core.libs import schema_handlers

        # Should not raise even if accessed via __getattr__
        func = schema_handlers.load_pydantic_model_from_schema
        assert callable(func)

    def test_invalid_attribute_raises(self):
        """Verify __getattr__ raises for invalid attributes."""
        from lionherd_core.libs import schema_handlers

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = schema_handlers.nonexistent_function  # type: ignore[attr-defined]

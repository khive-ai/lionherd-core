# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

from lionherd_core.libs.schema_handlers._typescript import typescript_schema


class SimpleModel(BaseModel):
    """Simple model with basic types."""

    name: str = Field(..., description="User name")
    age: int = Field(..., description="User age")
    email: str | None = Field(None, description="Email address")


class ModelWithDefaults(BaseModel):
    """Model with default values."""

    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Result limit")
    enabled: bool = Field(True, description="Feature enabled")
    tags: list[str] | None = Field(None, description="Filter tags")


class ModelWithEnums(BaseModel):
    """Model with enum fields."""

    model_config = {
        "json_schema_extra": {
            "properties": {
                "status": {"enum": ["active", "inactive", "pending"]},
                "priority": {"enum": ["low", "medium", "high"]},
            }
        }
    }

    status: str = Field(..., description="Status")
    priority: str = Field("medium", description="Priority level")


class ModelWithArrays(BaseModel):
    """Model with array types."""

    tags: list[str] = Field(..., description="Tag list")
    scores: list[int] | None = Field(None, description="Score array")
    nested: list[dict] | None = Field(None, description="Nested objects")


class TestTypeScriptSchema:
    """Test TypeScript-style schema formatter."""

    def test_simple_types(self):
        """Test basic type formatting."""
        schema = SimpleModel.model_json_schema()
        result = typescript_schema(schema)

        assert "name: string" in result
        assert "age: int" in result
        assert "email?: string | null" in result
        assert "User name" in result
        assert "User age" in result

    def test_default_values(self):
        """Test default value formatting."""
        schema = ModelWithDefaults.model_json_schema()
        result = typescript_schema(schema)

        assert "query: string" in result  # Required, no default
        assert "limit?: int = 10" in result  # Optional with default
        assert "enabled?: bool = true" in result  # Boolean default
        assert "tags?: string[] | null" in result  # Optional array

    def test_array_types(self):
        """Test array type formatting."""
        schema = ModelWithArrays.model_json_schema()
        result = typescript_schema(schema)

        assert "tags: string[]" in result  # Required array
        assert "scores?: int[] | null" in result  # Optional array
        assert "nested?: object[] | null" in result  # Nested objects

    def test_descriptions_included(self):
        """Test that descriptions are included."""
        schema = SimpleModel.model_json_schema()
        result = typescript_schema(schema)

        assert "- User name" in result
        assert "- User age" in result
        assert "- Email address" in result

    def test_required_vs_optional(self):
        """Test required vs optional field detection."""
        schema = ModelWithDefaults.model_json_schema()
        result = typescript_schema(schema)

        # Required field has no ?
        assert "query: string -" in result
        # Optional fields have ?
        assert "limit?: int" in result
        assert "enabled?: bool" in result

    def test_compact_output(self):
        """Test that output is compact (no JSON verbosity)."""
        schema = SimpleModel.model_json_schema()
        result = typescript_schema(schema)

        # Should NOT contain JSON Schema keywords
        assert "$ref" not in result
        assert "properties" not in result
        assert "required" not in result
        assert "type" not in result.split(":")[0]  # Not as keyword

        # Should be concise
        lines = result.split("\n")
        assert len(lines) == 3  # One line per field

    def test_null_handling(self):
        """Test null/None type handling."""
        schema = SimpleModel.model_json_schema()
        result = typescript_schema(schema)

        # Optional fields should show nullable
        assert "email?: string | null" in result

    def test_boolean_defaults(self):
        """Test boolean default value formatting."""
        schema = ModelWithDefaults.model_json_schema()
        result = typescript_schema(schema)

        assert "= true" in result  # Not "= True"

    def test_empty_schema(self):
        """Test handling of empty schema."""

        class EmptyModel(BaseModel):
            pass

        schema = EmptyModel.model_json_schema()
        result = typescript_schema(schema)

        assert result == ""  # No properties = empty output

    def test_nested_models(self):
        """Test handling of nested Pydantic models."""

        class Address(BaseModel):
            street: str
            city: str

        class User(BaseModel):
            name: str = Field(..., description="User name")
            address: Address | None = Field(None, description="User address")

        schema = User.model_json_schema()
        result = typescript_schema(schema)

        assert "name: string" in result
        assert "address?: Address | null" in result  # Should reference by name

    def test_multiline_format(self):
        """Test that each field is on its own line."""
        schema = SimpleModel.model_json_schema()
        result = typescript_schema(schema)

        lines = result.split("\n")
        assert len(lines) == 3  # Three fields
        assert all(line.strip() for line in lines)  # No empty lines

    def test_token_reduction(self):
        """Test that TypeScript format is significantly shorter than JSON Schema."""
        schema = ModelWithDefaults.model_json_schema()

        import json

        json_format = json.dumps(schema, indent=2)
        ts_format = typescript_schema(schema)

        # TypeScript should be much shorter
        assert len(ts_format) < len(json_format) * 0.3  # At least 70% reduction
        assert ts_format.count("\n") < json_format.count("\n") * 0.2  # Fewer lines


class TestTypeScriptSchemaEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_description(self):
        """Test fields without descriptions."""

        class NoDescModel(BaseModel):
            field: str

        schema = NoDescModel.model_json_schema()
        result = typescript_schema(schema)

        # Should still work, just no description
        assert "field: string" in result

    def test_complex_unions(self):
        """Test complex union types."""

        class UnionModel(BaseModel):
            value: str | int | None = Field(None, description="Multi-type field")

        schema = UnionModel.model_json_schema()
        result = typescript_schema(schema)

        # Should handle union of multiple types
        assert "value?" in result
        assert "Multi-type field" in result

    def test_indentation(self):
        """Test indentation parameter."""
        schema = SimpleModel.model_json_schema()

        result_indent_0 = typescript_schema(schema, indent=0)
        result_indent_2 = typescript_schema(schema, indent=2)

        # Indent 0 should have no leading spaces
        assert not result_indent_0.startswith(" ")

        # Indent 2 should have 4 spaces (2 * 2)
        assert result_indent_2.startswith("    ")

    def test_enum_with_integer_values(self):
        """Test enum fields with integer values."""
        schema = {
            "properties": {
                "priority": {
                    "enum": [1, 2, 3, None],
                    "description": "Priority level",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should format integers correctly
        assert "priority?: 1 | 2 | 3 | null" in result

    def test_enum_with_boolean_values(self):
        """Test enum fields with boolean values."""
        schema = {
            "properties": {
                "flag": {
                    "enum": [True, False],
                    "description": "Flag value",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should format booleans correctly
        assert "flag?: True | False" in result

    def test_enum_field_required(self):
        """Test required enum field."""
        schema = {
            "properties": {
                "status": {
                    "enum": ["active", "inactive", "pending"],
                    "description": "Status",
                }
            },
            "required": ["status"],
        }
        result = typescript_schema(schema)

        # Required enum should not have ?
        assert 'status: "active" | "inactive" | "pending"' in result

    def test_ref_field(self):
        """Test $ref field handling."""
        schema = {
            "properties": {
                "user": {
                    "$ref": "#/definitions/User",
                    "description": "User reference",
                }
            },
            "required": ["user"],
        }
        result = typescript_schema(schema)

        # Should extract reference name
        assert "user: User" in result

    def test_ref_field_optional(self):
        """Test optional $ref field."""
        schema = {
            "properties": {
                "address": {
                    "$ref": "#/definitions/Address",
                    "description": "Address reference",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Optional ref should have ?
        assert "address?: Address" in result

    def test_array_with_ref_items(self):
        """Test array with $ref items."""
        schema = {
            "properties": {
                "users": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/User"},
                    "description": "User list",
                }
            },
            "required": ["users"],
        }
        result = typescript_schema(schema)

        # Should format as User[]
        assert "users: User[]" in result

    def test_array_with_enum_items(self):
        """Test array with enum items."""
        schema = {
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"enum": ["tag1", "tag2", "tag3"]},
                    "description": "Tag list",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should format enum in array
        assert "tags?: " in result
        assert "[]" in result

    def test_anyof_with_ref(self):
        """Test anyOf with $ref."""
        schema = {
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"$ref": "#/definitions/CustomType"},
                    ],
                    "description": "Value field",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should handle anyOf with ref
        assert "value?: " in result
        assert "CustomType" in result

    def test_anyof_with_enum(self):
        """Test anyOf with enum."""
        schema = {
            "properties": {
                "status": {
                    "anyOf": [
                        {"enum": ["active", "inactive"]},
                        {"type": "null"},
                    ],
                    "description": "Status field",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should handle anyOf with enum
        assert "status?: " in result

    def test_anyof_with_array_ref(self):
        """Test anyOf with array containing $ref."""
        schema = {
            "properties": {
                "items": {
                    "anyOf": [
                        {"type": "array", "items": {"$ref": "#/definitions/Item"}},
                        {"type": "null"},
                    ],
                    "description": "Items field",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should handle array with ref in anyOf
        assert "items?: " in result

    def test_type_with_question_suffix(self):
        """Test type field with ? suffix."""
        schema = {
            "properties": {
                "value": {
                    "type": "string?",
                    "description": "Optional string",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should handle type with ?
        assert "value?: string" in result

    def test_fallback_any_type(self):
        """Test fallback to any type."""
        schema = {
            "properties": {
                "unknown": {
                    "description": "Unknown field",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should fallback to any
        assert "unknown?: any" in result

    def test_string_default_value(self):
        """Test string default value formatting."""
        schema = {
            "properties": {
                "name": {
                    "type": "string",
                    "default": "John Doe",
                    "description": "User name",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should format string default with quotes
        assert 'name?: string = "John Doe"' in result

    def test_null_default_value(self):
        """Test null default value formatting."""
        schema = {
            "properties": {
                "value": {
                    "type": "string",
                    "default": None,
                    "description": "Nullable value",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should format null default
        assert "value?: string = null" in result

    def test_integer_default_value(self):
        """Test integer default value formatting."""
        schema = {
            "properties": {
                "count": {
                    "type": "integer",
                    "default": 42,
                    "description": "Count value",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should format integer default
        assert "count?: int = 42" in result

    def test_array_items_no_spec(self):
        """Test array with items having no type/ref/enum."""
        schema = {
            "properties": {
                "data": {
                    "type": "array",
                    "items": {},  # No type, ref, or enum
                    "description": "Data array",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should fallback to any[]
        assert "data?: any[]" in result

    def test_schema_without_properties(self):
        """Test schema without properties field."""
        schema = {
            "description": "Schema with no properties",
        }
        result = typescript_schema(schema)

        # Should return empty string
        assert result == ""

    def test_anyof_enum_with_null(self):
        """Test anyOf with enum containing null."""
        schema = {
            "properties": {
                "status": {
                    "anyOf": [
                        {"enum": ["active", "inactive", None]},
                    ],
                    "description": "Status field",
                }
            },
            "required": [],
        }
        result = typescript_schema(schema)

        # Should handle enum with null in anyOf
        assert "status?: " in result
        assert "null" in result or "active" in result

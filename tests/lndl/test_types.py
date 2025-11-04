# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from lionherd_core.lndl.types import LNDLOutput, ParsedConstructor


def test_lndloutput_internal_attribute_access():
    """Test LNDLOutput internal attribute access."""
    output = LNDLOutput(
        fields={"key": "value"}, lvars={}, lacts={}, actions={}, raw_out_block="raw"
    )

    # Access internal attributes directly (line 50)
    assert object.__getattribute__(output, "fields") == {"key": "value"}
    assert object.__getattribute__(output, "lvars") == {}
    assert object.__getattribute__(output, "raw_out_block") == "raw"


class TestPhase3TypesCoverage:
    """Phase 3: Tests for remaining uncovered lines."""

    def test_parsed_constructor_has_dict_unpack(self):
        """Test has_dict_unpack property."""
        # With dict unpack
        ctor = ParsedConstructor(
            class_name="Report", kwargs={"**dict": {"key": "value"}}, raw="Report(**dict)"
        )
        assert ctor.has_dict_unpack is True

        # Without dict unpack
        ctor2 = ParsedConstructor(
            class_name="Report", kwargs={"key": "value"}, raw="Report(key='value')"
        )
        assert ctor2.has_dict_unpack is False

    def test_lndloutput_getattr_field_access(self):
        """Test LNDLOutput field access via __getattr__."""

        class TestModel(BaseModel):
            value: str

        output = LNDLOutput(
            fields={"my_field": TestModel(value="test")},
            lvars={},
            lacts={},
            actions={},
            raw_out_block="raw",
        )

        # Test line 51: __getattr__ returns from fields dict
        accessed = output.my_field
        assert accessed.value == "test"

        # Test line 50: defensive check for internal attributes
        # Call __getattr__ directly to test the special case
        fields_result = output.__getattr__("fields")
        assert fields_result is output.fields

        lvars_result = output.__getattr__("lvars")
        assert lvars_result is output.lvars

        raw_result = output.__getattr__("raw_out_block")
        assert raw_result is output.raw_out_block

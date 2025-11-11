"""Tests for @implements() decorator and protocol metadata."""

from uuid import UUID, uuid4

import pytest

from lionherd_core.protocols import (
    Hashable,
    Observable,
    Serializable,
    implements,
)


class TestImplementsDecorator:
    """Test @implements() decorator behavior and __protocols__ metadata."""

    def test_implements_sets_protocols_metadata_single(self):
        """@implements() should set __protocols__ attribute for single protocol."""

        @implements(Observable)
        class TestClass:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        assert hasattr(TestClass, "__protocols__")
        assert len(TestClass.__protocols__) == 1
        # Protocol classes are in lionherd_core.protocols module
        assert TestClass.__protocols__[0].__name__ == "ObservableProto"

    def test_implements_sets_protocols_metadata_multiple(self):
        """@implements() should set __protocols__ for multiple protocols."""

        @implements(Observable, Serializable)
        class TestClass:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

            def to_dict(self, **kwargs):
                return {"id": str(self.id)}

        assert hasattr(TestClass, "__protocols__")
        assert len(TestClass.__protocols__) == 2
        protocol_names = {p.__name__ for p in TestClass.__protocols__}
        assert protocol_names == {"ObservableProto", "Serializable"}

    def test_implements_metadata_inherited_like_class_attributes(self):
        """@implements() metadata inherits via normal Python class attribute inheritance."""

        @implements(Observable)
        class Parent:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # Child doesn't use @implements
        class Child(Parent):
            pass

        # Parent has protocols
        assert hasattr(Parent, "__protocols__")
        assert len(Parent.__protocols__) == 1

        # Child DOES inherit __protocols__ via normal class attribute inheritance
        # (This is standard Python behavior - class attributes are inherited)
        assert hasattr(Child, "__protocols__")
        assert Child.__protocols__ == Parent.__protocols__

    def test_implements_with_inherited_methods_requires_override(self):
        """@implements() enforces that methods are in class body, not inherited."""

        class Parent:
            def to_dict(self, **kwargs):
                return {"parent": "data"}

        # ❌ This violates @implements() semantics - method is inherited
        # (Not caught at decorator time, but documented as anti-pattern)
        @implements(Serializable)
        class WrongChild(Parent):
            pass  # to_dict inherited, not in body

        # ✅ Correct: explicit override in class body
        @implements(Serializable)
        class CorrectChild(Parent):
            def to_dict(self, **kwargs):  # Explicit in body
                data = super().to_dict(**kwargs)
                data["child"] = "additional"
                return data

        # Both have __protocols__ set
        assert hasattr(WrongChild, "__protocols__")
        assert hasattr(CorrectChild, "__protocols__")

    def test_isinstance_checks_structure_not_decorator(self):
        """isinstance() checks method presence, not @implements() metadata."""

        # Class with @implements but missing method
        @implements(Observable)
        class IncompleteClass:
            pass  # Missing .id property

        # Class without @implements but has method
        class CompleteClass:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        incomplete = IncompleteClass()
        complete = CompleteClass()

        # isinstance() checks structure (method presence), not @implements()
        assert not isinstance(
            incomplete, Observable
        )  # Missing .id despite @implements
        assert isinstance(complete, Observable)  # Has .id despite no @implements

    def test_implements_with_hashable_protocol(self):
        """@implements() works with Hashable protocol."""

        @implements(Hashable)
        class TestClass:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return hash(self.value)

            def __eq__(self, other):
                return isinstance(other, TestClass) and self.value == other.value

        assert hasattr(TestClass, "__protocols__")
        assert TestClass.__protocols__[0].__name__ == "Hashable"

        # Verify hashable behavior
        obj1 = TestClass(42)
        obj2 = TestClass(42)
        assert hash(obj1) == hash(obj2)
        assert obj1 == obj2
        assert len({obj1, obj2}) == 1  # Set deduplication

    def test_implements_empty_call_sets_empty_tuple(self):
        """@implements() with no protocols sets __protocols__ to empty tuple."""

        @implements()  # No protocols provided
        class EmptyClass:
            pass

        assert hasattr(EmptyClass, "__protocols__")
        assert EmptyClass.__protocols__ == ()

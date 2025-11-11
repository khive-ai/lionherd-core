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

    def test_implements_raises_when_method_inherited_not_defined(self):
        """@implements() should raise TypeError when method is inherited, not defined in class body."""

        class Parent:
            def to_dict(self, **kwargs):
                return {"parent": "data"}

        # ❌ This violates @implements() semantics - method is inherited
        with pytest.raises(
            TypeError,
            match=r"WrongChild declares @implements\(Serializable\) but does not define 'to_dict' in its class body",
        ):

            @implements(Serializable)
            class WrongChild(Parent):
                pass  # to_dict inherited, not in body - should raise!

    def test_implements_allows_explicit_override(self):
        """@implements() should allow methods defined in class body (even if calling super)."""

        class Parent:
            def to_dict(self, **kwargs):
                return {"parent": "data"}

        # ✅ Correct: explicit override in class body
        @implements(Serializable)
        class CorrectChild(Parent):
            def to_dict(self, **kwargs):  # Explicit in body
                data = super().to_dict(**kwargs)
                data["child"] = "additional"
                return data

        # Should succeed without raising
        assert hasattr(CorrectChild, "__protocols__")
        assert Serializable in CorrectChild.__protocols__

    def test_implements_raises_when_property_inherited(self):
        """@implements() should raise TypeError when property is inherited, not defined in class body."""

        class Parent:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # ❌ Property inherited, not defined in Child
        with pytest.raises(
            TypeError,
            match=r"WrongChild declares @implements\(ObservableProto\) but does not define 'id' in its class body",
        ):

            @implements(Observable)
            class WrongChild(Parent):
                pass

    def test_implements_allows_property_in_class_body(self):
        """@implements() should allow properties defined in class body."""

        class Parent:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # ✅ Property explicitly defined in child
        @implements(Observable)
        class CorrectChild(Parent):
            @property
            def id(self) -> UUID:
                return self._id

        assert hasattr(CorrectChild, "__protocols__")
        assert Observable in CorrectChild.__protocols__

    def test_implements_raises_when_classmethod_inherited(self):
        """@implements() should raise TypeError when classmethod is inherited."""
        from lionherd_core.protocols import Deserializable

        class Parent:
            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        # ❌ Classmethod inherited
        with pytest.raises(
            TypeError,
            match=r"WrongChild declares @implements\(Deserializable\) but does not define 'from_dict' in its class body",
        ):

            @implements(Deserializable)
            class WrongChild(Parent):
                pass

    def test_implements_allows_classmethod_in_class_body(self):
        """@implements() should allow classmethods defined in class body."""
        from lionherd_core.protocols import Deserializable

        class Parent:
            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        # ✅ Classmethod explicitly defined
        @implements(Deserializable)
        class CorrectChild(Parent):
            @classmethod
            def from_dict(cls, data, **kwargs):
                instance = super().from_dict(data, **kwargs)
                return instance

        assert hasattr(CorrectChild, "__protocols__")
        assert Deserializable in CorrectChild.__protocols__

    def test_implements_validates_all_protocol_methods(self):
        """@implements() should validate ALL methods required by protocol."""

        # ✅ All methods defined
        @implements(Serializable)
        class Complete:
            def to_dict(self, **kwargs):
                return {}

        # ❌ Missing required method
        with pytest.raises(
            TypeError,
            match=r"Incomplete declares @implements\(Serializable\) but does not define 'to_dict' in its class body",
        ):

            @implements(Serializable)
            class Incomplete:
                pass

    def test_implements_validates_multiple_protocols(self):
        """@implements() should validate all methods for multiple protocols."""

        # ✅ All methods for both protocols defined
        @implements(Observable, Serializable)
        class CompleteMulti:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

            def to_dict(self, **kwargs):
                return {"id": str(self.id)}

        # ❌ Missing method from second protocol
        with pytest.raises(TypeError, match=r"but does not define 'to_dict' in its class body"):

            @implements(Observable, Serializable)
            class IncompleteMult:
                def __init__(self):
                    self._id = uuid4()

                @property
                def id(self) -> UUID:
                    return self._id

                # Missing to_dict!

    def test_isinstance_checks_structure_not_decorator(self):
        """isinstance() checks method presence, not @implements() metadata."""

        # Class without @implements but has method
        class CompleteClass:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # Class without @implements and missing method
        class IncompleteClass:
            pass  # Missing .id property

        incomplete = IncompleteClass()
        complete = CompleteClass()

        # isinstance() checks structure (method presence), not @implements()
        assert not isinstance(incomplete, Observable)  # Missing .id, no @implements
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

    def test_implements_validates_pydantic_fields_via_annotations(self):
        """@implements() should recognize Pydantic fields via __annotations__."""
        from pydantic import BaseModel

        # ✅ Pydantic field in __annotations__ satisfies Observable protocol
        @implements(Observable)
        class PydanticObservable(BaseModel):
            id: UUID  # Field annotation (no @property needed)

        # Should succeed without raising
        assert hasattr(PydanticObservable, "__protocols__")
        assert Observable in PydanticObservable.__protocols__

        # Verify it actually works
        instance = PydanticObservable(id=uuid4())
        assert isinstance(instance.id, UUID)

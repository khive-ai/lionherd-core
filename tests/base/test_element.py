# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Element base class test suite: identity, polymorphic serialization, validation, security.

Design Philosophy:
    Element is the foundational abstraction for all lionherd entities. Its design enforces
    three critical architectural principles:

    1. Immutable Identity: UUID + timestamp frozen at creation (entity lifecycle tracking)
    2. Polymorphic Serialization: lion_class metadata enables type-safe deserialization
    3. Security-First: Validation prevents arbitrary code execution via malicious metadata

Test Architecture:
    Tests organized by concern domains:
    - Instantiation: Auto-generation, defaults, custom values, validation
    - Serialization: python/json modes, lion_class injection, field exclusion
    - Deserialization: Type coercion, roundtrips, polymorphic routing
    - Validation: Frozen fields, timezone/UUID flexibility, metadata mutability
    - Subclassing: Inheritance, automatic registration, type preservation
    - Polymorphism: lion_class routing, security checks, recursion prevention
    - Security: Arbitrary code execution prevention, equality/hashing semantics

Mathematical Properties Verified:
    - Identity: ∀e: Element, e.id ∈ UUID ∧ e.created_at ∈ DateTime(UTC)
    - Immutability: ∀e: Element, ¬(e.id := new_id) ∧ ¬(e.created_at := new_ts)
    - Roundtrip Fidelity: ∀e, from_dict(to_dict(e)) ≡ e (by ID equality)
    - Polymorphic Correctness: ∀S <: Element, isinstance(from_dict(S().to_dict()), S)
    - Hash Stability: ∀e, hash(e) = hash(e.id) (constant over element lifetime)

Why These Tests Matter:
    Element is inherited by Node, Event, Flow, Message - all core abstractions. Bugs here
    cascade to the entire system. These tests protect against:
    - Identity corruption (broken UUID/timestamp generation)
    - Serialization loss (missing lion_class breaks polymorphism)
    - Security vulnerabilities (arbitrary code exec via malicious lion_class)
    - Hash instability (breaks sets/dicts if fields used instead of ID)
    - Type confusion (deserialization returns wrong subclass)

Test Data Strategy:
    Module-level test subclasses (PersonElement, DocumentElement, etc.) enable dynamic
    import via load_type_from_string() for polymorphic deserialization testing.
"""

import datetime as dt
from uuid import UUID, uuid4

import orjson
import pytest
from pydantic import ValidationError

from lionherd_core.base import Element

# Module-level test Element subclasses for polymorphic deserialization tests
# (must be module-level so they can be dynamically imported via load_type_from_string)


class PersonElement(Element):
    """Test subclass with custom fields."""

    name: str
    age: int


class DocumentElement(Element):
    """Test subclass for document-like objects."""

    title: str
    content: str


class CustomElement(Element):
    """Test subclass with simple value."""

    value: str


class TaggedElement(Element):
    """Test subclass with tag field."""

    tag: str


class SimpleElement(Element):
    """Test subclass that uses inherited from_dict."""

    value: str


class SpecificElement(Element):
    """Test subclass for no-lion_class test."""

    field: str


class TestElementInstantiation:
    """Element instantiation: auto-generation, custom values, validation boundaries.

    Design Rationale:
        Element's constructor implements the "sensible defaults with override" pattern.
        Auto-generation of UUID and timestamp enables creation without ceremony while
        supporting deterministic testing via custom values.

    Invariants Protected:
        - ∀e: Element(), isinstance(e.id, UUID) ∧ e.id.version == 4 (default)
        - ∀e: Element(), e.created_at.tzinfo == UTC ∧ e.created_at ≈ now()
        - ∀e: Element(), e.metadata == {} (empty dict, not None)
        - Element(id=x) ⇒ e.id == x (custom identity preserved)
        - Element(unknown_field=x) ⇒ ValidationError (Pydantic extra='forbid')

    Architecture:
        Uses Pydantic Field(default_factory=...) for lazy generation:
        - uuid4() called per instance (not shared sentinel)
        - datetime.now(UTC) called per instance (not import-time constant)
        - dict() creates new empty dict per instance (not shared mutable default)

    Edge Cases Covered:
        - Timezone flexibility: Accepts naive datetime (coerced to UTC) and any timezone
        - UUID flexibility: Accepts any UUID version (1/3/4/5), not just UUID4
        - Metadata flexibility: Empty dict default, but accepts any dict content
        - Validation: Rejects unknown fields (prevents typos/API drift)

    Why This Matters:
        Without auto-generation, every Element creation requires boilerplate. Without
        frozen ID/timestamp, entity identity can drift. Without validation, typos pass
        silently. These tests prevent all three failure modes.
    """

    def test_auto_generated_id(self):
        """ID should be auto-generated UUID."""
        elem = Element()
        assert isinstance(elem.id, UUID)

    def test_auto_generated_created_at(self):
        """created_at should be auto-generated UTC datetime with millisecond precision.

        Architecture: Uses default_factory=lambda: datetime.now(dt.UTC) for per-instance
        generation. Ensures timezone-aware timestamps (not naive datetime).

        Test strategy: Capture before/after timestamps to verify elem.created_at falls
        within execution window. Validates both auto-generation and UTC enforcement.
        """
        before = dt.datetime.now(dt.UTC)
        elem = Element()
        after = dt.datetime.now(dt.UTC)

        assert isinstance(elem.created_at, dt.datetime)
        assert elem.created_at.tzinfo == dt.UTC
        assert before <= elem.created_at <= after

    def test_empty_metadata_default(self):
        """metadata should default to empty dict."""
        elem = Element()
        assert elem.metadata == {}

    def test_custom_id(self):
        """Can provide custom UUID."""
        custom_id = uuid4()
        elem = Element(id=custom_id)
        assert elem.id == custom_id

    def test_custom_created_at(self):
        """Can provide custom timestamp."""
        custom_time = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        elem = Element(created_at=custom_time)
        assert elem.created_at == custom_time

    def test_custom_metadata(self):
        """Can provide custom metadata."""
        meta = {"key": "value", "count": 42}
        elem = Element(metadata=meta)
        assert elem.metadata == meta

    def test_extra_fields_forbidden(self):
        """Should reject unknown fields (Pydantic extra='forbid' enforcement).

        Design rationale: Fail-fast on typos and API drift. Without this, typos like
        Element(metdata={...}) silently create wrong field. Strict validation catches
        mistakes at creation time, not later during serialization.

        Edge case: Prevents accidental forward compatibility (new field added to class
        but not to old deserialization code). Forces explicit schema evolution.
        """
        with pytest.raises(ValidationError):
            Element(unknown_field="value")


class TestElementSerialization:
    """Element serialization: python/json/db modes, lion_class injection, field control.

    Design Rationale:
        Multi-mode serialization enables seamless interop across contexts:
        - python mode: Native types (UUID, datetime) for in-memory processing
        - json mode: String types for API responses and file storage
        - db mode: Renames metadata → node_metadata for Neo4j compatibility

        lion_class injection is the foundation of polymorphic deserialization. Without it,
        Element.from_dict() cannot determine the correct subclass to instantiate.

    Invariants Protected:
        - ∀e, "lion_class" ∈ e.to_dict()["metadata"] (injection always happens)
        - ∀e, e.to_dict(mode="python")["id"] ∈ UUID (native types preserved)
        - ∀e, e.to_dict(mode="json")["id"] ∈ str (JSON-safe types)
        - ∀e, orjson.loads(e.to_json()) == e.to_dict(mode="json") (consistency)
        - ∀e, e.to_dict(exclude={"metadata"}) → "metadata" ∉ result (Pydantic exclusion works)

    Architecture:
        - python mode: Direct model_dump() with lion_class post-injection
        - json mode: to_json() → orjson.loads() roundtrip for nested object handling
        - db mode: Same as json + metadata key renaming
        - lion_class: Fully qualified name (module.Class) for unambiguous import

    Format Options:
        - to_json(pretty=True): Human-readable with indentation
        - to_json(sort_keys=True): Deterministic key order for testing/hashing
        - to_dict(exclude={...}): Selective field omission (Pydantic passthrough)

    Why This Matters:
        Without mode separation, UUID/datetime serialization breaks JSON APIs. Without
        lion_class injection, polymorphic deserialization is impossible. Without field
        exclusion, sensitive data leaks. These tests prevent all three.
    """

    def test_to_dict_python_mode(self):
        """to_dict(mode='python') returns native types (UUID, datetime objects).

        Architecture: Python mode preserves native types for in-memory processing. No
        coercion to strings - UUID and datetime objects passed through unchanged.

        Critical: lion_class ALWAYS injected into metadata during serialization. This
        enables polymorphic deserialization. Without it, Element.from_dict() cannot
        determine correct subclass.

        Use case: Inter-process communication via pickle, in-memory caching, testing.
        """
        elem = Element()
        data = elem.to_dict(mode="python")

        assert isinstance(data["id"], UUID)
        assert isinstance(data["created_at"], dt.datetime)
        # lion_class injected during serialization
        assert data["metadata"]["lion_class"] == "lionherd_core.base.element.Element"

    def test_to_dict_json_mode(self):
        """to_dict(mode='json') returns JSON-serializable types (all strings/primitives).

        Architecture: JSON mode uses to_json() → orjson.loads() roundtrip to handle
        nested objects correctly. This ensures UUID → str, datetime → ISO8601 string.

        Critical: lion_class survives JSON roundtrip. Without it, polymorphic
        deserialization breaks for JSON-serialized data.

        Format: ISO8601 for datetime (standard), hex string for UUID (RFC 4122).
        Verifies parsing works by constructing UUID/datetime from strings.

        Use case: REST APIs, file storage (JSON documents), message queues.
        """
        elem = Element()
        data = elem.to_dict(mode="json")

        # Should be strings after orjson roundtrip
        assert isinstance(data["id"], str)
        assert isinstance(data["created_at"], str)
        # lion_class preserved through JSON mode
        assert data["metadata"]["lion_class"] == "lionherd_core.base.element.Element"

        # Verify can parse back
        UUID(data["id"])  # Should not raise
        dt.datetime.fromisoformat(data["created_at"])  # Should not raise

    def test_to_dict_invalid_mode(self):
        """to_dict with invalid mode should raise."""
        elem = Element()
        with pytest.raises(ValueError, match="Invalid mode"):
            elem.to_dict(mode="invalid")

    def test_to_json_basic(self):
        """to_json returns valid JSON string."""
        elem = Element()
        json_str = elem.to_json()

        # Should be valid JSON
        parsed = orjson.loads(json_str)
        assert "id" in parsed
        assert "created_at" in parsed
        assert "metadata" in parsed

    def test_to_json_pretty(self):
        """to_json(pretty=True) should format with indentation."""
        elem = Element()
        json_str = elem.to_json(pretty=True)

        assert "\n" in json_str
        assert "  " in json_str  # Indentation

    def test_to_json_sort_keys(self):
        """to_json(sort_keys=True) should sort keys alphabetically."""
        elem = Element(metadata={"z": 1, "a": 2})
        json_str = elem.to_json(sort_keys=True)

        # Keys should be sorted
        keys = list(orjson.loads(json_str).keys())
        assert keys == sorted(keys)

    def test_to_dict_exclude(self):
        """to_dict should support Pydantic exclude."""
        elem = Element(metadata={"secret": "value"})
        data = elem.to_dict(exclude={"metadata"})

        assert "id" in data
        assert "created_at" in data
        assert "metadata" not in data

    def test_created_at_format_python_mode_defaults(self):
        """Python mode defaults to datetime object for created_at."""
        elem = Element()
        data = elem.to_dict(mode="python")
        assert isinstance(data["created_at"], dt.datetime)

    def test_created_at_format_python_mode_isoformat(self):
        """Python mode with created_at_format='isoformat' returns ISO string."""
        elem = Element()
        data = elem.to_dict(mode="python", created_at_format="isoformat")
        assert isinstance(data["created_at"], str)
        # Verify it's valid ISO format
        dt.datetime.fromisoformat(data["created_at"])

    def test_created_at_format_python_mode_timestamp(self):
        """Python mode with created_at_format='timestamp' returns float."""
        elem = Element()
        data = elem.to_dict(mode="python", created_at_format="timestamp")
        assert isinstance(data["created_at"], float)
        # Verify it's a valid timestamp
        assert data["created_at"] > 0

    def test_created_at_format_json_mode_defaults(self):
        """JSON mode defaults to isoformat string for created_at."""
        elem = Element()
        data = elem.to_dict(mode="json")
        assert isinstance(data["created_at"], str)
        # Verify it's valid ISO format
        dt.datetime.fromisoformat(data["created_at"])

    def test_created_at_format_json_mode_timestamp(self):
        """JSON mode with created_at_format='timestamp' returns float."""
        elem = Element()
        data = elem.to_dict(mode="json", created_at_format="timestamp")
        assert isinstance(data["created_at"], float)
        assert data["created_at"] > 0

    def test_created_at_format_json_mode_datetime_raises(self):
        """JSON mode with created_at_format='datetime' should raise (not JSON-serializable)."""
        elem = Element()
        with pytest.raises(
            ValueError, match="created_at_format='datetime' not valid for mode='json'"
        ):
            elem.to_dict(mode="json", created_at_format="datetime")

    def test_created_at_format_db_mode_defaults(self):
        """DB mode defaults to datetime object for created_at."""
        elem = Element()
        data = elem.to_dict(mode="db")
        assert isinstance(data["created_at"], dt.datetime)
        # Verify metadata renamed to node_metadata in db mode
        assert "node_metadata" in data
        assert "metadata" not in data

    def test_created_at_format_db_mode_isoformat(self):
        """DB mode with created_at_format='isoformat' returns ISO string."""
        elem = Element()
        data = elem.to_dict(mode="db", created_at_format="isoformat")
        assert isinstance(data["created_at"], str)
        dt.datetime.fromisoformat(data["created_at"])

    def test_created_at_format_db_mode_timestamp(self):
        """DB mode with created_at_format='timestamp' returns float."""
        elem = Element()
        data = elem.to_dict(mode="db", created_at_format="timestamp")
        assert isinstance(data["created_at"], float)
        assert data["created_at"] > 0

    def test_regression_json_mode_respects_timestamp_format(self):
        """Regression: json mode must apply created_at_format='timestamp' transformation.

        Original bug: created_at_format only applied to python mode, json mode ignored it.
        This test verifies json mode converts to timestamp when requested.
        """
        elem = Element()

        # json mode with timestamp format should return float, not ISO string
        data_timestamp = elem.to_dict(mode="json", created_at_format="timestamp")
        assert isinstance(data_timestamp["created_at"], float), (
            "json mode must apply timestamp format transformation"
        )

        # Compare to default (isoformat) - should be different types
        data_default = elem.to_dict(mode="json")
        assert isinstance(data_default["created_at"], str), (
            "json mode default should be isoformat string"
        )

        # Verify timestamp is actually the datetime as float
        expected_timestamp = elem.created_at.timestamp()
        assert abs(data_timestamp["created_at"] - expected_timestamp) < 0.001

    def test_regression_db_mode_respects_format_parameter(self):
        """Regression: db mode must apply created_at_format transformations.

        Original bug: db mode defaulted to isoformat and ignored format parameter.
        This test verifies db mode correctly applies all format options.
        """
        elem = Element()

        # db mode default should be datetime (not isoformat)
        data_default = elem.to_dict(mode="db")
        assert isinstance(data_default["created_at"], dt.datetime), (
            "db mode default must be datetime object"
        )

        # db mode with isoformat should convert to string
        data_iso = elem.to_dict(mode="db", created_at_format="isoformat")
        assert isinstance(data_iso["created_at"], str), (
            "db mode must apply isoformat transformation"
        )

        # db mode with timestamp should convert to float
        data_ts = elem.to_dict(mode="db", created_at_format="timestamp")
        assert isinstance(data_ts["created_at"], float), (
            "db mode must apply timestamp transformation"
        )

    def test_regression_all_modes_apply_format_consistently(self):
        """Regression: all modes must consistently apply created_at_format parameter.

        Original bug: only python mode applied format transformation, json/db ignored it.
        This test verifies format parameter works across all modes.
        """
        elem = Element()

        # timestamp format should return float for ALL modes (except json with datetime)
        python_ts = elem.to_dict(mode="python", created_at_format="timestamp")
        json_ts = elem.to_dict(mode="json", created_at_format="timestamp")
        db_ts = elem.to_dict(mode="db", created_at_format="timestamp")

        assert isinstance(python_ts["created_at"], float)
        assert isinstance(json_ts["created_at"], float)
        assert isinstance(db_ts["created_at"], float)

        # All should be approximately equal
        assert abs(python_ts["created_at"] - json_ts["created_at"]) < 0.001
        assert abs(python_ts["created_at"] - db_ts["created_at"]) < 0.001


class TestElementDeserialization:
    """Element deserialization: type coercion, roundtrip fidelity, partial data handling.

    Design Rationale:
        from_dict() must accept both native types (in-memory) and string types (from JSON).
        This dual-mode support enables seamless data flow: DB → Element → API → Element.

        Pydantic validators handle coercion, but roundtrip fidelity requires careful
        coordination between to_dict() and from_dict() - especially for lion_class metadata.

    Invariants Protected:
        - ∀e, from_dict(e.to_dict(mode="python")) ≡ e (by ID, roundtrip fidelity)
        - ∀e, from_dict(e.to_dict(mode="json")) ≡ e (string coercion works)
        - from_dict({"id": str(uuid)}) → Element(id=uuid) (string → UUID coercion)
        - from_dict({"created_at": isoformat}) → Element(created_at=dt) (ISO → datetime)
        - from_dict({}) → Element(id=auto, created_at=auto) (defaults applied)

    Architecture:
        - Pydantic validators (_coerce_id, _coerce_created_at) handle type conversion
        - model_validate() provides the Pydantic validation pipeline
        - lion_class metadata extracted and removed before validation (serialization-only field)
        - meta_key parameter enables db mode deserialization (node_metadata → metadata)

    Edge Cases Covered:
        - Minimal data: from_dict({}) generates defaults (no required fields)
        - String coercion: UUID/datetime strings automatically parsed
        - Native types: UUID/datetime objects passed through unchanged
        - Metadata restoration: node_metadata → metadata for db mode compatibility

    Why This Matters:
        Without coercion, JSON deserialization breaks. Without roundtrip fidelity,
        serialization becomes lossy. Without default handling, API clients must provide
        all fields. These tests prevent all three failure modes.
    """

    def test_from_dict_with_uuid_objects(self):
        """from_dict should accept native UUID and datetime."""
        elem_id = uuid4()
        created = dt.datetime.now(dt.UTC)
        data = {
            "id": elem_id,
            "created_at": created,
            "metadata": {"key": "value"},
        }

        elem = Element.from_dict(data)
        assert elem.id == elem_id
        assert elem.created_at == created
        assert elem.metadata == {"key": "value"}

    def test_from_dict_with_strings(self):
        """from_dict should parse UUID and datetime strings."""
        elem_id = uuid4()
        created = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)
        data = {
            "id": str(elem_id),
            "created_at": created.isoformat(),
            "metadata": {},
        }

        elem = Element.from_dict(data)
        assert elem.id == elem_id
        assert elem.created_at == created

    def test_from_dict_minimal(self):
        """from_dict with minimal data should use defaults."""
        data = {}
        elem = Element.from_dict(data)

        assert isinstance(elem.id, UUID)
        assert isinstance(elem.created_at, dt.datetime)
        assert elem.metadata == {}

    def test_from_dict_roundtrip_python(self):
        """Roundtrip through to_dict(mode='python') and from_dict."""
        original = Element(metadata={"test": "value"})
        data = original.to_dict(mode="python")
        restored = Element.from_dict(data)

        assert restored.id == original.id
        assert restored.created_at == original.created_at
        assert restored.metadata == original.metadata

    def test_from_dict_roundtrip_json(self):
        """Roundtrip through to_dict(mode='json') and from_dict."""
        original = Element(metadata={"test": "value"})
        data = original.to_dict(mode="json")
        restored = Element.from_dict(data)

        assert restored.id == original.id
        assert restored.created_at == original.created_at
        assert restored.metadata == original.metadata

    def test_from_dict_with_non_dict_metadata(self):
        """Non-dict metadata disables polymorphism; Pydantic coerces or rejects.

        Pattern:
            Graceful handling of malformed serialization data

        Edge Case:
            metadata field is string/int instead of dict (malformed data)

        Expected:
            Element created but lion_class extraction fails (no polymorphism)

        Design Rationale:
            Safety vs Strictness trade-off:
            - Strict: Reject non-dict metadata → ValueError
            - Safe: Accept and coerce → Element with non-dict metadata
            Current implementation: Safe (Pydantic coercion)

        Use Case:
            Legacy data migration where metadata format changed
            External systems providing non-standard serialization

        Complexity:
            O(1) metadata validation
        """
        # Create data with metadata as string (not dict)
        data = {"metadata": "not a dict"}

        # Should handle non-dict metadata (lion_class = None, no polymorphism)
        elem = Element.from_dict(data)

        # Should still create element, just no polymorphism
        assert isinstance(elem, Element)

    def test_from_json_basic(self):
        """from_json() wraps orjson.loads + from_dict; accepts str or bytes.

        Pattern:
            Convenience method for JSON deserialization (reduces boilerplate)

        Expected:
            One-step deserialization: JSON string → Element

        Design Rationale:
            Ergonomics: Avoid orjson.loads() boilerplate in calling code
            Consistency: Mirror to_json() for symmetry
            Type safety: Ensures JSON → Element (not JSON → dict)

        Use Case:
            API responses: response.text → Element
            File loading: json_file.read() → Element
            Message queues: message.body → Element

        Complexity:
            O(n) JSON parsing where n = JSON string length
        """

        # Create element and serialize to JSON
        elem = Element(metadata={"key": "value"})
        json_str = elem.to_json()

        # Deserialize using from_json convenience method
        restored = Element.from_json(json_str)

        assert restored.id == elem.id
        assert restored.metadata["key"] == "value"

    def test_from_json_with_bytes(self):
        """from_json() accepts bytes (orjson returns bytes by default).

        Pattern:
            Type flexibility for common JSON library behavior

        Edge Case:
            orjson.dumps() returns bytes, not str (unlike json.dumps)

        Expected:
            from_json(bytes) → Element (no decode step required)

        Design Rationale:
            Compatibility: orjson.dumps() default is bytes
            Ergonomics: Avoid manual decode in calling code
            Performance: orjson operates on bytes natively

        Use Case:
            Direct deserialization from orjson.dumps() output
            Network protocols sending JSON as bytes
            File I/O reading JSON as binary

        Complexity:
            O(n) JSON parsing where n = bytes length
        """
        import orjson

        elem = Element(metadata={"test": "data"})

        # Serialize to JSON bytes
        json_bytes = elem.to_json(decode=False)
        assert isinstance(json_bytes, bytes)

        # Deserialize from bytes
        restored = Element.from_json(json_bytes)
        assert restored.id == elem.id


class TestElementRepr:
    """Element string representation: diagnostic format for logging and debugging.

    Design Rationale:
        __repr__() provides a concise identity representation for logging, REPL inspection,
        and error messages. Format: ClassName(id=UUID) reveals both type and identity
        without overwhelming detail.

        Design choice: Show ID (immutable identity) but not created_at or metadata
        (too verbose for most debugging). If full detail needed, use to_dict().

    Invariants Protected:
        - ∀e, repr(e).startswith(e.__class__.__name__) (type visible)
        - ∀e, str(e.id) in repr(e) (identity visible)
        - ∀e, repr(e) follows "ClassName(id=...)" format (consistency)

    Architecture:
        Simple string interpolation: f"{class.__name__}(id={self.id})". No complex
        logic needed - immutable fields guarantee repr stability.

    Why This Matters:
        Without informative __repr__(), debugging multi-element systems becomes painful.
        Log messages show <Element object at 0x...> instead of Element(id=abc-123).
        This test ensures diagnostic value.
    """

    def test_repr_format(self):
        """__repr__ should show class name and id."""
        elem = Element()
        repr_str = repr(elem)

        assert "Element" in repr_str
        assert str(elem.id) in repr_str
        assert repr_str.startswith("Element(id=")


class TestElementValidation:
    """Element validation: frozen identity, coercion, mutability, flexibility boundaries.

    Design Rationale:
        Validation strategy balances three goals:
        1. Identity immutability: ID and timestamp frozen after creation (entity semantics)
        2. Data flexibility: Accept multiple UUID versions, any timezone (pragmatic interop)
        3. Metadata mutability: Dict can be modified in-place (workflow state tracking)

        Pydantic frozen=True on id/created_at prevents accidental identity corruption.
        Validators coerce but don't reject valid alternatives (UUID5 vs UUID4, EST vs UTC).

    Invariants Protected:
        - ∀e: Element(), e.id := new_id ⇒ ValidationError (frozen enforcement)
        - ∀e: Element(), e.created_at := new_ts ⇒ ValidationError (frozen enforcement)
        - ∀e: Element(id=uuid1()) → e.id.version == 1 (any UUID version accepted)
        - ∀e: Element(created_at=naive_dt) → e.created_at.tzinfo == UTC (naive → UTC coercion)
        - ∀e: Element(created_at=est_dt) → e.created_at.tzinfo != UTC (any tz accepted)
        - ∀e: Element(), e.metadata["key"] = value (in-place mutation allowed)

    Architecture:
        - Pydantic Field(frozen=True): Compile-time immutability enforcement
        - @field_validator(mode="before"): Pre-validation coercion pipeline
        - validate_assignment=True: Runtime validation on attribute assignment
        - extra="forbid": Reject unknown fields (fail-fast on typos)

    Coercion Strategy:
        - UUID: Accepts string/UUID/int, converts to UUID object via to_uuid()
        - Timestamp: Naive datetime coerced to UTC (convenience), any timezone accepted (flexibility)
        - Metadata: Dict validation via to_dict() with suppression (graceful fallback)

    Why This Matters:
        Without frozen fields, identity can drift (e.id = other_id). Without coercion,
        JSON deserialization breaks. Without flexibility, valid alternatives rejected.
        Without mutability, workflow state can't be tracked. These tests protect all four.
    """

    def test_validate_assignment(self):
        """Should validate on attribute assignment (validate_assignment=True)."""
        elem = Element()

        # Should reject invalid UUID (frozen field)
        with pytest.raises(ValidationError):
            elem.id = "not-a-uuid"

    def test_identity_fields_frozen(self):
        """Identity fields (id, created_at) must be immutable (frozen=True)."""
        elem = Element()

        # Cannot reassign id even with valid UUID
        with pytest.raises(ValidationError, match="frozen"):
            elem.id = uuid4()

        # Cannot reassign created_at even with valid datetime
        with pytest.raises(ValidationError, match="frozen"):
            elem.created_at = dt.datetime.now(dt.UTC)

    def test_metadata_mutability(self):
        """Metadata dict should be mutable."""
        elem = Element()
        elem.metadata["new_key"] = "new_value"
        assert elem.metadata["new_key"] == "new_value"

    def test_created_at_coerces_naive_to_utc(self):
        """Naive datetimes coerced to UTC for convenience (not rejected).

        Design rationale: Pragmatic interop. Many datetime operations produce naive
        datetimes. Rejecting them creates friction. Instead, assume UTC (sensible
        default for server timestamps).

        Alternative rejected: Raise ValueError on naive datetime (too strict, breaks
        common patterns like datetime(2025, 1, 1, 12, 0, 0) for testing).

        Edge case: Ambiguous semantics - naive datetime could mean local time or UTC.
        We choose UTC for consistency with created_at default_factory.
        """
        # Naive datetime is automatically coerced to UTC
        naive_dt = dt.datetime(2025, 1, 1, 12, 0, 0)
        element = Element(created_at=naive_dt)
        assert element.created_at.tzinfo == dt.UTC
        assert element.created_at == dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt.UTC)

    def test_created_at_accepts_any_timezone(self):
        """created_at accepts any timezone (not just UTC)."""
        # Eastern timezone
        import zoneinfo

        eastern = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=zoneinfo.ZoneInfo("US/Eastern"))
        elem = Element(created_at=eastern)
        assert elem.created_at.tzinfo is not None

    def test_id_accepts_any_uuid_version(self):
        """id accepts any valid UUID (not just UUID4)."""
        from uuid import NAMESPACE_DNS, uuid1, uuid3, uuid4, uuid5

        # UUID4 (default)
        elem4 = Element(id=uuid4())
        assert elem4.id.version == 4

        # UUID1 accepted
        elem1 = Element(id=uuid1())
        assert elem1.id.version == 1

        # UUID3 accepted
        elem3 = Element(id=uuid3(NAMESPACE_DNS, "test"))
        assert elem3.id.version == 3

        # UUID5 accepted
        elem5 = Element(id=uuid5(NAMESPACE_DNS, "test"))
        assert elem5.id.version == 5

    def test_id_accepts_uuid4(self):
        """id accepts valid UUID4."""
        custom_id = uuid4()
        elem = Element(id=custom_id)
        assert elem.id == custom_id
        assert elem.id.version == 4

    def test_validate_metadata_with_non_dict_convertible(self):
        """Metadata validator auto-converts objects with to_dict() method.

        Pattern:
            Duck-typing for dict-like objects (structural typing)

        Edge Case:
            Object with to_dict() method passed as metadata

        Expected:
            Validator calls to_dict() and stores resulting dict

        Design Rationale:
            Ergonomics: Allows passing Pydantic models, dataclasses, custom objects
            without manual conversion. Validator handles coercion transparently.

        Use Case:
            Element(metadata=my_pydantic_model) → auto-converts to dict
            Common when wrapping external objects as metadata.

        Complexity:
            O(1) to_dict() call + dict validation
        """

        class DictLike:
            """Object with to_dict method for duck-typing test."""

            def to_dict(self):
                return {"converted": True, "nested": {"value": 123}}

        # Pass object that can be converted to dict
        # Validator should call to_dict() automatically
        elem = Element(metadata=DictLike())

        # Should have converted metadata
        assert elem.metadata == {"converted": True, "nested": {"value": 123}}


class TestElementSubclass:
    """Element subclassing: inheritance, method preservation, type fidelity.

    Design Rationale:
        Element serves as the base for Node, Event, Flow, Message. Subclasses must
        inherit both data (id, created_at, metadata) and behavior (to_dict, from_dict).

        from_dict() on subclass must return subclass instance, not Element. This
        requires Pydantic class methods to be polymorphic (cls.model_validate, not
        Element.model_validate).

    Invariants Protected:
        - ∀S <: Element, S().id ∈ UUID (inherited field works)
        - ∀S <: Element, S().to_dict()["lion_class"].endswith(S.__name__) (correct class name)
        - ∀S <: Element, isinstance(S.from_dict({...}), S) (from_dict returns correct type)
        - ∀S <: Element, S().to_json() works (inherited method works)

    Architecture:
        Standard Python inheritance with Pydantic model_validate():
        - Inherited fields: Pydantic merges field definitions from base + subclass
        - Inherited methods: Python MRO (Method Resolution Order) handles lookups
        - Type preservation: cls parameter in classmethods ensures correct type

    Why This Matters:
        Without inheritance, every subclass must reimplement id/created_at/metadata.
        Without type preservation, from_dict(PersonElement(...)) returns Element.
        Without method inheritance, to_dict/to_json must be reimplemented. These
        tests prevent all three failure modes.

    Test Strategy:
        Use inline class definitions (class MyEntity(Element): ...) to verify
        inheritance works without registration boilerplate.
    """

    def test_subclass_inheritance(self):
        """Subclass should inherit Element fields and methods."""

        class MyEntity(Element):
            name: str
            value: int

        entity = MyEntity(name="test", value=42)

        # Inherited fields
        assert isinstance(entity.id, UUID)
        assert isinstance(entity.created_at, dt.datetime)

        # Inherited methods
        data = entity.to_dict(mode="python")
        assert data["name"] == "test"
        assert data["value"] == 42

        json_str = entity.to_json()
        assert "test" in json_str

    def test_subclass_from_dict(self):
        """Subclass from_dict should create correct type."""

        class MyEntity(Element):
            name: str

        data = {"name": "test"}
        entity = MyEntity.from_dict(data)

        assert isinstance(entity, MyEntity)
        assert entity.name == "test"


class TestElementPolymorphicDeserialization:
    """Polymorphic deserialization: lion_class routing, security, recursion prevention.

    Design Rationale:
        Polymorphic deserialization solves a fundamental problem: Given serialized data
        from an Element subclass, how do we restore the correct subclass type?

        Solution: lion_class metadata (injected during to_dict) specifies the fully
        qualified class name. from_dict() imports that class dynamically and delegates
        deserialization to it.

        This enables serialization round-trip fidelity: ∀S <: Element, from_dict(S().to_dict()) ∈ S

    Invariants Protected:
        - ∀S <: Element, isinstance(from_dict(S().to_dict()), S) (type restoration)
        - lion_class == "nonexistent.Class" → ValueError (import failure detected)
        - lion_class == "builtins.dict" → ValueError (non-Element rejected)
        - ∀S: S.from_dict == Element.from_dict, no recursion (same impl detection)
        - from_dict({}) uses calling class (no lion_class → cls.model_validate)

    Architecture:
        Dynamic class loading via load_type_from_string():
        1. Parse "module.path.ClassName" into module + class name
        2. importlib.import_module(module) loads module
        3. getattr(module, classname) retrieves class object
        4. issubclass(cls, Element) validates security
        5. cls.from_dict(data) delegates deserialization

        Recursion prevention:
        - Compare target_cls.from_dict.__func__ vs cls.from_dict.__func__
        - If same implementation, use model_validate() instead of recursive from_dict()

    Security Model:
        Three layers of defense against arbitrary code execution:
        1. Import validation: load_type_from_string raises ValueError on import failure
        2. Type validation: issubclass(cls, Element) rejects non-Element classes
        3. Method validation: from_dict() callable check prevents attribute injection

        Attack vector blocked: {"metadata": {"lion_class": "os.system"}}
        → ValueError("'os.system' is not an Element subclass")

    Edge Cases Covered:
        - Unknown class: Import failure raises ValueError with diagnostic message
        - Non-Element class: Security check rejects with clear error
        - Metadata preservation: Custom metadata survives polymorphic roundtrip
        - No lion_class: Falls back to calling class (SpecificElement.from_dict → SpecificElement)
        - Recursion: Same from_dict impl uses model_validate to avoid infinite loop

    Why This Matters:
        Without polymorphism, deserialization always returns Element (type information lost).
        Without security checks, malicious data can execute arbitrary code. Without
        recursion prevention, simple subclasses hit stack overflow. These tests protect
        against all three failure modes.

    Test Data Strategy:
        Module-level classes (PersonElement, DocumentElement, etc.) enable dynamic import
        via load_type_from_string("tests.base.test_element.PersonElement").
    """

    def test_polymorphic_with_lion_class(self):
        """from_dict should deserialize to correct subclass via lion_class."""
        # Create and serialize
        person = PersonElement(name="Alice", age=30)
        data = person.to_dict(mode="python")

        # Deserialize via base class - should get PersonElement
        restored = Element.from_dict(data)
        assert isinstance(restored, PersonElement)
        assert restored.name == "Alice"
        assert restored.age == 30

    def test_polymorphic_roundtrip_json(self):
        """Polymorphic deserialization should work through JSON."""
        doc = DocumentElement(title="Test", content="Body")
        json_data = doc.to_dict(mode="json")

        # Deserialize via Element.from_dict
        restored = Element.from_dict(json_data)
        assert isinstance(restored, DocumentElement)
        assert restored.title == "Test"

    def test_polymorphic_with_full_qualified_name(self):
        """Should work with both short and fully qualified class names."""
        obj = CustomElement(value="test")
        data = obj.to_dict(mode="python")

        # Should contain fully qualified name
        assert "." in data["metadata"]["lion_class"]
        assert data["metadata"]["lion_class"].endswith("CustomElement")

        # Should deserialize correctly
        restored = Element.from_dict(data)
        assert isinstance(restored, CustomElement)

    def test_polymorphic_unknown_class_error(self):
        """Should raise error for unknown lion_class."""
        data = {"metadata": {"lion_class": "nonexistent.module.FakeClass"}}

        with pytest.raises(ValueError, match="Failed to deserialize"):
            Element.from_dict(data)

    def test_polymorphic_non_element_subclass_rejected(self):
        """Should reject non-Element classes (security: prevent arbitrary code execution).

        Attack vector: Malicious data provides {"metadata": {"lion_class": "os.system"}}.
        Without validation, dynamic import loads os.system and attempts deserialization.

        Defense: issubclass(loaded_class, Element) check rejects non-Element classes
        before any instantiation occurs. Fail-fast with clear error message.

        Test strategy: Use builtins.dict (loads successfully but not Element subclass)
        to verify security check works without triggering import errors.
        """
        # Test with a fake class name that would load to a non-Element class
        # Dynamic import will try to load 'builtins.dict' which is not an Element
        data = {"metadata": {"lion_class": "builtins.dict"}}

        with pytest.raises(ValueError, match="not an Element subclass"):
            Element.from_dict(data)

    def test_polymorphic_preserves_metadata(self):
        """Should preserve custom metadata during polymorphic deserialization."""
        elem = TaggedElement(tag="important", metadata={"priority": "high", "owner": "alice"})
        data = elem.to_dict(mode="python")

        restored = Element.from_dict(data)
        assert isinstance(restored, TaggedElement)
        assert restored.metadata["priority"] == "high"
        assert restored.metadata["owner"] == "alice"
        # lion_class should be removed from metadata after deserialization
        assert "lion_class" not in restored.metadata

    def test_polymorphic_recursion_prevention(self):
        """Should prevent infinite recursion when subclass inherits from_dict.

        Problem: SimpleElement doesn't override from_dict, so it inherits Element.from_dict.
        Without recursion detection: Element.from_dict sees lion_class="...SimpleElement",
        loads SimpleElement, calls SimpleElement.from_dict, which is Element.from_dict,
        which loads SimpleElement again → infinite loop.

        Solution: Compare target_cls.from_dict.__func__ vs cls.from_dict.__func__. If
        same implementation, use model_validate() directly (skip from_dict recursion).

        Edge case: Subclasses that override from_dict won't trigger this (different __func__).
        Only affects subclasses that inherit from_dict without changes.
        """
        obj = SimpleElement(value="test")
        data = obj.to_dict(mode="python")

        # Should use model_validate instead of recursing
        restored = Element.from_dict(data)
        assert isinstance(restored, SimpleElement)
        assert restored.value == "test"

    def test_polymorphic_no_lion_class_uses_target_class(self):
        """Without lion_class, should use the calling class."""
        data = {"field": "value"}  # No lion_class in metadata

        # Calling SpecificElement.from_dict should create SpecificElement
        restored = SpecificElement.from_dict(data)
        assert isinstance(restored, SpecificElement)
        assert restored.field == "value"


class TestElementSecurity:
    """Test security validations and element semantics.

    Why this matters: Element is the foundational base class and must be secure
    against malicious input and have well-defined equality/hashing semantics.

    Security concerns:

    1. Arbitrary code execution via lion_class
       - Threat: Attacker provides {"metadata": {"lion_class": "os.system"}}
       - Defense: Validate lion_class is Element subclass before instantiation
       - Test: test_polymorphic_non_element_subclass_rejected

    Equality and hashing semantics:

    1. Identity-based equality (__eq__ by ID)
       - Why: Matches Python object identity semantics
       - Consequence: Two Elements with same field values but different IDs are NOT equal
       - Use case: Elements as dict keys (hash stable even if fields change)

    2. Immutable hash (__hash__ by ID)
       - Why: ID is frozen (immutable), so hash never changes
       - Consequence: Safe to use in sets and as dict keys
       - Alternative rejected: Hash by fields (breaks if fields change)

    3. Always truthy (__bool__)
       - Why: Elements represent entities (entities always "exist")
       - Consequence: if elem: always True (even if fields are empty)

    Test coverage:
    - class_name() returns correct short/full name
    - Equality by ID (not by values)
    - Hashing by ID (stable, works in sets)
    - Always truthy (even if empty)
    """

    def test_class_name_method(self):
        """class_name() should return correct format."""
        # Short name
        assert Element.class_name(full=False) == "Element"

        # Full name
        full_name = Element.class_name(full=True)
        assert "." in full_name
        assert full_name.endswith("Element")
        assert "lionherd_core" in full_name

    def test_dunder_equality_by_id(self):
        """Elements equal by ID (identity), not by field values (structural equality).

        Design rationale: Element represents entities (persistent objects with identity).
        Entity equality is identity-based, not value-based. Two persons named "Alice"
        are different people.

        Alternative rejected: Value-based equality (elem1 == elem2 if all fields match).
        This breaks when used in sets/dicts and fields change over time.

        Consequence: Element(id=x, metadata={...}) == Element(id=x, metadata={}) is True.
        If structural equality needed, compare to_dict() directly.

        Mathematical property: ∀e1, e2: (e1.id == e2.id) ⟺ (e1 == e2)
        """
        elem1 = Element(metadata={"key": "value1"})
        elem2 = Element(id=elem1.id, metadata={"key": "value2"})
        elem3 = Element(metadata={"key": "value1"})

        # Same ID = equal
        assert elem1 == elem2

        # Different ID = not equal
        assert elem1 != elem3

    def test_dunder_hash_by_id(self):
        """Elements hash by ID (immutable identity) for safe use in sets/dicts.

        Design rationale: Hash must be stable (never change during object lifetime).
        ID is frozen=True (immutable), so hash(element) = hash(element.id) is stable.

        Alternative rejected: Hash by field values. If fields change, hash changes,
        breaking sets/dicts (can't find element after modification).

        Consequence: Elements safe as dict keys: d[elem] = value won't break if
        elem.metadata changes. Hash stability guaranteed by ID immutability.

        Mathematical property: ∀e1, e2: (e1 == e2) ⟹ (hash(e1) == hash(e2))
        (hash consistency with equality)
        """
        elem1 = Element()
        elem2 = Element(id=elem1.id)
        elem3 = Element()

        # Same ID = same hash
        assert hash(elem1) == hash(elem2)

        # Different ID = (probably) different hash
        assert hash(elem1) != hash(elem3)

        # Should work in sets
        s = {elem1, elem2, elem3}
        assert len(s) == 2  # elem1 and elem2 are same

    def test_dunder_bool_always_true(self):
        """Elements should always be truthy."""
        elem = Element()
        assert bool(elem) is True
        assert elem  # Should be truthy in conditionals

    def test_eq_with_non_element(self):
        """Returns NotImplemented (not False) to allow reverse comparison per Python protocol.

        Pattern:
            Rich comparison protocol with NotImplemented sentinel

        Edge Case:
            Element compared to non-Element types (str, int, dict, None)

        Expected:
            elem.__eq__(other) returns NotImplemented
            Python interpreter interprets as False

        Design Rationale:
            Python comparison protocol:
            - Return NotImplemented → allows reverse comparison (other.__eq__(elem))
            - Return False → blocks reverse comparison
            NotImplemented enables symmetric comparison with custom types.

        Use Case:
            Extensibility: Custom types can define __eq__ to compare with Element
            Example: MyType.__eq__(elem) called if elem.__eq__(myobj) returns NotImplemented

        Complexity:
            O(1) type check
        """
        elem = Element()

        # Compare with non-Element types
        # Should return NotImplemented, which Python interprets as False
        assert elem != "string"
        assert elem != 123
        assert elem is not None
        assert elem != {"dict": "value"}

        # Verify it returns NotImplemented (not False directly)
        result = elem.__eq__("not an element")
        assert result is NotImplemented

    def test_eq_with_different_element_types(self):
        """Same ID equals even across different Element subclasses.

        Pattern:
            Identity-based equality across type hierarchy

        Edge Case:
            Element and CustomElement (subclass) with same ID

        Expected:
            Equal if IDs match, regardless of class difference

        Design Rationale:
            Entity identity transcends implementation details:
            - Same entity can have different representations (base vs subclass)
            - ID is the source of truth for identity
            - Type checking would break polymorphic equality

        Use Case:
            Heterogeneous collections: List[Element] with mixed types
            Identity tracking: Same entity represented as different types
            Deserialization: Original type lost but ID preserved

        Complexity:
            O(1) ID comparison
        """

        class CustomElement(Element):
            value: int = 0

        elem1 = Element()
        custom = CustomElement(id=elem1.id, value=42)

        # Same ID = equal, even if different types
        assert elem1 == custom
        assert custom == elem1


# ==================== class_name() Integration Tests ====================


def test_class_name_strips_generics():
    """Test class_name() strips generic type parameters using typing.get_origin().

    Design Rationale:
        Generic classes (Flow[E, P]) need to serialize as base class name (Flow)
        for polymorphic deserialization. Using typing.get_origin() is robust and
        handles all generic forms correctly.

    Edge Cases:
        - Non-generic classes: Element -> Element
        - Generic classes: Flow[Item, Prog] -> Flow
        - Nested generics: Pile[Flow[Item, Prog]] -> Pile
    """
    from lionherd_core.base import Flow, Pile, Progression

    # Non-generic Element
    assert Element.class_name(full=False) == "Element"
    assert Element.class_name(full=True) == "lionherd_core.base.element.Element"

    # Generic Flow without parameters
    assert Flow.class_name(full=False) == "Flow"
    assert Flow.class_name(full=True) == "lionherd_core.base.flow.Flow"

    # Generic Flow with parameters
    flow_generic = Flow[Element, Progression]
    assert flow_generic.class_name(full=False) == "Flow"
    assert flow_generic.class_name(full=True) == "lionherd_core.base.flow.Flow"

    # Generic Pile
    pile_generic = Pile[Element]
    assert pile_generic.class_name(full=False) == "Pile"
    assert pile_generic.class_name(full=True) == "lionherd_core.base.pile.Pile"


def test_class_name_serialization_across_subclasses():
    """Integration test: class_name() enables correct polymorphic deserialization.

    Tests that class_name() works correctly across the Element hierarchy:
    - Element, Flow, Pile, Node, Graph, Event
    - Generic and non-generic forms
    - Serialization round-trip preserves type

    This verifies the critic's concern: class_name() affects 9+ subclasses,
    not just Flow.
    """
    from lionherd_core.base import Flow, Node, Pile, Progression
    from lionherd_core.ln import to_dict

    # Test Element
    elem = Element()
    elem_data = to_dict(elem)
    assert elem_data["metadata"]["lion_class"] == "lionherd_core.base.element.Element"
    elem2 = Element.from_dict(elem_data)
    assert isinstance(elem2, Element)
    assert elem2.id == elem.id

    # Test Flow (generic)
    flow = Flow[Element, Progression](name="test_flow")
    flow_data = to_dict(flow)
    assert flow_data["metadata"]["lion_class"] == "lionherd_core.base.flow.Flow"
    flow2 = Flow.from_dict(flow_data)
    assert isinstance(flow2, Flow)
    assert flow2.name == "test_flow"

    # Test Pile (generic)
    pile = Pile[Element]()
    pile.add(Element())
    pile_data = to_dict(pile)
    assert pile_data["metadata"]["lion_class"] == "lionherd_core.base.pile.Pile"
    pile2 = Pile.from_dict(pile_data)
    assert isinstance(pile2, Pile)
    assert len(pile2) == 1

    # Test Node
    node = Node(content={"value": "test content"})
    node_data = to_dict(node)
    assert node_data["metadata"]["lion_class"] == "lionherd_core.base.node.Node"
    node2 = Node.from_dict(node_data)
    assert isinstance(node2, Node)
    assert node2.content == {"value": "test content"}

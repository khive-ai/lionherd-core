# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Testing utilities for lionherd-core and downstream projects.

This module provides reusable test fixtures, factories, and strategies for
testing code that uses lionherd-core primitives.

Basic usage:
    from lionherd_core.testing import create_test_pile, mock_element

    # In your tests
    def test_my_function():
        pile = create_test_pile(count=10)
        element = mock_element(value=42)
        ...

Property-based testing (requires hypothesis):
    from lionherd_core.testing import element_strategy

    @given(elem=element_strategy())
    def test_element_property(elem):
        assert elem.id is not None
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .base import Edge, Element, Event, Flow, Graph, Node, Pile, Processor, Progression

if TYPE_CHECKING:
    pass

__all__ = (
    "FailingTestEvent",
    "SimpleTestEvent",
    "SlowTestEvent",
    "StreamingTestEvent",
    "TestProcessor",
    "create_cyclic_graph",
    "create_dag_graph",
    "create_empty_graph",
    "create_simple_graph",
    "create_spec",
    "create_test_elements",
    "create_test_flow",
    "create_test_pile",
    "create_test_progression",
    "create_typed_pile",
    "element_strategy",
    "get_sample_hashable_models",
    "get_sample_lndl_text",
    "get_sample_params_classes",
    "get_sample_pydantic_models",
    "get_sample_validators",
    "mock_element",
    "mock_node",
    "node_strategy",
    "progression_strategy",
    "validate_and_dump_pydantic",
)


# =============================================================================
# Element Factories
# =============================================================================


class TestElement(Element):
    """Simple Element subclass for testing."""

    value: int = 0
    name: str = "test"


def mock_element(
    *,
    value: int = 0,
    name: str = "test",
    metadata: dict[str, Any] | None = None,
) -> TestElement:
    """Create a single mock Element for testing.

    Args:
        value: Integer value for the element
        name: Name string for the element
        metadata: Optional metadata dict

    Returns:
        TestElement instance

    Example:
        >>> elem = mock_element(value=42, name="test1")
        >>> assert elem.value == 42
        >>> assert elem.name == "test1"
    """
    elem = TestElement(value=value, name=name)
    if metadata:
        elem.metadata.update(metadata)
    return elem


def create_test_elements(count: int = 5, start_value: int = 0) -> list[TestElement]:
    """Create a list of test Elements.

    Args:
        count: Number of elements to create
        start_value: Starting value for sequential values

    Returns:
        List of TestElement instances

    Example:
        >>> elements = create_test_elements(count=3, start_value=10)
        >>> assert len(elements) == 3
        >>> assert elements[0].value == 10
        >>> assert elements[2].value == 12
    """
    return [TestElement(value=start_value + i, name=f"element_{i}") for i in range(count)]


# =============================================================================
# Node Factories
# =============================================================================


def mock_node(
    *,
    content: dict[str, Any] | BaseModel | None = None,
    value: str = "test",
) -> Node:
    """Create a single mock Node for testing.

    Args:
        content: Node content (dict or BaseModel)
        value: Default value if content not provided

    Returns:
        Node instance

    Example:
        >>> node = mock_node(content={"key": "value"})
        >>> assert node.content == {"key": "value"}
    """
    if content is None:
        content = {"value": value}
    return Node(content=content)


def create_test_nodes(count: int = 5) -> list[Node]:
    """Create a list of test Nodes.

    Args:
        count: Number of nodes to create

    Returns:
        List of Node instances

    Example:
        >>> nodes = create_test_nodes(count=3)
        >>> assert len(nodes) == 3
        >>> assert nodes[0].content == {"value": "node_0"}
    """
    return [Node(content={"value": f"node_{i}"}) for i in range(count)]


# =============================================================================
# Event and Processor Classes
# =============================================================================


class SimpleTestEvent(Event):
    """Simple Event that returns a configurable value.

    Example:
        >>> event = SimpleTestEvent(return_value=42)
        >>> result = await event.invoke()
        >>> assert result == 42
    """

    return_value: Any = None
    streaming: bool = False

    async def _invoke(self) -> Any:
        """Return the configured value."""
        return self.return_value


class FailingTestEvent(Event):
    """Event that raises a configurable exception.

    Example:
        >>> event = FailingTestEvent(error_message="Test error")
        >>> await event.invoke()  # Raises ValueError
    """

    error_message: str = "Test error"
    error_type: type[Exception] = ValueError
    streaming: bool = False

    async def _invoke(self) -> Any:
        """Raise configured error."""
        raise self.error_type(self.error_message)


class SlowTestEvent(Event):
    """Event that takes time to complete.

    Example:
        >>> event = SlowTestEvent(delay=0.1, return_value="done")
        >>> result = await event.invoke()
        >>> assert result == "done"
    """

    delay: float = 0.1
    return_value: Any = "completed"
    streaming: bool = False

    async def _invoke(self) -> Any:
        """Wait then return value."""
        import anyio

        await anyio.sleep(self.delay)
        return self.return_value


class StreamingTestEvent(Event):
    """Event that yields values via async generator.

    Example:
        >>> event = StreamingTestEvent(stream_count=3)
        >>> async for value in event.stream():
        ...     print(value)
        0
        1
        2
    """

    stream_count: int = 3
    streaming: bool = True

    async def _invoke(self) -> Any:
        """Not used for streaming events."""
        raise NotImplementedError("Use stream() instead")

    async def stream(self):
        """Yield values asynchronously."""
        import anyio

        from .base.event import EventStatus

        for i in range(self.stream_count):
            await anyio.sleep(0.01)
            yield i
        # Mark as completed after streaming
        self.execution.status = EventStatus.COMPLETED
        self.execution.response = f"streamed {self.stream_count} items"


class TestProcessor(Processor):
    """Basic Processor for SimpleTestEvent.

    Example:
        >>> pile = Pile[Event]()
        >>> processor = TestProcessor(pile=pile)
        >>> event = SimpleTestEvent(return_value=42)
        >>> await processor.process(event)
    """

    event_type = SimpleTestEvent


# =============================================================================
# Pile Factories
# =============================================================================


def create_test_pile(
    count: int = 5,
    *,
    item_type: type | None = None,
    strict_type: bool = False,
) -> Pile[TestElement]:
    """Create a test Pile with mock elements.

    Args:
        count: Number of elements in pile
        item_type: Optional type constraint
        strict_type: Enforce exact type match

    Returns:
        Pile instance with test elements

    Example:
        >>> pile = create_test_pile(count=10)
        >>> assert len(pile) == 10
        >>> assert pile[0].value == 0
    """
    items = create_test_elements(count=count)
    return Pile(items=items, item_type=item_type, strict_type=strict_type)


def create_typed_pile(
    items: list[Element],
    *,
    item_type: type | set[type] | None = None,
    strict_type: bool = False,
) -> Pile:
    """Create a Pile with custom items and type constraints.

    Args:
        items: List of Element instances
        item_type: Type constraint(s)
        strict_type: Enforce exact type match

    Returns:
        Pile instance with provided items

    Example:
        >>> elements = [TestElement(value=i) for i in range(5)]
        >>> pile = create_typed_pile(elements, item_type=TestElement)
        >>> assert len(pile) == 5
    """
    return Pile(items=items, item_type=item_type, strict_type=strict_type)


# =============================================================================
# Graph Factories
# =============================================================================


def create_empty_graph() -> Graph:
    """Create an empty Graph for testing.

    Returns:
        Empty Graph instance

    Example:
        >>> graph = create_empty_graph()
        >>> assert len(graph.nodes) == 0
        >>> assert len(graph.edges) == 0
    """
    return Graph()


def create_simple_graph() -> tuple[Graph, list[Node]]:
    """Create a simple graph with 3 nodes in a chain: A -> B -> C.

    Returns:
        Tuple of (Graph, list of nodes)

    Example:
        >>> graph, nodes = create_simple_graph()
        >>> assert len(graph.nodes) == 3
        >>> assert len(graph.edges) == 2
    """
    graph = Graph()

    # Create nodes
    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    # Create edges
    graph.add_edge(Edge(head=n1.id, tail=n2.id))
    graph.add_edge(Edge(head=n2.id, tail=n3.id))

    return graph, [n1, n2, n3]


def create_cyclic_graph() -> tuple[Graph, list[Node]]:
    """Create a graph with a cycle: A -> B -> C -> A.

    Returns:
        Tuple of (Graph, list of nodes)

    Example:
        >>> graph, nodes = create_cyclic_graph()
        >>> assert not graph.is_acyclic()
    """
    graph = Graph()

    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    # Create cycle
    graph.add_edge(Edge(head=n1.id, tail=n2.id))
    graph.add_edge(Edge(head=n2.id, tail=n3.id))
    graph.add_edge(Edge(head=n3.id, tail=n1.id))  # Cycle back

    return graph, [n1, n2, n3]


def create_dag_graph() -> tuple[Graph, list[Node]]:
    """Create a directed acyclic graph (DAG) with multiple paths.

    Structure:
        A -> B -> D
        A -> C -> D

    Returns:
        Tuple of (Graph, list of nodes)

    Example:
        >>> graph, nodes = create_dag_graph()
        >>> assert graph.is_acyclic()
        >>> assert len(graph.nodes) == 4
        >>> assert len(graph.edges) == 4
    """
    graph = Graph()

    n1 = Node(content={"value": "A"})
    n2 = Node(content={"value": "B"})
    n3 = Node(content={"value": "C"})
    n4 = Node(content={"value": "D"})

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    # Create DAG
    graph.add_edge(Edge(head=n1.id, tail=n2.id))
    graph.add_edge(Edge(head=n1.id, tail=n3.id))
    graph.add_edge(Edge(head=n2.id, tail=n4.id))
    graph.add_edge(Edge(head=n3.id, tail=n4.id))

    return graph, [n1, n2, n3, n4]


# =============================================================================
# Flow Factories
# =============================================================================


def create_test_flow(
    item_count: int = 5,
    progression_count: int = 2,
) -> Flow[TestElement, Progression]:
    """Create a test Flow with elements and progressions.

    Args:
        item_count: Number of items to create
        progression_count: Number of progressions to create

    Returns:
        Flow instance with test data

    Example:
        >>> flow = create_test_flow(item_count=10, progression_count=3)
        >>> assert len(flow.items) == 10
        >>> assert len(flow.progressions) == 3
    """
    items = create_test_elements(count=item_count)
    progressions = [Progression(name=f"prog_{i}", order=[]) for i in range(progression_count)]

    return Flow(
        items=items,
        progressions=progressions,
        item_type=TestElement,
    )


# =============================================================================
# Progression Factories
# =============================================================================


def create_test_progression(
    items: list[Element] | None = None,
    name: str = "test_progression",
) -> Progression:
    """Create a test Progression from items.

    Args:
        items: List of Elements (creates progression with their IDs)
        name: Name for the progression

    Returns:
        Progression instance

    Example:
        >>> elements = create_test_elements(count=5)
        >>> prog = create_test_progression(items=elements, name="my_prog")
        >>> assert len(prog) == 5
        >>> assert prog.name == "my_prog"
    """
    if items is None:
        items = create_test_elements(count=5)

    return Progression(order=[item.id for item in items], name=name)


# =============================================================================
# Spec Factories
# =============================================================================


def create_spec(
    base_type: type,
    name: str,
    *,
    default: Any = None,
    nullable: bool = False,
    listable: bool = False,
    validator: Any = None,
    description: str | None = None,
    **kwargs: Any,
) -> Any:
    """Factory for Spec creation in tests.

    More flexible than test-local helpers - supports all metadata keys.

    Args:
        base_type: The base type for the Spec (e.g., str, int)
        name: Name of the Spec
        default: Default value (if None provided, no default set unless specified)
        nullable: Whether the field can be None
        listable: Whether the field is a list
        validator: Optional validator function
        description: Optional description
        **kwargs: Additional Spec metadata

    Returns:
        Spec instance

    Example:
        >>> spec = create_spec(str, name="username", nullable=True)
        >>> spec = create_spec(int, name="age", validator=lambda x: x >= 0)
    """
    from lionherd_core.types import Spec

    spec_kwargs: dict[str, Any] = {"name": name}

    # Only add default if explicitly provided and not None
    # This allows distinguishing between "no default" and "default=None"
    if default is not None:
        spec_kwargs["default"] = default

    if nullable:
        spec_kwargs["nullable"] = nullable

    if listable:
        spec_kwargs["listable"] = listable

    if validator is not None:
        spec_kwargs["validator"] = validator

    if description is not None:
        spec_kwargs["description"] = description

    # Add any extra kwargs
    spec_kwargs.update(kwargs)

    return Spec(base_type, **spec_kwargs)


# =============================================================================
# Type System Test Fixtures
# =============================================================================


def get_sample_validators() -> dict[str, Any]:
    """Returns dict of common validator functions for testing.

    Keys:
        - 'nonneg': Non-negative integer validator
        - 'string_length': String length > 0 validator
        - 'email_format': Simple email format validator
        - 'range_validator': Integer range validator (0-100)

    Usage:
        validators = get_sample_validators()
        spec = create_spec(int, name="age", validator=validators['nonneg'])

    Example:
        >>> validators = get_sample_validators()
        >>> nonneg = validators["nonneg"]
        >>> nonneg(5)  # Returns 5
        >>> nonneg(-1)  # Raises ValueError
    """

    def nonneg(v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    def string_length(v: str) -> str:
        if len(v) == 0:
            raise ValueError("string cannot be empty")
        return v

    def email_format(v: str) -> str:
        if "@" not in v:
            raise ValueError("invalid email format")
        return v

    def range_validator(v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("value must be between 0 and 100")
        return v

    return {
        "nonneg": nonneg,
        "string_length": string_length,
        "email_format": email_format,
        "range_validator": range_validator,
    }


def get_sample_params_classes() -> dict[str, type]:
    """Returns dict of Params/DataClass test fixtures.

    Keys:
        - 'basic_params': MyParams (3 fields, all Unset)
        - 'params_none_sentinel': MyParamsNoneSentinel (none_as_sentinel=True)
        - 'params_strict': MyParamsStrict (strict=True)
        - 'basic_dataclass': MyDataClass
        - 'dataclass_strict': MyDataClassStrict
        - 'dataclass_prefill': MyDataClassPrefillUnset

    Usage:
        classes = get_sample_params_classes()
        MyParams = classes['basic_params']
        params = MyParams(field1="value")

    Example:
        >>> classes = get_sample_params_classes()
        >>> MyParams = classes["basic_params"]
        >>> params = MyParams(field1="test")
        >>> assert params.field1 == "test"
    """
    from dataclasses import dataclass
    from typing import ClassVar

    from lionherd_core.types import DataClass, ModelConfig, Params, Unset

    @dataclass(slots=True, frozen=True, init=False)
    class MyParams(Params):
        field1: str = Unset
        field2: int = Unset
        field3: bool = Unset

    @dataclass(slots=True, frozen=True, init=False)
    class MyParamsNoneSentinel(Params):
        _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
        field1: str = Unset

    @dataclass(slots=True, frozen=True, init=False)
    class MyParamsStrict(Params):
        _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
        field1: str = Unset
        field2: int = Unset

    @dataclass(slots=True)
    class MyDataClass(DataClass):
        field1: str = Unset
        field2: int = Unset

    @dataclass(slots=True)
    class MyDataClassStrict(DataClass):
        _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
        field1: str = Unset
        field2: int = Unset

    @dataclass(slots=True)
    class MyDataClassPrefillUnset(DataClass):
        _config: ClassVar[ModelConfig] = ModelConfig(prefill_unset=True)
        field1: str = Unset
        field2: int = Unset

    return {
        "basic_params": MyParams,
        "params_none_sentinel": MyParamsNoneSentinel,
        "params_strict": MyParamsStrict,
        "basic_dataclass": MyDataClass,
        "dataclass_strict": MyDataClassStrict,
        "dataclass_prefill": MyDataClassPrefillUnset,
    }


def get_sample_hashable_models() -> dict[str, type]:
    """Returns dict of HashableModel test fixtures.

    Keys:
        - 'simple': SimpleConfig (name, value)
        - 'nested': NestedConfig (config: SimpleConfig)
        - 'optional': ConfigWithOptional (required + optional)

    Usage:
        models = get_sample_hashable_models()
        SimpleConfig = models['simple']
        config = SimpleConfig(name="test", value=42)

    Example:
        >>> models = get_sample_hashable_models()
        >>> SimpleConfig = models["simple"]
        >>> config = SimpleConfig(name="test", value=42)
        >>> assert config.name == "test"
    """
    from lionherd_core.types import HashableModel

    class SimpleConfig(HashableModel):
        name: str
        value: int

    class NestedConfig(HashableModel):
        config: SimpleConfig
        enabled: bool = True

    class ConfigWithOptional(HashableModel):
        required: str
        optional: str | None = None

    return {
        "simple": SimpleConfig,
        "nested": NestedConfig,
        "optional": ConfigWithOptional,
    }


def validate_and_dump_pydantic(
    model_cls: type,
    data: dict,
) -> dict:
    """Helper to validate data and dump back to dict.

    Args:
        model_cls: Pydantic model class
        data: Data to validate

    Returns:
        Validated and dumped dict

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> result = validate_and_dump_pydantic(User, {"name": "alice", "age": 30})
        >>> assert result == {"name": "alice", "age": 30}
    """
    try:
        from lionherd_core.types.spec_adapters.pydantic_field import PydanticSpecAdapter
    except ImportError as e:
        raise ImportError("PydanticSpecAdapter not available") from e

    inst = PydanticSpecAdapter.validate_model(model_cls, data)
    return PydanticSpecAdapter.dump_model(inst)


# =============================================================================
# LNDL Fixtures
# =============================================================================


def get_sample_lndl_text(variant: str = "simple") -> str:
    """Get sample LNDL text for testing.

    Args:
        variant: Type of sample text:
            - "simple": Single lvar with OUT block
            - "multi": Multiple lvars
            - "lact": With action call
            - "raw": Raw lvar (no namespace)
            - "mixed": Mix of lvars, rlvars, lacts
            - "invalid": Syntax errors
            - "empty": Empty string

    Returns:
        Sample LNDL text

    Example:
        >>> text = get_sample_lndl_text("simple")
        >>> assert "OUT{" in text
    """
    samples = {
        "simple": """\
<lvar Report.title t>AI Safety Analysis</lvar>

OUT{title: [t]}
""",
        "multi": """\
<lvar Report.title t>AI Safety</lvar>
<lvar Report.content c>Analysis of AI safety measures.</lvar>
<lvar Report.score s>0.95</lvar>

OUT{title: [t], content: [c], score: [s]}
""",
        "lact": """\
<lact SearchResult.results r>search(query="AI safety")</lact>

OUT{results: [r]}
""",
        "raw": """\
<lvar reasoning>This is intermediate reasoning text.</lvar>

OUT{reasoning: [reasoning]}
""",
        "mixed": """\
<lvar Report.title t>Title</lvar>
<lvar reasoning>Reasoning text here</lvar>
<lact Analysis.summary s>summarize(text="...")</lact>

OUT{title: [t], reasoning: [reasoning], summary: [s], confidence: 0.85}
""",
        "invalid": """\
<lvar Report.title>Missing closing tag
OUT{title: [t]
""",
        "empty": "",
    }

    if variant not in samples:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {', '.join(samples.keys())}")

    return samples[variant]


def get_sample_pydantic_models() -> dict[str, type[BaseModel]]:
    """Get sample Pydantic models for LNDL testing.

    Returns:
        Dict of model name -> BaseModel class

    Example:
        >>> models = get_sample_pydantic_models()
        >>> Report = models["Report"]
        >>> report = Report(title="Test", content="Content")
    """

    class Report(BaseModel):
        title: str
        content: str
        score: float = 0.0

    class Analysis(BaseModel):
        summary: str
        findings: list[str]
        confidence: float
        metadata: dict[str, str] | None = None

    class SearchResult(BaseModel):
        query: str
        results: list[str]
        count: int

    return {
        "Report": Report,
        "Analysis": Analysis,
        "SearchResult": SearchResult,
    }


# =============================================================================
# Hypothesis Strategies (optional dependency)
# =============================================================================

try:
    from hypothesis import strategies as st

    def element_strategy() -> st.SearchStrategy[TestElement]:
        """Hypothesis strategy for generating TestElement instances.

        Returns:
            SearchStrategy that generates TestElements

        Example:
            >>> from hypothesis import given
            >>> @given(elem=element_strategy())
            ... def test_element_has_id(elem):
            ...     assert elem.id is not None
        """
        return st.builds(
            TestElement,
            value=st.integers(min_value=0, max_value=1000),
            name=st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"),
                    min_codepoint=ord("0"),
                    max_codepoint=ord("z"),
                ),
            ),
        )

    def node_strategy() -> st.SearchStrategy[Node]:
        """Hypothesis strategy for generating Node instances.

        Returns:
            SearchStrategy that generates Nodes

        Example:
            >>> from hypothesis import given
            >>> @given(node=node_strategy())
            ... def test_node_has_content(node):
            ...     assert node.content is not None
        """
        return st.builds(
            Node,
            content=st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.one_of(
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.text(max_size=50),
                    st.booleans(),
                ),
                min_size=1,
                max_size=5,
            ),
        )

    def progression_strategy(
        min_items: int = 0,
        max_items: int = 10,
    ) -> st.SearchStrategy[Progression]:
        """Hypothesis strategy for generating Progression instances.

        Args:
            min_items: Minimum number of items in progression
            max_items: Maximum number of items in progression

        Returns:
            SearchStrategy that generates Progressions

        Example:
            >>> from hypothesis import given
            >>> @given(prog=progression_strategy(min_items=1, max_items=5))
            ... def test_progression_not_empty(prog):
            ...     assert len(prog) > 0
        """
        return st.builds(
            Progression,
            order=st.lists(
                st.uuids().map(lambda u: u),
                min_size=min_items,
                max_size=max_items,
            ),
            name=st.text(min_size=1, max_size=20),
        )

except ImportError:
    # Hypothesis not installed, provide stub functions
    def element_strategy():
        """Hypothesis not installed. Install with: pip install hypothesis"""
        raise ImportError(
            "hypothesis is required for property-based testing strategies. "
            "Install with: pip install hypothesis"
        )

    def node_strategy():
        """Hypothesis not installed. Install with: pip install hypothesis"""
        raise ImportError(
            "hypothesis is required for property-based testing strategies. "
            "Install with: pip install hypothesis"
        )

    def progression_strategy(min_items: int = 0, max_items: int = 10):
        """Hypothesis not installed. Install with: pip install hypothesis"""
        raise ImportError(
            "hypothesis is required for property-based testing strategies. "
            "Install with: pip install hypothesis"
        )

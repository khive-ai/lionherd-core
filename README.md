# lionherd-core

> The kernel layer for production AI agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/license/apache-2-0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/github/khive-ai/lionherd-core/graph/badge.svg?token=FAE47FY26T)](https://app.codecov.io/github/khive-ai/lionherd-core)
[![PyPI version](https://img.shields.io/pypi/v/lionherd-core.svg)](https://pypi.org/project/lionherd-core/)

---

## Why lionherd-core

Zero framework lock-in. Use what you need, ignore the rest. Build production AI
systems your way.

- ✅ **Protocol-based architecture** (Rust-inspired) - compose capabilities
  without inheritance hell
- ✅ **Type-safe runtime validation** (Pydantic V2) - catch bugs before they
  bite
- ✅ **Async-first** with thread-safe operations - scale without tears
- ✅ **99% test coverage** - production-ready from day one
- ✅ **Minimal dependencies** (pydapter + anyio) - no dependency hell

lionherd-core gives you composable primitives that work exactly how you want
them to.

---

## When to use this

### Perfect for

1. **Multi-agent orchestration**
   - Define workflow DAGs with conditional edges
   - Type-safe agent state management
   - Protocol-based capability composition

2. **Structured LLM outputs**
   - Parse messy LLM responses → validated Python objects
   - Fuzzy parsing tolerates formatting variations
   - Declarative schemas with Pydantic integration

3. **Production AI systems**
   - Thread-safe collections for concurrent operations
   - Async-first architecture scales naturally
   - Protocol system enables clean interfaces

4. **Custom AI frameworks**
   - Build your own framework on solid primitives
   - Protocol composition beats inheritance
   - Adapter pattern for storage/serialization flexibility

### Not for

- Quick prototypes (try LangChain)
- Learning AI agents (too low-level)
- No-code solutions (this is code-first)

---

## Installation

```bash
pip install lionherd-core
```

**Requirements**: Python ≥3.11

---

## Quick Examples

### 1. Type-Safe Agent Collections

```python
from lionherd_core import Element, Pile
from uuid import uuid4

class Agent(Element):
    name: str
    role: str
    status: str = "idle"

# Type-safe collection
agents = Pile(item_type=Agent)
researcher = Agent(id=uuid4(), name="Alice", role="researcher")
agents.include(researcher)  # Returns True if in pile

# O(1) UUID lookup
found = agents[researcher.id]

# Predicate queries return new Pile
idle_agents = agents[lambda a: a.status == "idle"]
```

### 2. Directed Graphs

```python
from lionherd_core import Graph, Node, Edge

graph = Graph()

# Add nodes
research = Node(content="Research")
analyze = Node(content="Analyze")
report = Node(content="Report")

graph.add_node(research)
graph.add_node(analyze)
graph.add_node(report)

# Define execution flow with edges
graph.add_edge(Edge(head=research.id, tail=analyze.id))
graph.add_edge(Edge(head=analyze.id, tail=report.id))

# Traverse graph
current = research
while current:
    print(f"Executing: {current.content}")
    successors = graph.get_successors(current.id)
    current = successors[0] if successors else None
```

### 3. Structured LLM Outputs (LNDL - Language InterOperable Network Directive Language)

```python
from lionherd_core import Spec, Operable
from lionherd_core.lndl import parse_lndl_fuzzy
from pydantic import BaseModel

class Research(BaseModel):
    query: str
    findings: list[str]
    confidence: float = 0.8

# Define schema
operable = Operable([Spec(Research, name="research")])

# Parse LLM output (tolerates typos and formatting variations)
llm_response = """
<lvar Research.query q>AI architectures</lvar>
<lvar Research.findings f>["Protocol-based", "Async-first"]</lvar>
<lvar Research.confidence c>0.92</lvar>
OUT{research: [q, f, c]}
"""

result = parse_lndl_fuzzy(llm_response, operable)
print(result.research.confidence)  # 0.92
print(result.research.query)       # "AI architectures"
```

### 4. Protocol-Based Design

```python
from lionherd_core.protocols import Observable, Serializable, Adaptable
from lionherd_core.protocols import implements
from uuid import uuid4

# Check capabilities at runtime
if isinstance(obj, Observable):
    print(obj.id)  # UUID guaranteed

if isinstance(obj, Serializable):
    data = obj.to_dict()  # Serialization guaranteed

# Compose capabilities without inheritance
@implements(Observable, Serializable, Adaptable)
class CustomAgent:
    def __init__(self):
        self.id = uuid4()

    def to_dict(self, **kwargs):
        return {"id": str(self.id)}
```

---

## Core Components

| Component | Purpose | Use When |
|-----------|---------|----------|
| **Element** | UUID + metadata | You need unique identity |
| **Node** | Polymorphic content | You need flexible content storage |
| **Pile[T]** | Type-safe collections | You need thread-safe typed collections |
| **Graph** | Directed graph with edges | You need workflow DAGs |
| **Flow** | Pile of progressions + items | You need multi-stage workflows |
| **Progression** | Ordered UUID sequence | You need to track execution order |
| **LNDL** | LLM output parser | You need structured LLM outputs |

### Protocols (Rust-Inspired)

```python
from lionherd_core.protocols import (
    Observable,      # UUID + metadata
    Serializable,    # to_dict(), to_json()
    Deserializable,  # from_dict()
    Adaptable,       # Multi-format conversion
    AsyncAdaptable,  # Async adaptation
)
```

**Why protocols?**

- Structural typing beats inheritance
- Runtime checks with `isinstance()`
- Compose capabilities à la carte
- Zero performance overhead

---

## Use Cases in Detail

### Multi-Agent Systems

```python
# Define agent types with protocols
class ResearchAgent(Element):
    expertise: str
    status: str

class AnalystAgent(Element):
    domain: str
    status: str

# Type-safe agent registry
researchers = Pile(item_type=ResearchAgent)
analysts = Pile(item_type=AnalystAgent)

# Workflow orchestration with Graph
workflow = Graph()
research_phase = Node(content="research")
analysis_phase = Node(content="analysis")

workflow.add_node(research_phase)
workflow.add_node(analysis_phase)
workflow.add_edge(Edge(head=research_phase.id, tail=analysis_phase.id))

# Execute with conditional branching
current = research_phase
while current:
    # Dispatch to appropriate agents
    if current.content == "research":
        execute_research(researchers)
    elif current.content == "analysis":
        execute_analysis(analysts)

    # Progress workflow
    successors = workflow.get_successors(current.id)
    current = successors[0] if successors else None
```

### Tool Calling & Function Execution

```python
from lionherd_core import Element, Pile, Spec, Operable
from lionherd_core.lndl import parse_lndl_fuzzy
from collections.abc import Callable
from pydantic import BaseModel, ConfigDict
from typing import Any

class Tool(Element):
    name: str
    description: str
    func: Callable[..., Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Tool registry
tools = Pile(item_type=Tool)
tools.include(Tool(name="search", description="Search web", func=search_fn))
tools.include(Tool(name="calculate", description="Math ops", func=calc_fn))

# Parse LLM tool call
class ToolCall(BaseModel):
    tool: str
    args: dict

operable = Operable([Spec(ToolCall, name="call")])

llm_output = """
<lvar ToolCall.tool t>search</lvar>
<lvar ToolCall.args a>{"query": "AI agents"}</lvar>
OUT{call: [t, a]}
"""

parsed = parse_lndl_fuzzy(llm_output, operable)

# Execute - use predicate query with [] not get()
matching_tools = tools[lambda t: t.name == parsed.call.tool]
if matching_tools:
    tool = list(matching_tools)[0]
    result = tool.func(**parsed.call.args)
```

### Memory Systems

```python
from lionherd_core import Node, Graph, Edge
from datetime import datetime

class Memory(Node):
    timestamp: datetime
    importance: float
    tags: list[str]

# Memory graph (semantic connections)
memory_graph = Graph()

# Add memories
mem1 = Memory(
    content="User likes Python",
    timestamp=datetime.now(),
    importance=0.9,
    tags=["preference"]
)
mem2 = Memory(
    content="User dislikes Java",
    timestamp=datetime.now(),
    importance=0.7,
    tags=["preference"]
)

memory_graph.add_node(mem1)
memory_graph.add_node(mem2)

# Connect related memories
memory_graph.add_edge(Edge(head=mem1.id, tail=mem2.id, label=["preference"]))

# Query by importance using predicate (returns new Pile)
important_memories = memory_graph.nodes[lambda m: m.importance > 0.8]

# Traverse connections
related = memory_graph.get_successors(mem1.id)
```

### RAG Pipelines

```python
from lionherd_core import Pile, Element

class Document(Element):
    content: str
    embedding: list[float]
    metadata: dict

# Document store
docs = Pile(item_type=Document)

# Add documents with embeddings
doc = Document(
    content="Protocol-based design enables...",
    embedding=get_embedding(content),
    metadata={"source": "paper.pdf", "page": 12}
)
docs.include(doc)

# Retrieve by predicate - use [] not get()
results = docs[lambda d: d.metadata["source"] == "paper.pdf"]

# Integrate with vector DB via adapters
doc_dict = doc.to_dict()
vector_db.insert(doc_dict)
```

---

## Architecture

```text
Your Application
    ↓
lionherd-core ← You are here
    ├── Protocols (Observable, Serializable, Adaptable)
    ├── Base Classes (Element, Node, Pile, Graph, Flow)
    ├── LNDL Parser (LLM output → Python objects)
    └── Utilities (async, serialization, adapters)
    ↓
Python Ecosystem (Pydantic, asyncio, pydapter)
```

**Design Principles:**

1. **Protocols over inheritance** - Compose capabilities structurally
2. **Operations as morphisms** - Preserve semantics through composition
3. **Async-first** - Native asyncio with thread-safe operations
4. **Isolated adapters** - Per-class registries, zero pollution
5. **Minimal dependencies** - Only pydapter + anyio

---

## Development

```bash
# Setup
git clone https://github.com/khive-ai/lionherd-core.git
cd lionherd-core
uv sync --all-extras

# Test
uv run pytest --cov=lionherd_core

# Lint
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/
```

**Test Coverage**: Maintained at 99%+ with comprehensive test suite

---

## Roadmap

### v1.0.0-beta (Q1 2025)

- API stabilization
- Comprehensive docs
- Performance benchmarks
- Additional adapters (Protobuf, MessagePack)

### v1.0.0 (Q2 2025)

- Frozen public API
- Production-hardened
- Ecosystem integrations

---

## Related Projects

Part of the Lion ecosystem:

- **[lionagi](https://github.com/khive-ai/lionagi)**: v0 of the Lion ecosystem
  - full agentic AI framework with advanced orchestration capabilities
- **[pydapter](https://github.com/khive-ai/pydapter)**: Universal data adapter
  (JSON/YAML/TOML/SQL/Neo4j/Redis/MongoDB/Weaviate/etc.)

---

## License

Apache 2.0 - Free for commercial use, no strings attached.

---

## Support

- [GitHub Issues](https://github.com/khive-ai/lionherd-core/issues)
- [Contributing Guide](./CONTRIBUTING.md)

---

## Created by

**[HaiyangLi (Ocean)](https://github.com/ohdearquant)** - [khive.ai](https://khive.ai)

Inspired by Rust traits, Pydantic validation, and functional programming.

---

**Ready to build?**

```bash
pip install lionherd-core
```

*Alpha release - APIs may evolve. Feedback shapes the future.*

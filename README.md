# lionherd-core

> The kernel layer for production AI agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/github/khive-ai/lionherd-core/graph/badge.svg?token=FAE47FY26T)](https://codecov.io/github/khive-ai/lionherd-core)
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
- ✅ **99% test coverage** (1851 tests) - production-ready from day one
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
agents = Pile[Agent](item_type=Agent)
researcher = Agent(id=uuid4(), name="Alice", role="researcher")
agents.include(researcher)

# O(1) UUID lookup
found = agents[researcher.id]

# Predicate queries
idle_agents = agents.get(lambda a: a.status == "idle")
```

### 2. Workflow State Machines

```python
from lionherd_core import Flow, Node

workflow = Flow()

# Define workflow steps
research_id = workflow.register_node("research", Node(content="Research"))
analyze_id = workflow.register_node("analyze", Node(content="Analyze"))
report_id = workflow.register_node("report", Node(content="Report"))

# Define execution flow
workflow.register_edge(research_id, analyze_id)
workflow.register_edge(analyze_id, report_id)

# Create progression
workflow.register_progression(
    name="main",
    order=[research_id, analyze_id, report_id],
    progressive=True
)

# Execute
for step_id in workflow.get_progression("main"):
    node = workflow.get_node(step_id)
    print(f"Executing: {node.content}")
```

### 3. Structured LLM Outputs

```python
from lionherd_core import Spec, Operable
from lionherd_core.lndl import parse_lndl_fuzzy
from pydantic import BaseModel

class ResearchOutput(BaseModel):
    query: str
    findings: list[str]
    confidence: float

# Define schema
operable = Operable([
    Spec(str, name="query", description="Research query"),
    Spec(list[str], name="findings", description="Key findings"),
    Spec(float, name="confidence", description="Score 0-1", default=0.8)
], name="Research")

# Parse LLM output (tolerates formatting chaos)
llm_response = """
OUT{research: [{
    query: "AI architectures",
    findings: ["Protocol-based", "Async-first"],
    confidence: 0.92
}]}
"""

result = parse_lndl_fuzzy(llm_response, operable)
print(result.confidence)  # 0.92
```

### 4. Protocol-Based Design

```python
from lionherd_core.protocols import Observable, Serializable, Adaptable

# Check capabilities at runtime
if isinstance(obj, Observable):
    print(obj.id)  # UUID guaranteed

if isinstance(obj, Serializable):
    data = obj.to_dict()  # Serialization guaranteed

# Compose capabilities without inheritance
from lionherd_core.protocols import implements

@implements(Observable, Serializable, Adaptable)
class CustomAgent:
    def __init__(self):
        self.id = uuid4()

    def to_dict(self):
        return {"id": str(self.id)}
```

---

## Core Components

| Component | Purpose | Use When |
|-----------|---------|----------|
| **Element** | UUID + metadata | You need unique identity |
| **Node** | Polymorphic content | You need flexible content storage |
| **Pile[T]** | Type-safe collections | You need thread-safe typed collections |
| **Graph** | Directed graph with conditions | You need workflow DAGs |
| **Flow** | Workflow state machine | You need stateful orchestration |
| **Progression** | Named UUID ordering | You need execution sequences |
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
researchers = Pile[ResearchAgent](item_type=ResearchAgent)
analysts = Pile[AnalystAgent](item_type=AnalystAgent)

# Workflow orchestration
workflow = Flow()
research_phase = workflow.register_node("research")
analysis_phase = workflow.register_node("analysis")
workflow.register_edge(research_phase, analysis_phase)

# Execute with conditional branching
current = research_phase
while current:
    # Dispatch to appropriate agents
    if current == research_phase:
        execute_research(researchers)
    elif current == analysis_phase:
        execute_analysis(analysts)

    # Progress workflow
    successors = workflow.get_successors(current)
    current = successors[0] if successors else None
```

### Tool Calling & Function Execution

```python
from lionherd_core import Node, Pile

class Tool(Element):
    name: str
    description: str
    func: callable

# Tool registry
tools = Pile[Tool](item_type=Tool)
tools.include([
    Tool(name="search", description="Search web", func=search_fn),
    Tool(name="calculate", description="Math ops", func=calc_fn),
])

# Parse LLM tool call
tool_call_spec = Operable([
    Spec(str, name="tool", description="Tool name"),
    Spec(dict, name="args", description="Arguments"),
], name="ToolCall")

parsed = parse_lndl_fuzzy(llm_output, tool_call_spec)

# Execute
tool = tools.get(lambda t: t.name == parsed.tool)[0]
result = tool.func(**parsed.args)
```

### Memory Systems

```python
from lionherd_core import Node, Graph

class Memory(Node):
    timestamp: datetime
    importance: float
    tags: list[str]

# Memory graph (semantic connections)
memory_graph = Graph()

# Add memories
mem1_id = memory_graph.add_node(Memory(content="User likes Python"))
mem2_id = memory_graph.add_node(Memory(content="User dislikes Java"))

# Connect related memories
memory_graph.add_edge(mem1_id, mem2_id, label="preference")

# Query by importance
important_memories = memory_pile.get(lambda m: m.importance > 0.8)

# Traverse connections
related = memory_graph.get_successors(mem1_id)
```

### RAG Pipelines

```python
from lionherd_core import Pile, Element

class Document(Element):
    content: str
    embedding: list[float]
    metadata: dict

# Document store
docs = Pile[Document](item_type=Document)

# Add documents with embeddings
doc = Document(
    content="Protocol-based design enables...",
    embedding=get_embedding(content),
    metadata={"source": "paper.pdf", "page": 12}
)
docs.include(doc)

# Retrieve by predicate
results = docs.get(lambda d: d.metadata["source"] == "paper.pdf")

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

**Test Coverage**: 99% (1851 tests, 31k lines)

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
- [Discussions](https://github.com/khive-ai/lionherd-core/discussions)
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

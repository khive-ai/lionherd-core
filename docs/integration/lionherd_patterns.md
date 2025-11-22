# lionherd Integration Patterns

**Purpose**: Document production-proven patterns for building with lionherd-core primitives

**Audience**: Developers building on lionherd-core (including lionherd, khive, and custom frameworks)

**Source**: Extracted from lionherd alpha1 codebase (2024-2025)

---

## Table of Contents

1. [Session Management with Flow](#1-session-management-with-flow)
2. [ServiceRegistry with Pile](#2-serviceregistry-with-pile)
3. [Branch-based Conversation Management](#3-branch-based-conversation-management)
4. [Message Pile Queries](#4-message-pile-queries)
5. [Event-based Lifecycle Tracking](#5-event-based-lifecycle-tracking)
6. [Operation Graphs with Graph + Flow](#6-operation-graphs-with-graph--flow)

---

## 1. Session Management with Flow

**Pattern**: Session = Flow[Message, Branch] + ServiceRegistry + metadata

**Why Flow**: Enables O(1) message lookup by UUID while preserving multiple conversation branches (Progressions).

### Core Architecture

```python
from lionherd_core import Element, Flow
from pydantic import Field

class Session(Element):
    """Central storage for messages, branches, and services."""

    user: str | None = Field(default=None)
    conversations: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message)
    )
    services: ServiceRegistry = Field(
        default_factory=ServiceRegistry
    )

    @property
    def messages(self):
        """Alias for conversations.items (Pile[Message])."""
        return self.conversations.items

    @property
    def branches(self):
        """Alias for conversations.progressions (Pile[Branch])."""
        return self.conversations.progressions
```

### Access Patterns

```python
# O(1) Message access by UUID
msg = session.messages[uuid]

# O(1) Branch access by name
branch = session.conversations.get_progression("main")

# Filter messages by predicate
user_messages = session.messages[lambda m: m.role == "user"]

# Get branch messages in order
branch = session.branches["main"]
messages = [session.messages[uid] for uid in branch]

# Add message to specific branch
session.conversations.add_item(message, progressions=["main", "experiment"])
```

### Why This Works

1. **Flow composition**: `conversations.items` (Pile[Message]) stores all messages once
2. **Progression tracking**: Each Branch (Progression) tracks message order by UUID
3. **No duplication**: Message shared across multiple branches = single storage, multiple references
4. **Thread-safe**: Pile's built-in RLock protects concurrent access

---

## 2. ServiceRegistry with Pile

**Pattern**: ServiceRegistry = Pile[iModel] + name index (dict[str, UUID])

**Why Pile**: Type-safe storage with O(1) UUID lookup. Name index provides O(1) name-based access.

### Core Architecture

```python
from lionherd_core import Pile, Element
from uuid import UUID
from typing import Any

# Stub types for documentation (from lionherd)
class iModel(Element):
    name: str = "model"

class Tool(Element):
    name: str = "tool"

class ServiceRegistry:
    """Unified service boundary for models and tools."""

    def __init__(self):
        # Pile provides type-safe storage
        self._pile: Pile[iModel | Tool] = Pile(item_type=[iModel, Tool])
        # Name index for O(1) lookup
        self._name_index: dict[str, UUID] = {}

    def register(self, model: iModel | Tool, update: bool = False) -> UUID:
        """Register service with name-based lookup."""
        if model.name in self._name_index and not update:
            raise ValueError(f"Service '{model.name}' already registered")

        # Add to Pile (UUID storage)
        self._pile.add(model)
        # Index by name
        self._name_index[model.name] = model.id

        return model.id

    def get(self, name: str) -> iModel | Tool:
        """Get service by name (O(1))."""
        if name not in self._name_index:
            raise KeyError(f"Service '{name}' not found")

        uuid = self._name_index[name]
        return self._pile[uuid]

    def list_by_tag(self, tag: str) -> list[iModel | Tool]:
        """Filter services by tag using Pile predicate."""
        return list(self._pile[lambda s: tag in s.tags])
```

### Why This Pattern

1. **Pile for storage**: Thread-safe, type-validated, UUID-indexed
2. **Separate name index**: O(1) name lookup without sacrificing Pile benefits
3. **Predicate filtering**: Pile's `pile[lambda x: ...]` enables tag-based queries
4. **Single source of truth**: Pile stores once, name index points to it

---

## 3. Branch-based Conversation Management

**Pattern**: Branch = Progression + metadata (system message, capabilities)

**Why Progression**: Ordered UUID sequence tracks message order. Element identity enables branching/cloning.

### Core Architecture

```python
from lionherd_core import Progression
from uuid import UUID
from pydantic import Field

class Branch(Progression):
    """Named conversation thread with metadata."""

    session_id: UUID = Field(description="Parent session")
    user: UUID | str = Field(description="User identifier")
    capabilities: set[str] = Field(
        default_factory=set,
        description="Available service names"
    )
    _system_message_id: UUID | None = None

    def set_system_message(self, msg_id: UUID):
        """Set system message at order[0]."""
        self._system_message_id = msg_id
        # Ensure system message is first
        if msg_id in self.order:
            self.order.remove(msg_id)
        self.order.insert(0, msg_id)
```

### Usage Patterns

```python
# Create branch with system message
session = Session()
sys_msg = Message(content={"system_message": "You are helpful."})
session.messages.add(sys_msg)

branch = Branch(
    session_id=session.id,
    user=session.id,
    name="main",
    capabilities={"gpt-4"},
)
branch.set_system_message(sys_msg.id)

# Add branch to session
session.conversations.add_progression(branch)

# Branch with multiple conversations
branch.append(msg1.id)
branch.append(msg2.id)

# Get messages in order
messages = [session.messages[uid] for uid in branch]

# Clone branch for experimentation
experiment = branch.clone()
experiment.name = "experiment"
session.conversations.add_progression(experiment)
```

### Why Progression for Branches

1. **Element identity**: Each branch has UUID, can be cloned/forked
2. **Order preservation**: `order` list tracks message sequence
3. **Referential integrity**: UUIDs reference messages in Pile, not copies
4. **Serializable**: Can save/load branches across sessions

---

## 4. Message Pile Queries

**Pattern**: Use Pile's type-dispatched `__getitem__` for expressive queries

**Why**: Single interface (`pile[key]`) replaces multiple query methods.

### Query Patterns

```python
# 1. By UUID (returns Message)
msg = session.messages[uuid]

# 2. By index (returns Message)
first = session.messages[0]
last = session.messages[-1]

# 3. By slice (returns Pile[Message])
recent = session.messages[-10:]  # Last 10 messages

# 4. By list of indices (returns Pile[Message])
selected = session.messages[[0, 5, 10]]

# 5. By list of UUIDs (returns Pile[Message])
specific = session.messages[[uuid1, uuid2, uuid3]]

# 6. By predicate (returns Pile[Message])
user_msgs = session.messages[lambda m: m.role == "user"]
recent_errors = session.messages[lambda m: m.metadata.get("error")]

# 7. By Progression (returns Pile[Message])
branch_msgs = session.messages[branch]  # Branch is Progression
```

### Advanced Queries

```python
# Chain queries (Pile operations)
# Note: Assumes session is defined in Pattern 4 context
user_msgs = session.messages[lambda m: m.role == "user"]
recent_user = user_msgs[-5:]  # Last 5 user messages

# Filter by type (stub for documentation)
class InstructionContent:
    pass

instructions = session.messages[
    lambda m: isinstance(m.content, InstructionContent)
]

# Complex predicates
failed = session.messages[lambda m:
    m.metadata.get("status") == "failed" and
    m.metadata.get("retries", 0) < 3
]

# Convert to list when needed
msg_list = list(session.messages[0:10])
```

### Performance Notes

- **UUID lookup**: O(1) via Pile's internal dict
- **Index access**: O(1) via Progression index + O(1) dict lookup
- **Slice**: O(k) where k = slice width
- **Predicate**: O(n) where n = pile size (must check all items)
- **List/tuple**: O(k) where k = list length

---

## 5. Event-based Lifecycle Tracking

**Pattern**: Event = async execution + status tracking + metadata

**Why Event**: Async-native, tracks execution state, enables retry/monitoring.

### Core Usage

```python
from lionherd_core import Event, EventStatus

class GenerationEvent(Event):
    """Event for LLM generation."""

    model_name: str
    prompt: str

    async def _invoke(self):
        # Your async logic here (stub for documentation)
        return f"Response for {self.prompt}"

async def main():
    # Execute event
    event = GenerationEvent(model_name="gpt-4", prompt="Hello")
    result = await event.invoke()

    # Check status
    assert event.status == EventStatus.COMPLETED
    assert event.execution.result == result

    # Retry on failure
    if event.status == EventStatus.FAILED:
        retry_event = event.as_fresh_event()
        result = await retry_event.invoke()
```

### Integration with Pile

```python
# Track all events in session
class Session(Element):
    events: Pile[Event] = Field(default_factory=lambda: Pile(item_type=Event))

# Query events
failed_events = session.events[lambda e: e.status == EventStatus.FAILED]
recent_events = session.events[-10:]

# Retry all failed events
for event in failed_events:
    retry = event.as_fresh_event()
    await retry.invoke()
```

---

## 6. Operation Graphs with Graph + Flow

**Pattern**: Graph for dependencies, Flow for execution state

**Why**: Graph validates structure (DAG, cycles), Flow tracks execution progress.

### Operation Graph Pattern

```python
from lionherd_core import Graph, Flow, Node, Edge
from uuid import UUID

class OperationGraph:
    """Directed graph of operations with execution tracking."""

    def __init__(self):
        self.graph = Graph()
        self.execution = Flow[Node, Progression](item_type=Node)

    def add_operation(
        self,
        op: Node,
        depends_on: list[UUID] | None = None
    ):
        """Add operation with dependencies."""
        self.graph.add_node(op)

        if depends_on:
            for dep_id in depends_on:
                self.graph.add_edge(Edge(head=dep_id, tail=op.id))

    async def execute(self):
        """Execute operations in topological order."""
        # Validate DAG
        if not self.graph.is_acyclic():
            raise ValueError("Operation graph contains cycles")

        # Get execution order
        order = self.graph.topological_sort()

        # Execute in order
        results = {}
        for node_id in order:
            node = self.graph.nodes[node_id]
            # Execute operation
            result = await self._execute_operation(node)
            results[node_id] = result

            # Track in Flow
            self.execution.add_item(node, progressions=["completed"])

        return results

    async def _execute_operation(self, node: Node):
        """Execute single operation."""
        # Your logic here
        ...
```

### Usage

```python
# Build operation graph
op_graph = OperationGraph()

# Add operations
fetch = Node(content={"op": "fetch_data", "url": "..."})
process = Node(content={"op": "process", "algorithm": "..."})
store = Node(content={"op": "store", "dest": "..."})

op_graph.add_operation(fetch)
op_graph.add_operation(process, depends_on=[fetch.id])
op_graph.add_operation(store, depends_on=[process.id])

# Execute
results = await op_graph.execute()

# Query completed operations
completed = op_graph.execution.get_progression("completed")
```

---

## Testing with lionherd-core.testing

**New in alpha7**: Reusable test fixtures for downstream projects

```python
from lionherd_core.testing import (
    create_test_pile,
    mock_element,
    create_simple_graph,
    create_test_flow,
)

def test_session_message_storage():
    # Use lionherd-core fixtures
    pile = create_test_pile(count=10)
    assert len(pile) == 10

def test_graph_operations():
    graph, nodes = create_simple_graph()
    assert graph.is_acyclic()
    assert len(nodes) == 3
```

See `lionherd_core.testing` module for full API.

---

## Common Pitfalls

### 1. Don't Store Messages Twice

```python
# ❌ WRONG: Storing messages in both Pile and list
class Session:
    messages: Pile[Message]
    branch_messages: list[Message]  # Duplication!

# ✅ CORRECT: Store once in Pile, reference by UUID
class Session:
    messages: Pile[Message]  # Single source of truth

class Branch(Progression):
    # order: list[UUID] references messages in Pile
    ...
```

### 2. Use Element Identity for Cloning

```python
# ❌ WRONG: Manual copying
new_branch = Branch(
    name=f"{old_branch.name}_copy",
    order=list(old_branch.order)  # Loses metadata
)

# ✅ CORRECT: Use Element.clone()
new_branch = old_branch.clone()
new_branch.name = f"{old_branch.name}_copy"
```

### 3. Don't Mix Pile and Progression

```python
# ❌ WRONG: Progression with direct item storage
class Branch(Progression):
    messages: list[Message]  # Don't store items here!

# ✅ CORRECT: Progression references items in Pile
class Branch(Progression):
    # order: list[UUID] built-in from Progression
    ...

# Items live in Flow.items (Pile[Message])
session.conversations.items[uuid]  # O(1) access
```

---

## Performance Guidelines

### When to Use What

| Use Case | Primitive | Why |
|----------|-----------|-----|
| Store items with UUID lookup | Pile[T] | O(1) UUID access, thread-safe |
| Track ordered sequences | Progression | Ordered UUIDs, serializable |
| Manage multiple orderings | Flow | Pile + Progressions composition |
| Validate dependencies | Graph | DAG detection, topological sort |
| Async execution tracking | Event | Status tracking, retry support |

### Complexity Reference

```python
# Pile operations
pile[uuid]              # O(1) - dict lookup
pile[0]                 # O(1) - progression index + dict
pile[0:10]              # O(k) - k items
pile[lambda x: ...]     # O(n) - must check all n items

# Progression operations
prog.append(uuid)       # O(1) - list append
prog[index]             # O(1) - list index
prog.move(from, to)     # O(1) - list operations

# Graph operations
graph.add_node(n)       # O(1) - pile add + dict init
graph.add_edge(e)       # O(1) - pile add + set operations
graph.topological_sort()# O(V + E) - Kahn's algorithm
```

---

## Summary

**Key Takeaways**:

1. **Session = Flow[Message, Branch]**: Messages stored once in Pile, referenced by Branches (Progressions)
2. **ServiceRegistry = Pile + name index**: Type-safe storage + O(1) name lookup
3. **Branch = Progression + metadata**: Ordered message references + conversation context
4. **Pile queries are expressive**: `pile[uuid]`, `pile[0:10]`, `pile[lambda x: ...]` - single interface
5. **Event for async lifecycle**: Built-in status tracking + retry support
6. **Graph for structure, Flow for state**: Separate concerns for clarity

**Next Steps**:

- Review lionherd-core API: `docs/api/`
- Explore test utilities: `lionherd_core.testing`
- Study lionherd source: `github.com/khive-ai/lionherd`

---

**Version**: 1.0 (2025-11-21)
**Status**: Production
**Source**: lionherd alpha1

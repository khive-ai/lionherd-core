# Graph

> Directed graph with Pile-backed storage, O(1) adjacency queries, and classical graph algorithms.

---

## Overview

**Graph** is a directed graph implementation built on [Pile](pile.md) storage with pre-computed adjacency lists for O(1) neighbor queries. It combines efficient graph operations with classical algorithms for cycle detection, topological sorting, and pathfinding.

**Architecture**:

- **`nodes`**: Pile[Node] - Vertex storage
- **`edges`**: Pile[Edge] - Directed edge storage
- **`_out_edges`**: dict[UUID, set[UUID]] - O(1) successor lookup
- **`_in_edges`**: dict[UUID, set[UUID]] - O(1) predecessor lookup

**Thread Safety**: All operations are thread-safe via @synchronized decorator (RLock).

---

## When to Use Graph

**Use Graph when you need:**

- **Directed relationships**: Dependencies, workflows, state machines
- **O(1) adjacency queries**: Fast predecessor/successor lookups
- **Graph algorithms**: Cycle detection, topological sort, pathfinding
- **Edge properties**: Labels, conditions, custom attributes
- **Conditional traversal**: Runtime edge filtering via EdgeCondition

**Don't use Graph when:**

- Undirected edges are primary requirement → implement wrapper or use specialized library
- Very large graphs (millions of nodes) → consider database-backed graph (Neo4j, etc.)
- Simple linear sequence → use [Progression](progression.md)
- Tree structure without multiple parents → use simpler tree data structure

---

## Class Signature

```python
class Graph(Element, PydapterAdaptable, PydapterAsyncAdaptable):
    """Directed graph with Pile-backed storage, O(1) operations, graph algorithms."""
```

---

## Components

### Edge

Directed edge with labels, properties, and optional traversal conditions.

**Signature:**

```python
class Edge(Element):
    """Directed edge with labels, conditions, properties."""

    head: UUID  # Source node ID
    tail: UUID  # Target node ID
    label: list[str] = []  # Edge labels
    condition: EdgeCondition | None = None  # Runtime traversal predicate (not serialized)
    properties: dict[str, Any] = {}  # Custom edge attributes
```

**Example:**

```python
edge = Edge(
    head=source_node.id,
    tail=target_node.id,
    label=["dependency", "required"],
    properties={"weight": 1.5, "type": "strong"}
)
```

---

### EdgeCondition

Runtime predicate for conditional edge traversal (not serialized).

**Signature:**

```python
class EdgeCondition:
    """Runtime predicate for edge traversal (not serialized)."""

    def __init__(self, **kwargs: Any): ...

    async def apply(self, *args: Any, **kwargs: Any) -> bool:
        """Evaluate condition. Override for custom logic. Default: always True."""
        return True

    async def __call__(self, *args: Any, **kwargs: Any) -> bool:
        """Async callable interface. Calls apply() directly."""
```

**Example:**

```python
class WeightCondition(EdgeCondition):
    async def apply(self, max_weight: float = 10.0) -> bool:
        weight = self.properties.get("weight", 0.0)
        return weight <= max_weight

# Use in edge
edge = Edge(
    head=n1.id,
    tail=n2.id,
    condition=WeightCondition(properties={"weight": 15.0})
)
```

**Note**: Conditions are not serialized. They're runtime constructs for dynamic graph traversal logic.

---

## Constructor

```python
def __init__(self, **data: Any) -> None: ...
```

**Parameters:**

- `**data`: Element fields (id, created_at, metadata)

**Example:**

```python
from lionherd_core.base import Graph, Node, Edge

# Create empty graph
graph = Graph()

# Or with metadata
graph = Graph(metadata={"version": "1.0", "type": "dependency_graph"})
```

---

## Attributes

### Core Piles

#### `nodes`

```python
nodes: Pile[Node]
```

Pile of Node instances representing graph vertices.

**Type:** Pile[Node]

**Access:** Read-only via property (returns MappingProxyType)

**Example:**

```python
print(f"Graph has {len(graph.nodes)} nodes")
for node in graph.nodes:
    print(node.content)
```

#### `edges`

```python
edges: Pile[Edge]
```

Pile of Edge instances representing directed connections.

**Type:** Pile[Edge]

**Access:** Read-only via property (returns MappingProxyType)

**Example:**

```python
print(f"Graph has {len(graph.edges)} edges")
for edge in graph.edges:
    print(f"{edge.head} → {edge.tail}")
```

---

## Methods

### Node Operations

#### `add_node()`

Add node to graph with adjacency list initialization.

**Signature:**

```python
@synchronized
def add_node(self, node: Node) -> None
```

**Parameters:**

- `node` (Node): Node instance to add

**Raises:**

- `ExistsError`: If node already exists in graph

**Thread Safety:** Synchronized via @synchronized decorator

**Time Complexity:** O(1)

**Example:**

```python
node = Node(content={"name": "TaskA"})
graph.add_node(node)
```

**Implementation Note**: Atomically adds node to pile and initializes empty adjacency sets in `_out_edges` and `_in_edges`.

---

#### `remove_node()`

Remove node and all connected edges from graph.

**Signature:**

```python
@synchronized
def remove_node(self, node_id: UUID | Node) -> Node
```

**Parameters:**

- `node_id` (UUID | Node): Node UUID or Node instance

**Returns:** Node - The removed node

**Raises:**

- `NotFoundError`: If node not found in graph

**Thread Safety:** Synchronized with RLock (allows nested remove_edge calls)

**Time Complexity:** O(d) where d = degree (in + out edges)

**Example:**

```python
removed = graph.remove_node(node_id)
print(f"Removed node with {len(removed_edges)} connected edges")
```

**Cascade Behavior**: Automatically removes all edges connected to the node (both incoming and outgoing) before removing the node itself.

---

### Edge Operations

#### `add_edge()`

Add directed edge to graph with adjacency list updates.

**Signature:**

```python
@synchronized
def add_edge(self, edge: Edge) -> None
```

**Parameters:**

- `edge` (Edge): Edge instance to add

**Raises:**

- `ExistsError`: If edge already exists in graph
- `NotFoundError`: If head or tail node not in graph

**Thread Safety:** Synchronized via @synchronized decorator

**Time Complexity:** O(1)

**Example:**

```python
edge = Edge(head=n1.id, tail=n2.id, label=["depends_on"])
graph.add_edge(edge)
```

**Validation**: Both head and tail nodes must exist in graph before adding edge.

---

#### `remove_edge()`

Remove edge from graph with adjacency list cleanup.

**Signature:**

```python
@synchronized
def remove_edge(self, edge_id: UUID | Edge) -> Edge
```

**Parameters:**

- `edge_id` (UUID | Edge): Edge UUID or Edge instance

**Returns:** Edge - The removed edge

**Raises:**

- `NotFoundError`: If edge not found in graph

**Thread Safety:** Synchronized (RLock allows calls from remove_node)

**Time Complexity:** O(1)

**Example:**

```python
removed_edge = graph.remove_edge(edge_id)
print(f"Removed edge: {removed_edge.head} → {removed_edge.tail}")
```

---

### Graph Queries

#### `get_predecessors()`

Get all nodes with edges pointing to the specified node.

**Signature:**

```python
def get_predecessors(self, node_id: UUID | Node) -> list[Node]
```

**Parameters:**

- `node_id` (UUID | Node): Target node UUID or instance

**Returns:** list[Node] - Nodes with outgoing edges to target

**Time Complexity:** O(k) where k = in-degree of node

**Example:**

```python
# Get all nodes that depend on task_node
predecessors = graph.get_predecessors(task_node)
print(f"Dependencies: {[n.content['name'] for n in predecessors]}")
```

**Implementation**: Uses `_in_edges` adjacency list for O(1) edge set lookup, then O(k) node retrievals.

---

#### `get_successors()`

Get all nodes that the specified node points to.

**Signature:**

```python
def get_successors(self, node_id: UUID | Node) -> list[Node]
```

**Parameters:**

- `node_id` (UUID | Node): Source node UUID or instance

**Returns:** list[Node] - Nodes with incoming edges from source

**Time Complexity:** O(k) where k = out-degree of node

**Example:**

```python
# Get all tasks that depend on this task
successors = graph.get_successors(task_node)
print(f"Dependents: {[n.content['name'] for n in successors]}")
```

**Implementation**: Uses `_out_edges` adjacency list for O(1) edge set lookup, then O(k) node retrievals.

---

#### `get_node_edges()`

Get edges connected to node by direction.

**Signature:**

```python
def get_node_edges(
    self,
    node_id: UUID | Node,
    direction: Literal["in", "out", "both"] = "both",
) -> list[Edge]
```

**Parameters:**

- `node_id` (UUID | Node): Node UUID or instance
- `direction` (Literal["in", "out", "both"]): Edge direction filter. Default: "both"

**Returns:** list[Edge] - Edges connected to node

**Raises:** ValueError if invalid direction

**Time Complexity:** O(k) where k = degree in specified direction

**Example:**

```python
# Get all edges connected to node
all_edges = graph.get_node_edges(node, direction="both")

# Get only incoming edges
incoming = graph.get_node_edges(node, direction="in")

# Get only outgoing edges
outgoing = graph.get_node_edges(node, direction="out")
```

---

#### `get_heads()`

Get all source nodes (no incoming edges).

**Signature:**

```python
def get_heads(self) -> list[Node]
```

**Returns:** list[Node] - Nodes with in-degree = 0

**Time Complexity:** O(V) - scans all nodes

**Example:**

```python
# Get starting points in dependency graph
sources = graph.get_heads()
print(f"Root tasks: {[n.content['name'] for n in sources]}")
```

**Use Case**: Identify entry points in workflows, DAG roots, independent tasks.

---

#### `get_tails()`

Get all sink nodes (no outgoing edges).

**Signature:**

```python
def get_tails(self) -> list[Node]
```

**Returns:** list[Node] - Nodes with out-degree = 0

**Time Complexity:** O(V) - scans all nodes

**Example:**

```python
# Get final outputs in dependency graph
sinks = graph.get_tails()
print(f"Final outputs: {[n.content['name'] for n in sinks]}")
```

**Use Case**: Identify terminal nodes, final deliverables, workflow endpoints.

---

### Graph Algorithms

#### `is_acyclic()`

Check if graph is a DAG (Directed Acyclic Graph) using three-color DFS.

**Signature:**

```python
def is_acyclic(self) -> bool
```

**Returns:** bool - True if graph has no cycles, False otherwise

**Algorithm:** Three-color DFS (WHITE/GRAY/BLACK)

**Time Complexity:** O(V + E)

**Example:**

```python
if graph.is_acyclic():
    print("Graph is a DAG - can perform topological sort")
else:
    print("Graph contains cycles")
```

**Implementation**:

- WHITE: Unvisited node
- GRAY: Currently visiting (in DFS stack)
- BLACK: Finished visiting
- Cycle detected when encountering GRAY node (back edge)

---

#### `topological_sort()`

Return topologically sorted list of nodes using Kahn's algorithm.

**Signature:**

```python
def topological_sort(self) -> list[Node]
```

**Returns:** list[Node] - Nodes in topological order

**Raises:** ValueError if graph contains cycles

**Algorithm:** Kahn's algorithm (BFS-based)

**Time Complexity:** O(V + E)

**Example:**

```python
try:
    sorted_tasks = graph.topological_sort()
    print("Execution order:")
    for task in sorted_tasks:
        print(f"  {task.content['name']}")
except ValueError:
    print("Cannot sort - graph has cycles")
```

**Property**: For every directed edge (u, v), node u appears before v in the sorted list.

**Use Cases**: Build order resolution, task scheduling, dependency resolution.

---

#### `find_path()`

Find shortest path between two nodes using BFS.

**Signature:**

```python
async def find_path(
    self,
    start: UUID | Node,
    end: UUID | Node,
    check_conditions: bool = False,
) -> list[Edge] | None
```

**Parameters:**

- `start` (UUID | Node): Starting node
- `end` (UUID | Node): Target node
- `check_conditions` (bool): If True, evaluate edge conditions during traversal. Default: False

**Returns:** list[Edge] | None - List of edges forming path, or None if no path exists

**Raises:**

- `NotFoundError`: If start or end node not in graph

**Algorithm:** Breadth-First Search (BFS)

**Time Complexity:** O(V + E)

**Example:**

```python
# Find path without condition checking (async-first design)
path = await graph.find_path(start_node, end_node)
if path:
    print(f"Path length: {len(path)} edges")
    for edge in path:
        print(f"  {edge.head} → {edge.tail}")

# Find path with condition checking
conditional_path = await graph.find_path(start_node, end_node, check_conditions=True)
```

**Property**: Returns shortest path by edge count (BFS guarantees shortest path in unweighted graphs).

**Conditional Traversal**: When `check_conditions=True`, edges with conditions that evaluate to False are skipped.

**Note**: Graph uses async-first design. In Jupyter notebooks, use `await` syntax directly. In sync contexts, use `run_async()` from `lionherd_core.libs.concurrency`.

---

### Serialization

#### `to_dict()`

Serialize graph with proper Pile serialization for nodes and edges.

**Signature:**

```python
def to_dict(
    self,
    mode: Literal["python", "json", "db"] = "python",
    created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
    meta_key: str | None = None,
    item_meta_key: str | None = None,
    item_created_at_format: Literal["datetime", "isoformat", "timestamp"] | None = None,
    **kwargs: Any,
) -> dict[str, Any]
```

**Parameters:**

- `mode`: Serialization mode (python/json/db)
- `created_at_format`: Timestamp format for Graph
- `meta_key`: Rename Graph metadata field
- `item_meta_key`: Pass to Pile.to_dict for node/edge metadata renaming
- `item_created_at_format`: Pass to Pile.to_dict for node/edge timestamp format
- `**kwargs`: Additional serialization options

**Returns:** dict[str, Any]

**Example:**

```python
# Serialize to JSON
data = graph.to_dict(mode="json")

# Database mode with custom metadata key
db_data = graph.to_dict(
    mode="db",
    meta_key="graph_metadata",
    item_meta_key="node_metadata"
)
```

**Note**: Adjacency lists (`_out_edges`, `_in_edges`) are NOT serialized. They're rebuilt from edges during deserialization.

---

#### `from_dict()`

Deserialize graph from dictionary with automatic adjacency list rebuilding.

**Signature:**

```python
@classmethod
def from_dict(
    cls,
    data: dict[str, Any],
    meta_key: str | None = None,
    item_meta_key: str | None = None,
    **kwargs: Any,
) -> Graph
```

**Parameters:**

- `data` (dict): Serialized graph data from to_dict()
- `meta_key`: Restore Graph metadata from this key (db compatibility)
- `item_meta_key`: Pass to Pile.from_dict for node/edge deserialization
- `**kwargs`: Additional arguments

**Returns:** Graph instance with rebuilt adjacency lists

**Example:**

```python
# Round-trip serialization
data = graph.to_dict(mode="json")
restored = Graph.from_dict(data)

# Adjacency lists automatically rebuilt
predecessors = restored.get_predecessors(node_id)
```

**Post-Init Hook**: The `_rebuild_adjacency_after_init` validator automatically rebuilds adjacency lists from deserialized nodes and edges.

---

### Containment

#### `__contains__()`

Check if node or edge is in graph.

**Signature:**

```python
def __contains__(self, item: object) -> bool
```

**Parameters:**

- `item` (Node | Edge | UUID): Item to check

**Returns:** bool - True if item in graph

**Example:**

```python
if node in graph:
    print("Node exists")

if edge in graph:
    print("Edge exists")

if node_uuid in graph:
    print("UUID found (node or edge)")
```

---

#### `__len__()`

Return number of nodes in graph.

**Signature:**

```python
def __len__(self) -> int
```

**Returns:** int - Number of nodes

**Example:**

```python
print(f"Graph has {len(graph)} nodes")
print(f"Graph has {len(graph.edges)} edges")
```

**Note**: `len(graph)` returns node count, not edge count. Use `len(graph.edges)` for edge count.

---

### Adapter Methods

Graph implements pydapter integration for external format conversion (e.g., Neo4j, NetworkX, JSON Graph Format).

#### `adapt_to()`

Convert graph to external format via registered adapter.

**Signature:**

```python
def adapt_to(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any
```

**Parameters:**

- `obj_key` (str): Adapter key (e.g., "neo4j", "networkx")
- `many` (bool): Whether to adapt multiple Graph instances. Default: False
- `**kwargs`: Additional arguments passed to adapter

**Returns:** Any - Adapted object (format depends on adapter)

**Default Behavior**: Uses `to_dict(mode="db")` for serialization

**Example:**

```python
# Convert to Neo4j format (requires neo4j adapter registered)
neo4j_data = graph.adapt_to("neo4j")

# Convert to NetworkX format
nx_graph = graph.adapt_to("networkx")
```

---

#### `adapt_from()`

Create Graph from external format via registered adapter.

**Signature:**

```python
@classmethod
def adapt_from(cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any) -> Graph
```

**Parameters:**

- `obj` (Any): Source object in external format
- `obj_key` (str): Adapter key
- `many` (bool): Whether to deserialize multiple Graph instances. Default: False
- `**kwargs`: Additional arguments passed to adapter

**Returns:** Graph - Deserialized Graph instance

**Default Behavior**: Uses `from_dict()` for deserialization

**Example:**

```python
# Create from Neo4j query result
graph = Graph.adapt_from(neo4j_result, "neo4j")

# Create from NetworkX graph
graph = Graph.adapt_from(nx_graph, "networkx")
```

---

#### `adapt_to_async()`

Async version of `adapt_to()` for async adapters.

**Signature:**

```python
async def adapt_to_async(self, obj_key: str, many: bool = False, **kwargs: Any) -> Any
```

**Parameters:** Same as `adapt_to()`

**Returns:** Any - Adapted object

**Use Case**: Async database operations, streaming serialization

**Example:**

```python
# Async conversion to database format
async with db_session() as session:
    db_data = await graph.adapt_to_async("postgres")
    await session.execute(insert_query(db_data))
```

---

#### `adapt_from_async()`

Async version of `adapt_from()` for async adapters.

**Signature:**

```python
@classmethod
async def adapt_from_async(
    cls, obj: Any, obj_key: str, many: bool = False, **kwargs: Any
) -> Graph
```

**Parameters:** Same as `adapt_from()`

**Returns:** Graph - Deserialized Graph instance

**Use Case**: Async database queries, streaming deserialization

**Example:**

```python
# Async load from database
async with db_session() as session:
    result = await session.execute(graph_query)
    graph = await Graph.adapt_from_async(result, "postgres")
```

---

**Adapter Registration**:

Adapters must be registered before use. See [pydapter documentation](https://github.com/khive-ai/pydapter) for registration details.

```python
from pydapter import Adapter

# Register custom adapter
@Graph.register_adapter("custom_format")
class CustomAdapter(Adapter):
    def adapt(self, graph: Graph) -> dict:
        return {"nodes": [...], "edges": [...]}
```

---

## Design Rationale

### Pile-Backed Storage

**Why Piles instead of plain dicts?**

1. **Type Safety**: Pile enforces Node/Edge type constraints
2. **Serialization**: Piles handle to_dict/from_dict automatically
3. **Thread Safety**: Piles provide RLock synchronization
4. **Consistency**: Unified interface with other lionherd primitives
5. **Progression Ordering**: Piles preserve insertion order

### Pre-Computed Adjacency Lists

**Why `_out_edges` and `_in_edges` dicts?**

1. **O(1) Neighbor Queries**: Constant-time predecessor/successor lookup
2. **Algorithm Performance**: Graph algorithms need fast adjacency access
3. **Rebuild on Deserialize**: Adjacency lists are derived data, rebuilt from edges
4. **Memory/Speed Tradeoff**: 2× edge storage (forward + backward) for O(1) queries

### Thread Safety with RLock

**Why RLock instead of Lock?**

1. **Nested Calls**: `remove_node()` calls `remove_edge()` - both synchronized
2. **Reentrant**: Same thread can acquire lock multiple times
3. **Future-Proof**: Rust port and Python 3.13+ nogil require explicit synchronization
4. **Cascade Operations**: Node removal cascades to edge removal atomically

### EdgeCondition Not Serialized

**Why are conditions runtime-only?**

1. **Arbitrary Code**: Conditions can contain closures, lambdas, external refs
2. **Non-Portable**: Cannot serialize Python code to JSON/DB
3. **Rebuild Pattern**: Conditions reconstructed from properties after deserialization
4. **Separation**: Data (properties) vs logic (conditions) separated

---

## Usage Patterns

### Pattern 1: Dependency Graph

```python
# Build dependency graph
graph = Graph()

# Add tasks as nodes
tasks = [
    Node(content={"name": "compile"}),
    Node(content={"name": "test"}),
    Node(content={"name": "deploy"}),
]

for task in tasks:
    graph.add_node(task)

# Add dependencies as edges
graph.add_edge(Edge(head=tasks[0].id, tail=tasks[1].id))  # compile → test
graph.add_edge(Edge(head=tasks[1].id, tail=tasks[2].id))  # test → deploy

# Get execution order
if graph.is_acyclic():
    execution_order = graph.topological_sort()
    print("Execute in order:")
    for task in execution_order:
        print(f"  {task.content['name']}")
```

---

### Pattern 2: State Machine

```python
# Create state machine graph
fsm = Graph()

# States
states = {
    "idle": Node(content={"state": "idle"}),
    "processing": Node(content={"state": "processing"}),
    "complete": Node(content={"state": "complete"}),
    "error": Node(content={"state": "error"}),
}

for state in states.values():
    fsm.add_node(state)

# Transitions
fsm.add_edge(Edge(head=states["idle"].id, tail=states["processing"].id, label=["start"]))
fsm.add_edge(Edge(head=states["processing"].id, tail=states["complete"].id, label=["success"]))
fsm.add_edge(Edge(head=states["processing"].id, tail=states["error"].id, label=["failure"]))
fsm.add_edge(Edge(head=states["error"].id, tail=states["idle"].id, label=["retry"]))

# Query transitions from state
current_state = states["processing"]
next_states = fsm.get_successors(current_state)
print(f"From {current_state.content['state']}, can transition to:")
for s in next_states:
    print(f"  {s.content['state']}")
```

---

### Pattern 3: Conditional Pathfinding

```python
# noqa:validation
# Define condition: only traverse high-capacity edges
class CapacityCondition(EdgeCondition):
    async def apply(self, min_capacity: float = 100.0) -> bool:
        capacity = self.properties.get("capacity", 0.0)
        return capacity >= min_capacity

# Build network graph
network = Graph()
n1, n2, n3 = Node(content={"id": "A"}), Node(content={"id": "B"}), Node(content={"id": "C"})

for node in [n1, n2, n3]:
    network.add_node(node)

# Low capacity edge (will be blocked)
low_cap = Edge(
    head=n1.id,
    tail=n2.id,
    condition=CapacityCondition(properties={"capacity": 50.0})
)

# High capacity edge
high_cap = Edge(
    head=n1.id,
    tail=n3.id,
    condition=CapacityCondition(properties={"capacity": 200.0})
)

network.add_edge(low_cap)
network.add_edge(high_cap)

# Find path with capacity constraints
path = await network.find_path(n1, n3, check_conditions=True)
if path:
    print(f"High-capacity path found: {len(path)} edges")
```

---

### Pattern 4: Graph Analysis

```python
# Analyze graph structure
print(f"Graph Statistics:")
print(f"  Nodes: {len(graph)}")
print(f"  Edges: {len(graph.edges)}")

# Find entry points
heads = graph.get_heads()
print(f"  Entry points: {len(heads)}")

# Find endpoints
tails = graph.get_tails()
print(f"  Endpoints: {len(tails)}")

# Check for cycles
if graph.is_acyclic():
    print("  Type: DAG (Directed Acyclic Graph)")
    # Topological sort possible
    sorted_nodes = graph.topological_sort()
    print(f"  Topological levels: {len(sorted_nodes)}")
else:
    print("  Type: Cyclic graph")
    print("  ⚠️ Cannot perform topological sort")
```

---

## Common Pitfalls

### Pitfall 1: Adding Edges Before Nodes

**Issue**: Attempting to add edge when head or tail nodes don't exist in graph.

```python
graph = Graph()
edge = Edge(head=node1.id, tail=node2.id)
graph.add_edge(edge)  # ❌ NotFoundError: Head node not in graph
```

**Solution**: Always add nodes before connecting them with edges.

```python
# ✓ CORRECT: Add nodes first
graph.add_node(node1)
graph.add_node(node2)
graph.add_edge(edge)
```

---

### Pitfall 2: Modifying Adjacency Lists Directly

**Issue**: Adjacency lists are private implementation details, not public API.

```python
graph._out_edges[node_id].add(edge_id)  # ❌ Breaks encapsulation
```

**Solution**: Use graph methods to modify structure.

```python
# ✓ CORRECT: Use add_edge()
graph.add_edge(Edge(head=node1.id, tail=node2.id))
```

---

### Pitfall 3: Expecting Undirected Edges

**Issue**: Graph is directed by design. Undirected behavior requires explicit bidirectional edges.

```python
# Only creates A → B, not B → A
graph.add_edge(Edge(head=a.id, tail=b.id))

# B → A path doesn't exist
path = await graph.find_path(b, a)  # Returns None
```

**Solution**: Add edges in both directions for undirected behavior.

```python
# ✓ CORRECT: Create bidirectional connection
graph.add_edge(Edge(head=a.id, tail=b.id))
graph.add_edge(Edge(head=b.id, tail=a.id))
```

---

### Pitfall 4: Topological Sort on Cyclic Graph

**Issue**: Calling topological_sort() on graph with cycles raises ValueError.

```python
# Create cycle: A → B → A
graph.add_edge(Edge(head=a.id, tail=b.id))
graph.add_edge(Edge(head=b.id, tail=a.id))

sorted_nodes = graph.topological_sort()  # ❌ ValueError: Cannot topologically sort graph with cycles
```

**Solution**: Check acyclicity before sorting.

```python
# ✓ CORRECT: Check before sorting
if graph.is_acyclic():
    sorted_nodes = graph.topological_sort()
else:
    print("Cannot sort - graph has cycles")
```

---

## Examples

### Example 1: Build System Dependency Graph

```python
from lionherd_core.base import Graph, Node, Edge

# Create build targets
graph = Graph()

targets = {
    "compile": Node(content={"task": "compile", "time": 30}),
    "test": Node(content={"task": "test", "time": 60}),
    "lint": Node(content={"task": "lint", "time": 10}),
    "deploy": Node(content={"task": "deploy", "time": 120}),
}

for target in targets.values():
    graph.add_node(target)

# Define dependencies (target → dependency)
dependencies = [
    ("test", "compile"),     # test depends on compile
    ("deploy", "compile"),   # deploy depends on compile
    ("deploy", "test"),      # deploy depends on test
    ("deploy", "lint"),      # deploy depends on lint
]

for dependent, dependency in dependencies:
    graph.add_edge(Edge(
        head=targets[dependency].id,
        tail=targets[dependent].id,
        label=["depends_on"]
    ))

# Get build order
build_order = graph.topological_sort()
print("Build execution order:")
total_time = 0
for target in build_order:
    task = target.content["task"]
    time = target.content["time"]
    total_time += time
    print(f"  {task} ({time}s)")

print(f"Total sequential time: {total_time}s")

# Find parallel opportunities (nodes at same topological level)
heads = graph.get_heads()
print(f"\nCan run in parallel first: {[t.content['task'] for t in heads]}")
```

---

### Example 2: Workflow State Machine

```python
# Create approval workflow
workflow = Graph()

# Define states
states = ["draft", "review", "approved", "rejected", "published"]
nodes = {state: Node(content={"status": state}) for state in states}

for node in nodes.values():
    workflow.add_node(node)

# Define transitions
transitions = [
    ("draft", "review", "submit"),
    ("review", "approved", "approve"),
    ("review", "rejected", "reject"),
    ("rejected", "draft", "revise"),
    ("approved", "published", "publish"),
]

for source, target, action in transitions:
    workflow.add_edge(Edge(
        head=nodes[source].id,
        tail=nodes[target].id,
        label=[action]
    ))

# Query valid transitions from state
current = "review"
next_states = workflow.get_successors(nodes[current])
transitions = workflow.get_node_edges(nodes[current], direction="out")

print(f"From '{current}', can:")
for edge in transitions:
    target_node = workflow.nodes[edge.tail]
    action = edge.label[0]
    print(f"  {action} → {target_node.content['status']}")
```

---

### Example 3: Cycle Detection in Dependencies

```python
# Detect circular dependencies
dependencies = Graph()

# Add packages
packages = {
    "auth": Node(content={"name": "auth", "version": "1.0"}),
    "db": Node(content={"name": "db", "version": "2.0"}),
    "api": Node(content={"name": "api", "version": "1.5"}),
}

for pkg in packages.values():
    dependencies.add_node(pkg)

# Add dependency edges
dependencies.add_edge(Edge(head=packages["auth"].id, tail=packages["db"].id))
dependencies.add_edge(Edge(head=packages["api"].id, tail=packages["auth"].id))

# Accidentally create cycle
dependencies.add_edge(Edge(head=packages["db"].id, tail=packages["api"].id))

# Detect cycle
if not dependencies.is_acyclic():
    print("❌ Circular dependency detected!")
    print("Cannot determine installation order")

    # Remove problematic edge
    problem_edge = list(dependencies.edges)[-1]
    dependencies.remove_edge(problem_edge)
    print(f"Removed edge to break cycle")

# Now can compute install order
if dependencies.is_acyclic():
    install_order = dependencies.topological_sort()
    print("\n✓ Install order:")
    for pkg in install_order:
        print(f"  {pkg.content['name']} v{pkg.content['version']}")
```

---

### Example 4: Shortest Path with Visualization

```python
# Find shortest path in network
network = Graph()

# Create network topology
locations = ["NYC", "LAX", "CHI", "ATL", "SEA"]
nodes = {loc: Node(content={"city": loc}) for loc in locations}

for node in nodes.values():
    network.add_node(node)

# Add connections (bidirectional)
connections = [
    ("NYC", "CHI"), ("NYC", "ATL"),
    ("CHI", "LAX"), ("CHI", "SEA"),
    ("ATL", "LAX"), ("SEA", "LAX"),
]

for city1, city2 in connections:
    # Bidirectional: add edges in both directions
    network.add_edge(Edge(head=nodes[city1].id, tail=nodes[city2].id, label=["route"]))
    network.add_edge(Edge(head=nodes[city2].id, tail=nodes[city1].id, label=["route"]))

# Find shortest path
start, end = "NYC", "LAX"
path = await network.find_path(nodes[start], nodes[end])

if path:
    print(f"Shortest path from {start} to {end}:")
    current = start
    for edge in path:
        next_city = network.nodes[edge.tail].content["city"]
        print(f"  {current} → {next_city}")
        current = next_city
    print(f"Total hops: {len(path)}")
else:
    print(f"No path from {start} to {end}")
```

---

## Summary

**Graph** provides directed graph operations with algorithm support:

**Key Features**:

- Pile-backed storage (nodes + edges)
- O(1) adjacency queries via pre-computed lists
- Thread-safe operations (@synchronized with RLock)
- Classical algorithms (cycle detection, topological sort, BFS pathfinding)
- Edge conditions for dynamic traversal logic
- Full serialization with automatic adjacency rebuilding

**Core Operations**:

- `add_node()`, `remove_node()` - O(1) / O(d)
- `add_edge()`, `remove_edge()` - O(1)
- `get_predecessors()`, `get_successors()` - O(k) where k = degree
- `get_heads()`, `get_tails()` - O(V)
- Direct pile access: `graph.nodes[id]`, `graph.edges[id]` - O(1)

**Algorithms**:

- `is_acyclic()` - O(V + E) three-color DFS
- `topological_sort()` - O(V + E) Kahn's algorithm
- `find_path()` - O(V + E) BFS shortest path

**Design Principles**:

- Adjacency lists derived from edges (rebuild on deserialize)
- RLock enables nested synchronized calls
- EdgeCondition runtime-only (not serialized)
- Directed by design (explicit bidirectional edges for undirected behavior)

**Use Cases**: Dependency resolution, workflow graphs, state machines, network topology, build systems, task scheduling.

See `src/lionherd_core/base/graph.py` for full implementation details.

---

## See Also

- [Element](element.md) - Base class with identity and serialization
- [Node](node.md) - Content-bearing vertex for graphs
- [Pile](pile.md) - Thread-safe typed collection backing storage
- [Progression](progression.md) - Ordered UUID sequence
- [Flow](flow.md) - Dual-pile workflow state machine

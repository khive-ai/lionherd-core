# CLAUDE.md - Claude Code Guidance

This file provides specific guidance for Claude Code (claude.ai/code) when working with this repository.

**Repository**: <https://github.com/khive-ai/lionherd-core>
**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

---

## Quick Commands Reference

```bash
# Setup
uv sync --all-extras
uv run pre-commit install

# Testing & Quality
uv run pytest --cov=lionherd_core --cov-report=term-missing  # Coverage ≥80%
uv run ruff format .   # Format (line length: 100)
uv run ruff check .    # Lint (must pass)
uv run mypy src/       # Type check (all public functions)
```

---

## Architecture Philosophy

**Core Principle**: Protocol-based composition over inheritance (Rust traits, Go interfaces).

### Protocol Pattern

```python
from uuid import uuid4
from lionherd_core.protocols import Observable, Serializable, implements

# ❌ WRONG: Inheritance creates tight coupling
class Agent(Observable, Serializable): pass

# ✅ CORRECT: Structural typing, explicit capabilities
@implements(Observable, Serializable)
class Agent:
    def __init__(self):
        self.id = uuid4()                # Observable requirement
    def to_dict(self, **kwargs):         # Serializable requirement
        return {"id": str(self.id)}
```

**Design Intent**: `@implements()` enforces that the class defines protocol methods in its body, not via inheritance.

---

## Three-Layer Architecture

```text
1. Protocols: Observable, Serializable, Adaptable (structural typing)
2. Base: Element, Node, Pile[T], Graph, Flow, Progression (core data structures)
3. Types: Spec, Operable, LNDL Parser (Pydantic integration + LLM output parsing)
```

---

## LNDL Parser

**Purpose**: Fuzzy parser for LLM output with high tolerance for errors.

**Trade-off**: +~50-90μs overhead, <5% failure rate (vs 40-60% with strict JSON)

**Workflow**:

```python
# Define Pydantic model
Spec(MyModel, name="result") → Operable([specs]) → parse_lndl_fuzzy(llm_response, operable)
```

**Pipeline**: parser.py (tokenize) → resolver.py (map fields) → fuzzy.py (handle variations)

---

## Key Data Structures

### Node - Polymorphic Content Container

```python
from lionherd_core import Node

# Node has toml/yaml adapters by default
node = Node(content={"key": "value"})  # Content must be dict/Serializable/BaseModel
toml_str = node.adapt_to("toml")  # Works out of the box
yaml_str = node.adapt_to("yaml")  # Works out of the box

# Subclasses have isolated registries - must register explicitly
from pydapter.adapters import TomlAdapter

class CustomNode(Node):
    custom_field: str = "data"

CustomNode.register_adapter(TomlAdapter)  # Required for subclasses
custom = CustomNode(content={"x": 1})
custom.adapt_to("toml")  # Now works
```

### Pile[T] - Type-Safe Collections

- O(1) lookup: `pile[uuid]`
- Predicate queries, thread-safe
- Generic type support

### Graph - Directed with Conditional Edges

```python
# ❌ WRONG: Removed in v1.0.0-alpha4
node = graph.get_node(uuid)

# ✅ CORRECT: Direct Pile access
node = graph.nodes[uuid]

# Async operations
path = await graph.find_path(start, end)
```

### Flow - Composition Pattern

```python
# ❌ WRONG: Removed in v1.0.0-alpha4
flow.pile.add(item)

# ✅ CORRECT: Explicit composition API
flow.items.add(item)  # or flow.add_item(item)
```

---

## Breaking Changes (v1.0.0-alpha4)

1. **Exceptions**: `ValueError` → `NotFoundError`/`ExistsError` (from `lionherd_core.errors`)
2. **Graph access**: `graph.get_node()` removed → `graph.nodes[uuid]`
3. **Flow composition**: `flow.pile` removed → `flow.items` or `flow.add_item()`

---

## Common Pitfalls

### 1. Don't inherit from protocols

```python
# ❌ WRONG                          # ✅ CORRECT
class MyClass(Observable): pass    @implements(Observable)
                                   class MyClass:
                                       def __init__(self): self.id = uuid4()
```

### 2. Don't use `@implements()` for inherited methods

```python
class Parent:
    def to_dict(self): ...

# ❌ WRONG                          # ✅ CORRECT
@implements(Serializable)          @implements(Serializable)
class Child(Parent): pass          class Child(Parent):
                                       def to_dict(self): return super().to_dict()
```

### 3. Node.content type constraint

```python
# ❌ WRONG: String not allowed
Node(content="x")

# ✅ CORRECT: Dict, Serializable, or BaseModel
Node(content={"key": "value"})
```

### 4. Remember async

```python
# ❌ WRONG
path = graph.find_path(start, end)

# ✅ CORRECT
path = await graph.find_path(start, end)
```

---

## Adapter Pattern Details

**Supported**: Node, Pile, Graph
**NOT Supported**: Element, Flow, Progression, Edge

**Isolation**: Each subclass has isolated adapter registry (no inheritance from parent).

**Rationale**: Prevents adapter pollution, explicit over implicit (Rust-like).

---

## CI/CD Expectations

PRs must pass:

- ✅ Tests (pytest)
- ✅ Coverage ≥80%
- ✅ Type checking (mypy)
- ✅ Linting (ruff check)
- ✅ Formatting (ruff format)
- ✅ Notebooks (if modified)
- ✅ Links (if docs modified)

**Workflows**: `.github/workflows/validate-notebooks.yml`, `validate-links.yml`

---

## Contributing Guidelines

1. Check existing patterns (multi-alpha evolution)
2. Read related tests (behavior + edge cases)
3. Maintain type safety (runtime validation is core)
4. Preserve protocol semantics (`@implements()` strict)
5. Minimal dependencies (pydapter + anyio only)
6. Document breaking changes (CHANGELOG.md)

**Commit Format**: `type(scope): subject` where type ∈ {feat, fix, docs, test, refactor, perf, chore}

---

## Additional Resources

- **README.md**: Use cases, examples, installation
- **CONTRIBUTING.md**: Full contribution workflow
- **AGENTS.md**: Quick reference for AI agents
- **CHANGELOG.md**: API evolution
- **notebooks/tutorials/**: Executable examples
- **docs/**: MkDocs documentation

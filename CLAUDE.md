# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Repository**: <https://github.com/khive-ai/lionherd-core>
**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

## Repository Structure

```text
lionherd-core/
├── docs/api/           # Sphinx API reference
├── notebooks/          # tutorials/ (CI validated), references/ (advanced)
├── src/lionherd_core/
│   ├── base/           # Element, Node, Pile, Graph, Flow, Progression
│   ├── libs/           # concurrency, schema_handlers, string_handlers
│   ├── lndl/           # parser, resolver, fuzzy (LLM output parsing)
│   ├── types/          # Spec, Operable (Pydantic integration)
│   └── protocols.py    # Observable, Serializable, Adaptable
└── tests/              # 99% coverage, property-based + unit
```

## Essential Commands

```bash
# Setup
uv sync --all-extras
uv run pre-commit install

# Testing
uv run pytest                                              # All tests
uv run pytest --cov=lionherd_core --cov-report=term-missing  # Coverage (≥80%)
uv run pytest -m unit                                      # Unit only
uv run pytest --nbmake notebooks/tutorials/                # Validate notebooks

# Quality
uv run ruff format .   # Format (line length: 100)
uv run ruff check .    # Lint (must pass)
uv run mypy src/       # Type check (all public functions)
```

## Architecture Overview

### Core Philosophy: Protocol-Based Composition

Structural typing (Rust traits, Go interfaces) → loose coupling, no inheritance.

```python
from uuid import uuid4
from lionherd_core.protocols import Observable, Serializable, implements

# ❌ Inheritance: Multiple inheritance complexity
class Agent(Observable, Serializable, Adaptable): pass

# ✅ Protocol: No inheritance, explicit capabilities
@implements(Observable, Serializable)
class Agent:
    def __init__(self):
        self.id = uuid4()                # Observable
    def to_dict(self, **kwargs):         # Serializable
        return {"id": str(self.id)}
```

**Design intent**: `@implements()` declares class implements protocol methods in its body (not via inheritance). Enforces explicit capability declaration.

### Three-Layer Architecture

```text
1. Protocols: Observable, Serializable, Adaptable
2. Base: Element (UUID+metadata), Node (polymorphic content), Pile[T] (type-safe collections),
         Graph (directed+conditional edges), Flow (Pile of Progressions), Progression (UUID sequences)
3. Types: Spec (Pydantic schema), Operable (Spec collection), LNDL Parser (LLM output → Python)
```

### LNDL (Language InterOperable Network Directive Language)

Fuzzy parser for LLM output → tolerates typos, formatting inconsistencies, missing tags.

**Syntax**: `<lvar ModelName.field var_name>value</lvar>` | `OUT{result_name: [var1, ...]}`

**Workflow**: Define Pydantic model → `Spec(MyModel, name="result")` → `Operable([specs])` → `parse_lndl_fuzzy(llm_response, operable)` → access `result.result_name.field`

**Trade-off**: +10-50ms for <5% failure (vs 40-60% strict JSON). Pipeline: parser.py (tokenize) → resolver.py (map fields) → fuzzy.py (handle variations) → `parse_lndl_fuzzy()` (entry point).

### Key Data Structures

- **Element**: UUID identity, timestamps, metadata (foundation)
- **Pile[T]**: Type-safe collection, O(1) lookup (`pile[uuid]`), predicate queries, thread-safe
- **Graph**: Directed graph, conditional edges, `await find_path()`, storage via `nodes`/`edges` (Piles)
- **Flow**: Composition pattern (`items` Pile + `progressions` Pile), NOT inheritance
- **Progression**: Ordered UUID sequence, thread-safe, represents execution order

### Adapter Pattern

Per-class registries (isolated state). **Supported**: Node, Pile, Graph. **NOT**: Element, Flow, Progression, Edge.

```python
from lionherd_core import Node
from pydapter import Adapter, to_json, to_dict

Node.register_adapter(Adapter(to_json=to_json(), to_dict=to_dict()))
node = Node(content="x")
data = node.adapt_to("json")  # Sync
data = await node.adapt_to_async("postgres")  # Async
```

## Important Conventions

### Breaking Changes (v1.0.0-alpha4)

1. `ValueError` → `NotFoundError`/`ExistsError` (`lionherd_core.errors`)
2. `graph.get_node()` removed → `graph.nodes[uuid]`
3. `flow.pile` removed → `flow.items` or `flow.add_item()`

### Type Hints, Testing, Docstrings

- **Types**: Required for public functions, `from __future__ import annotations`, protocol types, generics (`Pile[Agent]`)
- **Testing**: Hypothesis property tests, markers (`@pytest.mark.unit/property/slow`), 80%+ coverage, async auto-enabled
- **Docstrings**: Google-style (Args, Returns, Raises)

### Commit Messages

`type(scope): subject` where type ∈ {feat, fix, docs, test, refactor, perf, chore}

```text
feat(lndl): add fuzzy parsing for malformed tags

Handles typos in LNDL tag names and missing closing tags.

Closes #123
```

## Code Structure Insights

**libs/**: Low-level utilities (concurrency: async primitives; schema_handlers: TS schema, YAML; string_handlers: conversions, JSON extraction)

**Protocol Semantics**: Runtime checkable via `isinstance(obj, Protocol)` → structural typing, no inheritance needed.

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

### 3. Don't access Flow via `self.pile`

```python
# ❌ flow.pile.add(item)            # ✅ flow.items.add(item) or flow.add_item(item)
```

### 4. Don't use `graph.get_node()`

```python
# ❌ node = graph.get_node(uuid)    # ✅ node = graph.nodes[uuid]
```

### 5. Remember async

```python
# ✅ path = await graph.find_path(start, end)    # ❌ path = graph.find_path(start, end)
```

## CI/CD Expectations

PRs must pass: ✅ Tests | ✅ Coverage ≥80% | ✅ mypy | ✅ ruff check | ✅ ruff format | ✅ Notebooks (if modified) | ✅ Links (if docs modified)

**Workflows**: `validate-notebooks.yml` (tutorials strict, references lenient), `validate-links.yml` (markdown links)

## When Contributing

1. Check existing patterns (multi-alpha evolution)
2. Read related tests (behavior + edge cases)
3. Maintain type safety (runtime validation core)
4. Preserve protocol semantics (`@implements()` strict)
5. Minimal dependencies (pydapter + anyio only)
6. Document breaking changes (CHANGELOG.md)

## Related Resources

- **README.md**: Use cases, examples, installation
- **CONTRIBUTING.md**: Workflow details
- **CHANGELOG.md**: API evolution
- **notebooks/tutorials/**: Executable examples
- **docs/api/**: API reference

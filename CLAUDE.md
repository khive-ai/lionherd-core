# CLAUDE.md - Navigation Guide

**Repository**: <https://github.com/khive-ai/lionherd-core>
**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

---

## Quick Commands

| Task | Command |
|------|---------|
| Install | `uv sync --all-extras` |
| Test | `uv run pytest --cov=lionherd_core --cov-report=term-missing` |
| Format | `uv run ruff format .` |
| Lint | `uv run ruff check .` |
| Type Check | `uv run mypy src/` |
| Pre-commit | `uv run pre-commit install` |

**CI Requirements**: Tests ✓ | Coverage ≥80% | mypy ✓ | ruff ✓ | Notebooks ✓ (if modified)

---

## Directory Map

| Path | Contents |
|------|----------|
| `src/lionherd_core/base/` | Element, Node, Pile, Graph, Flow, Progression |
| `src/lionherd_core/protocols.py` | Observable, Serializable, Adaptable |
| `src/lionherd_core/types/` | Spec, Operable (Pydantic integration) |
| `src/lionherd_core/lndl/` | Fuzzy LLM output parser (lexer/parser/AST) |
| `tests/` | Unit tests (≥80% coverage) |
| `notebooks/tutorials/` | Executable examples |

---

## Resource Guide

| I want to... | Go here |
|--------------|---------|
| Understand architecture | README.md (Philosophy section) |
| See examples | `notebooks/tutorials/*.ipynb` |
| Contribute | CONTRIBUTING.md |
| Check API evolution | CHANGELOG.md |
| Quick reference | AGENTS.md |
| Test a pattern | `tests/` (mirror structure) |

---

## Key Patterns

### Protocol-Based Composition (NOT inheritance)

```python
from lionherd_core.protocols import Observable, Serializable, implements

# ❌ WRONG: Inheritance
class Agent(Observable, Serializable): pass

# ✅ CORRECT: Structural typing
@implements(Observable, Serializable)
class Agent:
    def __init__(self): self.id = uuid4()
    def to_dict(self): return {"id": str(self.id)}
```

### Node Adapters (Isolated Per Subclass)

```python
from lionherd_core import Node
from pydapter.adapters import TomlAdapter

# Node has toml/yaml adapters by default
node = Node(content={"key": "value"})
toml_str = node.adapt_to("toml")  # Works

# Subclasses have isolated registries
class CustomNode(Node):
    custom_field: str = "data"

CustomNode.register_adapter(TomlAdapter)  # Required
custom = CustomNode(content={"x": 1})
custom.adapt_to("toml")  # Now works
```

### Pile[T] Access Patterns

```python
pile = Pile(items=[el1, el2], item_type=Element)

pile[uuid]           # O(1) UUID lookup
pile[0]              # Index access
pile[1:3]            # Slice access
pile[lambda x: ...]  # Predicate filter
```

### Graph Direct Access

```python
# ❌ WRONG (removed in alpha4)
node = graph.get_node(uuid)

# ✅ CORRECT
node = graph.nodes[uuid]
path = await graph.find_path(start, end)
```

### Flow Composition

```python
# ❌ WRONG (removed in alpha4)
flow.pile.add(item)

# ✅ CORRECT
flow.items.add(item)
flow.add_item(item, progressions=["branch1"])
```

### LNDL Parser Workflow

```python
from lionherd_core.types import Spec, Operable
from lionherd_core.lndl import parse_lndl_fuzzy

# Define Pydantic model
spec = Spec(MyModel, name="result")
operable = Operable([spec])

# Parse fuzzy LLM output (+50-90μs overhead, <5% failure vs 40-60% with strict JSON)
result = parse_lndl_fuzzy(llm_response, operable)
```

---

## Common Pitfalls

| Issue | Solution |
|-------|----------|
| Inheriting from protocols | Use `@implements()` for structural typing |
| Node.content = string | Requires dict/Serializable/BaseModel |
| Missing await on graph ops | Always `await graph.find_path()` |
| Using `@implements()` on inherited methods | Only apply to methods defined in class body |
| Adapter inheritance | Subclasses don't inherit adapters - register explicitly |

---

## Breaking Changes

### v1.0.0-alpha6 (Latest)

| Change | Before | After |
|--------|--------|-------|
| LNDL architecture | Regex-based parser | Lexer → Parser → AST |
| Error handling | String exceptions | Typed exceptions (ParseError, ValidationError) |
| Performance | ~100-200μs | ~50-90μs (lexer optimization) |

### v1.0.0-alpha5

| Change | Before | After |
|--------|--------|-------|
| Spec/Operable | Complex internal refs | Pydantic-based validation |
| Type resolution | Runtime inspection | Static type mapping |

### v1.0.0-alpha4

| Change | Before | After |
|--------|--------|-------|
| Exceptions | `ValueError` | `NotFoundError`/`ExistsError` |
| Graph access | `graph.get_node(uuid)` | `graph.nodes[uuid]` |
| Flow composition | `flow.pile` | `flow.items` |

See CHANGELOG.md for complete history.

---

## Contributing

**Commit Format**: `type(scope): subject`

| Type | Use Case |
|------|----------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `test` | Tests |
| `refactor` | Code restructure |
| `perf` | Performance |
| `chore` | Maintenance |

**Full details**: CONTRIBUTING.md

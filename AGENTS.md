# AGENTS.md - Quick Reference for AI Agents

**Repository**: <https://github.com/khive-ai/lionherd-core>
**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

This file provides quick reference information for AI agents (like Claude Code) working with this codebase. For comprehensive documentation, see README.md and CONTRIBUTING.md.

---

## Quick Start

```bash
uv sync --all-extras          # Install dependencies
uv run pre-commit install     # Setup hooks
uv run pytest                 # Run tests
uv run ruff format . && uv run ruff check .  # Format & lint
```

---

## Architecture Quick Reference

### Core Components

```text
src/lionherd_core/
├── base/           # Element, Node, Pile, Graph, Flow, Progression
├── protocols.py    # Observable, Serializable, Adaptable
├── types/          # Spec, Operable (Pydantic integration)
└── lndl/           # Fuzzy LLM output parser
```

### Key Patterns

**Protocol-Based Composition** (NOT inheritance):

```python
from lionherd_core.protocols import Observable, Serializable, implements

@implements(Observable, Serializable)
class Agent:
    def __init__(self):
        self.id = uuid4()  # Observable requirement
    def to_dict(self): ...  # Serializable requirement
```

**Composition Over Inheritance**:

```python
flow.items.add(item)     # ✅ Correct (Flow HAS-A Pile)
flow.pile.add(item)      # ❌ Removed in v1.0.0-alpha4
```

**Adapter Support**:

- Node/Pile/Graph: ✅ Support pydapter
- Element/Flow/Progression: ❌ No adapter support

---

## Common Pitfalls

1. **Don't inherit from protocols** → Use `@implements()`
2. **Node.content requires dict/Serializable/BaseModel** → NOT strings
3. **Graph operations are async** → Use `await graph.find_path()`
4. **Direct Pile access** → `graph.nodes[uuid]` not `graph.get_node(uuid)`
5. **Flow composition** → `flow.items` not `flow.pile`

---

## Breaking Changes (v1.0.0-alpha4)

- `ValueError` → `NotFoundError`/`ExistsError` (from `lionherd_core.errors`)
- `graph.get_node()` removed → `graph.nodes[uuid]`
- `flow.pile` removed → `flow.items`

---

## Testing

```bash
uv run pytest --cov=lionherd_core --cov-report=term-missing  # Coverage ≥80%
uv run pytest -m unit           # Unit tests only
uv run pytest --nbmake notebooks/tutorials/  # Validate notebooks
```

**CI Requirements**: Tests pass | Coverage ≥80% | mypy | ruff | Notebooks (if modified)

---

## References

See comprehensive documentation in:

- **README.md**: Installation, use cases, examples
- **CONTRIBUTING.md**: Full contribution workflow
- **CLAUDE.md**: Detailed architecture for Claude Code
- **notebooks/tutorials/**: Executable examples

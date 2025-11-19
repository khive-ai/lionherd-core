# AGENTS.md - Quick Reference

**Repository**: <https://github.com/khive-ai/lionherd-core>
**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

---

## Quick Start

```bash
uv sync --all-extras                  # Install
uv run pre-commit install             # Setup hooks
uv run pytest --cov                   # Test (≥80% coverage)
uv run ruff format . && ruff check .  # Format & lint
```

---

## Codebase Structure

```text
src/lionherd_core/
├── base/
│   ├── element.py        # UUID + created_at + metadata (foundation)
│   ├── node.py           # Element + content + embedding (polymorphic container)
│   ├── pile.py           # Pile[T]: O(1) UUID lookup, thread-safe collections
│   ├── graph.py          # Directed graph with conditional edges
│   ├── flow.py           # Flow[E, P]: items + progressions (composition)
│   └── progression.py    # Ordered UUID sequence (list + Element)
├── protocols.py          # Observable, Serializable, Adaptable (structural typing)
├── types/
│   ├── spec.py           # Pydantic model specifications
│   └── operable.py       # Multi-spec validation container
└── lndl/
    ├── lexer.py          # Tokenization (50-90μs)
    ├── parser.py         # AST construction
    ├── ast.py            # Abstract syntax tree
    └── fuzzy.py          # Error-tolerant parsing (<5% failure rate)

tests/                    # Mirror structure, ≥80% coverage
notebooks/tutorials/      # Executable examples
```

---

## Finding Examples

| Pattern | File Path |
|---------|-----------|
| Protocol usage | `tests/test_protocols.py` |
| Node adapters | `tests/base/test_node.py` |
| Pile operations | `tests/base/test_pile.py` |
| Graph algorithms | `tests/base/test_graph.py` |
| Flow composition | `tests/base/test_flow.py` |
| LNDL parsing | `tests/lndl/test_fuzzy.py` |
| Spec/Operable | `tests/types/test_spec.py` |
| Tutorials | `notebooks/tutorials/*.ipynb` |

---

## Common Workflows

### 1. Add New Feature

- Write test in `tests/` (mirror structure)
- Implement in `src/lionherd_core/`
- Run `uv run pytest --cov` (ensure ≥80%)
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Type check: `uv run mypy src/`

### 2. Fix Bug

- Add regression test in `tests/`
- Fix in `src/`
- Verify all tests pass
- Update CHANGELOG.md if breaking

### 3. Add Adapter

- Implement pydapter `Adapter` subclass
- Register on target class: `MyNode.register_adapter(MyAdapter)`
- Test in `tests/base/test_node.py` or similar
- **Note**: Subclasses have isolated adapter registries

### 4. Extend LNDL Parser

- Modify lexer (`lndl/lexer.py`) for new tokens
- Update parser (`lndl/parser.py`) for grammar
- Adjust AST (`lndl/ast.py`) for new node types
- Test in `tests/lndl/`

---

## Anti-Patterns

| ❌ Don't | ✅ Do |
|---------|------|
| `class X(Observable)` | `@implements(Observable)` + define methods |
| `Node(content="string")` | `Node(content={"key": "value"})` |
| `graph.get_node(uuid)` | `graph.nodes[uuid]` |
| `flow.pile.add(item)` | `flow.items.add(item)` |
| Forget `await` on graph ops | `await graph.find_path()` |
| Inherit adapters | `CustomNode.register_adapter()` |

---

## References

- **README.md**: Installation, philosophy, use cases
- **CLAUDE.md**: Architecture details, breaking changes
- **CONTRIBUTING.md**: Full contribution workflow
- **CHANGELOG.md**: API evolution
- **notebooks/tutorials/**: Executable examples

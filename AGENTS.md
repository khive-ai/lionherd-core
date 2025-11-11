# AGENTS.md

**Repository**: <https://github.com/khive-ai/lionherd-core>
**Copyright**: © 2025 HaiyangLi (Ocean) - Apache 2.0 License

---

## Dev Environment Tips

### Setup

- **Python**: ≥3.11 required
- **Package manager**: Use `uv` (NOT pip/poetry)
- **Install**: `uv sync --all-extras`
- **Pre-commit hooks**: `uv run pre-commit install`

### Architecture

- **Base classes**: `src/lionherd_core/base/` (Element, Node, Pile, Graph, Flow, Progression)
- **Protocols**: `src/lionherd_core/protocols.py` (Observable, Serializable, Adaptable)
- **Type system**: `src/lionherd_core/types/` (Spec, Operable, LNDL parser)
- **Tests**: `tests/` mirrors `src/` structure

### Key Patterns

- **Protocols over inheritance**: Use `@implements()` decorator, define methods in class body
- **Composition**: Flow HAS-A Pile (not IS-A), use `self.items` not `self.pile`
- **Adapters**: Only Node/Pile/Graph support pydapter (NOT Element/Flow/Progression)
- **Async**: Graph operations are async (`await graph.find_path()`)

### Breaking Changes (v1.0.0-alpha4)

- `ValueError` → `NotFoundError`/`ExistsError` (from `lionherd_core.errors`)
- `graph.get_node()` → `graph.nodes[uuid]` (direct Pile access)
- `flow.pile` → `flow.items` (composition API)

---

## Testing Instructions

### CI/CD Workflows

- **Location**: `.github/workflows/`
- **Notebooks**: `validate-notebooks.yml` (tutorials strict, references lenient)
- **Links**: `validate-links.yml` (markdown validation)

### Running Tests

```bash
# All tests
uv run pytest

# With coverage (maintain ≥80%)
uv run pytest --cov=lionherd_core --cov-report=term-missing

# By marker
uv run pytest -m unit           # Unit tests only
uv run pytest -m property       # Property-based (Hypothesis)
uv run pytest -m "not slow"     # Skip slow tests

# Single file
uv run pytest tests/base/test_element.py

# Notebooks
uv run pytest --nbmake notebooks/tutorials/
```

### Quality Gates

```bash
# Format (line length: 100)
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```

### Required Checks

- ✅ All tests pass
- ✅ Coverage ≥80%
- ✅ Type checking passes (`mypy`)
- ✅ Linting passes (`ruff check`)
- ✅ Formatting applied (`ruff format`)
- ✅ Notebooks execute (if modified)
- ✅ Links valid (if docs modified)

---

## PR Instructions

### Commit Message Format

`type(scope): subject`

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

**Example**: `feat(lndl): add fuzzy parsing for malformed tags`

### Pre-commit Requirements

- Run `uv run pre-commit run --all-files`
- Fix all linting/formatting issues
- Ensure tests pass locally
- Verify notebook execution (if applicable)

### PR Checklist

- [ ] Tests added for new functionality
- [ ] Documentation updated (docstrings, notebooks, API docs)
- [ ] Breaking changes documented in CHANGELOG.md
- [ ] Type hints on all public functions
- [ ] Coverage maintained at ≥80%
- [ ] Related issue referenced (`Closes #N`)

### Code Standards

- **Type hints**: Required on all public functions (`from __future__ import annotations`)
- **Docstrings**: Google-style with Args/Returns/Raises
- **Protocols**: Use `@implements()` for explicit capability declaration
- **Tests**: Property-based (Hypothesis) for edge cases, unit tests for behavior
- **Dependencies**: Minimal (only pydapter + anyio in core)

---

## Architecture Notes for Agents

### Protocol System

- `@implements()` declares protocol implementation (Rust-like trait semantics)
- Design intent: class defines methods in its body, not via inheritance
- Runtime: `isinstance(obj, Protocol)` checks structural typing

### LNDL Parser

- **Purpose**: Parse LLM output with tolerance for typos/formatting variations
- **Trade-off**: +10-50ms overhead, <5% failure vs 40-60% with strict JSON
- **Pipeline**: parser.py (tokenize) → resolver.py (map fields) → fuzzy.py (handle variations)
- **Entry point**: `parse_lndl_fuzzy(llm_response, operable)`

### Data Structure Semantics

- **Element**: UUID + metadata foundation (no adapter support)
- **Node**: Element + polymorphic content (supports adapters)
- **Pile[T]**: Type-safe collection, O(1) lookup, thread-safe
- **Graph**: Directed with conditional edges, async path finding
- **Flow**: Composition (items + progressions), NOT inheritance
- **Progression**: Ordered UUID sequence, thread-safe

### Common Mistakes

1. Inheriting from protocols → Use `@implements()` instead
2. Using `@implements()` without defining methods in class body
3. Accessing `flow.pile` → Use `flow.items`
4. Using `graph.get_node()` → Use `graph.nodes[uuid]`
5. Forgetting `await` on `graph.find_path()`
6. Assuming Element has adapters → Only Node/Pile/Graph do

---

## References

- **CLAUDE.md**: Detailed architecture guide for Claude Code
- **CONTRIBUTING.md**: Full contribution workflow
- **README.md**: Use cases and examples
- **notebooks/tutorials/**: Executable examples
- **docs/api/**: API reference documentation

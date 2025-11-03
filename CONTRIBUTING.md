# Contributing

We welcome contributions to lionherd-core! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Clone and Install

```bash
git clone https://github.com/khive-ai/lionherd-core.git
cd lionherd-core

# Install all dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=lionherd_core --cov-report=term-missing

# Run specific markers
uv run pytest -m unit          # Unit tests only
uv run pytest -m property      # Property-based tests
uv run pytest -m "not slow"    # Skip slow tests

# Run specific test file
uv run pytest tests/base/test_element.py
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## Code Style

### Formatting

- Use `ruff format` for all Python code
- Line length: 100 characters
- Follow PEP 8 conventions

### Type Hints

- All public functions must have type hints
- Use `from __future__ import annotations` for forward references
- Use protocol types for structural typing

```python
from __future__ import annotations
from typing import Protocol

def process_items(items: list[Element]) -> dict[str, Any]:
    """Process elements and return summary."""
    ...
```

### Documentation

- All public classes/functions need docstrings
- Use Google-style docstrings
- Include type information in docstrings

```python
def alcall(
    func: Callable,
    items: list[Any],
    max_concurrent: int = 10,
) -> list[Any]:
    """Execute function concurrently over items.

    Args:
        func: Async function to execute
        items: List of items to process
        max_concurrent: Maximum concurrent executions

    Returns:
        List of results in order

    Raises:
        ValueError: If max_concurrent < 1
    """
    ...
```

## Testing Guidelines

### Test Structure

```text
tests/
├── base/           # Base classes (Element, Node, Pile, Graph)
├── lndl/           # LNDL parser tests
├── ln/            # Utility function tests
└── libs/          # Library-specific tests
```

### Writing Tests

```python
import pytest
from lionherd_core import Element

class TestElement:
    """Test Element class."""

    @pytest.mark.unit
    def test_element_creation(self):
        """Test creating element with UUID."""
        element = Element()
        assert element.id is not None
        assert element.created_at is not None

    @pytest.mark.unit
    def test_element_equality(self):
        """Test element equality by ID."""
        e1 = Element()
        e2 = Element()
        assert e1 != e2  # Different IDs
```

### Property-Based Testing

Use Hypothesis for edge cases:

```python
from hypothesis import given
from hypothesis import strategies as st

@pytest.mark.property
@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_pile_operations(items):
    """Test Pile with various list sizes."""
    pile = Pile[int](item_type=int)
    for item in items:
        pile.add(item)
    assert len(pile) == len(items)
```

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation

### 3. Run Tests and Checks

```bash
# Run full test suite
uv run pytest

# Check coverage (should maintain 80%+)
uv run pytest --cov=lionherd_core

# Run linters
uv run ruff check .
uv run mypy src/
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature

Detailed description of changes.

Closes #123"
```

### Commit Message Format

```text
type(scope): subject

body

footer
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:

- Clear description of changes
- Reference to related issues
- Test coverage maintained
- Documentation updated

## Code Review

Pull requests require:

- ✅ All tests passing
- ✅ Code coverage ≥80%
- ✅ Type checking passes
- ✅ Linting passes
- ✅ Documentation updated
- ✅ Approval from maintainer

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions builds and publishes to PyPI

## Project Structure

```text
lionherd-core/
├── src/lionherd_core/
│   ├── base/              # Base classes
│   ├── lndl/              # LNDL parser
│   ├── ln/               # Utilities
│   │   ├── types/        # Spec/Operable
│   │   └── spec_adapters/ # Framework adapters
│   ├── libs/             # Internal libraries
│   └── protocols.py      # Protocol definitions
├── tests/                # Test suite
├── docs/                 # Documentation
└── pyproject.toml        # Project config
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Reach out to maintainers for guidance

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0.

# lionherd-core

Core utilities, base classes, and protocols for the lionherd ecosystem

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/github/khive-ai/lionherd-core/graph/badge.svg?token=FAE47FY26T)](https://codecov.io/github/khive-ai/lionherd-core)

## Overview

lionherd-core provides foundational components for building agentic AI systems:

- **Protocols**: Structural typing for Observable, Serializable,
  Adaptable objects (Rust-style traits)
- **Base Classes**: Element, Node, Pile, Graph, Flow, Progression
- **Utilities**: JSON operations, async helpers, type utilities
- **LNDL Parser**: Lion Directive Language for natural structured outputs
- **Event System**: EventBus and Broadcaster for pub/sub patterns

## Installation

```bash
pip install lionherd-core
```

### From Source

```bash
git clone https://github.com/khive-ai/lionherd-core.git
cd lionherd-core
uv sync
uv pip install -e .
```

## Quick Start

```python
from lionherd_core import Element, Pile, Observable, Serializable
from uuid import uuid4

# Create elements with protocols
class Task(Element):
    title: str
    status: str = "pending"

# Use collections
pile = Pile[Task](item_type=Task)
task = Task(id=uuid4(), title="Build agent", status="in_progress")
pile.add(task)

# Protocol checks
assert isinstance(task, Observable)
assert isinstance(task, Serializable)
```

## Core Components

### Protocols (Structural Typing)

```python
from lionherd_core.protocols import (
    Observable,      # Objects with unique identifiers
    Serializable,    # Objects that can be serialized
    Deserializable,  # Objects that can be deserialized
    Adaptable,       # Objects that adapt to external formats
    AsyncAdaptable,  # Async adaptation support
    implements,      # Decorator for explicit protocol declaration
)
```

### Base Classes

- **Element**: Minimal identity (UUID + metadata)
- **Node**: Polymorphic element with content
- **Pile**: Thread-safe typed collection
- **Graph**: Directed graph with edges and conditions
- **Flow**: Workflow state machine (Pile + Progressions)
- **Progression**: Named, trackable ordering of UUIDs

### Utilities (ln module)

- JSON operations: `json_dumps`, `json_dumpb`, `to_dict`
- Async utilities: `alcall`, `bcall`, `lcall`
- Type utilities: `Undefined`, `Unset`, `MaybeSentinel`

### LNDL (Lion Directive Language)

Natural structured output parser for LLMs:

```python
from lionherd_core.lndl import parse_lndl, Spec, Operable

class ResearchTask(Operable):
    query: str = Spec(description="Research query")
    depth: int = Spec(description="Research depth", default=3)

# Parse LLM output into structured objects
result = parse_lndl(llm_output, ResearchTask)
```

## Architecture

lionherd-core is designed for:

- **Enterprise-grade production systems**
- **Type safety with runtime validation**
- **Async-first architecture**
- **Composable protocols (trait-based design)**
- **Zero-dependency core** (optional dependencies for adapters)

## Documentation

- [API Reference](./docs/api/)
- [Architecture Guide](./docs/architecture.md)
- [Examples](./examples/)

## Development

### Setup

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/base/test_element.py

# Run with coverage
uv run pytest --cov=lionherd_core
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](./LICENSE) for details.

## Credits

Created by [HaiyangLi (Ocean)](https://github.com/ohdearquant)

Part of the [khive.ai](https://khive.ai) ecosystem.

## Related Projects

- **lionherd**: Full agentic AI orchestration framework (builds on lionherd-core)
- **lionagi**: Original agentic AI framework (5 years of evolution)
- **pydapter**: Universal data connection layer

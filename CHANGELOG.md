# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Background Processing** (#185):

- `Processor`: Priority queue-based event execution with capacity control and concurrency limiting
- `Executor`: Flow-based state tracking with O(1) status queries via named progressions
- `Event.streaming` field for processor streaming support

**Benchmarking Infrastructure** (#174, #182, #183):

- Component-organized benchmark suites (`benchmarks/graph/`, `benchmarks/flow/`, `benchmarks/pile/`, `benchmarks/lndl/`)
- CI workflow for benchmark regression detection (>10% slower fails PR, 5-10% warns)
- Manual workflow to generate benchmark baselines as GitHub artifacts

### Changed

**LNDL Architecture** (#194):

- Refactored from regex-based parsing to unified Lexer/Parser/AST architecture
- New files: `ast.py` (AST nodes), `lexer.py` (context-aware tokenizer with 17 token types)
- Context-aware lexing (strings only tokenized inside OUT{} blocks)
- Position tracking for improved error messages
- Fully backward compatible via adapter functions

### Fixed

- **LNDL**: Removed dead code in OUT{} block parsing (#194)
- **Documentation**: Fixed broken links and redirects (#173)

## [1.0.0-alpha5](https://github.com/khive-ai/lionherd-core/releases/tag/v1.0.0-alpha5) - 2025-11-12

### Changed

**Pile API**:

- **BREAKING**: `item_type` and `strict_type` now frozen (#156). Set at initialization only.
- **BREAKING**: `include()`/`exclude()` return guaranteed state (True = in pile) not action taken (#157).
- **BREAKING**: `items` changed from property to method (#159). Returns `Iterator[tuple[UUID, T]]`.

**Flow API**:

- **BREAKING**: `__init__` accepts `progressions` parameter, creates configured Pile upfront (#156).
- **BREAKING**: Validates referential integrity at construction (#156). All progression UUIDs must exist in items.
- **BREAKING**: `add_item()` parameter renamed: `progression_ids` → `progressions` (#162).
- **BREAKING**: `remove_item()` always removes from all progressions. `remove_from_progressions` parameter removed (#162).

**Progression API**:

- **BREAKING**: `__init__` removed—validation moved to `@field_validator` (#156). Invalid items raise `ValidationError` (no silent drops).
- **BREAKING**: `IndexError` → `NotFoundError` for `pop()`, `popleft()`, `_validate_index()` (#153). Consistent with Pile/Graph/Flow.

**Protocol System**:

- **BREAKING**: Protocol separation (#147, #149). Adaptable protocols split from registry mutation:
  - `Adaptable`/`AsyncAdaptable` - read-only adaptation
  - `AdapterRegisterable`/`AsyncAdapterRegisterable` - mutable registry
  - **Migration**: Update `@implements()` declarations to include both if registering adapters.
- **BREAKING**: `@implements()` strict runtime enforcement (#149). Methods must be in class body (inheritance doesn't count). Raises `TypeError` on violation.

### Removed

- **BREAKING**: Async Pile methods: `add_async()`, `remove_async()`, `get_async()` (#162). Use sync methods (O(1) CPU-bound).
- **BREAKING**: `Pile.__list__()` and `to_list()` (#162). Use built-in `list(pile)`.

### Added

- **Top-level exports** (#148, #171): All protocols/errors exported at top level. Backwards compatible.
  - Import: `from lionherd_core import Observable, NotFoundError` (was `from lionherd_core.protocols import ...`)
- **Bool protocols** (#156, #159): `Pile.__bool__`, `Progression.__bool__` for empty checks.
- **Pile iteration** (#159): `keys()` and `items()` methods for dict-like access.
- **Comprehensive documentation** (#165-#169): Migration guide, user guides (type safety, API design, validation, protocols), updated notebooks.

### Fixed

- **Flow**: `item_type`/`strict_type` correctly applied to items Pile (#156). Previous design mutated frozen fields.
- **Flow**: `add_progression()` validates referential integrity before mutation (#164, #170). Prevents inconsistent state.

## [1.0.0a4](https://github.com/khive-ai/lionherd-core/releases/tag/v1.0.0-alpha4) - 2025-11-11

### Changed

- **Error Handling**: `Graph`, `Flow`, and `Pile` now raise `NotFoundError` and
  `ExistsError` instead of `ValueError` for missing/duplicate items (#129, #131,
  #133). Exception metadata (`.details`, `.retryable`, `.__cause__`) is now
  preserved for retry logic. Update exception handlers from `except ValueError`
  to `except NotFoundError` or `except ExistsError` as appropriate.
  - `Pile.pop()` now raises `NotFoundError` (was `ValueError`) for consistency
    with `Pile.get()` and `Pile.remove()`.
- **Performance**: `Pile` methods now use single-lookup pattern (try/except vs
  if/check) for 50% performance improvement on failed lookups (#128).
- **Module Organization**: Async utilities consolidated to `libs/concurrency` for
  cleaner structure (#114).

### Removed

- **BREAKING**: `Graph.get_node()` and `Graph.get_edge()` removed in favor of
  direct Pile access (#117, #124, #132).

  **Migration**:

  ```python
  # Before
  node = graph.get_node(node_id)
  edge = graph.get_edge(edge_id)

  # After
  node = graph.nodes[node_id]
  edge = graph.edges[edge_id]
  ```

  **Rationale**: Eliminates unnecessary wrapper methods. Direct Pile access is
  more Pythonic and consistent with dict/list-like interfaces.

### Fixed

- **BREAKING**: `Element.to_dict()` `created_at_format` now applies to ALL modes
  (#39). DB mode default changed from `isoformat` (string) to `datetime` (object)
  for ORM compatibility. Migration: use `to_dict(mode='db',
  created_at_format='isoformat')` for backward compatibility.

### Added

- **Tutorial Infrastructure**: 29 executable tutorials covering concurrency
  patterns, schema/string handlers, ln utilities, and advanced workflows (#99-106).
  Includes circuit breakers, deadline management, fuzzy matching, pipelines, and
  resource lifecycle patterns.
- **API Documentation**: Complete reference docs and Jupyter notebooks for all
  base types (Element, Node, Pile, Progression, Flow, Graph, Event, Broadcaster,
  EventBus), types system (HashableModel, Operable/Spec, Sentinel), and libs
  (concurrency, schema, string, ln utilities) (#39, #43-46, #52, #54-59).
- **LNDL Documentation**: Complete API documentation and Jupyter notebooks for LNDL
  (Language InterOperable Network Directive Language) system (#53). Includes 6
  module docs (types, parser, resolver, fuzzy, prompt, errors) and 6 reference
  notebooks with 100% execution coverage (185/185 cells).
- **Node Features**: Content constraint (dict|Serializable|BaseModel|None, no
  primitives) and embedding serialization support for pgvector + JSONB (#50, #113).
- **Pile.pop() default**: Optional default parameter for safer fallback, consistent
  with `dict.pop()` behavior (#118, #123).
- **Test Coverage**: Race condition tests for Event timeout/exception/cancellation
  scenarios and fuzzy JSON kwargs regression tests (#32, #107, #112).

## [1.0.0a3](https://github.com/khive-ai/lionherd-core/releases/tag/v1.0.0-alpha3) - 2025-11-06

### Fixed

- **Memory Leaks**: `EventBus` (#22) and `Broadcaster` (#24) now use
  `weakref` for automatic callback cleanup. Prevents unbounded growth in
  long-running apps.
- **TOCTOU Races**: `Graph.add_edge()` (#21) and `Event.invoke()` (#26)
  synchronized with decorators. Eliminates 10% duplicate execution rate
  under concurrency.
- **LNDL Guard**: `ensure_no_action_calls()` (#23) prevents `ActionCall`
  persistence. Recursively detects placeholders in nested models/collections.
- **Backend Agnostic**: `Event._async_lock` now anyio-based (was
  `asyncio.Lock`). Enables Trio support.

### Changed

- **Event Idempotency**: Clarified `invoke()` caches results after
  COMPLETED/FAILED. Use `as_fresh_event()` for retry.

### Added

- Race/memory leak tests with GC validation. 100% coverage for guards.

## [1.0.0a2](https://github.com/khive-ai/lionherd-core/releases/tag/v1.0.0-alpha2) - 2025-11-05

### Added

- **Event Timeout Support**: Optional `timeout` field with validation.
  Converts `TimeoutError` to `LionherdTimeoutError` with `status=CANCELLED`,
  `retryable=True`.
- **PriorityQueue**: Async priority queue using `heapq` + anyio primitives
  with `put()`, `get()`, `put_nowait()`, `get_nowait()`. 100% test coverage.
- **LNDL Reserved Keyword Validation**: Python keyword checking for action
  names with `UserWarning`.

### Fixed

- **PriorityQueue Deadlock**: `get_nowait()` now notifies waiting putters,
  preventing deadlock with bounded queues.
- **LNDL**: Fixed typo in error messages and improved system prompt examples.

### Changed

- **Flow Architecture** (breaking): Composition over inheritance. `add()` →
  `add_progression()`, `pile` → `items`.
- **Copy Semantics** (breaking): `with_updates()` now uses
  `Literal["shallow", "deep"] | None` instead of two booleans.
- **Event Documentation**: Simplified docstrings, added `@final` to
  `invoke()`, moved rationale to tests.
- **EventStatus**: Uses `lionherd_core.types.Enum` instead of `(str, Enum)`
  for `Allowable` protocol support.

## [1.0.0a1](https://github.com/khive-ai/lionherd-core/releases/tag/v1.0.0-alpha1) - 2025-11-03

### Added

- **LNDL Action Syntax**: Added support for tool/function invocations within
  LNDL responses using `<lact>` tags. Supports both namespaced actions
  (`<lact Model.field alias>function(...)</lact>`) for mixing with lvars and
  direct actions (`<lact name>function(...)</lact>`) for entire output.
  Includes fuzzy matching support and complete validation lifecycle with
  re-validation after action execution.
- Added `py.typed` marker file for PEP 561 compliance to enable type checking support

## [1.0.0a0](https://github.com/khive-ai/lionherd-core/releases/tag/v1.0.0-alpha) - 2025-11-02

### Added

- Initial release of lionherd-core
- Core orchestration framework
- Base abstractions and protocols

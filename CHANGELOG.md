# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-alpha5](https://github.com/khive-ai/lionherd-core/releases/tag/v1.0.0-alpha5) - 2025-11-12

### Changed

- **BREAKING**: `Pile.item_type` and `Pile.strict_type` are now frozen fields (#156). Type configuration must be set at initialization and cannot be mutated afterward. Prevents runtime type confusion.
- **BREAKING**: `Pile.include()` and `Pile.exclude()` return value semantics changed (#157). Now return guaranteed state (True = in pile) rather than action taken (True = was added).
- **BREAKING**: `Pile.items` changed from property to method (#159). Returns `Iterator[tuple[UUID, T]]`.
- **BREAKING**: `Progression.__init__` removed—all normalization moved to `@field_validator` (#156). Validation is stricter: invalid items raise `ValidationError` instead of being silently dropped.
- **BREAKING**: `Flow.__init__` signature redesigned to accept `progressions` parameter and create configured `Pile` upfront (#156). Respects frozen `item_type`/`strict_type`.
- **BREAKING**: `Flow` now validates referential integrity at construction (#156). All UUIDs in progressions must exist in items pile.
- **BREAKING**: `Flow.add_item()` parameter renamed: `progression_ids` → `progressions`.
- **BREAKING**: `Flow.remove_item()` parameter removed: `remove_from_progressions` no longer supported. Always removes item from all progressions.
- **Progression exceptions**: Migrate `IndexError` → `NotFoundError` for semantic consistency
  - `pop()` without default raises `NotFoundError` instead of `IndexError`
  - `popleft()` on empty raises `NotFoundError` instead of `IndexError`
  - `_validate_index()` raises `NotFoundError` for out-of-bounds/empty
  - Rationale: "Index not found" is semantically same as "item not found" (consistent with Pile/Graph/Flow)
- **Progression docstrings**: Trimmed to API contract only (params, returns, raises)
  - Moved design rationale and "why" explanations to test docstrings
  - Pattern: Source code = what/how, tests = why/design intent
- **BREAKING**: Protocol separation - `Adaptable`/`AsyncAdaptable` split from registry mutation (#147). Classes now explicitly compose capabilities:
  - `Adaptable` / `AsyncAdaptable` - read-only adaptation (adapt_to/from)
  - `AdapterRegisterable` / `AsyncAdapterRegisterable` - mutable registry (register_adapter)

  **Migration**: Update `@implements()` declarations on custom classes inheriting from Node/Pile/Graph to include both protocols if registering adapters.

### Removed

- **BREAKING**: Async Pile methods removed: `add_async()`, `remove_async()`, `get_async()`. Use sync methods (Pile operations are O(1) CPU-bound, not I/O).
- **BREAKING**: `Pile.__list__()` and `Pile.to_list()` removed. Use built-in `list(pile)`.

### Added

- **API Exports**: All protocols and errors now exported at top level for simplified imports (#148, #171). Backwards compatible - old import paths still work.
  - Protocols: `Observable`, `Serializable`, `Adaptable`, `AdapterRegisterable`, `AsyncAdaptable`, `AsyncAdapterRegisterable`, `Deserializable`, `Containable`, `Allowable`, `Invocable`, `Hashable`, `implements`
  - Errors: `NotFoundError`, `ExistsError`, `ValidationError`, `ConfigurationError`, `ConnectionError`, `ExecutionError`, `TimeoutError`
  - Migration: `from lionherd_core import Observable, NotFoundError` (was `from lionherd_core.protocols import Observable`)
- `Pile.__bool__` protocol for empty checks (#159). `if pile:` is False when empty.
- `Pile` dict-like iteration protocol (#159): `keys()` and `items()` methods for dict-like access.
- `Progression.__bool__` protocol for empty checks (#156). Empty progressions are falsy.
- `Flow` referential integrity validation via `@model_validator` (#156).
- **BREAKING**: `@implements()` strict runtime enforcement (#147). Classes MUST define protocol methods in class body (inheritance doesn't count). Enforces Rust-like explicit trait implementation. Raises `TypeError` on violation with clear error message.
- **Documentation**: Comprehensive migration guide (`docs/migration/v1.0.0-alpha5.md`), user guides (type safety, API design, validation, protocols), and updated notebooks for all API changes (#165-#169).

### Fixed

- **Flow**: `item_type`/`strict_type` now correctly applied to items `Pile` (#156). Previous design created default Pile then mutated frozen fields.
- **Flow**: `add_progression()` now validates referential integrity before adding to pile (#164, #170). Prevents inconsistent state if progression contains invalid UUIDs. Consistent with `@model_validator` pattern.

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

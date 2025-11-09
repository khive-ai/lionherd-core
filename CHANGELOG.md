# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **BREAKING**: `Element.to_dict()` `created_at_format` now applies to ALL modes
  (#39). DB mode default changed from `isoformat` (string) to `datetime` (object)
  for ORM compatibility. Migration: use `to_dict(mode='db',
  created_at_format='isoformat')` for backward compatibility.

### Added

- Comprehensive Element API documentation and reference notebook (#39)
- **LNDL Documentation**: Complete API documentation and Jupyter notebooks for LNDL
  (Language InterOperable Network Directive Language) system (#53). Includes 6
  module docs (types, parser, resolver, fuzzy, prompt, errors) and 6 reference
  notebooks with 100% execution coverage (185/185 cells)

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

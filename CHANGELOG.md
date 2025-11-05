# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0a2] - 2025-11-05

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

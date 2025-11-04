# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-a1] - 2025-11-03

### Added

- **LNDL Action Syntax**: Added support for tool/function invocations within
  LNDL responses using `<lact>` tags. Supports both namespaced actions
  (`<lact Model.field alias>function(...)</lact>`) for mixing with lvars and
  direct actions (`<lact name>function(...)</lact>`) for entire output.
  Includes fuzzy matching support and complete validation lifecycle with
  re-validation after action execution.
- Added `py.typed` marker file for PEP 561 compliance to enable type checking support

## [1.0.0a0] - 2025-11-02

### Added

- Initial release of lionherd-core
- Core orchestration framework
- Base abstractions and protocols

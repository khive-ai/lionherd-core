# Link Check Exclusions

This file documents intentional exclusions in the link validation workflow (`.github/workflows/validate-links.yml`).

## Purpose

The link checker (lychee) excludes certain patterns to avoid false positives from pre-existing documentation debt while still catching real broken links in new contributions.

**Total Excluded**: ~80 unique broken links (as of PR #115)
**Tracking Issue**: #120 - Documentation Debt: Complete Missing User Guides and API Docs

## Why These Exclusions Exist

All excluded links represent **pre-existing documentation debt**, not issues introduced by recent changes. They fall into three categories:

1. **Path formatting issues** - Notebooks using absolute paths that don't resolve in CI
2. **Missing documentation** - Planned user guides and API docs not yet written
3. **External issues** - Broken external links or disabled features

## Excluded Patterns

### 1. Absolute `/docs/` Path References
**Pattern**: `--exclude 'file:///docs/.*'`

**Reason**: Notebooks contain absolute paths like `/docs/api/libs/concurrency.md` instead of relative paths. The actual files exist at `docs/api/libs/concurrency.md` but the absolute path format causes lychee to fail.

**Affected**: ~22 API documentation files

**Fix Required**: Update notebooks to use proper relative paths (e.g., `../../docs/api/...`)

---

### 2. Temporary Extraction Paths
**Pattern**: `--exclude 'file:///tmp/.*'`

**Reason**: The CI workflow extracts notebooks to `/tmp/nb-markdown/` for link checking, but cross-references between notebooks break because:
- Source notebooks are in `notebooks/**/*.ipynb`
- They reference each other using relative paths
- When extracted to `/tmp/`, these paths no longer resolve

**Affected**: ~24 inter-notebook references

**Fix Required**: None - this is a workflow limitation, not a real issue

---

### 3. Wrong Base Path References
**Patterns**:
- `--exclude 'file:///lndl/.*'`
- `--exclude 'file:///references/.*'`

**Reason**: Some notebooks use absolute paths missing the `notebooks/` prefix:
- `/lndl/structured_output_parsing.ipynb` → should be `notebooks/lndl/...`
- `/references/string_similarity.ipynb` → should be `notebooks/references/...`

**Affected**: ~3 files

**Fix Required**: Update notebook cross-references to include proper base path

---

### 4. Missing User Guide Documentation
**Pattern**: `--exclude 'file:///home/runner/.../docs/user_guide/.*'`

**Reason**: API documentation references user guides that haven't been written yet:
- `docs/user_guide/concurrency.md`
- `docs/user_guide/error_handling.md`
- `docs/user_guide/hashing.md`
- `docs/user_guide/polymorphism.md`
- `docs/user_guide/protocols.md`
- `docs/user_guide/pydapter.md`
- `docs/user_guide/serialization.md`
- `docs/user_guide/type_conversion.md`

**Affected**: 8 planned user guide files

**Fix Required**: Write comprehensive user guide documentation

---

### 5. Wrong Path Structure - docs/api/user_guide/*
**Pattern**: Included in composite pattern above

**Reason**: Some API docs incorrectly reference `docs/api/user_guide/*` when they should reference `docs/user_guide/*`:
- `docs/api/user_guide/concurrency.md` → should be `docs/user_guide/concurrency.md`
- `docs/api/user_guide/error_handling.md` → should be `docs/user_guide/error_handling.md`
- `docs/api/user_guide/orchestration.md` → should be `docs/user_guide/orchestration.md`
- `docs/api/user_guide/task_scheduling.md` → should be `docs/user_guide/task_scheduling.md`

**Affected**: 4 files

**Fix Required**: Update source API documentation to use correct paths

---

### 6. Wrong Base Path - user_guide/* (Missing docs/ Prefix)
**Pattern**: `--exclude 'file:///home/runner/.../user_guide/.*'`

**Reason**: Some docs reference `user_guide/*` without the `docs/` prefix:
- `user_guide/concurrency.md` → should be `docs/user_guide/concurrency.md`
- `user_guide/debugging.md` → should be `docs/user_guide/debugging.md`

**Affected**: 2 files

**Fix Required**: Add `docs/` prefix to path references

---

### 7. Missing API Documentation
**Pattern**: Included in composite pattern for `docs/api/`

**Reason**: API documentation files that should be auto-generated but don't exist yet:
- `docs/api/graph/pile.md`
- `docs/api/libs/concurrency.md`
- `docs/api/ln/alcall.md`
- `docs/api/ln/rcall.md`
- `docs/api/utils/to_dict.md`
- `docs/libs/string_handlers.md`

**Affected**: 6-7 files

**Fix Required**: Generate API documentation via automated tooling

---

### 8. Missing Tutorial Documentation
**Pattern**: `--exclude 'file:///home/runner/.../docs/tutorials/.*'`

**Reason**: Tutorial documentation that hasn't been converted from notebooks yet:
- `docs/tutorials/advanced_types.md`
- `docs/tutorials/content_caching.md`
- `docs/tutorials/json_logging.md`
- `docs/tutorials/large_scale_export.md`

**Affected**: 4-5 files

**Fix Required**: Convert tutorial notebooks to markdown documentation

---

### 9. Missing Reference Notebooks in docs/
**Pattern**: `--exclude 'file:///home/runner/.../docs/notebooks/references/.*'`

**Reason**: References to notebooks in `docs/notebooks/references/` that may exist elsewhere but not in the docs directory:
- `docs/notebooks/references/event.ipynb`
- `docs/notebooks/references/types_spec.ipynb`
- `docs/notebooks/references/types_spec_advanced.ipynb`

**Affected**: 3 files

**Fix Required**: Clarify documentation structure or copy notebooks to docs/

---

### 10. External Broken Links

#### GitHub Discussions (Not Enabled)
**Pattern**: `--exclude 'https://github.com/khive-ai/lionherd-core/discussions'`

**Reason**: GitHub Discussions feature is not enabled for this repository

**Fix Required**: Either enable GitHub Discussions or remove links to it

#### ONS UK Government Site
**Pattern**: `--exclude 'https://www\.ons\.gov\.uk/methodology/methodologicalpublications/.*'`

**Reason**: External link to UK Office of National Statistics paper - site has been restructured and link no longer works

**Fix Required**: Find updated URL or mark as archived/broken with note

---

## Summary Statistics

| Category | Count | Fix Type |
|----------|-------|----------|
| Absolute `/docs/` paths | 22 | Update notebook paths to relative |
| Missing user_guide/*.md | 8 | Write documentation |
| Wrong path structure (docs/api/user_guide) | 4 | Fix source API docs |
| Wrong base path (user_guide/*) | 2 | Add docs/ prefix |
| Missing API docs | 7 | Generate via tooling |
| Missing tutorial .md | 5 | Convert notebooks |
| Missing reference notebooks in docs/ | 3 | Clarify structure |
| /tmp/ references | 24 | Workflow limitation (no fix needed) |
| Absolute path (wrong base) | 3 | Fix path references |
| External 404s | 2 | Update or document |
| **TOTAL** | **80** | **Various** |

## How to Update This File

When adding new exclusions to `.github/workflows/validate-links.yml`:

1. Add the pattern to the workflow file
2. Document the pattern here with:
   - Clear description of what's excluded
   - Why it's excluded
   - What needs to be fixed to remove the exclusion
   - Affected file count
3. Update the tracking issue if it represents new documentation debt

## When to Remove Exclusions

Remove exclusion patterns when:

- ✅ Documentation is written and committed
- ✅ Notebooks are updated to use correct relative paths
- ✅ External links are fixed or replaced
- ✅ API documentation is generated
- ✅ Path structure issues are corrected

## References

- **PR #115**: Notebook validation workflow fixes (where these exclusions were added)
- **Analysis**: `.khive/workspaces/fix_links_115/analysis.md` (detailed categorization)
- **Tracking Issue**: #120 - Documentation Debt: Complete Missing User Guides and API Docs

---

**Last Updated**: PR #115
**Status**: All exclusions represent pre-existing debt, not new issues

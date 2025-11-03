# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import orjson
import yaml  # type: ignore[import-untyped]

__all__ = ("minimal_yaml",)


class MinimalDumper(yaml.SafeDumper):
    """YAML dumper with minimal, readable settings."""

    def ignore_aliases(self, data: Any) -> bool:  # type: ignore[override]
        """Disable anchors/aliases (&id001, *id001) for repeated objects."""
        return True


def _represent_str(dumper: yaml.SafeDumper, data: str):
    """Use block scalars for multiline text; plain style otherwise."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


MinimalDumper.add_representer(str, _represent_str)


def _is_empty(x: Any) -> bool:
    """Define 'empty' for pruning. Keeps 0 and False."""
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, dict):
        return len(x) == 0
    if isinstance(x, list | tuple | set):
        return len(x) == 0
    return False


def _prune(x: Any) -> Any:
    """Recursively remove empty leaves and empty containers."""
    if isinstance(x, dict):
        pruned = {k: _prune(v) for k, v in x.items() if not _is_empty(v)}
        return {k: v for k, v in pruned.items() if not _is_empty(v)}
    if isinstance(x, list):
        pruned_list = [_prune(v) for v in x if not _is_empty(v)]
        return [v for v in pruned_list if not _is_empty(v)]
    if isinstance(x, tuple):
        pruned_list = [_prune(v) for v in x if not _is_empty(v)]
        return tuple(v for v in pruned_list if not _is_empty(v))
    if isinstance(x, set):
        pruned_set = {_prune(v) for v in x if not _is_empty(v)}
        return {v for v in pruned_set if not _is_empty(v)}
    return x


def minimal_yaml(
    value: Any,
    *,
    drop_empties: bool = True,
    indent: int = 2,
    line_width: int = 2**31 - 1,
    sort_keys: bool = False,
) -> str:
    """Convert value to minimal YAML string."""
    # Auto-parse JSON strings for convenience (fails gracefully on invalid JSON)
    if isinstance(value, str):
        try:
            value = orjson.loads(value)
        except orjson.JSONDecodeError:
            # Not valid JSON - treat as plain string
            pass

    data = _prune(value) if drop_empties else value
    return yaml.dump(
        data,
        Dumper=MinimalDumper,
        default_flow_style=False,
        sort_keys=sort_keys,
        allow_unicode=True,
        indent=indent,
        width=line_width,
    )

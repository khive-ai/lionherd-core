# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import uuid
from datetime import UTC, datetime
from pathlib import Path as StdPath
from typing import Any

from anyio import Path as AsyncPath

__all__ = (
    "acreate_path",
    "get_bins",
    "import_module",
    "is_import_installed",
    "now_utc",
)


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


async def acreate_path(
    directory: StdPath | AsyncPath | str,
    filename: str,
    extension: str | None = None,
    timestamp: bool = False,
    dir_exist_ok: bool = True,
    file_exist_ok: bool = False,
    time_prefix: bool = False,
    timestamp_format: str | None = None,
    random_hash_digits: int = 0,
    timeout: float | None = None,
) -> AsyncPath:
    """Generate file path asynchronously with optional timeout.

    Args:
        directory: Base directory path
        filename: Target filename (may contain subdirectory with /)
        extension: File extension (if filename doesn't have one)
        timestamp: Add timestamp to filename
        dir_exist_ok: Allow existing directories
        file_exist_ok: Allow existing files
        time_prefix: Put timestamp before filename instead of after
        timestamp_format: Custom strftime format for timestamp
        random_hash_digits: Add random hash suffix (0 = disabled)
        timeout: Maximum time in seconds for async I/O operations (None = no timeout)

    Returns:
        AsyncPath to the created/validated file path

    Raises:
        ValueError: If filename contains backslash
        FileExistsError: If file exists and file_exist_ok is False
        TimeoutError: If timeout is exceeded
    """
    from lionherd_core.libs.concurrency import move_on_after

    async def _impl() -> AsyncPath:
        # Use AsyncPath for construction and execution
        nonlocal directory, filename

        if "/" in filename:
            sub_dir, filename = filename.split("/")[:-1], filename.split("/")[-1]
            directory = AsyncPath(directory) / "/".join(sub_dir)

        if "\\" in filename:
            raise ValueError("Filename cannot contain directory separators.")

        # Ensure directory is an AsyncPath
        directory = AsyncPath(directory)
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
        else:
            name = filename
            ext = extension or ""
        ext = f".{ext.lstrip('.')}" if ext else ""

        if timestamp:
            # datetime.now() is generally non-blocking
            ts_str = datetime.now().strftime(timestamp_format or "%Y%m%d%H%M%S")
            name = f"{ts_str}_{name}" if time_prefix else f"{name}_{ts_str}"

        if random_hash_digits > 0:
            random_suffix = uuid.uuid4().hex[:random_hash_digits]
            name = f"{name}-{random_suffix}"

        full_path = directory / f"{name}{ext}"

        # --- CRITICAL: ASYNC I/O Operations ---
        await full_path.parent.mkdir(parents=True, exist_ok=dir_exist_ok)

        if await full_path.exists() and not file_exist_ok:
            raise FileExistsError(f"File {full_path} already exists and file_exist_ok is False.")

        return full_path

    if timeout is None:
        return await _impl()

    with move_on_after(timeout) as cancel_scope:
        result = await _impl()
    if cancel_scope.cancelled_caught:
        raise TimeoutError(f"acreate_path timed out after {timeout}s")
    return result


def get_bins(input_: list[str], upper: int) -> list[list[int]]:
    """Organize indices into bins by cumulative length."""
    current = 0
    bins = []
    current_bin = []
    for idx, item in enumerate(input_):
        if current + len(item) < upper:
            current_bin.append(idx)
            current += len(item)
        else:
            bins.append(current_bin)
            current_bin = [idx]
            current = len(item)
    if current_bin:
        bins.append(current_bin)
    return bins


def import_module(
    package_name: str,
    module_name: str | None = None,
    import_name: str | list | None = None,
) -> Any:
    """Import module by path."""
    try:
        full_import_path = f"{package_name}.{module_name}" if module_name else package_name

        if import_name:
            import_name = [import_name] if not isinstance(import_name, list) else import_name
            a = __import__(
                full_import_path,
                fromlist=import_name,
            )
            if len(import_name) == 1:
                return getattr(a, import_name[0])
            return [getattr(a, name) for name in import_name]
        else:
            return __import__(full_import_path)

    except ImportError as e:
        raise ImportError(f"Failed to import module {full_import_path}: {e}") from e


def is_import_installed(package_name: str) -> bool:
    """Check if package is installed."""
    return importlib.util.find_spec(package_name) is not None

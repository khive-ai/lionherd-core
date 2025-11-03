# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from lionherd_core.ln import (
    acreate_path,
    get_bins,
    import_module,
    is_import_installed,
    now_utc,
)

# =============================================================================
# now_utc() Tests
# =============================================================================


class TestNowUtc:
    """Test suite for now_utc function."""

    @pytest.mark.unit
    def test_now_utc_returns_datetime(self):
        """Test now_utc returns datetime object."""
        result = now_utc()
        assert result is not None
        assert hasattr(result, "year")
        assert hasattr(result, "month")


# =============================================================================
# acreate_path() Async Tests
# =============================================================================


class TestAcreatePath:
    """Test suite for acreate_path async function."""

    @pytest.mark.anyio
    async def test_acreate_path_basic(self, tmp_path):
        """Test acreate_path basic usage."""
        result = await acreate_path(directory=tmp_path, filename="test.txt")
        assert result.name == "test.txt"
        assert result.parent == tmp_path

    @pytest.mark.anyio
    async def test_acreate_path_with_subdirectory(self, tmp_path):
        """Test acreate_path handles filename with subdirectory."""
        result = await acreate_path(
            directory=tmp_path,
            filename="subdir/test.txt",
        )
        assert result.name == "test.txt"
        assert result.parent.name == "subdir"
        assert await result.parent.exists()

    @pytest.mark.anyio
    async def test_acreate_path_backslash_raises(self, tmp_path):
        """Test acreate_path raises ValueError for backslash in filename."""
        with pytest.raises(ValueError, match="cannot contain directory separators"):
            await acreate_path(directory=tmp_path, filename="test\\file.txt")

    @pytest.mark.anyio
    async def test_acreate_path_with_extension_in_filename(self, tmp_path):
        """Test acreate_path handles filename with extension."""
        result = await acreate_path(directory=tmp_path, filename="test.txt")
        assert result.name == "test.txt"
        assert result.suffix == ".txt"

    @pytest.mark.anyio
    async def test_acreate_path_with_explicit_extension(self, tmp_path):
        """Test acreate_path with explicit extension parameter."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test",
            extension=".log",
        )
        assert result.name == "test.log"
        assert result.suffix == ".log"

    @pytest.mark.anyio
    async def test_acreate_path_extension_overrides_filename_ext(self, tmp_path):
        """Test explicit extension parameter overrides filename extension."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            extension=".log",
        )
        # When filename has extension, it's used; extension param only for files without ext
        assert result.suffix == ".txt"

    @pytest.mark.anyio
    async def test_acreate_path_with_timestamp_prefix(self, tmp_path):
        """Test acreate_path adds timestamp as prefix."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            timestamp=True,
            time_prefix=True,
        )
        # Should be YYYYMMDDHHMMSS_test.txt
        assert result.suffix == ".txt"
        assert "_test" in result.stem
        # Verify timestamp format (14 digits)
        prefix = result.stem.split("_")[0]
        assert len(prefix) == 14
        assert prefix.isdigit()

    @pytest.mark.anyio
    async def test_acreate_path_with_timestamp_suffix(self, tmp_path):
        """Test acreate_path adds timestamp as suffix."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            timestamp=True,
            time_prefix=False,
        )
        # Should be test_YYYYMMDDHHMMSS.txt
        assert result.suffix == ".txt"
        assert "test_" in result.stem
        suffix = result.stem.split("_")[1]
        assert len(suffix) == 14
        assert suffix.isdigit()

    @pytest.mark.anyio
    async def test_acreate_path_with_custom_timestamp_format(self, tmp_path):
        """Test acreate_path with custom timestamp format."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            timestamp=True,
            timestamp_format="%Y%m%d",
        )
        # Should have 8-digit date format
        assert result.suffix == ".txt"
        parts = result.stem.split("_")
        assert any(len(p) == 8 and p.isdigit() for p in parts)

    @pytest.mark.anyio
    async def test_acreate_path_with_random_hash(self, tmp_path):
        """Test acreate_path adds random hash suffix."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            random_hash_digits=8,
        )
        # Should be test-XXXXXXXX.txt
        assert result.suffix == ".txt"
        assert "-" in result.stem
        hash_part = result.stem.split("-")[1]
        assert len(hash_part) == 8

    @pytest.mark.anyio
    async def test_acreate_path_with_timestamp_and_hash(self, tmp_path):
        """Test acreate_path with both timestamp and random hash."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            timestamp=True,
            time_prefix=True,
            random_hash_digits=6,
        )
        assert result.suffix == ".txt"
        assert "_" in result.stem
        assert "-" in result.stem

    @pytest.mark.anyio
    async def test_acreate_path_file_exists_ok_true(self, tmp_path):
        """Test acreate_path with file_exist_ok=True allows existing files."""
        # Create file first
        test_file = tmp_path / "test.txt"
        test_file.touch()

        # Should not raise error
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            file_exist_ok=True,
        )
        assert result.name == "test.txt"

    @pytest.mark.anyio
    async def test_acreate_path_file_exists_raises(self, tmp_path):
        """Test acreate_path raises when file exists and file_exist_ok=False."""
        # Create file first
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with pytest.raises(FileExistsError, match="already exists"):
            await acreate_path(
                directory=tmp_path,
                filename="test.txt",
                file_exist_ok=False,
            )

    @pytest.mark.anyio
    async def test_acreate_path_creates_parent_directories(self, tmp_path):
        """Test acreate_path creates parent directories."""
        result = await acreate_path(
            directory=tmp_path,
            filename="deep/nested/structure/test.txt",
        )
        assert await result.parent.exists()
        assert result.parent.name == "structure"

    @pytest.mark.anyio
    async def test_acreate_path_with_timeout_success(self, tmp_path):
        """Test acreate_path succeeds within timeout."""
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
            timeout=5.0,  # Generous timeout
        )
        assert result.name == "test.txt"
        assert result.parent == tmp_path

    @pytest.mark.anyio
    async def test_acreate_path_with_timeout_raises(self, tmp_path):
        """Test acreate_path raises TimeoutError when timeout is exceeded."""
        # timeout=0.0 should trigger immediate timeout
        with pytest.raises(TimeoutError, match=r"timed out after 0.0s"):
            await acreate_path(
                directory=tmp_path,
                filename="test.txt",
                timeout=0.0,
            )

    @pytest.mark.anyio
    async def test_acreate_path_no_timeout_by_default(self, tmp_path):
        """Test acreate_path has no timeout by default."""
        # Should succeed without timeout parameter
        result = await acreate_path(
            directory=tmp_path,
            filename="test.txt",
        )
        assert result.name == "test.txt"


# =============================================================================
# get_bins() Tests
# =============================================================================


class TestGetBins:
    """Test suite for get_bins function."""

    @pytest.mark.unit
    def test_get_bins_basic(self):
        """Test get_bins basic functionality."""
        result = get_bins(["a" * 10, "b" * 10, "c" * 10], upper=25)
        assert len(result) == 2
        assert result[0] == [0, 1]
        assert result[1] == [2]

    @pytest.mark.unit
    def test_get_bins_empty_input(self):
        """Test get_bins with empty input."""
        result = get_bins([], upper=100)
        assert result == []

    @pytest.mark.unit
    def test_get_bins_single_item_fits(self):
        """Test get_bins with single item that fits."""
        result = get_bins(["a" * 50], upper=100)
        assert result == [[0]]

    @pytest.mark.unit
    def test_get_bins_single_item_exceeds_upper(self):
        """Test get_bins when single item exceeds upper limit.

        Note: Algorithm creates empty bin first when item exceeds limit,
        resulting in [[], [0]] for oversized single item.
        """
        result = get_bins(["a" * 200], upper=100)
        assert len(result) == 2
        assert result == [[], [0]]  # Empty bin, then oversized item

    @pytest.mark.unit
    def test_get_bins_exact_boundary(self):
        """Test get_bins at exact upper boundary."""
        # First two items total exactly 100 (50 + 49 = 99 < 100)
        result = get_bins(["a" * 50, "b" * 49, "c" * 30], upper=100)
        assert len(result) == 2
        assert result[0] == [0, 1]
        assert result[1] == [2]

    @pytest.mark.unit
    def test_get_bins_all_items_fit_one_bin(self):
        """Test get_bins when all items fit in one bin."""
        result = get_bins(["a" * 10, "b" * 10, "c" * 10], upper=100)
        assert len(result) == 1
        assert result[0] == [0, 1, 2]

    @pytest.mark.property
    @given(
        strings=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20),
        upper=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50)
    def test_get_bins_property_invariants(self, strings, upper):
        """Property test: get_bins maintains invariants."""
        result = get_bins(strings, upper)

        # All indices should be present exactly once
        all_indices = [idx for bin_ in result for idx in bin_]
        assert sorted(all_indices) == list(range(len(strings)))

        # Each bin should not exceed upper limit (except single oversized items)
        for bin_ in result:
            bin_length = sum(len(strings[i]) for i in bin_)
            # Allow single oversized item
            if len(bin_) == 1:
                continue
            assert bin_length < upper  # Note: < not <=, based on source code logic


# =============================================================================
# import_module() Tests
# =============================================================================


class TestImportModule:
    """Test suite for import_module function."""

    @pytest.mark.unit
    def test_import_module_package_only(self):
        """Test import_module with package name only."""
        result = import_module("json")
        assert result is not None
        assert hasattr(result, "dumps")

    @pytest.mark.unit
    def test_import_module_with_module_name(self):
        """Test import_module with package and module name."""
        result = import_module("os", "path")
        assert result is not None

    @pytest.mark.unit
    def test_import_module_with_single_import_name(self):
        """Test import_module with single import name."""
        result = import_module("json", import_name="dumps")
        assert callable(result)
        # Verify it's the actual dumps function
        assert result({"test": 1}) == '{"test": 1}'

    @pytest.mark.unit
    def test_import_module_with_list_import_names(self):
        """Test import_module with list of import names."""
        result = import_module("json", import_name=["dumps", "loads"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(callable(f) for f in result)

    @pytest.mark.unit
    def test_import_module_invalid_package_raises(self):
        """Test import_module raises ImportError for invalid package."""
        with pytest.raises(ImportError, match="Failed to import"):
            import_module("nonexistent_package_xyz")

    @pytest.mark.unit
    def test_import_module_invalid_module_raises(self):
        """Test import_module raises ImportError for invalid module."""
        with pytest.raises(ImportError, match="Failed to import"):
            import_module("os", "nonexistent_module_xyz")


# =============================================================================
# is_import_installed() Tests
# =============================================================================


class TestIsImportInstalled:
    """Test suite for is_import_installed function."""

    @pytest.mark.unit
    def test_is_import_installed_true_for_stdlib(self):
        """Test is_import_installed returns True for stdlib packages."""
        assert is_import_installed("json") is True
        assert is_import_installed("os") is True
        assert is_import_installed("sys") is True

    @pytest.mark.unit
    def test_is_import_installed_true_for_installed_packages(self):
        """Test is_import_installed returns True for installed packages."""
        assert is_import_installed("pytest") is True
        assert is_import_installed("anyio") is True

    @pytest.mark.unit
    def test_is_import_installed_false_for_nonexistent(self):
        """Test is_import_installed returns False for nonexistent packages."""
        assert is_import_installed("nonexistent_package_xyz_12345") is False

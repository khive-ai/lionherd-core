# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest


class TestSplitCancellation:
    """Test suite for split_cancellation() function."""

    @pytest.mark.anyio
    async def test_split_cancellation_with_mixed_exceptions(self):
        """Test split_cancellation() splits cancellation from non-cancellation exceptions (line 64)."""
        import anyio

        from lionherd_core.libs.concurrency._errors import split_cancellation

        # Create exception group with mix of cancel and non-cancel exceptions
        cancel_exc = anyio.get_cancelled_exc_class()()
        value_error = ValueError("test error")

        eg = BaseExceptionGroup("mixed", [cancel_exc, value_error])

        cancel_group, non_cancel_group = split_cancellation(eg)

        # Should split correctly
        assert cancel_group is not None
        assert non_cancel_group is not None
        assert len(cancel_group.exceptions) == 1
        assert len(non_cancel_group.exceptions) == 1
        assert isinstance(cancel_group.exceptions[0], anyio.get_cancelled_exc_class())
        assert isinstance(non_cancel_group.exceptions[0], ValueError)

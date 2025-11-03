# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import anyio
import pytest

from lionherd_core.libs.concurrency import (
    fail_after,
    get_cancelled_exc_class,
    is_cancelled,
    shield,
)


@pytest.mark.anyio
async def test_shield_propagates_inner_exception(anyio_backend):
    async def bad():
        await anyio.sleep(0)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await shield(bad)


@pytest.mark.anyio
async def test_shield_does_not_block_internal_timeout(anyio_backend):
    async def slow():
        with fail_after(0.01):
            await anyio.sleep(0.1)

    with pytest.raises(TimeoutError):
        await shield(slow)


@pytest.mark.anyio
async def test_is_cancelled_true_for_backend_exception(anyio_backend):
    caught = {}

    async def victim():
        try:
            await anyio.sleep(0.1)  # Further reduced for faster tests
        except BaseException as e:
            caught["e"] = e
            raise

    async with anyio.create_task_group() as tg:
        tg.start_soon(victim)
        await anyio.sleep(0.001)  # Small delay
        tg.cancel_scope.cancel()

    assert "e" in caught and is_cancelled(caught["e"])


@pytest.mark.anyio
async def test_shield_protects_from_external_cancellation(anyio_backend):
    """Test that shield protects an operation from external cancellation."""
    completed = False
    shield_worked = False

    async def protected_work():
        nonlocal completed
        await anyio.sleep(0.005)  # Reduced
        completed = True
        return "done"

    async def outer_work():
        nonlocal shield_worked
        try:
            # Shield protects the inner work
            result = await shield(protected_work)
            shield_worked = result == "done"
        except BaseException:
            # Should not get here if shield works
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(outer_work)
        await anyio.sleep(0.001)  # Let work start
        tg.cancel_scope.cancel()  # Cancel the group

    # Give shielded work time to complete
    await anyio.sleep(0.01)  # Reduced

    assert completed is True
    assert shield_worked is True


@pytest.mark.anyio
async def test_shield_still_allows_internal_cancellation(anyio_backend):
    """Test that shield doesn't prevent internal cancellation (e.g. timeouts)."""

    async def work_with_timeout():
        with fail_after(0.01):
            await anyio.sleep(1.0)
        return "should_not_reach"

    with pytest.raises(TimeoutError):
        await shield(work_with_timeout)

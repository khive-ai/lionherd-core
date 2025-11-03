# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import threading
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, ParamSpec, TypeVar

from lionherd_core.libs.concurrency import (
    Semaphore,
    create_task_group,
    get_cancelled_exc_class,
    is_coro_func,
    move_on_after,
    non_cancel_subgroup,
    run_sync,
    sleep,
)
from lionherd_core.types import ModelConfig, Params, Unset, not_sentinel

from ._to_list import to_list

T = TypeVar("T")
P = ParamSpec("P")

_INITIALIZED = False
_MODEL_LIKE = None
_INIT_LOCK = threading.RLock()


__all__ = (
    "AlcallParams",
    "BcallParams",
    "alcall",
    "bcall",
)


async def alcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    *,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    delay_before_start: float = 0,
    retry_initial_delay: float = 0,
    retry_backoff: float = 1,
    retry_default: Any = Unset,
    retry_timeout: float | None = None,
    retry_attempts: int = 0,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[T | BaseException]:
    """Apply function to each list element asynchronously with retry and concurrency control.

    Args:
        input_: List of items to process (or iterable that will be converted)
        func: Callable to apply (sync or async)
        input_flatten: Flatten nested input structures
        input_dropna: Remove None/undefined from input
        input_unique: Remove duplicate inputs (requires flatten)
        input_flatten_tuple_set: Include tuples/sets in flattening
        output_flatten: Flatten nested output structures
        output_dropna: Remove None/undefined from output
        output_unique: Remove duplicate outputs (requires flatten)
        output_flatten_tuple_set: Include tuples/sets in output flattening
        delay_before_start: Initial delay before processing (seconds)
        retry_initial_delay: Initial retry delay (seconds)
        retry_backoff: Backoff multiplier for retry delays
        retry_default: Default value on retry exhaustion (Unset = raise)
        retry_timeout: Timeout per function call (seconds)
        retry_attempts: Maximum retry attempts (0 = no retry)
        max_concurrent: Max concurrent executions (None = unlimited)
        throttle_period: Delay between starting tasks (seconds)
        return_exceptions: Return exceptions instead of raising
        **kwargs: Additional arguments passed to func

    Returns:
        List of results (preserves input order, may include exceptions if return_exceptions=True)

    Raises:
        ValueError: If func is not callable
        TimeoutError: If retry_timeout exceeded
        ExceptionGroup: If return_exceptions=False and tasks raise
    """

    global _INITIALIZED, _MODEL_LIKE
    if _INITIALIZED is False:
        with _INIT_LOCK:
            # Double-checked locking pattern
            if _INITIALIZED is False:
                from pydantic import BaseModel

                _MODEL_LIKE = (BaseModel,)
                _INITIALIZED = True

    # Validate func is a single callable
    if not callable(func):
        # If func is not callable, maybe it's an iterable. Extract one callable if possible.
        try:
            func_list = list(func)  # Convert iterable to list
        except TypeError:
            raise ValueError("func must be callable or an iterable containing one callable.")

        # Ensure exactly one callable is present
        if len(func_list) != 1 or not callable(func_list[0]):
            raise ValueError("Only one callable function is allowed.")

        func = func_list[0]

    # Process input if requested
    if any((input_flatten, input_dropna)):
        input_ = to_list(
            input_,
            flatten=input_flatten,
            dropna=input_dropna,
            unique=input_unique,
            flatten_tuple_set=input_flatten_tuple_set,
        )
    else:
        if not isinstance(input_, list):
            # Attempt to iterate
            if isinstance(input_, _MODEL_LIKE):
                # Pydantic model, convert to list
                input_ = [input_]
            else:
                try:
                    iter(input_)
                    # It's iterable (tuple), convert to list of its contents
                    input_ = list(input_)
                except TypeError:
                    # Not iterable, just wrap in a list
                    input_ = [input_]

    # Optional initial delay before processing
    if delay_before_start:
        await sleep(delay_before_start)

    semaphore = Semaphore(max_concurrent) if max_concurrent else None
    throttle_delay = throttle_period or 0
    coro_func = is_coro_func(func)

    async def call_func(item: Any) -> T:
        if coro_func:
            # Async function
            if retry_timeout is not None:
                with move_on_after(retry_timeout) as cancel_scope:
                    result = await func(item, **kwargs)
                if cancel_scope.cancelled_caught:
                    raise TimeoutError(f"Function call timed out after {retry_timeout}s")
                return result
            else:
                return await func(item, **kwargs)
        else:
            # Sync function
            if retry_timeout is not None:
                with move_on_after(retry_timeout) as cancel_scope:
                    result = await run_sync(func, item, **kwargs)
                if cancel_scope.cancelled_caught:
                    raise TimeoutError(f"Function call timed out after {retry_timeout}s")
                return result
            else:
                return await run_sync(func, item, **kwargs)

    async def execute_task(i: Any, index: int) -> Any:
        attempts = 0
        current_delay = retry_initial_delay
        while True:
            try:
                result = await call_func(i)
                return index, result

            # if cancelled, re-raise
            except get_cancelled_exc_class():
                raise

            # handle other exceptions
            except Exception:
                attempts += 1
                if attempts <= retry_attempts:
                    if current_delay:
                        await sleep(current_delay)
                        current_delay *= retry_backoff
                    # Retry loop continues
                else:
                    # Exhausted retries
                    if not_sentinel(retry_default):
                        return index, retry_default
                    # No default, re-raise
                    raise

    # Preallocate result list and fill by index — preserves order with no lock/sort
    n_items = len(input_)
    out: list[Any] = [None] * n_items

    async def task_wrapper(item: Any, idx: int) -> None:
        try:
            if semaphore:
                async with semaphore:
                    _, result = await execute_task(item, idx)
            else:
                _, result = await execute_task(item, idx)
            out[idx] = result
        except BaseException as exc:
            out[idx] = exc
            if not return_exceptions:
                raise  # Propagate to TaskGroup

    # Execute all tasks using task group
    try:
        async with create_task_group() as tg:
            for idx, item in enumerate(input_):
                tg.start_soon(task_wrapper, item, idx)
                # Apply throttle delay between starting tasks
                if throttle_delay and idx < n_items - 1:
                    await sleep(throttle_delay)
    except ExceptionGroup as eg:
        if not return_exceptions:
            # Surface only the non-cancellation subgroup to preserve structure & tracebacks
            rest = non_cancel_subgroup(eg)
            if rest is not None:
                raise rest
            raise

    output_list = out  # already in original order
    return to_list(
        output_list,
        flatten=output_flatten,
        dropna=output_dropna,
        unique=output_unique,
        flatten_tuple_set=output_flatten_tuple_set,
    )


async def bcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    batch_size: int,
    **kwargs: Any,
) -> AsyncGenerator[list[T | BaseException], None]:
    """Process input in batches using alcall. Yields results batch by batch.

    Args:
        input_: Items to process
        func: Callable to apply
        batch_size: Number of items per batch
        **kwargs: Arguments passed to alcall (see alcall for details)

    Yields:
        List of results for each batch
    """
    input_ = to_list(input_, flatten=True, dropna=True)

    for i in range(0, len(input_), batch_size):
        batch = input_[i : i + batch_size]
        yield await alcall(batch, func, **kwargs)


@dataclass(slots=True, init=False, frozen=True)
class AlcallParams(Params):
    # ClassVar attributes
    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
    _func: ClassVar[Any] = alcall

    # input processing
    input_flatten: bool
    input_dropna: bool
    input_unique: bool
    input_flatten_tuple_set: bool

    # output processing
    output_flatten: bool
    output_dropna: bool
    output_unique: bool
    output_flatten_tuple_set: bool

    # retry and timeout
    delay_before_start: float
    retry_initial_delay: float
    retry_backoff: float
    retry_default: Any
    retry_timeout: float
    retry_attempts: int

    # concurrency and throttling
    max_concurrent: int
    throttle_period: float

    kw: dict[str, Any] = Unset

    async def __call__(self, input_: list[Any], func: Callable[..., T], **kw: Any) -> list[T]:
        kwargs = {**self.default_kw(), **kw}
        return await alcall(input_, func, **kwargs)


@dataclass(slots=True, init=False, frozen=True)
class BcallParams(AlcallParams):
    _func: ClassVar[Any] = bcall

    batch_size: int

    async def __call__(self, input_: list[Any], func: Callable[..., T], **kw: Any) -> list[T]:
        kwargs = {**self.default_kw(), **kw}
        return await bcall(input_, func, self.batch_size, **kwargs)

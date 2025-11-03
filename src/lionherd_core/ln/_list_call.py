# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import Any, ParamSpec, TypeVar

from ._to_list import to_list

R = TypeVar("R")
T = TypeVar("T")
P = ParamSpec("P")

__all__ = ("lcall",)


def lcall(
    input_: Iterable[T] | T,
    func: Callable[[T], R] | Iterable[Callable[[T], R]],
    /,
    *args: Any,
    input_flatten: bool = False,
    input_dropna: bool = False,
    input_unique: bool = False,
    input_use_values: bool = False,
    input_flatten_tuple_set: bool = False,
    output_flatten: bool = False,
    output_dropna: bool = False,
    output_unique: bool = False,
    output_flatten_tuple_set: bool = False,
    **kwargs: Any,
) -> list[R]:
    """Apply function to each element synchronously with optional input/output processing.

    Args:
        input_: Items to process
        func: Callable to apply to each element
        *args: Positional arguments passed to func
        input_flatten: Flatten input structures
        input_dropna: Remove None/undefined from input
        input_unique: Remove duplicate inputs
        input_use_values: Extract values from enums/mappings
        input_flatten_tuple_set: Include tuples/sets in input flattening
        output_flatten: Flatten output structures
        output_dropna: Remove None/undefined from output
        output_unique: Remove duplicate outputs
        output_flatten_tuple_set: Include tuples/sets in output flattening
        **kwargs: Keyword arguments passed to func

    Returns:
        List of results

    Raises:
        ValueError: If func is not callable or output_unique without flatten/dropna
        TypeError: If func or input processing fails
    """
    # Validate and extract callable function
    if not callable(func):
        try:
            func_list = list(func)
            if len(func_list) != 1 or not callable(func_list[0]):
                raise ValueError("func must contain exactly one callable function.")
            func = func_list[0]
        except TypeError as e:
            raise ValueError("func must be callable or iterable with one callable.") from e

    # Validate output processing options
    if output_unique and not (output_flatten or output_dropna):
        raise ValueError("output_unique requires output_flatten or output_dropna.")

    # Process input based on sanitization flag
    if input_flatten or input_dropna:
        input_ = to_list(
            input_,
            flatten=input_flatten,
            dropna=input_dropna,
            unique=input_unique,
            flatten_tuple_set=input_flatten_tuple_set,
            use_values=input_use_values,
        )
    else:
        if not isinstance(input_, list):
            try:
                input_ = list(input_)
            except TypeError:
                input_ = [input_]

    # Process elements and collect results
    out = []
    append = out.append

    for item in input_:
        try:
            result = func(item, *args, **kwargs)
            append(result)
        except InterruptedError:
            return out
        except Exception:
            raise

    # Apply output processing if requested
    if output_flatten or output_dropna:
        out = to_list(
            out,
            flatten=output_flatten,
            dropna=output_dropna,
            unique=output_unique,
            flatten_tuple_set=output_flatten_tuple_set,
        )

    return out

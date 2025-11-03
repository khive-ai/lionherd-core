# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from typing import Any


def to_num(
    input_: Any,
    /,
    *,
    upper_bound: int | float | None = None,
    lower_bound: int | float | None = None,
    num_type: type[int] | type[float] = float,
    precision: int | None = None,
) -> int | float:
    """Convert input to numeric type with validation and bounds checking.

    Args:
        input_: Value to convert to number
        upper_bound: Maximum allowed value (inclusive)
        lower_bound: Minimum allowed value (inclusive)
        num_type: Target numeric type (int or float)
        precision: Number of decimal places for rounding (float only)

    Returns:
        Converted number

    Raises:
        ValueError: For invalid input or out of bounds values
        TypeError: For invalid input types
    """
    # Validate num_type
    if num_type not in (int, float):
        raise ValueError(f"Invalid number type: {num_type}")

    # Handle boolean (special case - must check before int)
    if isinstance(input_, (bool, int, float, Decimal)):
        value = float(input_)
    # Handle string input
    elif isinstance(input_, str):
        input_ = input_.strip()
        if not input_:
            raise ValueError("Empty string cannot be converted to number")
        try:
            value = float(input_)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{input_}' to number") from e
    else:
        raise TypeError(f"Cannot convert {type(input_).__name__} to number")

    # Apply bounds checking
    if upper_bound is not None and value > upper_bound:
        raise ValueError(f"Value {value} exceeds upper bound {upper_bound}")
    if lower_bound is not None and value < lower_bound:
        raise ValueError(f"Value {value} below lower bound {lower_bound}")

    # Apply precision for float
    if precision is not None and num_type is float:
        value = round(value, precision)

    # Convert to target type
    return num_type(value)

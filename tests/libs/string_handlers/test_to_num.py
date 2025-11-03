# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionherd_core.libs.string_handlers._to_num import to_num


class TestToNumBasic:
    """Test basic number conversion."""

    def test_int_to_float(self):
        """Test converting int to float."""
        assert to_num(42) == 42.0
        assert isinstance(to_num(42), float)

    def test_int_to_int(self):
        """Test converting int to int."""
        assert to_num(42, num_type=int) == 42
        assert isinstance(to_num(42, num_type=int), int)

    def test_float_to_float(self):
        """Test converting float to float."""
        assert to_num(3.14) == 3.14
        assert isinstance(to_num(3.14), float)

    def test_float_to_int(self):
        """Test converting float to int (truncates)."""
        assert to_num(3.14, num_type=int) == 3
        assert to_num(3.99, num_type=int) == 3
        assert isinstance(to_num(3.14, num_type=int), int)

    def test_string_to_float(self):
        """Test converting string to float."""
        assert to_num("42") == 42.0
        assert to_num("3.14") == 3.14
        assert to_num("-5.5") == -5.5

    def test_string_to_int(self):
        """Test converting string to int."""
        assert to_num("42", num_type=int) == 42
        assert to_num("3.14", num_type=int) == 3
        assert to_num("-5", num_type=int) == -5

    def test_string_with_whitespace(self):
        """Test strings with leading/trailing whitespace."""
        assert to_num("  42  ") == 42.0
        assert to_num("\t3.14\n") == 3.14

    def test_boolean_conversion(self):
        """Test boolean to number conversion."""
        assert to_num(True) == 1.0
        assert to_num(False) == 0.0
        assert to_num(True, num_type=int) == 1
        assert to_num(False, num_type=int) == 0


class TestToNumBounds:
    """Test bounds checking."""

    def test_upper_bound_valid(self):
        """Test values within upper bound."""
        assert to_num(50, upper_bound=100) == 50.0
        assert to_num(100, upper_bound=100) == 100.0

    def test_upper_bound_exceeded(self):
        """Test values exceeding upper bound."""
        with pytest.raises(ValueError, match="exceeds upper bound"):
            to_num(101, upper_bound=100)

    def test_lower_bound_valid(self):
        """Test values within lower bound."""
        assert to_num(50, lower_bound=10) == 50.0
        assert to_num(10, lower_bound=10) == 10.0

    def test_lower_bound_violated(self):
        """Test values below lower bound."""
        with pytest.raises(ValueError, match="below lower bound"):
            to_num(9, lower_bound=10)

    def test_both_bounds(self):
        """Test values with both upper and lower bounds."""
        assert to_num(50, lower_bound=10, upper_bound=100) == 50.0
        assert to_num(10, lower_bound=10, upper_bound=100) == 10.0
        assert to_num(100, lower_bound=10, upper_bound=100) == 100.0

        with pytest.raises(ValueError, match="below lower bound"):
            to_num(9, lower_bound=10, upper_bound=100)

        with pytest.raises(ValueError, match="exceeds upper bound"):
            to_num(101, lower_bound=10, upper_bound=100)


class TestToNumPrecision:
    """Test precision rounding."""

    def test_precision_float(self):
        """Test precision with float type."""
        assert to_num(3.14159, precision=2) == 3.14
        assert to_num(3.14159, precision=3) == 3.142
        assert to_num(3.5, precision=0) == 4.0  # Rounds to nearest

    def test_precision_ignored_for_int(self):
        """Test precision is ignored for int type."""
        # Precision is only applied to float type
        assert to_num(3.14159, num_type=int, precision=2) == 3
        assert to_num(3.99, num_type=int, precision=2) == 3

    def test_precision_with_bounds(self):
        """Test precision combined with bounds."""
        result = to_num(50.6789, lower_bound=10, upper_bound=100, precision=2)
        assert result == 50.68


class TestToNumErrors:
    """Test error handling."""

    def test_empty_string(self):
        """Test empty string raises error."""
        with pytest.raises(ValueError, match="Empty string"):
            to_num("")

        with pytest.raises(ValueError, match="Empty string"):
            to_num("   ")

    def test_invalid_string(self):
        """Test invalid string raises error."""
        with pytest.raises(ValueError, match="Cannot convert"):
            to_num("not a number")

        with pytest.raises(ValueError, match="Cannot convert"):
            to_num("abc123")

    def test_invalid_type(self):
        """Test invalid input type raises error."""
        with pytest.raises(TypeError, match="Cannot convert"):
            to_num([1, 2, 3])

        with pytest.raises(TypeError, match="Cannot convert"):
            to_num({"value": 42})

        with pytest.raises(TypeError, match="Cannot convert"):
            to_num(None)

    def test_invalid_num_type(self):
        """Test invalid num_type parameter."""
        with pytest.raises(ValueError, match="Invalid number type"):
            to_num(42, num_type=str)

        with pytest.raises(ValueError, match="Invalid number type"):
            to_num(42, num_type=complex)


class TestToNumDecimal:
    """Test Decimal type support."""

    def test_decimal_to_float(self):
        """Test converting Decimal to float."""
        from decimal import Decimal

        assert to_num(Decimal("3.14")) == 3.14
        assert isinstance(to_num(Decimal("3.14")), float)

    def test_decimal_to_int(self):
        """Test converting Decimal to int."""
        from decimal import Decimal

        assert to_num(Decimal("42"), num_type=int) == 42
        assert isinstance(to_num(Decimal("42"), num_type=int), int)

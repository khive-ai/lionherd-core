# to_num

> Convert strings and values to numbers with validation, bounds checking, and precision control

## Overview

`to_num()` is a robust utility function for converting various input types to numeric values (int or float) with built-in validation, bounds checking, and precision control. It provides safe type conversion with clear error messages for invalid inputs.

**Key Capabilities:**

- **Flexible Input**: Accepts strings, booleans, integers, floats, and Decimal objects
- **Type Conversion**: Convert to int or float with automatic type coercion
- **Bounds Validation**: Optional upper and lower bound checking (inclusive)
- **Precision Control**: Configurable decimal precision for float outputs
- **Safe Conversion**: Clear error messages for invalid inputs and out-of-bounds values

**When to Use to_num:**

- Converting user input strings to validated numeric values
- Parsing numeric data with known valid ranges (e.g., percentages 0-100)
- Ensuring numeric values meet business constraints before processing
- Standardizing numeric precision for consistent calculations
- Converting API responses or file data to validated numbers

**When NOT to Use to_num:**

- Simple type casting without validation (use `int()` or `float()` directly)
- Performance-critical tight loops (validation overhead)
- Data already guaranteed to be valid numeric types
- Complex parsing requiring custom error handling logic

## Function Signature

```python
from lionherd_core.libs.string_handlers import to_num

def to_num(
    input_: Any,
    /,
    *,
    upper_bound: int | float | None = None,
    lower_bound: int | float | None = None,
    num_type: type[int] | type[float] = float,
    precision: int | None = None,
) -> int | float: ...
```

## Parameters

**input_** : Any (positional-only)

Value to convert to numeric type. Supports strings, booleans, integers, floats, and Decimal objects.

- **Strings**: Whitespace is automatically stripped. Empty strings raise ValueError
- **Booleans**: Converted to 1 (True) or 0 (False)
- **Numeric types**: int, float, Decimal are converted directly
- **Invalid types**: Lists, dicts, None, and other non-numeric types raise TypeError

**upper_bound** : int or float, optional

Maximum allowed value (inclusive). If provided, values exceeding this bound raise ValueError.

- Validation performed after type conversion but before precision rounding
- Works with both int and float target types
- Default: None (no upper bound validation)

**lower_bound** : int or float, optional

Minimum allowed value (inclusive). If provided, values below this bound raise ValueError.

- Validation performed after type conversion but before precision rounding
- Works with both int and float target types
- Default: None (no lower bound validation)

**num_type** : type[int] or type[float], default float

Target numeric type for the conversion.

- Must be exactly `int` or `float` (not subclasses)
- Determines final return type
- Default: `float` (preserves decimal values)
- When `int`: decimal values are truncated (not rounded)

**precision** : int, optional

Number of decimal places for rounding (float only). Ignored when `num_type=int`.

- Applied after bounds validation
- Uses Python's `round()` for banker's rounding (round half to even)
- Default: None (no rounding)
- Example: `precision=2` converts 3.14159 to 3.14

## Returns

`int` or `float` - Converted and validated numeric value matching `num_type`.

- Type guaranteed to match `num_type` parameter
- Value guaranteed to be within bounds (if specified)
- Precision guaranteed to match specification (if float with precision)

## Raises

### ValueError

Raised in the following cases:

- Invalid `num_type` (not `int` or `float`)
- Empty string input (`input_=""` or `input_="   "`)
- String cannot be converted to number (e.g., `"abc"`, `"12.34.56"`)
- Value exceeds `upper_bound`
- Value below `lower_bound`

### TypeError

Raised when input type cannot be converted to number:

- Lists, tuples, dicts, sets
- None, custom objects without numeric conversion
- Non-numeric types without `__float__()` or `__int__()` methods

## Usage Patterns

### Basic Conversion

```python
from lionherd_core.libs.string_handlers import to_num

# String to float (default)
result = to_num("3.14")
# 3.14 (float)

# String to int
result = to_num("42", num_type=int)
# 42 (int)

# Boolean conversion
result = to_num(True)
# 1.0 (float)

result = to_num(False, num_type=int)
# 0 (int)
```

### Bounds Validation

```python
from lionherd_core.libs.string_handlers import to_num

# Percentage validation (0-100)
def validate_percentage(value: str) -> float:
    return to_num(value, lower_bound=0, upper_bound=100)

validate_percentage("50.5")  # 50.5 (valid)
# validate_percentage("150")  # ValueError: exceeds upper bound
# validate_percentage("-10")  # ValueError: below lower bound

# Age validation (0-150)
age = to_num("25", num_type=int, lower_bound=0, upper_bound=150)
# 25 (int)

# Temperature validation (Celsius: -273.15 to infinity)
temp = to_num("-40.5", lower_bound=-273.15)
# -40.5 (float)
```

### Precision Control

```python
from lionherd_core.libs.string_handlers import to_num

# Round to 2 decimal places (currency)
price = to_num("19.999", precision=2)
# 20.0 (banker's rounding)

# Round to 3 decimal places (scientific)
measurement = to_num("3.14159265", precision=3)
# 3.142 (float)

# Precision ignored for int
count = to_num("42.7", num_type=int, precision=2)
# 42 (int, precision ignored)
```

### Error Handling

```python
from lionherd_core.libs.string_handlers import to_num

# Handle invalid input
try:
    result = to_num("not_a_number")
except ValueError as e:
    print(f"Conversion failed: {e}")
    # "Cannot convert 'not_a_number' to number"

# Handle type errors
try:
    result = to_num([1, 2, 3])
except TypeError as e:
    print(f"Invalid type: {e}")
    # "Cannot convert list to number"

# Handle bounds violations
try:
    result = to_num("150", upper_bound=100)
except ValueError as e:
    print(f"Out of bounds: {e}")
    # "Value 150.0 exceeds upper bound 100"
```

### Combined Validation

```python
from lionherd_core.libs.string_handlers import to_num

# Complex validation: score (0-100, 1 decimal place)
def validate_score(input_value: str) -> float:
    return to_num(
        input_value,
        num_type=float,
        lower_bound=0,
        upper_bound=100,
        precision=1
    )

validate_score("95.67")  # 95.7 (rounded to 1 decimal)
validate_score("100")    # 100.0 (valid)
# validate_score("100.1")  # ValueError: exceeds upper bound

# API response parsing with validation
def parse_confidence(response: dict) -> float:
    """Parse confidence score from API response (0.0-1.0, 4 decimals)."""
    raw_value = response.get("confidence", "0")
    return to_num(
        raw_value,
        lower_bound=0.0,
        upper_bound=1.0,
        precision=4
    )
```

## Design Rationale

### Why Positional-Only Input?

The first parameter `input_` is positional-only (using `/` syntax) to:

1. **Prevent naming conflicts**: Avoids collision with keyword arguments
2. **Enforce clear API**: Input is always first argument, never named
3. **Allow generic naming**: `input_` can be any type without confusion

### Why Default to Float?

The `num_type` parameter defaults to `float` rather than `int` because:

1. **Information preservation**: Floats preserve decimal values; converting to int loses precision
2. **API compatibility**: Most APIs return decimal numbers (confidence scores, percentages)
3. **User expectations**: Safer default for string inputs like "3.14" (explicit `num_type=int` required for truncation)

### Why Bounds Before Precision?

Bounds validation occurs before precision rounding to ensure:

1. **Logical ordering**: Validate raw converted value first
2. **Prevent edge cases**: Rounding might bring out-of-bounds values into range (unintended)
3. **Clear semantics**: Bounds apply to input value, not rounded result

Example:

```python
# With current order (bounds before precision)
to_num("100.6", upper_bound=100, precision=0)  # ValueError (100.6 > 100)

# If precision came first (would round to 101, then fail bounds)
# More confusing error message
```

### Why Banker's Rounding?

Uses Python's built-in `round()` which implements banker's rounding (round half to even):

1. **Statistical fairness**: Reduces cumulative rounding bias over many operations
2. **Standard behavior**: Matches Python's rounding semantics
3. **Predictable**: Consistent with IEEE 754 floating-point standard

### Why Separate Validation Function?

`to_num()` is designed as a focused utility rather than a class because:

1. **Single responsibility**: One function, one clear purpose
2. **No state**: Pure function with no internal state to manage
3. **Composability**: Easy to combine with other validation functions
4. **Performance**: No object instantiation overhead

## See Also

- **Related Functions**:
  - `to_str()`: Convert values to string with validation
  - `to_list()`: Convert values to list with validation
  - `strip_lower()`: String normalization utilities
- **Related Modules**:
  - `decimal.Decimal`: High-precision decimal arithmetic
  - Python built-ins: `int()`, `float()`, `round()`

## Examples

### Example 1: User Input Validation

```python
from lionherd_core.libs.string_handlers import to_num

def get_age_from_user() -> int:
    """Get and validate user age input."""
    while True:
        user_input = input("Enter your age: ")
        try:
            age = to_num(
                user_input,
                num_type=int,
                lower_bound=0,
                upper_bound=150
            )
            return age
        except (ValueError, TypeError) as e:
            print(f"Invalid age: {e}. Please try again.")

# Usage
# age = get_age_from_user()
# Enter your age: 25
# Returns: 25
```

### Example 2: API Response Parsing

```python
from lionherd_core.libs.string_handlers import to_num

def parse_llm_response(response: dict) -> dict:
    """Parse and validate LLM API response."""
    return {
        "confidence": to_num(
            response["confidence"],
            lower_bound=0.0,
            upper_bound=1.0,
            precision=4
        ),
        "temperature": to_num(
            response["temperature"],
            lower_bound=0.0,
            upper_bound=2.0,
            precision=2
        ),
        "max_tokens": to_num(
            response["max_tokens"],
            num_type=int,
            lower_bound=1,
            upper_bound=32000
        ),
    }

# Usage
response_data = {
    "confidence": "0.87654",
    "temperature": "0.7",
    "max_tokens": "2000"
}
parsed = parse_llm_response(response_data)
# {
#     "confidence": 0.8765,
#     "temperature": 0.7,
#     "max_tokens": 2000
# }
```

### Example 3: Configuration Validation

```python
from lionherd_core.libs.string_handlers import to_num

class ServerConfig:
    """Server configuration with validated numeric fields."""

    def __init__(self, config: dict):
        self.port = to_num(
            config.get("port", "8080"),
            num_type=int,
            lower_bound=1,
            upper_bound=65535
        )
        self.timeout = to_num(
            config.get("timeout", "30.0"),
            lower_bound=0.1,
            upper_bound=300.0,
            precision=1
        )
        self.max_connections = to_num(
            config.get("max_connections", "100"),
            num_type=int,
            lower_bound=1,
            upper_bound=10000
        )

# Usage
config = ServerConfig({
    "port": "3000",
    "timeout": "60.5",
    "max_connections": "500"
})
# config.port = 3000 (int)
# config.timeout = 60.5 (float)
# config.max_connections = 500 (int)
```

### Example 4: Batch Data Processing

```python
from lionherd_core.libs.string_handlers import to_num

def process_survey_scores(responses: list[str]) -> dict:
    """Process survey scores (1-5 scale) and calculate statistics."""
    scores = []
    errors = []

    for idx, response in enumerate(responses):
        try:
            score = to_num(
                response,
                num_type=int,
                lower_bound=1,
                upper_bound=5
            )
            scores.append(score)
        except (ValueError, TypeError) as e:
            errors.append({"index": idx, "value": response, "error": str(e)})

    if scores:
        avg = sum(scores) / len(scores)
        avg_rounded = to_num(avg, precision=2)
    else:
        avg_rounded = 0.0

    return {
        "valid_count": len(scores),
        "error_count": len(errors),
        "average": avg_rounded,
        "errors": errors
    }

# Usage
responses = ["5", "4", "3", "5", "invalid", "4", "6", "2"]
result = process_survey_scores(responses)
# {
#     "valid_count": 6,
#     "error_count": 2,
#     "average": 3.83,
#     "errors": [
#         {"index": 4, "value": "invalid", "error": "Cannot convert 'invalid' to number"},
#         {"index": 6, "value": "6", "error": "Value 6 exceeds upper bound 5"}
#     ]
# }
```

### Example 5: Financial Calculations

```python
from lionherd_core.libs.string_handlers import to_num

def calculate_tax(price_str: str, tax_rate_str: str) -> dict:
    """Calculate tax and total with precision control."""
    # Validate inputs
    price = to_num(
        price_str,
        lower_bound=0,
        precision=2  # Currency precision
    )
    tax_rate = to_num(
        tax_rate_str,
        lower_bound=0,
        upper_bound=100,  # Percentage
        precision=2
    )

    # Calculate tax and total
    tax_decimal = tax_rate / 100
    tax_amount = to_num(price * tax_decimal, precision=2)
    total = to_num(price + tax_amount, precision=2)

    return {
        "subtotal": price,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "total": total
    }

# Usage
result = calculate_tax("99.99", "8.875")
# {
#     "subtotal": 99.99,
#     "tax_rate": 8.88,      # Rounded from 8.875
#     "tax_amount": 8.88,    # 99.99 * 0.0888
#     "total": 108.87        # 99.99 + 8.88
# }
```

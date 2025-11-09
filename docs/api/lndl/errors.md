# LNDL Errors

> Exception hierarchy for LNDL parsing, validation, and extraction errors

## Overview

The `errors` module defines the **exception hierarchy** for LNDL (Language InterOperable Network Directive Language) parsing and validation failures. These exceptions are raised during structured output extraction when LLM responses fail to match expected specifications or contain invalid constructor syntax.

**Key Capabilities:**

- **Structured Error Reporting**: Specific exception types for each failure mode
- **Hierarchical Organization**: All LNDL errors inherit from `LNDLError` for easy catching
- **Clear Failure Semantics**: Exception names and messages clearly indicate what went wrong
- **Validation Integration**: Exceptions raised during strict/lenient/fuzzy validation strategies

**When to Use:**

- **Parsing LLM Responses**: Catch LNDL errors when extracting structured data from completions
- **Validation Pipelines**: Handle different error types with different retry/fallback strategies
- **Error Reporting**: Provide specific feedback about what's wrong with LLM output
- **Testing**: Verify LNDL parsers correctly detect malformed responses

**When NOT to Use:**

- **Pydantic Validation Errors**: Use Pydantic's built-in exceptions for data validation within parsed objects
- **Generic Parsing**: For non-LNDL parsing errors, use standard Python exceptions
- **Business Logic Errors**: LNDL errors are parsing-specific, not for application-level failures

## Module Exports

```python
from lionherd_core.lndl.errors import (
    LNDLError,
    MissingLvarError,
    MissingFieldError,
    TypeMismatchError,
    InvalidConstructorError,
    MissingOutBlockError,
    AmbiguousMatchError,
)
```

## Exceptions

### `LNDLError`

Base exception for all LNDL parsing and validation errors.

**Inheritance:**

```python
class LNDLError(Exception): ...
```

**Purpose:**

Serves as the **root exception class** for the LNDL error hierarchy. Catch this to handle any LNDL-related failure without distinguishing specific error types.

**When Raised:**

Never raised directly. All specific LNDL errors inherit from this class.

**Usage:**

```python
from lionherd_core.lndl.errors import LNDLError

try:
    result = await parse_lndl_response(llm_output, spec=UserSpec)
except LNDLError as e:
    # Catch any LNDL parsing/validation error
    logger.error(f"LNDL parsing failed: {e}")
    # Retry with different prompt or fallback to lenient validation
```

**See Also:**

- Specific subclasses for targeted error handling

---

### `MissingLvarError`

Referenced lvar (local variable) does not exist in the LNDL context.

**Inheritance:**

```python
class MissingLvarError(LNDLError): ...
```

**When Raised:**

During LNDL parsing when:

- A constructor references an `@lvar` that wasn't defined earlier
- Variable scoping issues cause a variable to be out of scope
- Typos in variable names (e.g., `@usr` instead of `@user`)

**Example Scenarios:**

```python
# LLM Output with missing lvar
"""
@user = User{name="Alice"}
@profile = Profile{user=@usr}  # Typo: @usr doesn't exist, should be @user
"""

# Raises: MissingLvarError: Referenced lvar '@usr' does not exist
```

**Handling:**

```python
from lionherd_core.lndl.errors import MissingLvarError

try:
    result = parse_lndl(llm_output)
except MissingLvarError as e:
    # Variable reference error - likely LLM hallucination or typo
    logger.warning(f"LLM used undefined variable: {e}")
    # Retry with explicit variable listing in prompt
```

---

### `MissingFieldError`

Required Spec field is missing from the OUT{} block.

**Inheritance:**

```python
class MissingFieldError(LNDLError): ...
```

**When Raised:**

During LNDL validation when:

- A required field in the target Spec is not present in the `OUT{}` block
- Field name mismatch prevents mapping (even with fuzzy matching)
- LLM omitted a mandatory field from the response

**Example Scenarios:**

```python
# Target Spec
class UserSpec(BaseModel):
    name: str
    email: str  # Required field

# LLM Output missing required field
"""
OUT{
    name = "Alice"
    # Missing 'email' field
}
"""

# Raises: MissingFieldError: Required field 'email' not found in OUT{} block
```

**Handling:**

```python
from lionherd_core.lndl.errors import MissingFieldError

try:
    result = parse_lndl(llm_output, spec=UserSpec)
except MissingFieldError as e:
    # Required field missing - prompt wasn't clear enough
    logger.error(f"LLM omitted required field: {e}")
    # Retry with more explicit field requirements in prompt
```

---

### `TypeMismatchError`

Constructor class doesn't match the expected Spec type.

**Inheritance:**

```python
class TypeMismatchError(LNDLError): ...
```

**When Raised:**

During LNDL parsing when:

- Constructor type name doesn't match the expected Spec class
- LLM used wrong type name (e.g., `Person{}` instead of `User{}`)
- Type inference failed due to ambiguous constructor syntax

**Example Scenarios:**

```python
# Target Spec
class User(BaseModel):
    name: str

# LLM Output with wrong type
"""
OUT{
    result = Person{name="Alice"}  # Should be User{}, not Person{}
}
"""

# Raises: TypeMismatchError: Constructor 'Person' doesn't match expected type 'User'
```

**Handling:**

```python
from lionherd_core.lndl.errors import TypeMismatchError

try:
    result = parse_lndl(llm_output, spec=User)
except TypeMismatchError as e:
    # Type name mismatch - LLM used wrong class name
    logger.warning(f"Type mismatch: {e}")
    # Retry with explicit type name in prompt, or use lenient mode
```

---

### `InvalidConstructorError`

Cannot parse constructor syntax - malformed or invalid structure.

**Inheritance:**

```python
class InvalidConstructorError(LNDLError): ...
```

**When Raised:**

During LNDL parsing when:

- Constructor syntax is malformed (missing braces, invalid nesting)
- Unsupported syntax used (e.g., arbitrary Python expressions)
- Syntax errors prevent parsing (unclosed quotes, unmatched brackets)

**Example Scenarios:**

```python
# Invalid constructor syntax
"""
OUT{
    user = User{name="Alice", email=  # Missing closing quote and brace
}
"""

# Raises: InvalidConstructorError: Cannot parse constructor syntax

# Another example - invalid nesting
"""
OUT{
    user = User{profile={name="Alice"}}  # Dict literal instead of constructor
}
"""

# Raises: InvalidConstructorError: Invalid constructor structure
```

**Handling:**

```python
from lionherd_core.lndl.errors import InvalidConstructorError

try:
    result = parse_lndl(llm_output)
except InvalidConstructorError as e:
    # Syntax error in LLM output
    logger.error(f"Invalid LNDL syntax: {e}")
    # Retry with syntax examples in prompt, or fall back to JSON
```

---

### `MissingOutBlockError`

No OUT{} block found in the LLM response.

**Inheritance:**

```python
class MissingOutBlockError(LNDLError): ...
```

**When Raised:**

During LNDL extraction when:

- LLM response doesn't contain an `OUT{}` block
- OUT block is commented out or inside a code fence
- LLM misunderstood the instruction and returned plain text

**Example Scenarios:**

```python
# LLM Output without OUT{} block
"""
Based on the analysis, the user's name is Alice and email is alice@example.com.
I recommend creating a profile with these details.
"""

# Raises: MissingOutBlockError: No OUT{} block found in response
```

**Handling:**

```python
from lionherd_core.lndl.errors import MissingOutBlockError

try:
    result = parse_lndl(llm_output)
except MissingOutBlockError as e:
    # LLM didn't follow LNDL format
    logger.warning(f"No OUT block in response: {e}")
    # Retry with stronger prompt, or try JSON extraction
```

---

### `AmbiguousMatchError`

Multiple fields match with similar similarity scores during fuzzy validation.

**Inheritance:**

```python
class AmbiguousMatchError(LNDLError): ...
```

**When Raised:**

During fuzzy field matching when:

- Two or more candidate fields have similarity scores within threshold (tie)
- Cannot confidently determine which field the LLM intended
- Ambiguous field names in the response (e.g., "user_name" vs "username")

**Example Scenarios:**

```python
# Target Spec
class UserSpec(BaseModel):
    user_name: str
    username: str  # Both fields exist

# LLM Output with ambiguous field
"""
OUT{
    userName = "Alice"  # Fuzzy matches both 'user_name' and 'username' equally
}
"""

# Raises: AmbiguousMatchError: Multiple fields match 'userName' with similar scores
```

**Handling:**

```python
from lionherd_core.lndl.errors import AmbiguousMatchError

try:
    result = parse_lndl(llm_output, validation_mode="fuzzy")
except AmbiguousMatchError as e:
    # Field name ambiguity - fuzzy matching failed
    logger.error(f"Ambiguous field match: {e}")
    # Fall back to strict mode, or retry with explicit field names
```

---

## Usage Patterns

### Pattern 1: Graceful Degradation by Error Type

```python
from lionherd_core.lndl.errors import (
    LNDLError,
    MissingFieldError,
    TypeMismatchError,
    InvalidConstructorError,
    MissingOutBlockError,
)

async def extract_with_fallback(llm_output: str, spec: type[BaseModel]):
    """Try LNDL parsing with type-specific fallback strategies."""
    try:
        # Attempt strict LNDL parsing
        return parse_lndl(llm_output, spec=spec, mode="strict")

    except MissingOutBlockError:
        # No OUT block - try JSON extraction instead
        logger.info("No OUT block, falling back to JSON extraction")
        return extract_json(llm_output, spec=spec)

    except TypeMismatchError:
        # Type name mismatch - retry with lenient mode
        logger.warning("Type mismatch, retrying with lenient mode")
        return parse_lndl(llm_output, spec=spec, mode="lenient")

    except (MissingFieldError, InvalidConstructorError):
        # Structural issues - retry LLM call with better prompt
        logger.error("Structural parsing error, retrying with explicit schema")
        return await retry_with_schema_prompt(spec)

    except LNDLError as e:
        # Other LNDL errors - log and raise
        logger.exception(f"Unhandled LNDL error: {e}")
        raise
```

### Pattern 2: Validation Strategy Selection

```python
from lionherd_core.lndl.errors import MissingFieldError, AmbiguousMatchError

async def adaptive_parsing(llm_output: str, spec: type[BaseModel]):
    """Start strict, fall back to lenient, then fuzzy."""

    # Try strict validation first
    try:
        return parse_lndl(llm_output, spec=spec, mode="strict")
    except MissingFieldError:
        logger.info("Strict validation failed, trying lenient")

    # Fall back to lenient (allows type mismatches)
    try:
        return parse_lndl(llm_output, spec=spec, mode="lenient")
    except MissingFieldError:
        logger.info("Lenient validation failed, trying fuzzy")

    # Last resort: fuzzy field matching
    try:
        return parse_lndl(llm_output, spec=spec, mode="fuzzy")
    except AmbiguousMatchError as e:
        # Fuzzy matching found ties - can't resolve
        logger.error(f"Fuzzy matching ambiguous: {e}")
        raise
```

### Pattern 3: Error Context for LLM Feedback

```python
from lionherd_core.lndl.errors import (
    LNDLError,
    MissingFieldError,
    TypeMismatchError,
    InvalidConstructorError,
)

def create_retry_prompt(original_prompt: str, error: LNDLError, spec: type[BaseModel]) -> str:
    """Generate improved prompt based on specific error type."""

    base_prompt = f"{original_prompt}\n\nIMPORTANT CORRECTIONS:\n"

    if isinstance(error, MissingFieldError):
        required_fields = [f.name for f in spec.__fields__.values() if f.is_required()]
        return base_prompt + f"You MUST include these fields: {', '.join(required_fields)}"

    elif isinstance(error, TypeMismatchError):
        return base_prompt + f"Use constructor type '{spec.__name__}{{}}', not other types"

    elif isinstance(error, InvalidConstructorError):
        return base_prompt + """
Constructor syntax must be:
OUT{{
    field = Type{{key=value, key2=value2}}
}}
"""

    else:
        return base_prompt + "Follow the exact LNDL format with OUT{} block"

# Usage
try:
    result = await llm_call(prompt, spec=UserSpec)
except LNDLError as e:
    # Retry with error-specific guidance
    retry_prompt = create_retry_prompt(prompt, e, UserSpec)
    result = await llm_call(retry_prompt, spec=UserSpec)
```

### Pattern 4: Error Metrics and Monitoring

```python
from collections import Counter
from lionherd_core.lndl.errors import LNDLError

class LNDLMetrics:
    """Track LNDL parsing error patterns for monitoring."""

    def __init__(self):
        self.error_counts = Counter()
        self.total_parses = 0

    def record_success(self):
        self.total_parses += 1

    def record_error(self, error: LNDLError):
        self.total_parses += 1
        error_type = type(error).__name__
        self.error_counts[error_type] += 1

    def get_error_rate(self) -> float:
        if self.total_parses == 0:
            return 0.0
        total_errors = sum(self.error_counts.values())
        return total_errors / self.total_parses

    def get_report(self) -> dict:
        return {
            "total_parses": self.total_parses,
            "error_rate": self.get_error_rate(),
            "errors_by_type": dict(self.error_counts),
            "most_common_error": self.error_counts.most_common(1)[0] if self.error_counts else None,
        }

# Usage
metrics = LNDLMetrics()

try:
    result = parse_lndl(llm_output, spec=UserSpec)
    metrics.record_success()
except LNDLError as e:
    metrics.record_error(e)
    # Handle error...

# Periodic reporting
if metrics.total_parses % 100 == 0:
    report = metrics.get_report()
    logger.info(f"LNDL Metrics: {report}")
```

## Common Pitfalls

### Pitfall 1: Catching Too Broadly

**Issue**: Catching `Exception` instead of `LNDLError` hides the error source.

```python
# ❌ BAD: Catches everything, including system errors
try:
    result = parse_lndl(llm_output)
except Exception as e:
    logger.error(f"Something went wrong: {e}")
    # Is this a parsing error, network error, or system error?
```

**Solution**: Catch `LNDLError` specifically to distinguish parsing failures.

```python
# ✅ GOOD: Distinguish LNDL errors from other failures
try:
    result = parse_lndl(llm_output)
except LNDLError as e:
    # Known parsing error - can retry with different strategy
    logger.warning(f"LNDL parsing failed: {e}")
    handle_parsing_error(e)
except Exception as e:
    # System error - different handling
    logger.exception(f"Unexpected error: {e}")
    raise
```

### Pitfall 2: Ignoring Error Context

**Issue**: Not using error information to improve retry attempts.

```python
# ❌ BAD: Retry with same prompt regardless of error
try:
    result = parse_lndl(llm_output)
except LNDLError:
    result = parse_lndl(llm_output)  # Same error will repeat
```

**Solution**: Use error type to adjust retry strategy.

```python
# ✅ GOOD: Adapt strategy based on error type
try:
    result = parse_lndl(llm_output, mode="strict")
except TypeMismatchError:
    # Type issues - use lenient mode
    result = parse_lndl(llm_output, mode="lenient")
except MissingFieldError:
    # Field issues - retry with fuzzy matching
    result = parse_lndl(llm_output, mode="fuzzy")
```

### Pitfall 3: No Fallback Strategy

**Issue**: Raising errors without fallback leaves system fragile.

```python
# ❌ BAD: No fallback - single point of failure
def extract_data(llm_output: str):
    return parse_lndl(llm_output)  # Raises on any LNDL error
```

**Solution**: Implement fallback extraction methods.

```python
# ✅ GOOD: Multiple extraction strategies
def extract_data(llm_output: str):
    try:
        # Prefer LNDL structured extraction
        return parse_lndl(llm_output)
    except MissingOutBlockError:
        # Fall back to JSON extraction
        return extract_json(llm_output)
    except LNDLError as e:
        # Last resort: keyword extraction
        logger.warning(f"Structured extraction failed: {e}")
        return extract_keywords(llm_output)
```

### Pitfall 4: Not Logging Error Details

**Issue**: Generic error messages don't help debug LLM behavior.

```python
# ❌ BAD: Loses error context
try:
    result = parse_lndl(llm_output)
except LNDLError:
    logger.error("Parsing failed")  # What went wrong?
```

**Solution**: Log error type, message, and context.

```python
# ✅ GOOD: Rich error logging
try:
    result = parse_lndl(llm_output, spec=UserSpec)
except LNDLError as e:
    logger.error(
        f"LNDL parsing failed: {type(e).__name__}: {e}",
        extra={
            "error_type": type(e).__name__,
            "spec": UserSpec.__name__,
            "llm_output_preview": llm_output[:200],
        }
    )
```

## Design Rationale

### Why Specific Exception Types?

LNDL parsing has **distinct failure modes** with different remediation strategies:

1. **MissingOutBlockError**: LLM didn't follow format → Retry with stronger prompt
2. **TypeMismatchError**: Wrong type name → Use lenient validation or fix prompt
3. **MissingFieldError**: Omitted fields → Retry with explicit field list
4. **InvalidConstructorError**: Syntax errors → Provide syntax examples
5. **AmbiguousMatchError**: Fuzzy matching ties → Fall back to strict mode

Specific exceptions enable **targeted error handling** - different errors require different fixes. Generic exceptions would force one-size-fits-all handling.

### Why Inherit from Common Base?

`LNDLError` as the base class enables:

1. **Catch-all handling**: `except LNDLError` catches any LNDL-specific error
2. **Clear error boundaries**: Distinguish LNDL errors from Pydantic, network, or system errors
3. **Type safety**: Static analysis can verify LNDL error handling
4. **Future extensibility**: New error types automatically work with existing catch blocks

### Why Not Use Pydantic ValidationError?

Pydantic's `ValidationError` is for **data validation** (wrong types, invalid values), while LNDL errors are for **syntax and structure** (missing blocks, malformed constructors). Separating concerns:

- **LNDL errors**: Parsing and structure issues (before Pydantic)
- **Pydantic errors**: Data validation issues (after successful parsing)

This separation enables different handling strategies at different pipeline stages.

## See Also

- **Related Modules**:
  - [Spec](../types/spec.md): Spec definition and validation
  - [Operable](../types/operable.md): Structured output integration
  - [LNDL Parser](parser.md): Parser API documentation
  - [LNDL Resolver](resolver.md): Resolver and validation
  - [LNDL Fuzzy](fuzzy.md): Fuzzy matching for error tolerance

## Examples

### Example 1: Production Error Handling

```python
from lionherd_core.lndl.errors import (
    LNDLError,
    MissingOutBlockError,
    MissingFieldError,
    TypeMismatchError,
    InvalidConstructorError,
    AmbiguousMatchError,
)
import logging

logger = logging.getLogger(__name__)

async def robust_lndl_extraction(
    llm_output: str,
    spec: type[BaseModel],
    max_retries: int = 3
) -> BaseModel:
    """Production-grade LNDL extraction with comprehensive error handling."""

    # Strategy progression: strict → lenient → fuzzy → JSON fallback
    strategies = ["strict", "lenient", "fuzzy"]

    for attempt in range(max_retries):
        for strategy in strategies:
            try:
                result = parse_lndl(llm_output, spec=spec, mode=strategy)
                logger.info(f"LNDL extraction succeeded with {strategy} mode")
                return result

            except MissingOutBlockError:
                # No OUT block - skip to JSON fallback immediately
                logger.warning("No OUT{} block found, trying JSON extraction")
                break

            except AmbiguousMatchError as e:
                # Fuzzy matching failed - can't resolve
                if strategy == "fuzzy":
                    logger.error(f"Fuzzy matching ambiguous: {e}")
                    raise
                else:
                    # Try next strategy
                    continue

            except (MissingFieldError, TypeMismatchError, InvalidConstructorError) as e:
                # Structural issues - try next strategy
                logger.debug(f"{strategy.capitalize()} mode failed: {e}")
                continue

            except LNDLError as e:
                # Other LNDL errors
                logger.warning(f"Unexpected LNDL error in {strategy} mode: {e}")
                continue

        # All LNDL strategies failed - try JSON extraction
        try:
            result = extract_json(llm_output, spec=spec)
            logger.info("JSON extraction succeeded after LNDL failure")
            return result
        except Exception as e:
            logger.error(f"JSON extraction also failed: {e}")

        # All extraction methods failed
        if attempt < max_retries - 1:
            logger.warning(f"Retry {attempt + 1}/{max_retries}: All extraction failed")
            # Optionally: retry LLM call with improved prompt
        else:
            logger.error(f"All extraction attempts exhausted ({max_retries} retries)")
            raise ValueError(f"Failed to extract {spec.__name__} after {max_retries} attempts")
```

### Example 2: Error-Driven Prompt Engineering

```python
from lionherd_core.lndl.errors import (
    LNDLError,
    MissingFieldError,
    TypeMismatchError,
    InvalidConstructorError,
    MissingOutBlockError,
)

class AdaptivePromptBuilder:
    """Build prompts that adapt based on previous errors."""

    def __init__(self, base_prompt: str, spec: type[BaseModel]):
        self.base_prompt = base_prompt
        self.spec = spec
        self.error_history: list[LNDLError] = []

    def add_error(self, error: LNDLError):
        """Record error to improve future prompts."""
        self.error_history.append(error)

    def build_prompt(self) -> str:
        """Generate prompt with corrections based on error history."""
        prompt = self.base_prompt

        # Add corrections based on observed errors
        if any(isinstance(e, MissingOutBlockError) for e in self.error_history):
            prompt += "\n\nCRITICAL: You MUST include an OUT{} block in your response!"

        if any(isinstance(e, MissingFieldError) for e in self.error_history):
            required_fields = [
                f.name for f in self.spec.__fields__.values() if f.is_required()
            ]
            prompt += f"\n\nREQUIRED FIELDS: {', '.join(required_fields)}"

        if any(isinstance(e, TypeMismatchError) for e in self.error_history):
            prompt += f"\n\nUSE EXACT TYPE: {self.spec.__name__}{{}}"

        if any(isinstance(e, InvalidConstructorError) for e in self.error_history):
            prompt += """

CORRECT SYNTAX:
OUT{{
    field = Type{{key=value, key2=value2}}
}}
"""

        return prompt

# Usage
async def adaptive_extraction(base_prompt: str, spec: type[BaseModel]):
    """Extract with adaptive prompt refinement."""
    builder = AdaptivePromptBuilder(base_prompt, spec)
    max_attempts = 3

    for attempt in range(max_attempts):
        prompt = builder.build_prompt()
        llm_output = await llm_call(prompt)

        try:
            return parse_lndl(llm_output, spec=spec)
        except LNDLError as e:
            logger.warning(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            builder.add_error(e)

            if attempt == max_attempts - 1:
                logger.error("All adaptive attempts failed")
                raise
```

### Example 3: Error Analytics Dashboard

```python
from collections import defaultdict
from datetime import datetime, timedelta
from lionherd_core.lndl.errors import LNDLError

class LNDLErrorAnalytics:
    """Comprehensive error tracking and analytics."""

    def __init__(self):
        self.errors_by_type: defaultdict[str, list[tuple[datetime, str]]] = defaultdict(list)
        self.errors_by_spec: defaultdict[str, list[tuple[datetime, str]]] = defaultdict(list)
        self.total_attempts = 0
        self.successful_parses = 0

    def record_success(self, spec: type[BaseModel]):
        """Record successful parse."""
        self.total_attempts += 1
        self.successful_parses += 1

    def record_error(self, error: LNDLError, spec: type[BaseModel], context: str = ""):
        """Record error with context."""
        self.total_attempts += 1
        timestamp = datetime.now()
        error_type = type(error).__name__

        self.errors_by_type[error_type].append((timestamp, context))
        self.errors_by_spec[spec.__name__].append((timestamp, error_type))

    def get_error_rate(self, hours: int = 24) -> dict[str, float]:
        """Calculate error rates by type for recent time window."""
        cutoff = datetime.now() - timedelta(hours=hours)
        rates = {}

        for error_type, occurrences in self.errors_by_type.items():
            recent = [ts for ts, _ in occurrences if ts > cutoff]
            rates[error_type] = len(recent) / max(self.total_attempts, 1)

        return rates

    def get_problematic_specs(self, threshold: float = 0.3) -> list[str]:
        """Find Specs with high error rates."""
        problematic = []

        for spec_name, errors in self.errors_by_spec.items():
            error_rate = len(errors) / max(self.total_attempts, 1)
            if error_rate > threshold:
                problematic.append((spec_name, error_rate))

        return sorted(problematic, key=lambda x: x[1], reverse=True)

    def get_report(self) -> dict:
        """Generate comprehensive analytics report."""
        total_errors = sum(len(errors) for errors in self.errors_by_type.values())

        return {
            "total_attempts": self.total_attempts,
            "successful_parses": self.successful_parses,
            "total_errors": total_errors,
            "success_rate": self.successful_parses / max(self.total_attempts, 1),
            "error_rate": total_errors / max(self.total_attempts, 1),
            "errors_by_type": {
                error_type: len(occurrences)
                for error_type, occurrences in self.errors_by_type.items()
            },
            "most_common_errors": sorted(
                self.errors_by_type.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5],
            "problematic_specs": self.get_problematic_specs(),
            "recent_error_rates": self.get_error_rate(hours=24),
        }

# Global analytics instance
analytics = LNDLErrorAnalytics()

# Usage in production
try:
    result = parse_lndl(llm_output, spec=UserSpec)
    analytics.record_success(UserSpec)
except LNDLError as e:
    analytics.record_error(e, UserSpec, context=f"model={model_name}")
    # Handle error...

# Periodic reporting
if analytics.total_attempts % 1000 == 0:
    report = analytics.get_report()
    logger.info(f"LNDL Analytics Report: {json.dumps(report, indent=2)}")

    # Alert if error rate too high
    if report["error_rate"] > 0.2:
        alert_operations(f"High LNDL error rate: {report['error_rate']:.1%}")
```

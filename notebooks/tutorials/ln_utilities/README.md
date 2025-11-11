# ln Utilities Tutorials

Master lionherd-core's core utilities (`ln` module) for data transformation, validation, LLM integration, and async operations. Learn production-ready patterns for type conversion, fuzzy matching, data pipelines, and structured output parsing.

## Overview

These tutorials teach you to use lionherd-core's most frequently used utilities:

- **Type Conversion**: `to_dict()`, `to_list()` for flexible data transformation
- **Fuzzy Validation**: `fuzzy_validate()`, `fuzzy_match()` for resilient data validation
- **Async Utilities**: `lcall()`, `alcall()` for callable invocation patterns
- **Data Integrity**: `hash_dict()` for content-based hashing and deduplication
- **LLM Integration**: Parsing and validating LLM outputs with fuzzy tolerance

## Prerequisites

- Python 3.11+
- Basic Pydantic knowledge (for validation tutorials)
- Familiarity with async/await (for async tutorials)

## Quick Start

```bash
pip install lionherd-core
jupyter notebook fuzzy_validation.ipynb
```

## Tutorials (11 total: all available)

### Core Utilities (Start Here)

| Tutorial | Status | Time | What You'll Learn |
|----------|--------|------|-------------------|
| [**Fuzzy Validation**](./fuzzy_validation.ipynb) | ✅ Available | 15-20 min | Validate data with field name variations using `fuzzy_validate_pydantic()` |
| [**Advanced to_dict**](./advanced_to_dict.ipynb) | ✅ Available | 15-20 min | Convert complex types to dicts with `to_dict()` (Pydantic, dataclasses, custom) |
| [**Content Deduplication**](./content_deduplication.ipynb) | ✅ Available | 15-20 min | Detect duplicates using `hash_dict()` (order-independent content hashing) |
| [**Multi-Stage Pipeline**](./multistage_pipeline.ipynb) | ✅ Available | 15-20 min | Build data pipelines with `lcall()` (lazy callable invocation) |

### LLM Integration

| Tutorial | Status | Time | What You'll Learn |
|----------|--------|------|-------------------|
| [**Fuzzy JSON Parsing**](./fuzzy_json_parsing.ipynb) | ✅ Available | 20-30 min | Parse malformed LLM JSON with markdown extraction and error correction |
| [**LLM Complex Models**](./llm_complex_models.ipynb) | ✅ Available | 15-20 min | Extract structured data from LLM outputs into Pydantic models |
| [**API Field Flattening**](./api_field_flattening.ipynb) | ✅ Available | 20-30 min | Normalize nested API responses with field flattening patterns |

### Advanced Patterns

| Tutorial | Status | Time | What You'll Learn |
|----------|--------|------|-------------------|
| [**Custom JSON Serialization**](./custom_json_serialization.ipynb) | ✅ Available | 15-20 min | Handle non-JSON types (datetime, Decimal, custom classes) |
| [**Async Path Creation**](./async_path_creation.ipynb) | ✅ Available | 15-20 min | Create paths asynchronously with timeouts using `alcall()` |
| [**Nested Cleaning**](./nested_cleaning.ipynb) | ✅ Available | 15-20 min | Sanitize nested dicts by removing null values and empty collections |
| [**Data Migration**](./data_migration.ipynb) | ✅ Available | 15-20 min | Map legacy schemas to new schemas with field transformations |

## Learning Paths

### Path 1: Essential ln Utilities (1 hour)

1. **Fuzzy Validation** - Handle field name variations
2. **Advanced to_dict** - Type conversion fundamentals
3. **Content Deduplication** - Hash-based duplicate detection

**Outcome**: Use core ln utilities for data transformation and validation

### Path 2: LLM Integration Focus (1.5 hours)

1. **Fuzzy JSON Parsing** - Parse LLM outputs with error tolerance
2. **LLM Complex Models** - Extract structured data from unstructured LLM responses
3. **API Field Flattening** - Normalize nested data from various sources

**Outcome**: Build resilient LLM integration pipelines

### Path 3: Advanced Data Processing (1.5 hours)

1. **Multi-Stage Pipeline** - Build composable data pipelines
2. **Nested Cleaning** - Sanitize complex nested structures
3. **Data Migration** - Handle schema evolution and mapping
4. **Custom JSON Serialization** - Extend serialization for custom types

**Outcome**: Build production-grade data processing systems

## Key Concepts

### Fuzzy Validation

Handle data with field name variations (typos, camelCase vs snake_case):

```python
from lionherd_core.ln import fuzzy_validate_pydantic
from pydantic import BaseModel

class User(BaseModel):
    user_name: str
    email: str

# Data with field name variations
messy_data = {
    "usr_name": "Alice",  # typo
    "email": "alice@example.com"
}

# Validates successfully despite field name mismatch
user = fuzzy_validate_pydantic(
    messy_data,
    User,
    fuzzy_match=True,
    fuzzy_match_params={"similarity_threshold": 0.75}
)
```

**Tutorial**: [Fuzzy Validation](./fuzzy_validation.ipynb)

### Type Conversion with to_dict()

Convert any Python object to a dictionary:

```python
from lionherd_core.ln import to_dict
from pydantic import BaseModel
from datetime import datetime

class Event(BaseModel):
    name: str
    timestamp: datetime

event = Event(name="Deploy", timestamp=datetime.now())

# Converts Pydantic model to dict, handling datetime serialization
data = to_dict(event, serialize_all=True)
# {"name": "Deploy", "timestamp": "2025-11-09T18:30:00"}
```

**Tutorial**: [Advanced to_dict](./advanced_to_dict.ipynb)

### Content-Based Hashing

Detect duplicates using order-independent content hashing:

```python
from lionherd_core.ln import hash_dict

data1 = {"name": "Alice", "age": 30, "tags": ["admin", "user"]}
data2 = {"age": 30, "name": "Alice", "tags": ["user", "admin"]}

# Same content hash despite different key/list order
assert hash_dict(data1) == hash_dict(data2)
```

**Tutorial**: [Content Deduplication](./content_deduplication.ipynb)

### LLM Output Parsing

Parse malformed JSON from LLM responses:

```python
from lionherd_core.ln import fuzzy_validate_pydantic

# LLM response with markdown wrapping and formatting errors
llm_output = """
Here's the data:
{
    taskName: 'Deploy',  // Single quotes, unquoted key
    priority: "HIGH",
}
"""

# Extracts, corrects, and validates despite errors
task = fuzzy_validate_pydantic(
    llm_output,
    TaskModel,
    fuzzy_parse=True,   # Fix JSON formatting errors
    fuzzy_match=True    # Match field names (taskName → task_name)
)
```

**Tutorial**: [Fuzzy JSON Parsing](./fuzzy_json_parsing.ipynb)

## Common Patterns

### Pattern 1: Lenient External Data Validation

Validate data from unreliable sources with fuzzy matching:

```python
from lionherd_core.ln._fuzzy_match import FuzzyMatchKeysParams

# Reusable config for lenient validation
lenient = FuzzyMatchKeysParams(
    similarity_threshold=0.75,  # Accept more variations
    handle_unmatched="ignore",  # Skip unknown fields
    strict=False                # Don't raise on issues
)

result = fuzzy_validate_pydantic(
    third_party_data,
    ExpectedModel,
    fuzzy_match=True,
    fuzzy_match_params=lenient
)
```

### Pattern 2: Data Pipeline with lcall()

Build composable data transformation pipelines:

```python
from lionherd_core.ln import lcall

def pipeline_stage_1(data):
    return {**data, "stage1_processed": True}

def pipeline_stage_2(data):
    return {**data, "stage2_processed": True}

# Lazy callable invocation (evaluates on demand)
result = lcall([pipeline_stage_1, pipeline_stage_2], initial_data)
```

**Tutorial**: [Multi-Stage Pipeline](./multistage_pipeline.ipynb)

### Pattern 3: Duplicate Detection

Detect and remove duplicates using content hashing:

```python
from lionherd_core.ln import hash_dict

def deduplicate(items: list[dict]) -> list[dict]:
    seen = set()
    unique = []

    for item in items:
        h = hash_dict(item)
        if h not in seen:
            seen.add(h)
            unique.append(item)

    return unique
```

**Tutorial**: [Content Deduplication](./content_deduplication.ipynb)

## Production Considerations

### Fuzzy Matching Thresholds

| Threshold | Use Case | Trade-off |
|-----------|----------|-----------|
| 0.95+ | Near-exact matching (minor typos only) | May reject reasonable variations |
| 0.85 | Naming conventions (camelCase ↔ snake_case) | **Recommended default** |
| 0.75 | More lenient (typos, abbreviations) | Risk of false positives |
| <0.6 | Too permissive | High risk of incorrect matches |

### Performance

- `to_dict()`: <1ms for typical objects (<100 fields)
- `hash_dict()`: ~0.5ms for dicts with <50 keys
- `fuzzy_validate()`: ~5-15ms (depends on field count and similarity algorithm)

### Error Handling

```python
from lionherd_core.errors import ValidationError

try:
    result = fuzzy_validate_pydantic(data, Model, fuzzy_match=True)
except ValidationError as e:
    # Handle validation failure
    logger.error(f"Validation failed: {e}")
    # Fall back to lenient mode or return None
```

## Troubleshooting

### Common Issues

**Issue**: Fuzzy validation rejects reasonable field name variations
**Solution**: Lower `similarity_threshold` (try 0.75) or use custom similarity function

**Issue**: `to_dict()` doesn't serialize custom types
**Solution**: Set `serialize_all=True` or provide custom serializers via `handlers` parameter

**Issue**: `hash_dict()` produces different hashes for "same" content
**Solution**: Ensure nested structures are also dicts (not custom objects). Use `to_dict()` first if needed.

**Issue**: LLM JSON parsing still fails after `fuzzy_parse=True`
**Solution**: JSON may be too malformed. Inspect with `extract_json()` to see what was extracted.

## Related Resources

- **API Reference**: [ln module](../../../docs/api/ln/)
- **Fuzzy Validation**: [fuzzy_validate](../../../docs/api/ln/fuzzy_validate.md)
- **Type Conversion**: [to_dict](../../../docs/api/ln/to_dict.md), [to_list](../../../docs/api/ln/to_list.md)
- **Hashing**: [hash_dict](../../../docs/api/ln/hash_dict.md)

## Contributing

Found issues or have suggestions? Open an issue at [lionherd-core GitHub](https://github.com/khive-ai/lionherd-core/issues).

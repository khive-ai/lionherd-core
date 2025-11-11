# lionherd-core Tutorials

Hands-on tutorials demonstrating lionherd-core features through practical, copy-paste ready examples.

## Quick Start

1. **Install**: `pip install lionherd-core`
2. **Pick a tutorial**: Browse by category below
3. **Run**: Open in Jupyter and execute cells sequentially
4. **Time**: Most tutorials complete in 15-30 min

## Tutorial Categories

### 🔄 [Concurrency](./concurrency/) (6 tutorials)

Production-ready async/await patterns: timeout management, resource leak detection, graceful shutdown, transaction handling.

**Key Topics**: TaskGroups, timeouts, leak tracking, service lifecycle, transaction shielding

**Start**: [task_coordination.ipynb](./concurrency/task_coordination.ipynb) - Basic async coordination patterns

### 🛠️ [ln Utilities](./ln_utilities/) (11 tutorials)

Core utilities for data transformation, validation, and LLM integration. Type conversion, fuzzy matching, async operations, pipelines.

**Key Topics**: `to_dict()`, `fuzzy_validate()`, `lcall()`, `hash_dict()`, async utilities, LLM parsing

**Start**: [fuzzy_validation.ipynb](./ln_utilities/fuzzy_validation.ipynb) - Flexible validation with field name variations

### 📝 [String Handlers](./string_handlers/) (4 tutorials)

String similarity and fuzzy matching: user input correction, deduplication, phonetic matching, multi-algorithm consensus.

**Key Topics**: Jaro-Winkler, Levenshtein, Soundex, similarity algorithms, deduplication patterns

**Start**: [cli_fuzzy_matching.ipynb](./string_handlers/cli_fuzzy_matching.ipynb) - Command typo correction

### 🔧 [Schema Handlers](./schema_handlers/) (2 tutorials)

Function call parsing and dynamic schema selection for tool-calling systems (MCP servers, OpenAI tools, LLM function calling).

**Key Topics**: Function parsing, argument mapping, schema dictionaries, nested structures, MCP integration

**Start**: [dynamic_schema_selection.ipynb](./schema_handlers/dynamic_schema_selection.ipynb) - Schema dict pattern

## All Tutorials (23)

### Concurrency (6)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [Task Coordination](./concurrency/task_coordination.ipynb) | 🔵 Intermediate | 20-30 min | TaskGroups, result aggregation, error handling |
| [Deadline Patterns](./concurrency/deadline_patterns.ipynb) | 🔵 Intermediate | 20-30 min | `fail_at`, deadline propagation, timeout coordination |
| [Error Handling Timeouts](./concurrency/error_handling_timeouts.ipynb) | 🔵 Intermediate | 20-30 min | `move_on_after`, timeout recovery, partial failures |
| [Transaction Shielding](./concurrency/transaction_shielding.ipynb) | 🔵 Intermediate | 20-30 min | `shield`, commit/rollback, cleanup patterns |
| [Resource Leak Detection](./concurrency/resource_leak_detection.ipynb) | 🔴 Advanced | 25-35min | `LeakTracker`, connection pools, file handles |
| [Service Lifecycle](./concurrency/service_lifecycle.ipynb) | 🔴 Advanced | 25-35min | Service management, graceful shutdown, startup/cleanup |

### ln Utilities (11)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [Fuzzy Validation](./ln_utilities/fuzzy_validation.ipynb) | 🔵 Intermediate | 15-20min | `fuzzy_validate()`, lenient/strict modes, param objects |
| [Advanced to_dict](./ln_utilities/advanced_to_dict.ipynb) | 🔵 Intermediate | 15-20min | `to_dict()`, type conversion, custom serialization |
| [Custom JSON Serialization](./ln_utilities/custom_json_serialization.ipynb) | 🔵 Intermediate | 15-20min | JSON handlers, datetime, complex types |
| [Nested Cleaning](./ln_utilities/nested_cleaning.ipynb) | 🔵 Intermediate | 15-20min | Recursive sanitization, nested structures |
| [Content Deduplication](./ln_utilities/content_deduplication.ipynb) | 🔵 Intermediate | 15-20min | `hash_dict()`, duplicate detection, stable hashing |
| [Multi-Stage Pipeline](./ln_utilities/multistage_pipeline.ipynb) | 🔵 Intermediate | 15-20min | `lcall()`, data transformations, pipeline composition |
| [Async Path Creation](./ln_utilities/async_path_creation.ipynb) | 🔵 Intermediate | 15-20min | `alcall()`, async operations, path handling |
| [Data Migration](./ln_utilities/data_migration.ipynb) | 🔵 Intermediate | 15-20min | Schema mapping, validation, migration patterns |
| [API Field Flattening](./ln_utilities/api_field_flattening.ipynb) | 🔵 Intermediate | 20-30 min | Nested data, dot notation, normalization |
| [Fuzzy JSON Parsing](./ln_utilities/fuzzy_json_parsing.ipynb) | 🔵 Intermediate | 20-30 min | LLM output, markdown extraction, lenient parsing |
| [LLM Complex Models](./ln_utilities/llm_complex_models.ipynb) | 🔵 Intermediate | 15-20min | Pydantic models, LLM parsing, structured extraction |

### String Handlers (4)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [CLI Fuzzy Matching](./string_handlers/cli_fuzzy_matching.ipynb) | 🔵 Intermediate | 15-20min | Command correction, Jaro-Winkler, typo tolerance |
| [Consensus Matching](./string_handlers/consensus_matching.ipynb) | 🔵 Intermediate | 15-20min | Multi-algorithm voting, confidence scoring |
| [Fuzzy Deduplication](./string_handlers/fuzzy_deduplication.ipynb) | 🔵 Intermediate | 15-25min | Similarity clustering, union-find, dedup pipeline |
| [Phonetic Matching](./string_handlers/phonetic_matching.ipynb) | 🔵 Intermediate | 15-30 min | Soundex, phonetic similarity, custom callables |

### Schema Handlers (2)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [Dynamic Schema Selection](./schema_handlers/dynamic_schema_selection.ipynb) | 🔵 Intermediate | 15-25min | Schema dict, dynamic routing, positional/keyword args |
| [MCP Tool Pipeline](./schema_handlers/mcp_tool_pipeline.ipynb) | 🔵 Intermediate | 20-30 min | Function parsing, MCP integration, nested arguments |

## Learning Paths

### New to lionherd-core?

1. [Fuzzy Validation](./ln_utilities/fuzzy_validation.ipynb) - Core utility patterns
2. [Task Coordination](./concurrency/task_coordination.ipynb) - Basic concurrency
3. [CLI Fuzzy Matching](./string_handlers/cli_fuzzy_matching.ipynb) - String utilities

### LLM Integration Focus

1. [Fuzzy JSON Parsing](./ln_utilities/fuzzy_json_parsing.ipynb) - Parse LLM outputs
2. [LLM Complex Models](./ln_utilities/llm_complex_models.ipynb) - Structured extraction
3. [MCP Tool Pipeline](./schema_handlers/mcp_tool_pipeline.ipynb) - Tool calling

### Production Async Patterns

1. [Task Coordination](./concurrency/task_coordination.ipynb) - Basic patterns
2. [Error Handling Timeouts](./concurrency/error_handling_timeouts.ipynb) - Failure handling
3. [Service Lifecycle](./concurrency/service_lifecycle.ipynb) - Lifecycle management
4. [Resource Leak Detection](./concurrency/resource_leak_detection.ipynb) - Resource management

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Problem Statement**: Real-world scenario and motivation
2. **Prerequisites**: Required knowledge and packages
3. **Solution Overview**: High-level approach
4. **Implementation Steps**: Progressive examples building the solution
5. **Complete Example**: Production-ready copy-paste code
6. **Production Considerations**: Error handling, performance, testing
7. **Variations**: Alternative approaches and trade-offs
8. **Summary**: Key takeaways and related resources

## Contributing

Found an issue or want to suggest improvements? Open an issue at [lionherd-core GitHub](https://github.com/khive-ai/lionherd-core/issues).

## API Documentation

For detailed API references, see [docs/api/](../../docs/api/).

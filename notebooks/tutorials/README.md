# lionherd-core Tutorials

Hands-on tutorials demonstrating lionherd-core features through practical, copy-paste ready examples.

## Quick Start

1. **Install**: `pip install lionherd-core`
2. **Pick a tutorial**: Browse by category below
3. **Run**: Open in Jupyter and execute cells sequentially
4. **Time**: Most tutorials complete in 15-30 minutes

## Tutorial Categories

### 🔄 [Concurrency](./concurrency/) (13 tutorials)

Master async/await patterns using lionherd-core's concurrency utilities. Learn timeout management, graceful shutdown, resource leak detection, and production patterns for async Python.

**Topics**: TaskGroups, timeouts, circuit breakers, worker pools, leak detection, graceful shutdown

**Start here**: [parallel_timeout.ipynb](./concurrency/parallel_timeout.ipynb) (20 min)

### 🛠️ [ln Utilities](./ln_utilities/) (11 tutorials)

Core lionherd-core utilities for data transformation, validation, and LLM integration. Type conversion, fuzzy matching, async operations, and production-ready data pipelines.

**Topics**: `to_dict()`, `fuzzy_validate()`, `lcall()`, `hash_dict()`, async utilities, LLM parsing

**Start here**: [fuzzy_validation.ipynb](./ln_utilities/fuzzy_validation.ipynb) (15 min)

### 📝 [String Handlers](./string_handlers/) (4 tutorials)

String similarity algorithms and fuzzy matching for user input, deduplication, and approximate matching use cases.

**Topics**: Jaro-Winkler, Levenshtein, Soundex, multi-algorithm consensus, deduplication

**Start here**: [cli_fuzzy_matching.ipynb](./string_handlers/cli_fuzzy_matching.ipynb) (15 min)

### 🔧 [Schema Handlers](./schema_handlers/) (2 tutorials)

Function call parsing, argument mapping, and dynamic schema selection for tool-calling patterns (MCP, OpenAI tools, LLM function calling).

**Topics**: Function call parsing, positional/keyword mapping, schema dictionaries, nesting

**Start here**: [mcp_tool_pipeline.ipynb](./schema_handlers/mcp_tool_pipeline.ipynb) (20 min)

## All Tutorials (31)

### Concurrency (13)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [Parallel Timeout](./concurrency/parallel_timeout.ipynb) | Intermediate | 20-30min | `move_on_after`, parallel ops |
| [Deadline Task Queue](./concurrency/deadline_task_queue.ipynb) | Intermediate | 20-30min | `fail_at`, queue processing |
| [Circuit Breaker](./concurrency/circuit_breaker_timeout.ipynb) | Intermediate | 20-30min | Failure isolation, timeouts |
| [Graceful Shutdown](./concurrency/graceful_shutdown.ipynb) | Advanced | 20-30min | Cleanup, error handling |
| [Batch Partial Failure](./concurrency/batch_partial_failure.ipynb) | Intermediate | 20-30min | Resilient batch processing |
| [DB Transactions Shielded](./concurrency/db_transactions_shielded.ipynb) | Intermediate | 20-30min | `shield`, commit/rollback |
| [Connection Pool Leak](./concurrency/connection_pool_leak.ipynb) | Advanced | 25-35min | `LeakTracker`, resource management |
| [File Handle Tracking](./concurrency/file_handle_tracking.ipynb) | Intermediate | 20-25min | `LeakTracker`, file handles |
| [Lock Debugging](./concurrency/lock_debugging.ipynb) | Advanced | 25-35min | Deadlock detection, debugging |
| [Service Lifecycle](./concurrency/service_lifecycle.ipynb) | Advanced | 25-35min | Service management, TaskGroups |
| [Fan-Out/Fan-In](./concurrency/fan_out_fan_in.ipynb) | Advanced | 25-35min | Worker pools, result aggregation |
| [Rate-Limited Batch](./concurrency/rate_limited_batch.ipynb) | Intermediate | 20-30min | Token bucket, rate limiting |
| [Deadline Worker Pool](./concurrency/deadline_worker_pool.ipynb) | Advanced | 25-35min | Worker pools, deadline propagation |

### ln Utilities (11)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [Fuzzy Validation](./ln_utilities/fuzzy_validation.ipynb) | Intermediate | 15-20min | `fuzzy_validate()`, param objects |
| [LLM Complex Models](./ln_utilities/llm_complex_models.ipynb) | Intermediate | 15-20min | Pydantic, LLM parsing |
| [Advanced to_dict](./ln_utilities/advanced_to_dict.ipynb) | Intermediate | 15-20min | `to_dict()`, type conversion |
| [Custom JSON Serialization](./ln_utilities/custom_json_serialization.ipynb) | Intermediate | 15-20min | JSON, custom types |
| [Async Path Creation](./ln_utilities/async_path_creation.ipynb) | Intermediate | 15-20min | `alcall()`, async operations |
| [Fuzzy JSON Parsing](./ln_utilities/fuzzy_json_parsing.ipynb) | Intermediate | 20-30min | LLM output, markdown extraction |
| [API Field Flattening](./ln_utilities/api_field_flattening.ipynb) | Intermediate | 20-30min | Nested data, normalization |
| [Multi-Stage Pipeline](./ln_utilities/multistage_pipeline.ipynb) | Intermediate | 15-20min | `lcall()`, data pipelines |
| [Content Deduplication](./ln_utilities/content_deduplication.ipynb) | Intermediate | 15-20min | `hash_dict()`, deduplication |
| [Nested Cleaning](./ln_utilities/nested_cleaning.ipynb) | Intermediate | 15-20min | Nested structures, sanitization |
| [Data Migration](./ln_utilities/data_migration.ipynb) | Intermediate | 15-20min | Schema mapping, migration |

### String Handlers (4)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [CLI Fuzzy Matching](./string_handlers/cli_fuzzy_matching.ipynb) | Intermediate | 15-20min | Command matching, Jaro-Winkler |
| [Fuzzy Deduplication](./string_handlers/fuzzy_deduplication.ipynb) | Intermediate | 15-25min | Similarity-based dedup |
| [Consensus Matching](./string_handlers/consensus_matching.ipynb) | Intermediate | 15-20min | Multi-algorithm, voting |
| [Phonetic Matching](./string_handlers/phonetic_matching.ipynb) | Intermediate | 15-30min | Soundex, phonetic similarity |

### Schema Handlers (2)

| Tutorial | Difficulty | Time | Topics |
|----------|------------|------|--------|
| [MCP Tool Pipeline](./schema_handlers/mcp_tool_pipeline.ipynb) | Intermediate | 20-30min | Function parsing, MCP tools |
| [Dynamic Schema Selection](./schema_handlers/dynamic_schema_selection.ipynb) | Intermediate | 15-25min | Schema dictionaries, routing |

## Learning Paths

### New to lionherd-core?
1. [Fuzzy Validation](./ln_utilities/fuzzy_validation.ipynb) - Core utility patterns
2. [Parallel Timeout](./concurrency/parallel_timeout.ipynb) - Basic concurrency
3. [CLI Fuzzy Matching](./string_handlers/cli_fuzzy_matching.ipynb) - String utilities

### LLM Integration Focus
1. [Fuzzy JSON Parsing](./ln_utilities/fuzzy_json_parsing.ipynb) - Parse LLM outputs
2. [LLM Complex Models](./ln_utilities/llm_complex_models.ipynb) - Structured extraction
3. [MCP Tool Pipeline](./schema_handlers/mcp_tool_pipeline.ipynb) - Tool calling

### Production Async Patterns
1. [Parallel Timeout](./concurrency/parallel_timeout.ipynb) - Basic patterns
2. [Circuit Breaker](./concurrency/circuit_breaker_timeout.ipynb) - Failure handling
3. [Graceful Shutdown](./concurrency/graceful_shutdown.ipynb) - Lifecycle management
4. [Connection Pool Leak](./concurrency/connection_pool_leak.ipynb) - Resource management

## Tutorial Structure

Each tutorial follows a consistent 9-section structure:

1. **Problem Statement**: Real-world scenario and why it matters
2. **Prerequisites**: Required knowledge and packages
3. **Imports**: All necessary imports with comments
4. **Solution Overview**: High-level approach and key components
5. **Implementation Steps**: 3-4 progressive steps building the solution
6. **Complete Example**: Production-ready copy-paste code
7. **Production Considerations**: Error handling, performance, testing
8. **Variations**: Alternative approaches and trade-offs
9. **Summary**: Key takeaways and related resources

## Contributing

Found an issue or want to suggest improvements? Open an issue at [lionherd-core GitHub](https://github.com/khive-ai/lionherd-core/issues).

## API Documentation

For detailed API references, see [docs/api/](../../docs/api/).

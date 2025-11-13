# Shared Benchmark Utilities

Common fixtures, helpers, and profiling utilities used across benchmarks.

## Contents

- `fixtures.py` - Shared test fixtures for generating test data
- `comparisons.py` - Baseline comparison utilities (pandas, polars, etc)
- `profiling.py` - CPU and memory profiling helpers

## Usage

```python
from benchmarks.utils.fixtures import create_large_graph
from benchmarks.utils.profiling import profile_cpu

graph = create_large_graph(nodes=10000, edges=50000)
with profile_cpu():
    # ... benchmark code
```

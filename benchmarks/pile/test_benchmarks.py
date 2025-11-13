# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive benchmarks for Pile[T] data structure.

This benchmark suite measures Pile performance against baseline implementations
(dict, OrderedDict) to quantify the overhead of type safety, protocol support,
and progression tracking.

Methodology
-----------
- Size scales: 100, 1K, 10K items
- Operations: add, remove, get, contains, iteration, filtering
- Comparisons: dict, OrderedDict
- Memory profiling: Overhead of Pile vs dict

Design Intent
-------------
Pile trades some performance for:
    1. Type safety (runtime validation)
    2. Observable protocol (UUID-based identity)
    3. Progression tracking (ordered collection)
    4. Thread safety (RLock synchronization)

Benchmark Goals
---------------
1. Quantify overhead: Is Pile 2x slower? 10x? When does it matter?
2. Decision matrix: When to use Pile vs dict?
3. Identify bottlenecks: What operations are expensive?
4. Memory overhead: How much RAM does Pile use vs dict?

Performance Expectations
------------------------
Pile should be competitive with dict for:
    - O(1) operations: add, get, contains (dict-backed)
    - Iteration: similar speed (dict iteration + progression order)

Pile will be slower for:
    - Type validation: isinstance checks on add
    - Progression ops: remove is O(n) due to list scan
    - Thread safety: lock overhead

Usage
-----
Run all benchmarks:
    uv run pytest benchmarks/pile/ --benchmark-only

Run specific benchmark:
    uv run pytest benchmarks/pile/test_benchmarks.py::test_pile_add

Save results:
    uv run pytest benchmarks/pile/ --benchmark-save=pile

Compare results:
    pytest-benchmark compare pile_*

"""

import sys
from collections import OrderedDict
from typing import Any
from uuid import UUID, uuid4

import pytest

from lionherd_core.base import Element, Pile, Progression

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class BenchElement(Element):
    """Simple element for benchmarking."""

    value: int = 0


@pytest.fixture(params=[100, 1_000, 10_000])
def size(request) -> int:
    """Parametrize benchmark sizes."""
    return request.param


@pytest.fixture
def elements(size: int) -> list[BenchElement]:
    """Create test elements."""
    return [BenchElement(value=i) for i in range(size)]


@pytest.fixture
def pile(elements: list[BenchElement]) -> Pile[BenchElement]:
    """Pre-populated Pile."""
    return Pile(items=elements)


@pytest.fixture
def dict_baseline(elements: list[BenchElement]) -> dict[UUID, BenchElement]:
    """Baseline dict for comparison."""
    return {elem.id: elem for elem in elements}


@pytest.fixture
def ordered_dict_baseline(elements: list[BenchElement]) -> OrderedDict[UUID, BenchElement]:
    """Baseline OrderedDict for comparison."""
    return OrderedDict((elem.id, elem) for elem in elements)


# =============================================================================
# Core Operations: Single Item
# =============================================================================


class TestCoreOperations:
    """Benchmark core operations: add, remove, get, contains."""

    def test_pile_add(self, benchmark, size) -> None:
        """Benchmark Pile.add() - single item addition."""

        def setup():
            pile = Pile(item_type=BenchElement)
            elements = [BenchElement(value=i) for i in range(size)]
            return (pile, elements), {}

        def run(pile, elements):
            for elem in elements:
                pile.add(elem)

        benchmark.pedantic(run, setup=setup, rounds=10)

    def test_dict_add(self, benchmark, size) -> None:
        """Baseline: dict insert."""

        def setup():
            d = {}
            elements = [BenchElement(value=i) for i in range(size)]
            return (d, elements), {}

        def run(d, elements):
            for elem in elements:
                d[elem.id] = elem

        benchmark.pedantic(run, setup=setup, rounds=10)

    def test_pile_remove(self, benchmark, size) -> None:
        """Benchmark Pile.remove() - single item removal."""

        def setup():
            elements = [BenchElement(value=i) for i in range(size)]
            pile = Pile(items=elements)
            to_remove = elements[: len(elements) // 2]
            return (pile, to_remove), {}

        def run(pile, to_remove):
            for elem in to_remove:
                pile.remove(elem.id)

        benchmark.pedantic(run, setup=setup, rounds=5)

    def test_dict_remove(self, benchmark, size) -> None:
        """Baseline: dict.pop()."""

        def setup():
            elements = [BenchElement(value=i) for i in range(size)]
            d = {elem.id: elem for elem in elements}
            to_remove = elements[: len(elements) // 2]
            return (d, to_remove), {}

        def run(d, to_remove):
            for elem in to_remove:
                d.pop(elem.id)

        benchmark.pedantic(run, setup=setup, rounds=5)

    def test_pile_get(self, benchmark, pile, elements) -> None:
        """Benchmark Pile.get() - UUID lookup."""
        uuids = [elem.id for elem in elements]

        def run():
            for uuid in uuids:
                pile.get(uuid)

        benchmark(run)

    def test_dict_get(self, benchmark, dict_baseline, elements) -> None:
        """Baseline: dict[key] lookup."""
        uuids = [elem.id for elem in elements]

        def run():
            for uuid in uuids:
                dict_baseline[uuid]

        benchmark(run)

    def test_pile_contains(self, benchmark, pile, elements) -> None:
        """Benchmark Pile.__contains__() - membership test."""
        uuids = [elem.id for elem in elements]

        def run():
            for uuid in uuids:
                _ = uuid in pile

        benchmark(run)

    def test_dict_contains(self, benchmark, dict_baseline, elements) -> None:
        """Baseline: dict.__contains__()."""
        uuids = [elem.id for elem in elements]

        def run():
            for uuid in uuids:
                _ = uuid in dict_baseline

        benchmark(run)

    def test_pile_len(self, benchmark, pile) -> None:
        """Benchmark Pile.__len__()."""
        benchmark(len, pile)

    def test_dict_len(self, benchmark, dict_baseline) -> None:
        """Baseline: dict.__len__()."""
        benchmark(len, dict_baseline)

    def test_pile_iteration(self, benchmark, pile) -> None:
        """Benchmark Pile.__iter__() - full iteration."""

        def run():
            for _ in pile:
                pass

        benchmark(run)

    def test_dict_iteration(self, benchmark, dict_baseline) -> None:
        """Baseline: dict.values() iteration."""

        def run():
            for _ in dict_baseline.values():
                pass

        benchmark(run)

    def test_ordered_dict_iteration(self, benchmark, ordered_dict_baseline) -> None:
        """Baseline: OrderedDict iteration (ordered like Pile)."""

        def run():
            for _ in ordered_dict_baseline.values():
                pass

        benchmark(run)


# =============================================================================
# Bulk Operations
# =============================================================================


class TestBulkOperations:
    """Benchmark bulk operations: bulk add, bulk remove, filtering."""

    def test_pile_bulk_add_via_init(self, benchmark, elements) -> None:
        """Benchmark Pile initialization with items."""

        def run():
            Pile(items=elements, item_type=BenchElement)

        benchmark(run)

    def test_dict_bulk_add_via_comprehension(self, benchmark, elements) -> None:
        """Baseline: dict comprehension."""

        def run():
            {elem.id: elem for elem in elements}

        benchmark(run)

    def test_pile_bulk_remove(self, benchmark, size) -> None:
        """Benchmark bulk remove (remove half)."""

        def setup():
            elements = [BenchElement(value=i) for i in range(size)]
            pile = Pile(items=elements)
            to_remove = elements[: size // 2]
            return (pile, to_remove), {}

        def run(pile, to_remove):
            for elem in to_remove:
                pile.remove(elem.id)

        benchmark.pedantic(run, setup=setup, rounds=5)

    def test_pile_filter_by_predicate(self, benchmark, pile) -> None:
        """Benchmark Pile.__getitem__[callable] - predicate filtering."""

        def run():
            filtered = pile[lambda x: x.value % 2 == 0]
            return filtered

        benchmark(run)

    def test_pile_filter_by_progression(self, benchmark, pile, elements) -> None:
        """Benchmark Pile.__getitem__[Progression] - progression filtering."""
        # Filter to 50% of items
        subset_ids = [elem.id for elem in elements[::2]]
        prog = Progression(order=subset_ids)

        def run():
            filtered = pile[prog]
            return filtered

        benchmark(run)

    def test_pile_slice(self, benchmark, pile, size) -> None:
        """Benchmark Pile.__getitem__[slice] - slice access."""
        start = size // 4
        end = 3 * size // 4

        def run():
            result = pile[start:end]
            return result

        benchmark(run)


# =============================================================================
# Type Safety Overhead
# =============================================================================


class TestTypeSafetyOverhead:
    """Benchmark type validation overhead."""

    def test_pile_add_with_type_validation(self, benchmark, size) -> None:
        """Benchmark add with type validation (non-strict)."""

        def setup():
            pile = Pile(item_type=BenchElement, strict_type=False)
            elements = [BenchElement(value=i) for i in range(size)]
            return (pile, elements), {}

        def run(pile, elements):
            for elem in elements:
                pile.add(elem)

        benchmark.pedantic(run, setup=setup, rounds=10)

    def test_pile_add_strict_type_validation(self, benchmark, size) -> None:
        """Benchmark add with strict type validation."""

        def setup():
            pile = Pile(item_type=BenchElement, strict_type=True)
            elements = [BenchElement(value=i) for i in range(size)]
            return (pile, elements), {}

        def run(pile, elements):
            for elem in elements:
                pile.add(elem)

        benchmark.pedantic(run, setup=setup, rounds=10)

    def test_pile_add_no_type_validation(self, benchmark, size) -> None:
        """Benchmark add without type validation (item_type=None)."""

        def setup():
            pile = Pile()  # No type validation
            elements = [BenchElement(value=i) for i in range(size)]
            return (pile, elements), {}

        def run(pile, elements):
            for elem in elements:
                pile.add(elem)

        benchmark.pedantic(run, setup=setup, rounds=10)


# =============================================================================
# Memory Benchmarks
# =============================================================================


class TestMemoryOverhead:
    """Measure memory overhead of Pile vs dict.

    Note: These are qualitative tests that print memory info.
    Use memory_profiler for detailed analysis: @profile decorator.
    """

    def test_pile_memory_overhead(self, benchmark, size) -> None:
        """Measure Pile memory footprint."""

        def run():
            elements = [BenchElement(value=i) for i in range(size)]
            pile = Pile(items=elements)
            return pile

        pile = benchmark(run)

        # Print memory stats
        print(f"\nPile with {size} items:")
        print(f"  sys.getsizeof(pile._items): {sys.getsizeof(pile._items)} bytes")
        print(f"  sys.getsizeof(pile._progression): {sys.getsizeof(pile._progression.order)} bytes")

    def test_dict_memory_overhead(self, benchmark, size) -> None:
        """Baseline: dict memory footprint."""

        def run():
            elements = [BenchElement(value=i) for i in range(size)]
            d = {elem.id: elem for elem in elements}
            return d

        d = benchmark(run)

        # Print memory stats
        print(f"\nDict with {size} items:")
        print(f"  sys.getsizeof(dict): {sys.getsizeof(d)} bytes")


# =============================================================================
# Special Operations
# =============================================================================


class TestSpecialOperations:
    """Benchmark special Pile features."""

    def test_pile_keys_iteration(self, benchmark, pile) -> None:
        """Benchmark Pile.keys() iteration."""

        def run():
            for _ in pile.keys():  # noqa: SIM118
                pass

        benchmark(run)

    def test_pile_items_iteration(self, benchmark, pile) -> None:
        """Benchmark Pile.items() iteration."""

        def run():
            for _ in pile.items():
                pass

        benchmark(run)

    def test_pile_filter_by_type(self, benchmark, size) -> None:
        """Benchmark Pile.filter_by_type()."""

        def setup():
            # Create mixed-type pile
            class TypeA(Element):
                pass

            class TypeB(Element):
                pass

            elements = [TypeA() if i % 2 == 0 else TypeB() for i in range(size)]
            pile = Pile(items=elements, item_type={TypeA, TypeB})
            return (pile, TypeA), {}

        def run(pile, type_filter):
            pile.filter_by_type(type_filter)

        benchmark.pedantic(run, setup=setup, rounds=5)

    def test_pile_include_idempotent(self, benchmark, pile, elements) -> None:
        """Benchmark Pile.include() - idempotent add."""
        # Test repeated include (should be fast after first add)

        def run():
            for elem in elements:
                pile.include(elem)

        benchmark(run)

    def test_pile_exclude_idempotent(self, benchmark, size) -> None:
        """Benchmark Pile.exclude() - idempotent remove."""

        def setup():
            elements = [BenchElement(value=i) for i in range(size)]
            pile = Pile(items=elements)
            return (pile, elements), {}

        def run(pile, elements):
            for elem in elements:
                pile.exclude(elem.id)

        benchmark.pedantic(run, setup=setup, rounds=5)


# =============================================================================
# Analysis Utilities
# =============================================================================


def analyze_results():
    """Utility to print analysis of benchmark results.

    Run after benchmarks complete:
        pytest benchmarks/pile/ --benchmark-save=pile
        python -c "from benchmarks.pile.test_benchmarks import analyze_results; analyze_results()"
    """
    try:
        import json

        # Load latest benchmark results
        # Results saved in .benchmarks/<machine>/<timestamp>/0001_pile.json
        # This is a placeholder - user should inspect results manually
        print("\n" + "=" * 80)
        print("PILE BENCHMARK ANALYSIS")
        print("=" * 80)
        print("\nTo analyze results:")
        print("  1. Run: pytest benchmarks/pile/ --benchmark-save=pile")
        print("  2. View: .benchmarks/*/0001_pile.json")
        print("  3. Compare: pytest-benchmark compare pile_*")
        print("\nKey Questions:")
        print("  - Is Pile overhead <2x for O(1) operations? (add, get, contains)")
        print("  - Is iteration speed competitive with OrderedDict?")
        print("  - Does type validation add >10% overhead?")
        print("  - Is remove O(n) bottleneck significant?")
        print("  - When does memory overhead matter? (>10K items?)")
        print("\nDecision Matrix:")
        print("  Use Pile when:")
        print("    - Need type safety (validated collections)")
        print("    - Need Observable protocol (UUID identity)")
        print("    - Need progression tracking (ordered iteration)")
        print("    - Collection size <100K (overhead acceptable)")
        print("\n  Use dict when:")
        print("    - Pure speed required (no validation overhead)")
        print("    - Simple key-value mapping (no protocols)")
        print("    - Very large collections (>100K items)")
        print("    - No ordering requirements")
        print("=" * 80)

    except Exception as e:
        print(f"Error analyzing results: {e}")


if __name__ == "__main__":
    analyze_results()

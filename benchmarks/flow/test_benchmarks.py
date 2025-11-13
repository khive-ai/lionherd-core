# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Flow performance benchmarks for validation against pandas-level performance.

This benchmark suite validates that Flow operations maintain performance
after architectural changes (e.g., PR #117 progression refactor).

Target: Pandas-level performance for data structure operations
Baseline: Pre-PR #117 (commit before feda160)
Regression threshold: <5%

Benchmark Organization:
    - Single operations: add_item, remove_item, add_progression, etc.
    - Bulk operations: batch add/remove 100 items
    - Progression operations: append, remove, traversal
    - Complex operations: cascade deletes, multi-progression items

Datasets:
    - 1K items + 10 progressions (small workflow)
    - 10K items + 100 progressions (large workflow)

Usage:
    # Run all benchmarks
    uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only

    # Compare with baseline
    uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-compare=0001

    # Save baseline
    uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only --benchmark-save=baseline

    # Specific size
    uv run pytest benchmarks/flow/test_benchmarks.py --benchmark-only -k "1k"
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from lionherd_core.base import Element, Flow, Progression

# ==================== Test Items ====================


class BenchmarkItem(Element):
    """Test item for benchmarks."""

    value: str = "test"


class BenchmarkProgression(Progression):
    """Test progression for benchmarks."""

    pass


# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def items_1k() -> list[BenchmarkItem]:
    """Create 1K test items (reusable across benchmarks)."""
    return [BenchmarkItem(value=f"item{i}") for i in range(1000)]


@pytest.fixture(scope="module")
def items_10k() -> list[BenchmarkItem]:
    """Create 10K test items (reusable across benchmarks)."""
    return [BenchmarkItem(value=f"item{i}") for i in range(10000)]


@pytest.fixture(scope="module")
def flow_1k(items_1k: list[BenchmarkItem]) -> Flow[BenchmarkItem, BenchmarkProgression]:
    """Flow with 1K items + 10 progressions (small workflow).

    Pre-populated fixture to avoid setup time contamination.
    Items are already in pile, progressions registered.
    """
    flow = Flow[BenchmarkItem, BenchmarkProgression](
        items=items_1k,
        name="benchmark_1k",
        item_type=BenchmarkItem,
    )

    # Add 10 progressions, each with 100 items
    for i in range(10):
        prog = BenchmarkProgression(name=f"prog{i}")
        # Pre-populate progression with first 100 items
        for item in items_1k[i * 100 : (i + 1) * 100]:
            prog.append(item.id)
        flow.add_progression(prog)

    return flow


@pytest.fixture(scope="module")
def flow_10k(items_10k: list[BenchmarkItem]) -> Flow[BenchmarkItem, BenchmarkProgression]:
    """Flow with 10K items + 100 progressions (large workflow).

    Pre-populated fixture to avoid setup time contamination.
    Items are already in pile, progressions registered.
    """
    flow = Flow[BenchmarkItem, BenchmarkProgression](
        items=items_10k,
        name="benchmark_10k",
        item_type=BenchmarkItem,
    )

    # Add 100 progressions, each with 100 items
    for i in range(100):
        prog = BenchmarkProgression(name=f"prog{i}")
        # Pre-populate progression with 100 items
        for item in items_10k[i * 100 : (i + 1) * 100]:
            prog.append(item.id)
        flow.add_progression(prog)

    return flow


@pytest.fixture(scope="function")
def fresh_flow_1k(items_1k):
    """Fresh flow for modification benchmarks (function scope)."""
    return Flow[BenchmarkItem, BenchmarkProgression](
        items=items_1k,
        name="fresh_1k",
        item_type=BenchmarkItem,
    )


@pytest.fixture(scope="function")
def fresh_flow_10k(items_10k):
    """Fresh flow for modification benchmarks (function scope)."""
    return Flow[BenchmarkItem, BenchmarkProgression](
        items=items_10k,
        name="fresh_10k",
        item_type=BenchmarkItem,
    )


# ==================== Single Operation Benchmarks ====================


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_add_item(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark adding a single item to flow.

    Operation: flow.add_item(item)
    Critical path: Pile.add() + optional progression update
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k

    # Warm up: ensure flow is initialized
    _ = len(flow.items)

    # Create new item for each iteration to avoid ExistsError
    def add_item():
        item = BenchmarkItem(value=f"new_item_{uuid4()}")
        flow.add_item(item)

    benchmark(add_item)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_remove_item(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark removing a single item from flow.

    Operation: flow.remove_item(item_id)
    Critical path: Pile.remove() + cascade to all progressions
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k

    # Pre-add items to remove in each iteration
    items_to_remove = [BenchmarkItem(value=f"remove_me_{i}") for i in range(1000)]
    for item in items_to_remove:
        flow.add_item(item)

    item_ids = [item.id for item in items_to_remove]
    counter = {"index": 0}

    # Warm up
    _ = len(flow.items)

    def remove_item():
        if counter["index"] < len(item_ids):
            flow.remove_item(item_ids[counter["index"]])
            counter["index"] += 1

    benchmark(remove_item)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_add_progression(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark adding a single progression to flow.

    Operation: flow.add_progression(progression)
    Critical path: Pile.add() + name registration + referential integrity check
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k

    # Warm up
    _ = len(flow.progressions)

    # Create new progression for each iteration
    counter = {"index": 0}

    def add_progression():
        prog = BenchmarkProgression(name=f"new_prog_{counter['index']}")
        flow.add_progression(prog)
        counter["index"] += 1

    benchmark(add_progression)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_remove_progression(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark removing a single progression from flow.

    Operation: flow.remove_progression(progression_id)
    Critical path: Pile.remove() + name cleanup
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k

    # Pre-add progressions to remove in each iteration
    progs_to_remove = [BenchmarkProgression(name=f"remove_prog_{i}") for i in range(1000)]
    for prog in progs_to_remove:
        flow.add_progression(prog)

    prog_ids = [prog.id for prog in progs_to_remove]
    counter = {"index": 0}

    # Warm up
    _ = len(flow.progressions)

    def remove_progression():
        if counter["index"] < len(prog_ids):
            flow.remove_progression(prog_ids[counter["index"]])
            counter["index"] += 1

    benchmark(remove_progression)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_get_progression_by_name(benchmark, size, flow_1k, flow_10k) -> None:
    """Benchmark retrieving progression by name.

    Operation: flow.get_progression("prog0")
    Critical path: Dict lookup in _progression_names + Pile.__getitem__
    Expected: O(1) performance
    """
    flow = flow_1k if size == "1k" else flow_10k
    name = "prog0"

    # Warm up
    _ = flow.get_progression(name)

    benchmark(flow.get_progression, name)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_get_progression_by_uuid(benchmark, size, flow_1k, flow_10k) -> None:
    """Benchmark retrieving progression by UUID.

    Operation: flow.get_progression(uuid)
    Critical path: Pile.__getitem__ (dict lookup)
    Expected: O(1) performance
    """
    flow = flow_1k if size == "1k" else flow_10k
    uuid = next(iter(flow.progressions.keys()))

    # Warm up
    _ = flow.get_progression(uuid)

    benchmark(flow.get_progression, uuid)


# ==================== Bulk Operation Benchmarks ====================


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_bulk_add_items(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark bulk adding 100 items.

    Operation: Add 100 items in a loop
    Critical path: 100x Pile.add()
    Validates: Linear scaling (100x single add)
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k
    counter = {"index": 0}

    # Warm up
    _ = len(flow.items)

    def bulk_add():
        # Generate 100 new items for each iteration
        items = [BenchmarkItem(value=f"bulk{counter['index']}_{i}") for i in range(100)]
        counter["index"] += 1
        for item in items:
            flow.add_item(item)

    benchmark(bulk_add)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_bulk_remove_items(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark bulk removing 100 items.

    Operation: Remove 100 items in a loop
    Critical path: 100x (Pile.remove() + cascade to progressions)
    Validates: Linear scaling with cascade overhead
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k

    # Pre-add many items that can be removed
    items_pool = []
    for i in range(10000):
        item = BenchmarkItem(value=f"bulk_remove_pool_{i}")
        flow.add_item(item)
        items_pool.append(item.id)

    counter = {"index": 0}

    # Warm up
    _ = len(flow.items)

    def bulk_remove():
        # Remove 100 items from the pool
        start_idx = counter["index"] * 100
        end_idx = start_idx + 100
        if end_idx <= len(items_pool):
            for item_id in items_pool[start_idx:end_idx]:
                flow.remove_item(item_id)
            counter["index"] += 1

    benchmark(bulk_remove)


# ==================== Progression Operation Benchmarks ====================


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_progression_append(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark appending item to progression.

    Operation: progression.append(item_id)
    Critical path: Progression.append() (list append + validation)
    Expected: O(1) amortized
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k
    prog = BenchmarkProgression(name="append_prog")
    flow.add_progression(prog)

    # Get item IDs to append
    item_ids = list(flow.items.keys())
    counter = {"index": 0}

    # Warm up
    _ = len(prog)

    def append_item():
        if counter["index"] < len(item_ids):
            prog.append(item_ids[counter["index"]])
            counter["index"] += 1

    benchmark(append_item)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_progression_remove(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark removing item from progression.

    Operation: progression.remove(item_id)
    Critical path: Progression.remove() (list.remove + UUID search)
    Expected: O(n) where n=len(progression)
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k
    prog = BenchmarkProgression(name="remove_prog")

    # Pre-populate progression with items
    item_ids = list(flow.items.keys())[:100]
    for item_id in item_ids:
        prog.append(item_id)

    flow.add_progression(prog)
    counter = {"index": 0}

    # Warm up
    _ = len(prog)

    def remove_item():
        if counter["index"] < len(item_ids):
            prog.remove(item_ids[counter["index"]])
            counter["index"] += 1

    benchmark(remove_item)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_progression_traversal(benchmark, size, flow_1k, flow_10k) -> None:
    """Benchmark traversing progression order.

    Operation: for item_id in progression.order: flow.items[item_id]
    Critical path: Iteration + dict lookups
    Expected: O(n) where n=len(progression)
    Validates: Iteration overhead + item retrieval
    """
    flow = flow_1k if size == "1k" else flow_10k
    prog = next(iter(flow.progressions))

    def traverse():
        items = []
        for item_id in prog.order:
            items.append(flow.items[item_id])
        return items

    # Warm up
    _ = traverse()

    benchmark(traverse)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_progression_contains(benchmark, size, flow_1k, flow_10k) -> None:
    """Benchmark checking if item is in progression.

    Operation: item_id in progression
    Critical path: Progression.__contains__() (set lookup)
    Expected: O(1) performance
    """
    flow = flow_1k if size == "1k" else flow_10k
    prog = next(iter(flow.progressions))
    item_id = list(prog)[50] if len(prog) > 50 else next(iter(prog))

    # Warm up
    _ = item_id in prog

    benchmark(lambda: item_id in prog)


# ==================== Complex Operation Benchmarks ====================


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_add_item_to_multiple_progressions(
    benchmark, size, fresh_flow_1k, fresh_flow_10k
) -> None:
    """Benchmark adding item to multiple progressions (M:N relationship).

    Operation: flow.add_item(item, progressions=[prog1, prog2, ..., prog10])
    Critical path: Pile.add() + 10x progression.append()
    Validates: Multi-progression overhead
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k

    # Create 10 progressions
    progs = []
    for i in range(10):
        prog = BenchmarkProgression(name=f"multi_prog{i}")
        flow.add_progression(prog)
        progs.append(prog.id)

    # Warm up
    _ = len(flow.items)

    counter = {"index": 0}

    def add_item_multi():
        item = BenchmarkItem(value=f"multi_item_{counter['index']}")
        flow.add_item(item, progs)
        counter["index"] += 1

    benchmark(add_item_multi)


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_remove_item_cascade(benchmark, size, fresh_flow_1k, fresh_flow_10k) -> None:
    """Benchmark removing item that exists in multiple progressions (cascade delete).

    Operation: flow.remove_item(item_id)
    Critical path: O(P) scan where P=number of progressions
    Validates: Cascade delete overhead scales with progression count
    """
    flow = fresh_flow_1k if size == "1k" else fresh_flow_10k

    # Create 10 progressions
    progs = []
    for i in range(10):
        prog = BenchmarkProgression(name=f"cascade_prog{i}")
        flow.add_progression(prog)
        progs.append(prog)

    # Add items that will be in all progressions
    items_to_remove = [BenchmarkItem(value=f"cascade_item_{i}") for i in range(100)]
    for item in items_to_remove:
        flow.add_item(item, [p.id for p in progs])

    item_ids = [item.id for item in items_to_remove]
    counter = {"index": 0}

    # Warm up
    _ = len(flow.items)

    def remove_item_cascade():
        if counter["index"] < len(item_ids):
            flow.remove_item(item_ids[counter["index"]])
            counter["index"] += 1

    benchmark(remove_item_cascade)


# ==================== Serialization Benchmarks ====================


@pytest.mark.parametrize("size", ["1k", "10k"])
def test_flow_to_dict(benchmark, size, flow_1k, flow_10k) -> None:
    """Benchmark Flow serialization to dict.

    Operation: flow.to_dict()
    Critical path: Element.to_dict() + Pile.to_dict() for items and progressions
    Validates: Serialization scales linearly with data size
    """
    flow = flow_1k if size == "1k" else flow_10k

    # Warm up
    _ = flow.to_dict()

    benchmark(flow.to_dict)


@pytest.mark.parametrize("size", ["1k"])  # 10K is too slow for from_dict
def test_flow_from_dict(benchmark, size, flow_1k) -> None:
    """Benchmark Flow deserialization from dict.

    Operation: Flow.from_dict(data)
    Critical path: Pydantic validation + Pile reconstruction + name index rebuild
    Validates: Deserialization overhead
    Note: Only run on 1K dataset (10K is too slow for baseline)
    """
    flow = flow_1k
    data = flow.to_dict()

    # Warm up
    _ = Flow.from_dict(data)

    benchmark(Flow.from_dict, data)


# ==================== Referential Integrity Benchmarks ====================


def test_flow_referential_integrity_validation_1k(benchmark, items_1k) -> None:
    """Benchmark referential integrity validation during Flow construction.

    Operation: Flow(items, progressions) with model_validator
    Critical path: Set operations to validate all progression UUIDs exist in items
    Expected: O(I + P*M) where I=items, P=progressions, M=items per progression
    """
    # Create progressions that reference items
    progs = []
    for i in range(10):
        prog = BenchmarkProgression(name=f"val_prog{i}")
        # Add 100 items to each progression
        for item in items_1k[i * 100 : (i + 1) * 100]:
            prog.append(item.id)
        progs.append(prog)

    def create_flow():
        return Flow[BenchmarkItem, BenchmarkProgression](
            items=items_1k,
            progressions=progs,
            name="validation_test",
        )

    # Warm up
    _ = create_flow()

    benchmark(create_flow)


# ==================== Name Index Benchmarks ====================


def test_flow_name_index_lookup_1k(benchmark, flow_1k) -> None:
    """Benchmark name index lookup performance.

    Operation: flow._progression_names[name]
    Critical path: Dict lookup
    Expected: O(1) performance
    Validates: Name index is efficient for ergonomic access
    """
    flow = flow_1k
    name = "prog0"

    # Warm up
    _ = flow._progression_names[name]

    benchmark(lambda: flow._progression_names[name])


def test_flow_name_index_rebuild_after_deserialization(benchmark, flow_1k) -> None:
    """Benchmark name index rebuild in model_post_init.

    Operation: Flow.from_dict() -> model_post_init() -> rebuild _progression_names
    Critical path: Iterate progressions + dict insertions
    Expected: O(P) where P=number of progressions
    Validates: Deserialization doesn't lose name index efficiency
    """
    flow = flow_1k
    data = flow.to_dict()

    def deserialize_and_rebuild():
        return Flow.from_dict(data)

    # Warm up
    _ = deserialize_and_rebuild()

    benchmark(deserialize_and_rebuild)

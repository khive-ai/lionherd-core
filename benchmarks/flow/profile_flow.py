#!/usr/bin/env python
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Flow Performance Profiling

Profiles Flow operations at scale (10K-100K) to identify hot paths and optimization opportunities.

Usage:
    uv run python scripts/profile_flow.py [--size SIZE] [--memory] [--cpu]

Options:
    --size SIZE     Number of operations (default: 10000)
    --memory        Enable memory profiling (slower, detailed allocations)
    --cpu           Enable CPU profiling (default: always on)
    --help          Show this help

Output:
    - CPU profiling: Top 30 functions by cumulative time
    - Memory profiling: Line-by-line memory usage (if --memory)

Examples:
    # Quick CPU profiling with 10K operations
    uv run python scripts/profile_flow.py

    # Large-scale profiling with memory analysis
    uv run python scripts/profile_flow.py --size 50000 --memory

    # Focus on hot paths with 100K operations
    uv run python scripts/profile_flow.py --size 100000
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lionherd_core import Flow, Node, Progression


def build_flow(size: int, num_progressions: int = 10) -> tuple[Flow, list[Node], list[Progression]]:
    """Build flow with specified number of items and progressions.

    Creates a flow with 'size' items distributed across 'num_progressions' progressions.

    Args:
        size: Number of items to create
        num_progressions: Number of progressions to create

    Returns:
        Tuple of (flow, items, progressions)
    """
    print(f"\n{'=' * 60}")
    print(f"Building flow with {size:,} items, {num_progressions} progressions...")
    print(f"{'=' * 60}")

    flow = Flow(name="perf_test_flow")
    items = []
    progressions = []

    # Phase 1: Add items
    start = time.perf_counter()
    for i in range(size):
        item = Node(content={"index": i, "data": f"item_{i}"})
        flow.add_item(item)
        items.append(item)
        if (i + 1) % (size // 10) == 0:
            print(f"  Added {i + 1:,}/{size:,} items...")
    elapsed = time.perf_counter() - start
    print(f"✓ Items added: {elapsed:.2f}s ({size / elapsed:,.0f} ops/s)")

    # Phase 2: Create progressions
    start = time.perf_counter()
    items_per_prog = size // num_progressions

    for p in range(num_progressions):
        # Each progression gets a slice of items
        start_idx = p * items_per_prog
        end_idx = start_idx + items_per_prog if p < num_progressions - 1 else size

        prog = Progression(name=f"progression_{p}")
        for i in range(start_idx, end_idx):
            prog.append(items[i].id)

        flow.add_progression(prog)
        progressions.append(prog)

        if (p + 1) % max(1, num_progressions // 5) == 0:
            print(f"  Created {p + 1}/{num_progressions} progressions...")

    elapsed = time.perf_counter() - start
    print(f"✓ Progressions created: {elapsed:.2f}s")

    print("\n📊 Flow Stats:")
    print(f"   Items: {len(flow.items):,}")
    print(f"   Progressions: {len(flow.progressions):,}")
    print(f"   Avg items/progression: {len(flow.items) / len(flow.progressions):,.0f}")

    return flow, items, progressions


def profile_flow_operations(size: int = 10000, num_progressions: int = 10):
    """Profile Flow add/remove/query operations.

    Args:
        size: Number of items to create
        num_progressions: Number of progressions
    """
    flow, items, progressions = build_flow(size, num_progressions)

    # Profile progression queries
    print(f"\n{'=' * 60}")
    print("Profiling progression queries...")
    print(f"{'=' * 60}")

    start = time.perf_counter()
    for p in progressions:
        _ = flow.get_progression(p.id)
        _ = flow.get_progression(p.name)
        _ = list(p)  # Iterate through progression
    elapsed = time.perf_counter() - start
    ops = len(progressions) * 3
    print(f"✓ Query operations: {elapsed:.2f}s ({ops / elapsed:,.0f} ops/s)")

    # Profile adding items to existing progressions
    print(f"\n{'=' * 60}")
    print("Profiling item additions to progressions...")
    print(f"{'=' * 60}")

    new_items = []
    add_count = size // 10
    start = time.perf_counter()
    for i in range(add_count):
        item = Node(content={"index": size + i, "data": f"new_item_{i}"})
        # Add to first progression
        flow.add_item(item, progressions=progressions[0].id)
        new_items.append(item)
        if (i + 1) % (add_count // 5) == 0:
            print(f"  Added {i + 1:,}/{add_count:,} items...")

    elapsed = time.perf_counter() - start
    print(f"✓ Item additions: {elapsed:.2f}s ({add_count / elapsed:,.0f} ops/s)")

    # Profile item removal (cascades through progressions)
    print(f"\n{'=' * 60}")
    print("Profiling item removal (cascading progression updates)...")
    print(f"{'=' * 60}")

    # Remove items from middle progressions
    remove_count = size // 20
    start = time.perf_counter()
    for i in range(remove_count):
        idx = size // 2 + i
        if idx < len(items):
            flow.remove_item(items[idx].id)
        if (i + 1) % (remove_count // 5) == 0:
            print(f"  Removed {i + 1:,}/{remove_count:,} items...")

    elapsed = time.perf_counter() - start
    print(f"✓ Item removal: {elapsed:.2f}s ({remove_count / elapsed:,.0f} ops/s)")

    print("\n📊 Final Flow Stats:")
    print(f"   Items: {len(flow.items):,}")
    print(f"   Progressions: {len(flow.progressions):,}")


def profile_flow_progression_management(size: int = 10000):
    """Profile progression add/remove operations.

    Args:
        size: Number of items in flow
    """
    print(f"\n{'=' * 60}")
    print("Profiling progression management...")
    print(f"{'=' * 60}")

    flow, items, _ = build_flow(size, num_progressions=5)

    # Profile adding new progressions
    add_count = 50
    start = time.perf_counter()
    new_progressions = []
    items_per_prog = size // add_count

    for i in range(add_count):
        prog = Progression(name=f"new_prog_{i}")
        # Add subset of items to progression
        start_idx = i * items_per_prog
        end_idx = min(start_idx + items_per_prog, len(items))
        for idx in range(start_idx, end_idx):
            prog.append(items[idx].id)

        flow.add_progression(prog)
        new_progressions.append(prog)

        if (i + 1) % (add_count // 5) == 0:
            print(f"  Added {i + 1}/{add_count} progressions...")

    elapsed = time.perf_counter() - start
    print(f"✓ Progression additions: {elapsed:.2f}s ({add_count / elapsed:,.0f} ops/s)")

    # Profile progression removal
    remove_count = 25
    start = time.perf_counter()
    for i in range(remove_count):
        flow.remove_progression(new_progressions[i].id)

    elapsed = time.perf_counter() - start
    print(f"✓ Progression removal: {elapsed:.2f}s ({remove_count / elapsed:,.0f} ops/s)")

    print("\n📊 Final Stats:")
    print(f"   Progressions: {len(flow.progressions):,}")


def profile_flow_serialization(size: int = 10000, num_progressions: int = 10):
    """Profile flow serialization/deserialization.

    Args:
        size: Number of items in flow
        num_progressions: Number of progressions
    """
    print(f"\n{'=' * 60}")
    print("Profiling serialization...")
    print(f"{'=' * 60}")

    flow, _, _ = build_flow(size, num_progressions)

    # Profile to_dict
    start = time.perf_counter()
    data = flow.to_dict(mode="json")
    elapsed = time.perf_counter() - start
    print(f"✓ to_dict(mode='json'): {elapsed:.2f}s")

    # Profile from_dict
    start = time.perf_counter()
    restored = Flow.from_dict(data)
    elapsed = time.perf_counter() - start
    print(f"✓ from_dict(): {elapsed:.2f}s")
    print(
        f"   Restored: {len(restored.items):,} items, {len(restored.progressions):,} progressions"
    )


def run_profiling(size: int, enable_memory: bool = False, enable_cpu: bool = True):
    """Run profiling with specified configuration.

    Args:
        size: Number of operations
        enable_memory: Enable memory profiling (slower)
        enable_cpu: Enable CPU profiling
    """
    print(f"\n{'=' * 60}")
    print("FLOW PROFILING")
    print(f"{'=' * 60}")
    print("Configuration:")
    print(f"  Operations: {size:,}")
    print(f"  CPU profiling: {'enabled' if enable_cpu else 'disabled'}")
    print(f"  Memory profiling: {'enabled' if enable_memory else 'disabled'}")

    if enable_cpu:
        # CPU profiling with cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    # Run workloads
    profile_flow_operations(size, num_progressions=10)
    profile_flow_progression_management(size)
    profile_flow_serialization(min(size, 5000), num_progressions=10)

    if enable_cpu:
        profiler.disable()

        # Print CPU profiling results
        print(f"\n{'=' * 60}")
        print("CPU PROFILING RESULTS (Top 30 by cumulative time)")
        print(f"{'=' * 60}")

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(30)
        print(stream.getvalue())

        # Hot path analysis
        print(f"\n{'=' * 60}")
        print("HOT PATH ANALYSIS (Top 20 by time)")
        print(f"{'=' * 60}")
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats("time")
        stats.print_stats(20)
        print(stream.getvalue())

    if enable_memory:
        print("\n⚠️  Memory profiling requires @profile decorator on functions.")
        print("To enable detailed memory profiling:")
        print("  1. Add @profile decorator to Flow methods in flow.py")
        print("  2. Run: python -m memory_profiler scripts/profile_flow.py")


def main():
    parser = argparse.ArgumentParser(
        description="Profile Flow operations for performance optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Number of operations (default: 10000)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable memory profiling (slower)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=True,
        help="Enable CPU profiling (default: enabled)",
    )

    args = parser.parse_args()

    run_profiling(size=args.size, enable_memory=args.memory, enable_cpu=args.cpu)

    print(f"\n{'=' * 60}")
    print("PROFILING COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("  1. Review hot paths in CPU profiling results")
    print("  2. Compare against pandas/polars for baseline")
    print("  3. Consider Cython for hot functions (>10% cumulative time)")
    print("  4. Profile memory with: python -m memory_profiler scripts/profile_flow.py")


if __name__ == "__main__":
    main()

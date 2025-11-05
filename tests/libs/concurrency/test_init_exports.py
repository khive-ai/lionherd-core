# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import lionherd_core.libs.concurrency as p


def test_public_exports_exist():
    expected = {
        "CancelScope",
        "fail_after",
        "move_on_after",
        "fail_at",
        "move_on_at",
        "effective_deadline",
        "get_cancelled_exc_class",
        "is_cancelled",
        "shield",
        "TaskGroup",
        "create_task_group",
        "Lock",
        "Semaphore",
        "CapacityLimiter",
        "Queue",
        "PriorityQueue",
        "QueueEmpty",
        "QueueFull",
        "gather",
        "race",
        "bounded_map",
        "retry",
        "track_resource",
        "untrack_resource",
        "LeakInfo",
        "LeakTracker",
        "ConcurrencyEvent",
        "ExceptionGroup",
    }
    for name in expected:
        assert hasattr(p, name), f"missing export: {name}"

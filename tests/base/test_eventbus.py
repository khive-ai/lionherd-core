# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""EventBus pub/sub pattern for observability and tracing.

Pattern Overview:
    Publisher: Emits events (fire-and-forget, no return value)
    Subscriber: Registers handlers with topic-based filtering
    Decoupling: Publishers don't know subscribers, subscribers don't block publishers

    This pattern enables distributed observability without coupling business logic
    to instrumentation concerns.

Handler Dispatch:
    Concurrent: All handlers run concurrently via asyncio.gather()
    Async-only: Handlers must be async coroutines
    Exception isolation: Handler failures don't propagate or affect other handlers
    No ordering guarantee: Handlers execute in arbitrary order

    Dispatch is fire-and-forget - emit() returns after all handlers complete/fail,
    but individual handler results are not captured.

Filtering:
    Topic-based: Subscribe to specific event types (e.g., "node.start", "node.error")
    No wildcards: Each topic is exact-match only
    Multiple topics: Handlers can subscribe to multiple topics independently

    Advanced filtering (predicates, payload matching) not currently supported.

Use Cases:
    - Distributed tracing: Emit span start/end events, handlers collect traces
    - Metrics collection: Count events, measure durations, track throughput
    - Audit logging: Record state changes, user actions, system events
    - Error reporting: Centralize exception handling, send to monitoring systems
    - Observable execution: Instrument async workflows without tight coupling

Test Coverage:
    - TestEventBusBasics: Core subscribe/emit/unsubscribe mechanics
    - TestEventBusConcurrency: Concurrent handler execution behavior
    - TestEventBusExceptionHandling: Exception isolation and safety
    - TestEventBusManagement: Bus introspection and lifecycle
    - TestEventBusIntegration: Real-world observability patterns
"""

import asyncio

import pytest

from lionherd_core.base import EventBus


class TestEventBusBasics:
    """Test fundamental subscribe/emit/unsubscribe operations.

    Pub/Sub Fundamentals:
        subscribe(topic, handler): Register async handler for topic
        emit(topic, **kwargs): Fire event to all subscribers (concurrent)
        unsubscribe(topic, handler): Remove specific handler

    Pattern Validation:
        - Publishers emit without knowing who's listening (decoupling)
        - Subscribers receive all events for their subscribed topics
        - Unsubscribe prevents future events, doesn't affect past emissions
        - Topics are isolated namespaces (no cross-contamination)
    """

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        """Handler receives emitted events.

        Tests basic pub/sub contract: subscriber registration → event emission → handler invocation.
        This is the foundation for all observability patterns.

        Pattern:
            1. Register handler (subscriber declares interest)
            2. Emit event (publisher fires, doesn't wait for specific handler)
            3. Handler executes asynchronously (receives event data)

        Real-world: Metrics handler subscribes to "request.complete",
                   receives latency data on each request.
        """
        bus = EventBus()
        received = []

        async def handler(value: int):
            received.append(value)

        bus.subscribe("test", handler)
        await bus.emit("test", value=42)

        assert received == [42]

    @pytest.mark.asyncio
    async def test_multiple_emits(self):
        """Handler processes multiple emissions.

        Tests event stream processing: single subscriber receives all events in topic.
        Critical for metrics aggregation and audit trail construction.

        Pattern:
            Stateful subscriber: Handler maintains state across events
            Sequential semantics: await emit() ensures order within single emitter
            Stream processing: Each event processed independently

        Real-world: Counter that tracks total requests by subscribing to "request.start"
                   and incrementing on each event.
        """
        bus = EventBus()
        received = []

        async def handler(value: int):
            received.append(value)

        bus.subscribe("test", handler)
        await bus.emit("test", value=1)
        await bus.emit("test", value=2)
        await bus.emit("test", value=3)

        assert received == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Unsubscribed handlers stop receiving events.

        Tests dynamic subscription management: handlers can opt-out at runtime.
        Essential for lifecycle management and resource cleanup.

        Pattern:
            Temporal subscription: Handler active for limited time window
            Clean removal: unsubscribe returns True on success
            Immediate effect: No events received after unsubscribe

        Real-world: Request-scoped tracing handler unsubscribes after request completes,
                   preventing memory leaks from accumulating handlers.
        """
        bus = EventBus()
        received = []

        async def handler(value: int):
            received.append(value)

        bus.subscribe("test", handler)
        await bus.emit("test", value=1)

        assert bus.unsubscribe("test", handler) is True
        await bus.emit("test", value=2)

        assert received == [1]  # Only received first emission

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self):
        """Unsubscribing nonexistent handler returns False.

        Tests error handling for invalid unsubscribe operations.
        Safe API: doesn't raise, returns False for non-existent handlers/topics.

        Pattern:
            Idempotent cleanup: Safe to call unsubscribe multiple times
            Error signaling: Boolean return indicates success/failure
            No exceptions: Defensive API design for cleanup code

        Real-world: Shutdown handlers can safely unsubscribe without checking
                   if subscription exists, simplifying cleanup logic.
        """
        bus = EventBus()

        async def handler():
            pass

        # Handler not subscribed
        assert bus.unsubscribe("test", handler) is False

        # Topic doesn't exist
        assert bus.unsubscribe("nonexistent", handler) is False

    @pytest.mark.asyncio
    async def test_emit_no_handlers(self):
        """Emitting to topic with no handlers is safe.

        Tests fire-and-forget semantics: publishers don't require subscribers.
        Critical for decoupling - instrumentation code doesn't break if no observers.

        Pattern:
            Optional observability: Emit events unconditionally
            No-op when unused: Zero overhead if no subscribers
            Defensive: No validation that topic exists or has subscribers

        Real-world: Development code emits trace events, production may have zero
                   trace subscribers. Code continues working without modification.
        """
        bus = EventBus()

        # Should not raise
        await bus.emit("nonexistent", value=42)

    @pytest.mark.asyncio
    async def test_multiple_topics(self):
        """Different topics maintain separate subscriptions.

        Tests topic isolation: handlers only receive events from subscribed topics.
        Foundation for event routing and selective observability.

        Pattern:
            Topic namespacing: Each topic is independent event stream
            Selective subscription: Handlers choose which events to observe
            No cross-talk: Events in one topic don't affect other topics

        Real-world: Error handler subscribes to "node.error" only, doesn't receive
                   "node.start" or "node.complete" events, reducing noise.
        """
        bus = EventBus()
        topic1_received = []
        topic2_received = []

        async def handler1(value: int):
            topic1_received.append(value)

        async def handler2(value: int):
            topic2_received.append(value)

        bus.subscribe("topic1", handler1)
        bus.subscribe("topic2", handler2)

        await bus.emit("topic1", value=1)
        await bus.emit("topic2", value=2)

        assert topic1_received == [1]
        assert topic2_received == [2]


class TestEventBusConcurrency:
    """Test concurrent handler execution via asyncio.gather().

    Concurrency Model:
        All handlers for a topic execute concurrently (not sequentially)
        Implementation: asyncio.gather(*[handler(**kwargs) for handler in handlers])
        No ordering guarantees: Fast handlers may complete before slow ones start
        Structured concurrency: emit() awaits all handlers before returning

    Performance:
        Concurrent dispatch reduces total latency for multiple handlers
        Slow handlers don't block fast handlers (parallelism)
        All handlers share event loop (cooperative multitasking)

    Use Cases:
        - Parallel metrics collection (Prometheus, StatsD, CloudWatch simultaneously)
        - Concurrent logging (local file + remote syslog + stdout)
        - Multi-destination tracing (Jaeger + DataDog + local spans)
    """

    @pytest.mark.asyncio
    async def test_multiple_handlers_concurrent(self):
        """Multiple handlers run concurrently.

        Tests asyncio.gather() dispatch: fast handler completes before slow handler finishes.
        Critical for performance when multiple instrumentation systems subscribe.

        Pattern:
            Concurrent start: All handlers begin execution immediately
            Independent progress: Slow handlers don't block fast ones
            Structured completion: emit() waits for all handlers

        Mechanics:
            1. emit() creates list of handler coroutines
            2. asyncio.gather() schedules all concurrently
            3. Event loop interleaves execution
            4. Fast handler completes while slow handler awaits

        Real-world: StatsD metrics (fast UDP) complete while CloudWatch API call
                   (slow HTTPS) is still pending. Total latency = max(handlers),
                   not sum(handlers).
        """
        bus = EventBus()
        execution_order = []

        async def slow_handler():
            execution_order.append("slow_start")
            await asyncio.sleep(0.01)
            execution_order.append("slow_end")

        async def fast_handler():
            execution_order.append("fast")

        bus.subscribe("test", slow_handler)
        bus.subscribe("test", fast_handler)

        await bus.emit("test")

        # Fast handler should complete before slow_handler finishes
        # Both start concurrently, so fast completes before slow_end
        assert execution_order[0] == "slow_start"
        assert "fast" in execution_order
        assert execution_order[-1] == "slow_end"

    @pytest.mark.asyncio
    async def test_handler_execution_order_unspecified(self):
        """Handler execution order is not guaranteed.

        Tests non-deterministic ordering: asyncio.gather() doesn't guarantee execution order.
        Important constraint for handler design - don't assume ordering.

        Pattern:
            Unordered execution: Handlers may run in any order
            All-or-nothing completion: All handlers complete, order irrelevant
            Stateless handlers: Don't depend on execution order

        Design Implications:
            - Handlers must be independent (no shared mutable state)
            - Order-sensitive operations need explicit coordination
            - Metrics/logging naturally order-independent

        Real-world: Three logging handlers (file, syslog, stdout) execute in arbitrary
                   order. Each handler independent, final logs contain all three.
        """
        bus = EventBus()
        received = []

        async def handler1():
            received.append(1)

        async def handler2():
            received.append(2)

        async def handler3():
            received.append(3)

        bus.subscribe("test", handler1)
        bus.subscribe("test", handler2)
        bus.subscribe("test", handler3)

        await bus.emit("test")

        # All handlers executed (order unspecified)
        assert sorted(received) == [1, 2, 3]


class TestEventBusExceptionHandling:
    """Test exception isolation and safety via asyncio.gather(return_exceptions=True).

    Exception Isolation:
        Handler failures don't propagate: emit() never raises
        Implementation: asyncio.gather(..., return_exceptions=True)
        Independent failure: One handler's exception doesn't affect others
        Silent suppression: Exceptions caught, not logged (by design)

    Rationale:
        Observability failures shouldn't break business logic
        Instrumentation is non-critical by definition
        Publisher doesn't care if observers fail

    Design Tradeoffs:
        ✓ Robustness: Business logic protected from instrumentation bugs
        ✓ Isolation: Faulty handler doesn't cascade
        ⚠ Silent failures: Handler exceptions not reported (add monitoring handler if needed)

    Use Cases:
        - Metrics endpoint down: Other handlers continue
        - Logging disk full: Metrics collection unaffected
        - Tracing service timeout: Application continues
    """

    @pytest.mark.asyncio
    async def test_handler_exception_suppressed(self):
        """Handler exceptions don't propagate to emit().

        Tests fire-and-forget safety: instrumentation failures don't break business logic.
        Foundation of robust observability - failures isolated.

        Pattern:
            Defensive emit: Never raises, even if all handlers fail
            Exception swallowing: asyncio.gather(return_exceptions=True)
            Publisher protection: Business code unaware of observer failures

        Mechanics:
            1. Handler raises during execution
            2. asyncio.gather() captures exception (doesn't propagate)
            3. emit() completes normally (no re-raise)

        Real-world: Metrics handler crashes due to network error. Application
                   continues processing requests, unaffected by monitoring failure.
        """
        bus = EventBus()

        async def failing_handler():
            raise ValueError("Handler error")

        bus.subscribe("test", failing_handler)

        # Should not raise
        await bus.emit("test")

    @pytest.mark.asyncio
    async def test_exception_isolation(self):
        """One handler's exception doesn't affect others.

        Tests independent failure: concurrent handlers isolated from each other's exceptions.
        Critical for multi-handler reliability.

        Pattern:
            Blast radius = 1: Single handler failure contained
            Continue on error: Working handlers execute fully
            No cascade: Exception in one doesn't propagate to others

        Mechanics:
            asyncio.gather(return_exceptions=True) catches each handler's exception
            independently, allowing other handlers to complete normally.

        Real-world: Logging handler fails (disk full), but metrics handler successfully
                   records count. Both subscribed to "request.complete", only one fails.
        """
        bus = EventBus()
        received = []

        async def failing_handler(**kwargs):
            raise ValueError("Handler error")

        async def working_handler(value: int, **kwargs):
            received.append(value)

        bus.subscribe("test", failing_handler)
        bus.subscribe("test", working_handler)

        await bus.emit("test", value=42)

        # Working handler still executed
        assert received == [42]

    @pytest.mark.asyncio
    async def test_multiple_exceptions(self):
        """Multiple handler exceptions are all suppressed.

        Tests worst-case resilience: all handlers fail, emit() still completes normally.
        Demonstrates fire-and-forget commitment - publisher never sees failures.

        Pattern:
            Total failure tolerance: Even 100% handler failure rate is safe
            No partial success reporting: emit() doesn't return success/failure counts
            Pure fire-and-forget: Publisher completely decoupled from handler health

        Mechanics:
            All handlers raise → asyncio.gather() captures all exceptions → emit() returns.
            No indication of failures propagated to caller.

        Real-world: Network partition causes all remote monitoring handlers to fail.
                   Application continues emitting events (no-op), ready for reconnection.
        """
        bus = EventBus()

        async def failing_handler1():
            raise ValueError("Error 1")

        async def failing_handler2():
            raise RuntimeError("Error 2")

        async def failing_handler3():
            raise KeyError("Error 3")

        bus.subscribe("test", failing_handler1)
        bus.subscribe("test", failing_handler2)
        bus.subscribe("test", failing_handler3)

        # Should not raise despite all handlers failing
        await bus.emit("test")


class TestEventBusManagement:
    """Test bus management operations: introspection and lifecycle.

    Management API:
        topics() -> list[str]: List all active topics (with subscribers)
        handler_count(topic) -> int: Count handlers for topic
        clear(topic=None): Remove topic or all topics

    Introspection Use Cases:
        - Debugging: Verify expected subscriptions active
        - Monitoring: Track subscription growth/leaks
        - Testing: Assert correct wiring in tests
        - Health checks: Verify required handlers registered

    Lifecycle Management:
        Clear during: Shutdown, test cleanup, dynamic reconfiguration
        Topics auto-created: subscribe() creates topic if not exists
        Topics auto-removed: unsubscribe last handler removes topic
    """

    def test_topics_empty(self):
        """New bus has no topics.

        Tests initial state: freshly created bus has empty topic registry.
        Important for test isolation and clean initialization.

        Pattern:
            Lazy creation: Topics only exist after first subscribe()
            No pre-registration: Dynamic topic creation on demand
            Clean slate: Each EventBus instance independent

        Real-world: Test setup creates new EventBus per test, guaranteeing isolation.
        """
        bus = EventBus()
        assert bus.topics() == []

    def test_topics_after_subscription(self):
        """Topics reflect active subscriptions.

        Tests topic discovery: topics() returns all topics with active handlers.
        Useful for runtime introspection and debugging.

        Pattern:
            Implicit registration: subscribe() auto-creates topic
            Live view: topics() reflects current state
            Multiple topics: Independent namespaces

        Real-world: Health check endpoint lists active topics to verify
                   expected instrumentation is wired up.
        """
        bus = EventBus()

        async def handler():
            pass

        bus.subscribe("topic1", handler)
        bus.subscribe("topic2", handler)

        topics = bus.topics()
        assert sorted(topics) == ["topic1", "topic2"]

    def test_handler_count(self):
        """handler_count returns number of subscribers.

        Tests per-topic handler counting: monitor subscription density.
        Useful for detecting subscription leaks or verifying registration.

        Pattern:
            Per-topic granularity: Count specific to each topic
            Real-time: Reflects current registration state
            Zero for nonexistent: Unsubscribed or never-subscribed returns 0

        Real-world: Monitoring dashboard tracks handler_count over time,
                   alerting on unexpected growth (potential memory leak).
        """
        bus = EventBus()

        async def handler1():
            pass

        async def handler2():
            pass

        assert bus.handler_count("test") == 0

        bus.subscribe("test", handler1)
        assert bus.handler_count("test") == 1

        bus.subscribe("test", handler2)
        assert bus.handler_count("test") == 2

    def test_clear_specific_topic(self):
        """clear() removes specific topic.

        Tests selective cleanup: remove one topic, others remain.
        Useful for dynamic reconfiguration or partial reset.

        Pattern:
            Surgical removal: Target specific topic
            Other topics unaffected: Isolated cleanup
            Immediate effect: Subsequent emits become no-ops

        Real-world: Request-scoped tracing unsubscribes handlers after request,
                   using clear("request.{id}") to batch-remove all request handlers.
        """
        bus = EventBus()

        async def handler():
            pass

        bus.subscribe("topic1", handler)
        bus.subscribe("topic2", handler)

        bus.clear("topic1")

        assert "topic1" not in bus.topics()
        assert "topic2" in bus.topics()

    def test_clear_all(self):
        """clear() with no args removes all topics.

        Tests full reset: remove all subscriptions at once.
        Useful for shutdown, test cleanup, or complete reconfiguration.

        Pattern:
            Nuclear option: clear() with no args = reset to empty state
            Equivalent to fresh EventBus: After clear(), bus.topics() == []
            Idempotent: Safe to call multiple times

        Real-world: Application shutdown calls bus.clear() to release
                   all handler references, allowing garbage collection.
        """
        bus = EventBus()

        async def handler():
            pass

        bus.subscribe("topic1", handler)
        bus.subscribe("topic2", handler)
        bus.subscribe("topic3", handler)

        bus.clear()

        assert bus.topics() == []

    @pytest.mark.asyncio
    async def test_clear_prevents_emission(self):
        """Cleared topic no longer receives events.

        Tests clear() effect on emit(): cleared topics become no-ops.
        Verifies lifecycle interaction between management and dispatch.

        Pattern:
            Immediate cutoff: clear() takes effect instantly
            Past events preserved: Handlers already executed not affected
            Future events dropped: No handlers = no execution

        Mechanics:
            clear() removes topic from internal registry → emit() finds no handlers
            → no-op dispatch (safe, doesn't raise).

        Real-world: Feature flag disables tracing by calling clear("trace.*") topics,
                   instantly stopping trace collection without code changes.
        """
        bus = EventBus()
        received = []

        async def handler(value: int):
            received.append(value)

        bus.subscribe("test", handler)
        await bus.emit("test", value=1)

        bus.clear("test")
        await bus.emit("test", value=2)

        assert received == [1]  # Only received before clear


class TestEventBusIntegration:
    """Test realistic integration scenarios for observability patterns.

    Integration Tests:
        Real-world patterns: Metrics, tracing, audit logging combined
        Multi-topic coordination: Different event types with shared handlers
        Complex argument passing: Rich event payloads

    Patterns Demonstrated:
        - Observable execution: start/finish event pairs
        - Metrics collection: Counters aggregating across events
        - Cross-cutting concerns: Log handler subscribes to all topics
        - Structured payloads: Complex nested data in events

    These tests validate that EventBus supports production observability needs,
    not just simple pub/sub mechanics.
    """

    @pytest.mark.asyncio
    async def test_multiple_topics_multiple_handlers(self):
        """Complex subscription patterns work correctly.

        Tests realistic metrics collection: multiple topics, multiple handlers per topic.
        Demonstrates separation of concerns (counting vs logging).

        Pattern:
            Topic per event type: node.start, node.complete, node.error
            Specialized handlers: count_starts only subscribes to node.start
            Cross-cutting handler: log_all subscribes to all topics

        Architecture:
            Metrics collector: Maintains state (counters), subscribes selectively
            Logger: Stateless, subscribes broadly
            Decoupling: Each handler independent, can be added/removed independently

        Real-world: Prometheus metrics (specialized counters) + structured logging
                   (broad subscription) in production service.
        """
        bus = EventBus()
        metrics = {"starts": 0, "completions": 0, "errors": 0}

        async def count_starts(**kwargs):
            metrics["starts"] += 1

        async def count_completions(**kwargs):
            metrics["completions"] += 1

        async def count_errors(**kwargs):
            metrics["errors"] += 1

        async def log_all(**kwargs):
            pass  # Simulate logging

        # Subscribe to multiple topics
        bus.subscribe("node.start", count_starts)
        bus.subscribe("node.start", log_all)
        bus.subscribe("node.complete", count_completions)
        bus.subscribe("node.complete", log_all)
        bus.subscribe("node.error", count_errors)
        bus.subscribe("node.error", log_all)

        # Emit various events
        await bus.emit("node.start", node_id="n1")
        await bus.emit("node.complete", node_id="n1", duration=0.5)
        await bus.emit("node.start", node_id="n2")
        await bus.emit("node.error", node_id="n2", error="timeout")

        assert metrics["starts"] == 2
        assert metrics["completions"] == 1
        assert metrics["errors"] == 1

    @pytest.mark.asyncio
    async def test_observable_execution_pattern(self):
        """Simulate observable execution with start/finish events.

        Tests distributed tracing pattern: paired events (start/finish) build timeline.
        Foundation for span-based tracing (OpenTelemetry, Jaeger).

        Pattern:
            Event pairs: execution.start → work → execution.finish
            Timeline construction: Handlers maintain ordered event sequence
            Correlation: node_id links start/finish events

        Architecture:
            Trace collector: Subscribes to start/finish, builds spans
            Each event carries context (node_id, result, etc.)
            Sequential emissions create causal ordering

        Real-world: OpenTelemetry trace handler subscribes to execution.start/finish,
                   creates spans with start_time/end_time/result for distributed trace.
        """
        bus = EventBus()
        timeline = []

        async def trace_start(node_id: str):
            timeline.append(f"start:{node_id}")

        async def trace_finish(node_id: str, result: int):
            timeline.append(f"finish:{node_id}:{result}")

        bus.subscribe("execution.start", trace_start)
        bus.subscribe("execution.finish", trace_finish)

        # Simulate execution lifecycle
        await bus.emit("execution.start", node_id="node_1")
        # ... do work ...
        await bus.emit("execution.finish", node_id="node_1", result=42)

        await bus.emit("execution.start", node_id="node_2")
        # ... do work ...
        await bus.emit("execution.finish", node_id="node_2", result=100)

        assert timeline == [
            "start:node_1",
            "finish:node_1:42",
            "start:node_2",
            "finish:node_2:100",
        ]

    @pytest.mark.asyncio
    async def test_handler_with_complex_arguments(self):
        """Handlers receive complex argument structures.

        Tests rich event payloads: nested dicts, multiple arguments, structured data.
        Essential for real-world events carrying context, metadata, domain objects.

        Pattern:
            Structured events: event_type + domain objects + metadata
            Type safety: Handlers declare expected arguments (Python type hints)
            Flexible payloads: **kwargs allows optional fields

        Architecture:
            Event as record: Rich data structure, not just primitives
            Handler signatures: Explicit parameters for type checking/IDE support
            Forward compatibility: **kwargs tolerates extra fields

        Real-world: Audit log handler receives event with user, action, timestamp,
                   before_state, after_state - full context for compliance reporting.
        """
        bus = EventBus()
        received_data = {}

        async def handler(event_type: str, node: dict, metadata: dict):
            received_data["event_type"] = event_type
            received_data["node"] = node
            received_data["metadata"] = metadata

        bus.subscribe("test", handler)

        await bus.emit(
            "test",
            event_type="execution",
            node={"id": "abc", "type": "compute"},
            metadata={"timestamp": 1234, "user": "ocean"},
        )

        assert received_data["event_type"] == "execution"
        assert received_data["node"]["id"] == "abc"
        assert received_data["metadata"]["user"] == "ocean"

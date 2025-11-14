# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Event processing with Flow-based state management and priority queue execution.

Architecture:
    Executor: Flow-based state tracking (EventStatus → Progressions) + Processor coordination
    Processor: Background execution loop with PriorityQueue and capacity control

Design Insight:
    Flow progressions map 1:1 with EventStatus enum values:
        - EventStatus.PENDING → progression "pending"
        - EventStatus.PROCESSING → progression "processing"
        - EventStatus.COMPLETED → progression "completed"
        etc.

    This gives O(1) status queries without scanning all events:
        executor.states.get_progression("completed") → all completed events

Key Benefits:
    - Explainability: Inspect state at any moment
    - Performance: O(1) stage queries, not O(n) pile scans
    - Safety: Referential integrity (Flow validates UUIDs exist)
    - Audit trail: Full state serialization via Flow.to_dict()
    - Type safety: EventStatus enum defines valid progressions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from ..libs import concurrency
from .event import Event, EventStatus
from .flow import Flow
from .pile import Pile
from .progression import Progression

if TYPE_CHECKING:
    from uuid import UUID


__all__ = (
    "Executor",
    "Processor",
)


class Processor:
    """Background event processor with priority queue and capacity control.

    Manages a priority queue of event UUIDs with capacity-limited async processing.
    Events are dequeued by priority (lower values first) and processed in batches
    respecting capacity limits that refresh periodically.

    Design: Queue stores UUID references only, events live in executor's Flow.items.
    This avoids redundancy while preserving type safety.

    After execution, automatically updates Flow progressions to match event status.

    Attributes:
        event_type: ClassVar specifying Event subclass this processor handles
        queue_capacity: Maximum events processed per batch
        capacity_refresh_time: Seconds before capacity reset
        concurrency_limit: Max concurrent event executions
        pile: Reference to executor's Flow.items for fetching events
        executor: Reference to executor for progression updates
    """

    event_type: ClassVar[type[Event]]

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event],
        executor: Executor | None = None,
        concurrency_limit: int | None = None,
    ) -> None:
        """Initialize processor with capacity constraints.

        Args:
            queue_capacity: Max events per batch (must be > 0)
            capacity_refresh_time: Refresh interval in seconds (must be > 0)
            pile: Reference to executor's Flow.items for event storage
            executor: Reference to executor for progression updates (optional)
            concurrency_limit: Max concurrent executions (None = unlimited)

        Raises:
            ValueError: If queue_capacity < 1 or capacity_refresh_time <= 0
        """
        if queue_capacity < 1:
            raise ValueError("Queue capacity must be greater than 0.")
        if capacity_refresh_time <= 0:
            raise ValueError("Capacity refresh time must be larger than 0.")

        self.queue_capacity = queue_capacity
        self.capacity_refresh_time = capacity_refresh_time
        self.pile = pile  # Reference to executor's event storage
        self.executor = executor  # For progression updates

        # Priority queue: items are (priority, event_uuid) tuples
        # Lower priority values are processed first
        # Queue stores UUIDs only, events live in pile
        self.queue: concurrency.PriorityQueue[tuple[float, UUID]] = concurrency.PriorityQueue()

        self._available_capacity = queue_capacity
        self._execution_mode = False
        self._stop_event = concurrency.ConcurrencyEvent()

        if concurrency_limit:
            self._concurrency_sem = concurrency.Semaphore(concurrency_limit)
        else:
            self._concurrency_sem = None

    @property
    def available_capacity(self) -> int:
        """Current capacity available for processing."""
        return self._available_capacity

    @available_capacity.setter
    def available_capacity(self, value: int) -> None:
        self._available_capacity = value

    @property
    def execution_mode(self) -> bool:
        """Whether processor is actively executing events."""
        return self._execution_mode

    @execution_mode.setter
    def execution_mode(self, value: bool) -> None:
        self._execution_mode = value

    async def enqueue(self, event_id: UUID, priority: float | None = None) -> None:
        """Add event UUID to priority queue.

        Args:
            event_id: UUID of event in pile
            priority: Priority value (lower = higher priority).
                     If None, fetches event from pile and uses created_at.
        """
        if priority is None:
            # Default: earlier events have lower priority value (processed first)
            event = self.pile[event_id]
            priority = event.created_at

        await self.queue.put((priority, event_id))

    async def dequeue(self) -> Event:
        """Retrieve highest priority event from queue.

        Returns:
            Event instance fetched from pile (lowest priority value first)
        """
        _, event_id = await self.queue.get()
        return self.pile[event_id]

    async def join(self) -> None:
        """Block until queue is empty and all tasks done.

        Note: PriorityQueue doesn't have task_done/join pattern like asyncio.Queue.
        This waits until queue is empty.
        """
        while not self.queue.empty():
            await concurrency.sleep(0.1)

    async def stop(self) -> None:
        """Signal processor to stop processing events."""
        self._stop_event.set()

    async def start(self) -> None:
        """Clear stop signal, allowing processing to resume."""
        # Create new event since ConcurrencyEvent doesn't have clear()
        if self._stop_event.is_set():
            self._stop_event = concurrency.ConcurrencyEvent()

    def is_stopped(self) -> bool:
        """Check if processor is in stopped state.

        Returns:
            True if signaled to stop
        """
        return self._stop_event.is_set()

    @classmethod
    async def create(
        cls,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event],
        executor: Executor | None = None,
        concurrency_limit: int | None = None,
    ) -> Self:
        """Asynchronously construct new Processor.

        Args:
            queue_capacity: Max events per batch
            capacity_refresh_time: Refresh interval in seconds
            pile: Reference to executor's Flow.items
            executor: Reference to executor for progression updates
            concurrency_limit: Max concurrent executions

        Returns:
            New processor instance
        """
        return cls(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            pile=pile,
            executor=executor,
            concurrency_limit=concurrency_limit,
        )

    async def process(self) -> None:
        """Dequeue and process events up to available capacity.

        Marks events as PROCESSING, invokes them asynchronously via TaskGroup,
        and waits for completion. Updates Flow progressions to match event status
        after execution. Resets capacity afterward if events processed.
        """
        prev_event: Event | None = None
        events_processed = 0

        async with concurrency.create_task_group() as tg:
            while self.available_capacity > 0 and not self.queue.empty():
                next_event = None

                # Wait if previous event still pending
                if prev_event and prev_event.execution.status == EventStatus.PENDING:
                    await concurrency.sleep(self.capacity_refresh_time)
                    next_event = prev_event
                else:
                    next_event = await self.dequeue()

                # Permission check (override for rate limiting, auth, etc.)
                if await self.request_permission(**next_event.request):
                    # Update to PROCESSING status
                    if self.executor:
                        self.executor._update_progression(next_event, EventStatus.PROCESSING)

                    if next_event.streaming:
                        # Streaming: consume async generator
                        async def consume_stream(event):
                            try:
                                async for _ in event.stream():
                                    pass
                                # Update progression after completion
                                if self.executor:
                                    self.executor._update_progression(event)
                            except Exception:
                                # Update progression after failure
                                if self.executor:
                                    self.executor._update_progression(event)

                        if self._concurrency_sem:

                            async def stream_with_sem(event):
                                async with self._concurrency_sem:
                                    await consume_stream(event)

                            tg.start_soon(stream_with_sem, next_event)
                        else:
                            tg.start_soon(consume_stream, next_event)
                    else:
                        # Non-streaming: just invoke
                        async def invoke_and_update(event):
                            try:
                                await event.invoke()
                            finally:
                                # Update progression to match final status
                                if self.executor:
                                    self.executor._update_progression(event)

                        if self._concurrency_sem:

                            async def invoke_with_sem(event):
                                async with self._concurrency_sem:
                                    await invoke_and_update(event)

                            tg.start_soon(invoke_with_sem, next_event)
                        else:
                            tg.start_soon(invoke_and_update, next_event)

                    events_processed += 1

                prev_event = next_event
                self._available_capacity -= 1

        # Reset capacity after batch
        if events_processed > 0:
            self.available_capacity = self.queue_capacity

    async def request_permission(self, **kwargs: Any) -> bool:
        """Determine if event may proceed.

        Override for custom checks (rate limits, permissions, quotas).

        Args:
            **kwargs: Event request parameters

        Returns:
            True if event allowed, False otherwise
        """
        return True

    async def execute(self) -> None:
        """Continuously process events until stop() called.

        Background loop that processes events in batches, sleeping for
        capacity_refresh_time between cycles.
        """
        self.execution_mode = True
        await self.start()

        while not self.is_stopped():
            await self.process()
            await concurrency.sleep(self.capacity_refresh_time)

        self.execution_mode = False


class Executor:
    """Event executor with Flow-based state tracking and background processing.

    Architecture:
        - Flow.items: All events (single source of truth, type-safe storage)
        - Flow.progressions: One progression per EventStatus (state tracking)
        - PriorityQueue (in Processor): Execution order via UUID references
        - Processor: Background execution loop

    Design Insight:
        Flow progressions map 1:1 with EventStatus enum:
            EventStatus.PENDING → progression "pending"
            EventStatus.PROCESSING → progression "processing"
            EventStatus.COMPLETED → progression "completed"
            EventStatus.FAILED → progression "failed"
            EventStatus.CANCELLED → progression "cancelled"
            EventStatus.SKIPPED → progression "skipped"
            EventStatus.ABORTED → progression "aborted"

    Benefits:
        - O(1) status queries: executor.get_events_by_status("completed")
        - Explainability: Inspect state at any moment
        - Audit trail: Full state serialization via states.to_dict()
        - Type safety: EventStatus enum defines valid progressions
        - No redundancy: Queue holds UUIDs, Flow holds events

    Typical usage:
        1. Create events and append to executor (stored in Flow + queued)
        2. Processor executes in background (dequeues UUIDs, fetches from Flow)
        3. Events update their own status during execution
        4. Processor updates Flow progressions to match event status
        5. Query events by status: executor.get_events_by_status("completed")

    Attributes:
        processor_type: ClassVar specifying Processor subclass
        states: Flow with EventStatus-aligned progressions
        processor: Background processor instance
    """

    processor_type: ClassVar[type[Processor]]

    def __init__(
        self,
        processor_config: dict[str, Any] | None = None,
        strict_event_type: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize executor with Flow-based state management.

        Args:
            processor_config: Config dict for creating Processor
            strict_event_type: If True, Flow enforces exact type matching
            name: Optional name for the executor Flow
        """
        self.processor_config = processor_config or {}
        self.processor: Processor | None = None

        # Create Flow with progressions for each EventStatus
        self.states = Flow[Event, Progression](
            name=name or "executor_states",
            item_type=self.processor_type.event_type,
            strict_type=strict_event_type,
        )

        # Create progression for each EventStatus value
        for status in EventStatus:
            self.states.add_progression(Progression(name=status.value))

    @property
    def event_type(self) -> type[Event]:
        """Event subclass handled by processor."""
        return self.processor_type.event_type

    @property
    def strict_event_type(self) -> bool:
        """Whether Flow enforces exact event type matching."""
        return self.states.items.strict_type

    def _update_progression(self, event: Event, force_status: EventStatus | None = None) -> None:
        """Update Flow progression to match event's execution status.

        Args:
            event: Event to update
            force_status: Override event.execution.status (for PROCESSING transition)
        """
        # Determine target status
        target_status = force_status if force_status else event.execution.status

        # Remove from all progressions (event moves between states)
        for prog in self.states.progressions:
            if event.id in prog:
                prog.remove(event.id)

        # Add to progression matching current status
        status_prog = self.states.get_progression(target_status.value)
        status_prog.append(event.id)

    async def forward(self) -> None:
        """Process any queued events immediately.

        Triggers processor.process() to handle queued events.
        Note: Events are queued automatically when appended.
        """
        if self.processor:
            await self.processor.process()

    async def start(self) -> None:
        """Initialize and start processor if not created."""
        if not self.processor:
            await self._create_processor()
        await self.processor.start()

    async def stop(self) -> None:
        """Stop processor if exists."""
        if self.processor:
            await self.processor.stop()

    async def _create_processor(self) -> None:
        """Instantiate processor using stored config + Flow.items + executor reference."""
        self.processor = await self.processor_type.create(
            pile=self.states.items,  # Pass Flow.items as pile reference
            executor=self,  # Pass self for progression updates
            **self.processor_config,
        )

    async def append(self, event: Event, priority: float | None = None) -> None:
        """Add event to Flow and enqueue for processing.

        Event starts in "pending" progression, matching EventStatus.PENDING.

        Args:
            event: Event to add
            priority: Optional priority (lower = higher priority).
                     If None, uses event.created_at.
        """
        # Add to Flow items + "pending" progression (initial state)
        self.states.add_item(event, progressions="pending")

        # Enqueue UUID reference for processing
        if self.processor:
            await self.processor.enqueue(event.id, priority=priority)

    def get_events_by_status(self, status: EventStatus | str) -> list[Event]:
        """Get all events with given status.

        O(1) progression lookup + O(k) iteration where k = events with this status.

        Args:
            status: EventStatus enum or status string ("completed", "failed", etc.)

        Returns:
            List of events in this status
        """
        status_str = status.value if isinstance(status, EventStatus) else status
        prog = self.states.get_progression(status_str)
        return [self.states.items[uid] for uid in prog]

    @property
    def completed_events(self) -> list[Event]:
        """All events with COMPLETED status (O(1) lookup)."""
        return self.get_events_by_status(EventStatus.COMPLETED)

    @property
    def pending_events(self) -> list[Event]:
        """All events with PENDING status (O(1) lookup)."""
        return self.get_events_by_status(EventStatus.PENDING)

    @property
    def failed_events(self) -> list[Event]:
        """All events with FAILED status (O(1) lookup)."""
        return self.get_events_by_status(EventStatus.FAILED)

    @property
    def processing_events(self) -> list[Event]:
        """All events with PROCESSING status (O(1) lookup)."""
        return self.get_events_by_status(EventStatus.PROCESSING)

    def status_counts(self) -> dict[str, int]:
        """Get event count per status (O(|EventStatus|) = O(7)).

        Returns:
            Dict mapping status name to count
        """
        return {prog.name: len(prog) for prog in self.states.progressions}

    def inspect_state(self) -> str:
        """Debug helper: show counts per status.

        Returns:
            Formatted string with status counts
        """
        lines = [f"Executor State ({self.states.name}):"]
        for status in EventStatus:
            count = len(self.states.get_progression(status.value))
            lines.append(f"  {status.value}: {count} events")
        return "\n".join(lines)

    def __contains__(self, event: Event | UUID) -> bool:
        """Check if event is in executor's Flow.

        Args:
            event: Event instance or UUID

        Returns:
            True if event in Flow.items
        """
        return event in self.states.items

    def __repr__(self) -> str:
        """String representation with status counts."""
        counts = self.status_counts()
        total = sum(counts.values())
        return f"Executor(total={total}, {', '.join(f'{k}={v}' for k, v in counts.items())})"

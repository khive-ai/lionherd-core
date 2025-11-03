# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import Field, field_serializer

from ..protocols import Invocable, Serializable, implements
from ..types import MaybeSentinel, MaybeUnset, Unset, is_sentinel
from .element import LN_ELEMENT_FIELDS, Element

__all__ = (
    "Event",
    "EventStatus",
    "Execution",
)


class EventStatus(str, Enum):
    """Event execution status states.

    Values:
        PENDING: Not yet started
        PROCESSING: Currently executing
        COMPLETED: Finished successfully
        FAILED: Execution failed with error
        CANCELLED: Interrupted by timeout or cancellation
        SKIPPED: Bypassed due to condition
        ABORTED: Pre-validation rejected, never started
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    ABORTED = "aborted"


@implements(Serializable)
@dataclass(slots=True)
class Execution:
    """Execution state tracking for Events.

    Attributes:
        status: Current execution status
        duration: Elapsed time in seconds (Unset until complete)
        response: Execution result (Unset/value/None)
        error: Exception if failed (Unset/None/BaseException for proper exception hierarchy)
        retryable: Whether retry is safe (Unset/bool)
    """

    status: EventStatus = EventStatus.PENDING
    duration: MaybeUnset[float] = Unset
    response: MaybeSentinel[Any] = Unset
    error: MaybeUnset[BaseException] | None = Unset
    retryable: MaybeUnset[bool] = Unset

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict with sentinel handling."""
        from ._utils import get_json_serializable

        # Serialize response: Unset → None, else serialize value
        if is_sentinel(self.response):
            res_ = None  # Unset: no response value
        else:
            # Try to serialize the actual response
            res_ = get_json_serializable(self.response)
            if res_ is Unset:
                res_ = "<unserializable>"

        # Serialize error
        error_dict = None
        if self.error is not Unset and self.error is not None:
            from lionherd_core.errors import LionherdError

            if isinstance(self.error, LionherdError):
                error_dict = self.error.to_dict()  # rich structured info
            elif isinstance(self.error, ExceptionGroup):
                error_dict = self._serialize_exception_group(self.error)  # serialize all exceptions
            else:
                error_dict = {
                    "error": type(self.error).__name__,
                    "message": str(self.error),
                }

        # Convert sentinels to None for JSON serialization
        duration_value = None if self.duration is Unset else self.duration
        retryable_value = None if self.retryable is Unset else self.retryable

        return {
            "status": self.status.value,
            "duration": duration_value,
            "response": res_,
            "error": error_dict,
            "retryable": retryable_value,
        }

    def _serialize_exception_group(self, eg: ExceptionGroup) -> dict[str, Any]:
        """Recursively serialize ExceptionGroup and nested exceptions."""
        from lionherd_core.errors import LionherdError

        exceptions = []
        for exc in eg.exceptions:
            if isinstance(exc, LionherdError):
                exceptions.append(exc.to_dict())
            elif isinstance(exc, ExceptionGroup):
                exceptions.append(self._serialize_exception_group(exc))
            else:
                exceptions.append(
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )

        return {
            "error": type(eg).__name__,
            "message": str(eg),
            "exceptions": exceptions,
        }

    def add_error(self, exc: BaseException) -> None:
        """Add error to execution. Creates ExceptionGroup if multiple errors."""
        if self.error is Unset or self.error is None:
            self.error = exc
        elif isinstance(self.error, ExceptionGroup):
            # Already have group - extend it
            self.error = ExceptionGroup(  # type: ignore[type-var]
                "multiple errors",
                [*self.error.exceptions, exc],
            )
        else:
            self.error = ExceptionGroup(  # type: ignore[type-var]
                "multiple errors",
                [self.error, exc],
            )


@implements(Invocable)
class Event(Element):
    """Base event with lifecycle tracking and execution.

    Subclasses implement _invoke(). invoke() manages status transitions, timing, error capture.
    Supports ExceptionGroup for multi-error scenarios.

    Attributes:
        execution: Execution state
        status: Property for execution.status
        response: Property for execution.response (read-only)
    """

    execution: Execution = Field(default_factory=Execution)

    @field_serializer("execution")
    def _serialize_execution(self, val: Execution) -> dict:
        """Serialize Execution to dict."""
        return val.to_dict()

    @property
    def request(self) -> dict:
        """Get request info."""
        return {}

    @property
    def status(self) -> EventStatus:
        """Get execution status."""
        return self.execution.status

    @status.setter
    def status(self, val: EventStatus | str) -> None:
        """Set execution status."""
        if isinstance(val, str):
            val = EventStatus(val)
        elif not isinstance(val, EventStatus):
            raise ValueError(f"Invalid status type: {type(val).__name__}")
        self.execution.status = val

    @property
    def response(self) -> Any:
        """Get execution response (read-only)."""
        return self.execution.response

    async def _invoke(self) -> Any:
        """Execute event. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _invoke()")

    async def invoke(self) -> Any:
        """Execute with status tracking, timing, error capture. Returns result or None (check status)."""
        from lionherd_core.libs.concurrency import current_time

        start = current_time()

        try:
            self.execution.status = EventStatus.PROCESSING
            result = await self._invoke()

            # Success path: set response and clear error
            self.execution.response = result  # Can be None or any value
            self.execution.error = None  # Explicitly no error
            self.execution.status = EventStatus.COMPLETED
            self.execution.retryable = False  # Success - no need to retry
            return result

        except Exception as e:
            # Catch all regular exceptions - execution state is the API
            # Handle ExceptionGroup specially (Python 3.11+ async task groups)
            from lionherd_core.errors import LionherdError

            if isinstance(e, ExceptionGroup):
                # ExceptionGroup: ALL exceptions must be retryable for group to be retryable
                # If even ONE exception is non-retryable, don't retry the whole group
                retryable = True  # Start optimistic
                for exc in e.exceptions:
                    if isinstance(exc, LionherdError) and not exc.retryable:
                        retryable = False
                        break
                    # Unknown exceptions keep default (True)

                self.execution.retryable = retryable
            else:
                # Single exception: use standard logic
                if isinstance(e, LionherdError):
                    # Use retryable flag from our error hierarchy
                    self.execution.retryable = e.retryable
                else:
                    # Unknown exceptions are retryable by default (safe assumption)
                    self.execution.retryable = True

            # Failure path: response is Unset (execution failed, no valid response)
            self.execution.response = Unset
            self.execution.error = e  # Store exception (single or ExceptionGroup)
            self.execution.status = EventStatus.FAILED
            return None  # Return None on failure, caller checks status

        except BaseException as e:
            # Catch cancellation signals (CancelledError, KeyboardInterrupt, etc.)
            # These are BaseException subclasses that are NOT Exception subclasses
            from lionherd_core.libs.concurrency import get_cancelled_exc_class

            if isinstance(e, get_cancelled_exc_class()):
                # CancelledError from anyio - set state and propagate
                self.execution.response = Unset  # Cancelled before completion, no response
                self.execution.error = e  # Store the cancellation exception
                self.execution.status = EventStatus.CANCELLED
                self.execution.retryable = True  # Cancellations (esp. timeouts) are retryable

            # Always propagate BaseException (cancellation, KeyboardInterrupt, etc.)
            raise

        finally:
            self.execution.duration = current_time() - start

    async def stream(self) -> Any:
        """Stream execution. Override if supported."""
        raise NotImplementedError("Subclasses must implement stream() if streaming=True")

    def as_fresh_event(self, copy_meta: bool = False) -> Event:
        """Create pristine clone with reset execution, fresh ID, PENDING status."""
        # Get dict representation and remove execution state
        d_ = self.to_dict()
        for key in ["execution", *LN_ELEMENT_FIELDS]:
            d_.pop(key, None)

        # Create fresh instance with same configuration
        fresh = self.__class__(**d_)

        # Optionally copy metadata
        if copy_meta and hasattr(self, "metadata"):
            fresh.metadata = self.metadata.copy()

        # Track original event in metadata
        if hasattr(fresh, "metadata"):
            fresh.metadata["original"] = {
                "id": str(self.id),
                "created_at": self.created_at,
            }

        return fresh

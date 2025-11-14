# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, final

from pydantic import Field, field_serializer, field_validator

from ..errors import TimeoutError as LionherdTimeoutError
from ..libs.concurrency import Lock
from ..protocols import Invocable, Serializable, implements
from ..types import Enum, MaybeSentinel, MaybeUnset, Unset, is_sentinel
from ._utils import async_synchronized
from .element import LN_ELEMENT_FIELDS, Element

__all__ = (
    "Event",
    "EventStatus",
    "Execution",
)


class EventStatus(Enum):
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
    """Execution state (status, duration, response, error, retryable).

    Attributes:
        status: Current execution status
        duration: Elapsed time in seconds (Unset until complete)
        response: Result (Unset if unavailable, None if legitimate null)
        error: Exception if failed (Unset/None/BaseException)
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

        if is_sentinel(self.response):
            res_ = None
        else:
            res_ = get_json_serializable(self.response)
            if res_ is Unset:
                res_ = "<unserializable>"

        error_dict = None
        if self.error is not Unset and self.error is not None:
            from lionherd_core.errors import LionherdError

            if isinstance(self.error, LionherdError):
                error_dict = self.error.to_dict()
            elif isinstance(self.error, ExceptionGroup):
                error_dict = self._serialize_exception_group(self.error)
            else:
                error_dict = {
                    "error": type(self.error).__name__,
                    "message": str(self.error),
                }

        duration_value = None if self.duration is Unset else self.duration
        retryable_value = None if self.retryable is Unset else self.retryable

        return {
            "status": self.status.value,
            "duration": duration_value,
            "response": res_,
            "error": error_dict,
            "retryable": retryable_value,
        }

    def _serialize_exception_group(
        self,
        eg: ExceptionGroup,
        depth: int = 0,
        _seen: set[int] | None = None,
    ) -> dict[str, Any]:
        """Recursively serialize ExceptionGroup with depth limit and cycle detection.

        Args:
            eg: ExceptionGroup to serialize
            depth: Current recursion depth (internal)
            _seen: Set of seen exception IDs for cycle detection (internal)

        Returns:
            Serialized exception group dict

        Note:
            Maximum depth is 100 to prevent stack overflow.
            Circular references are detected and handled gracefully.
        """
        from lionherd_core.errors import LionherdError

        # Depth limit to prevent stack overflow
        MAX_DEPTH = 100
        if depth > MAX_DEPTH:
            return {
                "error": "ExceptionGroup",
                "message": f"Max nesting depth ({MAX_DEPTH}) exceeded",
                "nested_count": len(eg.exceptions) if hasattr(eg, "exceptions") else 0,
            }

        # Initialize cycle detection set on first call
        if _seen is None:
            _seen = set()

        # Cycle detection using object id tracking
        eg_id = id(eg)
        if eg_id in _seen:
            return {
                "error": "ExceptionGroup",
                "message": "Circular reference detected",
            }

        _seen.add(eg_id)

        try:
            exceptions = []
            for exc in eg.exceptions:
                if isinstance(exc, LionherdError):
                    exceptions.append(exc.to_dict())
                elif isinstance(exc, ExceptionGroup):
                    exceptions.append(self._serialize_exception_group(exc, depth + 1, _seen))
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
        finally:
            # Cleanup seen set for this exception group
            _seen.discard(eg_id)

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
    """Base event with lifecycle tracking and execution state.

    Subclasses implement _invoke(). invoke() manages transitions, timing, errors.

    Attributes:
        execution: Execution state
        timeout: Optional timeout in seconds (None = no timeout)
    """

    execution: Execution = Field(default_factory=Execution)
    timeout: float | None = Field(None, exclude=True)
    streaming: bool = Field(False, exclude=True)

    def model_post_init(self, __context) -> None:
        """Initialize async lock for thread-safe invoke()."""
        super().model_post_init(__context)
        self._async_lock = Lock()

    @field_validator("timeout")
    @classmethod
    def _validate_timeout(cls, v: float | None) -> float | None:
        """Validate timeout is positive and finite (raises ValueError if not)."""
        if v is not None:
            if not math.isfinite(v):
                raise ValueError(f"timeout must be finite, got {v}")
            if v <= 0:
                raise ValueError(f"timeout must be positive, got {v}")
        return v

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

    @final
    @async_synchronized
    async def invoke(self) -> Any:
        """Execute with status tracking, timing, error capture (idempotent).

        **Idempotency**: Multiple concurrent calls execute _invoke() exactly once.
        Subsequent calls return cached result without re-execution. Once COMPLETED
        or FAILED, invoke() will return the cached response.

        **Retry Pattern**: To retry after FAILED status, use `as_fresh_event()` to
        create a new Event with reset execution state. Direct invoke() calls on
        completed events always return cached results.

        Returns:
            Execution result (same for all concurrent callers)
        """
        from lionherd_core.libs.concurrency import current_time

        # Idempotency: Return cached result if already executed
        if self.execution.status != EventStatus.PENDING:
            return self.execution.response

        start = current_time()

        try:
            self.execution.status = EventStatus.PROCESSING

            if self.timeout is not None:
                from lionherd_core.libs.concurrency import fail_after

                with fail_after(self.timeout):
                    result = await self._invoke()
            else:
                result = await self._invoke()

            # Success path
            self.execution.response = result
            self.execution.error = None
            self.execution.status = EventStatus.COMPLETED
            self.execution.retryable = False
            return result

        except TimeoutError:
            lionherd_timeout = LionherdTimeoutError(
                f"Operation timed out after {self.timeout}s",
                retryable=True,
            )

            self.execution.response = Unset
            self.execution.error = lionherd_timeout
            self.execution.status = EventStatus.CANCELLED
            self.execution.retryable = lionherd_timeout.retryable
            return None

        except Exception as e:
            from lionherd_core.errors import LionherdError

            if isinstance(e, ExceptionGroup):
                # All exceptions must be retryable for group to be retryable
                retryable = True
                for exc in e.exceptions:
                    if isinstance(exc, LionherdError) and not exc.retryable:
                        retryable = False
                        break

                self.execution.retryable = retryable
            else:
                if isinstance(e, LionherdError):
                    self.execution.retryable = e.retryable
                else:
                    self.execution.retryable = True

            self.execution.response = Unset
            self.execution.error = e
            self.execution.status = EventStatus.FAILED
            return None

        except BaseException as e:
            from lionherd_core.libs.concurrency import get_cancelled_exc_class

            if isinstance(e, get_cancelled_exc_class()):
                self.execution.response = Unset
                self.execution.error = e
                self.execution.status = EventStatus.CANCELLED
                self.execution.retryable = True

            raise

        finally:
            self.execution.duration = current_time() - start

    async def stream(self) -> Any:
        """Stream execution. Override if supported."""
        raise NotImplementedError("Subclasses must implement stream() if streaming=True")

    def as_fresh_event(self, copy_meta: bool = False) -> Event:
        """Clone with reset execution (fresh ID, PENDING status)."""
        d_ = self.to_dict()
        for key in ["execution", *LN_ELEMENT_FIELDS]:
            d_.pop(key, None)

        fresh = self.__class__(**d_)

        if hasattr(self, "timeout") and self.timeout is not None:
            fresh.timeout = self.timeout

        if copy_meta and hasattr(self, "metadata"):
            fresh.metadata = self.metadata.copy()

        if hasattr(fresh, "metadata"):
            fresh.metadata["original"] = {
                "id": str(self.id),
                "created_at": self.created_at,
            }

        return fresh

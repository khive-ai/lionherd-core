# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .protocols import Serializable, implements

__all__ = (
    "ConfigurationError",
    "ConnectionError",
    "ExecutionError",
    "ExistsError",
    "LionherdError",
    "NotFoundError",
    "QueueFullError",
    "TimeoutError",
    "ValidationError",
)


@implements(Serializable)
class LionherdError(Exception):
    """Base exception for all lionherd errors.

    Attributes:
        message: Human-readable error message
        details: Additional structured context
        retryable: Whether this error can be retried
    """

    default_message: str = "Lionherd error"
    default_retryable: bool = True

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
        cause: Exception | None = None,
    ):
        """Initialize error.

        Args:
            message: Error message (uses default_message if None)
            details: Additional context dict
            retryable: Whether error can be retried (uses default_retryable if None)
            cause: Original exception that caused this error
        """
        self.message = message or self.default_message
        self.details = details or {}
        self.retryable = retryable if retryable is not None else self.default_retryable

        if cause:
            self.__cause__ = cause  # Preserve traceback

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize error to dict for logging/debugging.

        Returns:
            Dict with error type, message, details, retryable flag
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "retryable": self.retryable,
            **({"details": self.details} if self.details else {}),
        }


class ValidationError(LionherdError):
    """Validation failure. Not retryable."""

    default_message = "Validation failed"
    default_retryable = False  # Validation errors won't fix themselves


class ConfigurationError(LionherdError):
    """Configuration error. Not retryable."""

    default_message = "Configuration error"
    default_retryable = False  # Config errors need manual fixes


class ExecutionError(LionherdError):
    """Event/Calling execution failure. Retryable by default."""

    default_message = "Execution failed"
    default_retryable = True  # Most execution failures are transient


class ConnectionError(LionherdError):
    """Connection/network failure. Retryable by default."""

    default_message = "Connection error"
    default_retryable = True  # Network issues are often transient


class TimeoutError(LionherdError):
    """Operation timeout. Retryable by default."""

    default_message = "Operation timed out"
    default_retryable = True  # Timeouts might succeed with more time


class NotFoundError(LionherdError):
    """Item not found. Not retryable."""

    default_message = "Item not found"
    default_retryable = False  # Missing items won't appear on retry


class ExistsError(LionherdError):
    """Item already exists. Not retryable."""

    default_message = "Item already exists"
    default_retryable = False  # Duplicate items won't resolve on retry


class QueueFullError(LionherdError):
    """Queue capacity exceeded. Retryable."""

    default_message = "Queue is full"
    default_retryable = True  # Queue might have space later

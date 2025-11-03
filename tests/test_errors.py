# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Tests for lionherd exception hierarchy.

Error hierarchy features:
- Retry-aware exceptions (retryable flag for automatic retry strategies)
- Structured context (details dict for machine-readable debugging)
- Cause chain preservation (wraps original exceptions)
- Serialization (to_dict() for logging/metrics)

Error types:
- LionherdError: Base (retryable)
- ValidationError: Input validation (NOT retryable)
- ConfigurationError: Config issues (NOT retryable)
- ExecutionError: Runtime failures (retryable)
- ConnectionError: Network/API failures (retryable)
- TimeoutError: Operation timeouts (retryable)
"""

import pytest

from lionherd_core.errors import (
    ConfigurationError,
    ConnectionError,
    ExecutionError,
    LionherdError,
    TimeoutError,
    ValidationError,
)


class TestLionherdErrorBase:
    """Test base LionherdError functionality."""

    def test_create_with_default_message(self):
        """LionherdError should use default message when none provided."""
        err = LionherdError()
        assert err.message == "Lionherd error"
        assert str(err) == "Lionherd error"

    def test_create_with_custom_message(self):
        """LionherdError should accept custom message."""
        err = LionherdError("Custom error message")
        assert err.message == "Custom error message"
        assert str(err) == "Custom error message"

    def test_default_retryable(self):
        """LionherdError should be retryable by default."""
        err = LionherdError()
        assert err.retryable is True

    def test_custom_retryable(self):
        """Can override default retryable flag."""
        err = LionherdError(retryable=False)
        assert err.retryable is False

    def test_empty_details_by_default(self):
        """details should default to empty dict."""
        err = LionherdError()
        assert err.details == {}

    def test_custom_details(self):
        """Can provide structured error context via details."""
        details = {"component": "parser", "line": 42, "file": "test.py"}
        err = LionherdError(details=details)
        assert err.details == details

    def test_cause_preservation(self):
        """Error should preserve cause chain."""
        original = ValueError("Original error")
        err = LionherdError("Wrapped error", cause=original)

        assert err.__cause__ is original

    def test_to_dict_basic(self):
        """to_dict should serialize error information."""
        err = LionherdError("Test error")
        data = err.to_dict()

        assert data["error"] == "LionherdError"
        assert data["message"] == "Test error"
        assert data["retryable"] is True

    def test_to_dict_with_details(self):
        """to_dict should include details when present."""
        err = LionherdError("Test", details={"key": "value"})
        data = err.to_dict()

        assert "details" in data
        assert data["details"] == {"key": "value"}

    def test_to_dict_without_details_excludes_key(self):
        """to_dict should omit details key when empty."""
        err = LionherdError("Test")
        data = err.to_dict()

        assert "details" not in data


class TestValidationError:
    """Test ValidationError specifics."""

    def test_default_message(self):
        """ValidationError should have appropriate default message."""
        err = ValidationError()
        assert err.message == "Validation failed"

    def test_not_retryable_by_default(self):
        """ValidationError should NOT be retryable (data won't fix itself)."""
        err = ValidationError()
        assert err.retryable is False

    def test_can_override_retryable(self):
        """Can override retryable if needed (edge cases)."""
        err = ValidationError(retryable=True)
        assert err.retryable is True

    def test_to_dict_type_name(self):
        """to_dict should report correct error type."""
        err = ValidationError("Invalid input")
        data = err.to_dict()
        assert data["error"] == "ValidationError"


class TestConfigurationError:
    """Test ConfigurationError specifics."""

    def test_default_message(self):
        """ConfigurationError should have appropriate default message."""
        err = ConfigurationError()
        assert err.message == "Configuration error"

    def test_not_retryable_by_default(self):
        """ConfigurationError should NOT be retryable (needs manual fix)."""
        err = ConfigurationError()
        assert err.retryable is False

    def test_with_details(self):
        """ConfigurationError can include config context."""
        err = ConfigurationError(
            "Invalid API key",
            details={"config_file": ".env", "key": "ANTHROPIC_API_KEY"},
        )
        assert err.details["config_file"] == ".env"

    def test_to_dict_type_name(self):
        """to_dict should report correct error type."""
        err = ConfigurationError("Bad config")
        data = err.to_dict()
        assert data["error"] == "ConfigurationError"


class TestExecutionError:
    """Test ExecutionError specifics."""

    def test_default_message(self):
        """ExecutionError should have appropriate default message."""
        err = ExecutionError()
        assert err.message == "Execution failed"

    def test_retryable_by_default(self):
        """ExecutionError should be retryable (transient failures common)."""
        err = ExecutionError()
        assert err.retryable is True

    def test_can_override_retryable(self):
        """Can mark ExecutionError as non-retryable for critical failures."""
        err = ExecutionError("Critical failure", retryable=False)
        assert err.retryable is False

    def test_with_execution_context(self):
        """ExecutionError can include execution context."""
        err = ExecutionError(
            "Tool failed",
            details={
                "tool": "web_search",
                "query": "test query",
                "attempt": 3,
            },
        )
        assert err.details["tool"] == "web_search"
        assert err.details["attempt"] == 3

    def test_to_dict_type_name(self):
        """to_dict should report correct error type."""
        err = ExecutionError("Execution failed")
        data = err.to_dict()
        assert data["error"] == "ExecutionError"


class TestConnectionError:
    """Test ConnectionError specifics."""

    def test_default_message(self):
        """ConnectionError should have appropriate default message."""
        err = ConnectionError()
        assert err.message == "Connection error"

    def test_retryable_by_default(self):
        """ConnectionError should be retryable (network is unreliable)."""
        err = ConnectionError()
        assert err.retryable is True

    def test_with_network_context(self):
        """ConnectionError can include network context."""
        err = ConnectionError(
            "API unreachable",
            details={
                "host": "api.anthropic.com",
                "status_code": 503,
                "retry_after": 60,
            },
        )
        assert err.details["host"] == "api.anthropic.com"
        assert err.details["status_code"] == 503

    def test_to_dict_type_name(self):
        """to_dict should report correct error type."""
        err = ConnectionError("Network error")
        data = err.to_dict()
        assert data["error"] == "ConnectionError"


class TestTimeoutError:
    """Test TimeoutError specifics."""

    def test_default_message(self):
        """TimeoutError should have appropriate default message."""
        err = TimeoutError()
        assert err.message == "Operation timed out"

    def test_retryable_by_default(self):
        """TimeoutError should be retryable (might succeed with more time)."""
        err = TimeoutError()
        assert err.retryable is True

    def test_with_timeout_context(self):
        """TimeoutError can include timeout details."""
        err = TimeoutError(
            "Request timeout",
            details={
                "timeout_seconds": 30,
                "operation": "model_inference",
                "elapsed": 30.5,
            },
        )
        assert err.details["timeout_seconds"] == 30
        assert err.details["elapsed"] == 30.5

    def test_to_dict_type_name(self):
        """to_dict should report correct error type."""
        err = TimeoutError("Timed out")
        data = err.to_dict()
        assert data["error"] == "TimeoutError"


class TestErrorHierarchy:
    """Test error hierarchy and inheritance."""

    def test_all_errors_inherit_from_lionherd_error(self):
        """All error types should inherit from LionherdError."""
        assert issubclass(ValidationError, LionherdError)
        assert issubclass(ConfigurationError, LionherdError)
        assert issubclass(ExecutionError, LionherdError)
        assert issubclass(ConnectionError, LionherdError)
        assert issubclass(TimeoutError, LionherdError)

    def test_catch_base_class(self):
        """Can catch all errors via LionherdError."""
        try:
            raise ExecutionError("Test")
        except LionherdError as e:
            assert isinstance(e, ExecutionError)
            assert e.message == "Test"

    def test_catch_specific_type(self):
        """Can catch specific error types."""
        try:
            raise ValidationError("Invalid")
        except ValidationError as e:
            assert e.message == "Invalid"
            assert not e.retryable


class TestErrorDesignPatterns:
    """Design documentation tests for error usage patterns."""

    def test_retry_logic_design(self):
        """
        Design Pattern: Automatic retry based on retryable flag.

        Demonstrates how error types guide retry behavior without
        manual classification.
        """
        # Retryable errors (network, execution, timeout)
        retryable_errors = [
            ExecutionError("Transient failure"),
            ConnectionError("Network blip"),
            TimeoutError("Slow response"),
        ]

        for err in retryable_errors:
            assert err.retryable is True, f"{type(err).__name__} should be retryable"

        # Non-retryable errors (validation, configuration)
        non_retryable_errors = [
            ValidationError("Bad input"),
            ConfigurationError("Missing API key"),
        ]

        for err in non_retryable_errors:
            assert err.retryable is False, f"{type(err).__name__} should NOT be retryable"

    def test_structured_logging_design(self):
        """
        Design Pattern: Structured error context for logging.

        Demonstrates how details dict enables machine-readable
        error logging and monitoring.
        """
        err = ExecutionError(
            "Model inference failed",
            details={
                "model": "claude-3-opus",
                "tokens": 1500,
                "provider": "anthropic",
                "status_code": 429,
                "request_id": "req_abc123",
            },
        )

        # Serialized form suitable for structured logging
        log_data = err.to_dict()

        assert log_data["error"] == "ExecutionError"
        assert log_data["retryable"] is True
        assert log_data["details"]["model"] == "claude-3-opus"
        assert log_data["details"]["status_code"] == 429

        # Can extract metrics from error details
        assert "request_id" in log_data["details"]

    def test_cause_chain_design(self):
        """
        Design Pattern: Preserve root cause for debugging.

        Demonstrates how cause chain maintains full traceback
        while providing domain-specific error types.
        """
        # Simulate network error from underlying library
        original = Exception("Connection refused")

        # Wrap in domain-specific error
        wrapped = ConnectionError(
            "Failed to reach API",
            details={"endpoint": "/v1/messages"},
            cause=original,
        )

        # Cause chain preserved
        assert wrapped.__cause__ is original

        # Can access both domain error and root cause
        assert wrapped.message == "Failed to reach API"
        assert str(wrapped.__cause__) == "Connection refused"

    def test_override_defaults_design(self):
        """
        Design Pattern: Override defaults for edge cases.

        Demonstrates flexibility to override default retryable
        behavior when needed.
        """
        # Normally retryable, but critical failure
        critical = ExecutionError(
            "Unrecoverable state corruption",
            retryable=False,  # Override default
        )
        assert critical.retryable is False

        # Normally not retryable, but special case
        special_validation = ValidationError(
            "Temporary schema mismatch",
            retryable=True,  # Override default
        )
        assert special_validation.retryable is True

    def test_error_categorization_design(self):
        """
        Design Pattern: Error types as semantic categories.

        Demonstrates how error types encode semantic meaning
        beyond just retryability.
        """
        # Input validation - client's responsibility
        validation_err = ValidationError(
            "Invalid message format",
            details={"field": "content", "reason": "empty"},
        )

        # Configuration issue - deployment problem
        config_err = ConfigurationError(
            "Missing API credentials",
            details={"env_var": "ANTHROPIC_API_KEY"},
        )

        # Runtime failure - transient or bug
        execution_err = ExecutionError(
            "Tool execution failed",
            details={"tool": "web_search"},
        )

        # Network issue - infrastructure problem
        connection_err = ConnectionError(
            "API unreachable",
            details={"host": "api.anthropic.com"},
        )

        # Timeout - resource constraint
        timeout_err = TimeoutError(
            "Request timeout",
            details={"timeout_ms": 30000},
        )

        # Each type communicates different failure mode
        errors = [
            validation_err,
            config_err,
            execution_err,
            connection_err,
            timeout_err,
        ]

        # All have consistent interface
        for err in errors:
            assert hasattr(err, "message")
            assert hasattr(err, "retryable")
            assert hasattr(err, "details")
            assert callable(err.to_dict)

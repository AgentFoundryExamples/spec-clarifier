# Copyright 2025 John Brosnihan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for logging helper utilities."""

import json
import logging
import uuid
from uuid import UUID

from app.utils.logging_helper import (
    get_correlation_id,
    log_error,
    log_info,
    log_structured,
    log_warning,
    redact_sensitive_data,
)


class TestGetCorrelationId:
    """Tests for get_correlation_id function."""

    def test_returns_job_id_when_provided(self):
        """Test that job_id is returned as correlation ID."""
        job_id = uuid.uuid4()
        correlation_id = get_correlation_id(job_id)

        assert correlation_id == str(job_id)

    def test_generates_new_uuid_when_no_job_id(self):
        """Test that a new UUID is generated when no job_id provided."""
        correlation_id = get_correlation_id()

        # Should be a valid UUID string
        assert UUID(correlation_id)

    def test_different_uuids_generated(self):
        """Test that different UUIDs are generated on multiple calls."""
        id1 = get_correlation_id()
        id2 = get_correlation_id()

        assert id1 != id2


class TestRedactSensitiveData:
    """Tests for redact_sensitive_data function."""

    def test_redacts_api_key(self):
        """Test that API keys are redacted."""
        message = "Error with api_key=sk-abc123xyz"
        redacted = redact_sensitive_data(message)

        assert "sk-abc123xyz" not in redacted
        assert "[REDACTED]" in redacted

    def test_redacts_bearer_token(self):
        """Test that bearer tokens are redacted."""
        message = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = redact_sensitive_data(message)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "[REDACTED]" in redacted

    def test_redacts_json_api_key(self):
        """Test that JSON-formatted API keys are redacted."""
        message = '{"api_key": "sk-12345"}'
        redacted = redact_sensitive_data(message)

        assert "sk-12345" not in redacted
        assert "[REDACTED]" in redacted

    def test_redacts_prompts(self):
        """Test that prompts are redacted."""
        message = "prompt: You are a helpful assistant"
        redacted = redact_sensitive_data(message)

        assert "You are a helpful assistant" not in redacted
        assert "[REDACTED]" in redacted

    def test_preserves_non_sensitive_content(self):
        """Test that non-sensitive content is preserved."""
        message = "Processing job 12345 with model gpt-4"
        redacted = redact_sensitive_data(message)

        assert redacted == message

    def test_handles_empty_string(self):
        """Test that empty strings are handled."""
        message = ""
        redacted = redact_sensitive_data(message)

        assert redacted == ""


class TestLogStructured:
    """Tests for log_structured function."""

    def test_logs_with_job_id(self, caplog):
        """Test that structured logs include job_id."""
        logger = logging.getLogger("test")
        job_id = uuid.uuid4()

        with caplog.at_level(logging.INFO):
            log_structured(
                logger,
                logging.INFO,
                "test_event",
                job_id=job_id,
                custom_field="value"
            )

        assert len(caplog.records) == 1
        log_message = caplog.records[0].getMessage()

        # Parse JSON log
        log_data = json.loads(log_message)
        assert log_data["event"] == "test_event"
        assert log_data["job_id"] == str(job_id)
        assert log_data["custom_field"] == "value"

    def test_logs_with_correlation_id(self, caplog):
        """Test that structured logs can use correlation_id."""
        logger = logging.getLogger("test")
        correlation_id = "test-correlation-123"

        with caplog.at_level(logging.INFO):
            log_structured(
                logger,
                logging.INFO,
                "test_event",
                correlation_id=correlation_id
            )

        assert len(caplog.records) == 1
        log_message = caplog.records[0].getMessage()

        log_data = json.loads(log_message)
        assert log_data["correlation_id"] == correlation_id

    def test_generates_correlation_id_when_missing(self, caplog):
        """Test that correlation_id is generated if not provided."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.INFO):
            log_structured(
                logger,
                logging.INFO,
                "test_event"
            )

        assert len(caplog.records) == 1
        log_message = caplog.records[0].getMessage()

        log_data = json.loads(log_message)
        assert "correlation_id" in log_data
        # Should be a valid UUID
        assert UUID(log_data["correlation_id"])

    def test_redacts_sensitive_data_in_values(self, caplog):
        """Test that sensitive data in field values is redacted."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.INFO):
            log_structured(
                logger,
                logging.INFO,
                "test_event",
                api_response="api_key: sk-secret123"
            )

        log_message = caplog.records[0].getMessage()

        assert "sk-secret123" not in log_message
        assert "[REDACTED]" in log_message

    def test_fallback_redacts_sensitive_data(self, caplog):
        """Test that sensitive data is redacted in fallback format."""
        logger = logging.getLogger("test")

        # Create an object that can't be JSON serialized to trigger fallback
        class UnserializableObject:
            def __str__(self):
                return "api_key=sk-test123"

        with caplog.at_level(logging.INFO):
            log_structured(
                logger,
                logging.INFO,
                "test_event",
                unserializable=UnserializableObject()
            )

        log_message = caplog.records[0].getMessage()

        # Verify fallback was used and sensitive data was redacted
        assert "sk-test123" not in log_message
        assert "[REDACTED]" in log_message


class TestLogConvenienceFunctions:
    """Tests for log_info, log_warning, log_error convenience functions."""

    def test_log_info(self, caplog):
        """Test log_info function."""
        logger = logging.getLogger("test")
        job_id = uuid.uuid4()

        with caplog.at_level(logging.INFO):
            log_info(logger, "info_event", job_id=job_id, data="test")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"

        log_data = json.loads(caplog.records[0].getMessage())
        assert log_data["event"] == "info_event"
        assert log_data["job_id"] == str(job_id)

    def test_log_warning(self, caplog):
        """Test log_warning function."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.WARNING):
            log_warning(logger, "warning_event", reason="test")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

        log_data = json.loads(caplog.records[0].getMessage())
        assert log_data["event"] == "warning_event"

    def test_log_error_with_exception(self, caplog):
        """Test log_error function with exception."""
        logger = logging.getLogger("test")
        error = ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            log_error(logger, "error_event", error=error)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"

        log_data = json.loads(caplog.records[0].getMessage())
        assert log_data["event"] == "error_event"
        assert log_data["error_type"] == "ValueError"
        assert "Test error" in log_data["error_message"]

    def test_log_error_without_exception(self, caplog):
        """Test log_error function without exception."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.ERROR):
            log_error(logger, "error_event", details="something failed")

        assert len(caplog.records) == 1

        log_data = json.loads(caplog.records[0].getMessage())
        assert log_data["event"] == "error_event"
        assert "error_type" not in log_data

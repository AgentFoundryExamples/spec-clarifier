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
"""Centralized logging helper with structured logging support."""

import json
import logging
import re
import uuid
from typing import Any
from uuid import UUID


def get_correlation_id(job_id: UUID | None = None) -> str:
    """Get or generate a correlation ID for tracing requests.

    If a job_id is provided, it will be used as the correlation ID.
    Otherwise, a new UUID will be generated. This ensures every request
    can be traced even when job_id is missing (e.g., malformed requests).

    Args:
        job_id: Optional job UUID to use as correlation ID

    Returns:
        Correlation ID as a string
    """
    if job_id is not None:
        return str(job_id)
    return str(uuid.uuid4())


def redact_sensitive_data(message: str) -> str:
    """Redact sensitive data from log messages.

    Removes API keys, tokens, prompts, and other sensitive information
    that might accidentally be included in log messages.

    Args:
        message: Log message that may contain sensitive data

    Returns:
        Sanitized message with sensitive data replaced with [REDACTED]
    """
    # Remove API keys (various formats)
    message = re.sub(r'(api[_-]?key["\s:=]+)\S+', r"\1[REDACTED]", message, flags=re.IGNORECASE)
    message = re.sub(
        r'(["\']api[_-]?key["\']\s*:\s*["\'])[^"\'\n\r]+?(["\'])',
        r"\1[REDACTED]\2",
        message,
        flags=re.IGNORECASE,
    )

    # Remove bearer tokens
    message = re.sub(r"(bearer\s+)[a-zA-Z0-9_.-]+", r"\1[REDACTED]", message, flags=re.IGNORECASE)

    # Remove tokens (various formats)
    message = re.sub(r'(token["\s:=]+)\S+', r"\1[REDACTED]", message, flags=re.IGNORECASE)
    message = re.sub(
        r'(["\']token["\']\s*:\s*["\'])[^"\'\n\r]+?(["\'])',
        r"\1[REDACTED]\2",
        message,
        flags=re.IGNORECASE,
    )

    # Remove authorization headers
    message = re.sub(
        r'(authorization["\s:]+)[^\r\n]+', r"\1[REDACTED]", message, flags=re.IGNORECASE
    )

    # Remove x-api-key headers
    message = re.sub(r'(x-api-key["\s:]+)[^\r\n]+', r"\1[REDACTED]", message, flags=re.IGNORECASE)

    # Remove secrets in JSON format
    message = re.sub(
        r'(["\'](?:secret|key|password|apikey)["\']\s*:\s*["\'])[^"\'\n\r]+?(["\'])',
        r"\1[REDACTED]\2",
        message,
        flags=re.IGNORECASE,
    )

    # Remove prompts if explicitly labeled
    message = re.sub(r'(prompt["\s:=]+)[^\n]+', r"\1[REDACTED]", message, flags=re.IGNORECASE)

    return message


def log_structured(
    logger: logging.Logger,
    level: int,
    event: str,
    correlation_id: str | None = None,
    job_id: UUID | None = None,
    **context: Any,
) -> None:
    """Log a structured message with consistent format.

    Emits a log entry with structured key/value pairs in JSON-style format.
    Automatically injects correlation_id or job_id for request tracing.
    Sanitizes the message to prevent credential/prompt leakage.

    Args:
        logger: Logger instance to use
        level: Log level (logging.INFO, logging.ERROR, etc.)
        event: Event name describing what happened
        correlation_id: Optional correlation ID for tracing
        job_id: Optional job UUID for tracing
        **context: Additional context fields to include
    """
    # Build structured log entry
    log_data = {"event": event}

    # Add correlation/job ID
    if job_id is not None:
        log_data["job_id"] = str(job_id)
    elif correlation_id is not None:
        log_data["correlation_id"] = correlation_id
    else:
        # Generate a correlation ID if neither is provided
        log_data["correlation_id"] = get_correlation_id()

    # Add context fields
    for key, value in context.items():
        # Sanitize string values to prevent leakage
        if isinstance(value, str):
            log_data[key] = redact_sensitive_data(value)
        else:
            log_data[key] = value

    # Format as JSON-style structured log
    try:
        log_message = json.dumps(log_data, default=str)
        # Redact sensitive data from the serialized JSON to catch any
        # values that were converted to strings during serialization
        log_message = redact_sensitive_data(log_message)
    except (TypeError, ValueError):
        # Fallback to simple format if JSON serialization fails
        fallback_message = f"event={event} " + " ".join(f"{k}={v}" for k, v in log_data.items())
        log_message = redact_sensitive_data(fallback_message)

    # Emit log at specified level
    logger.log(level, log_message)


def log_info(
    logger: logging.Logger, event: str, job_id: UUID | None = None, **context: Any
) -> None:
    """Log an informational structured message.

    Args:
        logger: Logger instance to use
        event: Event name describing what happened
        job_id: Optional job UUID for tracing
        **context: Additional context fields
    """
    log_structured(logger, logging.INFO, event, job_id=job_id, **context)


def log_warning(
    logger: logging.Logger, event: str, job_id: UUID | None = None, **context: Any
) -> None:
    """Log a warning structured message.

    Args:
        logger: Logger instance to use
        event: Event name describing what happened
        job_id: Optional job UUID for tracing
        **context: Additional context fields
    """
    log_structured(logger, logging.WARNING, event, job_id=job_id, **context)


def log_error(
    logger: logging.Logger,
    event: str,
    job_id: UUID | None = None,
    error: Exception | None = None,
    **context: Any,
) -> None:
    """Log an error structured message.

    Args:
        logger: Logger instance to use
        event: Event name describing what happened
        job_id: Optional job UUID for tracing
        error: Optional exception that occurred
        **context: Additional context fields
    """
    if error is not None:
        context["error_type"] = type(error).__name__
        context["error_message"] = redact_sensitive_data(str(error))

    log_structured(logger, logging.ERROR, event, job_id=job_id, **context)

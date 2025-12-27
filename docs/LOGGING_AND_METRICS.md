# Structured Logging and Metrics

This document describes the structured logging and basic metrics system implemented in the spec-clarifier service.

## Overview

The service now includes:
1. **Structured JSON Logging** - All key lifecycle events are logged with structured key/value pairs
2. **Correlation IDs** - Every request can be traced through the system using job_id or generated correlation_id
3. **Basic Metrics** - Operational counters exposed via REST endpoint for monitoring
4. **Sensitive Data Redaction** - Automatic redaction of API keys, tokens, and prompts from logs

## Structured Logging

### Log Format

All logs are emitted as JSON objects with the following structure:

```json
{
  "event": "event_name",
  "job_id": "uuid-or-correlation-id",
  "key1": "value1",
  "key2": "value2"
}
```

### Key Events Logged

#### Job Lifecycle
- `job_created` - When a new job is created
- `job_status_transition` - When a job changes status (PENDING → RUNNING → SUCCESS/FAILED)
- `job_processing_start` - When job processing begins
- `job_processing_complete` - When job processing finishes successfully
- `job_processing_failed` - When job processing fails
- `job_error_recorded` - When an error is recorded for a job
- `job_deleted` - When a job is deleted from the store
- `jobs_cleanup_completed` - When expired jobs are cleaned up

#### LLM Operations
- `llm_client_initialized` - When an LLM client is successfully initialized
- `llm_client_init_failed` - When LLM client initialization fails
- `llm_call_start` - When an LLM API call begins
- `llm_call_success` - When an LLM API call completes successfully
- `llm_call_failed` - When an LLM API call fails
- `llm_response_parsed` - When LLM response is successfully parsed
- `llm_authentication_failed` - When LLM authentication fails
- `llm_rate_limit_exceeded` - When LLM rate limits are hit
- `llm_validation_failed` - When LLM request validation fails
- `llm_network_error` - When LLM network errors occur
- `llm_api_error` - When LLM API errors occur
- `llm_empty_response` - When LLM returns empty response
- `llm_completion_success` - When LLM completion succeeds

### Usage

Use the structured logging helpers from `app.utils.logging_helper`:

```python
from app.utils.logging_helper import log_info, log_warning, log_error

# Log informational message
log_info(
    logger,
    "job_created",
    job_id=job_id,
    num_specs=5,
    num_answers=3
)

# Log warning
log_warning(
    logger,
    "job_skip_not_pending",
    job_id=job_id,
    current_status="RUNNING"
)

# Log error
log_error(
    logger,
    "llm_call_failed",
    job_id=job_id,
    error=exception,
    provider="openai",
    model="gpt-5"
)
```

### Correlation IDs

Every log entry includes either:
- `job_id` - The UUID of the job being processed
- `correlation_id` - A generated UUID for requests without a job_id

This ensures all log entries can be traced back to their source request, even when jobs fail before being created.

### Sensitive Data Redaction

All string values in logs are automatically scanned and redacted for:
- API keys (api_key, apikey)
- Bearer tokens
- Authorization headers
- Secrets and passwords
- Prompts (when explicitly labeled)

Example:
```python
# Input: "Error with api_key=sk-abc123xyz"
# Logged: "Error with api_key=[REDACTED]"
```

## Basic Metrics

### Metrics Endpoint

**GET /v1/metrics/basic**

Returns lightweight operational counters as JSON:

```json
{
  "jobs_queued": 42,
  "jobs_pending": 2,
  "jobs_running": 1,
  "jobs_success": 35,
  "jobs_failed": 4,
  "llm_errors": 8
}
```

### Counter Definitions

- `jobs_queued` - Total number of jobs created since service start
- `jobs_pending` - Current number of jobs in PENDING state
- `jobs_running` - Current number of jobs in RUNNING state  
- `jobs_success` - Total number of successfully completed jobs
- `jobs_failed` - Total number of failed jobs
- `llm_errors` - Total number of LLM API errors encountered

### Thread Safety

All metrics counters are protected by threading locks to ensure accurate counts even under concurrent job processing.

### Counter Updates

Counters are automatically updated by:
- `job_store.create_job()` - Increments jobs_queued and jobs_pending
- `job_store.update_job()` - Updates state counters based on status transitions
- `job_store.delete_job()` - Decrements appropriate state counters
- LLM error handlers - Increment llm_errors on any LLM API failure

### Usage

```python
from app.utils.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Increment a counter
metrics.increment("jobs_queued")

# Decrement a counter
metrics.decrement("jobs_pending")

# Get all counter values
counters = metrics.get_all()
```

## Implementation Details

### Files Added

- `app/utils/__init__.py` - Utils package initialization
- `app/utils/logging_helper.py` - Structured logging helpers and redaction
- `app/utils/metrics.py` - Thread-safe metrics collector
- `tests/test_logging_helper.py` - Tests for logging functionality
- `tests/test_metrics.py` - Tests for metrics collection

### Files Modified

- `app/api/routes_health.py` - Added /v1/metrics/basic endpoint
- `app/services/job_store.py` - Added structured logging and metrics updates
- `app/services/clarification.py` - Added structured logging throughout
- `app/services/llm_clients.py` - Added structured logging for LLM calls
- `tests/test_health.py` - Added metrics endpoint tests
- `tests/test_llm_integration.py` - Updated for structured logging

## Best Practices

1. **Always use structured logging** - Don't use raw logger.info() with f-strings
2. **Include job_id when available** - Enables request tracing
3. **Use descriptive event names** - Makes log analysis easier
4. **Never log prompts or API keys** - Use the redaction helpers
5. **Log at appropriate levels**:
   - INFO: Normal lifecycle events
   - WARNING: Unexpected but recoverable situations
   - ERROR: Failures and exceptions

## Monitoring

The metrics endpoint can be integrated with monitoring systems:

```bash
# Check service health and metrics
curl http://localhost:8000/health
curl http://localhost:8000/v1/metrics/basic

# Example Prometheus-style scraping (custom script needed)
while true; do
  curl -s http://localhost:8000/v1/metrics/basic | \
    jq -r 'to_entries[] | "\(.key) \(.value)"'
  sleep 60
done
```

## Future Enhancements

Potential improvements for consideration:

1. **Prometheus metrics format** - Add /metrics endpoint with Prometheus exposition format
2. **Histogram metrics** - Track LLM call duration distributions
3. **Log aggregation** - Integration with ELK, Splunk, or CloudWatch
4. **Alerting** - Set up alerts on error rates or high pending job counts
5. **Distributed tracing** - Add OpenTelemetry spans for cross-service tracing
6. **Performance metrics** - Track request latency, queue depth, etc.

## Security Considerations

1. **No authentication on metrics endpoint** - Currently read-only and safe to expose
2. **Redaction is best-effort** - Always review logs before sharing externally
3. **In-memory metrics** - Reset on service restart (not persisted)
4. **No rate limiting** - Consider adding if endpoint is abused

## Testing

Run tests with:

```bash
# Test logging helpers
pytest tests/test_logging_helper.py -v

# Test metrics collection
pytest tests/test_metrics.py -v

# Test health and metrics endpoints
pytest tests/test_health.py -v

# Run all tests
pytest tests/ -v
```

All tests include:
- Unit tests for logging and metrics components
- Integration tests for endpoint behavior
- Thread safety tests for concurrent access
- Edge case coverage (missing job_id, invalid counters, etc.)

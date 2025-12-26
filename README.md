# Spec Clarifier

A FastAPI service for clarifying specifications.

## Features

- RESTful API built with FastAPI
- Health check endpoint
- OpenAPI documentation (Swagger UI)
- Configuration via environment variables
- In-memory job store for async clarification workflows
- Thread-safe job management with TTL cleanup

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spec-clarifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Service

You can run the service using either Python directly or Docker.

### Option 1: Running with Python (Recommended for Development)

Start the development server with auto-reload:

```bash
uvicorn app.main:app --reload
```

The service will be available at:
- API: http://localhost:8000
- Interactive API docs (Swagger UI): http://localhost:8000/docs
- Alternative API docs (ReDoc): http://localhost:8000/redoc
- OpenAPI schema: http://localhost:8000/openapi.json

### Option 2: Running with Docker (Optional)

Build the Docker image:

```bash
docker build -t spec-clarifier .
```

Run the container:

```bash
docker run -p 8000:8000 spec-clarifier
```

For development with hot-reload (mounts local code and adds --reload flag):

```bash
docker run -p 8000:8000 -v $(pwd)/app:/app/app spec-clarifier \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

To pass environment variables to the container:

```bash
docker run -p 8000:8000 \
  -e APP_DEBUG=true \
  -e APP_CORS_ORIGINS="http://localhost:3000,http://localhost:8080" \
  spec-clarifier
```

The service will be available at the same URLs as the Python option above.

### Configuration

The application can be configured via environment variables with the `APP_` prefix:

- `APP_APP_NAME`: Application name (default: "Spec Clarifier")
- `APP_APP_VERSION`: Application version (default: "0.1.0")
- `APP_APP_DESCRIPTION`: Application description (default: "A service for clarifying specifications")
- `APP_DEBUG`: Enable debug mode (default: False)

#### Development Flags

- `APP_SHOW_JOB_RESULT`: Include result payload in GET /v1/clarifications/{job_id} responses (default: False)
  - **Production Mode (False)**: Result field is always `null`, keeping responses lightweight
  - **Development Mode (True)**: Result field contains the ClarifiedPlan when job status is SUCCESS
  - **Important**: POST responses always return lightweight summaries regardless of this flag
  - **Use Case**: Enable in development/debugging to inspect results directly in the API response
  - **Note**: Settings are cached at application startup. To change this flag at runtime, restart the service or use environment variables before starting the application.

#### CORS Configuration

CORS (Cross-Origin Resource Sharing) is configured to allow requests from localhost by default:

- `APP_CORS_ORIGINS`: Comma-separated list of allowed origins (default: "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000")
- `APP_CORS_ALLOW_CREDENTIALS`: Allow credentials in CORS requests (default: True)
- `APP_CORS_ALLOW_METHODS`: Allowed HTTP methods (default: "*" for all methods)
- `APP_CORS_ALLOW_HEADERS`: Allowed HTTP headers (default: "*" for all headers)

Example:
```bash
export APP_APP_NAME="My Spec Clarifier"
export APP_DEBUG=true
export APP_SHOW_JOB_RESULT=true  # Enable for development/debugging
export APP_CORS_ORIGINS="http://localhost:3000,http://localhost:8080,https://myapp.com"
uvicorn app.main:app --reload
```

**Note:** For production deployments, configure CORS origins to specific domains instead of using wildcards.

## API Endpoints

### Schema Contracts

⚠️ **STRICT SCHEMA ENFORCEMENT**

The API enforces strict schema validation at all boundaries to ensure predictable behavior and prevent malformed data from entering the LLM pipeline. All endpoints reject requests with:
- Missing required fields
- Extra/unknown fields
- Wrong data types (e.g., strings instead of lists)
- Null values for required string fields

**ClarificationRequest Schema:**

The input schema for clarification requests (`POST /v1/clarifications` and `POST /v1/clarifications/preview`) contains exactly two fields:

```json
{
  "plan": {
    "specs": [
      {
        "purpose": "string (required)",
        "vision": "string (required)",
        "must": ["string array (optional, default: [])"],
        "dont": ["string array (optional, default: [])"],
        "nice": ["string array (optional, default: [])"],
        "open_questions": ["string array (optional, default: [])"],
        "assumptions": ["string array (optional, default: [])"]
      }
    ]
  },
  "answers": [
    {
      "spec_index": "integer (required, >= 0)",
      "question_index": "integer (required, >= 0)",
      "question": "string (required)",
      "answer": "string (required)"
    }
  ]
}
```

- The `answers` field is optional and defaults to an empty list if omitted
- Extra fields in the request will be rejected with a 422 validation error
- The service validates all fields before invoking any LLM processing

**ClarifiedPlan Schema (Output):**

The output schema for clarified specifications contains **exactly six fields per spec**. The `open_questions` field is intentionally excluded:

```json
{
  "specs": [
    {
      "purpose": "string (required)",
      "vision": "string (required)",
      "must": ["string array"],
      "dont": ["string array"],
      "nice": ["string array"],
      "assumptions": ["string array"]
    }
  ]
}
```

⚠️ **Important Notes:**
- `open_questions` are **NOT** present in `ClarifiedPlan` - they are removed during clarification
- Resolved question answers are integrated into the appropriate fields (`must`, `dont`, `nice`, `assumptions`)
- The output schema is strictly enforced - no additional fields will be present
- All validation errors return deterministic 422 responses with sanitized error messages

### Health Check

```bash
GET /health
```

Returns the health status of the service:
```json
{
  "status": "ok"
}
```

### Clarifications

#### Preview Clarifications (Synchronous, Developer-Only)

```bash
POST /v1/clarifications/preview
```

⚠️ **DEVELOPER-ONLY ENDPOINT - NOT FOR PRODUCTION USE**

This endpoint provides a synchronous preview of the clarification process for development and debugging purposes only. It returns immediately with the clarified specifications without async processing or LLM operations.

**For production use cases**, use POST /v1/clarifications to create an async job and poll GET /v1/clarifications/{job_id} for results.

**Request Body:**

```json
{
  "plan": {
    "specs": [
      {
        "purpose": "Build a web application",
        "vision": "A modern, scalable web app",
        "must": ["User authentication", "Database integration"],
        "dont": ["Complex UI frameworks"],
        "nice": ["Dark mode", "Mobile responsive"],
        "open_questions": ["Which database should we use?", "What auth provider?"],
        "assumptions": ["Users have modern browsers"]
      }
    ]
  },
  "answers": []
}
```

**Response (200 OK):**

```json
{
  "specs": [
    {
      "purpose": "Build a web application",
      "vision": "A modern, scalable web app",
      "must": ["User authentication", "Database integration"],
      "dont": ["Complex UI frameworks"],
      "nice": ["Dark mode", "Mobile responsive"],
      "assumptions": ["Users have modern browsers"]
    }
  ]
}
```

**Important:** The `open_questions` field present in the input specification is **NOT included** in the clarified output. Clarified specifications contain only the six fields listed above: `purpose`, `vision`, `must`, `dont`, `nice`, and `assumptions`. Question resolutions are integrated into these fields during the clarification process.

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/v1/clarifications/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "plan": {
      "specs": [
        {
          "purpose": "Build a web application",
          "vision": "A modern, scalable web app",
          "must": ["User authentication"],
          "dont": [],
          "nice": ["Dark mode"],
          "open_questions": ["Which database?"],
          "assumptions": []
        }
      ]
    },
    "answers": []
  }'
```

**Error Response (422 Unprocessable Entity):**

Returned when the request payload is malformed or missing required fields:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "plan", "specs", 0, "vision"],
      "msg": "Field required",
      "input": {"purpose": "Missing vision field"}
    }
  ]
}
```

#### Start Async Clarification Job

```bash
POST /v1/clarifications
```

Creates an asynchronous clarification job and returns immediately with lightweight job details. The job will be processed in the background, transitioning through PENDING → RUNNING → SUCCESS/FAILED states. Use the returned job ID to poll for status and retrieve results.

**Request Body:**

Same format as the preview endpoint:

```json
{
  "plan": {
    "specs": [
      {
        "purpose": "Build a web application",
        "vision": "A modern, scalable web app",
        "must": ["User authentication", "Database integration"],
        "dont": ["Complex UI frameworks"],
        "nice": ["Dark mode", "Mobile responsive"],
        "open_questions": ["Which database should we use?", "What auth provider?"],
        "assumptions": ["Users have modern browsers"]
      }
    ]
  },
  "answers": []
}
```

**Response (202 Accepted):**

Returns immediately with lightweight job summary in PENDING status. **Note**: The response does NOT include the full request or result payload to keep responses lightweight.

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "created_at": "2025-12-26T05:40:11.545362Z",
  "updated_at": "2025-12-26T05:40:11.545362Z",
  "last_error": null
}
```

**Example using curl:**

```bash
# Create job
JOB_ID=$(curl -s -X POST "http://localhost:8000/v1/clarifications" \
  -H "Content-Type: application/json" \
  -d '{
    "plan": {
      "specs": [
        {
          "purpose": "Build a web application",
          "vision": "A modern, scalable web app",
          "must": ["User authentication"],
          "dont": [],
          "nice": ["Dark mode"],
          "open_questions": ["Which database?"],
          "assumptions": []
        }
      ]
    },
    "answers": []
  }' | jq -r '.id')

echo "Created job: $JOB_ID"
```

#### Get Clarification Job Status

```bash
GET /v1/clarifications/{job_id}
```

Retrieves the current status and details of a clarification job. Use this endpoint to poll for job completion after creating an async job.

**Important**: By default (production mode), the `result` field is always `null` to keep responses lightweight. To view results in API responses during development, set `APP_SHOW_JOB_RESULT=true`.

**Path Parameters:**

- `job_id` (UUID): The unique identifier of the job

**Response (200 OK) - Job in Progress (Production Mode):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "RUNNING",
  "created_at": "2025-12-26T05:40:11.545362Z",
  "updated_at": "2025-12-26T05:40:11.546315Z",
  "last_error": null,
  "result": null
}
```

**Response (200 OK) - Job Completed Successfully (Production Mode):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "SUCCESS",
  "created_at": "2025-12-26T05:40:11.545362Z",
  "updated_at": "2025-12-26T05:40:11.890234Z",
  "last_error": null,
  "result": null
}
```

**Response (200 OK) - Job Completed Successfully (Development Mode with APP_SHOW_JOB_RESULT=true):**

When the development flag is enabled, the result field contains the `ClarifiedPlan` with **exactly six fields per spec** (no `open_questions`):

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "SUCCESS",
  "created_at": "2025-12-26T05:40:11.545362Z",
  "updated_at": "2025-12-26T05:40:11.890234Z",
  "last_error": null,
  "result": {
    "specs": [
      {
        "purpose": "Build a web application",
        "vision": "A modern, scalable web app",
        "must": ["User authentication"],
        "dont": [],
        "nice": ["Dark mode"],
        "assumptions": []
      }
    ]
  }
}
```

Note: Each spec in the `result.specs` array contains **only** the six permitted fields: `purpose`, `vision`, `must`, `dont`, `nice`, and `assumptions`. The `open_questions` field from the input is not present in the output.

**Response (200 OK) - Job Failed:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "FAILED",
  "created_at": "2025-12-26T05:40:11.545362Z",
  "updated_at": "2025-12-26T05:40:11.789012Z",
  "last_error": "ValueError: Invalid specification format",
  "result": null
}
```

**Response (404 Not Found):**

Returned when the job ID doesn't exist:

```json
{
  "detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

**Example using curl:**

```bash
# Poll for job completion
JOB_ID="550e8400-e29b-41d4-a716-446655440000"

while true; do
  STATUS=$(curl -s "http://localhost:8000/v1/clarifications/$JOB_ID" | jq -r '.status')
  echo "Job status: $STATUS"
  
  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILED" ]; then
    curl -s "http://localhost:8000/v1/clarifications/$JOB_ID" | jq .
    break
  fi
  
  sleep 0.5
done
```

### Complete Async Workflow Example

This section demonstrates a complete manual workflow for creating, polling, and handling both success and failure scenarios using curl.

#### Step 1: Create a Clarification Job

```bash
# Create a job with specifications
curl -X POST "http://localhost:8000/v1/clarifications" \
  -H "Content-Type: application/json" \
  -d '{
    "plan": {
      "specs": [
        {
          "purpose": "Build a web application",
          "vision": "A modern, scalable web app",
          "must": ["User authentication", "Database integration"],
          "dont": ["Complex UI frameworks"],
          "nice": ["Dark mode", "Mobile responsive"],
          "open_questions": ["Which database should we use?"],
          "assumptions": ["Users have modern browsers"]
        }
      ]
    },
    "answers": []
  }'

# Response (HTTP 202):
# {
#   "id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "PENDING",
#   "created_at": "2025-12-26T06:00:00.000000Z",
#   "updated_at": "2025-12-26T06:00:00.000000Z",
#   "last_error": null
# }

# Save the job ID for polling
export JOB_ID="550e8400-e29b-41d4-a716-446655440000"
```

#### Step 2: Poll for Job Completion

```bash
# Poll until job completes (SUCCESS or FAILED)
while true; do
  RESPONSE=$(curl -s "http://localhost:8000/v1/clarifications/$JOB_ID")
  STATUS=$(echo "$RESPONSE" | jq -r '.status')
  echo "Job status: $STATUS"
  
  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILED" ]; then
    echo "Job completed!"
    echo "$RESPONSE" | jq .
    break
  fi
  
  sleep 0.5
done
```

#### Step 3: Inspect Success Response (Production Mode)

In production mode (default, `APP_SHOW_JOB_RESULT=false`), the result field is always null:

```bash
curl -s "http://localhost:8000/v1/clarifications/$JOB_ID" | jq .

# Response:
# {
#   "id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "SUCCESS",
#   "created_at": "2025-12-26T06:00:00.000000Z",
#   "updated_at": "2025-12-26T06:00:01.500000Z",
#   "last_error": null,
#   "result": null    # Always null in production mode
# }
```

#### Step 4: Inspect Success Response (Development Mode)

To view results during development, set `APP_SHOW_JOB_RESULT=true`:

```bash
# Start server with flag enabled
export APP_SHOW_JOB_RESULT=true
uvicorn app.main:app --reload

# Create and poll job (same as above)
# GET response includes result:
# {
#   "id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "SUCCESS",
#   "created_at": "2025-12-26T06:00:00.000000Z",
#   "updated_at": "2025-12-26T06:00:01.500000Z",
#   "last_error": null,
#   "result": {
#     "specs": [
#       {
#         "purpose": "Build a web application",
#         "vision": "A modern, scalable web app",
#         "must": ["User authentication", "Database integration"],
#         "dont": ["Complex UI frameworks"],
#         "nice": ["Dark mode", "Mobile responsive"],
#         "assumptions": ["Users have modern browsers"]
#         # Note: open_questions is removed from clarified specs
#       }
#     ]
#   }
# }
```

#### Step 5: Inspect Failure Response

When a job fails, the `last_error` field contains the error message:

```bash
curl -s "http://localhost:8000/v1/clarifications/$JOB_ID" | jq .

# Response:
# {
#   "id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "FAILED",
#   "created_at": "2025-12-26T06:00:00.000000Z",
#   "updated_at": "2025-12-26T06:00:01.200000Z",
#   "last_error": "ValueError: Invalid specification format",
#   "result": null
# }
```

#### Step 6: Handle Edge Cases

```bash
# Non-existent job (404)
curl -s "http://localhost:8000/v1/clarifications/00000000-0000-0000-0000-000000000000"
# Response: {"detail": "Job 00000000-0000-0000-0000-000000000000 not found"}

# Invalid UUID (422)
curl -s "http://localhost:8000/v1/clarifications/invalid-uuid"
# Response: {"detail": [{"type": "uuid_parsing", "loc": ["path", "job_id"], ...}]}
```

### Job Lifecycle

The async clarification workflow follows this lifecycle:

1. **PENDING**: Job created and queued for processing
2. **RUNNING**: Job is actively being processed
3. **SUCCESS**: Job completed successfully, `result` contains the clarified plan
4. **FAILED**: Job encountered an error, `last_error` contains the error message

Jobs always update the `updated_at` timestamp when their status or result changes.

## Job Store

The spec-clarifier includes a thread-safe in-memory job store for managing asynchronous clarification workflows. This enables clients to create jobs, poll for status, and retrieve results.

### Job Model

Jobs are represented by the `ClarificationJob` model with the following fields:

- `id` (UUID): Unique identifier for the job
- `status` (JobStatus): Current job status (PENDING, RUNNING, SUCCESS, FAILED)
- `created_at` (datetime): UTC timestamp when job was created
- `updated_at` (datetime): UTC timestamp when job was last updated
- `last_error` (Optional[str]): Error message if job failed
- `request` (ClarificationRequest): The original clarification request
- `result` (Optional[ClarifiedPlan]): The clarified plan result (when SUCCESS)
- `config` (Optional[dict]): Optional configuration for job processing

### Job Store API

The job store provides the following operations:

```python
from app.services.job_store import (
    create_job,
    get_job,
    update_job,
    list_jobs,
    delete_job,
    cleanup_expired_jobs,
    JobNotFoundError
)

# Create a new job
job = create_job(request, config={"model": "gpt-4"})

# Retrieve a job by ID
job = get_job(job_id)

# Update job status and result
updated_job = update_job(
    job_id,
    status=JobStatus.SUCCESS,
    result=clarified_plan
)

# List all jobs or filter by status
all_jobs = list_jobs()
pending_jobs = list_jobs(status=JobStatus.PENDING, limit=10)

# Delete a job
delete_job(job_id)

# Clean up expired completed jobs (default TTL: 24 hours)
cleanup_count = cleanup_expired_jobs(ttl_seconds=86400)

# Clean up stale PENDING jobs (e.g., after 48 hours)
cleanup_count = cleanup_expired_jobs(
    ttl_seconds=86400,
    stale_pending_ttl_seconds=172800
)
```

### Thread Safety

All job store operations are protected by a module-level lock, ensuring thread-safe access in multi-worker environments. Jobs returned by `get_job()` and `list_jobs()` are deep copies, preventing external mutations from affecting stored data. The store is safe to use with:

- Development servers with multiple workers
- Concurrent read/write operations
- Multi-threaded background tasks

### TTL Cleanup

The job store includes a TTL (Time-To-Live) cleanup mechanism that automatically removes expired jobs:

- **Default TTL**: 24 hours (86,400 seconds)
- **Eligible for cleanup**: Jobs with SUCCESS or FAILED status
- **Stale PENDING cleanup**: Optional parameter `stale_pending_ttl_seconds` removes PENDING jobs that were never processed (useful for handling worker crashes)
- **Protected**: RUNNING jobs are never cleaned up to prevent data loss during processing
- **Configurable**: TTL can be adjusted via the `ttl_seconds` and `stale_pending_ttl_seconds` parameters

To implement automated cleanup, you can schedule periodic calls to `cleanup_expired_jobs()` in your application startup or background tasks.

### Edge Cases

The job store handles several important edge cases:

- **Concurrent access**: Lock protection prevents race conditions
- **Missing jobs**: Raises `JobNotFoundError` with clear error messages
- **Timestamp management**: All timestamps are UTC-aware and `updated_at` is always refreshed on updates
- **Optional config**: Config parameter defaults to None and doesn't require client input

## Development

### Running Tests

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```



# Permanents (License, Contributing, Author)

Do not change any of the below sections

## License

This Agent Foundry Project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Created by Agent Foundry and John Brosnihan

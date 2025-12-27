# Spec Clarifier

A FastAPI service for clarifying specifications.

## Features

- RESTful API built with FastAPI
- Health check endpoint
- Admin configuration endpoints for runtime config management
- **Structured JSON logging with correlation IDs**
- **Basic operational metrics endpoint (/v1/metrics/basic)**
- **Automatic sensitive data redaction (API keys, tokens, prompts)**
- **DownstreamDispatcher extension point for forwarding clarified plans**
- OpenAPI documentation (Swagger UI)
- Configuration via environment variables
- In-memory job store for async clarification workflows
- Thread-safe job management with TTL cleanup

## Documentation

- [Logging and Metrics Guide](docs/LOGGING_AND_METRICS.md) - Comprehensive guide to structured logging and metrics collection

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

## OpenAPI Documentation

The Agent Foundry Clarification Service provides comprehensive OpenAPI documentation with detailed endpoint descriptions, request/response examples, and schema definitions.

### Accessing the Documentation

**Interactive Swagger UI:**
```bash
http://localhost:8000/docs
```
The Swagger UI provides an interactive interface to explore and test all API endpoints with live examples.

**ReDoc Documentation:**
```bash
http://localhost:8000/redoc
```
ReDoc offers an alternative, clean documentation interface with a focus on readability.

**OpenAPI Schema (JSON):**
```bash
curl http://localhost:8000/openapi.json > openapi.json
```
Download the complete OpenAPI 3.1 specification for integration with API clients, code generators, or testing tools.

### Key OpenAPI Features

- **Service Metadata**: Clear title "Agent Foundry Clarification Service" with semantic versioning
- **Organized Tags**: Endpoints grouped by Clarifications, Configuration, and Health categories
- **Request Examples**: Realistic payloads for all POST/PUT operations showing required fields
- **Response Examples**: Multiple examples per endpoint demonstrating success, error, and edge cases
- **Async Workflow Documentation**: Explicit descriptions emphasizing asynchronous processing semantics
- **Valid UUIDs**: All job_id examples use valid UUID format (36 characters)

### API Tags and Endpoints

The OpenAPI specification organizes endpoints into three logical tags:

#### Clarifications Tag
Endpoints for creating and managing specification clarification jobs:
- `POST /v1/clarifications` - Create async clarification job (returns 202 Accepted with job_id)
- `GET /v1/clarifications/{job_id}` - Retrieve job status and metadata (result excluded in production by default)
- `POST /v1/clarifications/preview` - Synchronous preview for development only (not for production)
- `GET /v1/clarifications/{job_id}/debug` - Debug endpoint for job inspection (disabled by default, enable with `APP_ENABLE_DEBUG_ENDPOINT=true`)

#### Configuration Tag
Admin endpoints for runtime configuration management (use only in trusted environments):
- `GET /v1/config/defaults` - Retrieve current global default configuration
- `PUT /v1/config/defaults` - Update global default configuration (changes not persisted across restarts)

#### Health Tag
Operational endpoints for monitoring and health checks:
- `GET /health` - Basic health check (returns `{"status": "ok"}`)
- `GET /v1/metrics/basic` - Operational metrics counters (jobs queued, pending, running, success, failed, LLM errors)

### Asynchronous Processing Semantics

**Important:** The `/v1/clarifications` endpoint implements an asynchronous workflow:

1. **POST /v1/clarifications** - Returns immediately with a `job_id` and `202 Accepted` status
2. **GET /v1/clarifications/{job_id}** - Poll this endpoint to check job status (PENDING → RUNNING → SUCCESS/FAILED)
3. **Result Retrieval** - When status is SUCCESS, processing is complete (result field is null in production mode)

The service does **NOT** return clarified specifications inline in POST responses. Always poll the GET endpoint to monitor job progress.

**Critical Limitation**: The status lookup endpoint (`GET /v1/clarifications/{job_id}`) returns **metadata only** by default. In production mode (default), the `result` field is always `null` to keep responses lightweight. Downstream systems receive clarified plans exclusively through the **DownstreamDispatcher** integration (see [Downstream Integration](#downstream-integration) section). The debug payload controlled by `APP_SHOW_JOB_RESULT` is intended for development/debugging only and should never be relied upon in production workflows.

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

- `APP_ENABLE_DEBUG_ENDPOINT`: Enable the debug endpoint GET /v1/clarifications/{job_id}/debug (default: False)
  - **Production Mode (False)**: Debug endpoint returns 403 Forbidden
  - **Development Mode (True)**: Debug endpoint returns sanitized metadata about jobs
  - **Security**: Even when enabled, this endpoint intentionally excludes raw prompts and LLM responses
  - **Use Case**: Enable in development to inspect job configuration, timestamps, and metadata
  - **Important**: This is a security-sensitive feature. Only enable in trusted environments.

- `APP_ENABLE_CONFIG_ADMIN_ENDPOINTS`: Enable admin config endpoints GET/PUT /v1/config/defaults (default: True)
  - **Enabled (True)**: Admin endpoints allow runtime inspection and modification of global defaults
  - **Disabled (False)**: Admin endpoints return 403 Forbidden
  - **Security**: These endpoints have no built-in authentication and should only be accessible in trusted environments
  - **Use Case**: Enable to allow operators to view and update configuration without redeploying
  - **Important**: Changes are NOT persisted across restarts. Use network-level access controls to protect these endpoints.
  - **See**: Admin Configuration Endpoints section for detailed usage and examples

#### LLM Configuration

The service uses LLM providers for specification clarification. Configure the default provider and model:

- `APP_LLM_DEFAULT_PROVIDER`: Default LLM provider (default: "openai")
  - Supported values: "openai", "anthropic", "google"
  - This sets the default provider when no explicit configuration is provided

- `APP_LLM_DEFAULT_MODEL`: Default model identifier (default: "gpt-5")
  - For OpenAI: "gpt-5", "gpt-5.1" (uses Responses API)
  - For Anthropic: "claude-sonnet-4.5", "claude-opus-4" (uses Messages API)
  - For Google: "gemini-3.0-pro" (uses Gemini API)
  - See LLMs.md for detailed provider information

- `APP_ALLOWED_MODELS_OPENAI`: Comma-separated list of allowed OpenAI models (default: "gpt-5,gpt-5.1,gpt-4o")
  - Restricts which OpenAI models can be used in clarification requests and set as defaults
  - Only models in this list will be accepted by the validation layer

- `APP_ALLOWED_MODELS_ANTHROPIC`: Comma-separated list of allowed Anthropic models (default: "claude-sonnet-4.5,claude-opus-4")
  - Restricts which Anthropic models can be used in clarification requests and set as defaults
  - Only models in this list will be accepted by the validation layer

**Note**: The LLM pipeline intentionally redacts prompts, answers, and raw responses from logs to protect sensitive data. Only metadata (provider, model, elapsed time, error messages) is logged.

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
export APP_ENABLE_DEBUG_ENDPOINT=true  # Enable debug endpoint for development
export APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=true  # Enable admin config endpoints (default: true)
export APP_LLM_DEFAULT_PROVIDER="openai"  # or "anthropic", "google"
export APP_LLM_DEFAULT_MODEL="gpt-5"  # or "gpt-5.1", "claude-sonnet-4.5", etc.
export APP_ALLOWED_MODELS_OPENAI="gpt-5,gpt-5.1,gpt-4o"  # Customize allowed OpenAI models
export APP_ALLOWED_MODELS_ANTHROPIC="claude-sonnet-4.5,claude-opus-4"  # Customize allowed Anthropic models
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

**ClarificationRequest Schema (Input):**

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

**Input Schema Notes:**
- The `answers` field is optional and defaults to an empty list if omitted
- The `open_questions` field is **allowed in the input** specification to capture questions that need clarification
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

#### Step 6: Access Debug Information (Optional, Development Only)

If you need to inspect job configuration and metadata during development, you can enable the debug endpoint:

```bash
# Start server with debug endpoint enabled
export APP_ENABLE_DEBUG_ENDPOINT=true
uvicorn app.main:app --reload

# Access debug information for a job
curl -s "http://localhost:8000/v1/clarifications/$JOB_ID/debug" | jq .

# Response:
# {
#   "job_id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "SUCCESS",
#   "created_at": "2025-12-26T06:00:00.000000Z",
#   "updated_at": "2025-12-26T06:00:01.500000Z",
#   "last_error": null,
#   "has_request": true,
#   "has_result": true,
#   "config": {"llm_config": {"provider": "openai", "model": "gpt-5"}},
#   "request_metadata": {
#     "num_specs": 1,
#     "num_answers": 0,
#     "spec_summaries": [
#       {
#         "purpose_length": 22,
#         "vision_length": 28,
#         "num_must": 2,
#         "num_dont": 1,
#         "num_nice": 2,
#         "num_open_questions": 1,
#         "num_assumptions": 1
#       }
#     ]
#   },
#   "result_metadata": {
#     "num_specs": 1,
#     "spec_summaries": [
#       {
#         "purpose_length": 22,
#         "vision_length": 28,
#         "num_must": 2,
#         "num_dont": 1,
#         "num_nice": 2,
#         "num_assumptions": 1
#       }
#     ]
#   }
# }
```

**Important**: The debug endpoint is disabled by default for security. When enabled, it returns sanitized metadata only - raw prompts and LLM responses are intentionally excluded to prevent data leakage.

#### Step 7: Handle Edge Cases

```bash
# Non-existent job (404)
curl -s "http://localhost:8000/v1/clarifications/00000000-0000-0000-0000-000000000000"
# Response: {"detail": "Job 00000000-0000-0000-0000-000000000000 not found"}

# Invalid UUID (422)
curl -s "http://localhost:8000/v1/clarifications/invalid-uuid"
# Response: {"detail": [{"type": "uuid_parsing", "loc": ["path", "job_id"], ...}]}
```

### Admin Configuration Endpoints

⚠️ **ADMIN-ONLY ENDPOINTS - USE ONLY IN TRUSTED ENVIRONMENTS**

These endpoints allow runtime inspection and modification of global default configuration without redeploying the service. They should only be accessible in trusted environments with proper network-level access controls (e.g., firewall rules, VPCs, authentication gateways).

**Security Considerations:**
- Changes are NOT persisted across service restarts - defaults reset to initial values (from environment or built-ins)
- No built-in authentication - these endpoints assume a trusted environment
- Use network-level access controls to restrict access
- Can be disabled by setting `APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=false` (responds with 403 Forbidden)
- All operations are logged with warnings for audit trails

#### GET /v1/config/defaults

Get the current global default configuration and allowed models.

**Request:**
```bash
curl -s "http://localhost:8000/v1/config/defaults" | jq
```

**Response (200 OK):**
```json
{
  "default_config": {
    "provider": "openai",
    "model": "gpt-5.1",
    "system_prompt_id": "default",
    "temperature": 0.1,
    "max_tokens": null
  },
  "allowed_models": {
    "openai": ["gpt-5", "gpt-5.1", "gpt-4o"],
    "anthropic": ["claude-sonnet-4.5", "claude-opus-4"]
  }
}
```

**Response Fields:**
- `default_config`: Current default `ClarificationConfig` used when clarification requests don't provide explicit config
  - `provider`: LLM provider (must be 'openai' or 'anthropic')
  - `model`: Model identifier specific to the provider
  - `system_prompt_id`: System prompt template identifier (see LLMs.md for available templates)
  - `temperature`: Sampling temperature (0.0-2.0), defaults to 0.1
  - `max_tokens`: Optional maximum tokens to generate
- `allowed_models`: Dictionary mapping provider names to lists of allowed model names
  - Only provider/model combinations in this dictionary can be set as defaults

**Status Codes:**
- `200`: Success
- `403`: Endpoint disabled (set `APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=true` to enable)

#### PUT /v1/config/defaults

Update the global default configuration. Changes affect all subsequent clarification requests that don't provide explicit configuration overrides.

**Request:**
```bash
curl -X PUT "http://localhost:8000/v1/config/defaults" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "model": "claude-sonnet-4.5",
    "system_prompt_id": "strict_json",
    "temperature": 0.2,
    "max_tokens": 3000
  }'
```

**Request Body Requirements:**
- All fields are required in the request body
- `provider`: Must be 'openai' or 'anthropic'
- `model`: Must be in the provider's allowed model list
- `system_prompt_id`: System prompt template identifier (unknown IDs fall back to 'default' at runtime)
- `temperature`: Must be 0.0-2.0
- `max_tokens`: Positive integer or null

**Response (200 OK):**
```json
{
  "default_config": {
    "provider": "anthropic",
    "model": "claude-sonnet-4.5",
    "system_prompt_id": "strict_json",
    "temperature": 0.2,
    "max_tokens": 3000
  },
  "allowed_models": {
    "openai": ["gpt-5", "gpt-5.1", "gpt-4o"],
    "anthropic": ["claude-sonnet-4.5", "claude-opus-4"]
  }
}
```

**Status Codes:**
- `200`: Success - returns updated defaults
- `400`: Validation error - invalid provider/model combination or missing required fields
- `403`: Endpoint disabled (set `APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=true` to enable)
- `422`: Pydantic validation error - malformed request (wrong types, invalid temperature, etc.)

**Validation Examples:**

Invalid provider:
```bash
# Request with unsupported provider
curl -X PUT "http://localhost:8000/v1/config/defaults" \
  -H "Content-Type: application/json" \
  -d '{"provider": "google", "model": "gemini-pro", "system_prompt_id": "default", "temperature": 0.1, "max_tokens": null}'

# Response (422): {"detail": [{"type": "literal_error", ...}]}
```

Invalid model for provider:
```bash
# Request with model not in allowed list
curl -X PUT "http://localhost:8000/v1/config/defaults" \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "model": "gpt-3.5-turbo", "system_prompt_id": "default", "temperature": 0.1, "max_tokens": null}'

# Response (400): {"detail": "Model 'gpt-3.5-turbo' is not allowed for provider 'openai'. Allowed models: gpt-5, gpt-5.1, gpt-4o"}
```

**Thread Safety:**
Updates are atomic and protected by a lock to handle concurrent PUT requests safely. If multiple PUT requests arrive simultaneously, they are serialized - the last write wins.

**System Prompt IDs:**
See LLMs.md for details on available system prompt templates:
- `default`: Standard clarification instructions
- `strict_json`: Emphasizes JSON format compliance
- `verbose_explanation`: Provides detailed task explanation

Unknown system_prompt_id values are accepted but fall back to 'default' at runtime with a logged warning.

**Allowed Models Configuration:**
The allowed_models dictionary can be configured at startup via environment variables:
- `APP_ALLOWED_MODELS_OPENAI`: Comma-separated list of OpenAI models (e.g., "gpt-5,gpt-5.1,gpt-4o")
- `APP_ALLOWED_MODELS_ANTHROPIC`: Comma-separated list of Anthropic models (e.g., "claude-sonnet-4.5,claude-opus-4")

If not specified, the service uses built-in safe defaults.

**Disabling Admin Endpoints:**
```bash
export APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=false
uvicorn app.main:app --reload
```

When disabled, both GET and PUT return 403 Forbidden:
```json
{"detail": "Config admin endpoints are disabled. Set APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=true to enable."}
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

## Downstream Integration

### Overview

The spec-clarifier service includes a **DownstreamDispatcher** abstraction that serves as the **sole integration point** for forwarding clarified plans to downstream systems. After a clarification job completes successfully, the service invokes the configured dispatcher to send the clarified plan to your downstream processing pipeline.

**Important**: Downstream dispatch is the **only** way to receive clarified plans in production workflows. The API endpoint `GET /v1/clarifications/{job_id}` returns metadata only (the `result` field is `null` by default) and is not intended for retrieving the actual clarified plan data.

### File Location

The dispatcher implementation is located at:
```
app/services/downstream.py
```

This file contains:
- `DownstreamDispatcher` - Protocol defining the dispatcher interface
- `PlaceholderDownstreamDispatcher` - Temporary placeholder implementation
- `get_downstream_dispatcher()` - Factory function for obtaining the configured dispatcher

### PlaceholderDownstreamDispatcher

The service currently ships with a **placeholder implementation** that logs clarified plans to stdout/logs without making external calls. This placeholder serves as:

1. A reference implementation demonstrating the dispatcher interface
2. A clear hook point for future integrations (marked with TODO comments)
3. A debugging aid for operators to verify successful clarification

**TODO Marker**: The `PlaceholderDownstreamDispatcher` class docstring contains a comprehensive TODO comment with a complete integration checklist. **To locate it quickly**: Open `app/services/downstream.py` and navigate to lines 72-79 within the `PlaceholderDownstreamDispatcher` class. The TODO lists these integration tasks:
- Determine target downstream system (HTTP endpoint, message queue, storage, etc.)
- Implement error handling for network failures and timeouts
- Add retry logic with exponential backoff for transient failures
- Configure authentication/authorization requirements
- Add metrics for dispatch success/failure rates
- Ensure idempotency to handle duplicate dispatches
- Add configuration for endpoint URLs and credentials via environment variables

### Replacing the Placeholder

**Integration Approaches**: Consider these patterns for receiving clarified plans:

1. **Webhook/HTTP Callback (Recommended)**: Configure the service to POST results to your HTTP endpoint. This is the simplest approach and works well for most integrations. The HTTP dispatcher example below demonstrates this pattern.

2. **Message Queue**: Publish results to a message queue (RabbitMQ, Kafka, AWS SQS) for asynchronous processing, retry logic, and decoupling. Useful for high-volume or distributed systems.

3. **Database/Storage**: Write results directly to a shared database or object storage (S3, Azure Blob). Useful when multiple systems need access to results.

To integrate with a real downstream system, follow these steps:

#### 1. Implement a Custom Dispatcher

Create a new class that conforms to the `DownstreamDispatcher` protocol:

```python
# Example: HTTP-based dispatcher
class HTTPDownstreamDispatcher:
    def __init__(self, endpoint: str, api_key: str, timeout: int = 30):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
    
    async def dispatch(self, job: ClarificationJob, plan: ClarifiedPlan) -> None:
        """Send clarified plan to HTTP endpoint."""
        import httpx
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                json={
                    "job_id": str(job.id),
                    "status": job.status.value,
                    "clarified_plan": plan.model_dump(),
                    "created_at": job.created_at.isoformat(),
                },
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()

# Example: Message queue dispatcher (using AWS SQS as example)
class QueueDownstreamDispatcher:
    def __init__(self, queue_url: str, queue_name: str):
        self.queue_url = queue_url
        self.queue_name = queue_name
        # Initialize queue client (e.g., boto3 for SQS, pika for RabbitMQ)
        # Example: self.sqs_client = boto3.client('sqs')
    
    async def dispatch(self, job: ClarificationJob, plan: ClarifiedPlan) -> None:
        """Publish clarified plan to message queue."""
        import json
        
        message_body = json.dumps({
            "job_id": str(job.id),
            "status": job.status.value,
            "clarified_plan": plan.model_dump(),
            "created_at": job.created_at.isoformat(),
        })
        
        # Example for AWS SQS:
        # response = self.sqs_client.send_message(
        #     QueueUrl=self.queue_url,
        #     MessageBody=message_body
        # )
        
        # Example for RabbitMQ:
        # channel.basic_publish(
        #     exchange='',
        #     routing_key=self.queue_name,
        #     body=message_body
        # )
```

#### 2. Update the Factory Function

Modify `get_downstream_dispatcher()` in `app/services/downstream.py` to return your custom implementation based on environment configuration:

```python
def get_downstream_dispatcher() -> DownstreamDispatcher:
    """Factory function to obtain the configured downstream dispatcher."""
    import os
    
    dispatcher_type = os.getenv("DOWNSTREAM_DISPATCHER_TYPE", "placeholder")
    
    if dispatcher_type == "http":
        endpoint = os.getenv("DOWNSTREAM_HTTP_ENDPOINT")
        api_key = os.getenv("DOWNSTREAM_API_KEY")
        
        # Validate required environment variables
        if not endpoint or not api_key:
            raise ValueError(
                "For 'http' dispatcher, DOWNSTREAM_HTTP_ENDPOINT and "
                "DOWNSTREAM_API_KEY environment variables must be set."
            )
        
        return HTTPDownstreamDispatcher(endpoint=endpoint, api_key=api_key)
    
    elif dispatcher_type == "queue":
        queue_url = os.getenv("DOWNSTREAM_QUEUE_URL")
        queue_name = os.getenv("DOWNSTREAM_QUEUE_NAME")
        
        # Validate required environment variables
        if not queue_url or not queue_name:
            raise ValueError(
                "For 'queue' dispatcher, DOWNSTREAM_QUEUE_URL and "
                "DOWNSTREAM_QUEUE_NAME environment variables must be set."
            )
        
        return QueueDownstreamDispatcher(queue_url=queue_url, queue_name=queue_name)
    
    else:
        # Default to placeholder for development/testing
        return PlaceholderDownstreamDispatcher()
```

#### 3. Configure Environment Variables

Set environment variables to configure your dispatcher:

```bash
# For HTTP dispatcher
export DOWNSTREAM_DISPATCHER_TYPE=http
export DOWNSTREAM_HTTP_ENDPOINT=https://your-service.com/api/v1/clarified-plans
export DOWNSTREAM_API_KEY=your-api-key-here

# For queue dispatcher
export DOWNSTREAM_DISPATCHER_TYPE=queue
export DOWNSTREAM_QUEUE_URL=amqp://localhost:5672
export DOWNSTREAM_QUEUE_NAME=clarified-plans
```

**Security Best Practices**:

⚠️ **Critical Security Requirements**:
- **Never commit credentials to source control** - Use environment variables, AWS Secrets Manager, HashiCorp Vault, or similar secret management systems
- **Validate all environment variables** - The factory function example above shows validation to fail fast if required credentials are missing
- **Use HTTPS for HTTP dispatchers** - Never send clarified plans or credentials over unencrypted connections
- **Rotate credentials regularly** - Implement credential rotation policies for API keys and tokens
- **Restrict network access** - Use firewall rules, VPCs, or security groups to limit which services can receive dispatched results
- **Implement authentication** - Always use authentication (API keys, OAuth tokens, mutual TLS) for downstream endpoints
- **Log dispatch operations** - The service logs dispatch events with structured logging (see [Structured Logging and Metrics](#structured-logging-and-metrics)) but never logs credentials or full plan content

The example code intentionally shows credential handling via environment variables to prevent accidental hardcoding. Adapt the security measures to your organization's requirements.

### Dispatcher Behavior and Error Handling

The dispatcher is invoked **after** the job is marked as `SUCCESS` in the job store. This ensures that:

1. The clarified plan is persisted before dispatch attempts
2. Job status remains `SUCCESS` even if dispatch fails
3. Operators can retry failed dispatches without re-running clarification

**Error Handling**: If the dispatcher raises an exception:
- The exception is logged with event `downstream_dispatch_failed`
- The job status remains `SUCCESS` (dispatch is considered an optimization/notification)
- The service continues operating normally

This design ensures that dispatch failures do not affect core clarification functionality.

### Monitoring Dispatch Operations

Dispatcher operations emit structured log events for monitoring:

- `downstream_dispatch_start` - Dispatcher invoked for a job
- `downstream_dispatch_success` - Plan successfully dispatched
- `downstream_dispatch_failed` - Dispatch failed (includes error details)
- `downstream_dispatch_placeholder` - Placeholder dispatcher used (logs plan to console)

See [Structured Logging and Metrics](#structured-logging-and-metrics) for details on consuming these events.

## Structured Logging and Metrics

The spec-clarifier service provides comprehensive structured logging with correlation IDs and operational metrics for monitoring and debugging. For complete details, see [Logging and Metrics Guide](docs/LOGGING_AND_METRICS.md).

### Structured JSON Logging

All key lifecycle events are logged as structured JSON objects with key/value pairs for easy parsing by log aggregation systems. Each log entry includes a descriptive `event` field and relevant context:

```json
{
  "event": "job_created",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "num_specs": 3,
  "num_answers": 2
}
```

**Key Event Categories:**
- **Job Lifecycle**: `job_created`, `job_status_transition`, `job_processing_start`, `job_processing_complete`, `job_processing_failed`
- **LLM Operations**: `llm_call_start`, `llm_call_success`, `llm_call_failed`, `llm_response_parsed`
- **Downstream Dispatch**: `downstream_dispatch_start`, `downstream_dispatch_success`, `downstream_dispatch_failed`, `downstream_dispatch_placeholder`

For a complete list of logged events, see the [Structured Logging section](docs/LOGGING_AND_METRICS.md#structured-logging) in the Logging and Metrics Guide.

### Correlation IDs

Every request is assigned a correlation identifier for end-to-end tracing:

- **job_id**: UUID assigned to clarification jobs (included in all logs related to that job)
- **correlation_id**: Generated UUID for requests without a job_id

These identifiers enable operators to trace a request through the entire system, from API entry to job completion and downstream dispatch.

**Example**: To trace all logs for a specific job:
```bash
# Using grep
grep '"job_id": "550e8400-e29b-41d4-a716-446655440000"' application.log

# Using jq for JSON logs
cat application.log | jq 'select(.job_id == "550e8400-e29b-41d4-a716-446655440000")'
```

### Operational Metrics Endpoint

The service exposes operational metrics at `GET /v1/metrics/basic` for monitoring system health and job throughput:

```bash
curl http://localhost:8000/v1/metrics/basic
```

**Response:**
```json
{
  "jobs_queued": 150,
  "jobs_pending": 3,
  "jobs_running": 2,
  "jobs_success": 142,
  "jobs_failed": 3,
  "llm_errors": 5
}
```

**Metric Definitions:**
- `jobs_queued` - Total jobs created since service start (monotonic counter)
- `jobs_pending` - Current number of jobs waiting for processing (gauge)
- `jobs_running` - Current number of jobs actively being processed (gauge)
- `jobs_success` - Total successfully completed jobs (monotonic counter)
- `jobs_failed` - Total failed jobs (monotonic counter)
- `llm_errors` - Total LLM API errors encountered (monotonic counter)

**Monitoring Integration**: The metrics endpoint can be polled by monitoring systems (Prometheus, Datadog, CloudWatch, etc.) to track service health. Set up alerts on:
- High `jobs_failed` rate
- Growing `jobs_pending` backlog
- Elevated `llm_errors` count

**Feature Flag**: The metrics endpoint is always enabled and requires no configuration. No authentication is required as the metrics contain no sensitive data.

For implementation details and best practices, see the complete [Logging and Metrics Guide](docs/LOGGING_AND_METRICS.md).

### Privacy and Security

The service is designed with privacy and security in mind. The LLM pipeline and API endpoints intentionally redact sensitive information from logs and responses.

#### What is Logged

The service logs the following **non-sensitive** information:

- Job IDs and status transitions (PENDING → RUNNING → SUCCESS/FAILED)
- LLM provider and model names (e.g., "openai", "gpt-5")
- Request/response elapsed times and performance metrics
- Sanitized error messages (without prompts or data)
- HTTP request methods and status codes

#### What is NOT Logged

The following **sensitive information** is intentionally excluded from all logs:

- ❌ User prompts and system prompts sent to LLMs
- ❌ Raw LLM responses and completions
- ❌ Specification content (purpose, vision, must, dont, nice, assumptions, open_questions)
- ❌ Question answers provided by users
- ❌ ClarifiedPlan results and clarified specifications

#### Debug Endpoint Safety

Even when the debug endpoint (`/v1/clarifications/{job_id}/debug`) is enabled via `APP_ENABLE_DEBUG_ENDPOINT=true`, it returns only sanitized metadata:

- Job configuration (provider, model, temperature, etc.)
- Timestamps and status information
- **Counts** of specs, questions, and answers (not content)
- **Length** of text fields (not actual text)
- Error messages (already sanitized)

The debug endpoint **never** returns raw prompts, LLM responses, or full specification content.

#### Logging in Production

For production deployments:

1. Keep `APP_DEBUG=false` to use production logging levels
2. Keep `APP_ENABLE_DEBUG_ENDPOINT=false` to disable debug access
3. Keep `APP_SHOW_JOB_RESULT=false` to prevent result exposure in GET responses
4. Configure log aggregation to capture structured logs for monitoring
5. Review middleware and reverse proxy logs to ensure they don't capture request bodies

The service uses Python's standard logging module with the following format:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

All sensitive data is intentionally excluded from log messages to prevent accidental exposure through log aggregation systems, SIEM tools, or log files.

## Development

### Setup

Install development dependencies:
```bash
pip install -e ".[dev]"
```

This installs the project in editable mode along with development tools:
- **pytest** and **pytest-asyncio** for testing
- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking
- **httpx** for HTTP testing

### Development Tools

The project includes a Makefile with convenient commands for common development tasks:

```bash
# View all available commands
make help

# Install runtime dependencies only
make install

# Install development dependencies
make install-dev

# Format code with black (modifies files)
make format

# Lint code with ruff
make lint

# Type check with mypy
make type-check

# Run tests
make test

# Run tests with verbose output
make test-verbose

# Clean build artifacts and caches
make clean
```

#### Code Formatting with Black

Black formats Python code to a consistent style:

```bash
# Format all code (app/ and tests/)
black app/ tests/

# Check formatting without modifying files
black --check app/ tests/

# Or use the Makefile
make format
```

Configuration is in `pyproject.toml` under `[tool.black]`.

#### Linting with Ruff

Ruff is a fast Python linter that checks for errors and code quality issues:

```bash
# Lint all code
ruff check app/ tests/

# Auto-fix issues where possible
ruff check --fix app/ tests/

# Or use the Makefile
make lint
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

#### Type Checking with Mypy

Mypy performs static type checking:

```bash
# Type check the app code
mypy app/

# Or use the Makefile
make type-check
```

Configuration is in `pyproject.toml` under `[tool.mypy]`.

#### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_health.py

# Run tests matching a pattern
pytest -k "test_health"

# Or use the Makefile
make test          # normal output
make test-verbose  # verbose output
```

Configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Project Structure

```
spec-clarifier/
├── app/              # Application source code
│   ├── api/          # API route handlers
│   ├── models/       # Pydantic models and schemas
│   ├── services/     # Business logic and services
│   └── utils/        # Utility functions
├── tests/            # Test suite
├── pyproject.toml    # Project metadata and tool configuration
├── requirements.txt  # Runtime dependencies (mirrors pyproject.toml)
├── Makefile          # Development commands
└── README.md         # This file
```

### Configuration Files

All tool configurations are centralized in `pyproject.toml`:

- **[project]** - Project metadata, dependencies, and Python version (>=3.11)
- **[project.optional-dependencies]** - Development dependencies
- **[tool.pytest.ini_options]** - Pytest configuration (test paths, asyncio mode)
- **[tool.black]** - Black formatter settings (line length 100, Python 3.11+)
- **[tool.ruff]** - Ruff linter rules and ignores
- **[tool.mypy]** - Mypy type checker settings (non-strict mode with warnings)



# Permanents (License, Contributing, Author)

Do not change any of the below sections

## License

This Agent Foundry Project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Created by Agent Foundry and John Brosnihan

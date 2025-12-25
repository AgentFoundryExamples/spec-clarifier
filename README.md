# Spec Clarifier

A FastAPI service for clarifying specifications.

## Features

- RESTful API built with FastAPI
- Health check endpoint
- OpenAPI documentation (Swagger UI)
- Configuration via environment variables

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

Start the development server with auto-reload:

```bash
uvicorn app.main:app --reload
```

The service will be available at:
- API: http://localhost:8000
- Interactive API docs (Swagger UI): http://localhost:8000/docs
- Alternative API docs (ReDoc): http://localhost:8000/redoc
- OpenAPI schema: http://localhost:8000/openapi.json

### Configuration

The application can be configured via environment variables with the `APP_` prefix:

- `APP_APP_NAME`: Application name (default: "Spec Clarifier")
- `APP_APP_VERSION`: Application version (default: "0.1.0")
- `APP_APP_DESCRIPTION`: Application description (default: "A service for clarifying specifications")
- `APP_DEBUG`: Enable debug mode (default: False)

Example:
```bash
export APP_APP_NAME="My Spec Clarifier"
export APP_DEBUG=true
uvicorn app.main:app --reload
```

## API Endpoints

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

#### Preview Clarifications

```bash
POST /v1/clarifications/preview
```

Accepts a ClarificationRequest containing specifications with open questions and returns a ClarifiedPlan with the questions removed. This endpoint provides a synchronous preview of the clarification process without processing answers or triggering async/LLM operations.

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

Note that `open_questions` are omitted from the clarified specifications in the response.

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

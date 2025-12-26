# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 24
- **Intra-repo dependencies**: 41
- **External stdlib dependencies**: 27
- **External third-party dependencies**: 35

## External Dependencies

### Standard Library / Core Modules

Total: 27 unique modules

- `abc.ABC`
- `abc.abstractmethod`
- `asyncio`
- `copy`
- `datetime.datetime`
- `datetime.timedelta`
- `datetime.timezone`
- `enum.Enum`
- `functools.lru_cache`
- `inspect`
- `json`
- `logging`
- `os`
- `re`
- `threading`
- `time`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- ... and 7 more (see JSON for full list)

### Third-Party Packages

Total: 35 unique packages

- `anthropic.`
- `anthropic.APIConnectionError`
- `anthropic.APIError`
- `anthropic.APITimeoutError`
- `anthropic.AsyncAnthropic`
- `anthropic.AuthenticationError`
- `anthropic.BadRequestError`
- `anthropic.RateLimitError`
- `anthropic.UnprocessableEntityError`
- `fastapi.APIRouter`
- `fastapi.BackgroundTasks`
- `fastapi.FastAPI`
- `fastapi.HTTPException`
- `fastapi.Request`
- `fastapi.middleware.cors.CORSMiddleware`
- `fastapi.responses.JSONResponse`
- `fastapi.testclient.TestClient`
- `httpx`
- `openai.`
- `openai.APIConnectionError`
- ... and 15 more (see JSON for full list)

## Most Depended Upon Files (Intra-Repo)

- `app/models/specs.py` (11 dependents)
- `app/services/job_store.py` (8 dependents)
- `app/services/clarification.py` (6 dependents)
- `app/services/llm_clients.py` (6 dependents)
- `app/config.py` (5 dependents)
- `app/main.py` (3 dependents)
- `app/api/routes_health.py` (1 dependents)
- `app/api/routes_clarifications.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `tests/test_clarifications_api.py` (5 dependencies)
- `app/api/routes_clarifications.py` (4 dependencies)
- `tests/test_async_job_lifecycle.py` (4 dependencies)
- `tests/test_clarification_service.py` (4 dependencies)
- `tests/test_llm_integration.py` (4 dependencies)
- `app/main.py` (3 dependencies)
- `app/services/clarification.py` (3 dependencies)
- `app/services/__init__.py` (2 dependencies)
- `tests/test_job_store.py` (2 dependencies)
- `tests/test_main.py` (2 dependencies)

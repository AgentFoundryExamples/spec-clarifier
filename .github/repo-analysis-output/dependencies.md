# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 22
- **Intra-repo dependencies**: 31
- **External stdlib dependencies**: 22
- **External third-party dependencies**: 17

## External Dependencies

### Standard Library / Core Modules

Total: 22 unique modules

- `abc.ABC`
- `abc.abstractmethod`
- `datetime.datetime`
- `datetime.timedelta`
- `datetime.timezone`
- `enum.Enum`
- `functools.lru_cache`
- `logging`
- `os`
- `re`
- `threading`
- `time`
- `typing.Any`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `typing.Protocol`
- `unittest.mock.MagicMock`
- `unittest.mock.Mock`
- `unittest.mock.patch`
- ... and 2 more (see JSON for full list)

### Third-Party Packages

Total: 17 unique packages

- `fastapi.APIRouter`
- `fastapi.BackgroundTasks`
- `fastapi.FastAPI`
- `fastapi.HTTPException`
- `fastapi.Request`
- `fastapi.middleware.cors.CORSMiddleware`
- `fastapi.responses.JSONResponse`
- `fastapi.testclient.TestClient`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `pydantic.ValidationError`
- `pydantic.field_validator`
- `pydantic_settings.BaseSettings`
- `pydantic_settings.SettingsConfigDict`
- `pytest`
- `starlette.middleware.cors.CORSMiddleware`

## Most Depended Upon Files (Intra-Repo)

- `app/models/specs.py` (9 dependents)
- `app/services/job_store.py` (7 dependents)
- `app/config.py` (5 dependents)
- `app/services/clarification.py` (4 dependents)
- `app/main.py` (3 dependents)
- `app/api/routes_health.py` (1 dependents)
- `app/api/routes_clarifications.py` (1 dependents)
- `app/services/llm_clients.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `app/api/routes_clarifications.py` (4 dependencies)
- `tests/test_clarifications_api.py` (4 dependencies)
- `app/main.py` (3 dependencies)
- `tests/test_async_job_lifecycle.py` (3 dependencies)
- `tests/test_clarification_service.py` (3 dependencies)
- `app/services/__init__.py` (2 dependencies)
- `app/services/clarification.py` (2 dependencies)
- `tests/test_job_store.py` (2 dependencies)
- `tests/test_main.py` (2 dependencies)
- `app/models/__init__.py` (1 dependencies)

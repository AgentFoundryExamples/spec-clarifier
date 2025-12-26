# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 19
- **Intra-repo dependencies**: 20
- **External stdlib dependencies**: 15
- **External third-party dependencies**: 14

## External Dependencies

### Standard Library / Core Modules

Total: 15 unique modules

- `datetime.datetime`
- `datetime.timedelta`
- `datetime.timezone`
- `enum.Enum`
- `functools.lru_cache`
- `logging`
- `os`
- `threading`
- `time`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- `unittest.mock.patch`
- `uuid.UUID`
- `uuid.uuid4`

### Third-Party Packages

Total: 14 unique packages

- `fastapi.APIRouter`
- `fastapi.FastAPI`
- `fastapi.Request`
- `fastapi.middleware.cors.CORSMiddleware`
- `fastapi.responses.JSONResponse`
- `fastapi.testclient.TestClient`
- `pydantic.BaseModel`
- `pydantic.ConfigDict`
- `pydantic.Field`
- `pydantic.ValidationError`
- `pydantic_settings.BaseSettings`
- `pydantic_settings.SettingsConfigDict`
- `pytest`
- `starlette.middleware.cors.CORSMiddleware`

## Most Depended Upon Files (Intra-Repo)

- `app/models/specs.py` (7 dependents)
- `app/services/clarification.py` (3 dependents)
- `app/config.py` (3 dependents)
- `app/main.py` (3 dependents)
- `app/services/job_store.py` (2 dependents)
- `app/api/routes_health.py` (1 dependents)
- `app/api/routes_clarifications.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `app/main.py` (3 dependencies)
- `app/api/routes_clarifications.py` (2 dependencies)
- `app/services/__init__.py` (2 dependencies)
- `tests/test_clarification_service.py` (2 dependencies)
- `tests/test_job_store.py` (2 dependencies)
- `tests/test_main.py` (2 dependencies)
- `app/models/__init__.py` (1 dependencies)
- `app/services/clarification.py` (1 dependencies)
- `app/services/job_store.py` (1 dependencies)
- `tests/test_clarifications_api.py` (1 dependencies)

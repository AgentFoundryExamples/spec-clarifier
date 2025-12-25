# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 16
- **Intra-repo dependencies**: 15
- **External stdlib dependencies**: 5
- **External third-party dependencies**: 14

## External Dependencies

### Standard Library / Core Modules

Total: 5 unique modules

- `functools.lru_cache`
- `logging`
- `os`
- `typing.List`
- `unittest.mock.patch`

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

- `app/models/specs.py` (5 dependents)
- `app/config.py` (3 dependents)
- `app/main.py` (3 dependents)
- `app/__init__.py` (2 dependents)
- `app/api/routes_health.py` (1 dependents)
- `app/api/routes_clarifications.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `app/main.py` (3 dependencies)
- `app/api/routes_clarifications.py` (2 dependencies)
- `tests/test_clarification_service.py` (2 dependencies)
- `tests/test_main.py` (2 dependencies)
- `app/models/__init__.py` (1 dependencies)
- `app/services/clarification.py` (1 dependencies)
- `tests/test_clarifications_api.py` (1 dependencies)
- `tests/test_config.py` (1 dependencies)
- `tests/test_health.py` (1 dependencies)
- `tests/test_models.py` (1 dependencies)

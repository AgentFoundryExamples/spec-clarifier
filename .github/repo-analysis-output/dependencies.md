# Dependency Graph

Multi-language intra-repository dependency analysis.

Supports Python, JavaScript/TypeScript, C/C++, Rust, Go, Java, C#, Swift, HTML/CSS, and SQL.

Includes classification of external dependencies as stdlib vs third-party.

## Statistics

- **Total files**: 9
- **Intra-repo dependencies**: 6
- **External stdlib dependencies**: 4
- **External third-party dependencies**: 8

## External Dependencies

### Standard Library / Core Modules

Total: 4 unique modules

- `functools.lru_cache`
- `logging`
- `os`
- `unittest.mock.patch`

### Third-Party Packages

Total: 8 unique packages

- `fastapi.APIRouter`
- `fastapi.FastAPI`
- `fastapi.Request`
- `fastapi.responses.JSONResponse`
- `fastapi.testclient.TestClient`
- `pydantic_settings.BaseSettings`
- `pydantic_settings.SettingsConfigDict`
- `pytest`

## Most Depended Upon Files (Intra-Repo)

- `app/config.py` (3 dependents)
- `app/main.py` (2 dependents)
- `app/api/routes_health.py` (1 dependents)

## Files with Most Dependencies (Intra-Repo)

- `app/main.py` (2 dependencies)
- `tests/test_main.py` (2 dependencies)
- `tests/test_config.py` (1 dependencies)
- `tests/test_health.py` (1 dependencies)

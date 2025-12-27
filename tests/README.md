# Test Suite Documentation

This document provides an overview of the test suite for the spec-clarifier project, including test organization, fixtures, and how to run tests.

## Overview

The test suite provides comprehensive coverage of:
- **Models**: Pydantic model validation (SpecInput, ClarifiedSpec, ClarificationConfig)
- **LLM Clients**: DummyLLMClient, OpenAI, and Anthropic implementations
- **Services**: Clarification pipeline, job processing, and state management
- **API Endpoints**: FastAPI integration tests for clarifications and config admin
- **Integration**: End-to-end workflows from request to response

**Total Tests**: 594 tests covering all critical paths and edge cases.

## Test Files

### Core Test Files (as specified in ISS-X)

| File | Purpose | Test Count | Key Areas |
|------|---------|------------|-----------|
| `test_models.py` | Pydantic model validation | 64 | SpecInput, ClarifiedSpec, ClarificationConfig validation, edge cases |
| `test_llm_clients.py` | LLM client abstractions | 154 | DummyLLMClient, OpenAI/Anthropic clients, error handling, protocol compliance |
| `test_clarification_service.py` | Service layer logic | 68 | Pipeline processing, job lifecycle, LLM integration, downstream dispatch |
| `test_clarifications_api.py` | API endpoint integration | 82 | POST/GET /v1/clarifications, job polling, preview, debug endpoints |
| `test_config_admin_api.py` | Config admin endpoints | 14 | GET/PUT /v1/config/defaults, validation, persistence |

### Additional Test Files

| File | Purpose | Test Count |
|------|---------|------------|
| `test_async_job_lifecycle.py` | Async job processing | 34 |
| `test_config.py` | Configuration loading | 56 |
| `test_downstream.py` | Downstream dispatch | 16 |
| `test_dummy_mode.py` | Dummy mode behavior | 16 |
| `test_health.py` | Health endpoints | 8 |
| `test_job_store.py` | Job storage layer | 38 |
| `test_llm_integration.py` | LLM integration tests | 10 |
| `test_logging_helper.py` | Structured logging | 14 |
| `test_main.py` | App initialization | 12 |
| `test_metrics.py` | Metrics collection | 12 |
| `test_openapi_metadata.py` | OpenAPI schema | 16 |
| `test_prompt_builder.py` | Prompt construction | 60 |

## Fixtures (conftest.py)

The `conftest.py` file centralizes common fixtures used across multiple test modules:

### Test Client Fixtures

- **`client`**: Standard FastAPI TestClient with default settings
- **`enabled_debug_client`**: TestClient with debug endpoints enabled
- **`client_with_job_results`**: TestClient with job results visible in responses
- **`disabled_config_client`**: TestClient with config admin endpoints disabled

### Job Store Management

- **`clean_job_store`** (autouse): Clears job store before/after each test
- Ensures deterministic test state and prevents test pollution

### LLM Client Mocking

- **`mock_llm_client`**: Patches get_llm_client to return DummyLLMClient
- **`create_dummy_client_with_response(specs)`**: Helper to create client with specific response
- Enables offline testing without real API keys

### Configuration Management

- **`clear_settings_cache`** (autouse): Clears settings cache for environment-dependent tests
- **`reset_config`**: Resets ClarificationConfig to known defaults

### Metrics Management

- **`reset_metrics`**: Resets metrics collector to zero state

### Sample Data

- **`sample_spec_input`**: Reusable SpecInput with typical content
- **`sample_clarification_request`**: Reusable ClarificationRequest
- **`deterministic_job_timing`**: Configuration for predictable async behavior

## Running Tests

### Run All Tests
```bash
make test
# or
pytest
```

### Run Specific Test Files
```bash
pytest tests/test_models.py
pytest tests/test_llm_clients.py
pytest tests/test_clarification_service.py
pytest tests/test_clarifications_api.py
pytest tests/test_config_admin_api.py
```

### Run Tests by Category
```bash
# Unit tests only (models, LLM clients)
pytest tests/test_models.py tests/test_llm_clients.py

# Service layer tests
pytest tests/test_clarification_service.py tests/test_job_store.py

# API integration tests
pytest tests/test_clarifications_api.py tests/test_config_admin_api.py tests/test_health.py

# Async tests
pytest tests/test_async_job_lifecycle.py
```

### Verbose Output
```bash
make test-verbose
# or
pytest -v
```

### Run Specific Test Classes or Functions
```bash
# Run a specific test class
pytest tests/test_models.py::TestSpecInput

# Run a specific test function
pytest tests/test_models.py::TestSpecInput::test_spec_input_with_all_fields

# Run tests matching a pattern
pytest -k "test_dummy_client"
```

### Coverage Report
```bash
pytest --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Design Principles

### 1. Deterministic Execution

All tests run deterministically without external dependencies:
- **No network calls**: LLM clients are mocked with DummyLLMClient
- **No real API keys**: Tests use fixtures that don't require credentials
- **Clean state**: Job store and metrics are reset between tests
- **Fast execution**: Complete suite runs in ~4 seconds

### 2. Offline Operation

Tests are designed to run completely offline:
- DummyLLMClient provides canned responses
- No external API calls to OpenAI, Anthropic, or other services
- Environment variables are mocked via monkeypatch
- All HTTP requests use TestClient (in-memory)

### 3. Comprehensive Coverage

Tests cover:
- **Success paths**: Valid inputs produce expected outputs
- **Failure paths**: Invalid inputs trigger appropriate errors
- **Edge cases**: Boundary conditions, empty values, unicode, special characters
- **Error handling**: Proper error messages and sanitization
- **State management**: Job lifecycle, status transitions, timestamps
- **Integration**: Full workflows from API request to response

### 4. Test Organization

Tests are organized by component/layer:
- **Models**: Validation logic for Pydantic models
- **Clients**: LLM client implementations and protocols
- **Services**: Business logic and orchestration
- **API**: HTTP endpoints and request/response handling
- **Integration**: End-to-end workflows

## Acceptance Criteria Coverage

The test suite fulfills all acceptance criteria from ISS-X:

✅ **Unit tests cover SpecInput, ClarifiedSpec, and ClarificationConfig**
- 64 tests in `test_models.py` validate success/failure cases
- Tests match exact field requirements defined in models
- Edge cases: missing fields, wrong types, invalid UUIDs, extra fields

✅ **DummyLLMClient tests assert JSON parsing and error handling**
- 66 tests in `test_llm_clients.py` cover DummyLLMClient behavior
- Valid payloads parse successfully
- Malformed responses raise predictable errors
- Error messages are documented and sanitized

✅ **Clarification pipeline verifies open_questions removal**
- Tests confirm open_questions are stripped from results
- ClarifiedSpec exposes only the six defined fields (purpose, vision, must, dont, nice, assumptions)
- Field assertions adapt automatically if canonical list changes

✅ **Integration tests with FastAPI and DummyLLMClient**
- 82 tests in `test_clarifications_api.py` exercise POST/GET /v1/clarifications
- Job polling until SUCCESS with deterministic responses
- Config admin GET/PUT validation in `test_config_admin_api.py`
- HTTP status codes and payload shapes asserted

✅ **Centralized fixtures in conftest.py**
- App fixtures with DummyLLMClient configuration
- Deterministic job store timings
- Reusable plan/answers payloads
- Shared cleanup and setup logic

## Edge Cases

### Job Polling
- Tests cap retries/timeouts to prevent hanging
- Deterministic timing configuration for predictable behavior
- Fast polling intervals (10ms) for rapid test execution

### API Key Independence
- Tests remain stable whether or not real API keys exist
- DummyLLMClient forced via fixtures
- Environment variables are mocked

### Field Assertions
- ClarifiedSpec key assertions use `ClarifiedSpec.model_fields.keys()`
- Adapts automatically if canonical field list changes
- Avoids brittle literal string comparisons

## Common Test Patterns

### Testing Models
```python
def test_model_validation():
    # Valid case
    spec = SpecInput(purpose="Test", vision="Vision")
    assert spec.purpose == "Test"
    
    # Invalid case
    with pytest.raises(ValidationError) as exc_info:
        SpecInput(purpose="Test")  # Missing required vision
    
    errors = exc_info.value.errors()
    assert any(error["loc"] == ("vision",) for error in errors)
```

### Testing LLM Clients
```python
async def test_llm_client():
    client = DummyLLMClient(canned_response='{"result": "test"}')
    response = await client.complete(
        system_prompt="System",
        user_prompt="User",
        model="test-model"
    )
    assert response == '{"result": "test"}'
```

### Testing Services
```python
async def test_service_processing():
    spec = SpecInput(purpose="Test", vision="Vision")
    plan = PlanInput(specs=[spec])
    request = ClarificationRequest(plan=plan)
    
    job = start_clarification_job(request, background_tasks)
    assert job.status == JobStatus.PENDING
    
    await process_clarification_job(job.id, llm_client=dummy_client)
    
    processed = get_job(job.id)
    assert processed.status == JobStatus.SUCCESS
```

### Testing API Endpoints
```python
def test_api_endpoint(client):
    response = client.post("/v1/clarifications", json={
        "plan": {"specs": [{"purpose": "Test", "vision": "Vision"}]},
        "answers": []
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["status"] == "PENDING"
```

## Troubleshooting

### Tests Fail with "ModuleNotFoundError"
Install dev dependencies:
```bash
make install-dev
# or
pip install -e ".[dev]"
```

### Tests Fail with "OPENAI_API_KEY not set"
The test suite should not require real API keys. If you see this error:
1. Check that `mock_llm_client` fixture is being used
2. Verify `conftest.py` is present and loaded
3. Check for tests that directly import real LLM clients without mocking

### Tests Hang or Timeout
1. Check async test timing configuration in `deterministic_job_timing` fixture
2. Ensure job polling tests use fast intervals (10ms)
3. Verify tests cap retries to prevent infinite loops

### Flaky Tests
Tests should be deterministic. If you encounter flaky behavior:
1. Check that `clean_job_store` fixture is running (autouse)
2. Verify metrics are reset between tests (`reset_metrics`)
3. Check for shared state or race conditions in async tests

## Contributing

When adding new tests:

1. **Use existing fixtures** from `conftest.py` when possible
2. **Follow naming conventions**: `test_<component>_<behavior>`
3. **Add docstrings**: Explain what the test validates
4. **Test both success and failure paths**
5. **Include edge cases**: Empty values, boundaries, unicode
6. **Keep tests fast**: Mock external dependencies
7. **Make tests deterministic**: Use canned responses, clean state

## Maintenance

### Updating Fixtures

When modifying fixtures in `conftest.py`:
1. Document the change in docstring
2. Check for breaking changes in existing tests
3. Run full test suite to verify compatibility
4. Update this README if fixture behavior changes

### Adding New Test Files

New test files automatically inherit fixtures from `conftest.py`. Follow the naming convention `test_<component>.py` and organize tests into classes by feature area.

### Deprecating Tests

Don't delete tests unless:
1. The feature is completely removed
2. The test is redundant with better coverage elsewhere
3. Document the reason in commit message

## References

- **Project**: [spec-clarifier](https://github.com/AgentFoundryExamples/spec-clarifier)
- **Issue**: ISS-X "Add targeted pytest coverage"
- **Pytest Documentation**: https://docs.pytest.org/
- **FastAPI Testing**: https://fastapi.tiangolo.com/tutorial/testing/
- **Pydantic Validation**: https://docs.pydantic.dev/latest/concepts/models/

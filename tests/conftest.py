# Copyright 2025 John Brosnihan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Centralized test fixtures for spec-clarifier test suite.

This module provides reusable fixtures for:
- Test clients (FastAPI TestClient with various configurations)
- Job store management (cleanup, deterministic state)
- LLM client mocking (DummyLLMClient with canned responses)
- Configuration management (settings cache, config reset)
- Metrics reset (clean state for metrics tests)

Fixtures in this file are available to all test modules without explicit imports.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings, set_default_config
from app.main import create_app
from app.models.config_models import ClarificationConfig
from app.models.specs import SpecInput
from app.services.job_store import clear_all_jobs
from app.services.llm_clients import DummyLLMClient
from app.utils.metrics import get_metrics_collector

# =============================================================================
# Test Client Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a standard FastAPI TestClient.

    Returns a TestClient instance configured with the default app settings.
    This fixture is used across multiple test modules for API endpoint testing.

    Returns:
        TestClient: Configured test client for FastAPI app
    """
    app = create_app()
    return TestClient(app)


@pytest.fixture
def enabled_debug_client(monkeypatch):
    """Create a TestClient with debug endpoint enabled.

    Sets APP_ENABLE_DEBUG_ENDPOINT=true and creates a fresh app instance
    with debug endpoints available. Useful for testing debug-specific routes.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        TestClient: Test client with debug endpoints enabled
    """
    monkeypatch.setenv("APP_ENABLE_DEBUG_ENDPOINT", "true")
    get_settings.cache_clear()
    app = create_app()
    return TestClient(app)


@pytest.fixture
def client_with_job_results(monkeypatch):
    """Create a TestClient with job results visible in GET responses.

    Sets APP_SHOW_JOB_RESULT=true to enable result field in job status responses.
    Used for testing that results are properly exposed when the flag is enabled.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        TestClient: Test client with job results enabled
    """
    monkeypatch.setenv("APP_SHOW_JOB_RESULT", "true")
    get_settings.cache_clear()
    app = create_app()
    return TestClient(app)


@pytest.fixture
def disabled_config_client(monkeypatch):
    """Create a TestClient with config admin endpoints disabled.

    Sets APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=false and creates a fresh app
    instance without config admin endpoints. Useful for testing that endpoints
    are properly gated behind the configuration flag.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        TestClient: Test client with config admin endpoints disabled

    Note:
        The autouse `clear_settings_cache` fixture handles cache clearing
        before and after each test, so explicit cleanup is not needed here.
    """
    monkeypatch.setenv("APP_ENABLE_CONFIG_ADMIN_ENDPOINTS", "false")
    get_settings.cache_clear()
    app = create_app()
    return TestClient(app)


# =============================================================================
# Job Store Management Fixtures
# =============================================================================


@pytest.fixture(autouse=True, scope="function")
def clean_job_store():
    """Clean the job store before and after each test (autouse).

    This fixture runs automatically for all tests to ensure a clean job store
    state. Jobs are cleared before the test runs and after it completes, preventing
    test pollution and ensuring deterministic test behavior.

    This is the primary job store fixture used by most test modules.
    Explicit scope='function' ensures each test gets a clean job store.
    """
    clear_all_jobs()
    yield
    clear_all_jobs()


# =============================================================================
# LLM Client Mocking Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for API tests to avoid needing real API keys.

    This fixture patches get_llm_client to return a DummyLLMClient with a
    valid ClarifiedPlan JSON response. This allows tests to run offline without
    real LLM API keys while still exercising the full clarification pipeline.

    The dummy client returns a minimal valid spec with all required fields but
    no actual content (empty arrays). Tests that need specific content should
    use more targeted mocking or create custom DummyLLMClient instances.

    Yields:
        Context manager with patched LLM client factory
    """

    def create_dummy_client_from_request(provider, config):
        """Create dummy client that returns valid response."""
        return DummyLLMClient(
            canned_response=json.dumps(
                {
                    "specs": [
                        {
                            "purpose": "Test",
                            "vision": "Test vision",
                            "must": [],
                            "dont": [],
                            "nice": [],
                            "assumptions": [],
                        }
                    ]
                }
            )
        )

    with patch(
        "app.services.clarification.get_llm_client", side_effect=create_dummy_client_from_request
    ):
        yield


def create_dummy_client_with_response(specs):
    """Helper to create DummyLLMClient with valid ClarifiedPlan JSON response.

    This helper creates a DummyLLMClient configured to return a valid ClarifiedPlan
    JSON that matches the provided SpecInput list. It strips open_questions and
    returns only the six ClarifiedSpec fields (purpose, vision, must, dont, nice,
    assumptions).

    Args:
        specs: List of SpecInput objects to convert to clarified response

    Returns:
        DummyLLMClient: Client configured with appropriate canned response

    Example:
        >>> spec = SpecInput(purpose="Test", vision="Vision", must=["A"])
        >>> client = create_dummy_client_with_response([spec])
        >>> response = await client.complete("sys", "user", "model")
        >>> # response contains valid ClarifiedPlan JSON
    """
    clarified_specs = []
    for spec in specs:
        clarified_spec = {
            "purpose": spec.purpose,
            "vision": spec.vision,
            "must": spec.must,
            "dont": spec.dont,
            "nice": spec.nice,
            "assumptions": spec.assumptions,
        }
        clarified_specs.append(clarified_spec)

    response_json = json.dumps({"specs": clarified_specs}, indent=2)
    return DummyLLMClient(canned_response=response_json)


# =============================================================================
# Configuration Management Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before and after each test (autouse).

    This fixture ensures the settings cache is cleared before and after tests
    that modify environment variables. This prevents settings pollution across
    tests and ensures each test gets a fresh settings instance based on the
    current environment.

    Used primarily in config-related tests where environment variables are
    modified via monkeypatch.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def reset_config():
    """Reset default clarification config before and after each test.

    This fixture resets the default ClarificationConfig to known defaults
    (OpenAI gpt-5.1 with temperature 0.1) before and after tests. This ensures
    config tests start from a clean state and don't pollute other tests.

    Used primarily in config admin endpoint tests.
    """
    # Reset to known defaults
    default = ClarificationConfig(
        provider="openai",
        model="gpt-5.1",
        system_prompt_id="default",
        temperature=0.1,
        max_tokens=None,
    )
    set_default_config(default)
    yield
    # Reset after test
    set_default_config(default)


# =============================================================================
# Metrics Management Fixtures
# =============================================================================


@pytest.fixture
def reset_metrics():
    """Reset metrics collector before and after each test.

    This fixture resets the metrics collector to ensure metrics tests start
    from a clean state with zero counters. Prevents metrics pollution between
    tests that track request counts, error counts, etc.

    Used primarily in metrics and health endpoint tests.
    """
    metrics = get_metrics_collector()
    metrics.reset()
    yield
    metrics.reset()


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_spec_input():
    """Create a sample SpecInput for testing.

    Provides a reusable SpecInput with typical content for testing. Includes
    all fields with representative data including open_questions.

    Returns:
        SpecInput: Sample specification input object
    """
    return SpecInput(
        purpose="Build a web application",
        vision="Modern, scalable, and user-friendly",
        must=["Authentication", "Database persistence", "RESTful API"],
        dont=["Complex legacy patterns", "Monolithic architecture"],
        nice=["Dark mode", "Real-time updates", "Mobile app"],
        open_questions=["Which database?", "Which authentication method?"],
        assumptions=["Users have modern browsers", "Deploy to cloud infrastructure"],
    )


@pytest.fixture
def sample_clarification_request():
    """Create a sample ClarificationRequest for testing.

    Provides a reusable ClarificationRequest with a plan containing a sample
    spec and empty answers list. Useful for API and service layer tests.

    Returns:
        ClarificationRequest: Sample clarification request object
    """
    from app.models.specs import ClarificationRequest, PlanInput, SpecInput

    spec = SpecInput(
        purpose="Build a web application",
        vision="Modern and scalable",
        must=["Feature A", "Feature B"],
        dont=["Anti-pattern X"],
        nice=["Nice feature Y"],
        open_questions=["Question 1?", "Question 2?"],
        assumptions=["Assumption Z"],
    )
    plan = PlanInput(specs=[spec])
    return ClarificationRequest(plan=plan, answers=[])


# =============================================================================
# Deterministic Test Helpers
# =============================================================================


@pytest.fixture
def deterministic_job_timing():
    """Configure deterministic job timing for tests.

    This fixture can be used to patch time-related functions to make async job
    tests deterministic. Returns a context manager that can be used to control
    timing in job processing tests.

    Yields:
        Dict with timing configuration
    """
    return {
        "poll_interval": 0.01,  # Fast polling for tests (10ms)
        "max_retries": 10,  # Cap retries to prevent hangs
        "timeout": 2.0,  # 2 second timeout for tests
    }

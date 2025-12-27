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
"""Tests for the FastAPI application factory and configuration."""

import os
import uuid
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_create_app():
    """Test that create_app returns a FastAPI instance."""
    app = create_app()
    assert isinstance(app, FastAPI)


def test_app_metadata():
    """Test that the app has correct metadata."""
    app = create_app()

    assert app.title == "Agent Foundry Clarification Service"
    assert app.version == "0.1.0"
    assert "asynchronously clarifying specifications" in app.description


def test_app_has_openapi_endpoints():
    """Test that OpenAPI documentation endpoints are available."""
    app = create_app()
    client = TestClient(app)

    # Test /docs endpoint exists (Swagger UI)
    response = client.get("/docs")
    assert response.status_code == 200

    # Test /openapi.json endpoint exists
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Agent Foundry Clarification Service"


def test_unknown_route_returns_404():
    """Test that unknown routes return 404."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_global_exception_handler_in_production():
    """Test that the global exception handler catches unhandled exceptions in production mode."""
    # Ensure we're not in debug mode
    with patch.dict(os.environ, {"APP_DEBUG": "false"}, clear=False):
        get_settings.cache_clear()
        app = create_app()

        @app.get("/test-error")
        async def raise_error():
            raise ValueError("Test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-error")

        assert response.status_code == 500
        # Should return sanitized error with correlation_id
        response_json = response.json()
        assert response_json["detail"] == "Internal server error"
        assert "correlation_id" in response_json
        # correlation_id should be a valid UUID string
        uuid.UUID(response_json["correlation_id"])  # Should not raise


def test_exception_handler_not_registered_in_debug_mode():
    """Test that the exception handler is not registered in debug mode."""
    # Set debug mode
    with patch.dict(os.environ, {"APP_DEBUG": "true"}, clear=False):
        get_settings.cache_clear()
        app = create_app()

        @app.get("/test-error")
        async def raise_error():
            raise ValueError("Test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-error")

        # In debug mode, FastAPI's default error handling shows the full traceback
        assert response.status_code == 500


def test_cors_middleware_configured():
    """Test that CORS middleware is configured on the app."""
    app = create_app()

    # Check that CORSMiddleware is in the middleware stack
    # Middleware is wrapped in Middleware objects, so we check the cls attribute
    from starlette.middleware.cors import CORSMiddleware

    middleware_classes = [m.cls for m in app.user_middleware]
    assert CORSMiddleware in middleware_classes


def test_cors_allows_configured_origins():
    """Test that CORS allows requests from configured origins."""
    app = create_app()
    client = TestClient(app)

    # Test with default localhost origin
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_non_configured_origins():
    """Test that CORS rejects requests from non-configured origins."""
    app = create_app()
    client = TestClient(app)

    # Test with non-configured origin
    response = client.get("/health", headers={"Origin": "https://evil.com"})
    assert response.status_code == 200
    # The access-control-allow-origin header should not be present for a disallowed origin
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_request():
    """Test that CORS preflight requests are handled correctly."""
    app = create_app()
    client = TestClient(app)

    # Send OPTIONS preflight request
    response = client.options(
        "/v1/clarifications/preview",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers


def test_cors_with_credentials():
    """Test that CORS allows credentials when configured."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-credentials") == "true"


class TestBackgroundTasksWiring:
    """Tests for BackgroundTasks integration with async job processing."""

    def test_background_tasks_available_in_endpoints(self):
        """Test that BackgroundTasks can be used in endpoints."""
        from fastapi.testclient import TestClient

        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Create a job which uses BackgroundTasks
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                    }
                ]
            },
            "answers": [],
        }

        response = client.post("/v1/clarifications", json=request_data)

        # Should succeed with 202 (job created with background task scheduled)
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "PENDING"

    def test_preview_endpoint_remains_functional_after_async_changes(self):
        """Test that preview endpoint still works synchronously."""
        from fastapi.testclient import TestClient

        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": ["Feature 1"],
                        "dont": [],
                        "nice": [],
                        "open_questions": ["Question?"],
                        "assumptions": [],
                    }
                ]
            },
            "answers": [],
        }

        response = client.post("/v1/clarifications/preview", json=request_data)

        # Should return 200 with immediate result
        assert response.status_code == 200
        data = response.json()
        assert "specs" in data
        assert len(data["specs"]) == 1
        assert data["specs"][0]["purpose"] == "Test"
        # open_questions should not be in preview result
        assert "open_questions" not in data["specs"][0]

    def test_openapi_documentation_includes_both_sync_and_async_endpoints(self):
        """Test that OpenAPI schema documents both preview and async endpoints."""
        from fastapi.testclient import TestClient

        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        # Preview endpoint (synchronous)
        assert "/v1/clarifications/preview" in paths
        assert "post" in paths["/v1/clarifications/preview"]

        # Async endpoints
        assert "/v1/clarifications" in paths
        assert "post" in paths["/v1/clarifications"]

        assert "/v1/clarifications/{job_id}" in paths
        assert "get" in paths["/v1/clarifications/{job_id}"]

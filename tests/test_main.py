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
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import create_app
from app.config import get_settings


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
    """Test that the app has correct metadata from settings."""
    settings = get_settings()
    app = create_app()
    
    assert app.title == settings.app_name
    assert app.version == settings.app_version
    assert app.description == settings.app_description


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
    assert response.json()["info"]["title"] == get_settings().app_name


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
        assert response.json() == {"detail": "Internal server error"}


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

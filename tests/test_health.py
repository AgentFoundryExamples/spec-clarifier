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
"""Tests for health check endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.utils.metrics import get_metrics_collector


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics before each test."""
    metrics = get_metrics_collector()
    metrics.reset()
    yield
    metrics.reset()


def test_health_check_returns_ok(client):
    """Test that /health endpoint returns status ok."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_check_returns_json(client):
    """Test that /health endpoint returns JSON content type."""
    response = client.get("/health")
    
    assert response.headers["content-type"] == "application/json"


def test_health_check_is_synchronous(client):
    """Test that health check endpoint works and responds quickly."""
    response = client.get("/health")
    
    # Should return immediately without blocking
    assert response.status_code == 200
    assert "status" in response.json()


def test_metrics_endpoint_exists(client):
    """Test that /v1/metrics/basic endpoint exists."""
    response = client.get("/v1/metrics/basic")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


def test_metrics_endpoint_returns_expected_counters(client):
    """Test that metrics endpoint returns all expected counters."""
    response = client.get("/v1/metrics/basic")
    data = response.json()
    
    expected_counters = {
        "jobs_queued",
        "jobs_pending", 
        "jobs_running",
        "jobs_success",
        "jobs_failed",
        "llm_errors"
    }
    
    assert set(data.keys()) == expected_counters
    assert all(isinstance(v, int) for v in data.values())

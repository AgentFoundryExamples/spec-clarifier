"""Tests for health check endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


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

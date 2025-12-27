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
"""Tests for metrics collection and endpoint."""

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


class TestMetricsEndpoint:
    """Tests for the /v1/metrics/basic endpoint."""
    
    def test_metrics_endpoint_returns_200(self, client):
        """Test that metrics endpoint returns 200 OK."""
        response = client.get("/v1/metrics/basic")
        assert response.status_code == 200
    
    def test_metrics_endpoint_returns_json(self, client):
        """Test that metrics endpoint returns JSON."""
        response = client.get("/v1/metrics/basic")
        assert response.headers["content-type"] == "application/json"
    
    def test_metrics_endpoint_initial_state(self, client):
        """Test that metrics start at zero."""
        response = client.get("/v1/metrics/basic")
        data = response.json()
        
        assert data["jobs_queued"] == 0
        assert data["jobs_pending"] == 0
        assert data["jobs_running"] == 0
        assert data["jobs_success"] == 0
        assert data["jobs_failed"] == 0
        assert data["llm_errors"] == 0
    
    def test_metrics_endpoint_shape(self, client):
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
    
    def test_metrics_reflect_job_creation(self, client):
        """Test that creating jobs updates metrics."""
        from app.models.specs import ClarificationRequest, PlanInput, SpecInput
        
        spec = SpecInput(
            purpose="Test",
            vision="Test vision",
            must=["Requirement 1"],
        )
        plan = PlanInput(specs=[spec])
        request_data = ClarificationRequest(plan=plan, answers=[])
        
        # Create a job
        response = client.post(
            "/v1/clarifications",
            json=request_data.model_dump()
        )
        assert response.status_code == 202
        
        # Check metrics - job is queued but may have already been processed
        metrics_response = client.get("/v1/metrics/basic")
        data = metrics_response.json()
        
        # At minimum, jobs_queued should be incremented
        assert data["jobs_queued"] >= 1


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""
    
    def test_increment_counter(self):
        """Test incrementing a counter."""
        metrics = get_metrics_collector()
        metrics.increment("jobs_queued")
        
        counters = metrics.get_all()
        assert counters["jobs_queued"] == 1
    
    def test_increment_counter_by_amount(self):
        """Test incrementing a counter by a specific amount."""
        metrics = get_metrics_collector()
        metrics.increment("jobs_queued", delta=5)
        
        counters = metrics.get_all()
        assert counters["jobs_queued"] == 5
    
    def test_decrement_counter(self):
        """Test decrementing a counter."""
        metrics = get_metrics_collector()
        metrics.increment("jobs_pending", delta=3)
        metrics.decrement("jobs_pending")
        
        counters = metrics.get_all()
        assert counters["jobs_pending"] == 2
    
    def test_decrement_counter_does_not_go_negative(self):
        """Test that counters don't go below zero."""
        metrics = get_metrics_collector()
        metrics.decrement("jobs_pending", delta=5)
        
        counters = metrics.get_all()
        assert counters["jobs_pending"] == 0
    
    def test_increment_unknown_counter_raises_error(self):
        """Test that incrementing unknown counter raises ValueError."""
        metrics = get_metrics_collector()
        
        with pytest.raises(ValueError, match="Unknown counter"):
            metrics.increment("invalid_counter")
    
    def test_decrement_unknown_counter_raises_error(self):
        """Test that decrementing unknown counter raises ValueError."""
        metrics = get_metrics_collector()
        
        with pytest.raises(ValueError, match="Unknown counter"):
            metrics.decrement("invalid_counter")
    
    def test_reset_clears_all_counters(self):
        """Test that reset clears all counters."""
        metrics = get_metrics_collector()
        metrics.increment("jobs_queued", delta=10)
        metrics.increment("jobs_success", delta=5)
        
        metrics.reset()
        
        counters = metrics.get_all()
        assert all(v == 0 for v in counters.values())
    
    def test_concurrent_increments(self):
        """Test thread-safe concurrent increments."""
        import threading
        
        metrics = get_metrics_collector()
        num_threads = 10
        increments_per_thread = 100
        
        def increment_counter():
            for _ in range(increments_per_thread):
                metrics.increment("jobs_queued")
        
        threads = [threading.Thread(target=increment_counter) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        counters = metrics.get_all()
        assert counters["jobs_queued"] == num_threads * increments_per_thread

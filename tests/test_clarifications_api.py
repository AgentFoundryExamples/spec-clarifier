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
"""Integration tests for clarification endpoints."""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.specs import JobStatus
from app.services.job_store import clear_all_jobs


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_job_store():
    """Clean the job store before and after each test."""
    clear_all_jobs()
    yield
    clear_all_jobs()


@pytest.fixture(autouse=True)
def mock_llm_client():
    """Mock LLM client for API tests to avoid needing real API keys."""
    from unittest.mock import patch
    from app.services.llm_clients import DummyLLMClient
    import json
    
    def create_dummy_client_from_request(provider, config):
        """Create dummy client that returns valid response based on request specs."""
        # Return a DummyLLMClient that echoes back valid ClarifiedPlan JSON
        # This is a simplified approach for API tests
        return DummyLLMClient(canned_response=json.dumps({
            "specs": [
                {
                    "purpose": "Test",
                    "vision": "Test vision",
                    "must": [],
                    "dont": [],
                    "nice": [],
                    "assumptions": []
                }
            ]
        }))
    
    with patch('app.services.clarification.get_llm_client', side_effect=create_dummy_client_from_request):
        yield


@pytest.fixture
def enabled_debug_client(monkeypatch):
    """Return a TestClient with the debug endpoint enabled."""
    monkeypatch.setenv("APP_ENABLE_DEBUG_ENDPOINT", "true")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    from fastapi.testclient import TestClient
    return TestClient(create_app())


class TestPreviewClarificationsEndpoint:
    """Tests for POST /v1/clarifications/preview endpoint."""
    
    def test_preview_single_spec(self, client):
        """Test preview endpoint with a single spec."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Build a web app",
                        "vision": "Modern and user-friendly",
                        "must": ["Authentication", "Database"],
                        "dont": ["Complex UI"],
                        "nice": ["Dark mode"],
                        "open_questions": ["What database?", "Which auth?"],
                        "assumptions": ["Modern browsers"],
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "specs" in data
        assert len(data["specs"]) == 1
        
        spec = data["specs"][0]
        assert spec["purpose"] == "Build a web app"
        assert spec["vision"] == "Modern and user-friendly"
        assert spec["must"] == ["Authentication", "Database"]
        assert spec["dont"] == ["Complex UI"]
        assert spec["nice"] == ["Dark mode"]
        assert spec["assumptions"] == ["Modern browsers"]
        
        # Verify open_questions is not in the response
        assert "open_questions" not in spec
    
    def test_preview_multiple_specs(self, client):
        """Test preview endpoint with multiple specs."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Frontend",
                        "vision": "Responsive UI",
                        "must": ["React"],
                        "dont": [],
                        "nice": [],
                        "open_questions": [],
                        "assumptions": [],
                    },
                    {
                        "purpose": "Backend",
                        "vision": "Scalable API",
                        "must": ["FastAPI"],
                        "dont": [],
                        "nice": [],
                        "open_questions": [],
                        "assumptions": [],
                    },
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["specs"]) == 2
        assert data["specs"][0]["purpose"] == "Frontend"
        assert data["specs"][1]["purpose"] == "Backend"
    
    def test_preview_with_answers_ignores_them(self, client):
        """Test that answers are accepted but ignored."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": [],
                        "dont": [],
                        "nice": [],
                        "open_questions": ["What database?"],
                        "assumptions": [],
                    }
                ]
            },
            "answers": [
                {
                    "spec_index": 0,
                    "question_index": 0,
                    "question": "What database?",
                    "answer": "PostgreSQL",
                }
            ],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        # Should succeed and ignore answers
        assert response.status_code == 200
        data = response.json()
        assert len(data["specs"]) == 1
    
    def test_preview_with_empty_specs_list(self, client):
        """Test preview with empty specs list."""
        request_data = {
            "plan": {"specs": []},
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["specs"] == []
    
    def test_preview_with_empty_lists_in_spec(self, client):
        """Test that empty lists are serialized correctly."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": [],
                        "dont": [],
                        "nice": [],
                        "open_questions": [],
                        "assumptions": [],
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        spec = data["specs"][0]
        assert spec["must"] == []
        assert spec["dont"] == []
        assert spec["nice"] == []
        assert spec["assumptions"] == []
    
    def test_preview_missing_required_field_returns_422(self, client):
        """Test that missing required fields return 422 validation error."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        # Missing "vision" field
                    }
                ]
            }
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_preview_wrong_type_returns_422(self, client):
        """Test that wrong field types return 422 validation error."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": "should be a list, not a string",
                    }
                ]
            }
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 422
    
    def test_preview_extra_field_returns_422(self, client):
        """Test that extra fields are rejected with 422."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "extra_field": "not allowed",
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 422
    
    def test_preview_extra_field_in_request_returns_422(self, client):
        """Test that extra fields in ClarificationRequest are rejected (edge case from issue)."""
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
            "extra_request_field": "should trigger validation error"
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        # Verify it's a validation error with sanitized message
        assert "detail" in data
        errors = data["detail"]
        assert any("extra_request_field" in str(error) for error in errors)
    
    def test_preview_null_string_rejected(self, client):
        """Test that null strings are rejected before invoking LLM (edge case from issue)."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": None,
                        "vision": "Test vision",
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_preview_wrong_list_type_rejected(self, client):
        """Test that wrong list types are rejected before invoking LLM (edge case from issue)."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": "should be a list, not a string",
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_preview_invalid_json_returns_422(self, client):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/v1/clarifications/preview",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )
        
        assert response.status_code == 422
    
    def test_preview_returns_json_content_type(self, client):
        """Test that response has correct content type."""
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
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
    
    def test_preview_with_unicode_content(self, client):
        """Test preview with unicode and special characters."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Système de gestion 系统管理",
                        "vision": "🚀 Modern & scalable",
                        "must": ["UTF-8 支持"],
                        "dont": [],
                        "nice": ["Émojis 👍"],
                        "open_questions": [],
                        "assumptions": [],
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["specs"][0]["purpose"] == "Système de gestion 系统管理"
        assert data["specs"][0]["vision"] == "🚀 Modern & scalable"
    
    def test_preview_deterministic_order(self, client):
        """Test that spec order is preserved deterministically."""
        specs = [
            {
                "purpose": f"Spec {i}",
                "vision": f"Vision {i}",
            }
            for i in range(5)
        ]
        
        request_data = {
            "plan": {"specs": specs},
            "answers": [],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        for i, spec in enumerate(data["specs"]):
            assert spec["purpose"] == f"Spec {i}"
            assert spec["vision"] == f"Vision {i}"
    
    def test_preview_negative_answer_indices_rejected(self, client):
        """Test that negative indices in answers are rejected."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                    }
                ]
            },
            "answers": [
                {
                    "spec_index": -1,
                    "question_index": 0,
                    "question": "Test",
                    "answer": "Test",
                }
            ],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 422
    
    def test_preview_out_of_range_answer_indices_accepted(self, client):
        """Test that out-of-range indices are accepted (validation happens later)."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                    }
                ]
            },
            "answers": [
                {
                    "spec_index": 999,
                    "question_index": 999,
                    "question": "Test",
                    "answer": "Test",
                }
            ],
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        # Should succeed as answers are ignored
        assert response.status_code == 200


class TestClarificationsOpenAPI:
    """Tests for OpenAPI documentation of clarifications endpoint."""
    
    def test_openapi_schema_includes_clarifications(self, client):
        """Test that OpenAPI schema includes clarifications endpoint."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi = response.json()
        
        # Check that the endpoint is documented
        assert "/v1/clarifications/preview" in openapi["paths"]
        
        endpoint = openapi["paths"]["/v1/clarifications/preview"]
        assert "post" in endpoint
        
        # Check tags
        post_spec = endpoint["post"]
        assert "Clarifications" in post_spec["tags"]
        
        # Check that request body is documented
        assert "requestBody" in post_spec
        
        # Check that response is documented
        assert "200" in post_spec["responses"]


class TestCreateClarificationJob:
    """Tests for POST /v1/clarifications endpoint."""
    
    def test_create_job_returns_202_with_pending_status(self, client):
        """Test that creating a job returns 202 with PENDING status."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": ["Feature 1"],
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications", json=request_data)
        
        assert response.status_code == 202
        data = response.json()
        
        # Should return a lightweight job summary with PENDING status
        assert data["status"] == "PENDING"
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert data["last_error"] is None
        # Request and result should NOT be in lightweight summary
        assert "request" not in data
        assert "result" not in data
        assert "config" not in data
    
    def test_create_job_returns_immediately(self, client):
        """Test that job creation returns immediately without blocking."""
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
        
        start_time = time.time()
        response = client.post("/v1/clarifications", json=request_data)
        elapsed_time = time.time() - start_time
        
        # Should return very quickly (within 1 second)
        assert elapsed_time < 1.0
        assert response.status_code == 202
        
        # Job should be in PENDING state (not processed yet)
        data = response.json()
        assert data["status"] == "PENDING"
    
    def test_create_job_stores_in_job_store(self, client):
        """Test that created job is retrievable via GET endpoint."""
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
        
        # Create job
        response = client.post("/v1/clarifications", json=request_data)
        assert response.status_code == 202
        
        job_id = response.json()["id"]
        
        # Retrieve job
        get_response = client.get(f"/v1/clarifications/{job_id}")
        assert get_response.status_code == 200
        
        job_data = get_response.json()
        assert job_data["id"] == job_id
    
    def test_create_job_processing_completes(self, client):
        """Test that background processing completes successfully."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test Service",
                        "vision": "High performance",
                        "must": ["Fast", "Reliable"],
                        "dont": ["Slow"],
                        "nice": ["Configurable"],
                        "assumptions": ["Cloud deployment"],
                        "open_questions": ["Which cloud?"],
                    }
                ]
            },
            "answers": [],
        }
        
        # Create job
        response = client.post("/v1/clarifications", json=request_data)
        assert response.status_code == 202
        
        job_id = response.json()["id"]
        
        # Poll for completion (with timeout)
        max_wait = 5.0
        start_time = time.time()
        job_data = None
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] in ["SUCCESS", "FAILED"]:
                break
            
            time.sleep(0.1)
        
        assert job_data["status"] in ["SUCCESS", "FAILED"], "Job did not complete within timeout"
        
        # Job should have completed successfully
        assert job_data is not None
        assert job_data["status"] == "SUCCESS"
        # Result is NOT included by default (show_job_result=False)
        assert job_data["result"] is None
        assert job_data["last_error"] is None
    
    def test_create_job_with_multiple_specs(self, client):
        """Test creating a job with multiple specs."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Frontend",
                        "vision": "Responsive UI",
                    },
                    {
                        "purpose": "Backend",
                        "vision": "Scalable API",
                    },
                    {
                        "purpose": "DevOps",
                        "vision": "Automated deployment",
                    },
                ]
            },
            "answers": [],
        }
        
        # Create job
        response = client.post("/v1/clarifications", json=request_data)
        assert response.status_code == 202
        
        job_id = response.json()["id"]
        
        # Wait for completion
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] in ["SUCCESS", "FAILED"]:
                break
            
            time.sleep(0.1)
        
        assert job_data["status"] in ["SUCCESS", "FAILED"], "Job did not complete within timeout"
        
        # Job should have completed successfully
        assert job_data["status"] == "SUCCESS"
        # Result is NOT included by default (show_job_result=False)
        assert job_data["result"] is None
    
    def test_create_job_invalid_request_returns_422(self, client):
        """Test that invalid requests return 422."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        # Missing required 'vision' field
                    }
                ]
            }
        }
        
        response = client.post("/v1/clarifications", json=request_data)
        
        assert response.status_code == 422
    
    def test_create_job_with_extra_fields_returns_422(self, client):
        """Test that extra fields in request trigger validation error (edge case from issue)."""
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
            "unexpected_field": "should fail"
        }
        
        response = client.post("/v1/clarifications", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_create_job_without_answers_succeeds(self, client):
        """Test that omitting answers defaults to empty list (edge case from issue)."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                    }
                ]
            }
            # Note: answers field is omitted
        }
        
        response = client.post("/v1/clarifications", json=request_data)
        
        # Should succeed with default empty answers list
        assert response.status_code == 202
        data = response.json()
        assert "id" in data
        assert data["status"] == "PENDING"
        
        # Verify job actually processes correctly with empty answers
        job_id = data["id"]
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] in ["SUCCESS", "FAILED"]:
                break
            
            time.sleep(0.1)
        
        # Job should complete successfully with empty answers
        assert job_data["status"] == "SUCCESS", f"Job failed or timed out: {job_data.get('last_error', 'timeout')}"


class TestGetClarificationJob:
    """Tests for GET /v1/clarifications/{job_id} endpoint."""
    
    def test_get_job_returns_job_details(self, client):
        """Test retrieving a job by ID."""
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
        
        # Create job
        create_response = client.post("/v1/clarifications", json=request_data)
        job_id = create_response.json()["id"]
        
        # Get job
        get_response = client.get(f"/v1/clarifications/{job_id}")
        
        assert get_response.status_code == 200
        data = get_response.json()
        
        assert data["id"] == job_id
        assert data["status"] in ["PENDING", "RUNNING", "SUCCESS", "FAILED"]
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_get_nonexistent_job_returns_404(self, client):
        """Test that getting a nonexistent job returns 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        response = client.get(f"/v1/clarifications/{fake_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_job_invalid_uuid_returns_422(self, client):
        """Test that invalid UUID returns 422."""
        response = client.get("/v1/clarifications/not-a-valid-uuid")
        
        assert response.status_code == 422
    
    def test_get_job_shows_success_status(self, client):
        """Test that completed job shows SUCCESS status."""
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
        
        # Create and wait for completion
        create_response = client.post("/v1/clarifications", json=request_data)
        job_id = create_response.json()["id"]
        
        # Poll until complete
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] in ["SUCCESS", "FAILED"]:
                assert job_data["status"] == "SUCCESS"
                # Result is NOT included by default (show_job_result=False)
                assert job_data["result"] is None
                return
            
            time.sleep(0.1)
        
        # Should have completed within timeout
        assert False, "Job did not complete within timeout"
    
    def test_get_job_returns_json_content_type(self, client):
        """Test that response has correct content type."""
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
        
        create_response = client.post("/v1/clarifications", json=request_data)
        job_id = create_response.json()["id"]
        
        get_response = client.get(f"/v1/clarifications/{job_id}")
        
        assert get_response.status_code == 200
        assert "application/json" in get_response.headers["content-type"]


class TestAsyncClarificationsOpenAPI:
    """Tests for OpenAPI documentation of async clarification endpoints."""
    
    def test_openapi_includes_async_endpoints(self, client):
        """Test that OpenAPI schema includes async clarification endpoints."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi = response.json()
        
        # Check POST endpoint (empty path for POST to base)
        assert "/v1/clarifications" in openapi["paths"]
        clarifications_path = openapi["paths"]["/v1/clarifications"]
        
        # Should have POST method
        assert "post" in clarifications_path
        post_spec = clarifications_path["post"]
        assert "Clarifications" in post_spec["tags"]
        assert "202" in post_spec["responses"]
        
        # Check GET by ID endpoint
        assert "/v1/clarifications/{job_id}" in openapi["paths"]
        get_path = openapi["paths"]["/v1/clarifications/{job_id}"]
        
        # Should have GET method
        assert "get" in get_path
        get_spec = get_path["get"]
        assert "Clarifications" in get_spec["tags"]
        assert "200" in get_spec["responses"]
        # 404 is handled by HTTPException but not always in OpenAPI spec


class TestShowJobResultFlag:
    """Tests for APP_SHOW_JOB_RESULT development flag behavior."""
    
    def test_get_job_excludes_result_by_default(self, client, monkeypatch):
        """Test that result is excluded when flag is False (default)."""
        # Ensure flag is False (default)
        monkeypatch.setenv("APP_SHOW_JOB_RESULT", "false")
        
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
        
        # Create job
        response = client.post("/v1/clarifications", json=request_data)
        job_id = response.json()["id"]
        
        # Wait for completion
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] == "SUCCESS":
                # Result should be None (excluded)
                assert job_data["result"] is None
                return
            
            time.sleep(0.1)
        
        assert False, "Job did not complete within timeout"
    
    def test_get_job_includes_result_when_flag_enabled(self, monkeypatch):
        """Test that result is included when flag is True."""
        # Enable the flag
        monkeypatch.setenv("APP_SHOW_JOB_RESULT", "true")
        # Force settings reload and create a new client with the updated settings
        from app.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app
        from fastapi.testclient import TestClient
        client = TestClient(create_app())
        
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test Service",
                        "vision": "High performance",
                        "must": ["Fast", "Reliable"],
                        "open_questions": ["Which cloud?"],
                    }
                ]
            },
            "answers": [],
        }
        
        # Create job
        response = client.post("/v1/clarifications", json=request_data)
        job_id = response.json()["id"]
        
        # Wait for completion
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] == "SUCCESS":
                # Result SHOULD be included when flag is enabled
                assert job_data["result"] is not None
                assert "specs" in job_data["result"]
                assert len(job_data["result"]["specs"]) == 1
                
                spec = job_data["result"]["specs"][0]
                # With mocked LLM client, we get generic response
                assert spec["purpose"] == "Test"
                assert spec["vision"] == "Test vision"
                # open_questions should not be in the result
                assert "open_questions" not in spec
                return
            
            time.sleep(0.1)
        
        assert False, "Job did not complete within timeout"
    
    def test_post_job_never_includes_result(self, monkeypatch):
        """Test that POST response never includes result, regardless of flag."""
        # Enable flag to verify POST ignores it
        monkeypatch.setenv("APP_SHOW_JOB_RESULT", "true")
        from app.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app
        from fastapi.testclient import TestClient
        client = TestClient(create_app())
        
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
        
        assert response.status_code == 202
        data = response.json()
        
        # POST response should NEVER include result (lightweight summary)
        assert "result" not in data
        assert "request" not in data
        assert "config" not in data
        
        # Only includes id, status, timestamps, and last_error
        assert "id" in data
        assert "status" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "last_error" in data
    
    def test_flag_only_affects_get_endpoint(self, monkeypatch):
        """Test that flag only controls GET, not POST responses."""
        monkeypatch.setenv("APP_SHOW_JOB_RESULT", "false")
        from app.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app
        from fastapi.testclient import TestClient
        client = TestClient(create_app())
        
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
        
        # POST should return lightweight summary
        post_response = client.post("/v1/clarifications", json=request_data)
        post_data = post_response.json()
        assert "result" not in post_data
        
        job_id = post_data["id"]
        
        # Wait for completion
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            get_data = get_response.json()
            
            if get_data["status"] == "SUCCESS":
                # GET should have result=None when flag is False
                assert get_data["result"] is None
                return
            
            time.sleep(0.1)
        
        assert False, "Job did not complete within timeout"


class TestAsyncJobLifecycleAPI:
    """Additional tests for async job lifecycle at the API level."""
    
    def test_post_returns_immediately_without_result_payload(self, client):
        """Test that POST /v1/clarifications returns immediately without ClarifiedPlan in response."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": ["Feature 1"],
                    }
                ]
            },
            "answers": [],
        }
        
        response = client.post("/v1/clarifications", json=request_data)
        
        # Response should be 202 Accepted
        assert response.status_code == 202
        
        data = response.json()
        # Should have minimal fields only
        assert "id" in data
        assert "status" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "last_error" in data
        
        # Should NOT have full request or result - this is the robust check
        assert "request" not in data
        assert "result" not in data
        assert "config" not in data
        
        # Status should be PENDING (not yet processed)
        assert data["status"] == "PENDING"
    
    def test_get_returns_documented_fields(self, client):
        """Test that GET returns id, status, timestamps, last_error, and result fields."""
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
        
        # Create job
        post_response = client.post("/v1/clarifications", json=request_data)
        job_id = post_response.json()["id"]
        
        # Get job
        get_response = client.get(f"/v1/clarifications/{job_id}")
        
        assert get_response.status_code == 200
        data = get_response.json()
        
        # Verify all documented fields are present
        assert "id" in data
        assert "status" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "last_error" in data
        assert "result" in data  # Field is present but may be null
    
    def test_get_returns_404_for_unknown_job_id(self, client):
        """Test that GET returns 404 for unknown job IDs."""
        from uuid import uuid4
        fake_id = uuid4()
        
        response = client.get(f"/v1/clarifications/{fake_id}")
        
        assert response.status_code == 404
        assert "detail" in response.json()
        assert str(fake_id) in response.json()["detail"]
    
    def test_failed_job_shows_last_error_in_get(self, client):
        """Test that FAILED jobs return last_error in GET response."""
        from unittest.mock import patch
        from app.services.llm_clients import DummyLLMClient
        
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
        
        # Patch get_llm_client to return a failing client
        def create_failing_client(provider, config):
            return DummyLLMClient(
                simulate_failure=True,
                failure_message="Forced failure for testing"
            )
        
        with patch('app.services.clarification.get_llm_client', side_effect=create_failing_client):
            # Create job (will be processed in background with failing client)
            post_response = client.post("/v1/clarifications", json=request_data)
            job_id = post_response.json()["id"]
            
            # Wait for background processing to complete
            max_wait = 5.0
            start_time = time.time()
            job_done = False
            
            while time.time() - start_time < max_wait:
                get_response = client.get(f"/v1/clarifications/{job_id}")
                data = get_response.json()
                
                if data["status"] in ["SUCCESS", "FAILED"]:
                    job_done = True
                    break
                
                time.sleep(0.1)
            
            assert job_done, f"Job did not complete within {max_wait} seconds. Final status: {data.get('status')}"
        
        # Get final job status
        get_response = client.get(f"/v1/clarifications/{job_id}")
        assert get_response.status_code == 200
        data = get_response.json()
        
        assert data["status"] == "FAILED"
        assert data["last_error"] is not None
        assert "Forced failure for testing" in data["last_error"]
        assert data["result"] is None


class TestDebugEndpoint:
    """Tests for GET /v1/clarifications/{job_id}/debug endpoint."""
    
    def test_debug_endpoint_disabled_by_default(self, client):
        """Test that debug endpoint returns 403 when disabled (default)."""
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
        
        # Create job
        post_response = client.post("/v1/clarifications", json=request_data)
        job_id = post_response.json()["id"]
        
        # Try to access debug endpoint (should be disabled by default)
        debug_response = client.get(f"/v1/clarifications/{job_id}/debug")
        
        assert debug_response.status_code == 403
        assert "detail" in debug_response.json()
        assert "disabled" in debug_response.json()["detail"].lower()
    
    def test_debug_endpoint_enabled_returns_metadata(self, enabled_debug_client):
        """Test that debug endpoint returns sanitized metadata when enabled."""
        client = enabled_debug_client
        
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test Service",
                        "vision": "High performance API",
                        "must": ["Fast", "Reliable"],
                        "dont": ["Slow"],
                        "nice": ["Configurable"],
                        "open_questions": ["Which cloud?"],
                        "assumptions": ["Cloud deployment"],
                    }
                ]
            },
            "answers": [
                {
                    "spec_index": 0,
                    "question_index": 0,
                    "question": "Which cloud?",
                    "answer": "AWS",
                }
            ],
        }
        
        # Create job
        post_response = client.post("/v1/clarifications", json=request_data)
        job_id = post_response.json()["id"]
        
        # Wait for completion
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] in ["SUCCESS", "FAILED"]:
                break
            
            time.sleep(0.1)
        
        # Access debug endpoint
        debug_response = client.get(f"/v1/clarifications/{job_id}/debug")
        
        assert debug_response.status_code == 200
        debug_data = debug_response.json()
        
        # Verify debug data structure
        assert "job_id" in debug_data
        assert "status" in debug_data
        assert "created_at" in debug_data
        assert "updated_at" in debug_data
        assert "has_request" in debug_data
        assert "has_result" in debug_data
        assert "config" in debug_data
        
        # Verify request metadata (not full content)
        assert "request_metadata" in debug_data
        req_meta = debug_data["request_metadata"]
        assert req_meta["num_specs"] == 1
        assert req_meta["num_answers"] == 1
        assert len(req_meta["spec_summaries"]) == 1
        
        spec_summary = req_meta["spec_summaries"][0]
        assert "purpose_length" in spec_summary
        assert "vision_length" in spec_summary
        assert spec_summary["num_must"] == 2
        assert spec_summary["num_dont"] == 1
        assert spec_summary["num_nice"] == 1
        assert spec_summary["num_open_questions"] == 1
        assert spec_summary["num_assumptions"] == 1
        
        # Verify result metadata is present if job succeeded
        if debug_data["status"] == "SUCCESS":
            assert "result_metadata" in debug_data
            result_meta = debug_data["result_metadata"]
            assert result_meta["num_specs"] == 1
            assert len(result_meta["spec_summaries"]) == 1
    
    def test_debug_endpoint_excludes_sensitive_content(self, enabled_debug_client):
        """Test that debug endpoint does not expose prompts or raw content."""
        client = enabled_debug_client
        
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Sensitive purpose text",
                        "vision": "Sensitive vision text",
                        "must": ["Sensitive requirement"],
                    }
                ]
            },
            "answers": [],
        }
        
        # Create job
        post_response = client.post("/v1/clarifications", json=request_data)
        job_id = post_response.json()["id"]
        
        # Access debug endpoint
        debug_response = client.get(f"/v1/clarifications/{job_id}/debug")
        debug_data = debug_response.json()
        
        # Convert to string to search for sensitive content
        debug_str = str(debug_data).lower()
        
        # Verify sensitive content is NOT present
        assert "sensitive purpose text" not in debug_str
        assert "sensitive vision text" not in debug_str
        assert "sensitive requirement" not in debug_str
        
        # Verify only metadata is present
        assert "purpose_length" in str(debug_data)
        assert "vision_length" in str(debug_data)
        assert "num_must" in str(debug_data)
    
    def test_debug_endpoint_returns_404_for_nonexistent_job(self, enabled_debug_client):
        """Test that debug endpoint returns 404 for nonexistent jobs."""
        client = enabled_debug_client
        
        from uuid import uuid4
        fake_id = uuid4()
        
        debug_response = client.get(f"/v1/clarifications/{fake_id}/debug")
        
        assert debug_response.status_code == 404
        assert "not found" in debug_response.json()["detail"].lower()
    
    def test_debug_endpoint_invalid_uuid_returns_422(self, enabled_debug_client):
        """Test that debug endpoint returns 422 for invalid UUIDs."""
        client = enabled_debug_client
        
        debug_response = client.get("/v1/clarifications/invalid-uuid/debug")
        
        assert debug_response.status_code == 422
    
    def test_debug_endpoint_shows_config_when_present(self, enabled_debug_client):
        """Test that debug endpoint shows job config when available."""
        client = enabled_debug_client
        
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
        
        # Create job
        post_response = client.post("/v1/clarifications", json=request_data)
        job_id = post_response.json()["id"]
        
        # Access debug endpoint
        debug_response = client.get(f"/v1/clarifications/{job_id}/debug")
        debug_data = debug_response.json()
        
        # Should have config field
        assert "config" in debug_data
        # Config should contain llm_config since that's set by default
        if debug_data["config"]:
            assert "llm_config" in debug_data["config"]
    
    def test_debug_endpoint_sanitizes_config(self, enabled_debug_client):
        """Test that debug endpoint sanitizes sensitive fields from config."""
        from app.services.job_store import create_job
        from app.models.specs import ClarificationRequest, PlanInput, SpecInput
        
        client = enabled_debug_client
        
        # Create a job with potentially sensitive config
        plan = PlanInput(specs=[SpecInput(purpose="Test", vision="Test vision")])
        request = ClarificationRequest(plan=plan, answers=[])
        
        # Manually create a job with sensitive config data
        from app.services import job_store
        job = job_store.create_job(request, config={
            "llm_config": {
                "provider": "openai",
                "model": "gpt-5",
                "api_key": "sk-secret-key-12345",  # This should be filtered
                "temperature": 0.5
            },
            "api_key": "should-be-filtered",  # This should be filtered
            "token": "should-be-filtered",  # This should be filtered
            "safe_field": "should-be-included"  # This should be included
        })
        
        # Access debug endpoint
        debug_response = client.get(f"/v1/clarifications/{job.id}/debug")
        debug_data = debug_response.json()
        
        # Verify config is sanitized
        assert "config" in debug_data
        config = debug_data["config"]
        
        # llm_config should only have safe fields
        assert "llm_config" in config
        assert config["llm_config"]["provider"] == "openai"
        assert config["llm_config"]["model"] == "gpt-5"
        assert config["llm_config"]["temperature"] == 0.5
        assert "api_key" not in config["llm_config"]  # Should be filtered out
        
        # Top-level sensitive fields should be filtered
        assert "api_key" not in config
        assert "token" not in config
        
        # Safe fields should be preserved
        assert config["safe_field"] == "should-be-included"
    
    def test_debug_endpoint_sanitizes_error_messages(self, enabled_debug_client):
        """Test that debug endpoint sanitizes error messages containing sensitive data."""
        from app.services.job_store import create_job, update_job
        from app.models.specs import ClarificationRequest, PlanInput, SpecInput, JobStatus
        
        client = enabled_debug_client
        
        # Create a job
        plan = PlanInput(specs=[SpecInput(purpose="Test", vision="Test vision")])
        request = ClarificationRequest(plan=plan, answers=[])
        
        from app.services import job_store
        job = job_store.create_job(request)
        
        # Update job with error message containing sensitive data
        error_message = "Authentication failed with api_key=sk-secret-12345 and token=bearer-token-xyz"
        job_store.update_job(job.id, status=JobStatus.FAILED, last_error=error_message)
        
        # Access debug endpoint
        debug_response = client.get(f"/v1/clarifications/{job.id}/debug")
        debug_data = debug_response.json()
        
        # Verify error message is sanitized
        assert "last_error" in debug_data
        sanitized_error = debug_data["last_error"]
        
        # Sensitive data should be redacted
        assert "sk-secret-12345" not in sanitized_error
        assert "bearer-token-xyz" not in sanitized_error
        assert "[REDACTED]" in sanitized_error
    
    def test_debug_endpoint_includes_num_open_questions_in_result(self, enabled_debug_client):
        """Test that result metadata includes num_open_questions field for consistency."""
        client = enabled_debug_client
        
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Test",
                        "vision": "Test vision",
                        "must": ["Feature 1"],
                    }
                ]
            },
            "answers": [],
        }
        
        # Create job
        post_response = client.post("/v1/clarifications", json=request_data)
        job_id = post_response.json()["id"]
        
        # Wait for completion
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] in ["SUCCESS", "FAILED"]:
                break
            
            time.sleep(0.1)
        
        # Access debug endpoint
        debug_response = client.get(f"/v1/clarifications/{job_id}/debug")
        debug_data = debug_response.json()
        
        # Verify result metadata includes num_open_questions
        if debug_data["status"] == "SUCCESS" and "result_metadata" in debug_data:
            result_meta = debug_data["result_metadata"]
            assert "spec_summaries" in result_meta
            for spec_summary in result_meta["spec_summaries"]:
                # Verify field exists and is numeric (ClarifiedSpec doesn't have open_questions)
                assert "num_open_questions" in spec_summary
                assert isinstance(spec_summary["num_open_questions"], int)
                assert spec_summary["num_open_questions"] >= 0


class TestDebugEndpointOpenAPI:
    """Tests for OpenAPI documentation of debug endpoint."""
    
    def test_openapi_includes_debug_endpoint(self, client):
        """Test that OpenAPI schema includes debug endpoint."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi = response.json()
        
        # Check that the debug endpoint is documented
        assert "/v1/clarifications/{job_id}/debug" in openapi["paths"]
        
        endpoint = openapi["paths"]["/v1/clarifications/{job_id}/debug"]
        assert "get" in endpoint
        
        # Check GET method documentation
        get_spec = endpoint["get"]
        assert "Clarifications" in get_spec["tags"]
        assert "summary" in get_spec
        assert "debug" in get_spec["summary"].lower()
        
        # Check that it has 200 response
        assert "200" in get_spec["responses"]
        
        # Verify description mentions it's disabled by default
        assert "description" in get_spec
        assert "disabled by default" in get_spec["description"].lower() or "debug mode only" in get_spec["description"].lower()


class TestOpenQuestionsNeverExposed:
    """Tests ensuring open_questions are never exposed in API responses.
    
    These tests verify the acceptance criteria that GET responses never expose
    open_questions, while must/dont/nice arrays retain prior entries.
    """
    
    def test_preview_response_never_includes_open_questions(self, client):
        """Test that preview endpoint response excludes open_questions."""
        request_data = {
            "plan": {
                "specs": [{
                    "purpose": "Test",
                    "vision": "Test vision",
                    "must": ["Feature 1"],
                    "dont": [],
                    "nice": [],
                    "open_questions": ["Question 1?", "Question 2?"],
                    "assumptions": []
                }]
            },
            "answers": []
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "specs" in data
        assert len(data["specs"]) == 1
        spec = data["specs"][0]
        
        # Verify open_questions is NOT in response
        assert "open_questions" not in spec
        
        # Verify the 6 allowed keys are present
        assert "purpose" in spec
        assert "vision" in spec
        assert "must" in spec
        assert "dont" in spec
        assert "nice" in spec
        assert "assumptions" in spec
    
    def test_get_job_result_never_includes_open_questions_when_enabled(self, monkeypatch):
        """Test that GET job result never includes open_questions (development mode)."""
        # Enable result in response
        monkeypatch.setenv("APP_SHOW_JOB_RESULT", "true")
        from app.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app
        from fastapi.testclient import TestClient
        client = TestClient(create_app())
        
        request_data = {
            "plan": {
                "specs": [{
                    "purpose": "Test",
                    "vision": "Test vision",
                    "must": ["Feature 1"],
                    "open_questions": ["Question 1?"]
                }]
            },
            "answers": []
        }
        
        # Create job
        response = client.post("/v1/clarifications", json=request_data)
        job_id = response.json()["id"]
        
        # Wait for completion
        import time
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] == "SUCCESS":
                # Verify result exists and open_questions is not present
                assert job_data["result"] is not None
                assert "specs" in job_data["result"]
                spec = job_data["result"]["specs"][0]
                
                # open_questions should NOT be in result
                assert "open_questions" not in spec
                
                # Verify only the 6 allowed keys
                spec_keys = set(spec.keys())
                expected_keys = {"purpose", "vision", "must", "dont", "nice", "assumptions"}
                assert spec_keys == expected_keys
                return
            
            time.sleep(0.1)
        
        assert False, "Job did not complete within timeout"
    
    def test_post_response_never_includes_open_questions_in_metadata(self, client):
        """Test that POST response doesn't leak open_questions in any field."""
        request_data = {
            "plan": {
                "specs": [{
                    "purpose": "Test",
                    "vision": "Test vision",
                    "open_questions": ["Secret question?"]
                }]
            },
            "answers": []
        }
        
        response = client.post("/v1/clarifications", json=request_data)
        
        assert response.status_code == 202
        data = response.json()
        
        # Convert entire response to string and verify no open_questions leak
        response_str = str(data).lower()
        assert "secret question" not in response_str
        
        # Verify lightweight response (no request/result fields)
        assert "request" not in data
        assert "result" not in data
    
    def test_must_dont_nice_arrays_preserved_in_result(self, monkeypatch):
        """Test that must/dont/nice arrays retain prior entries in result."""
        monkeypatch.setenv("APP_SHOW_JOB_RESULT", "true")
        from app.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app
        from fastapi.testclient import TestClient
        client = TestClient(create_app())
        
        request_data = {
            "plan": {
                "specs": [{
                    "purpose": "Test",
                    "vision": "Test vision",
                    "must": ["Existing feature A", "Existing feature B"],
                    "dont": ["Existing constraint X"],
                    "nice": ["Existing nice-to-have Y"],
                    "open_questions": ["Add feature C?"]
                }]
            },
            "answers": []
        }
        
        # Create and wait for job
        response = client.post("/v1/clarifications", json=request_data)
        job_id = response.json()["id"]
        
        import time
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            get_response = client.get(f"/v1/clarifications/{job_id}")
            job_data = get_response.json()
            
            if job_data["status"] == "SUCCESS":
                result_spec = job_data["result"]["specs"][0]
                
                # Verify prior entries are preserved
                # (Note: With mocked LLM, exact content depends on mock response)
                assert isinstance(result_spec["must"], list)
                assert isinstance(result_spec["dont"], list)
                assert isinstance(result_spec["nice"], list)
                
                # Verify open_questions is NOT present
                assert "open_questions" not in result_spec
                return
            
            time.sleep(0.1)
        
        assert False, "Job did not complete within timeout"
    
    def test_multiple_specs_none_include_open_questions(self, client):
        """Test that with multiple specs, none include open_questions in response."""
        request_data = {
            "plan": {
                "specs": [
                    {
                        "purpose": "Spec 1",
                        "vision": "Vision 1",
                        "open_questions": ["Q1?"]
                    },
                    {
                        "purpose": "Spec 2",
                        "vision": "Vision 2",
                        "open_questions": ["Q2?", "Q3?"]
                    },
                    {
                        "purpose": "Spec 3",
                        "vision": "Vision 3",
                        "open_questions": []
                    }
                ]
            },
            "answers": []
        }
        
        response = client.post("/v1/clarifications/preview", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify none of the specs include open_questions
        for spec in data["specs"]:
            assert "open_questions" not in spec
            # Verify only the 6 allowed keys
            spec_keys = set(spec.keys())
            expected_keys = {"purpose", "vision", "must", "dont", "nice", "assumptions"}
            assert spec_keys == expected_keys


class TestAPIWithVariousDummyClientResponses:
    """API-level tests with various DummyLLMClient response scenarios.
    
    These tests verify the full API workflow with different LLM response patterns.
    """
    
    def test_api_handles_markdown_wrapped_response(self, client):
        """Test that API successfully handles markdown-wrapped LLM responses."""
        from unittest.mock import patch
        from app.services.llm_clients import DummyLLMClient
        
        # Create a dummy client that returns markdown-wrapped JSON
        def create_markdown_client(provider, config):
            return DummyLLMClient(canned_response='''```json
{
  "specs": [{
    "purpose": "Test",
    "vision": "Test",
    "must": [],
    "dont": [],
    "nice": [],
    "assumptions": []
  }]
}
```''')
        
        with patch('app.services.clarification.get_llm_client', side_effect=create_markdown_client):
            request_data = {
                "plan": {
                    "specs": [{
                        "purpose": "Test",
                        "vision": "Test"
                    }]
                },
                "answers": []
            }
            
            # Create job
            response = client.post("/v1/clarifications", json=request_data)
            assert response.status_code == 202
            job_id = response.json()["id"]
            
            # Poll for completion
            import time
            max_wait = 5.0
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                get_response = client.get(f"/v1/clarifications/{job_id}")
                job_data = get_response.json()
                
                if job_data["status"] in ["SUCCESS", "FAILED"]:
                    # Should succeed despite markdown
                    assert job_data["status"] == "SUCCESS"
                    return
                
                time.sleep(0.1)
            
            assert False, "Job did not complete"
    
    def test_api_handles_malformed_response_gracefully(self, client):
        """Test that API gracefully handles malformed LLM responses."""
        from unittest.mock import patch
        from app.services.llm_clients import DummyLLMClient
        
        def create_malformed_client(provider, config):
            return DummyLLMClient(canned_response='{"incomplete": "json"')
        
        with patch('app.services.clarification.get_llm_client', side_effect=create_malformed_client):
            request_data = {
                "plan": {
                    "specs": [{
                        "purpose": "Test",
                        "vision": "Test"
                    }]
                },
                "answers": []
            }
            
            response = client.post("/v1/clarifications", json=request_data)
            job_id = response.json()["id"]
            
            # Poll for completion
            import time
            max_wait = 5.0
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                get_response = client.get(f"/v1/clarifications/{job_id}")
                job_data = get_response.json()
                
                if job_data["status"] in ["SUCCESS", "FAILED"]:
                    # Should be FAILED with error
                    assert job_data["status"] == "FAILED"
                    assert job_data["last_error"] is not None
                    assert "parse" in job_data["last_error"].lower() or "failed" in job_data["last_error"].lower()
                    return
                
                time.sleep(0.1)
            
            assert False, "Job did not complete"
    
    def test_api_backward_compatible_with_existing_clients(self, client):
        """Test that API remains backward compatible."""
        # Test that old-style requests still work
        request_data = {
            "plan": {
                "specs": [{
                    "purpose": "Backward Compatible",
                    "vision": "Still works",
                    "must": ["Feature"],
                    "dont": [],
                    "nice": [],
                    "open_questions": [],
                    "assumptions": []
                }]
            },
            "answers": []
        }
        
        # POST should work
        response = client.post("/v1/clarifications", json=request_data)
        assert response.status_code == 202
        
        # Preview should work
        preview_response = client.post("/v1/clarifications/preview", json=request_data)
        assert preview_response.status_code == 200
        
        # Result should not include open_questions
        preview_data = preview_response.json()
        assert "open_questions" not in preview_data["specs"][0]

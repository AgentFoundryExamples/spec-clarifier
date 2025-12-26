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
        assert "clarifications" in post_spec["tags"]
        
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
        
        # Should return a job with PENDING status
        assert data["status"] == "PENDING"
        assert "id" in data
        # Request should be stored (with defaults filled in by Pydantic)
        assert data["request"]["plan"]["specs"][0]["purpose"] == "Test"
        assert data["request"]["plan"]["specs"][0]["must"] == ["Feature 1"]
        assert data["result"] is None
        assert data["last_error"] is None
        assert "created_at" in data
        assert "updated_at" in data
    
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
        assert job_data["result"] is not None
        assert job_data["last_error"] is None
        
        # Result should contain the clarified spec
        result = job_data["result"]
        assert len(result["specs"]) == 1
        
        spec = result["specs"][0]
        assert spec["purpose"] == "Test Service"
        assert spec["vision"] == "High performance"
        assert spec["must"] == ["Fast", "Reliable"]
        assert spec["dont"] == ["Slow"]
        assert spec["nice"] == ["Configurable"]
        assert spec["assumptions"] == ["Cloud deployment"]
        # open_questions should not be in the result
        assert "open_questions" not in spec
    
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
        
        # Should have all specs processed
        assert job_data["status"] == "SUCCESS"
        assert len(job_data["result"]["specs"]) == 3
        assert job_data["result"]["specs"][0]["purpose"] == "Frontend"
        assert job_data["result"]["specs"][1]["purpose"] == "Backend"
        assert job_data["result"]["specs"][2]["purpose"] == "DevOps"
    
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
                assert job_data["result"] is not None
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
        assert "clarifications" in post_spec["tags"]
        assert "202" in post_spec["responses"]
        
        # Check GET by ID endpoint
        assert "/v1/clarifications/{job_id}" in openapi["paths"]
        get_path = openapi["paths"]["/v1/clarifications/{job_id}"]
        
        # Should have GET method
        assert "get" in get_path
        get_spec = get_path["get"]
        assert "clarifications" in get_spec["tags"]
        assert "200" in get_spec["responses"]
        # 404 is handled by HTTPException but not always in OpenAPI spec

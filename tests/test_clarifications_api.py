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

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


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

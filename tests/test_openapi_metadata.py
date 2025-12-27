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
"""Tests for OpenAPI metadata, tags, and examples."""

import pytest
from fastapi.testclient import TestClient
from uuid import UUID

from app.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def openapi_schema(client):
    """Get the OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


class TestOpenAPIMetadata:
    """Test OpenAPI metadata and service information."""
    
    def test_service_title(self, openapi_schema):
        """Test that the service has the correct title."""
        assert openapi_schema["info"]["title"] == "Agent Foundry Clarification Service"
    
    def test_service_version(self, openapi_schema):
        """Test that the service has a semantic version."""
        version = openapi_schema["info"]["version"]
        assert version == "0.1.0"
        # Verify it's a semantic version format
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
    
    def test_service_description(self, openapi_schema):
        """Test that the service has a descriptive purpose statement."""
        description = openapi_schema["info"]["description"]
        assert description is not None
        assert len(description) > 50  # Should be descriptive
        assert "asynchronously" in description.lower() or "async" in description.lower()
        assert "clarifying" in description.lower() or "clarification" in description.lower()


class TestOpenAPITags:
    """Test that endpoints are properly tagged for grouping."""
    
    def test_clarifications_endpoints_have_tag(self, openapi_schema):
        """Test that clarifications endpoints have the Clarifications tag."""
        paths = openapi_schema["paths"]
        
        # POST /v1/clarifications
        assert "/v1/clarifications" in paths
        assert "post" in paths["/v1/clarifications"]
        tags = paths["/v1/clarifications"]["post"].get("tags", [])
        assert "Clarifications" in tags
        
        # GET /v1/clarifications/{job_id}
        assert "/v1/clarifications/{job_id}" in paths
        assert "get" in paths["/v1/clarifications/{job_id}"]
        tags = paths["/v1/clarifications/{job_id}"]["get"].get("tags", [])
        assert "Clarifications" in tags
    
    def test_config_endpoints_have_tag(self, openapi_schema):
        """Test that config endpoints have the Configuration tag."""
        paths = openapi_schema["paths"]
        
        # GET /v1/config/defaults
        assert "/v1/config/defaults" in paths
        assert "get" in paths["/v1/config/defaults"]
        tags = paths["/v1/config/defaults"]["get"].get("tags", [])
        assert "Configuration" in tags
        
        # PUT /v1/config/defaults
        assert "put" in paths["/v1/config/defaults"]
        tags = paths["/v1/config/defaults"]["put"].get("tags", [])
        assert "Configuration" in tags
    
    def test_health_endpoint_has_tag(self, openapi_schema):
        """Test that health endpoint has the Health tag."""
        paths = openapi_schema["paths"]
        
        # GET /health
        assert "/health" in paths
        assert "get" in paths["/health"]
        tags = paths["/health"]["get"].get("tags", [])
        assert "Health" in tags


class TestClarificationsEndpointDocumentation:
    """Test clarifications endpoint documentation and examples."""
    
    def test_post_clarifications_has_async_description(self, openapi_schema):
        """Test that POST /v1/clarifications emphasizes async workflow."""
        endpoint = openapi_schema["paths"]["/v1/clarifications"]["post"]
        description = endpoint.get("description", "")
        
        # Check for async keywords
        assert "async" in description.lower() or "asynchronous" in description.lower()
        assert "job_id" in description.lower() or "job id" in description.lower()
        assert "poll" in description.lower() or "polling" in description.lower()
    
    def test_post_clarifications_returns_202(self, openapi_schema):
        """Test that POST /v1/clarifications documents 202 status."""
        endpoint = openapi_schema["paths"]["/v1/clarifications"]["post"]
        responses = endpoint.get("responses", {})
        
        # Should have 202 response
        assert "202" in responses
        response_202 = responses["202"]
        assert "description" in response_202
    
    def test_post_clarifications_has_examples(self, openapi_schema):
        """Test that POST /v1/clarifications has response examples."""
        endpoint = openapi_schema["paths"]["/v1/clarifications"]["post"]
        responses = endpoint.get("responses", {})
        
        # Check 202 response has example
        if "202" in responses:
            response_202 = responses["202"]
            content = response_202.get("content", {})
            if "application/json" in content:
                json_content = content["application/json"]
                # Should have either example or examples
                assert "example" in json_content or "examples" in json_content
    
    def test_post_clarifications_example_has_valid_uuid(self, openapi_schema):
        """Test that POST /v1/clarifications example uses valid UUID."""
        endpoint = openapi_schema["paths"]["/v1/clarifications"]["post"]
        responses = endpoint.get("responses", {})
        
        if "202" in responses:
            response_202 = responses["202"]
            content = response_202.get("content", {}).get("application/json", {})
            example = content.get("example", {})
            
            if "id" in example:
                job_id = example["id"]
                # Should be a valid UUID string
                assert isinstance(job_id, str)
                assert len(job_id) == 36  # UUID string length
                # Verify it can be parsed as UUID
                try:
                    UUID(job_id)
                except ValueError:
                    pytest.fail(f"Invalid UUID in example: {job_id}")
    
    def test_get_clarifications_has_status_examples(self, openapi_schema):
        """Test that GET /v1/clarifications/{job_id} has multiple status examples."""
        endpoint = openapi_schema["paths"]["/v1/clarifications/{job_id}"]["get"]
        responses = endpoint.get("responses", {})
        
        # Should have 200 response
        assert "200" in responses
        response_200 = responses["200"]
        content = response_200.get("content", {}).get("application/json", {})
        
        # Should have multiple examples for different states
        if "examples" in content:
            examples = content["examples"]
            # Should show at least PENDING and SUCCESS states
            assert len(examples) >= 2
    
    def test_get_clarifications_emphasizes_null_result(self, openapi_schema):
        """Test that GET /v1/clarifications/{job_id} documents null result behavior."""
        endpoint = openapi_schema["paths"]["/v1/clarifications/{job_id}"]["get"]
        description = endpoint.get("description", "")
        
        # Should mention that result is null by default
        assert "null" in description.lower() or "production" in description.lower()


class TestConfigEndpointDocumentation:
    """Test config endpoint documentation and examples."""
    
    def test_get_config_defaults_has_example(self, openapi_schema):
        """Test that GET /v1/config/defaults has example response."""
        endpoint = openapi_schema["paths"]["/v1/config/defaults"]["get"]
        responses = endpoint.get("responses", {})
        
        # Should have 200 response with example
        if "200" in responses:
            response_200 = responses["200"]
            content = response_200.get("content", {}).get("application/json", {})
            assert "example" in content or "examples" in content
    
    def test_get_config_defaults_example_shows_required_fields(self, openapi_schema):
        """Test that GET /v1/config/defaults example shows realistic config."""
        endpoint = openapi_schema["paths"]["/v1/config/defaults"]["get"]
        responses = endpoint.get("responses", {})
        
        if "200" in responses:
            content = responses["200"].get("content", {}).get("application/json", {})
            example = content.get("example", {})
            
            if "default_config" in example:
                config = example["default_config"]
                # Should have key fields
                assert "provider" in config
                assert "model" in config
                assert "temperature" in config
            
            if "allowed_models" in example:
                models = example["allowed_models"]
                # Should have at least one provider
                assert len(models) > 0
    
    def test_put_config_defaults_has_examples(self, openapi_schema):
        """Test that PUT /v1/config/defaults has request/response examples."""
        endpoint = openapi_schema["paths"]["/v1/config/defaults"]["put"]
        responses = endpoint.get("responses", {})
        
        # Should have 200 response with examples
        if "200" in responses:
            response_200 = responses["200"]
            content = response_200.get("content", {}).get("application/json", {})
            # Should have either example or examples
            assert "example" in content or "examples" in content
    
    def test_put_config_defaults_shows_validation_error(self, openapi_schema):
        """Test that PUT /v1/config/defaults documents 400 validation error."""
        endpoint = openapi_schema["paths"]["/v1/config/defaults"]["put"]
        responses = endpoint.get("responses", {})
        
        # Should document 400 error
        assert "400" in responses
        response_400 = responses["400"]
        assert "description" in response_400


class TestRequestBodyExamples:
    """Test that request bodies have proper examples."""
    
    def test_clarification_request_has_example(self, openapi_schema):
        """Test that ClarificationRequest schema has example."""
        # Get the request body schema for POST /v1/clarifications
        endpoint = openapi_schema["paths"]["/v1/clarifications"]["post"]
        request_body = endpoint.get("requestBody", {})
        content = request_body.get("content", {}).get("application/json", {})
        
        # Schema should reference ClarificationRequestWithConfig
        schema_ref = content.get("schema", {})
        
        # Check if examples exist in the schema or inline
        if "examples" in content or "example" in content:
            # Has inline example
            assert True
        elif "$ref" in schema_ref:
            # Schema is referenced, examples should be in components
            ref_path = schema_ref["$ref"].split("/")[-1]
            components = openapi_schema.get("components", {}).get("schemas", {})
            if ref_path in components:
                schema = components[ref_path]
                # Should have examples in the schema definition
                assert "examples" in schema or "example" in schema
    
    def test_config_request_has_example(self, openapi_schema):
        """Test that ClarificationConfig schema has example."""
        components = openapi_schema.get("components", {}).get("schemas", {})
        
        # ClarificationConfig should exist
        if "ClarificationConfig" in components:
            config_schema = components["ClarificationConfig"]
            # Should have examples
            assert "examples" in config_schema or "example" in config_schema


class TestUUIDValidation:
    """Test that all UUID examples in the schema are valid."""
    
    def test_all_uuid_examples_are_valid(self, openapi_schema):
        """Test that all UUID examples in responses are valid 36-character UUIDs."""
        def check_for_uuids_iterative(obj):
            """Iteratively check for UUID fields and validate them."""
            stack = [(obj, "")]
            while stack:
                current_obj, path = stack.pop()
                if isinstance(current_obj, dict):
                    for key, value in current_obj.items():
                        current_path = f"{path}.{key}" if path else key
                        if key in ["id", "job_id"] and isinstance(value, str):
                            assert len(value) == 36, f"Invalid UUID length at {current_path}: {value}"
                            try:
                                UUID(value)
                            except ValueError:
                                pytest.fail(f"Invalid UUID format at {current_path}: {value}")
                        else:
                            stack.append((value, current_path))
                elif isinstance(current_obj, list):
                    for i, item in enumerate(current_obj):
                        stack.append((item, f"{path}[{i}]"))
        
        # Check all examples in the schema
        check_for_uuids_iterative(openapi_schema)


class TestBackwardCompatibility:
    """Test that changes don't break backward compatibility."""
    
    def test_endpoint_paths_unchanged(self, openapi_schema):
        """Test that existing endpoint paths are maintained."""
        paths = openapi_schema["paths"]
        
        # All expected endpoints should exist
        assert "/health" in paths
        assert "/v1/clarifications" in paths
        assert "/v1/clarifications/preview" in paths
        assert "/v1/clarifications/{job_id}" in paths
        assert "/v1/config/defaults" in paths
    
    def test_router_prefixes_maintained(self, openapi_schema):
        """Test that router prefixes are maintained."""
        paths = openapi_schema["paths"]
        
        # Clarifications should use /v1/clarifications prefix
        clarifications_paths = [p for p in paths if p.startswith("/v1/clarifications")]
        assert len(clarifications_paths) >= 3  # Main, preview, and job_id
        
        # Config should use /v1/config prefix
        config_paths = [p for p in paths if p.startswith("/v1/config")]
        assert len(config_paths) >= 1

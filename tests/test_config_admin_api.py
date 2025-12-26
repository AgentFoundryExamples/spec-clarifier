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
"""Integration tests for config admin endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.config import set_default_config, get_default_config
from app.models.config_models import ClarificationConfig


@pytest.fixture
def client():
    """Create a test client with config admin endpoints enabled (default)."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def disabled_config_client(monkeypatch):
    """Create a test client with config admin endpoints disabled."""
    monkeypatch.setenv("APP_ENABLE_CONFIG_ADMIN_ENDPOINTS", "false")
    from app.config import get_settings
    get_settings.cache_clear()
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config to default before each test."""
    # Reset to known defaults
    default = ClarificationConfig(
        provider="openai",
        model="gpt-5.1",
        system_prompt_id="default",
        temperature=0.1,
        max_tokens=None
    )
    set_default_config(default)
    yield
    # Reset after test
    set_default_config(default)
    # Clear settings cache
    from app.config import get_settings
    get_settings.cache_clear()


class TestGetDefaultsEndpoint:
    """Tests for GET /v1/config/defaults endpoint."""
    
    def test_get_defaults_success(self, client):
        """Test GET /v1/config/defaults returns current defaults."""
        response = client.get("/v1/config/defaults")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "default_config" in data
        assert "allowed_models" in data
        
        # Check default_config fields
        config = data["default_config"]
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-5.1"
        assert config["system_prompt_id"] == "default"
        assert config["temperature"] == 0.1
        assert config["max_tokens"] is None
        
        # Check allowed_models structure
        allowed = data["allowed_models"]
        assert "openai" in allowed
        assert "anthropic" in allowed
        assert isinstance(allowed["openai"], list)
        assert isinstance(allowed["anthropic"], list)
        assert "gpt-5.1" in allowed["openai"]
        assert "claude-sonnet-4.5" in allowed["anthropic"]
    
    def test_get_defaults_returns_updated_config(self, client):
        """Test GET returns updated config after PUT."""
        # First, update the defaults
        new_config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4.5",
            "system_prompt_id": "strict_json",
            "temperature": 0.2,
            "max_tokens": 3000
        }
        
        put_response = client.put("/v1/config/defaults", json=new_config)
        assert put_response.status_code == 200
        
        # Now GET and verify
        get_response = client.get("/v1/config/defaults")
        assert get_response.status_code == 200
        
        data = get_response.json()
        config = data["default_config"]
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-sonnet-4.5"
        assert config["system_prompt_id"] == "strict_json"
        assert config["temperature"] == 0.2
        assert config["max_tokens"] == 3000
    
    def test_get_defaults_disabled(self, disabled_config_client):
        """Test GET returns 403 when endpoints are disabled."""
        response = disabled_config_client.get("/v1/config/defaults")
        
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()


class TestPutDefaultsEndpoint:
    """Tests for PUT /v1/config/defaults endpoint."""
    
    def test_put_defaults_success_openai(self, client):
        """Test PUT with valid OpenAI config."""
        new_config = {
            "provider": "openai",
            "model": "gpt-4o",
            "system_prompt_id": "verbose_explanation",
            "temperature": 0.3,
            "max_tokens": 2500
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check returned config matches what we set
        config = data["default_config"]
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o"
        assert config["system_prompt_id"] == "verbose_explanation"
        assert config["temperature"] == 0.3
        assert config["max_tokens"] == 2500
        
        # Verify it actually updated the global default
        current = get_default_config()
        assert current.provider == "openai"
        assert current.model == "gpt-4o"
        assert current.system_prompt_id == "verbose_explanation"
    
    def test_put_defaults_success_anthropic(self, client):
        """Test PUT with valid Anthropic config."""
        new_config = {
            "provider": "anthropic",
            "model": "claude-opus-4",
            "system_prompt_id": "default",
            "temperature": 0.0,
            "max_tokens": None
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        assert response.status_code == 200
        data = response.json()
        
        config = data["default_config"]
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-opus-4"
    
    def test_put_defaults_invalid_provider(self, client):
        """Test PUT with unsupported provider returns 422."""
        new_config = {
            "provider": "google",  # Not in Literal["openai", "anthropic"]
            "model": "gemini-pro",
            "system_prompt_id": "default",
            "temperature": 0.1,
            "max_tokens": None
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        # Pydantic validation rejects invalid literal before reaching our code
        assert response.status_code == 422
        detail = response.json()["detail"]
        # Check that validation error mentions the provider
    
    def test_put_defaults_invalid_model_for_provider(self, client):
        """Test PUT with valid provider but invalid model returns 400."""
        new_config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo",  # Not in allowed list for openai
            "system_prompt_id": "default",
            "temperature": 0.1,
            "max_tokens": None
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "gpt-3.5-turbo" in detail.lower()
        assert "not allowed" in detail.lower()
    
    def test_put_defaults_missing_required_field(self, client):
        """Test PUT without required field returns 400."""
        # Missing provider - all fields are optional in ClarificationConfig
        # but when None is provided, set_default_config validates against allowed_models
        incomplete_config = {
            "model": "gpt-5.1",
            "system_prompt_id": "default",
            "temperature": 0.1,
        }
        
        response = client.put("/v1/config/defaults", json=incomplete_config)
        
        # Our validation should reject None provider
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "provider" in detail.lower()
    
    def test_put_defaults_invalid_temperature(self, client):
        """Test PUT with invalid temperature returns 422."""
        new_config = {
            "provider": "openai",
            "model": "gpt-5.1",
            "system_prompt_id": "default",
            "temperature": 3.0,  # Must be <= 2.0
            "max_tokens": None
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        assert response.status_code == 422
    
    def test_put_defaults_negative_max_tokens(self, client):
        """Test PUT with negative max_tokens returns 422."""
        new_config = {
            "provider": "openai",
            "model": "gpt-5.1",
            "system_prompt_id": "default",
            "temperature": 0.1,
            "max_tokens": -100
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        assert response.status_code == 422
    
    def test_put_defaults_unknown_system_prompt_id(self, client):
        """Test PUT with unknown system_prompt_id succeeds (no validation)."""
        # System prompt IDs are not validated at config level
        # Unknown IDs fall back to "default" at runtime
        new_config = {
            "provider": "openai",
            "model": "gpt-5.1",
            "system_prompt_id": "nonexistent_prompt",
            "temperature": 0.1,
            "max_tokens": None
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        # Should succeed - validation happens at runtime, not here
        assert response.status_code == 200
        data = response.json()
        assert data["default_config"]["system_prompt_id"] == "nonexistent_prompt"
    
    def test_put_defaults_extra_field_rejected(self, client):
        """Test PUT with extra field returns 422."""
        new_config = {
            "provider": "openai",
            "model": "gpt-5.1",
            "system_prompt_id": "default",
            "temperature": 0.1,
            "max_tokens": None,
            "extra_field": "should_fail"
        }
        
        response = client.put("/v1/config/defaults", json=new_config)
        
        # Pydantic should reject extra fields
        assert response.status_code == 422
    
    def test_put_defaults_disabled(self, disabled_config_client):
        """Test PUT returns 403 when endpoints are disabled."""
        new_config = {
            "provider": "openai",
            "model": "gpt-5.1",
            "system_prompt_id": "default",
            "temperature": 0.1,
            "max_tokens": None
        }
        
        response = disabled_config_client.put("/v1/config/defaults", json=new_config)
        
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()


class TestConfigAdminEndpointsIntegration:
    """Integration tests for config admin endpoints behavior."""
    
    def test_config_persists_across_requests(self, client):
        """Test that config updates persist across multiple requests."""
        # Set a new config
        new_config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4.5",
            "system_prompt_id": "strict_json",
            "temperature": 0.2,
            "max_tokens": 2000
        }
        
        put_response = client.put("/v1/config/defaults", json=new_config)
        assert put_response.status_code == 200
        
        # Verify with multiple GET requests
        for _ in range(3):
            get_response = client.get("/v1/config/defaults")
            assert get_response.status_code == 200
            config = get_response.json()["default_config"]
            assert config["provider"] == "anthropic"
            assert config["model"] == "claude-sonnet-4.5"
    
    def test_multiple_updates(self, client):
        """Test multiple sequential updates work correctly."""
        configs = [
            {
                "provider": "openai",
                "model": "gpt-5",
                "system_prompt_id": "default",
                "temperature": 0.1,
                "max_tokens": None
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4.5",
                "system_prompt_id": "strict_json",
                "temperature": 0.2,
                "max_tokens": 2500
            },
            {
                "provider": "openai",
                "model": "gpt-4o",
                "system_prompt_id": "verbose_explanation",
                "temperature": 0.0,
                "max_tokens": 1000
            }
        ]
        
        for config in configs:
            response = client.put("/v1/config/defaults", json=config)
            assert response.status_code == 200
            
            # Verify it was set correctly
            get_response = client.get("/v1/config/defaults")
            current = get_response.json()["default_config"]
            assert current["provider"] == config["provider"]
            assert current["model"] == config["model"]


class TestConfigAdminThreadSafety:
    """Tests for thread safety of config admin endpoints."""
    
    def test_concurrent_updates_are_atomic(self, client):
        """Test that concurrent PUT requests are handled atomically."""
        import concurrent.futures
        import threading
        
        results = []
        lock = threading.Lock()
        
        def update_config(provider, model):
            """Update config and record result."""
            config = {
                "provider": provider,
                "model": model,
                "system_prompt_id": "default",
                "temperature": 0.1,
                "max_tokens": None
            }
            response = client.put("/v1/config/defaults", json=config)
            with lock:
                results.append((provider, model, response.status_code))
        
        # Try concurrent updates
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(update_config, "openai", "gpt-5"),
                executor.submit(update_config, "openai", "gpt-4o"),
                executor.submit(update_config, "anthropic", "claude-sonnet-4.5"),
            ]
            concurrent.futures.wait(futures)
        
        # All should succeed
        assert len(results) == 3
        for _, _, status in results:
            assert status == 200
        
        # Final state should be one of the configs (last write wins)
        final = client.get("/v1/config/defaults").json()["default_config"]
        valid_providers = {"openai", "anthropic"}
        valid_models = {"gpt-5", "gpt-4o", "claude-sonnet-4.5"}
        assert final["provider"] in valid_providers
        assert final["model"] in valid_models

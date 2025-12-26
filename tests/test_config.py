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
"""Tests for configuration module."""

import os
import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_defaults():
    """Test that settings have correct default values."""
    settings = get_settings()
    
    assert settings.app_name == "Spec Clarifier"
    assert settings.app_version == "0.1.0"
    assert settings.app_description == "A service for clarifying specifications"
    assert settings.debug is False
    assert settings.show_job_result is False
    assert settings.enable_debug_endpoint is False
    assert settings.llm_default_provider == "openai"
    assert settings.llm_default_model == "gpt-5"
    assert settings.cors_origins == "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"
    assert settings.cors_allow_credentials is True
    assert settings.cors_allow_methods == "*"
    assert settings.cors_allow_headers == "*"


def test_settings_from_environment(monkeypatch):
    """Test that settings can be configured from environment variables."""
    monkeypatch.setenv("APP_APP_NAME", "Test App")
    monkeypatch.setenv("APP_DEBUG", "true")
    
    # Clear cache to ensure we get fresh settings with new env vars
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.app_name == "Test App"
    assert settings.debug is True


def test_get_settings_returns_settings():
    """Test that get_settings returns a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_get_settings_caches_instance():
    """Test that get_settings returns the same cached instance on multiple calls."""
    settings1 = get_settings()
    settings2 = get_settings()
    
    # Should be the exact same object due to lru_cache
    assert settings1 is settings2


def test_cors_origins_defaults():
    """Test that CORS origins default includes localhost origins."""
    settings = get_settings()
    origins = settings.get_cors_origins_list()
    
    assert len(origins) == 4
    assert "http://localhost:3000" in origins
    assert "http://localhost:8000" in origins
    assert "http://127.0.0.1:3000" in origins
    assert "http://127.0.0.1:8000" in origins


def test_cors_origins_from_environment(monkeypatch):
    """Test that CORS origins can be configured from environment variables."""
    monkeypatch.setenv("APP_CORS_ORIGINS", "https://example.com,https://app.example.com")
    
    get_settings.cache_clear()
    settings = get_settings()
    origins = settings.get_cors_origins_list()
    
    assert len(origins) == 2
    assert "https://example.com" in origins
    assert "https://app.example.com" in origins


def test_cors_origins_empty_string(monkeypatch):
    """Test that empty CORS origins string returns empty list."""
    monkeypatch.setenv("APP_CORS_ORIGINS", "")
    
    get_settings.cache_clear()
    settings = get_settings()
    origins = settings.get_cors_origins_list()
    
    assert origins == []


def test_cors_origins_with_spaces(monkeypatch):
    """Test that CORS origins are trimmed of whitespace."""
    monkeypatch.setenv("APP_CORS_ORIGINS", " https://example.com , https://app.example.com ")
    
    get_settings.cache_clear()
    settings = get_settings()
    origins = settings.get_cors_origins_list()
    
    assert len(origins) == 2
    assert "https://example.com" in origins
    assert "https://app.example.com" in origins


def test_cors_settings_from_environment(monkeypatch):
    """Test that all CORS settings can be configured from environment."""
    monkeypatch.setenv("APP_CORS_ORIGINS", "https://example.com")
    monkeypatch.setenv("APP_CORS_ALLOW_CREDENTIALS", "false")
    monkeypatch.setenv("APP_CORS_ALLOW_METHODS", "GET,POST")
    monkeypatch.setenv("APP_CORS_ALLOW_HEADERS", "Content-Type,Authorization")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.cors_origins == "https://example.com"
    assert settings.cors_allow_credentials is False
    assert settings.cors_allow_methods == "GET,POST"
    assert settings.cors_allow_headers == "Content-Type,Authorization"


def test_get_cors_methods_list_wildcard():
    """Test that wildcard methods return a list with asterisk."""
    settings = get_settings()
    methods = settings.get_cors_methods_list()
    
    assert methods == ["*"]


def test_get_cors_methods_list_explicit(monkeypatch):
    """Test that explicit methods are parsed correctly."""
    monkeypatch.setenv("APP_CORS_ALLOW_METHODS", "GET, POST, PUT")
    
    get_settings.cache_clear()
    settings = get_settings()
    methods = settings.get_cors_methods_list()
    
    assert len(methods) == 3
    assert "GET" in methods
    assert "POST" in methods
    assert "PUT" in methods


def test_get_cors_headers_list_wildcard():
    """Test that wildcard headers return a list with asterisk."""
    settings = get_settings()
    headers = settings.get_cors_headers_list()
    
    assert headers == ["*"]


def test_get_cors_headers_list_explicit(monkeypatch):
    """Test that explicit headers are parsed correctly."""
    monkeypatch.setenv("APP_CORS_ALLOW_HEADERS", "Content-Type, Authorization, X-Custom")
    
    get_settings.cache_clear()
    settings = get_settings()
    headers = settings.get_cors_headers_list()
    
    assert len(headers) == 3
    assert "Content-Type" in headers
    assert "Authorization" in headers
    assert "X-Custom" in headers


def test_show_job_result_default():
    """Test that show_job_result defaults to False."""
    settings = get_settings()
    assert settings.show_job_result is False


def test_show_job_result_from_environment(monkeypatch):
    """Test that show_job_result can be enabled via environment variable."""
    monkeypatch.setenv("APP_SHOW_JOB_RESULT", "true")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.show_job_result is True


def test_show_job_result_false_from_environment(monkeypatch):
    """Test that show_job_result can be explicitly disabled via environment variable."""
    monkeypatch.setenv("APP_SHOW_JOB_RESULT", "false")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.show_job_result is False


def test_enable_debug_endpoint_default():
    """Test that enable_debug_endpoint defaults to False."""
    settings = get_settings()
    
    assert settings.enable_debug_endpoint is False


def test_enable_debug_endpoint_from_environment(monkeypatch):
    """Test that enable_debug_endpoint can be set from environment."""
    monkeypatch.setenv("APP_ENABLE_DEBUG_ENDPOINT", "true")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.enable_debug_endpoint is True


def test_enable_debug_endpoint_false_from_environment(monkeypatch):
    """Test that enable_debug_endpoint can be explicitly set to False."""
    monkeypatch.setenv("APP_ENABLE_DEBUG_ENDPOINT", "false")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.enable_debug_endpoint is False


def test_llm_default_provider_default():
    """Test that llm_default_provider has correct default."""
    settings = get_settings()
    
    assert settings.llm_default_provider == "openai"


def test_llm_default_provider_from_environment(monkeypatch):
    """Test that llm_default_provider can be set from environment."""
    monkeypatch.setenv("APP_LLM_DEFAULT_PROVIDER", "anthropic")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.llm_default_provider == "anthropic"


def test_llm_default_model_default():
    """Test that llm_default_model has correct default."""
    settings = get_settings()
    
    assert settings.llm_default_model == "gpt-5"


def test_llm_default_model_from_environment(monkeypatch):
    """Test that llm_default_model can be set from environment."""
    monkeypatch.setenv("APP_LLM_DEFAULT_MODEL", "gpt-5.1")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.llm_default_model == "gpt-5.1"


def test_all_new_config_flags_together(monkeypatch):
    """Test that all new config flags work together."""
    monkeypatch.setenv("APP_ENABLE_DEBUG_ENDPOINT", "true")
    monkeypatch.setenv("APP_LLM_DEFAULT_PROVIDER", "google")
    monkeypatch.setenv("APP_LLM_DEFAULT_MODEL", "gemini-3.0-pro")
    monkeypatch.setenv("APP_SHOW_JOB_RESULT", "true")
    
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.enable_debug_endpoint is True
    assert settings.llm_default_provider == "google"
    assert settings.llm_default_model == "gemini-3.0-pro"
    assert settings.show_job_result is True


class TestGlobalDefaults:
    """Tests for GlobalDefaults class."""
    
    def test_global_defaults_initialization(self):
        """Test that GlobalDefaults initializes with built-in defaults."""
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        
        # Check allowed_models has default providers
        allowed = defaults.allowed_models
        assert "openai" in allowed
        assert "anthropic" in allowed
        
        # Check OpenAI has default models
        assert "gpt-5" in allowed["openai"]
        assert "gpt-5.1" in allowed["openai"]
        assert "gpt-4o" in allowed["openai"]
        
        # Check Anthropic has default models
        assert "claude-sonnet-4.5" in allowed["anthropic"]
        assert "claude-opus-4" in allowed["anthropic"]
    
    def test_global_defaults_default_config(self):
        """Test that GlobalDefaults has a default config."""
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        config = defaults.get_default_config()
        
        assert config.provider == "openai"
        assert config.model == "gpt-5.1"
        assert config.system_prompt_id == "default"
        assert config.temperature == 0.1
        assert config.max_tokens is None
    
    def test_global_defaults_allowed_models_returns_copy(self):
        """Test that allowed_models property returns a copy."""
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        allowed1 = defaults.allowed_models
        allowed2 = defaults.allowed_models
        
        # Should be equal but not the same object
        assert allowed1 == allowed2
        assert allowed1 is not allowed2
    
    def test_global_defaults_get_default_config_returns_copy(self):
        """Test that get_default_config returns a copy."""
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        config1 = defaults.get_default_config()
        config2 = defaults.get_default_config()
        
        # Should be equal but not the same object
        assert config1.model_dump() == config2.model_dump()
        assert config1 is not config2
    
    def test_global_defaults_set_default_config_valid(self):
        """Test setting a valid default config."""
        from app.config import GlobalDefaults
        from app.models.specs import ClarificationConfig
        
        defaults = GlobalDefaults()
        
        new_config = ClarificationConfig(
            provider="anthropic",
            model="claude-sonnet-4.5",
            system_prompt_id="advanced",
            temperature=0.5
        )
        
        defaults.set_default_config(new_config)
        
        retrieved = defaults.get_default_config()
        assert retrieved.provider == "anthropic"
        assert retrieved.model == "claude-sonnet-4.5"
        assert retrieved.system_prompt_id == "advanced"
        assert retrieved.temperature == 0.5
    
    def test_global_defaults_set_default_config_invalid_provider(self):
        """Test that setting config with provider not in allowed_models fails."""
        from app.config import GlobalDefaults, ConfigValidationError
        from app.models.specs import ClarificationConfig
        
        defaults = GlobalDefaults()
        
        # Manipulate allowed_models to remove a provider temporarily
        # Save original for restoration
        original_allowed = defaults._allowed_models.copy()
        
        try:
            # Remove anthropic from allowed_models
            del defaults._allowed_models["anthropic"]
            
            # Now try to set a config with anthropic provider
            config = ClarificationConfig(
                provider="anthropic",
                model="claude-sonnet-4.5",
                system_prompt_id="default"
            )
            
            with pytest.raises(ConfigValidationError) as exc_info:
                defaults.set_default_config(config)
            
            assert "not in allowed_models" in str(exc_info.value)
        finally:
            # Restore original allowed_models
            defaults._allowed_models = original_allowed
    
    def test_global_defaults_set_default_config_invalid_model(self):
        """Test that setting config with invalid model fails."""
        from app.config import GlobalDefaults, ConfigValidationError
        from app.models.specs import ClarificationConfig
        
        defaults = GlobalDefaults()
        
        config = ClarificationConfig(
            provider="openai",
            model="not-in-allowed-list",
            system_prompt_id="default"
        )
        
        with pytest.raises(ConfigValidationError) as exc_info:
            defaults.set_default_config(config)
        
        assert "not allowed for provider" in str(exc_info.value)
    
    def test_global_defaults_set_default_config_wrong_type(self):
        """Test that setting config with wrong type fails."""
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        
        with pytest.raises(TypeError) as exc_info:
            defaults.set_default_config("not a config")
        
        assert "must be a ClarificationConfig instance" in str(exc_info.value)


class TestConfigHelperFunctions:
    """Tests for config helper functions."""
    
    def test_get_default_config(self):
        """Test get_default_config helper function."""
        from app.config import get_default_config
        
        config = get_default_config()
        
        assert config.provider == "openai"
        assert config.model == "gpt-5.1"
        assert config.system_prompt_id == "default"
    
    def test_set_default_config(self):
        """Test set_default_config helper function."""
        from app.config import get_default_config, set_default_config
        from app.models.specs import ClarificationConfig
        
        new_config = ClarificationConfig(
            provider="anthropic",
            model="claude-opus-4",
            system_prompt_id="custom"
        )
        
        set_default_config(new_config)
        
        retrieved = get_default_config()
        assert retrieved.provider == "anthropic"
        assert retrieved.model == "claude-opus-4"
        assert retrieved.system_prompt_id == "custom"
        
        # Reset to default for other tests
        reset_config = ClarificationConfig(
            provider="openai",
            model="gpt-5.1",
            system_prompt_id="default"
        )
        set_default_config(reset_config)
    
    def test_get_allowed_models(self):
        """Test get_allowed_models helper function."""
        from app.config import get_allowed_models
        
        allowed = get_allowed_models()
        
        assert "openai" in allowed
        assert "anthropic" in allowed
        assert "gpt-5.1" in allowed["openai"]
        assert "claude-sonnet-4.5" in allowed["anthropic"]
    
    def test_validate_provider_model_valid(self):
        """Test validate_provider_model with valid combinations."""
        from app.config import validate_provider_model
        
        # Should not raise for valid combinations
        validate_provider_model("openai", "gpt-5.1")
        validate_provider_model("openai", "gpt-5")
        validate_provider_model("anthropic", "claude-sonnet-4.5")
        validate_provider_model("anthropic", "claude-opus-4")
    
    def test_validate_provider_model_invalid_provider(self):
        """Test validate_provider_model with invalid provider."""
        from app.config import validate_provider_model, ConfigValidationError
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_provider_model("invalid-provider", "some-model")
        
        assert "Unsupported provider" in str(exc_info.value)
    
    def test_validate_provider_model_invalid_model(self):
        """Test validate_provider_model with invalid model."""
        from app.config import validate_provider_model, ConfigValidationError
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_provider_model("openai", "invalid-model")
        
        assert "not allowed for provider" in str(exc_info.value)


class TestConfigValidationEdgeCases:
    """Tests for edge cases in config validation."""
    
    def test_validate_provider_model_empty_allowed_list(self):
        """Test validation when provider has empty allowed model list."""
        from app.config import GlobalDefaults, ConfigValidationError
        
        # Create a GlobalDefaults with manipulated allowed_models
        defaults = GlobalDefaults()
        
        # Directly modify internal state to simulate empty list
        defaults._allowed_models["openai"] = []
        
        from app.models.specs import ClarificationConfig
        
        config = ClarificationConfig(
            provider="openai",
            model="any-model",
            system_prompt_id="default"
        )
        
        with pytest.raises(ConfigValidationError) as exc_info:
            defaults.set_default_config(config)
        
        assert "No allowed models configured" in str(exc_info.value)
    
    def test_environment_variable_seeding_openai(self, monkeypatch):
        """Test that allowed models can be seeded from environment variables."""
        monkeypatch.setenv("APP_ALLOWED_MODELS_OPENAI", "gpt-6,gpt-7")
        
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        allowed = defaults.allowed_models
        
        assert "gpt-6" in allowed["openai"]
        assert "gpt-7" in allowed["openai"]
    
    def test_environment_variable_seeding_anthropic(self, monkeypatch):
        """Test that Anthropic models can be seeded from environment."""
        monkeypatch.setenv("APP_ALLOWED_MODELS_ANTHROPIC", "claude-5,claude-6")
        
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        allowed = defaults.allowed_models
        
        assert "claude-5" in allowed["anthropic"]
        assert "claude-6" in allowed["anthropic"]
    
    def test_environment_variable_empty_string_falls_back(self, monkeypatch):
        """Test that empty environment variable falls back to defaults."""
        monkeypatch.setenv("APP_ALLOWED_MODELS_OPENAI", "")
        
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        allowed = defaults.allowed_models
        
        # Should still have defaults
        assert "gpt-5.1" in allowed["openai"]
    
    def test_environment_variable_with_spaces(self, monkeypatch):
        """Test that environment variables handle spacing correctly."""
        monkeypatch.setenv("APP_ALLOWED_MODELS_OPENAI", " gpt-5 , gpt-5.1 , gpt-4o ")
        
        from app.config import GlobalDefaults
        
        defaults = GlobalDefaults()
        allowed = defaults.allowed_models
        
        # Should be trimmed
        assert "gpt-5" in allowed["openai"]
        assert "gpt-5.1" in allowed["openai"]
        assert "gpt-4o" in allowed["openai"]


class TestThreadSafety:
    """Tests for thread safety of GlobalDefaults."""
    
    def test_concurrent_reads(self):
        """Test that concurrent reads are safe."""
        import threading
        from app.config import get_default_config
        
        results = []
        
        def read_config():
            config = get_default_config()
            results.append(config.model_dump())
        
        threads = [threading.Thread(target=read_config) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All results should be equal
        assert len(results) == 10
        for result in results:
            assert result == results[0]
    
    def test_concurrent_writes_serialized(self):
        """Test that concurrent writes are serialized correctly."""
        import threading
        from app.config import set_default_config, get_default_config
        from app.models.specs import ClarificationConfig
        
        def set_config(model_name):
            config = ClarificationConfig(
                provider="openai",
                model=model_name,
                system_prompt_id="default"
            )
            set_default_config(config)
        
        # Set up initial state
        initial_config = ClarificationConfig(
            provider="openai",
            model="gpt-5.1",
            system_prompt_id="default"
        )
        set_default_config(initial_config)
        
        # Create threads that will write different models
        threads = [
            threading.Thread(target=set_config, args=("gpt-5",)),
            threading.Thread(target=set_config, args=("gpt-4o",)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Final config should be one of the valid options (writes serialized)
        final_config = get_default_config()
        assert final_config.model in ["gpt-5", "gpt-4o"]
        
        # Reset to default
        set_default_config(initial_config)

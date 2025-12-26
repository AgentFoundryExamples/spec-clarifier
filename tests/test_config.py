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

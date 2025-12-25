"""Tests for configuration module."""

import os
import pytest

from app.config import Settings, get_settings


def test_settings_defaults():
    """Test that settings have correct default values."""
    settings = Settings()
    
    assert settings.app_name == "Spec Clarifier"
    assert settings.app_version == "0.1.0"
    assert settings.app_description == "A service for clarifying specifications"
    assert settings.debug is False


def test_settings_from_environment(monkeypatch):
    """Test that settings can be configured from environment variables."""
    # Clear the cache first
    get_settings.cache_clear()
    
    monkeypatch.setenv("APP_APP_NAME", "Test App")
    monkeypatch.setenv("APP_DEBUG", "true")
    
    # Clear cache again to ensure we get fresh settings
    get_settings.cache_clear()
    settings = get_settings()
    
    assert settings.app_name == "Test App"
    assert settings.debug is True
    
    # Clean up
    get_settings.cache_clear()


def test_get_settings_returns_settings():
    """Test that get_settings returns a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_get_settings_caches_instance():
    """Test that get_settings returns the same cached instance on multiple calls."""
    get_settings.cache_clear()
    
    settings1 = get_settings()
    settings2 = get_settings()
    
    # Should be the exact same object due to lru_cache
    assert settings1 is settings2
    
    get_settings.cache_clear()

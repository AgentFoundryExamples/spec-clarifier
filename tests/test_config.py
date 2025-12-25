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

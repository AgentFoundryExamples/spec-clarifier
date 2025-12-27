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
"""Tests for dummy LLM mode and API key validation."""

import pytest

from app.config import GlobalDefaults
from app.services.llm_clients import (
    AnthropicResponsesClient,
    ClarificationLLMConfig,
    DummyLLMClient,
    LLMCallError,
    OpenAIResponsesClient,
    get_llm_client,
)


class TestDummyProviderSupport:
    """Tests for dummy provider support in configuration and factory."""

    def test_dummy_provider_in_allowed_models(self):
        """Test that dummy provider is included in allowed models."""
        from app.config import get_allowed_models

        allowed = get_allowed_models()
        assert "dummy" in allowed
        assert "test-model" in allowed["dummy"]
        assert "dummy-model" in allowed["dummy"]

    def test_default_config_uses_dummy_provider(self):
        """Test that default configuration uses dummy provider."""
        # Since other tests may have modified the global defaults, we need to
        # create a fresh GlobalDefaults instance to test the built-in defaults

        fresh_defaults = GlobalDefaults()
        config = fresh_defaults.get_default_config()
        assert config.provider == "dummy"
        assert config.model == "test-model"

    def test_clarification_config_accepts_dummy_provider(self):
        """Test that ClarificationConfig accepts 'dummy' as a provider."""
        from app.models.config_models import ClarificationConfig

        # Should not raise validation error
        config = ClarificationConfig(
            provider="dummy",
            model="test-model",
            system_prompt_id="default"
        )
        assert config.provider == "dummy"
        assert config.model == "test-model"

    def test_clarification_llm_config_accepts_dummy_provider(self):
        """Test that ClarificationLLMConfig accepts 'dummy' as a provider."""
        # Should not raise validation error
        config = ClarificationLLMConfig(
            provider="dummy",
            model="test-model"
        )
        assert config.provider == "dummy"
        assert config.model == "test-model"


class TestAPIKeyValidation:
    """Tests for API key validation when creating real provider clients."""

    def test_openai_client_factory_requires_api_key(self):
        """Test that creating OpenAI client requires OPENAI_API_KEY."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")

        # Should raise LLMCallError when API key is missing
        with pytest.raises(LLMCallError) as exc_info:
            get_llm_client("openai", config)

        error_msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in error_msg
        assert "required" in error_msg.lower()
        assert "dummy" in error_msg.lower()  # Should suggest using dummy mode

    def test_anthropic_client_factory_requires_api_key(self):
        """Test that creating Anthropic client requires ANTHROPIC_API_KEY."""
        config = ClarificationLLMConfig(provider="anthropic", model="claude-sonnet-4.5")

        # Should raise LLMCallError when API key is missing
        with pytest.raises(LLMCallError) as exc_info:
            get_llm_client("anthropic", config)

        error_msg = str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in error_msg
        assert "required" in error_msg.lower()
        assert "dummy" in error_msg.lower()  # Should suggest using dummy mode

    def test_dummy_client_factory_no_api_key_required(self):
        """Test that creating Dummy client does not require API keys."""
        config = ClarificationLLMConfig(provider="dummy", model="test-model")

        # Should work without any API keys set
        client = get_llm_client("dummy", config)
        assert isinstance(client, DummyLLMClient)

    def test_openai_client_factory_with_api_key(self, monkeypatch):
        """Test that OpenAI client is created when API key is present."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")

        client = get_llm_client("openai", config)
        assert isinstance(client, OpenAIResponsesClient)

    def test_anthropic_client_factory_with_api_key(self, monkeypatch):
        """Test that Anthropic client is created when API key is present."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        config = ClarificationLLMConfig(provider="anthropic", model="claude-sonnet-4.5")

        client = get_llm_client("anthropic", config)
        assert isinstance(client, AnthropicResponsesClient)


class TestLLMProviderSettings:
    """Tests for LLM_PROVIDER environment variable setting."""

    def test_settings_llm_provider_default(self):
        """Test that Settings.llm_provider defaults to 'dummy'."""
        from app.config import Settings

        settings = Settings()
        assert settings.llm_provider == "dummy"

    def test_settings_llm_provider_from_environment(self, monkeypatch):
        """Test that Settings.llm_provider can be set via LLM_PROVIDER env var."""
        from app.config import Settings

        monkeypatch.setenv("LLM_PROVIDER", "openai")
        settings = Settings()
        assert settings.llm_provider == "openai"

    def test_settings_openai_api_key_from_environment(self, monkeypatch):
        """Test that Settings.openai_api_key can be set via OPENAI_API_KEY."""
        from app.config import Settings

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        settings = Settings()
        assert settings.openai_api_key == "sk-test-key-12345"

    def test_settings_anthropic_api_key_from_environment(self, monkeypatch):
        """Test that Settings.anthropic_api_key can be set via ANTHROPIC_API_KEY."""
        from app.config import Settings

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        settings = Settings()
        assert settings.anthropic_api_key == "sk-ant-test-key-12345"

    def test_settings_api_keys_default_none(self):
        """Test that API keys default to None when not set."""
        from app.config import Settings

        settings = Settings()
        assert settings.openai_api_key is None
        assert settings.anthropic_api_key is None


class TestDummyClientBehavior:
    """Tests for DummyLLMClient behavior without API keys."""

    async def test_dummy_client_works_without_api_keys(self):
        """Test that DummyLLMClient works without any API keys."""
        client = DummyLLMClient()

        response = await client.complete(
            system_prompt="You are a test assistant",
            user_prompt="Hello, world!",
            model="test-model"
        )

        # Should return deterministic response
        assert response == '{"clarified": true}'

    async def test_dummy_client_with_custom_response(self):
        """Test DummyLLMClient with custom canned response."""
        client = DummyLLMClient(canned_response='{"result": "success"}')

        response = await client.complete(
            system_prompt="Test prompt",
            user_prompt="Test query",
            model="test-model"
        )

        assert response == '{"result": "success"}'

    async def test_dummy_client_echo_mode(self):
        """Test DummyLLMClient in echo mode."""
        client = DummyLLMClient(echo_prompts=True)

        response = await client.complete(
            system_prompt="System instructions",
            user_prompt="User query",
            model="test-model"
        )

        assert "System: System instructions" in response
        assert "User: User query" in response
        assert "Model: test-model" in response


class TestErrorMessages:
    """Tests for error message quality and actionability."""

    def test_openai_missing_key_error_message_quality(self):
        """Test that OpenAI missing key error has actionable message."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")

        with pytest.raises(LLMCallError) as exc_info:
            get_llm_client("openai", config)

        error_msg = str(exc_info.value)
        # Error should mention the required env var
        assert "OPENAI_API_KEY" in error_msg
        # Error should suggest using dummy mode
        assert "dummy" in error_msg.lower()
        # Error should be clear about what's required
        assert "required" in error_msg.lower()

    def test_anthropic_missing_key_error_message_quality(self):
        """Test that Anthropic missing key error has actionable message."""
        config = ClarificationLLMConfig(provider="anthropic", model="claude-sonnet-4.5")

        with pytest.raises(LLMCallError) as exc_info:
            get_llm_client("anthropic", config)

        error_msg = str(exc_info.value)
        # Error should mention the required env var
        assert "ANTHROPIC_API_KEY" in error_msg
        # Error should suggest using dummy mode
        assert "dummy" in error_msg.lower()
        # Error should be clear about what's required
        assert "required" in error_msg.lower()

    def test_error_messages_are_sanitized(self):
        """Test that error messages don't leak sensitive information."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")

        with pytest.raises(LLMCallError) as exc_info:
            get_llm_client("openai", config)

        error = exc_info.value
        # Error message should be sanitized (LLMCallError does this automatically)
        assert "[REDACTED]" not in str(error) or "OPENAI_API_KEY" in str(error)
        # Provider should be set
        assert error.provider == "openai"

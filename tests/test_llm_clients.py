"""Tests for LLM client abstractions and configuration."""

import pytest

from app.services.llm_clients import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    SUPPORTED_PROVIDERS,
    ClarificationLLMConfig,
    DummyLLMClient,
    LLMAuthenticationError,
    LLMCallError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMValidationError,
)


class TestProviderConstants:
    """Tests for provider constant definitions."""
    
    def test_provider_constants_defined(self):
        """Test that all provider constants are defined."""
        assert PROVIDER_OPENAI == "openai"
        assert PROVIDER_ANTHROPIC == "anthropic"
        assert PROVIDER_GOOGLE == "google"
    
    def test_supported_providers_frozen_set(self):
        """Test that SUPPORTED_PROVIDERS is a frozenset."""
        assert isinstance(SUPPORTED_PROVIDERS, frozenset)
        assert PROVIDER_OPENAI in SUPPORTED_PROVIDERS
        assert PROVIDER_ANTHROPIC in SUPPORTED_PROVIDERS
        assert PROVIDER_GOOGLE in SUPPORTED_PROVIDERS
    
    def test_supported_providers_count(self):
        """Test that we have exactly 3 supported providers."""
        assert len(SUPPORTED_PROVIDERS) == 3


class TestLLMCallError:
    """Tests for LLMCallError exception hierarchy."""
    
    def test_llm_call_error_basic(self):
        """Test basic LLMCallError creation."""
        error = LLMCallError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.original_error is None
        assert error.provider is None
    
    def test_llm_call_error_with_original_error(self):
        """Test LLMCallError wrapping original exception."""
        original = ValueError("Original error")
        error = LLMCallError("Wrapped error", original_error=original)
        assert error.original_error is original
        assert str(error) == "Wrapped error"
    
    def test_llm_call_error_with_provider(self):
        """Test LLMCallError with provider information."""
        error = LLMCallError("API error", provider="openai")
        assert error.provider == "openai"
    
    def test_llm_call_error_sanitizes_api_keys(self):
        """Test that API keys are sanitized from error messages."""
        message_with_key = "Error: api_key=sk-abc123xyz invalid"
        error = LLMCallError(message_with_key)
        assert "sk-abc123xyz" not in str(error)
        assert "[REDACTED]" in str(error)
    
    def test_llm_call_error_sanitizes_bearer_tokens(self):
        """Test that bearer tokens are sanitized."""
        message = "Authorization failed: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        error = LLMCallError(message)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in str(error)
        assert "[REDACTED]" in str(error)
    
    def test_llm_call_error_sanitizes_various_api_key_formats(self):
        """Test sanitization of various API key formats."""
        test_cases = [
            'api-key: "sk-test123"',
            'api_key="key-abc"',
            "apikey=token123",
            "API_KEY: secret456",
        ]
        for message in test_cases:
            error = LLMCallError(message)
            sanitized = str(error)
            # Ensure the actual key is not in the sanitized message
            assert "sk-test123" not in sanitized
            assert "key-abc" not in sanitized
            assert "token123" not in sanitized
            assert "secret456" not in sanitized
            assert "[REDACTED]" in sanitized
    
    def test_llm_call_error_sanitizes_authorization_headers(self):
        """Test sanitization of authorization headers."""
        message = 'Authorization: "Bearer sk-secret123"\nOther data'
        error = LLMCallError(message)
        assert "sk-secret123" not in str(error)
        assert "[REDACTED]" in str(error)
    
    def test_llm_call_error_sanitizes_x_api_key_headers(self):
        """Test sanitization of x-api-key headers."""
        message = 'x-api-key: "secret-key-789"'
        error = LLMCallError(message)
        assert "secret-key-789" not in str(error)
        assert "[REDACTED]" in str(error)
    
    def test_llm_call_error_preserves_safe_content(self):
        """Test that safe error content is preserved."""
        message = "Rate limit exceeded for model gpt-5.1"
        error = LLMCallError(message)
        assert "Rate limit exceeded" in str(error)
        assert "gpt-5.1" in str(error)


class TestLLMErrorHierarchy:
    """Tests for LLMCallError subclasses."""
    
    def test_llm_network_error_is_llm_call_error(self):
        """Test that LLMNetworkError is a subclass of LLMCallError."""
        error = LLMNetworkError("Connection failed")
        assert isinstance(error, LLMCallError)
        assert str(error) == "Connection failed"
    
    def test_llm_authentication_error_is_llm_call_error(self):
        """Test that LLMAuthenticationError is a subclass of LLMCallError."""
        error = LLMAuthenticationError("Invalid API key")
        assert isinstance(error, LLMCallError)
        assert str(error) == "Invalid API key"
    
    def test_llm_rate_limit_error_is_llm_call_error(self):
        """Test that LLMRateLimitError is a subclass of LLMCallError."""
        error = LLMRateLimitError("Rate limit exceeded")
        assert isinstance(error, LLMCallError)
        assert str(error) == "Rate limit exceeded"
    
    def test_llm_validation_error_is_llm_call_error(self):
        """Test that LLMValidationError is a subclass of LLMCallError."""
        error = LLMValidationError("Invalid request format")
        assert isinstance(error, LLMCallError)
        assert str(error) == "Invalid request format"
    
    def test_all_error_types_support_provider(self):
        """Test that all error types support provider parameter."""
        errors = [
            LLMNetworkError("Network error", provider="openai"),
            LLMAuthenticationError("Auth error", provider="anthropic"),
            LLMRateLimitError("Rate limit", provider="google"),
            LLMValidationError("Validation error", provider="openai"),
        ]
        for error in errors:
            assert error.provider is not None


class TestClarificationLLMConfig:
    """Tests for ClarificationLLMConfig model."""
    
    def test_config_with_all_fields(self):
        """Test creating config with all fields."""
        config = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=0.7,
            max_tokens=1000
        )
        assert config.provider == "openai"
        assert config.model == "gpt-5.1"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_config_with_minimal_fields(self):
        """Test creating config with only required fields."""
        config = ClarificationLLMConfig(
            provider="anthropic",
            model="claude-sonnet-4.5"
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4.5"
        assert config.temperature == 0.1  # Default
        assert config.max_tokens is None  # Optional
    
    def test_config_default_temperature(self):
        """Test that temperature defaults to 0.1."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        assert config.temperature == 0.1
    
    def test_config_validates_provider(self):
        """Test that invalid providers are rejected."""
        with pytest.raises(ValueError) as exc_info:
            ClarificationLLMConfig(provider="invalid", model="test-model")
        assert "Invalid provider 'invalid'" in str(exc_info.value)
        assert "openai" in str(exc_info.value)
        assert "anthropic" in str(exc_info.value)
        assert "google" in str(exc_info.value)
    
    def test_config_accepts_all_supported_providers(self):
        """Test that all supported providers are accepted."""
        for provider in SUPPORTED_PROVIDERS:
            config = ClarificationLLMConfig(provider=provider, model="test-model")
            assert config.provider == provider
    
    def test_config_validates_empty_model(self):
        """Test that empty model strings are rejected."""
        with pytest.raises(ValueError) as exc_info:
            ClarificationLLMConfig(provider="openai", model="")
        assert "Model must not be empty or blank" in str(exc_info.value)
    
    def test_config_validates_blank_model(self):
        """Test that blank model strings are rejected."""
        with pytest.raises(ValueError) as exc_info:
            ClarificationLLMConfig(provider="openai", model="   ")
        assert "Model must not be empty or blank" in str(exc_info.value)
    
    def test_config_strips_model_whitespace(self):
        """Test that model whitespace is stripped."""
        config = ClarificationLLMConfig(provider="openai", model="  gpt-5.1  ")
        assert config.model == "gpt-5.1"
    
    def test_config_validates_temperature_range_min(self):
        """Test that temperature below 0.0 is rejected."""
        with pytest.raises(ValueError):
            ClarificationLLMConfig(
                provider="openai",
                model="gpt-5.1",
                temperature=-0.1
            )
    
    def test_config_validates_temperature_range_max(self):
        """Test that temperature above 2.0 is rejected."""
        with pytest.raises(ValueError):
            ClarificationLLMConfig(
                provider="openai",
                model="gpt-5.1",
                temperature=2.1
            )
    
    def test_config_accepts_temperature_boundary_values(self):
        """Test that temperature boundary values (0.0, 2.0) are accepted."""
        config_min = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=0.0
        )
        assert config_min.temperature == 0.0
        
        config_max = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=2.0
        )
        assert config_max.temperature == 2.0
    
    def test_config_validates_max_tokens_positive(self):
        """Test that max_tokens must be positive."""
        with pytest.raises(ValueError):
            ClarificationLLMConfig(
                provider="openai",
                model="gpt-5.1",
                max_tokens=0
            )
        
        with pytest.raises(ValueError):
            ClarificationLLMConfig(
                provider="openai",
                model="gpt-5.1",
                max_tokens=-100
            )
    
    def test_config_max_tokens_none_is_valid(self):
        """Test that max_tokens can be None."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        assert config.max_tokens is None
    
    def test_config_serialization(self):
        """Test that config can be serialized to dict/JSON."""
        config = ClarificationLLMConfig(
            provider="google",
            model="gemini-3.0-pro",
            temperature=0.5,
            max_tokens=2000
        )
        data = config.model_dump()
        assert data["provider"] == "google"
        assert data["model"] == "gemini-3.0-pro"
        assert data["temperature"] == 0.5
        assert data["max_tokens"] == 2000
    
    def test_config_deserialization(self):
        """Test that config can be deserialized from dict."""
        data = {
            "provider": "anthropic",
            "model": "claude-opus-4",
            "temperature": 0.2,
            "max_tokens": 1500
        }
        config = ClarificationLLMConfig(**data)
        assert config.provider == "anthropic"
        assert config.model == "claude-opus-4"
        assert config.temperature == 0.2
        assert config.max_tokens == 1500


class TestDummyLLMClient:
    """Tests for DummyLLMClient test implementation."""
    
    async def test_dummy_client_default_response(self):
        """Test that DummyLLMClient returns default canned response."""
        client = DummyLLMClient()
        response = await client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello",
            model="test-model"
        )
        assert response == '{"clarified": true}'
    
    async def test_dummy_client_custom_canned_response(self):
        """Test DummyLLMClient with custom canned response."""
        custom_response = '{"result": "custom clarification"}'
        client = DummyLLMClient(canned_response=custom_response)
        response = await client.complete(
            system_prompt="System instruction",
            user_prompt="User query",
            model="test-model"
        )
        assert response == custom_response
    
    async def test_dummy_client_echo_prompts(self):
        """Test DummyLLMClient echo_prompts mode."""
        client = DummyLLMClient(echo_prompts=True)
        response = await client.complete(
            system_prompt="System instruction",
            user_prompt="User query",
            model="test-model"
        )
        assert "System: System instruction" in response
        assert "User: User query" in response
        assert "Model: test-model" in response
    
    async def test_dummy_client_canned_response_takes_precedence(self):
        """Test that canned_response overrides echo_prompts."""
        client = DummyLLMClient(
            canned_response="Custom response",
            echo_prompts=True
        )
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="model"
        )
        assert response == "Custom response"
        assert "System:" not in response
    
    async def test_dummy_client_validates_empty_system_prompt(self):
        """Test that DummyLLMClient validates empty system_prompt."""
        client = DummyLLMClient()
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="",
                user_prompt="User query",
                model="test-model"
            )
        assert "system_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_dummy_client_validates_blank_system_prompt(self):
        """Test that DummyLLMClient validates blank system_prompt."""
        client = DummyLLMClient()
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="   ",
                user_prompt="User query",
                model="test-model"
            )
        assert "system_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_dummy_client_validates_empty_user_prompt(self):
        """Test that DummyLLMClient validates empty user_prompt."""
        client = DummyLLMClient()
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="",
                model="test-model"
            )
        assert "user_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_dummy_client_validates_blank_user_prompt(self):
        """Test that DummyLLMClient validates blank user_prompt."""
        client = DummyLLMClient()
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="   ",
                model="test-model"
            )
        assert "user_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_dummy_client_validates_empty_model(self):
        """Test that DummyLLMClient validates empty model."""
        client = DummyLLMClient()
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="User query",
                model=""
            )
        assert "model must not be empty or blank" in str(exc_info.value)
    
    async def test_dummy_client_validates_blank_model(self):
        """Test that DummyLLMClient validates blank model."""
        client = DummyLLMClient()
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="User query",
                model="   "
            )
        assert "model must not be empty or blank" in str(exc_info.value)
    
    async def test_dummy_client_simulate_failure(self):
        """Test DummyLLMClient failure simulation."""
        client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Simulated API error"
        )
        with pytest.raises(LLMCallError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="model"
            )
        assert "Simulated API error" in str(exc_info.value)
        assert exc_info.value.provider == "dummy"
    
    async def test_dummy_client_simulate_network_error(self):
        """Test DummyLLMClient simulating network errors."""
        client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Connection timeout",
            failure_type=LLMNetworkError
        )
        with pytest.raises(LLMNetworkError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="model"
            )
        assert "Connection timeout" in str(exc_info.value)
    
    async def test_dummy_client_simulate_auth_error(self):
        """Test DummyLLMClient simulating authentication errors."""
        client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Invalid API key",
            failure_type=LLMAuthenticationError
        )
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="model"
            )
        assert "Invalid API key" in str(exc_info.value)
    
    async def test_dummy_client_simulate_rate_limit_error(self):
        """Test DummyLLMClient simulating rate limit errors."""
        client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Rate limit exceeded",
            failure_type=LLMRateLimitError
        )
        with pytest.raises(LLMRateLimitError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="model"
            )
        assert "Rate limit exceeded" in str(exc_info.value)
    
    async def test_dummy_client_simulate_validation_error(self):
        """Test DummyLLMClient simulating validation errors."""
        client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Invalid request format",
            failure_type=LLMValidationError
        )
        with pytest.raises(LLMValidationError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="model"
            )
        assert "Invalid request format" in str(exc_info.value)
    
    async def test_dummy_client_accepts_kwargs(self):
        """Test that DummyLLMClient accepts additional kwargs."""
        client = DummyLLMClient()
        # Should not raise any errors
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="model",
            temperature=0.7,
            max_tokens=1000,
            custom_param="value"
        )
        assert response is not None
    
    async def test_dummy_client_deterministic_output(self):
        """Test that DummyLLMClient returns deterministic output."""
        client = DummyLLMClient(canned_response="Deterministic")
        
        # Call multiple times with same inputs
        response1 = await client.complete("S", "U", "M")
        response2 = await client.complete("S", "U", "M")
        response3 = await client.complete("S", "U", "M")
        
        assert response1 == response2 == response3 == "Deterministic"
    
    async def test_dummy_client_different_instances_independent(self):
        """Test that different DummyLLMClient instances are independent."""
        client1 = DummyLLMClient(canned_response="Response 1")
        client2 = DummyLLMClient(canned_response="Response 2")
        
        response1 = await client1.complete("S", "U", "M")
        response2 = await client2.complete("S", "U", "M")
        
        assert response1 == "Response 1"
        assert response2 == "Response 2"


class TestLLMClientProtocol:
    """Tests for LLMClient protocol compliance."""
    
    async def test_dummy_client_implements_protocol(self):
        """Test that DummyLLMClient implements the LLMClient protocol."""
        client = DummyLLMClient()
        
        # Test that complete method exists and has the right signature
        assert hasattr(client, "complete")
        assert callable(client.complete)
        
        # Test that it's async
        result = client.complete(
            system_prompt="System",
            user_prompt="User",
            model="test-model"
        )
        # Should return a coroutine
        assert hasattr(result, '__await__')
        
        # Consume the coroutine
        response = await result
        assert isinstance(response, str)
    
    async def test_protocol_signature_returns_string(self):
        """Test that protocol requires string return type."""
        client = DummyLLMClient()
        response = await client.complete(
            system_prompt="Test",
            user_prompt="Test",
            model="model"
        )
        assert isinstance(response, str)
    
    async def test_protocol_signature_accepts_kwargs(self):
        """Test that protocol accepts arbitrary kwargs."""
        client = DummyLLMClient()
        # Should not raise TypeError for unexpected kwargs
        await client.complete(
            system_prompt="Test",
            user_prompt="Test",
            model="model",
            arbitrary_param=True,
            another_param="value"
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_error_sanitization_with_multiple_secrets(self):
        """Test sanitization when multiple secrets are in the same message."""
        message = "Error: api_key=sk-123 and token=abc456 and bearer xyz789"
        error = LLMCallError(message)
        sanitized = str(error)
        assert "sk-123" not in sanitized
        assert "abc456" not in sanitized
        assert "xyz789" not in sanitized
        assert sanitized.count("[REDACTED]") >= 2
    
    async def test_dummy_client_with_very_long_prompts(self):
        """Test DummyLLMClient handles very long prompts."""
        client = DummyLLMClient(echo_prompts=True)
        long_prompt = "X" * 100000  # 100k characters
        
        response = await client.complete(
            system_prompt=long_prompt,
            user_prompt=long_prompt,
            model="model"
        )
        assert long_prompt in response
    
    async def test_dummy_client_with_unicode_prompts(self):
        """Test DummyLLMClient handles unicode in prompts."""
        client = DummyLLMClient(echo_prompts=True)
        response = await client.complete(
            system_prompt="系统提示 🚀",
            user_prompt="用户查询 ✨",
            model="模型-1"
        )
        assert "系统提示 🚀" in response
        assert "用户查询 ✨" in response
        assert "模型-1" in response
    
    def test_config_with_extreme_temperature_values(self):
        """Test config with boundary temperature values."""
        # Test minimum
        config_min = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=0.0
        )
        assert config_min.temperature == 0.0
        
        # Test maximum
        config_max = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=2.0
        )
        assert config_max.temperature == 2.0
    
    def test_config_with_very_large_max_tokens(self):
        """Test config accepts very large max_tokens values."""
        config = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            max_tokens=1000000
        )
        assert config.max_tokens == 1000000

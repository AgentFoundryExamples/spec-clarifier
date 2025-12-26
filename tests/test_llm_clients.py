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
"""Tests for LLM client abstractions and configuration."""

import pytest

from app.services.llm_clients import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    SUPPORTED_PROVIDERS,
    AnthropicResponsesClient,
    ClarificationLLMConfig,
    DummyLLMClient,
    LLMAuthenticationError,
    LLMCallError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMValidationError,
    OpenAIResponsesClient,
    get_llm_client,
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
    
    def test_llm_call_error_sanitizes_json_format_secrets(self):
        """Test sanitization of secrets in JSON format."""
        message = '{"api_key":"sk-test123","status":"error"}'
        error = LLMCallError(message)
        assert "sk-test123" not in str(error)
        assert "[REDACTED]" in str(error)
        assert "status" in str(error)
    
    def test_llm_call_error_sanitizes_url_encoded_secrets(self):
        """Test sanitization of URL-encoded API keys."""
        message = "Request failed: api_key%3Dsk-secret123&model=gpt-5"
        error = LLMCallError(message)
        assert "sk-secret123" not in str(error)
        assert "[REDACTED]" in str(error)
    
    def test_llm_call_error_sanitizes_generic_json_secrets(self):
        """Test sanitization of generic secret patterns in JSON."""
        test_cases = [
            ('{"secret":"my-secret-value"}', "my-secret-value"),
            ('{"password":"pass123"}', "pass123"),
            ('{"apikey":"key-abc"}', "key-abc"),
        ]
        for message, secret_value in test_cases:
            error = LLMCallError(message)
            sanitized = str(error)
            assert secret_value not in sanitized
            assert "[REDACTED]" in sanitized
    
    def test_llm_call_error_message_sanitized_in_exception(self):
        """Test that exception message itself is sanitized."""
        message_with_secret = "Error with api_key=sk-secret123"
        error = LLMCallError(message_with_secret)
        # The exception message passed to super().__init__() should be sanitized
        assert str(error) == error.message
        assert "sk-secret123" not in str(error)


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
    
    def test_dummy_client_validates_failure_type(self):
        """Test that DummyLLMClient validates failure_type is subclass of LLMCallError."""
        # Should raise TypeError for non-LLMCallError types
        with pytest.raises(TypeError) as exc_info:
            DummyLLMClient(
                simulate_failure=True,
                failure_type=ValueError  # Not an LLMCallError subclass
            )
        assert "failure_type must be a subclass of LLMCallError" in str(exc_info.value)
    
    def test_dummy_client_accepts_valid_failure_types(self):
        """Test that DummyLLMClient accepts valid LLMCallError subclasses."""
        # Should not raise for LLMCallError and its subclasses
        valid_types = [
            LLMCallError,
            LLMNetworkError,
            LLMAuthenticationError,
            LLMRateLimitError,
            LLMValidationError,
        ]
        for failure_type in valid_types:
            client = DummyLLMClient(
                simulate_failure=True,
                failure_type=failure_type
            )
            assert client.failure_type == failure_type
    
    def test_dummy_client_no_validation_when_not_simulating_failure(self):
        """Test that failure_type is not validated when simulate_failure=False."""
        # Should not raise even with invalid type when not simulating failure
        client = DummyLLMClient(
            simulate_failure=False,
            failure_type=ValueError  # Invalid but ignored
        )
        assert client.simulate_failure is False


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


class TestOpenAIResponsesClient:
    """Tests for OpenAIResponsesClient implementation."""
    
    async def test_openai_client_successful_completion(self, monkeypatch):
        """Test successful completion with OpenAI Responses API."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        # Mock response object that matches OpenAI Responses API structure
        class MockResponse:
            @property
            def output_text(self):
                return "This is the AI response text."
        
        # Mock AsyncOpenAI client
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    return MockResponse()
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        # Monkeypatch the AsyncOpenAI import
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        
        client = OpenAIResponsesClient()
        # Inject mock client
        client._client = MockAsyncOpenAI(api_key="sk-test-key-12345")
        
        response = await client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello, world!",
            model="gpt-5.1"
        )
        
        assert response == "This is the AI response text."
    
    async def test_openai_client_kwargs_propagation(self, monkeypatch):
        """Test that kwargs are properly passed to OpenAI API."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        captured_params = {}
        
        # Mock response object
        class MockResponse:
            @property
            def output_text(self):
                return "Response with custom params"
        
        # Mock AsyncOpenAI client that captures parameters
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    captured_params.update(kwargs)
                    return MockResponse()
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key-12345")
        
        await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="gpt-5.1",
            temperature=0.7,
            max_tokens=500
        )
        
        # Verify parameters were passed correctly
        assert captured_params["model"] == "gpt-5.1"
        assert captured_params["instructions"] == "System"
        assert captured_params["input"] == "User"
        assert captured_params["temperature"] == 0.7
        assert captured_params["max_output_tokens"] == 500  # max_tokens renamed
    
    async def test_openai_client_content_array_fallback(self, monkeypatch):
        """Test extraction from output list when output_text is empty."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        # Mock content part
        class MockContentPart:
            def __init__(self, text):
                self.type = "output_text"
                self.text = text
        
        # Mock output item (message type)
        class MockOutputItem:
            def __init__(self, content_parts):
                self.type = "message"
                self.content = content_parts
        
        # Mock response with output list
        class MockResponse:
            def __init__(self):
                self.output = [
                    MockOutputItem([
                        MockContentPart("First part. "),
                        MockContentPart("Second part.")
                    ])
                ]
            
            @property
            def output_text(self):
                # Simulate empty output_text to force fallback
                return ""
        
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    return MockResponse()
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key-12345")
        
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="gpt-5.1"
        )
        
        assert response == "First part. Second part."
    
    async def test_openai_client_multiple_content_parts_ignore_non_text(self, monkeypatch):
        """Test that non-text content parts are ignored."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        # Mock content parts
        class MockTextPart:
            def __init__(self, text):
                self.type = "output_text"
                self.text = text
        
        class MockToolPart:
            def __init__(self):
                self.type = "tool_call"
                self.tool_name = "calculator"
        
        # Mock output item
        class MockOutputItem:
            def __init__(self, content_parts):
                self.type = "message"
                self.content = content_parts
        
        # Mock response with mixed content types
        class MockResponse:
            def __init__(self):
                self.output = [
                    MockOutputItem([
                        MockTextPart("Text before tool. "),
                        MockToolPart(),
                        MockTextPart("Text after tool.")
                    ])
                ]
            
            @property
            def output_text(self):
                # Simulate empty output_text to force fallback
                return ""
        
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    return MockResponse()
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key-12345")
        
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="gpt-5.1"
        )
        
        assert response == "Text before tool. Text after tool."
    
    async def test_openai_client_missing_api_key_error(self, monkeypatch):
        """Test that missing API key raises LLMCallError."""
        from app.services.llm_clients import OpenAIResponsesClient, LLMCallError
        
        # Remove API key from environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        client = OpenAIResponsesClient()
        
        with pytest.raises(LLMCallError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="gpt-5.1"
            )
        
        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
    
    async def test_openai_client_authentication_error(self, monkeypatch):
        """Test that OpenAI AuthenticationError is mapped to LLMAuthenticationError."""
        from app.services.llm_clients import OpenAIResponsesClient, LLMAuthenticationError
        import httpx
        
        # Create proper httpx request and response
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(401, json={"error": {"message": "Invalid API key"}}, request=request)
        
        # Mock AsyncOpenAI to raise AuthenticationError
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    from openai import AuthenticationError
                    raise AuthenticationError("Invalid API key", response=response, body=None)
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-key")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-invalid-key")
        
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="gpt-5.1"
            )
        
        assert "authentication failed" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert exc_info.value.original_error is not None
    
    async def test_openai_client_rate_limit_error(self, monkeypatch):
        """Test that OpenAI RateLimitError is mapped to LLMRateLimitError."""
        from app.services.llm_clients import OpenAIResponsesClient, LLMRateLimitError
        import httpx
        
        # Create proper httpx request and response
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}}, request=request)
        
        # Mock AsyncOpenAI to raise RateLimitError
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    from openai import RateLimitError
                    raise RateLimitError("Rate limit exceeded", response=response, body=None)
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key")
        
        with pytest.raises(LLMRateLimitError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="gpt-5.1"
            )
        
        assert "rate limit" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
    
    async def test_openai_client_validation_error(self, monkeypatch):
        """Test that OpenAI BadRequestError is mapped to LLMValidationError."""
        from app.services.llm_clients import OpenAIResponsesClient, LLMValidationError
        import httpx
        
        # Create proper httpx request and response
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(400, json={"error": {"message": "Invalid request format"}}, request=request)
        
        # Mock AsyncOpenAI to raise BadRequestError
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    from openai import BadRequestError
                    raise BadRequestError("Invalid request format", response=response, body=None)
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key")
        
        with pytest.raises(LLMValidationError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="gpt-5.1"
            )
        
        assert "validation failed" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
    
    async def test_openai_client_network_error(self, monkeypatch):
        """Test that OpenAI connection errors are mapped to LLMNetworkError."""
        from app.services.llm_clients import OpenAIResponsesClient, LLMNetworkError
        
        # Mock request object for OpenAI exception
        class MockRequest:
            url = "https://api.openai.com/v1/responses"
            method = "POST"
        
        # Mock AsyncOpenAI to raise APIConnectionError
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    from openai import APIConnectionError
                    raise APIConnectionError(request=MockRequest())
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key")
        
        with pytest.raises(LLMNetworkError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="gpt-5.1"
            )
        
        assert "network error" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
    
    async def test_openai_client_timeout_error(self, monkeypatch):
        """Test that OpenAI timeout errors are mapped to LLMNetworkError."""
        from app.services.llm_clients import OpenAIResponsesClient, LLMNetworkError
        
        # Mock request object for OpenAI exception
        class MockRequest:
            url = "https://api.openai.com/v1/responses"
            method = "POST"
        
        # Mock AsyncOpenAI to raise APITimeoutError
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    from openai import APITimeoutError
                    raise APITimeoutError(request=MockRequest())
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key")
        
        with pytest.raises(LLMNetworkError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="gpt-5.1"
            )
        
        assert "network error" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
    
    async def test_openai_client_generic_api_error(self, monkeypatch):
        """Test that generic OpenAI APIError is mapped to LLMCallError."""
        from app.services.llm_clients import OpenAIResponsesClient, LLMCallError
        
        # Mock response object for OpenAI exception
        class MockResponse:
            status_code = 500
            headers = {}
            
            def json(self):
                return {"error": {"message": "Internal server error"}}
        
        # Mock AsyncOpenAI to raise APIError
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    from openai import APIError
                    raise APIError("Internal server error", request=None, body=None)
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        client._client = MockAsyncOpenAI(api_key="sk-test-key")
        
        with pytest.raises(LLMCallError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="gpt-5.1"
            )
        
        # Should not be a more specific error type
        assert type(exc_info.value).__name__ == "LLMCallError"
        assert exc_info.value.provider == "openai"
    
    async def test_openai_client_validates_empty_system_prompt(self, monkeypatch):
        """Test that OpenAIResponsesClient validates empty system_prompt."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="",
                user_prompt="User query",
                model="gpt-5.1"
            )
        
        assert "system_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_openai_client_validates_empty_user_prompt(self, monkeypatch):
        """Test that OpenAIResponsesClient validates empty user_prompt."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="",
                model="gpt-5.1"
            )
        
        assert "user_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_openai_client_validates_empty_model(self, monkeypatch):
        """Test that OpenAIResponsesClient validates empty model."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="User query",
                model=""
            )
        
        assert "model must not be empty or blank" in str(exc_info.value)
    
    async def test_openai_client_lazy_initialization(self, monkeypatch):
        """Test that client is initialized lazily on first use."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        assert client._client is None  # Not initialized yet
        
        # Mock the client after instantiation
        class MockResponse:
            @property
            def output_text(self):
                return "Response"
        
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    return MockResponse()
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        client._client = MockAsyncOpenAI(api_key="sk-test-key")
        
        await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="gpt-5.1"
        )
        
        assert client._client is not None  # Now initialized
    
    async def test_openai_client_custom_api_key(self, monkeypatch):
        """Test that client can be initialized with custom API key."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        # Don't set environment variable
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        # Mock response
        class MockResponse:
            @property
            def output_text(self):
                return "Response with custom key"
        
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    return MockResponse()
            
            def __init__(self, api_key):
                self.api_key = api_key
                self.responses = self.MockResponses()
        
        client = OpenAIResponsesClient(api_key="sk-custom-key-789")
        client._client = MockAsyncOpenAI(api_key="sk-custom-key-789")
        
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="gpt-5.1"
        )
        
        assert response == "Response with custom key"
    
    async def test_openai_client_implements_protocol(self, monkeypatch):
        """Test that OpenAIResponsesClient implements the LLMClient protocol."""
        from app.services.llm_clients import OpenAIResponsesClient
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        
        client = OpenAIResponsesClient()
        
        # Test that complete method exists and has the right signature
        assert hasattr(client, "complete")
        assert callable(client.complete)
        
        # Mock the client
        class MockResponse:
            @property
            def output_text(self):
                return "Protocol test response"
        
        class MockAsyncOpenAI:
            class MockResponses:
                async def create(self, **kwargs):
                    return MockResponse()
            
            def __init__(self, api_key):
                self.responses = self.MockResponses()
        
        client._client = MockAsyncOpenAI(api_key="sk-test-key")
        
        # Test that it's async and returns a string
        result = client.complete(
            system_prompt="System",
            user_prompt="User",
            model="gpt-5.1"
        )
        assert hasattr(result, '__await__')
        
        response = await result
        assert isinstance(response, str)
    
    def test_openai_client_has_import_error_handling(self):
        """Verify that the complete method has ImportError handling for missing OpenAI SDK."""
        from app.services.llm_clients import OpenAIResponsesClient
        import inspect
        
        # Get the source code of the complete method
        source = inspect.getsource(OpenAIResponsesClient.complete)
        
        # Verify the ImportError handling is present
        assert "except ImportError" in source, "Missing ImportError exception handler"
        assert "OpenAI SDK not installed or accessible" in source, "Missing SDK not installed error message"
        assert "from openai import" in source, "Missing openai module import"
        
        # The code structure validates that if openai cannot be imported,
        # a clear LLMCallError will be raised with appropriate messaging


class TestAnthropicResponsesClient:
    """Tests for AnthropicResponsesClient implementation."""
    
    async def test_anthropic_client_successful_completion(self, monkeypatch):
        """Test successful completion with Anthropic Messages API."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        # Mock content block that matches Anthropic Messages API structure
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        # Mock response object
        class MockMessage:
            def __init__(self):
                self.content = [MockTextBlock("This is the AI response text.")]
        
        # Mock AsyncAnthropic client
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        # Monkeypatch the API key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        
        client = AnthropicResponsesClient()
        # Inject mock client
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key-12345")
        
        response = await client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Hello, world!",
            model="claude-sonnet-4.5"
        )
        
        assert response == "This is the AI response text."
    
    async def test_anthropic_client_default_model(self):
        """Test that DEFAULT_MODEL constant is correctly set."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        assert AnthropicResponsesClient.DEFAULT_MODEL == "claude-sonnet-4.5"
    
    async def test_anthropic_client_kwargs_propagation(self, monkeypatch):
        """Test that kwargs are properly passed to Anthropic API."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        captured_params = {}
        
        # Mock content block
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        # Mock response object
        class MockMessage:
            def __init__(self):
                self.content = [MockTextBlock("Response with custom params")]
        
        # Mock AsyncAnthropic client that captures parameters
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    captured_params.update(kwargs)
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key-12345")
        
        await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-opus-4",
            temperature=0.7,
            max_tokens=500
        )
        
        # Verify parameters were passed correctly
        assert captured_params["model"] == "claude-opus-4"
        assert captured_params["system"] == "System"
        assert captured_params["messages"] == [{"role": "user", "content": "User"}]
        assert captured_params["temperature"] == 0.7
        assert captured_params["max_tokens"] == 500
    
    async def test_anthropic_client_default_max_tokens(self, monkeypatch):
        """Test that max_tokens defaults to 4096 when not specified."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        captured_params = {}
        
        # Mock response
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        class MockMessage:
            def __init__(self):
                self.content = [MockTextBlock("Response")]
        
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    captured_params.update(kwargs)
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key-12345")
        
        await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-sonnet-4.5"
        )
        
        # Verify default max_tokens was set
        assert captured_params["max_tokens"] == 4096
    
    async def test_anthropic_client_multiple_content_blocks(self, monkeypatch):
        """Test extraction from multiple text content blocks."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        # Mock content blocks
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        # Mock response with multiple text blocks
        class MockMessage:
            def __init__(self):
                self.content = [
                    MockTextBlock("First part. "),
                    MockTextBlock("Second part.")
                ]
        
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key-12345")
        
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-sonnet-4.5"
        )
        
        assert response == "First part. Second part."
    
    async def test_anthropic_client_ignore_non_text_blocks(self, monkeypatch):
        """Test that non-text content blocks are ignored gracefully."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        # Mock content blocks
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        class MockToolUseBlock:
            def __init__(self):
                self.type = "tool_use"
                self.name = "calculator"
        
        # Mock response with mixed content types
        class MockMessage:
            def __init__(self):
                self.content = [
                    MockTextBlock("Text before tool. "),
                    MockToolUseBlock(),
                    MockTextBlock("Text after tool.")
                ]
        
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key-12345")
        
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-sonnet-4.5"
        )
        
        # Only text blocks should be extracted
        assert response == "Text before tool. Text after tool."
    
    async def test_anthropic_client_missing_api_key_error(self, monkeypatch):
        """Test that missing API key raises LLMCallError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMCallError
        
        # Remove API key from environment
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        client = AnthropicResponsesClient()
        
        with pytest.raises(LLMCallError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)
        assert exc_info.value.provider == "anthropic"
    
    async def test_anthropic_client_authentication_error(self, monkeypatch):
        """Test that Anthropic AuthenticationError is mapped to LLMAuthenticationError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMAuthenticationError
        import httpx
        
        # Create proper httpx request and response for Anthropic error
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(401, json={"error": {"message": "Invalid API key"}}, request=request)
        
        # Mock AsyncAnthropic to raise AuthenticationError
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    from anthropic import AuthenticationError
                    raise AuthenticationError("Invalid API key", response=response, body=None)
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-invalid-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-invalid-key")
        
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        assert "authentication failed" in str(exc_info.value).lower()
        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.original_error is not None
    
    async def test_anthropic_client_rate_limit_error(self, monkeypatch):
        """Test that Anthropic RateLimitError is mapped to LLMRateLimitError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMRateLimitError
        import httpx
        
        # Create proper httpx request and response
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}}, request=request)
        
        # Mock AsyncAnthropic to raise RateLimitError
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    from anthropic import RateLimitError
                    raise RateLimitError("Rate limit exceeded", response=response, body=None)
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        with pytest.raises(LLMRateLimitError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        assert "rate limit" in str(exc_info.value).lower()
        assert exc_info.value.provider == "anthropic"
    
    async def test_anthropic_client_validation_error(self, monkeypatch):
        """Test that Anthropic BadRequestError is mapped to LLMValidationError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMValidationError
        import httpx
        
        # Create proper httpx request and response
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(400, json={"error": {"message": "Invalid request format"}}, request=request)
        
        # Mock AsyncAnthropic to raise BadRequestError
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    from anthropic import BadRequestError
                    raise BadRequestError("Invalid request format", response=response, body=None)
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        with pytest.raises(LLMValidationError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        assert "validation failed" in str(exc_info.value).lower()
        assert exc_info.value.provider == "anthropic"
    
    async def test_anthropic_client_unprocessable_entity_error(self, monkeypatch):
        """Test that Anthropic UnprocessableEntityError is mapped to LLMValidationError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMValidationError
        import httpx
        
        # Create proper httpx request and response
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(422, json={"error": {"message": "Invalid parameter"}}, request=request)
        
        # Mock AsyncAnthropic to raise UnprocessableEntityError
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    from anthropic import UnprocessableEntityError
                    raise UnprocessableEntityError("Invalid parameter", response=response, body=None)
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        with pytest.raises(LLMValidationError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        assert "validation failed" in str(exc_info.value).lower()
        assert exc_info.value.provider == "anthropic"
    
    async def test_anthropic_client_network_error(self, monkeypatch):
        """Test that Anthropic connection errors are mapped to LLMNetworkError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMNetworkError
        
        # Mock request object for Anthropic exception
        class MockRequest:
            url = "https://api.anthropic.com/v1/messages"
            method = "POST"
        
        # Mock AsyncAnthropic to raise APIConnectionError
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    from anthropic import APIConnectionError
                    raise APIConnectionError(request=MockRequest())
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        with pytest.raises(LLMNetworkError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        assert "network error" in str(exc_info.value).lower()
        assert exc_info.value.provider == "anthropic"
    
    async def test_anthropic_client_timeout_error(self, monkeypatch):
        """Test that Anthropic timeout errors are mapped to LLMNetworkError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMNetworkError
        
        # Mock request object for Anthropic exception
        class MockRequest:
            url = "https://api.anthropic.com/v1/messages"
            method = "POST"
        
        # Mock AsyncAnthropic to raise APITimeoutError
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    from anthropic import APITimeoutError
                    raise APITimeoutError(request=MockRequest())
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        with pytest.raises(LLMNetworkError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        assert "network error" in str(exc_info.value).lower()
        assert exc_info.value.provider == "anthropic"
    
    async def test_anthropic_client_generic_api_error(self, monkeypatch):
        """Test that generic Anthropic APIError is mapped to LLMCallError."""
        from app.services.llm_clients import AnthropicResponsesClient, LLMCallError
        
        # Mock AsyncAnthropic to raise APIError
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    from anthropic import APIError
                    raise APIError("Internal server error", request=None, body=None)
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        with pytest.raises(LLMCallError) as exc_info:
            await client.complete(
                system_prompt="System",
                user_prompt="User",
                model="claude-sonnet-4.5"
            )
        
        # Should not be a more specific error type
        assert type(exc_info.value).__name__ == "LLMCallError"
        assert exc_info.value.provider == "anthropic"
    
    async def test_anthropic_client_validates_empty_system_prompt(self, monkeypatch):
        """Test that AnthropicResponsesClient validates empty system_prompt."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="",
                user_prompt="User query",
                model="claude-sonnet-4.5"
            )
        
        assert "system_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_anthropic_client_validates_empty_user_prompt(self, monkeypatch):
        """Test that AnthropicResponsesClient validates empty user_prompt."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="",
                model="claude-sonnet-4.5"
            )
        
        assert "user_prompt must not be empty or blank" in str(exc_info.value)
    
    async def test_anthropic_client_validates_empty_model(self, monkeypatch):
        """Test that AnthropicResponsesClient validates empty model."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        
        with pytest.raises(ValueError) as exc_info:
            await client.complete(
                system_prompt="System instruction",
                user_prompt="User query",
                model=""
            )
        
        assert "model must not be empty or blank" in str(exc_info.value)
    
    async def test_anthropic_client_lazy_initialization(self, monkeypatch):
        """Test that client is initialized lazily on first use."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        assert client._client is None  # Not initialized yet
        
        # Mock the client after instantiation
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        class MockMessage:
            def __init__(self):
                self.content = [MockTextBlock("Response")]
        
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-sonnet-4.5"
        )
        
        assert client._client is not None  # Now initialized
    
    async def test_anthropic_client_custom_api_key(self, monkeypatch):
        """Test that client can be initialized with custom API key."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        # Don't set environment variable
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        # Mock response
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        class MockMessage:
            def __init__(self):
                self.content = [MockTextBlock("Response with custom key")]
        
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    return MockMessage()
            
            def __init__(self, api_key):
                self.api_key = api_key
                self.messages = self.MockMessages()
        
        client = AnthropicResponsesClient(api_key="sk-ant-custom-key-789")
        client._client = MockAsyncAnthropic(api_key="sk-ant-custom-key-789")
        
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-sonnet-4.5"
        )
        
        assert response == "Response with custom key"
    
    async def test_anthropic_client_implements_protocol(self, monkeypatch):
        """Test that AnthropicResponsesClient implements the LLMClient protocol."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        
        # Test that complete method exists and has the right signature
        assert hasattr(client, "complete")
        assert callable(client.complete)
        
        # Mock the client
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        class MockMessage:
            def __init__(self):
                self.content = [MockTextBlock("Protocol test response")]
        
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        # Test that it's async and returns a string
        result = client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-sonnet-4.5"
        )
        assert hasattr(result, '__await__')
        
        response = await result
        assert isinstance(response, str)
    
    async def test_anthropic_client_opus_model_override(self, monkeypatch):
        """Test that client correctly handles claude-opus-4 model override."""
        from app.services.llm_clients import AnthropicResponsesClient
        
        captured_params = {}
        
        # Mock response
        class MockTextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        class MockMessage:
            def __init__(self):
                self.content = [MockTextBlock("Opus response")]
        
        class MockAsyncAnthropic:
            class MockMessages:
                async def create(self, **kwargs):
                    captured_params.update(kwargs)
                    return MockMessage()
            
            def __init__(self, api_key):
                self.messages = self.MockMessages()
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        
        client = AnthropicResponsesClient()
        client._client = MockAsyncAnthropic(api_key="sk-ant-test-key")
        
        response = await client.complete(
            system_prompt="System",
            user_prompt="User",
            model="claude-opus-4"
        )
        
        # Verify Opus model was passed correctly
        assert captured_params["model"] == "claude-opus-4"
        assert response == "Opus response"
    
    def test_anthropic_client_has_import_error_handling(self):
        """Verify that the complete method has ImportError handling for missing Anthropic SDK."""
        from app.services.llm_clients import AnthropicResponsesClient
        import inspect
        
        # Get the source code of the complete method
        source = inspect.getsource(AnthropicResponsesClient.complete)
        
        # Verify the ImportError handling is present
        assert "except ImportError" in source, "Missing ImportError exception handler"
        assert "Anthropic SDK not installed or accessible" in source, "Missing SDK not installed error message"
        assert "from anthropic import" in source, "Missing anthropic module import"
        
        # The code structure validates that if anthropic cannot be imported,
        # a clear LLMCallError will be raised with appropriate messaging


class TestGetLLMClientFactory:
    """Tests for get_llm_client factory function."""
    
    def test_factory_creates_openai_client(self):
        """Test that factory creates OpenAIResponsesClient for 'openai' provider."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        client = get_llm_client("openai", config)
        assert isinstance(client, OpenAIResponsesClient)
    
    def test_factory_creates_anthropic_client(self):
        """Test that factory creates AnthropicResponsesClient for 'anthropic' provider."""
        config = ClarificationLLMConfig(provider="anthropic", model="claude-sonnet-4.5")
        client = get_llm_client("anthropic", config)
        assert isinstance(client, AnthropicResponsesClient)
    
    def test_factory_creates_dummy_client_for_testing(self):
        """Test that factory creates DummyLLMClient for 'dummy' provider."""
        # Note: 'dummy' is not in SUPPORTED_PROVIDERS but is handled specially
        config = ClarificationLLMConfig(provider="openai", model="test-model")  # Config provider doesn't matter
        client = get_llm_client("dummy", config)
        assert isinstance(client, DummyLLMClient)
    
    def test_factory_raises_on_google_provider_not_yet_implemented(self):
        """Test that factory raises ValueError for 'google' provider (not yet implemented)."""
        config = ClarificationLLMConfig(provider="google", model="gemini-3.0-pro")
        with pytest.raises(ValueError) as exc_info:
            get_llm_client("google", config)
        assert "not yet implemented" in str(exc_info.value).lower()
        assert "google" in str(exc_info.value).lower()
    
    def test_factory_raises_on_unsupported_provider(self):
        """Test that factory raises ValueError for unsupported providers."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        with pytest.raises(ValueError) as exc_info:
            get_llm_client("unsupported", config)
        assert "unsupported provider" in str(exc_info.value).lower()
        assert "unsupported" in str(exc_info.value)
    
    def test_factory_raises_on_empty_provider(self):
        """Test that factory raises ValueError for empty provider string."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        with pytest.raises(ValueError) as exc_info:
            get_llm_client("", config)
        assert "provider must not be empty or blank" in str(exc_info.value)
    
    def test_factory_raises_on_blank_provider(self):
        """Test that factory raises ValueError for blank provider string."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        with pytest.raises(ValueError) as exc_info:
            get_llm_client("   ", config)
        assert "provider must not be empty or blank" in str(exc_info.value)
    
    def test_factory_normalizes_provider_case(self):
        """Test that factory handles different case variations of provider names."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        
        # All these should work
        client1 = get_llm_client("OpenAI", config)
        client2 = get_llm_client("OPENAI", config)
        client3 = get_llm_client("openai", config)
        client4 = get_llm_client("  openai  ", config)  # With whitespace
        
        assert isinstance(client1, OpenAIResponsesClient)
        assert isinstance(client2, OpenAIResponsesClient)
        assert isinstance(client3, OpenAIResponsesClient)
        assert isinstance(client4, OpenAIResponsesClient)
    
    def test_factory_normalizes_anthropic_case(self):
        """Test that factory handles different case variations for Anthropic."""
        config = ClarificationLLMConfig(provider="anthropic", model="claude-sonnet-4.5")
        
        client1 = get_llm_client("Anthropic", config)
        client2 = get_llm_client("ANTHROPIC", config)
        client3 = get_llm_client("anthropic", config)
        
        assert isinstance(client1, AnthropicResponsesClient)
        assert isinstance(client2, AnthropicResponsesClient)
        assert isinstance(client3, AnthropicResponsesClient)
    
    def test_factory_dummy_case_insensitive(self):
        """Test that 'dummy' provider is case-insensitive."""
        config = ClarificationLLMConfig(provider="openai", model="test-model")
        
        client1 = get_llm_client("dummy", config)
        client2 = get_llm_client("Dummy", config)
        client3 = get_llm_client("DUMMY", config)
        
        assert isinstance(client1, DummyLLMClient)
        assert isinstance(client2, DummyLLMClient)
        assert isinstance(client3, DummyLLMClient)
    
    def test_factory_creates_independent_client_instances(self):
        """Test that factory creates new client instances on each call."""
        # Use valid provider in config, but request 'dummy' from factory
        config = ClarificationLLMConfig(provider="openai", model="test-model")
        
        client1 = get_llm_client("dummy", config)
        client2 = get_llm_client("dummy", config)
        
        # Should be different instances
        assert client1 is not client2
    
    def test_factory_accepts_config_parameter(self):
        """Test that factory accepts and uses config parameter."""
        config1 = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        config2 = ClarificationLLMConfig(provider="anthropic", model="claude-sonnet-4.5")
        
        # Config parameter is accepted but not currently used by factory
        # (it will be used when we pass it to clients in future iterations)
        client1 = get_llm_client("openai", config1)
        client2 = get_llm_client("anthropic", config2)
        
        assert isinstance(client1, OpenAIResponsesClient)
        assert isinstance(client2, AnthropicResponsesClient)
    
    def test_factory_error_message_lists_supported_providers(self):
        """Test that error message for unsupported provider lists valid options."""
        config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        
        with pytest.raises(ValueError) as exc_info:
            get_llm_client("invalid_provider", config)
        
        error_msg = str(exc_info.value)
        assert "supported providers" in error_msg.lower()
        # Should list the actual supported providers
        assert "openai" in error_msg
        assert "anthropic" in error_msg
    
    def test_factory_unknown_provider_vs_unimplemented_provider(self):
        """Test distinction between unknown and unimplemented providers."""
        config = ClarificationLLMConfig(provider="openai", model="test-model")
        
        # Unknown provider (not in SUPPORTED_PROVIDERS)
        with pytest.raises(ValueError) as exc_info1:
            get_llm_client("unknown", config)
        assert "unsupported" in str(exc_info1.value).lower()
        
        # Known but unimplemented provider (in SUPPORTED_PROVIDERS)
        with pytest.raises(ValueError) as exc_info2:
            get_llm_client("google", config)
        assert "not yet implemented" in str(exc_info2.value).lower()
    
    def test_factory_returns_llm_client_protocol_implementations(self):
        """Test that all clients returned by factory implement LLMClient protocol."""
        config_openai = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        config_anthropic = ClarificationLLMConfig(provider="anthropic", model="claude-sonnet-4.5")
        # Use valid provider for config even though factory will create dummy
        config_for_dummy = ClarificationLLMConfig(provider="openai", model="test")
        
        client_openai = get_llm_client("openai", config_openai)
        client_anthropic = get_llm_client("anthropic", config_anthropic)
        client_dummy = get_llm_client("dummy", config_for_dummy)
        
        # All should have the 'complete' method
        assert hasattr(client_openai, "complete")
        assert callable(client_openai.complete)
        assert hasattr(client_anthropic, "complete")
        assert callable(client_anthropic.complete)
        assert hasattr(client_dummy, "complete")
        assert callable(client_dummy.complete)
    
    def test_factory_with_special_characters_in_provider(self):
        """Test that factory handles special characters in provider name correctly."""
        config = ClarificationLLMConfig(provider="openai", model="test-model")
        
        # These should all fail with appropriate errors
        invalid_providers = ["open-ai", "open_ai", "openai!", "open/ai"]
        for invalid in invalid_providers:
            with pytest.raises(ValueError):
                get_llm_client(invalid, config)

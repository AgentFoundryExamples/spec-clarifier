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
"""LLM client abstractions and configuration for specification clarification.

This module provides:
1. LLMClient Protocol/ABC - a unified interface for all LLM providers
2. ClarificationLLMConfig - configuration model for LLM settings
3. LLMCallError - exception hierarchy for handling LLM API failures
4. DummyLLMClient - a test implementation for deterministic testing

Provider-Specific Implementation Notes:
---------------------------------------
OpenAI GPT-5:
    - Use the Responses API (not Completions API, which is deprecated)
    - Target model: gpt-5.1
    - Response parsing: Extract text from the 'choices' array in the response
    - Use official 'openai' Python SDK

Anthropic Claude Sonnet/Opus 4:
    - Use the Messages API (API version 2023-06-01 or newer)
    - Target model: claude-sonnet-4.5 or claude-opus-4
    - Response parsing: Extract text from 'content' array in the message
    - Use official 'anthropic' Python SDK

Google Gemini 3:
    - Use the Gemini API
    - Target model: gemini-3.0-pro
    - Response parsing: Extract text from the 'candidates' response structure
    - Use official 'google-genai' Python SDK

Each provider implementation should:
1. Implement the LLMClient Protocol
2. Handle provider-specific response formats
3. Wrap API errors with LLMCallError
4. Sanitize error messages to prevent credential leakage
"""

import logging
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

from app.utils.logging_helper import log_error, log_info
from app.utils.metrics import get_metrics_collector

# Provider constants
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE = "google"

SUPPORTED_PROVIDERS = frozenset([PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GOOGLE])


class LLMCallError(Exception):
    """Base exception for LLM API call failures.
    
    This exception wraps downstream API/network failures and ensures
    error messages are sanitized to prevent credential leakage.
    
    Attributes:
        message: Sanitized error message safe for logging
        original_error: The original exception (if available)
        provider: The LLM provider that failed (if known)
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        provider: str | None = None
    ):
        """Initialize LLMCallError with sanitized message.
        
        Args:
            message: Human-readable error message
            original_error: Original exception that was wrapped
            provider: LLM provider identifier (e.g., 'openai', 'anthropic')
        """
        sanitized_message = self._sanitize_message(message)
        super().__init__(sanitized_message)
        self.message = sanitized_message
        self.original_error = original_error
        self.provider = provider

    @staticmethod
    def _sanitize_message(message: str) -> str:
        """Remove potential secrets from error messages.
        
        This method strips common patterns that might contain API keys,
        tokens, or other sensitive information from error messages.
        
        Args:
            message: Original error message
            
        Returns:
            Sanitized error message with secrets replaced
        """
        # Remove API keys (various formats including JSON and URL-encoded)
        message = re.sub(r'(api[_-]?key["\s:=]+)[^\s\'"]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)
        message = re.sub(r'(["\']api[_-]?key["\']\s*:\s*["\'])[^"\'\n\r]+?(["\'])', r'\1[REDACTED]\2', message, flags=re.IGNORECASE)
        message = re.sub(r'(api[_-]?key%3D)[^&\s]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)

        # Remove bearer tokens
        message = re.sub(r'(bearer\s+)[^\s]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)

        # Remove tokens (various formats including JSON and URL-encoded)
        message = re.sub(r'(token["\s:=]+)[^\s\'"]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)
        message = re.sub(r'(["\']token["\']\s*:\s*["\'])[^"\'\n\r]+?(["\'])', r'\1[REDACTED]\2', message, flags=re.IGNORECASE)
        message = re.sub(r'(token%3D)[^&\s]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)

        # Remove authorization headers
        message = re.sub(r'(authorization["\s:]+)[^\r\n]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)

        # Remove x-api-key headers
        message = re.sub(r'(x-api-key["\s:]+)[^\r\n]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)

        # Remove secrets in JSON format (generic patterns)
        message = re.sub(r'(["\'](?:secret|key|password|apikey)["\']\s*:\s*["\'])[^"\'\n\r]+?(["\'])', r'\1[REDACTED]\2', message, flags=re.IGNORECASE)

        return message

    def __str__(self) -> str:
        """Return sanitized error message."""
        return self.message


class LLMNetworkError(LLMCallError):
    """Exception raised when network/connectivity issues occur."""
    pass


class LLMAuthenticationError(LLMCallError):
    """Exception raised when authentication fails (invalid API key, etc.)."""
    pass


class LLMRateLimitError(LLMCallError):
    """Exception raised when rate limits are exceeded."""
    pass


class LLMValidationError(LLMCallError):
    """Exception raised when request validation fails."""
    pass


class ClarificationLLMConfig(BaseModel):
    """Configuration for LLM clients used in clarification workflows.
    
    This model captures all necessary configuration for making LLM API calls,
    including provider selection, model specification, and generation parameters.
    
    Attributes:
        provider: LLM provider identifier (openai, anthropic, google)
        model: Model identifier specific to the provider
        temperature: Sampling temperature (0.0-2.0), defaults to 0.1 for deterministic output
        max_tokens: Optional maximum tokens to generate in response
    """

    provider: str = Field(
        ...,
        description="LLM provider identifier (openai, anthropic, google)"
    )
    model: str = Field(
        ...,
        description="Model identifier specific to the provider"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation"
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate in response"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate that provider is one of the supported values.
        
        Args:
            v: Provider string to validate
            
        Returns:
            Validated provider string
            
        Raises:
            ValueError: If provider is not supported
        """
        # Include "dummy" as a valid provider for testing
        valid_providers = SUPPORTED_PROVIDERS | {"dummy"}
        if v not in valid_providers:
            raise ValueError(
                f"Invalid provider '{v}'. Must be one of: {', '.join(sorted(valid_providers))}"
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate that model string is not empty or blank.
        
        Args:
            v: Model string to validate
            
        Returns:
            Validated model string
            
        Raises:
            ValueError: If model is empty or blank
        """
        if not v or not v.strip():
            raise ValueError("Model must not be empty or blank")
        return v.strip()


class LLMClient(Protocol):
    """Protocol defining the interface for LLM clients.
    
    All LLM provider implementations must implement this protocol to ensure
    consistent behavior across different providers (OpenAI, Anthropic, Google).
    
    The protocol enforces:
    1. Async-only operations for non-blocking I/O
    2. Consistent signature across providers
    3. Raw text string responses (provider-agnostic)
    4. Error handling via LLMCallError hierarchy
    
    Example implementations:
        - OpenAIClient: Wraps OpenAI Responses API
        - AnthropicClient: Wraps Anthropic Messages API
        - GoogleClient: Wraps Google Gemini API
        - DummyLLMClient: Test double for deterministic testing
    """

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        **kwargs: Any
    ) -> str:
        """Generate a completion from the LLM.
        
        This method sends a request to the LLM provider with system and user prompts,
        waits for the response, and returns the raw text content. Provider-specific
        response parsing is handled by the implementation.
        
        Args:
            system_prompt: System-level instructions for the LLM
            user_prompt: User's actual request/query
            model: Model identifier to use for this completion
            **kwargs: Additional provider-specific parameters (temperature, max_tokens, etc.)
            
        Returns:
            Raw text string response from the LLM
            
        Raises:
            LLMCallError: When API call fails (network, auth, validation, etc.)
            ValueError: When prompts are empty or invalid
        """
        ...


class DummyLLMClient:
    """Test implementation of LLMClient that returns deterministic responses.
    
    This client is designed for testing purposes and never makes real API calls.
    It can be configured to:
    1. Return canned JSON responses
    2. Echo prompts back to the caller
    3. Simulate various failure scenarios
    
    Example usage:
        # Return canned JSON
        client = DummyLLMClient(
            canned_response='{"result": "clarified specification"}'
        )
        
        # Echo prompts
        client = DummyLLMClient(echo_prompts=True)
        
        # Simulate failures
        client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Rate limit exceeded"
        )
    
    Attributes:
        canned_response: Predefined response to return
        echo_prompts: If True, return a formatted echo of the input prompts
        simulate_failure: If True, raise LLMCallError on complete()
        failure_message: Custom error message for simulated failures
        failure_type: Type of error to simulate (LLMCallError subclass)
    """

    def __init__(
        self,
        canned_response: str | None = None,
        echo_prompts: bool = False,
        simulate_failure: bool = False,
        failure_message: str = "Simulated LLM failure",
        failure_type: type[LLMCallError] = LLMCallError
    ):
        """Initialize DummyLLMClient with test behavior configuration.
        
        Args:
            canned_response: Predefined response to return (overrides echo_prompts)
            echo_prompts: If True, echo prompts back in a formatted string
            simulate_failure: If True, raise errors instead of returning responses
            failure_message: Error message to use when simulating failures
            failure_type: Type of LLMCallError to raise (default: LLMCallError)
        """
        if simulate_failure and not issubclass(failure_type, LLMCallError):
            raise TypeError("failure_type must be a subclass of LLMCallError")

        self.canned_response = canned_response
        self.echo_prompts = echo_prompts
        self.simulate_failure = simulate_failure
        self.failure_message = failure_message
        self.failure_type = failure_type

        # Default canned response if nothing specified
        if not canned_response and not echo_prompts:
            self.canned_response = '{"clarified": true}'

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        **kwargs: Any
    ) -> str:
        """Generate a deterministic test response.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User's actual request/query
            model: Model identifier (recorded but not used)
            **kwargs: Additional parameters (recorded but not used)
            
        Returns:
            Deterministic test response based on configuration
            
        Raises:
            LLMCallError: When simulate_failure=True
            ValueError: When prompts are empty or blank
        """
        # Validate inputs like a real client would
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must not be empty or blank")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must not be empty or blank")
        if not model or not model.strip():
            raise ValueError("model must not be empty or blank")

        # Simulate failure if configured
        if self.simulate_failure:
            raise self.failure_type(
                message=self.failure_message,
                provider="dummy"
            )

        # Return canned response if provided
        if self.canned_response is not None:
            return self.canned_response

        # Echo prompts if configured
        if self.echo_prompts:
            return f"System: {system_prompt}\nUser: {user_prompt}\nModel: {model}"

        # Fallback (should never reach here due to __init__ logic)
        return '{"clarified": true}'


class OpenAIResponsesClient:
    """OpenAI implementation of LLMClient using the Responses API.
    
    This client wraps the OpenAI Python SDK (v2.x) and implements the LLMClient
    protocol using the modern Responses API (not the deprecated Completions API).
    
    The client:
    1. Initializes lazily for testability (client created on first use)
    2. Uses AsyncOpenAI for non-blocking I/O
    3. Formats messages according to Responses API schema
    4. Extracts text from response.output_text or response.content
    5. Maps OpenAI errors to LLMCallError hierarchy
    6. Logs provider/model/duration but not prompts/responses
    
    Configuration:
        - Requires OPENAI_API_KEY environment variable
        - Supports model specification per request
        - Passes through kwargs like temperature, max_tokens
    
    Example usage:
        client = OpenAIResponsesClient()
        response = await client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Explain quantum computing",
            model="gpt-5.1",
            temperature=0.7,
            max_tokens=500
        )
    """

    def __init__(self, api_key: str | None = None):
        """Initialize OpenAIResponsesClient with optional API key.
        
        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY
                    environment variable. Lazy initialization defers client
                    creation until first API call.
        """
        self._api_key = api_key
        self._client: Any | None = None
        self._logger = logging.getLogger(__name__)

    def _get_client(self) -> Any:
        """Get or create AsyncOpenAI client instance (lazy initialization).
        
        Returns:
            AsyncOpenAI client instance
            
        Raises:
            LLMCallError: If OPENAI_API_KEY is not available
        """
        if self._client is None:
            # Import here to avoid import errors if openai is not installed
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise LLMCallError(
                    "OpenAI SDK not installed. Install with: pip install openai",
                    original_error=e,
                    provider=PROVIDER_OPENAI
                )

            # Get API key from instance or environment
            import os
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")

            if not api_key:
                raise LLMCallError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Please set it with a valid OpenAI API key.",
                    provider=PROVIDER_OPENAI
                )

            self._client = AsyncOpenAI(api_key=api_key)
            self._logger.debug("Initialized AsyncOpenAI client")

        return self._client

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        **kwargs: Any
    ) -> str:
        """Generate a completion from OpenAI using the Responses API.
        
        This method sends a request to OpenAI's Responses API with system and
        user prompts, waits for the response, and extracts the text content.
        
        The Responses API uses:
        - `instructions` parameter for system-level instructions
        - `input` parameter for the user's actual query
        - `model` parameter for model selection
        
        Additional parameters (temperature, max_tokens, etc.) are passed through
        via kwargs and renamed to match OpenAI's parameter names.
        
        Args:
            system_prompt: System-level instructions for the LLM
            user_prompt: User's actual request/query
            model: Model identifier (e.g., "gpt-5.1", "gpt-4o")
            **kwargs: Additional parameters:
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum tokens to generate (renamed to max_output_tokens)
                - Other OpenAI-specific parameters
        
        Returns:
            Raw text string response from OpenAI
            
        Raises:
            ValueError: When prompts are empty or invalid
            LLMAuthenticationError: When API key is invalid
            LLMRateLimitError: When rate limits are exceeded
            LLMValidationError: When request validation fails
            LLMNetworkError: When network/connectivity issues occur
            LLMCallError: For other API errors
        """
        # Validate inputs
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must not be empty or blank")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must not be empty or blank")
        if not model or not model.strip():
            raise ValueError("model must not be empty or blank")

        # Get or create client
        client = self._get_client()

        # Prepare API parameters - map our kwargs to OpenAI's parameter names
        api_params = {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
        }

        # Map max_tokens to max_output_tokens for Responses API
        if "max_tokens" in kwargs:
            api_params["max_output_tokens"] = kwargs.pop("max_tokens")

        # Pass through other parameters
        api_params.update(kwargs)

        # Track timing for logging using perf_counter for better async performance
        import time
        start_time = time.perf_counter()

        try:
            from openai import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
            )
        except ImportError as import_err:
            raise LLMCallError(
                "OpenAI SDK not installed or accessible. Install with: pip install openai",
                original_error=import_err,
                provider=PROVIDER_OPENAI
            ) from import_err

        try:
            # Call OpenAI Responses API
            response = await client.responses.create(**api_params)

        except (AuthenticationError, RateLimitError, BadRequestError, APIConnectionError, APITimeoutError, APIError) as e:
            # Calculate elapsed time for error logging
            elapsed_time = time.perf_counter() - start_time
            metrics = get_metrics_collector()
            metrics.increment("llm_errors")

            # Map OpenAI errors to our error hierarchy
            if isinstance(e, AuthenticationError):
                log_error(
                    self._logger,
                    "llm_authentication_failed",
                    provider=PROVIDER_OPENAI,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMAuthenticationError(
                    "OpenAI authentication failed. Please check your API key.",
                    original_error=e,
                    provider=PROVIDER_OPENAI
                ) from e

            if isinstance(e, RateLimitError):
                log_error(
                    self._logger,
                    "llm_rate_limit_exceeded",
                    provider=PROVIDER_OPENAI,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMRateLimitError(
                    "OpenAI rate limit exceeded. Please retry after a delay.",
                    original_error=e,
                    provider=PROVIDER_OPENAI
                ) from e

            if isinstance(e, BadRequestError):
                log_error(
                    self._logger,
                    "llm_validation_failed",
                    provider=PROVIDER_OPENAI,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMValidationError(
                    f"OpenAI request validation failed: {str(e)}",
                    original_error=e,
                    provider=PROVIDER_OPENAI
                ) from e

            if isinstance(e, (APIConnectionError, APITimeoutError)):
                log_error(
                    self._logger,
                    "llm_network_error",
                    provider=PROVIDER_OPENAI,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMNetworkError(
                    f"OpenAI network error: {type(e).__name__}",
                    original_error=e,
                    provider=PROVIDER_OPENAI
                ) from e

            # Fallback for other APIError subclasses
            log_error(
                self._logger,
                "llm_api_error",
                provider=PROVIDER_OPENAI,
                model=model,
                elapsed_seconds=round(elapsed_time, 2),
                error=e
            )
            raise LLMCallError(
                f"OpenAI API error: {str(e)}",
                original_error=e,
                provider=PROVIDER_OPENAI
            ) from e

        # Extract text from response (outside the API error handling block)
        text_content = response.output_text

        # If output_text is empty and we have output, try to extract from output list manually
        if not text_content and hasattr(response, "output"):
            text_parts = []
            for output_item in response.output:
                # Handle message output items
                if hasattr(output_item, "type") and output_item.type == "message":
                    if hasattr(output_item, "content"):
                        for content_part in output_item.content:
                            # Extract text from output_text content blocks
                            if hasattr(content_part, "type") and content_part.type == "output_text":
                                if hasattr(content_part, "text"):
                                    text_parts.append(content_part.text)
            text_content = "".join(text_parts)

        # Validate that we got some content
        if not text_content:
            log_error(
                self._logger,
                "llm_empty_response",
                provider=PROVIDER_OPENAI,
                model=model
            )

        # Log success (provider, model, duration only)
        elapsed_time = time.perf_counter() - start_time
        log_info(
            self._logger,
            "llm_completion_success",
            provider=PROVIDER_OPENAI,
            model=model,
            elapsed_seconds=round(elapsed_time, 2)
        )

        return text_content


def get_llm_client(provider: str, config: ClarificationLLMConfig) -> Any:
    """Factory function to create LLM clients based on provider configuration.
    
    This factory provides a centralized way to construct LLM clients for different
    providers (OpenAI, Anthropic, Google, Dummy) while maintaining consistent
    configuration and error handling. It validates that required API keys are
    present before creating real provider clients.
    
    Provider-specific implementations are initialized lazily (client created on first
    use) and configured according to the provided ClarificationLLMConfig settings.
    
    Args:
        provider: LLM provider identifier ('openai', 'anthropic', 'google', 'dummy')
        config: Configuration object containing model, temperature, max_tokens, etc.
                Note: Currently, clients retrieve API keys from environment variables.
                The config parameter is reserved for future extensibility (e.g., passing
                API keys, custom endpoints, or other provider-specific settings).
        
    Returns:
        An instance implementing the LLMClient protocol:
        - OpenAIResponsesClient for 'openai' provider
        - AnthropicResponsesClient for 'anthropic' provider
        - DummyLLMClient for 'dummy' provider (testing)
        - Future: GoogleClient for 'google' provider
        
    Raises:
        ValueError: When provider is not supported or is empty/blank
        LLMCallError: When API keys are missing for real providers (openai/anthropic)
        
    Example:
        >>> config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        >>> client = get_llm_client("openai", config)
        >>> # Client is ready but not invoked yet
        
        >>> # For testing with deterministic responses
        >>> # Note: Use a valid provider in config, but request 'dummy' from factory
        >>> test_config = ClarificationLLMConfig(provider="openai", model="test-model")
        >>> test_client = get_llm_client("dummy", test_config)
    
    Note:
        The factory does not invoke the LLM - it only constructs and returns the
        client instance. Actual API calls happen when client.complete() is called.
        Model, temperature, and other parameters from config are passed during the
        complete() invocation, not during client construction.
    """
    # Validate provider parameter
    if not provider or not provider.strip():
        raise ValueError("provider must not be empty or blank")

    provider = provider.strip().lower()

    # Special case: 'dummy' provider for testing (not in SUPPORTED_PROVIDERS)
    if provider == "dummy":
        # For dummy client, we return a default canned response
        # Tests can customize behavior by creating DummyLLMClient directly
        return DummyLLMClient()

    # Validate against supported providers
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}, dummy"
        )

    # Check for required API keys before creating real provider clients
    import os

    # Route to provider-specific implementation
    # Note: Clients currently use environment variables for API keys.
    # Future enhancement: Pass config.api_key if/when added to ClarificationLLMConfig
    if provider == PROVIDER_OPENAI:
        # Validate API key is present
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMCallError(
                "OPENAI_API_KEY environment variable is required to use the OpenAI provider. "
                "Set it with a valid OpenAI API key or use LLM_PROVIDER=dummy for testing.",
                provider=PROVIDER_OPENAI
            )
        return OpenAIResponsesClient()

    elif provider == PROVIDER_ANTHROPIC:
        # Validate API key is present
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMCallError(
                "ANTHROPIC_API_KEY environment variable is required to use the Anthropic provider. "
                "Set it with a valid Anthropic API key or use LLM_PROVIDER=dummy for testing.",
                provider=PROVIDER_ANTHROPIC
            )
        return AnthropicResponsesClient()

    elif provider == PROVIDER_GOOGLE:
        # TODO: Implement GoogleClient when Google Gemini integration is ready
        # For now, raise an error indicating this is not yet implemented
        raise ValueError(
            f"Provider '{PROVIDER_GOOGLE}' is supported but not yet implemented. "
            "Google Gemini integration is coming soon."
        )

    else:
        # This should never be reached due to SUPPORTED_PROVIDERS check above,
        # but included for completeness and future-proofing
        raise ValueError(f"Provider '{provider}' routing not implemented")


class AnthropicResponsesClient:
    """Anthropic implementation of LLMClient using the Messages API.
    
    This client wraps the Anthropic Python SDK and implements the LLMClient
    protocol using the Messages API (API version 2023-06-01 or newer).
    
    The client:
    1. Initializes lazily for testability (client created on first use)
    2. Uses AsyncAnthropic for non-blocking I/O
    3. Formats messages according to Messages API schema (system + user messages)
    4. Extracts text from response.content array
    5. Maps Anthropic errors to LLMCallError hierarchy
    6. Logs provider/model/duration but not prompts/responses
    
    Configuration:
        - Requires ANTHROPIC_API_KEY environment variable
        - Defaults to claude-sonnet-4.5 (Claude Sonnet 4.5)
        - Supports claude-opus-4 (Claude Opus 4) via model parameter
        - Passes through kwargs like temperature, max_tokens
    
    Example usage:
        client = AnthropicResponsesClient()
        response = await client.complete(
            system_prompt="You are a helpful assistant",
            user_prompt="Explain quantum computing",
            model="claude-sonnet-4.5",
            temperature=0.7,
            max_tokens=500
        )
    """

    # Default model constants
    DEFAULT_MODEL = "claude-sonnet-4.5"

    def __init__(self, api_key: str | None = None):
        """Initialize AnthropicResponsesClient with optional API key.
        
        Args:
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY
                    environment variable. Lazy initialization defers client
                    creation until first API call.
        """
        self._api_key = api_key
        self._client: Any | None = None
        self._logger = logging.getLogger(__name__)

    def _get_client(self) -> Any:
        """Get or create AsyncAnthropic client instance (lazy initialization).
        
        Returns:
            AsyncAnthropic client instance
            
        Raises:
            LLMCallError: If ANTHROPIC_API_KEY is not available
        """
        if self._client is None:
            # Import here to avoid import errors if anthropic is not installed
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise LLMCallError(
                    "Anthropic SDK not installed. Install with: pip install anthropic",
                    original_error=e,
                    provider=PROVIDER_ANTHROPIC
                )

            # Get API key from instance or environment
            import os
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                raise LLMCallError(
                    "Anthropic API key is not set. Please provide it via the "
                    "'api_key' argument or set the ANTHROPIC_API_KEY environment variable.",
                    provider=PROVIDER_ANTHROPIC
                )

            self._client = AsyncAnthropic(api_key=api_key)
            self._logger.debug("Initialized AsyncAnthropic client")

        return self._client

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        **kwargs: Any
    ) -> str:
        """Generate a completion from Anthropic using the Messages API.
        
        This method sends a request to Anthropic's Messages API with system and
        user prompts, waits for the response, and extracts the text content.
        
        The Messages API uses:
        - `system` parameter for system-level instructions
        - `messages` parameter for user/assistant conversation history
        - `model` parameter for model selection
        - `max_tokens` parameter is required by Anthropic
        
        Additional parameters (temperature, etc.) are passed through via kwargs.
        
        Args:
            system_prompt: System-level instructions for the LLM
            user_prompt: User's actual request/query
            model: Model identifier (e.g., "claude-sonnet-4.5", "claude-opus-4")
            **kwargs: Additional parameters:
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum tokens to generate (required by Anthropic)
                - Other Anthropic-specific parameters
        
        Returns:
            Raw text string response from Anthropic
            
        Raises:
            ValueError: When prompts are empty or invalid
            LLMAuthenticationError: When API key is invalid
            LLMRateLimitError: When rate limits are exceeded
            LLMValidationError: When request validation fails
            LLMNetworkError: When network/connectivity issues occur
            LLMCallError: For other API errors
        """
        # Validate inputs
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must not be empty or blank")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must not be empty or blank")
        if not model or not model.strip():
            raise ValueError("model must not be empty or blank")

        # Get or create client
        client = self._get_client()

        # Import exception classes (after client is created, so anthropic is available)
        try:
            from anthropic import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
                UnprocessableEntityError,
            )
        except ImportError as import_err:
            raise LLMCallError(
                "Anthropic SDK not installed or accessible. Install with: pip install anthropic",
                original_error=import_err,
                provider=PROVIDER_ANTHROPIC
            ) from import_err

        # Prepare API parameters - Anthropic Messages API format
        api_params = {
            "model": model,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
        }

        # Ensure max_tokens is present (required by Anthropic)
        if "max_tokens" not in kwargs:
            # Use a reasonable default if not specified
            api_params["max_tokens"] = 4096
        else:
            api_params["max_tokens"] = kwargs.pop("max_tokens")

        # Pass through other parameters
        api_params.update(kwargs)

        # Track timing for logging using perf_counter for better async performance
        import time
        start_time = time.perf_counter()

        try:
            # Call Anthropic Messages API
            response = await client.messages.create(**api_params)

        except (AuthenticationError, RateLimitError, BadRequestError, UnprocessableEntityError,
                APIConnectionError, APITimeoutError, APIError) as e:
            # Calculate elapsed time for error logging
            elapsed_time = time.perf_counter() - start_time
            metrics = get_metrics_collector()
            metrics.increment("llm_errors")

            # Map Anthropic errors to our error hierarchy
            if isinstance(e, AuthenticationError):
                log_error(
                    self._logger,
                    "llm_authentication_failed",
                    provider=PROVIDER_ANTHROPIC,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMAuthenticationError(
                    "Anthropic authentication failed. Please check your API key.",
                    original_error=e,
                    provider=PROVIDER_ANTHROPIC
                ) from e

            elif isinstance(e, RateLimitError):
                log_error(
                    self._logger,
                    "llm_rate_limit_exceeded",
                    provider=PROVIDER_ANTHROPIC,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMRateLimitError(
                    "Anthropic rate limit exceeded. Please retry after a delay.",
                    original_error=e,
                    provider=PROVIDER_ANTHROPIC
                ) from e

            elif isinstance(e, (BadRequestError, UnprocessableEntityError)):
                log_error(
                    self._logger,
                    "llm_validation_failed",
                    provider=PROVIDER_ANTHROPIC,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMValidationError(
                    f"Anthropic request validation failed: {str(e)}",
                    original_error=e,
                    provider=PROVIDER_ANTHROPIC
                ) from e

            elif isinstance(e, (APIConnectionError, APITimeoutError)):
                log_error(
                    self._logger,
                    "llm_network_error",
                    provider=PROVIDER_ANTHROPIC,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMNetworkError(
                    f"Anthropic network error: {type(e).__name__}",
                    original_error=e,
                    provider=PROVIDER_ANTHROPIC
                ) from e

            else:
                # Fallback for other APIError subclasses
                log_error(
                    self._logger,
                    "llm_api_error",
                    provider=PROVIDER_ANTHROPIC,
                    model=model,
                    elapsed_seconds=round(elapsed_time, 2),
                    error=e
                )
                raise LLMCallError(
                    f"Anthropic API error: {str(e)}",
                    original_error=e,
                    provider=PROVIDER_ANTHROPIC
                ) from e

        # Extract text from response content array
        text_parts = []
        for content_block in response.content:
            # Handle text blocks
            if hasattr(content_block, "type") and content_block.type == "text":
                if hasattr(content_block, "text"):
                    text_parts.append(content_block.text)

        text_content = "".join(text_parts)

        # Validate that we got some content
        if not text_content:
            log_error(
                self._logger,
                "llm_empty_response",
                provider=PROVIDER_ANTHROPIC,
                model=model
            )

        # Log success (provider, model, duration only)
        elapsed_time = time.perf_counter() - start_time
        log_info(
            self._logger,
            "llm_completion_success",
            provider=PROVIDER_ANTHROPIC,
            model=model,
            elapsed_seconds=round(elapsed_time, 2)
        )

        return text_content

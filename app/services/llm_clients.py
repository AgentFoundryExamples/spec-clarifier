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

import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field, field_validator


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
        original_error: Optional[Exception] = None,
        provider: Optional[str] = None
    ):
        """Initialize LLMCallError with sanitized message.
        
        Args:
            message: Human-readable error message
            original_error: Original exception that was wrapped
            provider: LLM provider identifier (e.g., 'openai', 'anthropic')
        """
        super().__init__(message)
        self.message = self._sanitize_message(message)
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
        # Remove API keys (various formats)
        message = re.sub(r'(api[_-]?key["\s:=]+)[^\s\'"]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)
        message = re.sub(r'(bearer\s+)[^\s]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)
        message = re.sub(r'(token["\s:=]+)[^\s\'"]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)
        
        # Remove authorization headers
        message = re.sub(r'(authorization["\s:]+)[^\r\n]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)
        
        # Remove x-api-key headers
        message = re.sub(r'(x-api-key["\s:]+)[^\r\n]+', r'\1[REDACTED]', message, flags=re.IGNORECASE)
        
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
    max_tokens: Optional[int] = Field(
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
        if v not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Invalid provider '{v}'. Must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
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
        canned_response: Optional[str] = None,
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

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
"""Application configuration."""

import os
import threading
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Import ClarificationConfig from config_models to avoid circular dependency
from app.models.config_models import ClarificationConfig


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        case_sensitive=False,
    )
    
    app_name: str = "Spec Clarifier"
    app_version: str = "0.1.0"
    app_description: str = "A service for clarifying specifications"
    debug: bool = False
    
    # Development flag for exposing job results
    show_job_result: bool = False
    
    # Debug endpoint flag (off by default for security)
    enable_debug_endpoint: bool = False
    
    # LLM configuration defaults
    llm_default_provider: str = "openai"
    llm_default_model: str = "gpt-5"
    
    # CORS settings
    cors_origins: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "*"
    cors_allow_headers: str = "*"
    
    def get_cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string to list.
        
        Returns:
            List of allowed CORS origins
        """
        if not self.cors_origins:
            return []
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    def get_cors_methods_list(self) -> list[str]:
        """Parse CORS methods from comma-separated string to list.
        
        Returns:
            List of allowed CORS methods or ["*"] for wildcard
        """
        if self.cors_allow_methods == "*":
            return ["*"]
        return [method.strip() for method in self.cors_allow_methods.split(",") if method.strip()]
    
    def get_cors_headers_list(self) -> list[str]:
        """Parse CORS headers from comma-separated string to list.
        
        Returns:
            List of allowed CORS headers or ["*"] for wildcard
        """
        if self.cors_allow_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_allow_headers.split(",") if header.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


# ============================================================================
# GLOBAL DEFAULTS FOR CLARIFICATION CONFIG
# ============================================================================


class ConfigValidationError(ValueError):
    """Exception raised when config validation fails.
    
    This exception provides detailed error messages for configuration
    validation failures, including provider/model membership checks.
    """
    pass


class GlobalDefaults:
    """Global defaults for ClarificationConfig with validation.
    
    This class manages runtime defaults for clarification configuration,
    including allowed models per provider and the default configuration.
    It provides thread-safe access to defaults with validation to ensure
    provider/model membership constraints are satisfied.
    
    Design principles:
    - Thread-safe reads and writes using threading.Lock
    - Single writer, multiple readers pattern
    - Validates provider/model membership before mutations
    - Falls back to built-in defaults if environment config is invalid
    - Does NOT persist outside process memory
    
    Attributes:
        allowed_models: Dictionary mapping provider names to lists of allowed models
        default_config: Default ClarificationConfig to use when none is specified
    
    The class uses module-level state protected by a lock to ensure thread safety.
    Access defaults through get_default_config() and mutate through set_default_config().
    
    Example:
        >>> defaults = GlobalDefaults()
        >>> config = defaults.get_default_config()
        >>> # Modify defaults (with validation)
        >>> new_config = ClarificationConfig(provider="anthropic", model="claude-sonnet-4.5", system_prompt_id="default")
        >>> defaults.set_default_config(new_config)
    """
    
    def __init__(self):
        """Initialize GlobalDefaults with built-in safe defaults.
        
        Built-in defaults:
        - OpenAI: gpt-5, gpt-5.1, gpt-4o
        - Anthropic: claude-sonnet-4.5, claude-opus-4
        - Default config: provider=openai, model=gpt-5.1, system_prompt_id=default, temperature=0.1
        
        Environment variable overrides (if present and valid):
        - APP_ALLOWED_MODELS_OPENAI: Comma-separated list of OpenAI models
        - APP_ALLOWED_MODELS_ANTHROPIC: Comma-separated list of Anthropic models
        """
        self._lock = threading.Lock()
        
        # Built-in safe defaults for allowed models
        self._allowed_models: Dict[str, List[str]] = {
            "openai": ["gpt-5", "gpt-5.1", "gpt-4o"],
            "anthropic": ["claude-sonnet-4.5", "claude-opus-4"],
        }
        
        # Optionally seed from environment variables (deployment flexibility)
        # This is safe during __init__ as the instance is not yet shared
        self._seed_allowed_models_from_env()
        
        # Built-in default configuration
        self._default_config = ClarificationConfig(
            provider="openai",
            model="gpt-5.1",
            system_prompt_id="default",
            temperature=0.1,
            max_tokens=None
        )
    
    def _seed_allowed_models_from_env(self) -> None:
        """Seed allowed models from environment variables if present.
        
        Reads APP_ALLOWED_MODELS_OPENAI and APP_ALLOWED_MODELS_ANTHROPIC
        environment variables (comma-separated model lists) and updates
        allowed_models if valid. Falls back to built-in defaults on error.
        
        This allows deployments to customize allowed models without code changes
        while maintaining safe defaults if environment config is invalid.
        """
        # Try to read OpenAI models from environment
        openai_models_str = os.environ.get("APP_ALLOWED_MODELS_OPENAI", "").strip()
        if openai_models_str:
            models = [m.strip() for m in openai_models_str.split(",") if m.strip()]
            if models:  # Only override if we got at least one model
                self._allowed_models["openai"] = models
        
        # Try to read Anthropic models from environment
        anthropic_models_str = os.environ.get("APP_ALLOWED_MODELS_ANTHROPIC", "").strip()
        if anthropic_models_str:
            models = [m.strip() for m in anthropic_models_str.split(",") if m.strip()]
            if models:  # Only override if we got at least one model
                self._allowed_models["anthropic"] = models
    
    @property
    def allowed_models(self) -> Dict[str, List[str]]:
        """Get allowed models dictionary (thread-safe read).
        
        Returns a copy to prevent external mutations from bypassing validation.
        
        Returns:
            Dictionary mapping provider names to lists of allowed model names
        """
        with self._lock:
            return dict(self._allowed_models)  # Return copy to prevent external mutations
    
    def get_default_config(self):
        """Get the default ClarificationConfig (thread-safe read).
        
        Returns a copy of the default config to prevent external mutations
        from bypassing validation.
        
        Returns:
            ClarificationConfig: Copy of the default configuration
        """
        with self._lock:
            # Return a copy via model_copy() to prevent external mutations
            return self._default_config.model_copy()
    
    def set_default_config(self, config) -> None:
        """Set the default ClarificationConfig with validation (thread-safe write).
        
        Validates that the config's provider and model are in allowed_models
        before accepting the mutation. This ensures consistency and prevents
        invalid configurations from becoming defaults.
        
        Args:
            config: New default ClarificationConfig to set
        
        Raises:
            ConfigValidationError: If provider is missing from allowed_models
            ConfigValidationError: If model is not in provider's allowed list
            TypeError: If config is not a ClarificationConfig instance
        """
        if not isinstance(config, ClarificationConfig):
            raise TypeError(
                f"config must be a ClarificationConfig instance, got {type(config).__name__}"
            )
        
        with self._lock:
            # Delegate validation to the shared validation function
            # Note: We need to access allowed_models within the lock for consistency
            allowed = dict(self._allowed_models)
            
            # Check provider exists
            if config.provider not in allowed:
                raise ConfigValidationError(
                    f"Provider '{config.provider}' is not in allowed_models. "
                    f"Available providers: {', '.join(sorted(allowed.keys()))}"
                )
            
            # Check provider has at least one allowed model
            allowed_for_provider = allowed[config.provider]
            if not allowed_for_provider:
                raise ConfigValidationError(
                    f"No allowed models configured for provider '{config.provider}'. "
                    "Cannot set default config with empty allowed model list."
                )
            
            # Check model is in provider's allowed list
            if config.model not in allowed_for_provider:
                raise ConfigValidationError(
                    f"Model '{config.model}' is not allowed for provider '{config.provider}'. "
                    f"Allowed models: {', '.join(allowed_for_provider)}"
                )
            
            # Store a copy to prevent external mutations
            self._default_config = config.model_copy()


# Module-level singleton instance
_global_defaults = GlobalDefaults()


def get_default_config():
    """Get the current default ClarificationConfig.
    
    This is a convenience function that delegates to the module-level
    GlobalDefaults singleton. It provides thread-safe read access to
    the default configuration.
    
    Returns:
        ClarificationConfig: Copy of the current default configuration
    
    Example:
        >>> config = get_default_config()
        >>> print(config.provider, config.model)
        openai gpt-5.1
    """
    return _global_defaults.get_default_config()


def set_default_config(config) -> None:
    """Set the default ClarificationConfig with validation.
    
    This is a convenience function that delegates to the module-level
    GlobalDefaults singleton. It provides thread-safe write access with
    validation to ensure provider/model membership constraints.
    
    Args:
        config: New default ClarificationConfig to set
    
    Raises:
        ConfigValidationError: If validation fails (invalid provider/model)
        TypeError: If config is not a ClarificationConfig instance
    
    Example:
        >>> from app.models.specs import ClarificationConfig
        >>> new_config = ClarificationConfig(
        ...     provider="anthropic",
        ...     model="claude-sonnet-4.5",
        ...     system_prompt_id="default"
        ... )
        >>> set_default_config(new_config)
    """
    _global_defaults.set_default_config(config)


def get_allowed_models() -> Dict[str, List[str]]:
    """Get the allowed models dictionary.
    
    This is a convenience function that returns the allowed models
    configuration from the module-level GlobalDefaults singleton.
    
    Returns:
        Dictionary mapping provider names to lists of allowed model names
    
    Example:
        >>> models = get_allowed_models()
        >>> print(models["openai"])
        ['gpt-5', 'gpt-5.1', 'gpt-4o']
    """
    return _global_defaults.allowed_models


def validate_provider_model(provider: str, model: str) -> None:
    """Validate that a provider/model combination is allowed.
    
    This utility function provides a single source of truth for validating
    provider/model combinations against the allowed_models configuration.
    It is designed to be called by other layers (API, service) to enforce
    consistency.
    
    Args:
        provider: LLM provider identifier (e.g., 'openai', 'anthropic')
        model: Model identifier (e.g., 'gpt-5.1', 'claude-sonnet-4.5')
    
    Raises:
        ConfigValidationError: If provider is not in allowed_models
        ConfigValidationError: If model is not in provider's allowed list
        ConfigValidationError: If allowed model list is empty for provider
    
    Example:
        >>> validate_provider_model("openai", "gpt-5.1")  # OK
        >>> validate_provider_model("openai", "invalid-model")  # Raises ConfigValidationError
        >>> validate_provider_model("unknown", "any-model")  # Raises ConfigValidationError
    """
    allowed = _global_defaults.allowed_models
    
    # Check provider exists
    if provider not in allowed:
        raise ConfigValidationError(
            f"Unsupported provider '{provider}'. "
            f"Allowed providers: {', '.join(sorted(allowed.keys()))}"
        )
    
    # Check provider has at least one allowed model
    allowed_for_provider = allowed[provider]
    if not allowed_for_provider:
        raise ConfigValidationError(
            f"Provider '{provider}' has no allowed models configured. "
            "Cannot validate model."
        )
    
    # Check model is in provider's allowed list
    if model not in allowed_for_provider:
        raise ConfigValidationError(
            f"Model '{model}' is not allowed for provider '{provider}'. "
            f"Allowed models: {', '.join(allowed_for_provider)}"
        )


def validate_and_merge_config(
    request_config: Optional[ClarificationConfig],
) -> ClarificationConfig:
    """Validate and merge request config with defaults.
    
    Takes an optional per-request ClarificationConfig and merges it with the
    global default config. Request-provided fields override defaults, while
    missing fields inherit from the default. The merged config is validated
    to ensure provider/model membership constraints are satisfied.
    
    This function implements defensive copying to prevent mutation of the
    process-wide default config.
    
    Args:
        request_config: Optional config from request. If None, returns default config.
    
    Returns:
        ClarificationConfig: Validated merged config with request overrides applied
    
    Raises:
        ConfigValidationError: If provider/model combination is invalid
        TypeError: If request_config is not a ClarificationConfig or None
    
    Example:
        >>> # No override - returns default
        >>> config = validate_and_merge_config(None)
        >>> 
        >>> # Partial override - model only
        >>> request = ClarificationConfig(model="gpt-4o")
        >>> config = validate_and_merge_config(request)
        >>> 
        >>> # Full override
        >>> request = ClarificationConfig(
        ...     provider="anthropic",
        ...     model="claude-sonnet-4.5",
        ...     system_prompt_id="custom",
        ...     temperature=0.2,
        ...     max_tokens=3000
        ... )
        >>> config = validate_and_merge_config(request)
    """
    # If no request config, return a copy of the default
    if request_config is None:
        return get_default_config()
    
    # Validate type
    if not isinstance(request_config, ClarificationConfig):
        raise TypeError(
            f"request_config must be a ClarificationConfig instance or None, "
            f"got {type(request_config).__name__}"
        )
    
    # Get default config (this returns a copy, so it's safe)
    default = get_default_config()
    
    # Merge: request fields override defaults, None fields inherit from default
    # Build a dict with merged values
    merged_values = {}
    
    # Merge each field: use request value if not None, otherwise use default
    merged_values['provider'] = request_config.provider if request_config.provider is not None else default.provider
    merged_values['model'] = request_config.model if request_config.model is not None else default.model
    merged_values['system_prompt_id'] = request_config.system_prompt_id if request_config.system_prompt_id is not None else default.system_prompt_id
    merged_values['temperature'] = request_config.temperature if request_config.temperature is not None else default.temperature
    merged_values['max_tokens'] = request_config.max_tokens if request_config.max_tokens is not None else default.max_tokens
    
    # Create merged config
    merged = ClarificationConfig(**merged_values)
    
    # Validate the merged config's provider/model combination
    validate_provider_model(merged.provider, merged.model)
    
    return merged


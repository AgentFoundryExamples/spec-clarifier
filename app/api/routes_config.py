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
"""Admin configuration endpoints for managing global defaults."""

import logging
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    get_settings,
    get_default_config,
    set_default_config,
    get_allowed_models,
    ConfigValidationError,
)
from app.models.config_models import ClarificationConfig
from app.utils.logging_helper import log_info, log_warning, log_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/config", tags=["Configuration"])


class DefaultsResponse(BaseModel):
    """Response model for GET /v1/config/defaults.
    
    Contains the current default configuration and allowed models
    for all providers.
    """
    default_config: ClarificationConfig = Field(
        description="Current default ClarificationConfig used when no config is provided"
    )
    allowed_models: Dict[str, List[str]] = Field(
        description="Dictionary mapping provider names to lists of allowed model names"
    )


def _check_config_admin_enabled():
    """Check if config admin endpoints are enabled.
    
    Raises:
        HTTPException: 403 if endpoints are disabled
    """
    settings = get_settings()
    
    if not settings.enable_config_admin_endpoints:
        log_warning(logger, "config_admin_endpoint_disabled_access_attempt")
        raise HTTPException(
            status_code=403,
            detail="Config admin endpoints are disabled. Set APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=true to enable."
        )


@router.get(
    "/defaults",
    response_model=DefaultsResponse,
    summary="Get current default configuration (admin-only)",
    description=(
        "⚠️ ADMIN-ONLY ENDPOINT - USE ONLY IN TRUSTED ENVIRONMENTS\n\n"
        "Returns the current global default ClarificationConfig and allowed models "
        "for all providers. This endpoint exposes internal configuration state and "
        "should only be accessible in trusted environments with proper network-level "
        "access controls.\n\n"
        "The returned default_config contains:\n"
        "- provider: LLM provider (e.g., 'openai', 'anthropic')\n"
        "- model: Model identifier (e.g., 'gpt-5.1', 'claude-sonnet-4.5')\n"
        "- system_prompt_id: System prompt template identifier\n"
        "- temperature: Sampling temperature for response generation\n"
        "- max_tokens: Optional maximum tokens to generate\n\n"
        "The allowed_models dictionary maps provider names to lists of allowed "
        "model names. Only provider/model combinations in this dictionary can "
        "be set as defaults or used in clarification requests.\n\n"
        "This endpoint can be disabled by setting APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=false, "
        "which will return 403 Forbidden."
    ),
    responses={
        200: {
            "description": "Current default configuration retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "default_config": {
                            "provider": "openai",
                            "model": "gpt-5.1",
                            "system_prompt_id": "default",
                            "temperature": 0.1,
                            "max_tokens": None
                        },
                        "allowed_models": {
                            "openai": ["gpt-5", "gpt-5.1", "gpt-4o"],
                            "anthropic": ["claude-sonnet-4.5", "claude-opus-4"]
                        }
                    }
                }
            }
        },
        403: {
            "description": "Config admin endpoints are disabled",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Config admin endpoints are disabled. Set APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=true to enable."
                    }
                }
            }
        }
    },
)
def get_defaults() -> DefaultsResponse:
    """Get current default configuration and allowed models.
    
    This admin-only endpoint returns the current global defaults for
    clarification configuration. It should only be accessible in trusted
    environments with proper access controls.
    
    Returns:
        DefaultsResponse: Current defaults and allowed models
        
    Raises:
        HTTPException: 403 if config admin endpoints are disabled
    """
    _check_config_admin_enabled()
    
    log_info(logger, "config_admin_get_defaults_accessed")
    
    return DefaultsResponse(
        default_config=get_default_config(),
        allowed_models=get_allowed_models(),
    )


@router.put(
    "/defaults",
    response_model=DefaultsResponse,
    summary="Update default configuration (admin-only)",
    description=(
        "⚠️ ADMIN-ONLY ENDPOINT - USE ONLY IN TRUSTED ENVIRONMENTS\n\n"
        "Updates the global default ClarificationConfig. This change affects all "
        "subsequent clarification requests that don't provide explicit configuration "
        "overrides. The endpoint does NOT persist changes across restarts - defaults "
        "reset to initial values (from environment or built-ins) when the service restarts.\n\n"
        "The provided config is validated before being set as the new default:\n"
        "- Provider must be in allowed_models (e.g., 'openai', 'anthropic')\n"
        "- Model must be in the provider's allowed model list\n"
        "- System prompt ID is accepted but not validated (unknown IDs fall back to 'default' at runtime)\n"
        "- Temperature must be 0.0-2.0\n"
        "- Max tokens must be positive if provided\n\n"
        "All fields are required in the request. To see current defaults, use GET /v1/config/defaults.\n\n"
        "Invalid provider/model combinations return 400 Bad Request with actionable error messages.\n\n"
        "This endpoint can be disabled by setting APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=false, "
        "which will return 403 Forbidden.\n\n"
        "**Thread Safety:** Updates are atomic and protected by a lock to handle concurrent "
        "PUT requests safely."
    ),
    responses={
        200: {
            "description": "Configuration updated successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "openai": {
                            "summary": "Update to OpenAI GPT-5",
                            "value": {
                                "default_config": {
                                    "provider": "openai",
                                    "model": "gpt-5",
                                    "system_prompt_id": "default",
                                    "temperature": 0.1,
                                    "max_tokens": 2000
                                },
                                "allowed_models": {
                                    "openai": ["gpt-5", "gpt-5.1", "gpt-4o"],
                                    "anthropic": ["claude-sonnet-4.5", "claude-opus-4"]
                                }
                            }
                        },
                        "anthropic": {
                            "summary": "Update to Anthropic Claude",
                            "value": {
                                "default_config": {
                                    "provider": "anthropic",
                                    "model": "claude-sonnet-4.5",
                                    "system_prompt_id": "strict_json",
                                    "temperature": 0.2,
                                    "max_tokens": 3000
                                },
                                "allowed_models": {
                                    "openai": ["gpt-5", "gpt-5.1", "gpt-4o"],
                                    "anthropic": ["claude-sonnet-4.5", "claude-opus-4"]
                                }
                            }
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid configuration (validation failed)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model 'gpt-3.5-turbo' is not allowed for provider 'openai'. Allowed models: gpt-5, gpt-5.1, gpt-4o"
                    }
                }
            }
        },
        403: {
            "description": "Config admin endpoints are disabled",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Config admin endpoints are disabled. Set APP_ENABLE_CONFIG_ADMIN_ENDPOINTS=true to enable."
                    }
                }
            }
        },
        422: {
            "description": "Invalid request payload (wrong types or out of range values)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "literal_error",
                                "loc": ["body", "provider"],
                                "msg": "Input should be 'openai' or 'anthropic'"
                            }
                        ]
                    }
                }
            }
        }
    },
)
def update_defaults(config: ClarificationConfig) -> DefaultsResponse:
    """Update the global default configuration.
    
    This admin-only endpoint validates and sets a new default configuration.
    Changes are atomic and thread-safe but NOT persisted across restarts.
    
    All fields in the config are required. The config is validated to ensure
    the provider/model combination is in allowed_models.
    
    Args:
        config: New default ClarificationConfig to set
        
    Returns:
        DefaultsResponse: Updated defaults and allowed models
        
    Raises:
        HTTPException: 400 if validation fails (invalid provider/model)
        HTTPException: 403 if config admin endpoints are disabled
    """
    _check_config_admin_enabled()
    
    log_warning(
        logger,
        "config_admin_update_defaults_accessed",
        provider=config.provider,
        model=config.model
    )
    
    try:
        # Validate and set the new default
        # This will raise ConfigValidationError if provider/model is invalid
        set_default_config(config)
        
        log_info(
            logger,
            "config_admin_defaults_updated",
            provider=config.provider,
            model=config.model,
            system_prompt_id=config.system_prompt_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Return the config that was just set to avoid race conditions
        return DefaultsResponse(
            default_config=config,
            allowed_models=get_allowed_models(),
        )
        
    except ConfigValidationError as e:
        log_warning(logger, "config_validation_failed", error_message=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        log_error(logger, "config_type_error", error=e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_error(logger, "config_update_unexpected_error", error=e)
        raise HTTPException(status_code=500, detail="Internal server error")

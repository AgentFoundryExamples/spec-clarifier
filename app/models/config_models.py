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
"""Configuration models for specification clarification.

This module contains configuration-related Pydantic models that are shared
between the config and specs modules to avoid circular dependencies.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClarificationConfig(BaseModel):
    """Configuration for clarification LLM calls.
    
    This model defines the runtime configuration for LLM-based specification
    clarification, including provider selection, model specification, system
    prompt identification, and generation parameters.
    
    The config is designed to be:
    - Validated at creation time (provider/model membership checks)
    - Serializable for storage in job configs or API responses
    - Reusable across API and service layers
    - Compatible with global defaults and runtime overrides
    
    All fields are optional to support partial config overrides. When used
    in a request, missing fields will be filled from the default config
    during validation/merging.
    
    Attributes:
        provider: LLM provider identifier (must be 'openai' or 'anthropic')
        model: Model identifier specific to the provider (e.g., 'gpt-5.1', 'claude-sonnet-4.5')
        system_prompt_id: Identifier for the system prompt template to use
        temperature: Sampling temperature for response generation (0.0-2.0), defaults to 0.1
        max_tokens: Optional maximum tokens to generate in response
    
    Example:
        >>> # Full config
        >>> config = ClarificationConfig(
        ...     provider="openai",
        ...     model="gpt-5.1",
        ...     system_prompt_id="default",
        ...     temperature=0.1,
        ...     max_tokens=2000
        ... )
        >>> 
        >>> # Partial config (only override model)
        >>> config = ClarificationConfig(model="gpt-4o")
    """
    
    model_config = ConfigDict(extra="forbid")
    
    provider: Optional[Literal["openai", "anthropic"]] = Field(
        default=None,
        description="LLM provider identifier (must be 'openai' or 'anthropic')"
    )
    model: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Model identifier specific to the provider"
    )
    system_prompt_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Identifier for the system prompt template to use"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate in response"
    )

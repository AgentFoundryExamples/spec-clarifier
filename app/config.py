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

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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

"""Application configuration."""

import os
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


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()

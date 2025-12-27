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
"""FastAPI application factory and entrypoint."""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api import routes_health, routes_clarifications, routes_config
from app.utils.logging_helper import log_error, get_correlation_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = get_settings()
    
    # Use hardcoded metadata per OpenAPI documentation requirements
    # to ensure consistent, predictable service identification across
    # all deployment environments. This makes the service easily
    # discoverable and identifiable in API catalogs and documentation.
    app = FastAPI(
        title="Agent Foundry Clarification Service",
        version="0.1.0",
        description=(
            "A service for asynchronously clarifying specifications using LLM-powered processing. "
            "Submit specifications with open questions, receive clarified specifications with "
            "questions resolved and integrated into requirements, assumptions, and constraints."
        ),
    )
    
    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins_list(),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.get_cors_methods_list(),
        allow_headers=settings.get_cors_headers_list(),
    )
    
    # Register exception handlers only in non-debug mode
    if not settings.debug:
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
            """Handle uncaught exceptions globally with sanitized responses and structured logging."""
            # Generate correlation ID for tracing
            correlation_id = get_correlation_id()
            
            # Log the exception with full details including correlation ID and traceback
            log_error(
                logger,
                "unhandled_exception",
                error=exc,
                path=request.url.path,
                method=request.method,
                correlation_id=correlation_id
            )
            
            # Return sanitized error response (no stack trace)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "correlation_id": correlation_id
                },
            )
    
    # Register routers
    app.include_router(routes_health.router)
    app.include_router(routes_clarifications.router)
    app.include_router(routes_config.router)
    
    return app


# Create app instance for uvicorn
app = create_app()

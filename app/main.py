"""FastAPI application factory and entrypoint."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api import routes_health


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
    )
    
    # Register exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions globally."""
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    
    # Register routers
    app.include_router(routes_health.router)
    
    return app


# Create app instance for uvicorn
app = create_app()

"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """Health check endpoint.
    
    Returns:
        dict: Health status response
    """
    return {"status": "ok"}

"""Clarification endpoints."""

from fastapi import APIRouter

from app.models.specs import ClarificationRequest, ClarifiedPlan
from app.services.clarification import clarify_plan

router = APIRouter(prefix="/v1/clarifications", tags=["clarifications"])


@router.post(
    "/preview",
    response_model=ClarifiedPlan,
    summary="Preview clarified specifications",
    description=(
        "Accepts a ClarificationRequest containing specifications with open questions "
        "and returns a ClarifiedPlan with the questions removed. This endpoint provides "
        "a synchronous preview of the clarification process without processing answers "
        "or triggering async/LLM operations."
    ),
)
def preview_clarifications(request: ClarificationRequest) -> ClarifiedPlan:
    """Preview clarified specifications from a plan.
    
    This endpoint transforms specifications by copying the required fields
    (purpose, vision, must, dont, nice, assumptions) while omitting open_questions.
    Answers in the request are currently ignored.
    
    FastAPI automatically validates the request body and returns 422 Unprocessable
    Entity responses for malformed payloads or missing required fields.
    
    Args:
        request: The clarification request containing the plan and optional answers
        
    Returns:
        ClarifiedPlan: The clarified plan with specifications ready for use
    """
    return clarify_plan(request.plan)

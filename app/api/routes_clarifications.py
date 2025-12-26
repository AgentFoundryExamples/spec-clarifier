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
"""Clarification endpoints."""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import get_settings
from app.models.specs import (
    ClarificationRequest,
    ClarifiedPlan,
    JobStatusResponse,
    JobSummaryResponse,
)
from app.services.clarification import clarify_plan, start_clarification_job
from app.services.job_store import JobNotFoundError, get_job

router = APIRouter(prefix="/v1/clarifications", tags=["Clarifications"])


@router.post(
    "/preview",
    response_model=ClarifiedPlan,
    summary="Preview clarified specifications (synchronous, developer-only)",
    description=(
        "⚠️ DEVELOPER-ONLY ENDPOINT - NOT FOR PRODUCTION USE\n\n"
        "This endpoint provides a synchronous preview of the clarification process "
        "for development and debugging purposes only. It returns immediately with the "
        "clarified specifications without async processing or LLM operations.\n\n"
        "For production use cases, use POST /v1/clarifications to create an async job "
        "and poll GET /v1/clarifications/{job_id} for results.\n\n"
        "Accepts a ClarificationRequest containing specifications with open questions "
        "and returns a ClarifiedPlan with the questions removed. Answers in the request "
        "are currently ignored."
    ),
)
def preview_clarifications(request: ClarificationRequest) -> ClarifiedPlan:
    """Preview clarified specifications from a plan.
    
    ⚠️ DEVELOPER-ONLY - This is a synchronous endpoint for development/debugging.
    For production, use the async POST /v1/clarifications endpoint instead.
    
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


@router.post(
    "",
    response_model=JobSummaryResponse,
    status_code=202,
    summary="Start async clarification job",
    description=(
        "Creates an asynchronous clarification job and returns immediately with "
        "lightweight job details (id, status, timestamps). The job is initially in "
        "PENDING status and will be processed in the background.\n\n"
        "Returns HTTP 202 Accepted with minimal job metadata. The response does NOT "
        "include the full request payload or result to keep responses lightweight.\n\n"
        "Use GET /v1/clarifications/{job_id} to poll for job status and retrieve "
        "results when complete."
    ),
)
def create_clarification_job(
    request: ClarificationRequest,
    background_tasks: BackgroundTasks
) -> JobSummaryResponse:
    """Create and start an asynchronous clarification job.
    
    This endpoint creates a new clarification job with PENDING status,
    schedules it for background processing, and returns immediately with
    a lightweight summary (no request/result payloads).
    
    The job will transition through RUNNING to either SUCCESS or FAILED status.
    
    Args:
        request: The clarification request containing the plan and optional answers
        background_tasks: FastAPI BackgroundTasks for async processing
        
    Returns:
        JobSummaryResponse: Lightweight job summary with id, status, and timestamps
    """
    job = start_clarification_job(request, background_tasks)
    
    # Return lightweight summary without request/result payloads
    return JobSummaryResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        last_error=job.last_error,
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get clarification job status",
    description=(
        "Retrieves the current status and details of a clarification job by its ID. "
        "Use this endpoint to poll for job completion after creating an async job.\n\n"
        "When status is SUCCESS, the result field will contain the clarified plan "
        "(only if APP_SHOW_JOB_RESULT development flag is enabled).\n\n"
        "When status is FAILED, the last_error field will contain the error message.\n\n"
        "Returns 404 if the job ID is not found. Invalid UUIDs return 422 validation errors."
    ),
)
def get_clarification_job(job_id: UUID) -> JobStatusResponse:
    """Get the status and details of a clarification job.
    
    Returns job metadata and conditionally includes the result field based on
    the APP_SHOW_JOB_RESULT development flag. In production mode (flag=False),
    the result is excluded to prevent clients from depending on embedded results.
    
    Args:
        job_id: The UUID of the job to retrieve
        
    Returns:
        JobStatusResponse: Job status with conditional result field
        
    Raises:
        HTTPException: 404 if job not found, 422 if UUID is malformed
    """
    try:
        job = get_job(job_id)
        settings = get_settings()
        
        # Conditionally include result based on development flag and job status
        # Only expose result when flag is enabled AND job completed successfully
        from app.models.specs import JobStatus
        result = job.result if (settings.show_job_result and job.status == JobStatus.SUCCESS) else None
        
        return JobStatusResponse(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            last_error=job.last_error,
            result=result,
        )
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

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

from app.models.specs import ClarificationJob, ClarificationRequest, ClarifiedPlan
from app.services.clarification import clarify_plan, start_clarification_job
from app.services.job_store import JobNotFoundError, get_job

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


@router.post(
    "",
    response_model=ClarificationJob,
    status_code=202,
    summary="Start async clarification job",
    description=(
        "Accepts a ClarificationRequest and creates an asynchronous clarification job. "
        "The job is initially in PENDING status and will be processed in the background. "
        "Returns immediately with the job details. Use GET /v1/clarifications/{job_id} "
        "to poll for job status and retrieve results when complete."
    ),
)
def create_clarification_job(
    request: ClarificationRequest,
    background_tasks: BackgroundTasks
) -> ClarificationJob:
    """Create and start an asynchronous clarification job.
    
    This endpoint creates a new clarification job with PENDING status,
    schedules it for background processing, and returns immediately
    without blocking. The job will transition through RUNNING to either
    SUCCESS or FAILED status.
    
    Args:
        request: The clarification request containing the plan and optional answers
        background_tasks: FastAPI BackgroundTasks for async processing
        
    Returns:
        ClarificationJob: The newly created job with PENDING status
    """
    return start_clarification_job(request, background_tasks)


@router.get(
    "/{job_id}",
    response_model=ClarificationJob,
    summary="Get clarification job status",
    description=(
        "Retrieves the current status and details of a clarification job by its ID. "
        "Use this endpoint to poll for job completion after creating an async job. "
        "When status is SUCCESS, the result field will contain the clarified plan. "
        "When status is FAILED, the last_error field will contain the error message."
    ),
)
def get_clarification_job(job_id: UUID) -> ClarificationJob:
    """Get the status and details of a clarification job.
    
    Args:
        job_id: The UUID of the job to retrieve
        
    Returns:
        ClarificationJob: The job with current status and results (if complete)
        
    Raises:
        HTTPException: 404 if job not found
    """
    try:
        return get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

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

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import ConfigValidationError, get_settings, validate_and_merge_config
from app.models.specs import (
    ClarificationRequest,
    ClarificationRequestWithConfig,
    ClarifiedPlan,
    JobStatusResponse,
    JobSummaryResponse,
)
from app.services.clarification import clarify_plan, start_clarification_job
from app.services.job_store import JobNotFoundError, get_job
from app.utils.logging_helper import log_info, log_warning

logger = logging.getLogger(__name__)

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
        "**ASYNCHRONOUS PROCESSING - Returns immediately with job ID**\n\n"
        "Creates an asynchronous clarification job and returns immediately with "
        "lightweight job details (id, status, timestamps). The job is initially in "
        "PENDING status and will be processed in the background.\n\n"
        "**Important:** This endpoint returns a job_id, not the clarified plan. "
        "Processing happens asynchronously. Poll GET /v1/clarifications/{job_id} "
        "to check status and retrieve results when complete.\n\n"
        "Accepts an optional 'config' field to override default LLM configuration "
        "(provider, model, system_prompt_id, temperature, max_tokens). Config is "
        "validated and merged with defaults before processing. Invalid provider/model "
        "combinations return 400 Bad Request.\n\n"
        "Returns HTTP 202 Accepted with minimal job metadata. The response does NOT "
        "include the full request payload or result to keep responses lightweight.\n\n"
        "**Async Workflow:**\n"
        "1. POST /v1/clarifications → receive job_id\n"
        "2. Poll GET /v1/clarifications/{job_id} → check status (PENDING/RUNNING/SUCCESS/FAILED)\n"
        "3. When status is SUCCESS → processing complete"
    ),
    responses={
        202: {
            "description": "Job created successfully and queued for processing",
            "content": {
                "application/json": {
                    "example": {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "PENDING",
                        "created_at": "2025-12-27T03:00:00.000000Z",
                        "updated_at": "2025-12-27T03:00:00.000000Z",
                        "last_error": None,
                    }
                }
            },
        },
        400: {
            "description": "Invalid configuration (provider/model combination not allowed)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model 'invalid-model' is not allowed for provider 'openai'. Allowed models: gpt-5, gpt-5.1, gpt-4o"
                    }
                }
            },
        },
        422: {
            "description": "Invalid request payload (missing required fields or wrong types)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "missing",
                                "loc": ["body", "plan", "specs", 0, "vision"],
                                "msg": "Field required",
                            }
                        ]
                    }
                }
            },
        },
    },
)
def create_clarification_job(
    request: ClarificationRequestWithConfig, background_tasks: BackgroundTasks
) -> JobSummaryResponse:
    """Create and start an asynchronous clarification job.

    This endpoint creates a new clarification job with PENDING status,
    schedules it for background processing, and returns immediately with
    a lightweight summary (no request/result payloads).

    The job will transition through RUNNING to either SUCCESS or FAILED status.

    Accepts optional per-request config to override defaults. Config is validated
    and merged with global defaults (request fields override, missing fields inherit).
    Invalid provider/model combinations return 400 Bad Request.

    Args:
        request: The clarification request with plan, answers, and optional config
        background_tasks: FastAPI BackgroundTasks for async processing

    Returns:
        JobSummaryResponse: Lightweight job summary with id, status, and timestamps

    Raises:
        HTTPException: 400 if config validation fails (invalid provider/model)
    """
    # Validate and merge config with defaults
    try:
        merged_config = validate_and_merge_config(request.config)
    except ConfigValidationError as e:
        log_warning(
            logger,
            "config_validation_failed",
            error_message=str(e),
            provided_config=request.config.model_dump() if request.config else None,
        )
        raise HTTPException(status_code=400, detail=str(e))

    # Log if non-default config is used
    if request.config is not None:
        # Log only the fields that were actually provided in the request
        overridden_fields = {k: v for k, v in request.config.model_dump().items() if v is not None}
        log_info(logger, "job_created_with_config", overridden_fields=overridden_fields)

    # Convert to ClarificationRequest for service layer
    clarification_request = ClarificationRequest(plan=request.plan, answers=request.answers)

    # Start job with merged config
    job = start_clarification_job(
        clarification_request,
        background_tasks,
        config={"clarification_config": merged_config.model_dump()},
    )

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
        "**Important:** This endpoint returns only job metadata and status. The result "
        "field is null by default (production mode) to keep responses lightweight. "
        "To view results during development, set APP_SHOW_JOB_RESULT=true.\n\n"
        "**Status Values:**\n"
        "- PENDING: Job queued, not yet started\n"
        "- RUNNING: Job actively processing\n"
        "- SUCCESS: Job completed successfully\n"
        "- FAILED: Job encountered an error (see last_error field)\n\n"
        "When status is FAILED, the last_error field will contain the error message.\n\n"
        "Returns 404 if the job ID is not found. Invalid UUIDs return 422 validation errors."
    ),
    responses={
        200: {
            "description": "Job status retrieved successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "pending": {
                            "summary": "Job pending (queued, not yet started)",
                            "value": {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "status": "PENDING",
                                "created_at": "2025-12-27T03:00:00.000000Z",
                                "updated_at": "2025-12-27T03:00:00.000000Z",
                                "last_error": None,
                                "result": None,
                            },
                        },
                        "running": {
                            "summary": "Job running (actively processing)",
                            "value": {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "status": "RUNNING",
                                "created_at": "2025-12-27T03:00:00.000000Z",
                                "updated_at": "2025-12-27T03:00:01.500000Z",
                                "last_error": None,
                                "result": None,
                            },
                        },
                        "success": {
                            "summary": "Job completed successfully (production mode - result is null)",
                            "value": {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "status": "SUCCESS",
                                "created_at": "2025-12-27T03:00:00.000000Z",
                                "updated_at": "2025-12-27T03:00:03.250000Z",
                                "last_error": None,
                                "result": None,
                            },
                        },
                        "failed": {
                            "summary": "Job failed (with error message)",
                            "value": {
                                "id": "550e8400-e29b-41d4-a716-446655440000",
                                "status": "FAILED",
                                "created_at": "2025-12-27T03:00:00.000000Z",
                                "updated_at": "2025-12-27T03:00:02.100000Z",
                                "last_error": "ValueError: Invalid specification format",
                                "result": None,
                            },
                        },
                    }
                }
            },
        },
        404: {
            "description": "Job not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"}
                }
            },
        },
        422: {
            "description": "Invalid UUID format",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "uuid_parsing",
                                "loc": ["path", "job_id"],
                                "msg": "Input should be a valid UUID",
                            }
                        ]
                    }
                }
            },
        },
    },
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

        result = (
            job.result if (settings.show_job_result and job.status == JobStatus.SUCCESS) else None
        )

        return JobStatusResponse(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            last_error=job.last_error,
            result=result,
        )
    except JobNotFoundError:
        log_warning(logger, "job_not_found", job_id=job_id)
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.get(
    "/{job_id}/debug",
    response_model=dict,
    summary="Get debug information for a clarification job (debug mode only)",
    description=(
        "⚠️ DEBUG ENDPOINT - DISABLED BY DEFAULT\n\n"
        "Returns detailed debug information for a clarification job including configuration, "
        "timestamps, and metadata. This endpoint is only available when APP_ENABLE_DEBUG_ENDPOINT "
        "is set to true.\n\n"
        "This endpoint intentionally excludes raw prompts and LLM responses to prevent "
        "accidental exposure of sensitive data. Only metadata and sanitized information "
        "is returned.\n\n"
        "Returns 403 Forbidden if the debug endpoint is not enabled.\n"
        "Returns 404 if the job ID is not found."
    ),
)
def get_clarification_job_debug(job_id: UUID) -> dict:
    """Get debug information for a clarification job.

    This endpoint is protected by the APP_ENABLE_DEBUG_ENDPOINT flag and returns
    sanitized debug information about a job. Raw prompts and LLM responses are
    intentionally excluded to prevent accidental data leakage.

    Args:
        job_id: The UUID of the job to retrieve

    Returns:
        dict: Debug information including job metadata, config, and sanitized details

    Raises:
        HTTPException: 403 if debug endpoint disabled, 404 if job not found
    """
    settings = get_settings()

    # Check if debug endpoint is enabled
    if not settings.enable_debug_endpoint:
        raise HTTPException(
            status_code=403,
            detail="Debug endpoint is disabled. Set APP_ENABLE_DEBUG_ENDPOINT=true to enable.",
        )

    try:
        job = get_job(job_id)

        # Sanitize config to remove sensitive data (API keys, tokens, etc.)
        sanitized_config = None
        if job.config:
            sanitized_config = {}
            for key, value in job.config.items():
                # Only include safe config keys, exclude anything that might contain credentials
                if key == "clarification_config" and isinstance(value, dict):
                    # For clarification_config, only expose non-sensitive fields
                    clarification_config_safe = {
                        "provider": value.get("provider"),
                        "model": value.get("model"),
                        "system_prompt_id": value.get("system_prompt_id"),
                        "temperature": value.get("temperature"),
                        "max_tokens": value.get("max_tokens"),
                    }
                    # Filter out None values for cleaner output
                    sanitized_config["clarification_config"] = {
                        k: v for k, v in clarification_config_safe.items() if v is not None
                    }
                elif key == "llm_config" and isinstance(value, dict):
                    # For llm_config (legacy), only expose non-sensitive fields
                    llm_config_safe = {
                        "provider": value.get("provider"),
                        "model": value.get("model"),
                        "temperature": value.get("temperature"),
                        "max_tokens": value.get("max_tokens"),
                    }
                    # Filter out None values for cleaner output
                    sanitized_config["llm_config"] = {
                        k: v for k, v in llm_config_safe.items() if v is not None
                    }
                elif key not in ["api_key", "token", "secret", "password", "credential", "auth"]:
                    # For other keys, only include if they don't contain sensitive keywords
                    if not any(
                        sensitive in key.lower()
                        for sensitive in [
                            "key",
                            "token",
                            "secret",
                            "password",
                            "credential",
                            "auth",
                        ]
                    ):
                        sanitized_config[key] = value

        # Sanitize error message to remove potential sensitive information
        sanitized_error = None
        if job.last_error:
            # Use the same sanitization as LLMCallError for consistency
            from app.services.llm_clients import LLMCallError

            sanitized_error = LLMCallError._sanitize_message(job.last_error)

        # Build sanitized debug response
        debug_info = {
            "job_id": str(job.id),
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "last_error": sanitized_error,
            "has_request": job.request is not None,
            "has_result": job.result is not None,
            "config": sanitized_config,
        }

        # Add request metadata (not full content)
        if job.request:
            debug_info["request_metadata"] = {
                "num_specs": len(job.request.plan.specs),
                "num_answers": len(job.request.answers),
                "spec_summaries": [
                    {
                        "purpose_length": len(spec.purpose),
                        "vision_length": len(spec.vision),
                        "num_must": len(spec.must),
                        "num_dont": len(spec.dont),
                        "num_nice": len(spec.nice),
                        "num_open_questions": len(spec.open_questions),
                        "num_assumptions": len(spec.assumptions),
                    }
                    for spec in job.request.plan.specs
                ],
            }

        # Add result metadata (not full content)
        if job.result:
            debug_info["result_metadata"] = {
                "num_specs": len(job.result.specs),
                "spec_summaries": [
                    {
                        "purpose_length": len(spec.purpose),
                        "vision_length": len(spec.vision),
                        "num_must": len(spec.must),
                        "num_dont": len(spec.dont),
                        "num_nice": len(spec.nice),
                        "num_open_questions": 0,  # ClarifiedSpec never has open_questions
                        "num_assumptions": len(spec.assumptions),
                    }
                    for spec in job.result.specs
                ],
            }

        return debug_info

    except JobNotFoundError:
        log_warning(logger, "job_not_found_debug", job_id=job_id)
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

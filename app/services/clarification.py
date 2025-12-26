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
"""Service for clarifying specifications."""

import logging
from typing import Optional
from uuid import UUID

from fastapi import BackgroundTasks

from app.models.specs import ClarificationJob, ClarificationRequest, ClarifiedPlan, ClarifiedSpec, JobStatus, PlanInput
from app.services import job_store

logger = logging.getLogger(__name__)


def clarify_plan(plan_input: PlanInput) -> ClarifiedPlan:
    """Transform a PlanInput to ClarifiedPlan by copying required fields.
    
    This function processes each specification in the plan, copying the required
    fields (purpose, vision, must, dont, nice, assumptions) while omitting
    open_questions. Currently, answers are ignored as per the requirements.
    
    Args:
        plan_input: The input plan containing specifications with potential questions
        
    Returns:
        ClarifiedPlan: A plan with clarified specifications (no open_questions)
    """
    clarified_specs = []
    
    for spec_input in plan_input.specs:
        clarified_spec = ClarifiedSpec(
            purpose=spec_input.purpose,
            vision=spec_input.vision,
            must=spec_input.must,
            dont=spec_input.dont,
            nice=spec_input.nice,
            assumptions=spec_input.assumptions,
        )
        clarified_specs.append(clarified_spec)
    
    return ClarifiedPlan(specs=clarified_specs)


def start_clarification_job(
    request: ClarificationRequest,
    background_tasks: BackgroundTasks,
    config: Optional[dict] = None
) -> ClarificationJob:
    """Start a clarification job asynchronously.
    
    Creates a new job with PENDING status, stores it in the job store,
    and schedules background processing via FastAPI BackgroundTasks.
    Returns immediately without blocking.
    
    Args:
        request: The clarification request to process
        background_tasks: FastAPI BackgroundTasks instance for scheduling
        config: Optional configuration dictionary for job processing
        
    Returns:
        ClarificationJob: The newly created job with PENDING status
    """
    # Create job in PENDING state
    job = job_store.create_job(request, config=config)
    
    # Schedule background processing
    background_tasks.add_task(process_clarification_job, job.id)
    
    logger.info(f"Started clarification job {job.id} with status PENDING")
    return job


def process_clarification_job(job_id: UUID) -> None:
    """Process a clarification job asynchronously.
    
    Loads the job from the store, marks it RUNNING, invokes the synchronous
    clarification logic, saves the result, and marks it SUCCESS. If any
    exception occurs during processing, marks the job FAILED and captures
    the error message.
    
    This function is designed to be called via FastAPI BackgroundTasks but
    can also be invoked directly for testing purposes.
    
    Args:
        job_id: The UUID of the job to process
    """
    try:
        # Load the job
        job = job_store.get_job(job_id)
        logger.info(f"Processing clarification job {job_id}")
        
        # Mark as RUNNING
        job_store.update_job(job_id, status=JobStatus.RUNNING)
        
        # Perform the clarification using existing synchronous logic
        result = clarify_plan(job.request.plan)
        
        # Mark as SUCCESS with result
        job_store.update_job(job_id, status=JobStatus.SUCCESS, result=result)
        logger.info(f"Clarification job {job_id} completed successfully")
        
    except job_store.JobNotFoundError:
        # Job doesn't exist - log and return cleanly without crashing
        logger.warning(f"Job {job_id} not found during processing - skipping")
        return
        
    except Exception as e:
        # Capture any exception and mark job as FAILED
        error_message = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Clarification job {job_id} failed: {error_message}", exc_info=True)
        
        try:
            # Try to update the job with error status
            # Clear result to ensure no partial data is stored
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                last_error=error_message,
                result=None
            )
        except job_store.JobNotFoundError:
            # Job was deleted while processing - log and continue
            logger.warning(f"Job {job_id} not found when trying to mark as FAILED")
        except Exception as update_error:
            # Failed to update job status - log but don't raise
            logger.error(
                f"Failed to update job {job_id} with error status: {update_error}",
                exc_info=True
            )

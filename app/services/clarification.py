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

import copy
import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import BackgroundTasks

from app.models.specs import ClarificationJob, ClarificationRequest, ClarifiedPlan, ClarifiedSpec, JobStatus, PlanInput
from app.services import job_store
from app.services.llm_clients import ClarificationLLMConfig, get_llm_client, LLMCallError

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
    config: Optional[dict] = None,
    llm_config: Optional[ClarificationLLMConfig] = None
) -> ClarificationJob:
    """Start a clarification job asynchronously.
    
    Creates a new job with PENDING status, stores it in the job store,
    and schedules background processing via FastAPI BackgroundTasks.
    Returns immediately without blocking.
    
    Args:
        request: The clarification request to process
        background_tasks: FastAPI BackgroundTasks instance for scheduling
        config: Optional configuration dictionary for job processing
        llm_config: Optional LLM configuration for future AI-powered clarification.
                   If provided, an LLM client will be initialized during processing
                   but not invoked in this iteration. Defaults to None to preserve
                   existing deterministic behavior.
        
    Returns:
        ClarificationJob: The newly created job with PENDING status
    """
    # Create job in PENDING state, storing llm_config in the config dict if provided
    if llm_config is not None:
        # Merge llm_config into config dict for storage
        # Use deepcopy to avoid mutating nested structures in the original config
        job_config = copy.deepcopy(config) if config else {}
        job_config['llm_config'] = llm_config.model_dump()
        job = job_store.create_job(request, config=job_config)
    else:
        job = job_store.create_job(request, config=config)
    
    # Schedule background processing
    background_tasks.add_task(process_clarification_job, job.id)
    
    logger.info(f"Started clarification job {job.id} with status PENDING")
    return job


def process_clarification_job(job_id: UUID, llm_client: Optional[Any] = None) -> None:
    """Process a clarification job asynchronously.
    
    Loads the job from the store, marks it RUNNING, invokes the synchronous
    clarification logic, saves the result, and marks it SUCCESS. If any
    exception occurs during processing, marks the job FAILED and captures
    the error message.
    
    This function is designed to be called via FastAPI BackgroundTasks but
    can also be invoked directly for testing purposes.
    
    Args:
        job_id: The UUID of the job to process
        llm_client: Optional pre-configured LLM client for dependency injection.
                   If not provided and job has llm_config, a client will be created.
                   Used primarily for testing with DummyLLMClient.
    """
    try:
        # Load the job
        job = job_store.get_job(job_id)
        
        # Only process jobs that are in PENDING state
        if job.status != JobStatus.PENDING:
            logger.warning(
                f"Skipping processing for job {job_id} because its status is "
                f"'{job.status.value}' (expected PENDING)."
            )
            return
        
        logger.info(f"Processing clarification job {job_id}")
        
        # Mark as RUNNING
        job_store.update_job(job_id, status=JobStatus.RUNNING)
        
        # ====================================================================
        # LLM CLIENT INITIALIZATION (not yet invoked)
        # ====================================================================
        # Check if LLM configuration is provided in the job config
        client = llm_client  # Use injected client if provided (for testing)
        
        if client is None and job.config and 'llm_config' in job.config:
            # Reconstruct ClarificationLLMConfig from stored dict
            llm_config_dict = job.config['llm_config']
            llm_config = ClarificationLLMConfig(**llm_config_dict)
            
            # Initialize LLM client using factory (but don't invoke it yet)
            try:
                client = get_llm_client(llm_config.provider, llm_config)
                logger.info(
                    f"Initialized {llm_config.provider} LLM client for job {job_id} "
                    f"with model {llm_config.model}"
                )
            except (ValueError, LLMCallError) as e:
                # Log client initialization failure but continue with deterministic processing
                # ValueError: Invalid/unsupported provider
                # LLMCallError: SDK missing or other client-specific initialization errors
                logger.warning(
                    f"Failed to initialize LLM client for job {job_id}: {e}. "
                    "Continuing with deterministic clarification."
                )
                client = None
        
        # ====================================================================
        # PLACEHOLDER: FUTURE LLM INVOCATION POINT
        # ====================================================================
        # TODO: In a future iteration, invoke the LLM client here to perform
        # AI-powered specification clarification. The invocation should:
        #
        # 1. Format the plan into appropriate prompts (system + user)
        # 2. Call client.complete(system_prompt, user_prompt, model, **kwargs)
        # 3. Parse the LLM response and transform it into ClarifiedPlan
        # 4. Handle LLMCallError exceptions appropriately
        # 5. Fall back to deterministic clarification on errors
        #
        # Example pseudocode:
        #     if client is not None:
        #         try:
        #             system_prompt = "You are a specification clarifier..."
        #             user_prompt = format_plan_as_prompt(job.request.plan)
        #             response = await client.complete(
        #                 system_prompt=system_prompt,
        #                 user_prompt=user_prompt,
        #                 model=llm_config.model,
        #                 temperature=llm_config.temperature,
        #                 max_tokens=llm_config.max_tokens
        #             )
        #             result = parse_llm_response_to_plan(response)
        #         except LLMCallError as e:
        #             logger.error(f"LLM invocation failed: {e}")
        #             result = clarify_plan(job.request.plan)  # Fallback
        #     else:
        #         result = clarify_plan(job.request.plan)  # No LLM configured
        # ====================================================================
        
        # For now, use existing deterministic logic (LLM not invoked)
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

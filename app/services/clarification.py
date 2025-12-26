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
import json
import logging
import re
from typing import Any, Optional, Tuple
from uuid import UUID

from fastapi import BackgroundTasks

from app.models.specs import ClarificationJob, ClarificationRequest, ClarifiedPlan, ClarifiedSpec, JobStatus, PlanInput
from app.services import job_store
from app.services.llm_clients import ClarificationLLMConfig, get_llm_client

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT BUILDING UTILITIES
# ============================================================================

def build_clarification_prompts(
    request: ClarificationRequest,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Tuple[str, str]:
    """Build system and user prompts for LLM-based specification clarification.
    
    This function transforms a ClarificationRequest into a pair of prompts that
    guide the LLM to clarify specifications by incorporating provided answers.
    The prompts are designed to elicit structured JSON responses conforming to
    the ClarifiedPlan schema.
    
    System Prompt:
        - Describes the clarification task
        - Enumerates the exact ClarifiedPlan schema with six required keys per spec
        - Emphasizes that responses must be raw JSON (no markdown, no prose)
        - Requires single-call processing (no multi-turn interactions)
    
    User Prompt:
        - Serializes source specs (purpose, vision, must, dont, nice, open_questions, assumptions)
        - Includes indexed answers keyed by (spec_index, question_index)
        - Uses pretty-printed JSON for machine readability
        - Wrapped with guardrail reminders to emit valid JSON only
    
    Args:
        request: The clarification request containing plan and answers
        provider: Optional LLM provider for future metrics (not exposed in prompts)
        model: Optional model identifier for future metrics (not exposed in prompts)
    
    Returns:
        Tuple of (system_prompt, user_prompt) ready for LLM invocation
    
    Raises:
        ValueError: If request is None or invalid
    
    Example:
        >>> request = ClarificationRequest(plan=plan_input, answers=answers)
        >>> system_prompt, user_prompt = build_clarification_prompts(request)
        >>> # Use prompts with LLM client
    
    Note:
        Provider and model parameters are threaded through for future use in
        metrics/telemetry but are not included in the generated prompts.
    """
    if request is None:
        raise ValueError("request must not be None")
    
    if not hasattr(request, 'plan') or request.plan is None:
        raise ValueError("request.plan must not be None")
    
    # Build system prompt
    system_prompt = _build_system_prompt()
    
    # Build user prompt with request data
    user_prompt = _build_user_prompt(request)
    
    return system_prompt, user_prompt


def _build_system_prompt() -> str:
    """Build the system prompt for specification clarification.
    
    Returns:
        System prompt string describing the task and output schema
    """
    return """You are a specification clarification assistant. Your task is to transform specifications with open questions into clarified specifications by incorporating provided answers.

INPUT FORMAT:
You will receive a JSON object containing:
1. "specs": Array of specifications, each with:
   - purpose: The purpose of the specification
   - vision: The vision statement
   - must: Array of must-have requirements
   - dont: Array of things to avoid
   - nice: Array of nice-to-have features
   - open_questions: Array of questions needing clarification
   - assumptions: Array of assumptions made

2. "answers": Array of question answers, each with:
   - spec_index: Index of the specification (0-based)
   - question_index: Index of the question within that spec (0-based)
   - question: The question text
   - answer: The answer provided

OUTPUT REQUIREMENTS:
You must return ONLY a valid JSON object with this exact structure:
{
  "specs": [
    {
      "purpose": "string",
      "vision": "string",
      "must": ["string"],
      "dont": ["string"],
      "nice": ["string"],
      "assumptions": ["string"]
    }
  ]
}

IMPORTANT RULES:
1. Each spec in your output must have exactly these 6 keys: purpose, vision, must, dont, nice, assumptions
2. Do NOT include "open_questions" in the output - questions should be resolved using provided answers
3. Return ONLY the JSON object - no markdown fences, no explanations, no additional text
4. Incorporate answers into the appropriate fields (must, dont, nice, assumptions, or vision)
5. If no answer is provided for a question, use your best judgment based on context
6. Preserve all original content from must, dont, nice, and assumptions arrays
7. This is a single-call task - return the complete clarified plan in one response"""


def _build_user_prompt(request: ClarificationRequest) -> str:
    """Build the user prompt containing the request data.
    
    Args:
        request: The clarification request
    
    Returns:
        User prompt string with serialized request data
    """
    # Serialize the plan specs
    specs_data = []
    for i, spec in enumerate(request.plan.specs):
        spec_dict = {
            "purpose": spec.purpose,
            "vision": spec.vision,
            "must": spec.must,
            "dont": spec.dont,
            "nice": spec.nice,
            "open_questions": spec.open_questions,
            "assumptions": spec.assumptions
        }
        specs_data.append(spec_dict)
    
    # Serialize the answers
    answers_data = []
    for answer in request.answers:
        answer_dict = {
            "spec_index": answer.spec_index,
            "question_index": answer.question_index,
            "question": answer.question,
            "answer": answer.answer
        }
        answers_data.append(answer_dict)
    
    # Build the complete input payload
    input_payload = {
        "specs": specs_data,
        "answers": answers_data
    }
    
    # Serialize to pretty JSON for readability
    json_str = json.dumps(input_payload, indent=2, ensure_ascii=False)
    
    # Wrap with guardrail instructions
    prompt = f"""Please clarify the following specifications by incorporating the provided answers:

{json_str}

Remember: Return ONLY valid JSON with the exact structure specified. No markdown fences, no explanations."""
    
    return prompt


# ============================================================================
# JSON CLEANUP UTILITIES
# ============================================================================

class JSONCleanupError(Exception):
    """Exception raised when JSON cleanup and parsing fails.
    
    Attributes:
        message: Human-readable error message
        raw_content: The original content that failed to parse
        attempts: Number of cleanup attempts made
    """
    
    def __init__(self, message: str, raw_content: str, attempts: int):
        """Initialize JSONCleanupError.
        
        Args:
            message: Error message
            raw_content: Original content that failed
            attempts: Number of attempts made
        """
        super().__init__(message)
        self.message = message
        self.raw_content = raw_content
        self.attempts = attempts


def cleanup_and_parse_json(
    raw_response: str,
    max_attempts: int = 3
) -> dict:
    """Clean up LLM response and parse as JSON with retry logic.
    
    This function attempts to extract valid JSON from an LLM response by:
    1. Removing markdown code fences (```json, ```)
    2. Stripping leading/trailing whitespace and chatter
    3. Retrying with progressively aggressive cleanup strategies
    4. Surfacing structured errors if all attempts fail
    
    The function preserves legitimate whitespace within JSON strings while
    removing extraneous formatting that LLMs sometimes add.
    
    Args:
        raw_response: Raw text response from LLM
        max_attempts: Maximum number of cleanup attempts (default: 3)
    
    Returns:
        Parsed JSON object as a Python dictionary
    
    Raises:
        JSONCleanupError: If parsing fails after all attempts
        ValueError: If raw_response is None or empty, or max_attempts < 1
    
    Example:
        >>> response = "```json\\n{\\\"key\\\": \\\"value\\\"}\\n```"
        >>> data = cleanup_and_parse_json(response)
        >>> print(data)
        {'key': 'value'}
    
    Cleanup Strategies (applied in order until success):
        1. Remove markdown fences and basic whitespace trimming
        2. Extract JSON between first { and last }
        3. Remove common prose patterns (e.g., "Here is...", "The result is...")
    """
    if raw_response is None:
        raise ValueError("raw_response must not be None")
    
    if not raw_response.strip():
        raise ValueError("raw_response must not be empty or blank")
    
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    
    # Define cleanup strategies
    strategies = [
        # Strategy 1: Remove markdown fences and basic cleanup
        lambda text: _remove_markdown_fences(text).strip(),
        # Strategy 2: Extract JSON between first { and last }
        lambda text: _extract_json_object(text) or "",
        # Strategy 3: Remove common prose patterns then fences
        lambda text: _remove_markdown_fences(_remove_prose_patterns(text)).strip(),
    ]
    
    # Track attempts for error reporting
    last_error = None
    
    # Try each strategy up to max_attempts
    for attempt_num in range(min(max_attempts, len(strategies))):
        cleaned = strategies[attempt_num](raw_response)
        
        if not cleaned:
            continue
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
    
    # If all strategies failed, raise error
    attempt_count = min(max_attempts, len(strategies))
    raise JSONCleanupError(
        f"Failed to parse JSON after {attempt_count} attempt(s). "
        f"Last error: {str(last_error) if last_error else 'Unknown error'}",
        raw_response,
        attempt_count
    )


def _remove_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text.
    
    Removes patterns like:
    - ```json
    - ```
    - ` (backticks)
    
    Args:
        text: Input text potentially containing markdown fences
    
    Returns:
        Text with fences removed
    """
    # Remove ```json or ``` at start/end
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    
    # Remove inline backticks at boundaries (but preserve in JSON strings)
    text = re.sub(r'^`+', '', text)
    text = re.sub(r'`+$', '', text)
    
    return text


def _extract_json_object(text: str) -> Optional[str]:
    """Extract JSON object between first { and last }.
    
    Args:
        text: Input text potentially containing JSON surrounded by prose
    
    Returns:
        Extracted JSON string or None if not found
    """
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        return text[first_brace:last_brace + 1]
    
    return None


def _remove_prose_patterns(text: str) -> str:
    """Remove common prose patterns that LLMs add before/after JSON.
    
    Args:
        text: Input text potentially containing prose
    
    Returns:
        Text with prose patterns removed
    """
    # Common phrases LLMs use before JSON
    # Use word boundaries and more precise patterns to avoid false matches
    prose_patterns = [
        r'^\s*(?:here\s+is|here\'s|the\s+result\s+is|the\s+answer\s+is|the\s+clarified\s+plan\s+is)\s*[:\s]*',
        r'^\s*(?:sure|okay|certainly)[,\s]*',
        r'^\s*(?:i\s+can\s+help|let\s+me\s+help|i\'ll\s+help)\s+.*?\s*',
    ]
    
    for pattern in prose_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text


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
            except ValueError as e:
                # Log client initialization failure but continue with deterministic processing
                # ValueError: Invalid/unsupported provider or factory-level errors
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

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

from app.models.config_models import ClarificationConfig
from app.models.specs import ClarificationJob, ClarificationRequest, ClarifiedPlan, ClarifiedSpec, JobStatus, PlanInput
from app.services import job_store
from app.services.llm_clients import ClarificationLLMConfig, get_llm_client

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT BUILDING UTILITIES
# ============================================================================

# System prompt templates for specification clarification
# All templates enforce strict JSON output requirements
SYSTEM_PROMPT_TEMPLATES = {
    "default": """You are a specification clarification assistant. Your task is to transform specifications with open questions into clarified specifications by incorporating provided answers.

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
7. This is a single-call task - return the complete clarified plan in one response""",
    
    "strict_json": """You are a specification clarification assistant operating in STRICT JSON MODE.

INPUT FORMAT:
You will receive a JSON object containing:
1. "specs": Array of specifications with purpose, vision, must, dont, nice, open_questions, assumptions
2. "answers": Array of question answers with spec_index, question_index, question, answer

OUTPUT REQUIREMENTS - STRICT JSON ONLY:
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

CRITICAL RULES - NO EXCEPTIONS:
1. Output MUST be valid, parseable JSON
2. Output MUST start with { and end with }
3. NO markdown code fences (no ```, no ```json)
4. NO explanatory text before or after the JSON
5. NO comments in the JSON
6. Each spec MUST have exactly these 6 keys: purpose, vision, must, dont, nice, assumptions
7. Do NOT include "open_questions" in output - resolve them using provided answers
8. All string values must be properly escaped
9. All arrays must be valid JSON arrays
10. Return the complete clarified plan in one response""",
    
    "verbose_explanation": """You are a specification clarification assistant. Your task is to transform specifications with open questions into clarified specifications by incorporating provided answers.

INPUT FORMAT:
You will receive a JSON object containing:
1. "specs": Array of specifications, each with:
   - purpose: The purpose of the specification
   - vision: The vision statement
   - must: Array of must-have requirements (mandatory features)
   - dont: Array of things to avoid (anti-patterns, constraints)
   - nice: Array of nice-to-have features (optional enhancements)
   - open_questions: Array of questions needing clarification
   - assumptions: Array of assumptions made about the system

2. "answers": Array of question answers, each with:
   - spec_index: Index of the specification (0-based)
   - question_index: Index of the question within that spec (0-based)
   - question: The question text
   - answer: The answer provided

YOUR TASK:
Analyze the specifications and provided answers. Incorporate each answer into the most appropriate field:
- If an answer clarifies a requirement → add to "must" array
- If an answer identifies something to avoid → add to "dont" array  
- If an answer describes an optional feature → add to "nice" array
- If an answer clarifies an assumption → add to "assumptions" array
- If an answer refines the vision → incorporate into "vision" string

OUTPUT REQUIREMENTS - STRICT JSON ONLY:
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
3. Return ONLY the JSON object - no markdown fences (```, ```json), no explanations, no additional text
4. All fields must be present for each spec (use empty arrays [] if no items)
5. Preserve all original content from must, dont, nice, and assumptions arrays
6. Add new content from answers to appropriate arrays
7. This is a single-call task - return the complete clarified plan in one response
8. The JSON must be valid and parseable - proper escaping, no trailing commas, no comments"""
}


def get_system_prompt_template(system_prompt_id: str) -> str:
    """Retrieve system prompt template by ID with fallback to default.
    
    This function looks up a system prompt template by its identifier and
    returns the corresponding template text. If the requested ID is not found,
    it logs a warning and returns the 'default' template as a fallback. This
    ensures that jobs never fail due to unknown system_prompt_id values.
    
    All templates enforce strict JSON output requirements to ensure consistent
    response parsing across different prompt styles.
    
    Args:
        system_prompt_id: Identifier for the system prompt template
                         (e.g., 'default', 'strict_json', 'verbose_explanation')
    
    Returns:
        System prompt template text
    
    Example:
        >>> template = get_system_prompt_template('strict_json')
        >>> # Returns strict_json template
        
        >>> template = get_system_prompt_template('unknown_id')
        >>> # Logs warning and returns 'default' template
    
    Note:
        The fallback behavior is intentional - we prefer degraded behavior
        (using default template) over failing the entire clarification job.
        All templates maintain strict JSON output requirements.
    """
    template = SYSTEM_PROMPT_TEMPLATES.get(system_prompt_id)
    if template:
        return template
    
    logger.warning(
        f"Unknown system_prompt_id '{system_prompt_id}'. "
        f"Available templates: {', '.join(sorted(SYSTEM_PROMPT_TEMPLATES.keys()))}. "
        "Falling back to 'default' template."
    )
    return SYSTEM_PROMPT_TEMPLATES['default']


def build_clarification_prompts(
    request: ClarificationRequest,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt_id: Optional[str] = None
) -> Tuple[str, str]:
    """Build system and user prompts for LLM-based specification clarification.
    
    This function transforms a ClarificationRequest into a pair of prompts that
    guide the LLM to clarify specifications by incorporating provided answers.
    The prompts are designed to elicit structured JSON responses conforming to
    the ClarifiedPlan schema.
    
    System Prompt:
        - Selected from SYSTEM_PROMPT_TEMPLATES based on system_prompt_id
        - Describes the clarification task
        - Enumerates the exact ClarifiedPlan schema with six required keys per spec
        - Emphasizes that responses must be raw JSON (no markdown, no prose)
        - Requires single-call processing (no multi-turn interactions)
        - All templates enforce strict JSON output requirements
    
    User Prompt:
        - Serializes source specs (purpose, vision, must, dont, nice, open_questions, assumptions)
        - Includes indexed answers keyed by (spec_index, question_index)
        - Uses pretty-printed JSON for machine readability
        - Wrapped with guardrail reminders to emit valid JSON only
    
    Args:
        request: The clarification request containing plan and answers
        provider: Optional LLM provider for future metrics (not exposed in prompts)
        model: Optional model identifier for future metrics (not exposed in prompts)
        system_prompt_id: Optional identifier for system prompt template (defaults to 'default')
    
    Returns:
        Tuple of (system_prompt, user_prompt) ready for LLM invocation
    
    Raises:
        ValueError: If request is None or invalid
        TypeError: If request is not a ClarificationRequest instance
    
    Example:
        >>> request = ClarificationRequest(plan=plan_input, answers=answers)
        >>> system_prompt, user_prompt = build_clarification_prompts(request)
        >>> # Use prompts with LLM client
        
        >>> # With custom template
        >>> system_prompt, user_prompt = build_clarification_prompts(
        ...     request, system_prompt_id='strict_json'
        ... )
    
    Note:
        Provider and model parameters are threaded through for future use in
        metrics/telemetry but are not included in the generated prompts.
        If system_prompt_id is unknown, falls back to 'default' template with
        a warning log entry.
    """
    if request is None:
        raise ValueError("request must not be None")
    
    # Validate request type
    if not isinstance(request, ClarificationRequest):
        raise TypeError(
            f"request must be a ClarificationRequest instance, got {type(request).__name__}"
        )
    
    # Validate plan exists (should always be present in valid ClarificationRequest)
    if request.plan is None:
        raise ValueError("request.plan must not be None")
    
    # Build system prompt using template (defaults to 'default' if not specified)
    system_prompt = _build_system_prompt(system_prompt_id=system_prompt_id)
    
    # Build user prompt with request data
    user_prompt = _build_user_prompt(request)
    
    return system_prompt, user_prompt


def _build_system_prompt(system_prompt_id: Optional[str] = None) -> str:
    """Build the system prompt for specification clarification.
    
    Retrieves the appropriate system prompt template based on the provided
    system_prompt_id. If no ID is provided, defaults to 'default' template.
    If an unknown ID is provided, logs a warning and falls back to 'default'.
    
    Args:
        system_prompt_id: Optional identifier for system prompt template
                         Defaults to 'default' if not provided
    
    Returns:
        System prompt string describing the task and output schema
    """
    # Default to 'default' template if not specified
    if system_prompt_id is None:
        system_prompt_id = 'default'
    
    # Retrieve template (with fallback handling)
    return get_system_prompt_template(system_prompt_id)


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
                     Note: The function has 3 built-in cleanup strategies.
                     If max_attempts is greater than 3, only 3 attempts will be made.
                     If max_attempts is less than 3, only that many strategies will be tried.
    
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
        
        >>> # Try only first strategy
        >>> data = cleanup_and_parse_json(response, max_attempts=1)
    
    Cleanup Strategies (applied in order until success):
        1. Remove markdown fences and basic whitespace trimming
        2. Extract JSON between first { and last }
        3. Remove common prose patterns (e.g., "Here is...", "The result is...")
    
    Note:
        The number of attempts will be limited to the number of available strategies (3).
        If you request max_attempts=5, only 3 attempts will be made since there are
        only 3 cleanup strategies implemented.
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
    
    # Try each strategy up to max_attempts (limited by available strategies)
    num_attempts = min(max_attempts, len(strategies))
    for attempt_num in range(num_attempts):
        cleaned = strategies[attempt_num](raw_response)
        
        if not cleaned:
            continue
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
    
    # If all strategies failed, raise error
    raise JSONCleanupError(
        f"Failed to parse JSON after {num_attempts} attempt(s). "
        f"Last error: {str(last_error) if last_error else 'Unknown error'}",
        raw_response,
        num_attempts
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
    """Extract the first valid JSON object from the text.
    
    This function finds the first opening brace '{' and scans forward to find
    its matching closing brace '}', respecting nested structures.
    
    Args:
        text: Input text potentially containing JSON surrounded by prose
    
    Returns:
        Extracted JSON string or None if not found
    """
    try:
        first_brace_index = text.index('{')
    except ValueError:
        return None  # No opening brace found
    
    balance = 0
    in_string = False
    escape_next = False
    
    for i in range(first_brace_index, len(text)):
        char = text[i]
        
        # Handle string content to avoid counting braces inside strings
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        # Only count braces outside of strings
        if not in_string:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
            
            if balance == 0:
                # Found the matching closing brace
                return text[first_brace_index : i + 1]
    
    return None  # No matching closing brace found


def _remove_prose_patterns(text: str) -> str:
    """Remove common prose patterns that LLMs add before/after JSON.
    
    Args:
        text: Input text potentially containing prose
    
    Returns:
        Text with prose patterns removed
    """
    # Common phrases LLMs use before JSON
    # Use word boundaries and more precise patterns to avoid false matches
    leading_prose_patterns = [
        r'^\s*(?:here\s+is|here\'s|the\s+result\s+is|the\s+answer\s+is|the\s+clarified\s+plan\s+is)\s*[:\s]*',
        r'^\s*(?:sure|okay|certainly)[,\s]*',
        # Match "I can/let me/I'll help..." followed by anything up to newline or colon
        r'^\s*(?:i\s+can\s+help|let\s+me\s+help|i\'ll\s+help)(?:[^\n:]*?[:]\s*)?',
    ]
    
    # Common phrases LLMs use after JSON
    trailing_prose_patterns = [
        r'[\s,.]*(?:let\s+me\s+know|hope\s+this\s+helps|if\s+you\s+need\s+anything\s+else).*$',
    ]
    
    for pattern in leading_prose_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    for pattern in trailing_prose_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
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
        llm_config: Optional LLM configuration for LLM-powered clarification.
                   If not provided, defaults to provider="openai", model="gpt-5".
        
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
    
    # Schedule background processing (FastAPI handles async functions automatically)
    background_tasks.add_task(process_clarification_job, job.id)
    
    logger.info(f"Started clarification job {job.id} with status PENDING")
    return job


async def process_clarification_job(job_id: UUID, llm_client: Optional[Any] = None) -> None:
    """Process a clarification job asynchronously.
    
    Loads the job from the store, marks it RUNNING, invokes the LLM pipeline
    to clarify specifications, saves the result, and marks it SUCCESS. If any
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
    import asyncio
    import time
    from pydantic import ValidationError
    
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
        # LLM CONFIG RESOLUTION
        # ====================================================================
        # Resolve LLM configuration: use stored config or apply defaults
        llm_config = None
        system_prompt_id = 'default'  # Default system prompt template
        
        if job.config and 'clarification_config' in job.config:
            # Reconstruct ClarificationConfig from stored dict with error handling
            try:
                clarification_config_dict = job.config['clarification_config']
                clarification_config = ClarificationConfig(**clarification_config_dict)
                
                # Extract system_prompt_id if present
                if clarification_config.system_prompt_id:
                    system_prompt_id = clarification_config.system_prompt_id
                
                # Convert to ClarificationLLMConfig (without system_prompt_id)
                # Only pass non-None values to avoid validation errors
                llm_config_kwargs = {
                    'provider': clarification_config.provider,
                    'model': clarification_config.model,
                }
                if clarification_config.temperature is not None:
                    llm_config_kwargs['temperature'] = clarification_config.temperature
                if clarification_config.max_tokens is not None:
                    llm_config_kwargs['max_tokens'] = clarification_config.max_tokens
                
                llm_config = ClarificationLLMConfig(**llm_config_kwargs)
            except Exception as e:
                # If config reconstruction fails, log error and fall through to defaults
                logger.warning(
                    f"Failed to reconstruct clarification_config for job {job_id}: {e}. "
                    "Falling back to default configuration."
                )
                # llm_config remains None, will be handled below
        elif job.config and 'llm_config' in job.config:
            # Legacy support: Reconstruct ClarificationLLMConfig from stored dict
            llm_config_dict = job.config['llm_config']
            llm_config = ClarificationLLMConfig(**llm_config_dict)
            # Legacy configs don't have system_prompt_id, so 'default' is used
        
        # If no config loaded, apply default configuration
        if llm_config is None:
            llm_config = ClarificationLLMConfig(
                provider="openai",
                model="gpt-5"
            )
            # system_prompt_id is already 'default'
        
        # ====================================================================
        # LLM CLIENT INITIALIZATION
        # ====================================================================
        client = llm_client  # Use injected client if provided (for testing)
        
        if client is None:
            # Initialize LLM client using factory
            try:
                client = get_llm_client(llm_config.provider, llm_config)
                logger.info(
                    f"Initialized {llm_config.provider} LLM client for job {job_id} "
                    f"with model {llm_config.model}"
                )
            except ValueError as e:
                # Invalid/unsupported provider - fail the job immediately
                error_message = f"Invalid LLM provider '{llm_config.provider}': {str(e)}"
                logger.error(f"Job {job_id} failed: {error_message}")
                job_store.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    last_error=error_message,
                    result=None
                )
                return
        
        # ====================================================================
        # VALIDATE JOB HAS CLARIFICATION REQUEST
        # ====================================================================
        if job.request is None:
            error_message = "Job missing ClarificationRequest - cannot process"
            logger.error(f"Job {job_id} failed: {error_message}")
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                last_error=error_message,
                result=None
            )
            return
        
        # ====================================================================
        # BUILD PROMPTS
        # ====================================================================
        try:
            system_prompt, user_prompt = build_clarification_prompts(
                job.request,
                provider=llm_config.provider,
                model=llm_config.model,
                system_prompt_id=system_prompt_id
            )
        except Exception as e:
            error_message = f"Failed to build prompts: {type(e).__name__}: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_message}")
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                last_error=error_message,
                result=None
            )
            return
        
        # ====================================================================
        # INVOKE LLM
        # ====================================================================
        start_time = time.perf_counter()
        try:
            # Prepare kwargs for LLM call
            llm_kwargs = {}
            if llm_config.temperature is not None:
                llm_kwargs['temperature'] = llm_config.temperature
            if llm_config.max_tokens is not None:
                llm_kwargs['max_tokens'] = llm_config.max_tokens
            
            # Call LLM exactly once
            raw_response = await client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=llm_config.model,
                **llm_kwargs
            )
            elapsed_time = time.perf_counter() - start_time
            
            # Log success metrics (without prompts or full response)
            logger.info(
                f"LLM call successful for job {job_id}: "
                f"provider={llm_config.provider}, model={llm_config.model}, "
                f"elapsed_time={elapsed_time:.2f}s"
            )
            
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            # Sanitize error message (LLMCallError already does this, but be safe)
            error_message = str(e)
            logger.error(
                f"LLM call failed for job {job_id}: "
                f"provider={llm_config.provider}, model={llm_config.model}, "
                f"elapsed_time={elapsed_time:.2f}s, error={error_message}"
            )
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                last_error=f"LLM call failed: {error_message}",
                result=None
            )
            return
        
        # ====================================================================
        # PARSE AND VALIDATE LLM RESPONSE
        # ====================================================================
        try:
            # Clean up and parse JSON
            parsed_json = cleanup_and_parse_json(raw_response)
            
            # Validate as ClarifiedPlan
            result = ClarifiedPlan(**parsed_json)
            
            logger.info(f"Successfully parsed and validated LLM response for job {job_id}")
            
        except JSONCleanupError as e:
            error_message = f"Failed to parse LLM response: {e.message}"
            logger.error(f"Job {job_id} failed: {error_message}")
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                last_error=error_message,
                result=None
            )
            return
        except ValidationError as e:
            error_message = f"LLM response validation failed: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_message}")
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                last_error=error_message,
                result=None
            )
            return
        except Exception as e:
            error_message = f"Unexpected error parsing response: {type(e).__name__}: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_message}")
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                last_error=error_message,
                result=None
            )
            return
        
        # ====================================================================
        # PERSIST RESULT
        # ====================================================================
        # Mark as SUCCESS with result (updated_at will be refreshed automatically)
        job_store.update_job(job_id, status=JobStatus.SUCCESS, result=result)
        logger.info(f"Clarification job {job_id} completed successfully")
        
    except job_store.JobNotFoundError:
        # Job doesn't exist - log and return cleanly without crashing
        logger.warning(f"Job {job_id} not found during processing - skipping")
        return
        
    except Exception as e:
        # Capture any unexpected exception and mark job as FAILED
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

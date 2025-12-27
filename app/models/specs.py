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
"""Pydantic models for specification clarification."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Import and re-export ClarificationConfig to maintain backward compatibility
from app.models.config_models import ClarificationConfig


class JobStatus(str, Enum):
    """Status of a clarification job.

    Attributes:
        PENDING: Job created but not yet started
        RUNNING: Job is currently being processed
        SUCCESS: Job completed successfully
        FAILED: Job failed with an error
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class SpecInput(BaseModel):
    """Input specification with questions to be clarified.

    Attributes:
        purpose: The purpose of the specification
        vision: The vision statement
        must: List of must-have requirements
        dont: List of don't requirements (things to avoid)
        nice: List of nice-to-have features
        open_questions: List of questions needing clarification
        assumptions: List of assumptions made
    """

    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(..., description="The purpose of the specification")
    vision: str = Field(..., description="The vision statement")
    must: list[str] = Field(default_factory=list, description="List of must-have requirements")
    dont: list[str] = Field(default_factory=list, description="List of don't requirements")
    nice: list[str] = Field(default_factory=list, description="List of nice-to-have features")
    open_questions: list[str] = Field(
        default_factory=list, description="List of questions needing clarification"
    )
    assumptions: list[str] = Field(default_factory=list, description="List of assumptions made")


class PlanInput(BaseModel):
    """Input plan containing multiple specifications.

    Attributes:
        specs: List of specification inputs
    """

    model_config = ConfigDict(extra="forbid")

    specs: list[SpecInput] = Field(..., description="List of specification inputs")


class ClarifiedSpec(BaseModel):
    """Clarified specification after processing (no open questions).

    Attributes:
        purpose: The purpose of the specification
        vision: The vision statement
        must: List of must-have requirements
        dont: List of don't requirements (things to avoid)
        nice: List of nice-to-have features
        assumptions: List of assumptions made
    """

    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(..., description="The purpose of the specification")
    vision: str = Field(..., description="The vision statement")
    must: list[str] = Field(default_factory=list, description="List of must-have requirements")
    dont: list[str] = Field(default_factory=list, description="List of don't requirements")
    nice: list[str] = Field(default_factory=list, description="List of nice-to-have features")
    assumptions: list[str] = Field(default_factory=list, description="List of assumptions made")


class ClarifiedPlan(BaseModel):
    """Plan containing clarified specifications.

    Attributes:
        specs: List of clarified specifications
    """

    model_config = ConfigDict(extra="forbid")

    specs: list[ClarifiedSpec] = Field(..., description="List of clarified specifications")


class QuestionAnswer(BaseModel):
    """Answer to a specific question in a specification.

    Attributes:
        spec_index: Index of the specification containing the question
        question_index: Index of the question within the specification
        question: The question text
        answer: The answer provided
    """

    model_config = ConfigDict(extra="forbid")

    spec_index: int = Field(
        ..., ge=0, description="Index of the specification containing the question"
    )
    question_index: int = Field(
        ..., ge=0, description="Index of the question within the specification"
    )
    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer provided")


class ClarificationRequest(BaseModel):
    """Request to clarify specifications with optional answers.

    Attributes:
        plan: The plan input with specifications
        answers: List of answers to open questions (optional, currently ignored)
    """

    model_config = ConfigDict(extra="forbid")

    plan: PlanInput = Field(..., description="The plan input with specifications")
    answers: list[QuestionAnswer] = Field(
        default_factory=list, description="List of answers to open questions"
    )


class ClarificationRequestWithConfig(BaseModel):
    """Request to clarify specifications with optional answers and config.

    This schema extends the basic ClarificationRequest by allowing clients to
    optionally supply a ClarificationConfig that overrides process defaults.
    The config is validated and merged with defaults before job processing.

    Attributes:
        plan: The plan input with specifications
        answers: List of answers to open questions (optional, currently ignored)
        config: Optional ClarificationConfig to override defaults for this request
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plan": {
                        "specs": [
                            {
                                "purpose": "Build a user authentication system",
                                "vision": "A secure, scalable authentication system supporting multiple providers",
                                "must": [
                                    "Support OAuth 2.0",
                                    "Implement JWT tokens",
                                    "Store passwords securely with bcrypt",
                                ],
                                "dont": [
                                    "Store passwords in plain text",
                                    "Use deprecated authentication methods",
                                ],
                                "nice": [
                                    "Support biometric authentication",
                                    "Multi-factor authentication",
                                ],
                                "open_questions": [
                                    "Which OAuth providers should be supported?",
                                    "What is the token expiration policy?",
                                ],
                                "assumptions": [
                                    "Users have valid email addresses",
                                    "System will be deployed on HTTPS",
                                ],
                            }
                        ]
                    },
                    "answers": [],
                }
            ]
        },
    )

    plan: PlanInput = Field(..., description="The plan input with specifications")
    answers: list[QuestionAnswer] = Field(
        default_factory=list, description="List of answers to open questions"
    )
    config: ClarificationConfig | None = Field(
        None, description="Optional config to override defaults"
    )


class ClarificationJob(BaseModel):
    """Clarification job tracking status and results.

    Represents a clarification job with status tracking, timestamps, error handling,
    and optional configuration. Jobs are stored in-memory and can be polled for status.

    Attributes:
        id: Unique identifier for the job
        status: Current status of the job
        created_at: UTC timestamp when job was created
        updated_at: UTC timestamp when job was last updated
        last_error: Optional error message if job failed
        request: The original clarification request
        result: Optional clarified plan result when job succeeds
        config: Optional configuration dictionary for job processing
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current status of the job")
    created_at: datetime = Field(..., description="UTC timestamp when job was created")
    updated_at: datetime = Field(..., description="UTC timestamp when job was last updated")
    last_error: str | None = Field(None, description="Optional error message if job failed")
    request: ClarificationRequest = Field(..., description="The original clarification request")
    result: ClarifiedPlan | None = Field(
        None, description="Optional clarified plan result when job succeeds"
    )
    config: dict | None = Field(
        None, description="Optional configuration dictionary for job processing"
    )


class JobSummaryResponse(BaseModel):
    """Lightweight job summary for POST responses.

    Returns minimal job metadata without the full request or result payloads.
    This keeps POST responses lightweight as required by the API specification.

    Attributes:
        id: Unique identifier for the job
        status: Current status of the job
        created_at: UTC timestamp when job was created
        updated_at: UTC timestamp when job was last updated
        last_error: Optional error message if job failed
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current status of the job")
    created_at: datetime = Field(..., description="UTC timestamp when job was created")
    updated_at: datetime = Field(..., description="UTC timestamp when job was last updated")
    last_error: str | None = Field(None, description="Optional error message if job failed")


class JobStatusResponse(BaseModel):
    """Job status response with conditional result field.

    Returns job status and details. The result field is always present in the response
    schema but is conditionally populated based on the APP_SHOW_JOB_RESULT flag and
    job status. In production mode (flag=False), the result is always set to null to
    keep responses lightweight and prevent clients from depending on embedded results.

    The result field behavior:
    - Production mode (APP_SHOW_JOB_RESULT=false): result is always null
    - Development mode (APP_SHOW_JOB_RESULT=true): result contains ClarifiedPlan when
      job status is SUCCESS, otherwise null

    Attributes:
        id: Unique identifier for the job
        status: Current status of the job
        created_at: UTC timestamp when job was created
        updated_at: UTC timestamp when job was last updated
        last_error: Optional error message if job failed
        result: Optional clarified plan result (null in production, populated in
                development mode only when status is SUCCESS)
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current status of the job")
    created_at: datetime = Field(..., description="UTC timestamp when job was created")
    updated_at: datetime = Field(..., description="UTC timestamp when job was last updated")
    last_error: str | None = Field(None, description="Optional error message if job failed")
    result: ClarifiedPlan | None = Field(
        None, description="Optional clarified plan result when job succeeds (development mode only)"
    )

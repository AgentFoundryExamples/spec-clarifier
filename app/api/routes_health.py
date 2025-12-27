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
"""Health check endpoint."""

from fastapi import APIRouter

from app.utils.metrics import get_metrics_collector

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check() -> dict:
    """Health check endpoint.

    Returns:
        dict: Health status response
    """
    return {"status": "ok"}


@router.get(
    "/v1/metrics/basic",
    summary="Get basic operational metrics",
    description=(
        "Returns lightweight operational counters for monitoring job processing "
        "and LLM interactions. This endpoint is read-only and does not require "
        "authentication.\n\n"
        "**Metrics Tracked:**\n"
        "- jobs_queued: Total number of jobs created\n"
        "- jobs_pending: Current number of jobs in PENDING state\n"
        "- jobs_running: Current number of jobs in RUNNING state\n"
        "- jobs_success: Total number of jobs completed successfully\n"
        "- jobs_failed: Total number of jobs that failed\n"
        "- llm_errors: Total number of LLM API errors encountered\n\n"
        "All counters are maintained in memory and reset on service restart."
    ),
    response_model=dict,
)
def get_basic_metrics() -> dict:
    """Get basic operational metrics.

    Returns lightweight counters for job processing and LLM interactions.
    This endpoint is designed for monitoring and does not expose sensitive
    information about job contents or API keys.

    Returns:
        dict: Dictionary mapping counter names to current values
    """
    metrics = get_metrics_collector()
    return metrics.get_all()

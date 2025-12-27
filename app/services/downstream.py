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
"""Downstream dispatch abstraction for forwarding clarified plans."""

import json
import logging
from typing import Protocol

from app.models.specs import ClarificationJob, ClarifiedPlan
from app.utils.logging_helper import log_info

logger = logging.getLogger(__name__)


class DownstreamDispatcher(Protocol):
    """Protocol for dispatching clarified plans to downstream systems.

    Implementations of this protocol can forward clarified plans to:
    - HTTP endpoints
    - Message queues (e.g., RabbitMQ, Kafka)
    - Other microservices
    - Storage systems

    The dispatch method should be idempotent where possible and handle
    transient failures gracefully. Implementations must not raise exceptions
    that would crash the clarification worker.
    """

    async def dispatch(self, job: ClarificationJob, plan: ClarifiedPlan) -> None:
        """Dispatch a clarified plan to a downstream system.

        This method is called after a clarification job completes successfully.
        It should forward the clarified plan to the appropriate downstream
        system for further processing.

        Args:
            job: The completed clarification job with metadata
            plan: The clarified plan containing specifications

        Raises:
            Exception: Implementations may raise exceptions for unrecoverable
                      errors. The caller is responsible for handling these
                      exceptions without affecting job status.

        Note:
            This method should not mutate the job or plan objects.
            Implementations should be stateless or thread-safe for concurrent use.
        """
        ...


class PlaceholderDownstreamDispatcher:
    """Placeholder dispatcher that logs clarified plans without external calls.

    This is a temporary implementation that outputs clarified plans to logs
    with clear banner messages. It serves as:
    1. A reference for the dispatcher interface
    2. A hook point for future integrations
    3. A debugging aid for operators

    TODO: Replace this placeholder with a real downstream integration:
    - Determine the target downstream system (HTTP endpoint, message queue, etc.)
    - Implement error handling for network failures and timeouts
    - Add retry logic with exponential backoff for transient failures
    - Consider authentication/authorization requirements
    - Add metrics for dispatch success/failure rates
    - Ensure idempotency to handle duplicate dispatches
    - Add configuration for endpoint URLs and credentials (via environment)

    Example real implementations:
    - HTTPDownstreamDispatcher: POST plans to REST endpoint
    - QueueDownstreamDispatcher: Publish plans to message queue
    - StorageDownstreamDispatcher: Save plans to object storage
    """

    async def dispatch(self, job: ClarificationJob, plan: ClarifiedPlan) -> None:
        """Log the clarified plan with clear banner messages.

        This placeholder implementation:
        1. Logs START/END banners to make dispatch events visible
        2. Serializes the ClarifiedPlan to JSON for inspection
        3. Includes job_id for correlation with job processing logs
        4. Uses both structured logging and print for operator visibility

        Args:
            job: The completed clarification job
            plan: The clarified plan to dispatch
        """
        job_id = str(job.id)

        # Serialize plan to JSON for logging with error handling
        try:
            plan_json = json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            # Fallback to string representation if JSON serialization fails
            plan_json = f"[JSON serialization failed: {e}]\n{str(plan)}"

        # Banner start - use print for high visibility in production logs
        banner_start = f"{'=' * 80}\n" f"DOWNSTREAM DISPATCH START - Job {job_id}\n" f"{'=' * 80}"
        print(banner_start)
        logger.info(banner_start)

        # Log structured dispatch event
        log_info(
            logger,
            "downstream_dispatch_placeholder",
            job_id=job_id,
            num_specs=len(plan.specs),
            message="Placeholder dispatcher invoked - plan would be sent to downstream system",
        )

        # Log the plan JSON
        plan_output = f"Clarified Plan JSON:\n{plan_json}"
        print(plan_output)
        logger.info(plan_output)

        # Banner end
        banner_end = f"{'=' * 80}\n" f"DOWNSTREAM DISPATCH END - Job {job_id}\n" f"{'=' * 80}"
        print(banner_end)
        logger.info(banner_end)


def get_downstream_dispatcher() -> DownstreamDispatcher:
    """Factory function to obtain the configured downstream dispatcher.

    This function returns the appropriate dispatcher implementation based on
    configuration. Currently returns the placeholder, but can be extended to:
    - Read environment variables to determine dispatcher type
    - Initialize HTTP/queue clients with proper credentials
    - Return mock dispatchers in test environments

    Returns:
        DownstreamDispatcher: The configured dispatcher instance

    Example future implementation:
        dispatcher_type = os.getenv("DOWNSTREAM_DISPATCHER_TYPE", "placeholder")
        if dispatcher_type == "http":
            return HTTPDownstreamDispatcher(
                endpoint=os.getenv("DOWNSTREAM_HTTP_ENDPOINT"),
                api_key=os.getenv("DOWNSTREAM_API_KEY")
            )
        elif dispatcher_type == "queue":
            return QueueDownstreamDispatcher(
                queue_url=os.getenv("DOWNSTREAM_QUEUE_URL")
            )
        else:
            return PlaceholderDownstreamDispatcher()
    """
    return PlaceholderDownstreamDispatcher()

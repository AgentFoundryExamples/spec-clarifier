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
"""Thread-safe in-memory storage for clarification jobs."""

import logging
import threading
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.models.specs import ClarificationJob, ClarificationRequest, ClarifiedPlan, JobStatus
from app.utils.logging_helper import log_error, log_info
from app.utils.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


# Module-level job store with thread-safe access
_job_store: dict[UUID, ClarificationJob] = {}
_store_lock = threading.Lock()

# TTL configuration (in seconds) - default 24 hours
DEFAULT_JOB_TTL_SECONDS = 24 * 60 * 60


class JobNotFoundError(Exception):
    """Raised when a job is not found in the store."""
    pass


def create_job(
    request: ClarificationRequest,
    config: dict | None = None
) -> ClarificationJob:
    """Create a new clarification job in the store.
    
    Creates a new job with PENDING status and sets both created_at and updated_at
    to the current UTC time. The job is assigned a unique UUID identifier.
    
    Args:
        request: The clarification request to process
        config: Optional configuration dictionary for job processing
        
    Returns:
        ClarificationJob: The newly created job
    """
    now = datetime.now(UTC)
    job_id = uuid4()

    job = ClarificationJob(
        id=job_id,
        status=JobStatus.PENDING,
        created_at=now,
        updated_at=now,
        last_error=None,
        request=request,
        result=None,
        config=config,
    )

    with _store_lock:
        _job_store[job_id] = job

    # Update metrics
    metrics = get_metrics_collector()
    metrics.increment("jobs_queued")
    metrics.increment("jobs_pending")

    # Log job creation
    log_info(
        logger,
        "job_created",
        job_id=job_id,
        status=JobStatus.PENDING.value,
        num_specs=len(request.plan.specs) if request.plan else 0,
        num_answers=len(request.answers) if request.answers else 0
    )

    return job


def get_job(job_id: UUID) -> ClarificationJob:
    """Retrieve a job from the store by ID.
    
    Returns a deep copy of the job to ensure thread-safety when the job
    is accessed outside the lock.
    
    Args:
        job_id: The UUID of the job to retrieve
        
    Returns:
        ClarificationJob: A deep copy of the requested job
        
    Raises:
        JobNotFoundError: If the job does not exist in the store
    """
    with _store_lock:
        job = _job_store.get(job_id)

    if job is None:
        raise JobNotFoundError(f"Job with id {job_id} not found")

    return job.model_copy(deep=True)


def update_job(
    job_id: UUID,
    status: JobStatus | None = None,
    result: ClarifiedPlan | None = None,
    last_error: str | None = None
) -> ClarificationJob:
    """Update an existing job in the store.
    
    Updates the specified fields of a job and always refreshes the updated_at
    timestamp. The job must exist in the store.
    
    Args:
        job_id: The UUID of the job to update
        status: Optional new status for the job
        result: Optional clarified plan result
        last_error: Optional error message
        
    Returns:
        ClarificationJob: The updated job
        
    Raises:
        JobNotFoundError: If the job does not exist in the store
    """
    metrics = get_metrics_collector()

    with _store_lock:
        job = _job_store.get(job_id)

        if job is None:
            raise JobNotFoundError(f"Job with id {job_id} not found")

        old_status = job.status

        # Create updated job with new timestamp
        update_dict = {"updated_at": datetime.now(UTC)}

        if status is not None:
            update_dict["status"] = status
        if result is not None:
            update_dict["result"] = result
        if last_error is not None:
            update_dict["last_error"] = last_error

        updated_job = job.model_copy(update=update_dict)
        _job_store[job_id] = updated_job

        new_status = updated_job.status

    # Update metrics based on status transitions (outside lock)
    if status is not None and old_status != new_status:
        # Decrement old status counter
        if old_status == JobStatus.PENDING:
            metrics.decrement("jobs_pending")
        elif old_status == JobStatus.RUNNING:
            metrics.decrement("jobs_running")

        # Increment new status counter
        if new_status == JobStatus.PENDING:
            metrics.increment("jobs_pending")
        elif new_status == JobStatus.RUNNING:
            metrics.increment("jobs_running")
        elif new_status == JobStatus.SUCCESS:
            metrics.increment("jobs_success")
        elif new_status == JobStatus.FAILED:
            metrics.increment("jobs_failed")

        # Log status transition
        log_info(
            logger,
            "job_status_transition",
            job_id=job_id,
            old_status=old_status.value,
            new_status=new_status.value
        )

    # Log error if present
    if last_error is not None:
        log_error(
            logger,
            "job_error_recorded",
            job_id=job_id,
            error_message=last_error
        )

    return updated_job


def list_jobs(
    status: JobStatus | None = None,
    limit: int | None = None
) -> list[ClarificationJob]:
    """List jobs from the store, optionally filtered by status.
    
    Returns deep copies of jobs to ensure thread-safety when jobs
    are accessed outside the lock.
    
    Args:
        status: Optional status to filter jobs by
        limit: Optional maximum number of jobs to return
        
    Returns:
        List[ClarificationJob]: List of deep copies of jobs matching the criteria
    """
    with _store_lock:
        job_refs = list(_job_store.values())

    if status is not None:
        job_refs = [job for job in job_refs if job.status == status]

    # Sort by created_at descending (newest first)
    job_refs.sort(key=lambda j: j.created_at, reverse=True)

    if limit is not None:
        job_refs = job_refs[:limit]

    # Return deep copies to ensure thread-safety
    return [job.model_copy(deep=True) for job in job_refs]


def delete_job(job_id: UUID) -> None:
    """Delete a job from the store.
    
    Args:
        job_id: The UUID of the job to delete
        
    Raises:
        JobNotFoundError: If the job does not exist in the store
    """
    metrics = get_metrics_collector()

    with _store_lock:
        if job_id not in _job_store:
            raise JobNotFoundError(f"Job with id {job_id} not found")

        job = _job_store[job_id]
        job_status = job.status
        del _job_store[job_id]

    # Update metrics based on deleted job status (outside lock)
    if job_status == JobStatus.PENDING:
        metrics.decrement("jobs_pending")
    elif job_status == JobStatus.RUNNING:
        metrics.decrement("jobs_running")

    # Log deletion
    log_info(
        logger,
        "job_deleted",
        job_id=job_id,
        status=job_status.value
    )


def cleanup_expired_jobs(
    ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS,
    stale_pending_ttl_seconds: int | None = None
) -> int:
    """Clean up expired jobs from the store.
    
    Removes jobs that are completed (SUCCESS or FAILED status) and older than
    the specified TTL. Jobs with RUNNING status are never removed to prevent
    data loss during processing.
    
    Optionally removes PENDING jobs that have been stale for longer than
    stale_pending_ttl_seconds to prevent memory leaks from abandoned jobs.
    
    Args:
        ttl_seconds: Time-to-live in seconds for completed jobs (default: 24 hours)
        stale_pending_ttl_seconds: Optional TTL for stale PENDING jobs. If provided,
            PENDING jobs older than this threshold will be removed. Useful for
            cleaning up jobs that were never processed due to worker crashes.
        
    Returns:
        int: Number of jobs cleaned up
    """
    now = datetime.now(UTC)
    cutoff_time = now - timedelta(seconds=ttl_seconds)
    metrics = get_metrics_collector()

    jobs_to_remove = []
    with _store_lock:
        for job_id, job in _job_store.items():
            # Clean up completed jobs (SUCCESS or FAILED)
            if job.status in (JobStatus.SUCCESS, JobStatus.FAILED):
                if job.updated_at < cutoff_time:
                    jobs_to_remove.append((job_id, job.status))
            # Optionally clean up stale PENDING jobs
            elif job.status == JobStatus.PENDING and stale_pending_ttl_seconds is not None:
                stale_cutoff = now - timedelta(seconds=stale_pending_ttl_seconds)
                if job.created_at < stale_cutoff:
                    jobs_to_remove.append((job_id, job.status))

    if not jobs_to_remove:
        return 0

    # Update metrics for jobs to be removed (outside lock)
    for _, job_status in jobs_to_remove:
        if job_status == JobStatus.PENDING:
            metrics.decrement("jobs_pending")

    # Remove expired jobs
    with _store_lock:
        for job_id, _ in jobs_to_remove:
            if job_id in _job_store:
                del _job_store[job_id]

    cleanup_count = len(jobs_to_remove)

    # Log cleanup summary (outside lock)
    log_info(
        logger,
        "jobs_cleanup_completed",
        jobs_removed=cleanup_count,
        ttl_seconds=ttl_seconds
    )

    return cleanup_count


def clear_all_jobs() -> None:
    """Clear all jobs from the store.
    
    This is primarily useful for testing. Use with caution in production.
    """
    with _store_lock:
        _job_store.clear()

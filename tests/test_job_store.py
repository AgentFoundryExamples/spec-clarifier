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
"""Tests for job store service."""

import threading
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.models.specs import (
    ClarificationRequest,
    ClarifiedPlan,
    ClarifiedSpec,
    JobStatus,
    PlanInput,
    SpecInput,
)
from app.services import job_store
from app.services.job_store import (
    JobNotFoundError,
    cleanup_expired_jobs,
    clear_all_jobs,
    create_job,
    delete_job,
    get_job,
    list_jobs,
    update_job,
)


@pytest.fixture(autouse=True)
def clean_job_store():
    """Clean the job store before and after each test."""
    clear_all_jobs()
    yield
    clear_all_jobs()


class TestCreateJob:
    """Tests for create_job function."""

    def test_create_job_basic(self):
        """Test creating a basic job."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)

        assert job.id is not None
        assert job.status == JobStatus.PENDING
        assert job.request == request
        assert job.result is None
        assert job.last_error is None
        assert job.config is None
        assert job.created_at is not None
        assert job.updated_at is not None
        assert job.created_at == job.updated_at

    def test_create_job_with_config(self):
        """Test creating a job with configuration."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        config = {"model": "gpt-4", "temperature": 0.7}

        job = create_job(request, config=config)

        assert job.config == config

    def test_create_job_stores_in_store(self):
        """Test that created job is stored and retrievable."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)
        retrieved_job = get_job(job.id)

        assert retrieved_job.id == job.id
        assert retrieved_job.status == job.status

    def test_create_job_utc_timestamps(self):
        """Test that timestamps are UTC-aware."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)

        assert job.created_at.tzinfo is not None
        assert job.updated_at.tzinfo is not None
        assert job.created_at.tzinfo.utcoffset(None) == timedelta(0)

    def test_create_multiple_jobs_unique_ids(self):
        """Test that multiple jobs get unique IDs."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job1 = create_job(request)
        job2 = create_job(request)
        job3 = create_job(request)

        assert job1.id != job2.id
        assert job1.id != job3.id
        assert job2.id != job3.id


class TestGetJob:
    """Tests for get_job function."""

    def test_get_existing_job(self):
        """Test retrieving an existing job."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        created_job = create_job(request)
        retrieved_job = get_job(created_job.id)

        assert retrieved_job.id == created_job.id
        assert retrieved_job.status == created_job.status
        assert retrieved_job.request == created_job.request

    def test_get_nonexistent_job_raises_error(self):
        """Test that getting a nonexistent job raises JobNotFoundError."""
        fake_id = uuid4()

        with pytest.raises(JobNotFoundError) as exc_info:
            get_job(fake_id)

        assert str(fake_id) in str(exc_info.value)

    def test_get_job_does_not_mutate_store(self):
        """Test that getting a job doesn't modify the store."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)
        original_updated_at = job.updated_at

        # Small delay to ensure time difference would be detectable
        time.sleep(0.001)

        get_job(job.id)
        retrieved_again = get_job(job.id)

        assert retrieved_again.updated_at == original_updated_at


class TestUpdateJob:
    """Tests for update_job function."""

    def test_update_job_status(self):
        """Test updating job status."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)
        original_updated_at = job.updated_at

        # Small delay to ensure timestamp changes
        time.sleep(0.001)

        updated_job = update_job(job.id, status=JobStatus.RUNNING)

        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.updated_at > original_updated_at
        assert updated_job.created_at == job.created_at

    def test_update_job_result(self):
        """Test updating job with result."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)

        clarified_spec = ClarifiedSpec(purpose="Test", vision="Test vision")
        result = ClarifiedPlan(specs=[clarified_spec])

        updated_job = update_job(job.id, status=JobStatus.SUCCESS, result=result)

        assert updated_job.status == JobStatus.SUCCESS
        assert updated_job.result == result
        assert updated_job.last_error is None

    def test_update_job_error(self):
        """Test updating job with error."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)

        error_msg = "Processing failed due to network error"
        updated_job = update_job(job.id, status=JobStatus.FAILED, last_error=error_msg)

        assert updated_job.status == JobStatus.FAILED
        assert updated_job.last_error == error_msg

    def test_update_nonexistent_job_raises_error(self):
        """Test that updating a nonexistent job raises JobNotFoundError."""
        fake_id = uuid4()

        with pytest.raises(JobNotFoundError) as exc_info:
            update_job(fake_id, status=JobStatus.RUNNING)

        assert str(fake_id) in str(exc_info.value)

    def test_update_job_always_updates_timestamp(self):
        """Test that update always refreshes updated_at."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)
        original_updated_at = job.updated_at

        # Small delay to ensure timestamp changes
        time.sleep(0.001)

        # Update without changing any data
        updated_job = update_job(job.id)

        assert updated_job.updated_at > original_updated_at

    def test_update_job_persists_changes(self):
        """Test that job updates persist in the store."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)
        update_job(job.id, status=JobStatus.RUNNING)

        retrieved_job = get_job(job.id)

        assert retrieved_job.status == JobStatus.RUNNING


class TestListJobs:
    """Tests for list_jobs function."""

    def test_list_all_jobs(self):
        """Test listing all jobs."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job1 = create_job(request)
        job2 = create_job(request)
        job3 = create_job(request)

        jobs = list_jobs()

        assert len(jobs) == 3
        job_ids = {job.id for job in jobs}
        assert job1.id in job_ids
        assert job2.id in job_ids
        assert job3.id in job_ids

    def test_list_jobs_filtered_by_status(self):
        """Test listing jobs filtered by status."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job1 = create_job(request)
        job2 = create_job(request)
        job3 = create_job(request)

        update_job(job1.id, status=JobStatus.RUNNING)
        update_job(job2.id, status=JobStatus.SUCCESS)
        # job3 stays PENDING

        pending_jobs = list_jobs(status=JobStatus.PENDING)
        running_jobs = list_jobs(status=JobStatus.RUNNING)
        success_jobs = list_jobs(status=JobStatus.SUCCESS)

        assert len(pending_jobs) == 1
        assert pending_jobs[0].id == job3.id
        assert len(running_jobs) == 1
        assert running_jobs[0].id == job1.id
        assert len(success_jobs) == 1
        assert success_jobs[0].id == job2.id

    def test_list_jobs_with_limit(self):
        """Test listing jobs with a limit."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        for _ in range(10):
            create_job(request)

        jobs = list_jobs(limit=5)

        assert len(jobs) == 5

    def test_list_jobs_sorted_by_created_at_desc(self):
        """Test that jobs are sorted by creation time descending."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job1 = create_job(request)
        time.sleep(0.001)
        job2 = create_job(request)
        time.sleep(0.001)
        job3 = create_job(request)

        jobs = list_jobs()

        # Should be in reverse chronological order (newest first)
        assert jobs[0].id == job3.id
        assert jobs[1].id == job2.id
        assert jobs[2].id == job1.id

    def test_list_jobs_empty_store(self):
        """Test listing jobs when store is empty."""
        jobs = list_jobs()

        assert jobs == []


class TestDeleteJob:
    """Tests for delete_job function."""

    def test_delete_existing_job(self):
        """Test deleting an existing job."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)

        delete_job(job.id)

        with pytest.raises(JobNotFoundError):
            get_job(job.id)

    def test_delete_nonexistent_job_raises_error(self):
        """Test that deleting a nonexistent job raises JobNotFoundError."""
        fake_id = uuid4()

        with pytest.raises(JobNotFoundError) as exc_info:
            delete_job(fake_id)

        assert str(fake_id) in str(exc_info.value)

    def test_delete_job_removes_from_list(self):
        """Test that deleted job is removed from list."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job1 = create_job(request)
        job2 = create_job(request)

        delete_job(job1.id)

        jobs = list_jobs()

        assert len(jobs) == 1
        assert jobs[0].id == job2.id


class TestCleanupExpiredJobs:
    """Tests for cleanup_expired_jobs function."""

    def test_cleanup_expired_completed_jobs(self):
        """Test cleaning up expired completed jobs."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create old completed job
        old_job = create_job(request)
        # Manually update to make it old and completed
        with job_store._store_lock:
            old_time = datetime.now(UTC) - timedelta(hours=25)
            job_store._job_store[old_job.id] = old_job.model_copy(update={
                "status": JobStatus.SUCCESS,
                "updated_at": old_time
            })

        # Create recent completed job
        recent_job = create_job(request)
        update_job(recent_job.id, status=JobStatus.SUCCESS)

        # Cleanup with 24 hour TTL
        cleanup_count = cleanup_expired_jobs(ttl_seconds=24 * 60 * 60)

        assert cleanup_count == 1

        # Old job should be gone
        with pytest.raises(JobNotFoundError):
            get_job(old_job.id)

        # Recent job should remain
        assert get_job(recent_job.id) is not None

    def test_cleanup_never_removes_running_jobs(self):
        """Test that running jobs are never cleaned up regardless of age."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create old running job
        old_job = create_job(request)
        update_job(old_job.id, status=JobStatus.RUNNING)

        # Manually make it old
        with job_store._store_lock:
            old_time = datetime.now(UTC) - timedelta(hours=25)
            job_store._job_store[old_job.id] = job_store._job_store[old_job.id].model_copy(update={
                "updated_at": old_time
            })

        cleanup_count = cleanup_expired_jobs(ttl_seconds=24 * 60 * 60)

        assert cleanup_count == 0
        assert get_job(old_job.id) is not None

    def test_cleanup_removes_failed_jobs(self):
        """Test that expired failed jobs are cleaned up."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create old failed job
        old_job = create_job(request)
        update_job(old_job.id, status=JobStatus.FAILED, last_error="Test error")

        # Manually make it old
        with job_store._store_lock:
            old_time = datetime.now(UTC) - timedelta(hours=25)
            job_store._job_store[old_job.id] = job_store._job_store[old_job.id].model_copy(update={
                "updated_at": old_time
            })

        cleanup_count = cleanup_expired_jobs(ttl_seconds=24 * 60 * 60)

        assert cleanup_count == 1

        with pytest.raises(JobNotFoundError):
            get_job(old_job.id)

    def test_cleanup_never_removes_pending_jobs(self):
        """Test that pending jobs are never cleaned up."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create old pending job
        old_job = create_job(request)

        # Manually make it old
        with job_store._store_lock:
            old_time = datetime.now(UTC) - timedelta(hours=25)
            job_store._job_store[old_job.id] = job_store._job_store[old_job.id].model_copy(update={
                "updated_at": old_time
            })

        cleanup_count = cleanup_expired_jobs(ttl_seconds=24 * 60 * 60)

        assert cleanup_count == 0
        assert get_job(old_job.id) is not None

    def test_cleanup_custom_ttl(self):
        """Test cleanup with custom TTL."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create job that's 2 hours old
        job = create_job(request)
        update_job(job.id, status=JobStatus.SUCCESS)

        with job_store._store_lock:
            old_time = datetime.now(UTC) - timedelta(hours=2)
            job_store._job_store[job.id] = job_store._job_store[job.id].model_copy(update={
                "updated_at": old_time
            })

        # Cleanup with 1 hour TTL should remove it
        cleanup_count = cleanup_expired_jobs(ttl_seconds=60 * 60)

        assert cleanup_count == 1

    def test_cleanup_empty_store(self):
        """Test cleanup on empty store."""
        cleanup_count = cleanup_expired_jobs()

        assert cleanup_count == 0

    def test_cleanup_stale_pending_jobs(self):
        """Test cleaning up stale PENDING jobs with optional parameter."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create old PENDING job (48 hours old)
        old_pending = create_job(request)
        with job_store._store_lock:
            old_time = datetime.now(UTC) - timedelta(hours=48)
            job_store._job_store[old_pending.id] = job_store._job_store[old_pending.id].model_copy(update={
                "created_at": old_time,
                "updated_at": old_time
            })

        # Create recent PENDING job (1 hour old)
        recent_pending = create_job(request)
        with job_store._store_lock:
            recent_time = datetime.now(UTC) - timedelta(hours=1)
            job_store._job_store[recent_pending.id] = job_store._job_store[recent_pending.id].model_copy(update={
                "created_at": recent_time,
                "updated_at": recent_time
            })

        # Cleanup with stale_pending_ttl_seconds set to 24 hours
        cleanup_count = cleanup_expired_jobs(
            ttl_seconds=24 * 60 * 60,
            stale_pending_ttl_seconds=24 * 60 * 60
        )

        assert cleanup_count == 1

        # Old PENDING job should be gone
        with pytest.raises(JobNotFoundError):
            get_job(old_pending.id)

        # Recent PENDING job should remain
        assert get_job(recent_pending.id) is not None

    def test_cleanup_without_stale_pending_parameter(self):
        """Test that PENDING jobs are not cleaned up without stale_pending_ttl_seconds."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create very old PENDING job
        old_pending = create_job(request)
        with job_store._store_lock:
            old_time = datetime.now(UTC) - timedelta(hours=100)
            job_store._job_store[old_pending.id] = job_store._job_store[old_pending.id].model_copy(update={
                "created_at": old_time,
                "updated_at": old_time
            })

        # Cleanup without stale_pending_ttl_seconds
        cleanup_count = cleanup_expired_jobs(ttl_seconds=24 * 60 * 60)

        assert cleanup_count == 0

        # Old PENDING job should still exist
        assert get_job(old_pending.id) is not None


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_create_jobs(self):
        """Test creating jobs concurrently."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        created_jobs = []

        def create_jobs_worker():
            for _ in range(10):
                job = create_job(request)
                created_jobs.append(job)

        threads = [threading.Thread(target=create_jobs_worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have created 50 jobs total
        assert len(created_jobs) == 50

        # All IDs should be unique
        job_ids = [job.id for job in created_jobs]
        assert len(set(job_ids)) == 50

        # All jobs should be retrievable
        all_jobs = list_jobs()
        assert len(all_jobs) == 50

    def test_concurrent_update_same_job(self):
        """Test updating the same job concurrently."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)

        def update_worker(status):
            for _ in range(10):
                try:
                    update_job(job.id, status=status)
                    time.sleep(0.001)
                except JobNotFoundError:
                    # Job might have been deleted in another test scenario
                    pass

        threads = [
            threading.Thread(target=update_worker, args=(JobStatus.RUNNING,)),
            threading.Thread(target=update_worker, args=(JobStatus.SUCCESS,)),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Job should still exist and be in a valid state
        final_job = get_job(job.id)
        assert final_job.status in (JobStatus.RUNNING, JobStatus.SUCCESS)

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)
        read_count = [0]

        def reader():
            for _ in range(20):
                try:
                    get_job(job.id)
                    read_count[0] += 1
                except JobNotFoundError:
                    pass

        def writer():
            for i in range(10):
                update_job(job.id, status=JobStatus.RUNNING)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have completed all reads successfully
        assert read_count[0] > 0

        # Job should still exist and be valid
        final_job = get_job(job.id)
        assert final_job.status == JobStatus.RUNNING


class TestDeepCopy:
    """Tests for deep copy behavior to ensure thread-safety."""

    def test_get_job_returns_deep_copy(self):
        """Test that get_job returns a deep copy that can be mutated safely."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request, config={"key": "value"})

        # Get the job
        retrieved_job = get_job(job.id)

        # Mutate the retrieved job's config
        if retrieved_job.config:
            retrieved_job.config["key"] = "modified"

        # Get the job again - should have original config
        job_again = get_job(job.id)
        assert job_again.config == {"key": "value"}

    def test_list_jobs_returns_deep_copies(self):
        """Test that list_jobs returns deep copies that can be mutated safely."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request, config={"key": "value"})

        # List jobs
        jobs = list_jobs()
        assert len(jobs) == 1

        # Mutate the first job's config
        if jobs[0].config:
            jobs[0].config["key"] = "modified"

        # List jobs again - should have original config
        jobs_again = list_jobs()
        assert jobs_again[0].config == {"key": "value"}

    def test_get_job_nested_objects_are_deep_copied(self):
        """Test that nested objects in job are properly deep copied."""
        spec = SpecInput(purpose="Test", vision="Test vision", must=["Feature 1"])
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        job = create_job(request)

        # Get the job
        retrieved_job = get_job(job.id)

        # Mutate the retrieved job's nested request data
        retrieved_job.request.plan.specs[0].must.append("Feature 2")

        # Get the job again - should have original request
        job_again = get_job(job.id)
        assert len(job_again.request.plan.specs[0].must) == 1
        assert job_again.request.plan.specs[0].must[0] == "Feature 1"


class TestClearAllJobs:
    """Tests for clear_all_jobs function."""

    def test_clear_all_jobs(self):
        """Test clearing all jobs from the store."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)

        # Create multiple jobs
        for _ in range(5):
            create_job(request)

        assert len(list_jobs()) == 5

        clear_all_jobs()

        assert len(list_jobs()) == 0

    def test_clear_empty_store(self):
        """Test clearing an already empty store."""
        clear_all_jobs()

        assert len(list_jobs()) == 0

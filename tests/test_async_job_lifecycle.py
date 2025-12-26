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
"""Tests for async job lifecycle functions."""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from app.models.specs import ClarificationRequest, JobStatus, PlanInput, SpecInput
from app.services.clarification import process_clarification_job, start_clarification_job
from app.services.job_store import JobNotFoundError, clear_all_jobs, get_job


@pytest.fixture(autouse=True)
def clean_job_store():
    """Clean the job store before and after each test."""
    clear_all_jobs()
    yield
    clear_all_jobs()


class TestStartClarificationJob:
    """Tests for start_clarification_job function."""
    
    def test_start_job_creates_pending_job(self):
        """Test that starting a job creates it with PENDING status."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        assert job.status == JobStatus.PENDING
        assert job.request == request
        assert job.result is None
        assert job.last_error is None
    
    def test_start_job_stores_in_job_store(self):
        """Test that the job is stored and retrievable."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Job should be retrievable from store
        retrieved_job = get_job(job.id)
        assert retrieved_job.id == job.id
        assert retrieved_job.status == JobStatus.PENDING
    
    def test_start_job_schedules_background_task(self):
        """Test that background processing is scheduled."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Background task should be added
        background_tasks.add_task.assert_called_once()
        # First argument should be the processing function
        assert background_tasks.add_task.call_args[0][0] == process_clarification_job
        # Second argument should be the job ID
        assert background_tasks.add_task.call_args[0][1] == job.id
    
    def test_start_job_with_config(self):
        """Test starting a job with configuration."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        config = {"model": "gpt-4", "temperature": 0.7}
        
        job = start_clarification_job(request, background_tasks, config=config)
        
        assert job.config == config
    
    def test_start_job_returns_immediately(self):
        """Test that start_clarification_job returns without blocking."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Should return immediately without waiting for processing
        job = start_clarification_job(request, background_tasks)
        
        # Job should still be PENDING (not processed yet)
        assert job.status == JobStatus.PENDING


class TestProcessClarificationJob:
    """Tests for process_clarification_job function."""
    
    def test_process_job_success(self):
        """Test successful job processing."""
        spec = SpecInput(
            purpose="Test",
            vision="Test vision",
            must=["Feature 1"],
            open_questions=["Q1", "Q2"]
        )
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Start the job
        job = start_clarification_job(request, background_tasks)
        
        # Process it directly
        process_clarification_job(job.id)
        
        # Check the job status
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result is not None
        assert len(processed_job.result.specs) == 1
        assert processed_job.result.specs[0].purpose == "Test"
        assert processed_job.result.specs[0].must == ["Feature 1"]
        assert processed_job.last_error is None
    
    def test_process_job_transitions_through_running(self):
        """Test that job transitions through RUNNING status."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Mock update_job to capture status changes
        import app.services.job_store as job_store_module
        with patch.object(job_store_module, 'update_job', 
                         wraps=job_store_module.update_job) as mock_update:
            process_clarification_job(job.id)
            
            # Should have been called at least twice (RUNNING, then SUCCESS)
            assert mock_update.call_count >= 2
            
            # First call should be RUNNING
            first_call = mock_update.call_args_list[0]
            assert first_call[1]['status'] == JobStatus.RUNNING
            
            # Last call should be SUCCESS
            last_call = mock_update.call_args_list[-1]
            assert last_call[1]['status'] == JobStatus.SUCCESS
    
    def test_process_job_updates_timestamps(self):
        """Test that updated_at is refreshed during processing."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        original_updated_at = job.updated_at
        
        # Process the job
        process_clarification_job(job.id)
        
        # Check that updated_at changed
        processed_job = get_job(job.id)
        assert processed_job.updated_at > original_updated_at
    
    def test_process_job_with_unknown_job_id(self):
        """Test processing with an unknown job ID returns cleanly."""
        fake_id = uuid4()
        
        # Should not raise an exception
        process_clarification_job(fake_id)
        
        # No job should exist
        with pytest.raises(JobNotFoundError):
            get_job(fake_id)
    
    def test_process_job_exception_handling(self):
        """Test that exceptions during processing mark job as FAILED."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Mock clarify_plan to raise an exception
        with patch('app.services.clarification.clarify_plan') as mock_clarify:
            mock_clarify.side_effect = ValueError("Test error")
            
            # Process should not raise
            process_clarification_job(job.id)
        
        # Job should be FAILED
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.last_error is not None
        assert "ValueError" in failed_job.last_error
        assert "Test error" in failed_job.last_error
        assert failed_job.result is None
    
    def test_process_job_clears_result_on_failure(self):
        """Test that result is cleared when job fails."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Mock clarify_plan to raise exception
        with patch('app.services.clarification.clarify_plan') as mock_clarify:
            mock_clarify.side_effect = RuntimeError("Processing failed")
            
            process_clarification_job(job.id)
        
        # Result should be None (cleared)
        failed_job = get_job(job.id)
        assert failed_job.result is None
        assert failed_job.last_error is not None
    
    def test_process_job_multiple_specs(self):
        """Test processing a plan with multiple specs."""
        spec1 = SpecInput(purpose="Frontend", vision="UI")
        spec2 = SpecInput(purpose="Backend", vision="API")
        spec3 = SpecInput(purpose="DevOps", vision="CI/CD")
        plan = PlanInput(specs=[spec1, spec2, spec3])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        process_clarification_job(job.id)
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert len(processed_job.result.specs) == 3
        assert processed_job.result.specs[0].purpose == "Frontend"
        assert processed_job.result.specs[1].purpose == "Backend"
        assert processed_job.result.specs[2].purpose == "DevOps"
    
    def test_process_job_preserves_spec_fields(self):
        """Test that all spec fields are preserved during processing."""
        spec = SpecInput(
            purpose="Test Service",
            vision="High performance",
            must=["Fast", "Reliable"],
            dont=["Slow", "Complex"],
            nice=["Configurable", "Extensible"],
            assumptions=["Cloud deployment", "High bandwidth"],
            open_questions=["Which cloud?", "What latency?"]
        )
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        process_clarification_job(job.id)
        
        processed_job = get_job(job.id)
        result_spec = processed_job.result.specs[0]
        
        assert result_spec.purpose == "Test Service"
        assert result_spec.vision == "High performance"
        assert result_spec.must == ["Fast", "Reliable"]
        assert result_spec.dont == ["Slow", "Complex"]
        assert result_spec.nice == ["Configurable", "Extensible"]
        assert result_spec.assumptions == ["Cloud deployment", "High bandwidth"]
        # open_questions should not exist in ClarifiedSpec
        assert not hasattr(result_spec, "open_questions")
    
    def test_process_job_can_be_called_directly(self):
        """Test that process_clarification_job can be invoked directly for testing."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Clear the mock to verify direct invocation
        background_tasks.reset_mock()
        
        # Call process directly (not via background tasks)
        process_clarification_job(job.id)
        
        # Should have processed successfully
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        
        # Background tasks should not have been called during direct invocation
        background_tasks.add_task.assert_not_called()
    
    def test_process_job_exception_in_error_handling(self):
        """Test that exceptions during error handling don't crash the worker."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Mock to cause exception, then another exception when trying to update
        import app.services.job_store as job_store_module
        with patch('app.services.clarification.clarify_plan') as mock_clarify, \
             patch.object(job_store_module, 'update_job') as mock_update:
            
            mock_clarify.side_effect = ValueError("First error")
            
            # Make update_job fail when marking as FAILED (but succeed for RUNNING)
            def update_side_effect(job_id, **kwargs):
                if kwargs.get('status') == JobStatus.FAILED:
                    raise RuntimeError("Update failed")
                # For RUNNING status, call the real function
                return job_store_module.update_job(job_id, **kwargs)
            
            mock_update.side_effect = update_side_effect
            
            # Should not raise an exception
            process_clarification_job(job.id)


class TestAsyncJobLifecycleEdgeCases:
    """Tests for edge cases in async job lifecycle."""
    
    def test_job_deleted_during_processing(self):
        """Test handling of job deletion during processing."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Delete the job before processing completes
        from app.services.job_store import delete_job
        
        # Mock clarify_plan to delete job mid-processing
        with patch('app.services.clarification.clarify_plan') as mock_clarify:
            def delete_during_processing(plan):
                delete_job(job.id)
                raise ValueError("Job was deleted")
            
            mock_clarify.side_effect = delete_during_processing
            
            # Should handle gracefully without crashing
            process_clarification_job(job.id)
            
            # Job should not exist
            with pytest.raises(JobNotFoundError):
                get_job(job.id)
    
    def test_concurrent_job_creation(self):
        """Test creating multiple jobs concurrently."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create multiple jobs
        jobs = [start_clarification_job(request, background_tasks) for _ in range(5)]
        
        # All should have unique IDs
        job_ids = [job.id for job in jobs]
        assert len(set(job_ids)) == 5
        
        # All should be PENDING
        for job in jobs:
            assert job.status == JobStatus.PENDING
        
        # All should be scheduled for background processing
        assert background_tasks.add_task.call_count == 5
    
    def test_process_same_job_multiple_times(self):
        """Test that processing the same job multiple times doesn't cause issues."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Process the same job multiple times
        process_clarification_job(job.id)
        process_clarification_job(job.id)
        process_clarification_job(job.id)
        
        # Should end in SUCCESS state (second and third calls should skip)
        final_job = get_job(job.id)
        assert final_job.status == JobStatus.SUCCESS
        assert final_job.result is not None
    
    def test_process_job_skips_non_pending_status(self):
        """Test that process_clarification_job skips jobs not in PENDING state."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Manually set job to RUNNING
        from app.services.job_store import update_job
        update_job(job.id, status=JobStatus.RUNNING)
        
        # Try to process - should skip
        process_clarification_job(job.id)
        
        # Job should still be in RUNNING state (not processed)
        final_job = get_job(job.id)
        assert final_job.status == JobStatus.RUNNING
        assert final_job.result is None

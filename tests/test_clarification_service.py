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
"""Tests for clarification service."""

from unittest.mock import MagicMock, patch

import pytest

from app.models.specs import JobStatus, PlanInput, SpecInput
from app.services.clarification import clarify_plan, process_clarification_job, start_clarification_job
from app.services.job_store import clear_all_jobs, get_job
from app.services.llm_clients import ClarificationLLMConfig, DummyLLMClient


class TestClarifyPlan:
    """Tests for the clarify_plan service function."""
    
    def test_clarify_plan_single_spec(self):
        """Test clarifying a plan with a single spec."""
        spec = SpecInput(
            purpose="Build a web app",
            vision="Modern and user-friendly",
            must=["Authentication", "Database"],
            dont=["Complex UI"],
            nice=["Dark mode"],
            open_questions=["What database?", "Which auth provider?"],
            assumptions=["Users have modern browsers"],
        )
        plan = PlanInput(specs=[spec])
        
        result = clarify_plan(plan)
        
        assert len(result.specs) == 1
        clarified = result.specs[0]
        
        # Check required fields are copied
        assert clarified.purpose == "Build a web app"
        assert clarified.vision == "Modern and user-friendly"
        assert clarified.must == ["Authentication", "Database"]
        assert clarified.dont == ["Complex UI"]
        assert clarified.nice == ["Dark mode"]
        assert clarified.assumptions == ["Users have modern browsers"]
        
        # Verify open_questions field does not exist in ClarifiedSpec
        assert not hasattr(clarified, "open_questions")
    
    def test_clarify_plan_multiple_specs(self):
        """Test clarifying a plan with multiple specs."""
        spec1 = SpecInput(
            purpose="Frontend",
            vision="Responsive UI",
            must=["React"],
        )
        spec2 = SpecInput(
            purpose="Backend",
            vision="Scalable API",
            must=["FastAPI", "PostgreSQL"],
        )
        spec3 = SpecInput(
            purpose="DevOps",
            vision="Automated deployment",
            nice=["CI/CD"],
        )
        
        plan = PlanInput(specs=[spec1, spec2, spec3])
        
        result = clarify_plan(plan)
        
        assert len(result.specs) == 3
        assert result.specs[0].purpose == "Frontend"
        assert result.specs[1].purpose == "Backend"
        assert result.specs[2].purpose == "DevOps"
    
    def test_clarify_plan_preserves_order(self):
        """Test that spec order is preserved deterministically."""
        specs = [
            SpecInput(purpose=f"Spec {i}", vision=f"Vision {i}")
            for i in range(10)
        ]
        plan = PlanInput(specs=specs)
        
        result = clarify_plan(plan)
        
        for i, clarified in enumerate(result.specs):
            assert clarified.purpose == f"Spec {i}"
            assert clarified.vision == f"Vision {i}"
    
    def test_clarify_plan_with_empty_lists(self):
        """Test clarifying specs with empty lists."""
        spec = SpecInput(
            purpose="Test",
            vision="Test vision",
            must=[],
            dont=[],
            nice=[],
            open_questions=[],
            assumptions=[],
        )
        plan = PlanInput(specs=[spec])
        
        result = clarify_plan(plan)
        
        clarified = result.specs[0]
        assert clarified.must == []
        assert clarified.dont == []
        assert clarified.nice == []
        assert clarified.assumptions == []
    
    def test_clarify_plan_empty_specs_list(self):
        """Test clarifying a plan with no specs."""
        plan = PlanInput(specs=[])
        
        result = clarify_plan(plan)
        
        assert result.specs == []
    
    async def test_clarify_plan_ignores_open_questions(self):
        """Test that open_questions are not copied to clarified spec."""
        spec = SpecInput(
            purpose="Test",
            vision="Test vision",
            open_questions=["Question 1", "Question 2", "Question 3"],
        )
        plan = PlanInput(specs=[spec])
        
        result = clarify_plan(plan)
        
        # ClarifiedSpec should not have open_questions
        clarified = result.specs[0]
        assert not hasattr(clarified, "open_questions")
    
    async def test_clarify_plan_list_independence(self):
        """Test that list modifications don't affect original."""
        must_list = ["Feature 1", "Feature 2"]
        spec = SpecInput(
            purpose="Test",
            vision="Test vision",
            must=must_list,
        )
        plan = PlanInput(specs=[spec])
        
        result = clarify_plan(plan)
        
        # Modify the clarified spec's list
        result.specs[0].must.append("Feature 3")
        
        # Original should be unchanged
        assert len(plan.specs[0].must) == 2
        assert len(result.specs[0].must) == 3
    
    async def test_clarify_plan_with_unicode_and_special_chars(self):
        """Test clarifying specs with unicode and special characters."""
        spec = SpecInput(
            purpose="Système de gestion 系统管理",
            vision="🚀 Modern & scalable <system>",
            must=["UTF-8 支持", "Émojis 👍"],
            dont=["Legacy encoding"],
            nice=["Multi-language ⚡"],
            assumptions=["Unicode everywhere"],
        )
        plan = PlanInput(specs=[spec])
        
        result = clarify_plan(plan)
        
        clarified = result.specs[0]
        assert clarified.purpose == "Système de gestion 系统管理"
        assert clarified.vision == "🚀 Modern & scalable <system>"
        assert "UTF-8 支持" in clarified.must
        assert "Émojis 👍" in clarified.must


@pytest.fixture(autouse=True)
def clean_job_store_for_service_tests():
    """Clean the job store before and after each async test."""
    clear_all_jobs()
    yield
    clear_all_jobs()


class TestProcessClarificationJobService:
    """Tests for async job processing in the service layer."""
    
    async def test_job_status_transitions_pending_to_running_to_success(self):
        """Test that job transitions through PENDING -> RUNNING -> SUCCESS with timestamps."""
        spec = SpecInput(purpose="Test", vision="Test vision", must=["Feature 1"])
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create job
        job = start_clarification_job(request, background_tasks)
        assert job.status == JobStatus.PENDING
        original_created_at = job.created_at
        original_updated_at = job.updated_at
        
        # Track state transitions to verify RUNNING state occurs
        from app.services import job_store
        states_observed = []
        original_update = job_store.update_job
        
        def track_updates(job_id, **kwargs):
            if 'status' in kwargs:
                states_observed.append(kwargs['status'])
            return original_update(job_id, **kwargs)
        
        with patch.object(job_store, 'update_job', side_effect=track_updates):
            # Process job (manually invoke)
            await process_clarification_job(job.id)
        
        # Verify we observed RUNNING state during processing
        assert JobStatus.RUNNING in states_observed
        assert JobStatus.SUCCESS in states_observed
        
        # Verify final state
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result is not None
        assert len(processed_job.result.specs) == 1
        assert processed_job.result.specs[0].purpose == "Test"
        
        # Verify timestamps
        assert processed_job.created_at == original_created_at
        assert processed_job.updated_at > original_updated_at
        assert processed_job.last_error is None
    
    async def test_job_processing_failure_sets_failed_status_and_error(self):
        """Test that exceptions during processing mark job as FAILED with error message."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create job
        job = start_clarification_job(request, background_tasks)
        
        # Mock clarify_plan to raise an exception
        with patch('app.services.clarification.clarify_plan') as mock_clarify:
            mock_clarify.side_effect = ValueError("Simulated processing error")
            
            # Process should handle exception gracefully
            await process_clarification_job(job.id)
            
            mock_clarify.assert_called_once()
        
        # Verify job is marked as FAILED
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.last_error is not None
        assert "ValueError" in failed_job.last_error
        assert "Simulated processing error" in failed_job.last_error
        assert failed_job.result is None
    
    async def test_job_processing_updates_timestamps_on_each_state_change(self):
        """Test that timestamps are updated during state transitions."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        initial_updated_at = job.updated_at
        
        # Process the job
        await process_clarification_job(job.id)
        
        # Check updated_at changed
        final_job = get_job(job.id)
        assert final_job.updated_at > initial_updated_at
    
    async def test_manual_job_invocation_for_testing(self):
        """Test that process_clarification_job can be invoked directly for testing."""
        spec = SpecInput(
            purpose="Test Service",
            vision="High performance",
            must=["Fast", "Reliable"],
            dont=["Slow"],
            nice=["Configurable"],
            assumptions=["Cloud deployment"]
        )
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create job
        job = start_clarification_job(request, background_tasks)
        
        # Manually invoke processing (deterministic for testing)
        await process_clarification_job(job.id)
        
        # Verify successful processing
        result = get_job(job.id)
        assert result.status == JobStatus.SUCCESS
        assert result.result is not None
        assert result.result.specs[0].purpose == "Test Service"
        assert result.result.specs[0].must == ["Fast", "Reliable"]


class TestClarificationServiceWithLLMConfig:
    """Tests for clarification service with LLM configuration wiring."""
    
    async def test_service_accepts_llm_config_without_invoking_llm(self):
        """Test that service accepts LLM config but doesn't invoke it yet."""
        spec = SpecInput(purpose="Test", vision="Test vision", must=["Feature"])
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create LLM config for OpenAI
        llm_config = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Start job with LLM config
        job = start_clarification_job(request, background_tasks, llm_config=llm_config)
        assert job.status == JobStatus.PENDING
        
        # Process the job - should succeed with deterministic output
        await process_clarification_job(job.id)
        
        # Verify job completed successfully
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result is not None
        assert processed_job.result.specs[0].purpose == "Test"
    
    async def test_service_operates_without_llm_config_preserving_old_behavior(self):
        """Test that service works without LLM config (backward compatibility)."""
        spec = SpecInput(purpose="Legacy Test", vision="Old behavior")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Start job without LLM config (None is default)
        job = start_clarification_job(request, background_tasks, llm_config=None)
        
        # Process without LLM config
        await process_clarification_job(job.id)
        
        # Should complete successfully with deterministic behavior
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result.specs[0].purpose == "Legacy Test"
    
    async def test_service_with_dummy_llm_client_dependency_injection(self):
        """Test dependency injection with DummyLLMClient for testing."""
        spec = SpecInput(purpose="DI Test", vision="Dependency injection")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create a custom DummyLLMClient
        dummy_client = DummyLLMClient(canned_response='{"test": "injected"}')
        
        # Start job
        job = start_clarification_job(request, background_tasks)
        
        # Process with injected dummy client (LLM not invoked yet)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Should complete successfully
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
    
    async def test_service_initializes_llm_client_from_stored_config(self):
        """Test that service initializes LLM client from stored config."""
        spec = SpecInput(purpose="Config Test", vision="From storage")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create config with valid provider
        llm_config = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1"
        )
        
        job = start_clarification_job(request, background_tasks, llm_config=llm_config)
        
        # Verify config was stored
        assert 'llm_config' in job.config
        
        # Process job - should initialize client from stored config
        # Mock factory to return dummy client (avoid needing real API keys)
        with patch('app.services.clarification.get_llm_client') as mock_factory:
            mock_factory.return_value = DummyLLMClient()
            await process_clarification_job(job.id)
            # Verify factory was called
            mock_factory.assert_called_once()
        
        # Job should complete successfully
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
    
    async def test_service_handles_llm_client_initialization_failure_gracefully(self):
        """Test that service continues with deterministic behavior if LLM init fails."""
        spec = SpecInput(purpose="Fallback Test", vision="Error handling")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create config with valid provider
        llm_config = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1"
        )
        
        job = start_clarification_job(request, background_tasks, llm_config=llm_config)
        
        # Mock get_llm_client to raise an error
        with patch('app.services.clarification.get_llm_client') as mock_factory:
            mock_factory.side_effect = ValueError("Client initialization failed")
            
            # Process should handle error gracefully and continue
            await process_clarification_job(job.id)
        
        # Job should still complete successfully with deterministic output
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result.specs[0].purpose == "Fallback Test"
    
    async def test_service_does_not_invoke_llm_client_yet(self):
        """Test that LLM client is initialized but complete() is never called."""
        spec = SpecInput(purpose="No Invocation", vision="Client not called")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create a mock client to track method calls
        mock_client = MagicMock()
        mock_client.complete = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Inject mock client and process
        await process_clarification_job(job.id, llm_client=mock_client)
        
        # Verify complete was NEVER called
        mock_client.complete.assert_not_called()
        
        # Job should complete with deterministic behavior
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
    
    async def test_service_with_anthropic_config(self):
        """Test service with Anthropic LLM configuration."""
        spec = SpecInput(purpose="Anthropic Test", vision="Claude model")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        llm_config = ClarificationLLMConfig(
            provider="anthropic",
            model="claude-sonnet-4.5",
            temperature=0.2
        )
        
        job = start_clarification_job(request, background_tasks, llm_config=llm_config)
        
        # Mock factory to avoid needing real API key
        with patch('app.services.clarification.get_llm_client') as mock_factory:
            mock_factory.return_value = DummyLLMClient()
            await process_clarification_job(job.id)
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
    
    async def test_service_preserves_existing_config_when_adding_llm_config(self):
        """Test that adding llm_config doesn't overwrite existing config."""
        spec = SpecInput(purpose="Config Merge", vision="Preserve old config")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Existing config
        existing_config = {"custom_field": "preserved", "another_key": 123}
        
        llm_config = ClarificationLLMConfig(provider="openai", model="gpt-5.1")
        
        job = start_clarification_job(
            request, 
            background_tasks, 
            config=existing_config,
            llm_config=llm_config
        )
        
        # Verify both configs are present
        assert 'custom_field' in job.config
        assert job.config['custom_field'] == "preserved"
        assert 'llm_config' in job.config
        assert job.config['llm_config']['provider'] == "openai"
    
    async def test_service_llm_config_stored_as_dict(self):
        """Test that LLM config is properly serialized to dict for storage."""
        spec = SpecInput(purpose="Serialization Test", vision="Dict storage")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        llm_config = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=0.8,
            max_tokens=2000
        )
        
        job = start_clarification_job(request, background_tasks, llm_config=llm_config)
        
        # Verify stored as dict, not Pydantic model
        stored_config = job.config['llm_config']
        assert isinstance(stored_config, dict)
        assert stored_config['provider'] == "openai"
        assert stored_config['model'] == "gpt-5.1"
        assert stored_config['temperature'] == 0.8
        assert stored_config['max_tokens'] == 2000
    
    async def test_service_reconstructs_llm_config_from_dict(self):
        """Test that service correctly reconstructs ClarificationLLMConfig from dict."""
        spec = SpecInput(purpose="Reconstruction Test", vision="Dict to model")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        llm_config = ClarificationLLMConfig(
            provider="openai",
            model="gpt-5.1",
            temperature=0.5
        )
        
        job = start_clarification_job(request, background_tasks, llm_config=llm_config)
        
        # Track config reconstruction
        reconstructed_configs = []
        
        def capture_config(provider, config):
            reconstructed_configs.append(config)
            return DummyLLMClient()
        
        with patch('app.services.clarification.get_llm_client', side_effect=capture_config):
            await process_clarification_job(job.id)
        
        # Verify config was reconstructed correctly
        assert len(reconstructed_configs) == 1
        reconstructed = reconstructed_configs[0]
        assert isinstance(reconstructed, ClarificationLLMConfig)
        assert reconstructed.provider == "openai"
        assert reconstructed.model == "gpt-5.1"
        assert reconstructed.temperature == 0.5

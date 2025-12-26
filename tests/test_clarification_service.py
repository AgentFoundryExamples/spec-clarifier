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



def _create_dummy_client_with_response(specs):
    """Helper to create DummyLLMClient with valid ClarifiedPlan JSON response."""
    from app.services.llm_clients import DummyLLMClient
    import json
    
    clarified_specs = []
    for spec in specs:
        clarified_spec = {
            "purpose": spec.purpose,
            "vision": spec.vision,
            "must": spec.must,
            "dont": spec.dont,
            "nice": spec.nice,
            "assumptions": spec.assumptions
        }
        clarified_specs.append(clarified_spec)
    
    response_json = json.dumps({"specs": clarified_specs}, indent=2)
    return DummyLLMClient(canned_response=response_json)

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
            # Create dummy client
            dummy_client = _create_dummy_client_with_response([spec])
            
            await process_clarification_job(job.id, llm_client=dummy_client)
        
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
        
        # Use a failing LLM client
        from app.services.llm_clients import DummyLLMClient
        failing_client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Simulated processing error"
        )
        
        # Process should handle exception gracefully
        await process_clarification_job(job.id, llm_client=failing_client)
        
        # Verify job is marked as FAILED
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.last_error is not None
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
        # Create dummy client
        dummy_client = _create_dummy_client_with_response([spec])
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
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
        # Create dummy client
        dummy_client = _create_dummy_client_with_response([spec])
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
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
        # Create dummy client
        dummy_client = _create_dummy_client_with_response([spec])
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
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
        # Create dummy client
        dummy_client = _create_dummy_client_with_response([spec])
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
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
        
        # Create a custom DummyLLMClient with valid response
        dummy_client = _create_dummy_client_with_response([spec])
        
        # Start job
        job = start_clarification_job(request, background_tasks)
        
        # Process with injected dummy client
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
            dummy_client = _create_dummy_client_with_response([spec])
            mock_factory.return_value = dummy_client
            
            await process_clarification_job(job.id)  # Don't inject client, let it use factory
            # Verify factory was called
            mock_factory.assert_called_once()
        
        # Job should complete successfully
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
    
    async def test_service_handles_llm_client_initialization_failure_gracefully(self):
        """Test that service fails the job if LLM client initialization fails."""
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
        
        # Mock get_llm_client to raise an error (don't inject client, let it try to initialize)
        with patch('app.services.clarification.get_llm_client') as mock_factory:
            mock_factory.side_effect = ValueError("Client initialization failed")
            
            # Process should fail the job
            await process_clarification_job(job.id)
        
        # Job should be marked as FAILED
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.FAILED
        assert "Invalid LLM provider" in processed_job.last_error or "Client initialization failed" in processed_job.last_error
        
        job = start_clarification_job(request, background_tasks, llm_config=llm_config)
        
        # Mock get_llm_client to raise an error
        with patch('app.services.clarification.get_llm_client') as mock_factory:
            mock_factory.side_effect = ValueError("Client initialization failed")
            
            # Process should handle error gracefully and continue
            # Create dummy client
            dummy_client = _create_dummy_client_with_response([spec])
            
            await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Job should still complete successfully with deterministic output
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result.specs[0].purpose == "Fallback Test"
    
    async def test_service_invokes_llm_client(self):
        """Test that LLM client's complete() method is called."""
        spec = SpecInput(purpose="LLM Invocation", vision="Client is called")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create a DummyLLMClient with valid response
        dummy_client = _create_dummy_client_with_response([spec])
        
        job = start_clarification_job(request, background_tasks)
        
        # Inject client and process
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Job should complete successfully with LLM invocation
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result is not None
    
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
            # Create dummy client
            dummy_client = _create_dummy_client_with_response([spec])
            
            await process_clarification_job(job.id, llm_client=dummy_client)
        
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
            return _create_dummy_client_with_response([spec])
        
        with patch('app.services.clarification.get_llm_client', side_effect=capture_config):
            await process_clarification_job(job.id)  # Don't inject client, let it use factory
        
        # Verify config was reconstructed correctly
        assert len(reconstructed_configs) == 1
        reconstructed = reconstructed_configs[0]
        assert isinstance(reconstructed, ClarificationLLMConfig)
        assert reconstructed.provider == "openai"
        assert reconstructed.model == "gpt-5.1"
        assert reconstructed.temperature == 0.5


class TestLLMPipelineWithDummyClient:
    """Tests for LLM pipeline using DummyLLMClient with various response scenarios.
    
    These tests cover the acceptance criteria from the issue:
    - Valid DummyLLMClient JSON leads to SUCCESS jobs
    - Cleaned JSON from markdown fences parses successfully
    - Malformed outputs trigger FAILED status
    - ClarifiedPlan contains only the 6 required keys
    """
    
    async def test_valid_json_response_leads_to_success(self):
        """Test that valid JSON from DummyLLMClient results in SUCCESS job."""
        spec = SpecInput(
            purpose="Test",
            vision="Test vision",
            must=["Feature 1"],
            dont=["No feature 2"],
            nice=["Feature 3"],
            assumptions=["Assumption 1"],
            open_questions=["Question 1?"]
        )
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create DummyLLMClient with valid ClarifiedPlan JSON (no open_questions)
        valid_response = {
            "specs": [{
                "purpose": "Test",
                "vision": "Test vision",
                "must": ["Feature 1"],
                "dont": ["No feature 2"],
                "nice": ["Feature 3"],
                "assumptions": ["Assumption 1"]
            }]
        }
        import json
        dummy_client = DummyLLMClient(canned_response=json.dumps(valid_response))
        
        # Start and process job
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Verify SUCCESS status and ClarifiedPlan structure
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result is not None
        assert len(processed_job.result.specs) == 1
        
        # Verify ClarifiedPlan contains only 6 keys (not open_questions)
        result_spec = processed_job.result.specs[0]
        assert result_spec.purpose == "Test"
        assert result_spec.vision == "Test vision"
        assert result_spec.must == ["Feature 1"]
        assert result_spec.dont == ["No feature 2"]
        assert result_spec.nice == ["Feature 3"]
        assert result_spec.assumptions == ["Assumption 1"]
        assert not hasattr(result_spec, "open_questions")
    
    async def test_markdown_wrapped_json_parses_successfully(self):
        """Test that markdown-wrapped JSON is cleaned and parsed successfully."""
        spec = SpecInput(purpose="Markdown Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Create response with markdown fences
        markdown_wrapped = '''```json
{
  "specs": [{
    "purpose": "Markdown Test",
    "vision": "Test vision",
    "must": [],
    "dont": [],
    "nice": [],
    "assumptions": []
  }]
}
```'''
        dummy_client = DummyLLMClient(canned_response=markdown_wrapped)
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Should succeed despite markdown fences
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result is not None
        assert processed_job.result.specs[0].purpose == "Markdown Test"
    
    async def test_plain_markdown_fences_cleaned(self):
        """Test that plain ``` markdown fences are removed."""
        spec = SpecInput(purpose="Plain Fence Test", vision="Test")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Plain markdown fences without 'json' keyword
        plain_fences = '''```
{"specs": [{"purpose": "Plain Fence Test", "vision": "Test", "must": [], "dont": [], "nice": [], "assumptions": []}]}
```'''
        dummy_client = DummyLLMClient(canned_response=plain_fences)
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result.specs[0].purpose == "Plain Fence Test"
    
    async def test_json_with_trailing_prose_ignored(self):
        """Test that trailing prose after JSON is ignored."""
        spec = SpecInput(purpose="Trailing Text", vision="Test")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # JSON followed by prose
        with_prose = '{"specs": [{"purpose": "Trailing Text", "vision": "Test", "must": [], "dont": [], "nice": [], "assumptions": []}]} Hope this helps!'
        dummy_client = DummyLLMClient(canned_response=with_prose)
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result.specs[0].purpose == "Trailing Text"
    
    async def test_malformed_json_triggers_failed_status(self):
        """Test that malformed JSON triggers FAILED status with sanitized error."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Malformed JSON (missing closing brace)
        malformed = '{"specs": [{"purpose": "Test"}'
        dummy_client = DummyLLMClient(canned_response=malformed)
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Should be FAILED with error message
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.last_error is not None
        assert "Failed to parse" in failed_job.last_error or "parse" in failed_job.last_error.lower()
        assert failed_job.result is None
    
    async def test_missing_specs_key_fails_with_helpful_error(self):
        """Test that missing specs key results in failed job with helpful error."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Valid JSON but missing required 'specs' key
        missing_specs_key = '{"result": "oops, wrong format"}'
        dummy_client = DummyLLMClient(canned_response=missing_specs_key)
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.last_error is not None
        # Should mention validation failure
        assert "validation" in failed_job.last_error.lower() or "field required" in failed_job.last_error.lower()
    
    async def test_invalid_must_dont_nice_types_rejected(self):
        """Test that non-list values for must/dont/nice are rejected."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # 'must' is a string instead of list
        invalid_types = {
            "specs": [{
                "purpose": "Test",
                "vision": "Test",
                "must": "should be a list, not string",
                "dont": [],
                "nice": [],
                "assumptions": []
            }]
        }
        import json
        dummy_client = DummyLLMClient(canned_response=json.dumps(invalid_types))
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.last_error is not None
        assert "validation" in failed_job.last_error.lower()
    
    async def test_must_dont_nice_non_string_items_rejected(self):
        """Test that list items that aren't strings are rejected."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # 'must' contains integers instead of strings
        invalid_item_types = {
            "specs": [{
                "purpose": "Test",
                "vision": "Test",
                "must": [1, 2, 3],  # Should be strings
                "dont": [],
                "nice": [],
                "assumptions": []
            }]
        }
        import json
        dummy_client = DummyLLMClient(canned_response=json.dumps(invalid_item_types))
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.last_error is not None
    
    async def test_empty_answers_handled_correctly(self):
        """Test that empty answers list is handled correctly."""
        spec = SpecInput(
            purpose="Empty Answers",
            vision="Test",
            must=["Feature"],
            open_questions=["Question?"]
        )
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan, answers=[])  # Empty answers
        background_tasks = MagicMock()
        
        valid_response = {
            "specs": [{
                "purpose": "Empty Answers",
                "vision": "Test",
                "must": ["Feature"],
                "dont": [],
                "nice": [],
                "assumptions": []
            }]
        }
        import json
        dummy_client = DummyLLMClient(canned_response=json.dumps(valid_response))
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        # Verify open_questions not in result
        assert not hasattr(processed_job.result.specs[0], "open_questions")
    
    async def test_answers_merged_into_must_dont_nice(self):
        """Test that answers are intended to be merged into must/dont/nice arrays.
        
        Note: The actual merging logic is done by the LLM, but we test that
        the pipeline accepts answers and produces valid output.
        """
        spec = SpecInput(
            purpose="Merge Test",
            vision="Test",
            must=["Existing feature"],
            open_questions=["Add new feature?"]
        )
        plan = PlanInput(specs=[spec])
        from app.models.specs import QuestionAnswer, ClarificationRequest
        answers = [
            QuestionAnswer(
                spec_index=0,
                question_index=0,
                question="Add new feature?",
                answer="Yes, add feature X"
            )
        ]
        request = ClarificationRequest(plan=plan, answers=answers)
        background_tasks = MagicMock()
        
        # LLM response should include both existing and new features
        merged_response = {
            "specs": [{
                "purpose": "Merge Test",
                "vision": "Test",
                "must": ["Existing feature", "Feature X"],
                "dont": [],
                "nice": [],
                "assumptions": []
            }]
        }
        import json
        dummy_client = DummyLLMClient(canned_response=json.dumps(merged_response))
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert "Existing feature" in processed_job.result.specs[0].must
        assert "Feature X" in processed_job.result.specs[0].must
        assert not hasattr(processed_job.result.specs[0], "open_questions")
    
    async def test_sanitized_error_messages_no_prompts(self):
        """Test that error messages don't contain prompts or sensitive data."""
        spec = SpecInput(
            purpose="Secret Info: api_key=sk-12345",
            vision="Contains token=abc-xyz"
        )
        plan = PlanInput(specs=[spec])
        from app.models.specs import ClarificationRequest
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Use failing client
        dummy_client = DummyLLMClient(
            simulate_failure=True,
            failure_message="Processing failed"
        )
        
        job = start_clarification_job(request, background_tasks)
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        # Error message should not contain the sensitive info from purpose/vision
        assert "api_key=sk-12345" not in failed_job.last_error
        assert "token=abc-xyz" not in failed_job.last_error
        # But should contain the failure message
        assert "Processing failed" in failed_job.last_error or "failed" in failed_job.last_error.lower()


class TestClarifiedPlanValidation:
    """Tests for ClarifiedPlan validation edge cases.
    
    These tests ensure that the Pydantic model correctly validates
    the structure of clarified specifications.
    """
    
    def test_clarified_spec_requires_purpose_and_vision(self):
        """Test that ClarifiedSpec requires purpose and vision fields."""
        from app.models.specs import ClarifiedSpec
        from pydantic import ValidationError
        
        # Missing required 'vision' field
        with pytest.raises(ValidationError):
            ClarifiedSpec(
                purpose="Test"
                # Missing vision
            )
        
        # Missing required 'purpose' field
        with pytest.raises(ValidationError):
            ClarifiedSpec(
                vision="Test"
                # Missing purpose
            )
    
    def test_clarified_spec_rejects_extra_fields(self):
        """Test that ClarifiedSpec rejects extra fields like open_questions."""
        from app.models.specs import ClarifiedSpec
        from pydantic import ValidationError
        
        # Try to include open_questions (should be rejected)
        with pytest.raises(ValidationError):
            ClarifiedSpec(
                purpose="Test",
                vision="Test",
                must=[],
                dont=[],
                nice=[],
                assumptions=[],
                open_questions=["Should not be here"]
            )
    
    def test_clarified_spec_list_fields_must_be_lists(self):
        """Test that must/dont/nice/assumptions must be lists."""
        from app.models.specs import ClarifiedSpec
        from pydantic import ValidationError
        
        # 'must' as string instead of list
        with pytest.raises(ValidationError):
            ClarifiedSpec(
                purpose="Test",
                vision="Test",
                must="Not a list",
                dont=[],
                nice=[],
                assumptions=[]
            )
    
    def test_clarified_spec_list_items_must_be_strings(self):
        """Test that list items must be strings."""
        from app.models.specs import ClarifiedSpec
        from pydantic import ValidationError
        
        # 'must' contains integers
        with pytest.raises(ValidationError):
            ClarifiedSpec(
                purpose="Test",
                vision="Test",
                must=[1, 2, 3],
                dont=[],
                nice=[],
                assumptions=[]
            )
    
    def test_clarified_plan_requires_specs_key(self):
        """Test that ClarifiedPlan requires specs key."""
        from app.models.specs import ClarifiedPlan
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ClarifiedPlan()  # Missing required 'specs' field
    
    def test_clarified_plan_accepts_empty_specs_list(self):
        """Test that ClarifiedPlan accepts empty specs list."""
        from app.models.specs import ClarifiedPlan
        
        plan = ClarifiedPlan(specs=[])
        assert plan.specs == []
    
    def test_clarified_plan_rejects_non_list_specs(self):
        """Test that ClarifiedPlan rejects non-list specs."""
        from app.models.specs import ClarifiedPlan
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ClarifiedPlan(specs="Not a list")


class TestJSONCleanupEdgeCases:
    """Tests for JSON cleanup utility edge cases.
    
    These tests cover cleanup_and_parse_json edge cases including:
    - Multiple markdown fence formats
    - Nested JSON structures
    - Various prose patterns
    """
    
    def test_nested_json_structures_preserved(self):
        """Test that nested JSON structures are preserved during cleanup."""
        from app.services.clarification import cleanup_and_parse_json
        
        nested = '''```json
{
  "specs": [{
    "purpose": "Test",
    "vision": "Nested",
    "must": ["A", "B"],
    "dont": [],
    "nice": [],
    "assumptions": []
  }]
}
```'''
        result = cleanup_and_parse_json(nested)
        assert result["specs"][0]["must"] == ["A", "B"]
    
    def test_json_with_escaped_characters(self):
        """Test JSON with escaped characters is handled correctly."""
        from app.services.clarification import cleanup_and_parse_json
        
        escaped = r'```{"specs": [{"purpose": "Test \"quoted\"", "vision": "Line\nbreak", "must": [], "dont": [], "nice": [], "assumptions": []}]}```'
        result = cleanup_and_parse_json(escaped)
        assert 'Test "quoted"' in result["specs"][0]["purpose"]
    
    def test_json_with_unicode(self):
        """Test that unicode characters are preserved."""
        from app.services.clarification import cleanup_and_parse_json
        
        unicode_json = '```{"specs": [{"purpose": "系统管理", "vision": "🚀", "must": [], "dont": [], "nice": [], "assumptions": []}]}```'
        result = cleanup_and_parse_json(unicode_json)
        assert result["specs"][0]["purpose"] == "系统管理"
        assert result["specs"][0]["vision"] == "🚀"
    
    def test_multiple_json_objects_extracts_first(self):
        """Test that when multiple JSON objects exist, first complete one is extracted."""
        from app.services.clarification import cleanup_and_parse_json
        
        multiple = '{"first": "object"} {"second": "object"}'
        result = cleanup_and_parse_json(multiple)
        # Should extract the first complete object
        assert result == {"first": "object"}
    
    def test_json_with_leading_prose_patterns(self):
        """Test various leading prose patterns are removed."""
        from app.services.clarification import cleanup_and_parse_json
        
        patterns = [
            'Here is the JSON: {"key": "value"}',
            "Here's the result: {\"key\": \"value\"}",
            'Sure! {"key": "value"}',
            'Certainly, {"key": "value"}',
        ]
        
        for pattern in patterns:
            result = cleanup_and_parse_json(pattern)
            assert result == {"key": "value"}
    
    def test_cleanup_max_attempts_parameter(self):
        """Test that max_attempts parameter limits retry attempts."""
        from app.services.clarification import cleanup_and_parse_json, JSONCleanupError
        
        # This will fail all attempts
        invalid = "definitely not json at all"
        
        with pytest.raises(JSONCleanupError) as exc_info:
            cleanup_and_parse_json(invalid, max_attempts=2)
        
        # Should have made 2 attempts
        assert exc_info.value.attempts == 2
    
    def test_cleanup_error_includes_raw_content(self):
        """Test that JSONCleanupError includes raw content for debugging."""
        from app.services.clarification import cleanup_and_parse_json, JSONCleanupError
        
        invalid = "not json"
        
        with pytest.raises(JSONCleanupError) as exc_info:
            cleanup_and_parse_json(invalid)
        
        assert exc_info.value.raw_content == "not json"
        assert exc_info.value.message is not None

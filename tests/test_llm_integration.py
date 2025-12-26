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
"""Integration tests for LLM pipeline wiring in process_clarification_job."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.models.specs import (
    ClarificationRequest,
    JobStatus,
    PlanInput,
    SpecInput,
)
from app.services.clarification import process_clarification_job, start_clarification_job
from app.services.job_store import clear_all_jobs, get_job
from app.services.llm_clients import ClarificationLLMConfig, DummyLLMClient


@pytest.fixture(autouse=True)
def clean_job_store():
    """Clean the job store before and after each test."""
    clear_all_jobs()
    yield
    clear_all_jobs()


def _create_dummy_client_with_response(specs):
    """Helper to create DummyLLMClient with valid ClarifiedPlan JSON response."""
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


class TestLLMPipelineIntegration:
    """Integration tests for LLM pipeline in process_clarification_job."""
    
    async def test_default_llm_config_is_applied(self):
        """Test that default LLM config (provider=openai, model=gpt-5) is used when not specified."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Start job without llm_config (should use defaults)
        job = start_clarification_job(request, background_tasks)
        
        # Track what config is used
        captured_configs = []
        
        def capture_factory_call(provider, config):
            captured_configs.append((provider, config))
            return _create_dummy_client_with_response([spec])
        
        with patch('app.services.clarification.get_llm_client', side_effect=capture_factory_call):
            await process_clarification_job(job.id)
        
        # Verify default config was used
        assert len(captured_configs) == 1
        provider, config = captured_configs[0]
        assert provider == "openai"
        assert isinstance(config, ClarificationLLMConfig)
        assert config.provider == "openai"
        assert config.model == "gpt-5"
        
        # Job should succeed
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
    
    async def test_stored_llm_config_overrides_defaults(self):
        """Test that stored LLM config overrides defaults."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Start job with custom config
        custom_config = ClarificationLLMConfig(
            provider="anthropic",
            model="claude-sonnet-4.5",
            temperature=0.3
        )
        job = start_clarification_job(request, background_tasks, llm_config=custom_config)
        
        # Track what config is used
        captured_configs = []
        
        def capture_factory_call(provider, config):
            captured_configs.append((provider, config))
            return _create_dummy_client_with_response([spec])
        
        with patch('app.services.clarification.get_llm_client', side_effect=capture_factory_call):
            await process_clarification_job(job.id)
        
        # Verify custom config was used
        assert len(captured_configs) == 1
        provider, config = captured_configs[0]
        assert provider == "anthropic"
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4.5"
        assert config.temperature == 0.3
    
    async def test_job_missing_clarification_request_fails(self):
        """Test that job missing ClarificationRequest fails with helpful error."""
        # Create a job directly in the store without proper request
        from app.services.job_store import create_job
        from app.models.specs import ClarificationRequest, PlanInput
        
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        job = create_job(request)
        
        # Manually clear the request to simulate missing data
        job.request = None
        from app.services.job_store import _job_store, _store_lock
        with _store_lock:
            _job_store[job.id] = job
        
        # Process should fail with helpful error
        await process_clarification_job(job.id)
        
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert "missing ClarificationRequest" in failed_job.last_error
    
    async def test_llm_returns_markdown_wrapped_json(self):
        """Test that markdown-wrapped JSON is successfully cleaned and parsed."""
        spec = SpecInput(purpose="Markdown Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Create client that returns markdown-wrapped JSON
        markdown_response = '''```json
{
  "specs": [
    {
      "purpose": "Markdown Test",
      "vision": "Test vision",
      "must": [],
      "dont": [],
      "nice": [],
      "assumptions": []
    }
  ]
}
```'''
        
        dummy_client = DummyLLMClient(canned_response=markdown_response)
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Job should succeed despite markdown wrapping
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert processed_job.result is not None
        assert processed_job.result.specs[0].purpose == "Markdown Test"
    
    async def test_llm_returns_invalid_json_fails_job(self):
        """Test that invalid JSON marks job as FAILED."""
        spec = SpecInput(purpose="Invalid JSON Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Create client that returns invalid JSON
        dummy_client = DummyLLMClient(canned_response='{invalid json}')
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Job should fail
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert "Failed to parse LLM response" in failed_job.last_error
        assert failed_job.result is None
    
    async def test_llm_returns_json_missing_required_fields(self):
        """Test that JSON missing required fields fails validation."""
        spec = SpecInput(purpose="Validation Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Create client that returns JSON missing 'vision' field
        incomplete_response = json.dumps({
            "specs": [
                {
                    "purpose": "Validation Test",
                    # Missing vision, must, dont, nice, assumptions
                }
            ]
        })
        
        dummy_client = DummyLLMClient(canned_response=incomplete_response)
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        # Job should fail validation
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert "validation failed" in failed_job.last_error.lower()
        assert failed_job.result is None
    
    async def test_invalid_provider_name_fails_before_llm_call(self):
        """Test that invalid provider name raises deterministic error before attempting call."""
        spec = SpecInput(purpose="Invalid Provider Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        # Mock get_llm_client to raise ValueError for invalid provider
        def failing_factory(provider, config):
            raise ValueError(f"Unsupported provider '{provider}'")
        
        job = start_clarification_job(request, background_tasks)
        
        with patch('app.services.clarification.get_llm_client', side_effect=failing_factory):
            await process_clarification_job(job.id)
        
        # Job should fail with provider error
        failed_job = get_job(job.id)
        assert failed_job.status == JobStatus.FAILED
        assert "Invalid LLM provider" in failed_job.last_error
    
    async def test_llm_call_logs_metrics_without_prompts(self):
        """Test that LLM call logs structured metrics without including prompts."""
        spec = SpecInput(purpose="Logging Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        dummy_client = _create_dummy_client_with_response([spec])
        
        # Capture log messages
        import logging
        with patch.object(logging.getLogger('app.services.clarification'), 'info') as mock_log:
            await process_clarification_job(job.id, llm_client=dummy_client)
            
            # Find the success log message
            success_logs = [
                call for call in mock_log.call_args_list
                if 'LLM call successful' in str(call)
            ]
            
            assert len(success_logs) > 0
            log_message = str(success_logs[0])
            
            # Verify metrics are logged
            assert 'provider=openai' in log_message
            assert 'model=gpt-5' in log_message
            assert 'elapsed_time=' in log_message
            
            # Verify prompts are NOT logged
            assert 'system_prompt' not in log_message
            assert 'user_prompt' not in log_message
            assert 'Test vision' not in log_message  # Content from prompt
    
    async def test_multiple_specs_are_processed_correctly(self):
        """Test that jobs with multiple specs are processed correctly."""
        spec1 = SpecInput(purpose="Spec 1", vision="Vision 1", must=["Req 1"])
        spec2 = SpecInput(purpose="Spec 2", vision="Vision 2", must=["Req 2"])
        spec3 = SpecInput(purpose="Spec 3", vision="Vision 3", must=["Req 3"])
        
        plan = PlanInput(specs=[spec1, spec2, spec3])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        dummy_client = _create_dummy_client_with_response([spec1, spec2, spec3])
        
        await process_clarification_job(job.id, llm_client=dummy_client)
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS
        assert len(processed_job.result.specs) == 3
        assert processed_job.result.specs[0].purpose == "Spec 1"
        assert processed_job.result.specs[1].purpose == "Spec 2"
        assert processed_job.result.specs[2].purpose == "Spec 3"
    
    async def test_llm_call_exactly_once_per_job(self):
        """Test that LLM complete() is called exactly once per job."""
        spec = SpecInput(purpose="Single Call Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        background_tasks = MagicMock()
        
        job = start_clarification_job(request, background_tasks)
        
        # Create a mock client to track calls
        call_count = []
        
        class TrackingClient:
            async def complete(self, system_prompt, user_prompt, model, **kwargs):
                call_count.append(1)
                return json.dumps({
                    "specs": [{
                        "purpose": "Single Call Test",
                        "vision": "Test vision",
                        "must": [],
                        "dont": [],
                        "nice": [],
                        "assumptions": []
                    }]
                })
        
        await process_clarification_job(job.id, llm_client=TrackingClient())
        
        # Verify complete() was called exactly once
        assert len(call_count) == 1
        
        processed_job = get_job(job.id)
        assert processed_job.status == JobStatus.SUCCESS

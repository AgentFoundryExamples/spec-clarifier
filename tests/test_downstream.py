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
"""Tests for downstream dispatcher module."""

import logging
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.models.specs import ClarificationRequest, ClarifiedPlan, ClarifiedSpec, JobStatus, PlanInput, SpecInput
from app.services.downstream import PlaceholderDownstreamDispatcher, get_downstream_dispatcher
from app.services.job_store import create_job, clear_all_jobs, get_job


@pytest.fixture(autouse=True)
def clean_job_store_for_downstream_tests():
    """Clean the job store before and after each test."""
    clear_all_jobs()
    yield
    clear_all_jobs()


class TestPlaceholderDownstreamDispatcher:
    """Tests for PlaceholderDownstreamDispatcher implementation."""
    
    async def test_placeholder_dispatcher_logs_plan_with_banners(self, caplog):
        """Test that placeholder dispatcher outputs banner-delimited logs."""
        # Create a simple job and plan
        spec = SpecInput(
            purpose="Test Service",
            vision="High performance",
            must=["Fast", "Reliable"],
            dont=["Slow"],
            nice=["Configurable"],
            assumptions=["Cloud deployment"]
        )
        plan_input = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan_input)
        
        # Use job_store to create a proper job
        job = create_job(request)
        
        clarified_plan = ClarifiedPlan(specs=[
            ClarifiedSpec(
                purpose="Test Service",
                vision="High performance",
                must=["Fast", "Reliable"],
                dont=["Slow"],
                nice=["Configurable"],
                assumptions=["Cloud deployment"]
            )
        ])
        
        dispatcher = PlaceholderDownstreamDispatcher()
        
        with caplog.at_level(logging.INFO):
            await dispatcher.dispatch(job, clarified_plan)
        
        # Verify banner messages in logs
        log_text = caplog.text
        assert "DOWNSTREAM DISPATCH START" in log_text
        assert "DOWNSTREAM DISPATCH END" in log_text
        assert str(job.id) in log_text
        
        # Verify plan JSON is in logs
        assert "Clarified Plan JSON" in log_text
        assert "Test Service" in log_text
        assert "High performance" in log_text
    
    async def test_placeholder_dispatcher_includes_structured_logging(self, caplog):
        """Test that placeholder dispatcher emits structured log events."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan_input = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan_input)
        
        job = create_job(request)
        
        clarified_plan = ClarifiedPlan(specs=[
            ClarifiedSpec(purpose="Test", vision="Test vision")
        ])
        
        dispatcher = PlaceholderDownstreamDispatcher()
        
        with caplog.at_level(logging.INFO):
            await dispatcher.dispatch(job, clarified_plan)
        
        # Look for structured log event
        assert "downstream_dispatch_placeholder" in caplog.text
        assert str(job.id) in caplog.text
    
    async def test_placeholder_dispatcher_handles_multiple_specs(self, caplog):
        """Test dispatcher with multiple specs in plan."""
        specs = [
            SpecInput(purpose=f"Spec {i}", vision=f"Vision {i}")
            for i in range(3)
        ]
        plan_input = PlanInput(specs=specs)
        request = ClarificationRequest(plan=plan_input)
        
        job = create_job(request)
        
        clarified_specs = [
            ClarifiedSpec(purpose=f"Spec {i}", vision=f"Vision {i}")
            for i in range(3)
        ]
        clarified_plan = ClarifiedPlan(specs=clarified_specs)
        
        dispatcher = PlaceholderDownstreamDispatcher()
        
        with caplog.at_level(logging.INFO):
            await dispatcher.dispatch(job, clarified_plan)
        
        # Verify all specs are in logs
        log_text = caplog.text
        for i in range(3):
            assert f"Spec {i}" in log_text
            assert f"Vision {i}" in log_text
    
    async def test_placeholder_dispatcher_handles_unicode(self, caplog):
        """Test dispatcher with unicode characters in plan."""
        spec = SpecInput(
            purpose="系统管理",
            vision="🚀 Modern system",
            must=["UTF-8 支持"]
        )
        plan_input = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan_input)
        
        job = create_job(request)
        
        clarified_plan = ClarifiedPlan(specs=[
            ClarifiedSpec(
                purpose="系统管理",
                vision="🚀 Modern system",
                must=["UTF-8 支持"]
            )
        ])
        
        dispatcher = PlaceholderDownstreamDispatcher()
        
        with caplog.at_level(logging.INFO):
            await dispatcher.dispatch(job, clarified_plan)
        
        # Verify unicode is preserved in logs
        log_text = caplog.text
        assert "系统管理" in log_text
        assert "🚀" in log_text
        assert "UTF-8 支持" in log_text
    
    async def test_placeholder_dispatcher_with_empty_plan(self, caplog):
        """Test dispatcher with empty plan (no specs)."""
        plan_input = PlanInput(specs=[])
        request = ClarificationRequest(plan=plan_input)
        
        job = create_job(request)
        
        clarified_plan = ClarifiedPlan(specs=[])
        
        dispatcher = PlaceholderDownstreamDispatcher()
        
        with caplog.at_level(logging.INFO):
            await dispatcher.dispatch(job, clarified_plan)
        
        # Should still log banners even with empty plan
        log_text = caplog.text
        assert "DOWNSTREAM DISPATCH START" in log_text
        assert "DOWNSTREAM DISPATCH END" in log_text
    
    async def test_placeholder_dispatcher_handles_serialization_errors_gracefully(self, caplog):
        """Test that dispatcher handles JSON serialization errors gracefully."""
        from unittest.mock import patch
        
        spec = SpecInput(purpose="Test", vision="Test")
        plan_input = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan_input)
        
        job = create_job(request)
        
        clarified_plan = ClarifiedPlan(specs=[
            ClarifiedSpec(purpose="Test", vision="Test")
        ])
        
        dispatcher = PlaceholderDownstreamDispatcher()
        
        # Mock json.dumps to raise an error
        with patch('app.services.downstream.json.dumps', side_effect=TypeError("Cannot serialize")):
            with caplog.at_level(logging.INFO):
                await dispatcher.dispatch(job, clarified_plan)
        
        # Should still log banners despite serialization error
        log_text = caplog.text
        assert "DOWNSTREAM DISPATCH START" in log_text
        assert "DOWNSTREAM DISPATCH END" in log_text
        # Should mention serialization failure
        assert "JSON serialization failed" in log_text or "Cannot serialize" in log_text


class TestGetDownstreamDispatcher:
    """Tests for dispatcher factory function."""
    
    def test_get_dispatcher_returns_placeholder(self):
        """Test that factory returns placeholder dispatcher by default."""
        dispatcher = get_downstream_dispatcher()
        assert isinstance(dispatcher, PlaceholderDownstreamDispatcher)
    
    def test_get_dispatcher_returns_new_instance_each_call(self):
        """Test that factory returns a new instance on each call."""
        dispatcher1 = get_downstream_dispatcher()
        dispatcher2 = get_downstream_dispatcher()
        
        # Should be different instances (stateless dispatchers)
        assert dispatcher1 is not dispatcher2
    
    async def test_dispatcher_from_factory_is_functional(self, caplog):
        """Test that dispatcher from factory can dispatch successfully."""
        dispatcher = get_downstream_dispatcher()
        
        spec = SpecInput(purpose="Test", vision="Test")
        plan_input = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan_input)
        
        job = create_job(request)
        
        clarified_plan = ClarifiedPlan(specs=[
            ClarifiedSpec(purpose="Test", vision="Test")
        ])
        
        with caplog.at_level(logging.INFO):
            await dispatcher.dispatch(job, clarified_plan)
        
        # Should log successfully
        assert "DOWNSTREAM DISPATCH START" in caplog.text

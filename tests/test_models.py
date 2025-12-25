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
"""Tests for specification models."""

import pytest
from pydantic import ValidationError

from app.models.specs import (
    ClarificationRequest,
    ClarifiedPlan,
    ClarifiedSpec,
    PlanInput,
    QuestionAnswer,
    SpecInput,
)


class TestSpecInput:
    """Tests for SpecInput model."""
    
    def test_spec_input_with_all_fields(self):
        """Test creating SpecInput with all fields."""
        spec = SpecInput(
            purpose="Build a web app",
            vision="Modern and user-friendly",
            must=["Authentication", "Database"],
            dont=["Complex UI", "Legacy support"],
            nice=["Dark mode", "Mobile app"],
            open_questions=["What database?", "Which auth provider?"],
            assumptions=["Users have modern browsers", "Internet connection available"],
        )
        
        assert spec.purpose == "Build a web app"
        assert spec.vision == "Modern and user-friendly"
        assert len(spec.must) == 2
        assert len(spec.dont) == 2
        assert len(spec.nice) == 2
        assert len(spec.open_questions) == 2
        assert len(spec.assumptions) == 2
    
    def test_spec_input_with_minimal_fields(self):
        """Test creating SpecInput with only required fields."""
        spec = SpecInput(
            purpose="Build a web app",
            vision="Modern and user-friendly",
        )
        
        assert spec.purpose == "Build a web app"
        assert spec.vision == "Modern and user-friendly"
        assert spec.must == []
        assert spec.dont == []
        assert spec.nice == []
        assert spec.open_questions == []
        assert spec.assumptions == []
    
    def test_spec_input_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SpecInput(purpose="Build a web app")
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("vision",) for error in errors)
    
    def test_spec_input_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SpecInput(
                purpose="Build a web app",
                vision="Modern and user-friendly",
                extra_field="should not be allowed",
            )
        
        errors = exc_info.value.errors()
        assert any(error["type"] == "extra_forbidden" for error in errors)
    
    def test_spec_input_with_empty_lists(self):
        """Test that empty lists are valid."""
        spec = SpecInput(
            purpose="Build a web app",
            vision="Modern and user-friendly",
            must=[],
            dont=[],
            nice=[],
            open_questions=[],
            assumptions=[],
        )
        
        assert spec.must == []
        assert spec.dont == []
        assert spec.nice == []
        assert spec.open_questions == []
        assert spec.assumptions == []


class TestPlanInput:
    """Tests for PlanInput model."""
    
    def test_plan_input_with_multiple_specs(self):
        """Test creating PlanInput with multiple specs."""
        spec1 = SpecInput(purpose="Spec 1", vision="Vision 1")
        spec2 = SpecInput(purpose="Spec 2", vision="Vision 2")
        
        plan = PlanInput(specs=[spec1, spec2])
        
        assert len(plan.specs) == 2
        assert plan.specs[0].purpose == "Spec 1"
        assert plan.specs[1].purpose == "Spec 2"
    
    def test_plan_input_with_empty_specs_list(self):
        """Test creating PlanInput with empty specs list."""
        plan = PlanInput(specs=[])
        
        assert plan.specs == []
    
    def test_plan_input_rejects_extra_fields(self):
        """Test that extra fields are rejected."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        
        with pytest.raises(ValidationError) as exc_info:
            PlanInput(specs=[spec], extra_field="not allowed")
        
        errors = exc_info.value.errors()
        assert any(error["type"] == "extra_forbidden" for error in errors)


class TestClarifiedSpec:
    """Tests for ClarifiedSpec model."""
    
    def test_clarified_spec_with_all_fields(self):
        """Test creating ClarifiedSpec with all fields."""
        spec = ClarifiedSpec(
            purpose="Build a web app",
            vision="Modern and user-friendly",
            must=["Authentication", "Database"],
            dont=["Complex UI", "Legacy support"],
            nice=["Dark mode", "Mobile app"],
            assumptions=["Users have modern browsers", "Internet connection available"],
        )
        
        assert spec.purpose == "Build a web app"
        assert spec.vision == "Modern and user-friendly"
        assert len(spec.must) == 2
        assert len(spec.dont) == 2
        assert len(spec.nice) == 2
        assert len(spec.assumptions) == 2
        # Note: ClarifiedSpec does not have open_questions field
    
    def test_clarified_spec_has_no_open_questions(self):
        """Test that ClarifiedSpec does not accept open_questions field."""
        with pytest.raises(ValidationError) as exc_info:
            ClarifiedSpec(
                purpose="Build a web app",
                vision="Modern and user-friendly",
                open_questions=["Should not be allowed"],
            )
        
        errors = exc_info.value.errors()
        assert any(error["type"] == "extra_forbidden" for error in errors)


class TestClarifiedPlan:
    """Tests for ClarifiedPlan model."""
    
    def test_clarified_plan_with_multiple_specs(self):
        """Test creating ClarifiedPlan with multiple specs."""
        spec1 = ClarifiedSpec(purpose="Spec 1", vision="Vision 1")
        spec2 = ClarifiedSpec(purpose="Spec 2", vision="Vision 2")
        
        plan = ClarifiedPlan(specs=[spec1, spec2])
        
        assert len(plan.specs) == 2
        assert plan.specs[0].purpose == "Spec 1"
        assert plan.specs[1].purpose == "Spec 2"


class TestQuestionAnswer:
    """Tests for QuestionAnswer model."""
    
    def test_question_answer_valid(self):
        """Test creating valid QuestionAnswer."""
        qa = QuestionAnswer(
            spec_index=0,
            question_index=2,
            question="What database to use?",
            answer="PostgreSQL",
        )
        
        assert qa.spec_index == 0
        assert qa.question_index == 2
        assert qa.question == "What database to use?"
        assert qa.answer == "PostgreSQL"
    
    def test_question_answer_negative_index_rejected(self):
        """Test that negative indices are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QuestionAnswer(
                spec_index=-1,
                question_index=0,
                question="Test",
                answer="Test",
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("spec_index",) for error in errors)
    
    def test_question_answer_out_of_range_index_allowed(self):
        """Test that out-of-range indices are allowed (validation happens elsewhere)."""
        # This should not raise an error - validation logic handles this
        qa = QuestionAnswer(
            spec_index=999,
            question_index=999,
            question="Test",
            answer="Test",
        )
        
        assert qa.spec_index == 999
        assert qa.question_index == 999


class TestClarificationRequest:
    """Tests for ClarificationRequest model."""
    
    def test_clarification_request_with_answers(self):
        """Test creating ClarificationRequest with answers."""
        spec = SpecInput(
            purpose="Build a web app",
            vision="Modern",
            open_questions=["What database?"],
        )
        plan = PlanInput(specs=[spec])
        
        qa = QuestionAnswer(
            spec_index=0,
            question_index=0,
            question="What database?",
            answer="PostgreSQL",
        )
        
        request = ClarificationRequest(plan=plan, answers=[qa])
        
        assert request.plan == plan
        assert len(request.answers) == 1
        assert request.answers[0].answer == "PostgreSQL"
    
    def test_clarification_request_without_answers(self):
        """Test creating ClarificationRequest without answers."""
        spec = SpecInput(purpose="Build a web app", vision="Modern")
        plan = PlanInput(specs=[spec])
        
        request = ClarificationRequest(plan=plan)
        
        assert request.plan == plan
        assert request.answers == []
    
    def test_clarification_request_serialization(self):
        """Test that ClarificationRequest can be serialized to dict."""
        spec = SpecInput(
            purpose="Build a web app",
            vision="Modern",
            must=["Auth"],
            dont=["Legacy"],
            nice=["Dark mode"],
            open_questions=["What database?"],
            assumptions=["Modern browsers"],
        )
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan)
        
        data = request.model_dump()
        
        assert data["plan"]["specs"][0]["purpose"] == "Build a web app"
        assert data["plan"]["specs"][0]["must"] == ["Auth"]
        assert data["answers"] == []

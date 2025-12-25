"""Tests for clarification service."""

import pytest

from app.models.specs import PlanInput, SpecInput
from app.services.clarification import clarify_plan


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
    
    def test_clarify_plan_ignores_open_questions(self):
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
    
    def test_clarify_plan_list_independence(self):
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
    
    def test_clarify_plan_with_unicode_and_special_chars(self):
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

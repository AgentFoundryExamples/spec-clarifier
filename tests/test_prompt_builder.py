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
"""Tests for prompt builder and JSON cleanup utilities."""

import json

import pytest

from app.models.specs import ClarificationRequest, PlanInput, QuestionAnswer, SpecInput
from app.services.clarification import (
    SYSTEM_PROMPT_TEMPLATES,
    JSONCleanupError,
    build_clarification_prompts,
    cleanup_and_parse_json,
    get_system_prompt_template,
)


class TestSystemPromptTemplates:
    """Tests for system prompt template selection and fallback."""

    def test_get_system_prompt_template_default(self):
        """Test retrieving the default template."""
        template = get_system_prompt_template("default")

        assert template == SYSTEM_PROMPT_TEMPLATES["default"]
        assert "specification clarification assistant" in template.lower()
        assert "valid JSON" in template or "JSON object" in template
        assert "purpose" in template
        assert "vision" in template
        assert "must" in template
        assert "dont" in template
        assert "nice" in template
        assert "assumptions" in template

    def test_get_system_prompt_template_strict_json(self):
        """Test retrieving the strict_json template."""
        template = get_system_prompt_template("strict_json")

        assert template == SYSTEM_PROMPT_TEMPLATES["strict_json"]
        assert "STRICT JSON MODE" in template
        assert "CRITICAL RULES" in template
        assert "NO markdown code fences" in template
        assert "NO explanatory text" in template

    def test_get_system_prompt_template_verbose_explanation(self):
        """Test retrieving the verbose_explanation template."""
        template = get_system_prompt_template("verbose_explanation")

        assert template == SYSTEM_PROMPT_TEMPLATES["verbose_explanation"]
        assert "YOUR TASK:" in template
        assert "Analyze the specifications" in template
        assert "most appropriate field" in template

    def test_get_system_prompt_template_unknown_falls_back(self, caplog):
        """Test that unknown template ID falls back to default with warning."""
        import logging

        caplog.set_level(logging.WARNING)

        template = get_system_prompt_template("unknown_template_id")

        # Should return default template
        assert template == SYSTEM_PROMPT_TEMPLATES["default"]

        # Should log a warning
        assert any("unknown_template_id" in record.message.lower() for record in caplog.records)
        assert any("falling back to" in record.message.lower() for record in caplog.records)

    def test_all_templates_enforce_json_strictness(self):
        """Test that all templates contain strict JSON output requirements."""
        for template_id, template in SYSTEM_PROMPT_TEMPLATES.items():
            # All templates should mention JSON
            assert "JSON" in template, f"Template {template_id} doesn't mention JSON"

            # All templates should require ONLY JSON output
            assert (
                "ONLY" in template or "only" in template.lower()
            ), f"Template {template_id} doesn't emphasize 'ONLY' JSON"

            # All templates should specify the 6 required keys
            required_keys = ["purpose", "vision", "must", "dont", "nice", "assumptions"]
            for key in required_keys:
                assert key in template, f"Template {template_id} missing required key: {key}"

            # All templates should say not to include open_questions
            assert (
                "NOT include" in template or "do not include" in template.lower()
            ), f"Template {template_id} doesn't prohibit open_questions"
            assert (
                "open_questions" in template or "open questions" in template.lower()
            ), f"Template {template_id} doesn't mention open_questions"

    def test_build_prompts_with_default_template(self):
        """Test building prompts with default template (implicit)."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Should use default template
        assert "specification clarification assistant" in system_prompt.lower()
        assert "valid JSON" in system_prompt or "JSON object" in system_prompt

    def test_build_prompts_with_explicit_default_template(self):
        """Test building prompts with explicitly specified default template."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(
            request, system_prompt_id="default"
        )

        # Should use default template
        assert "specification clarification assistant" in system_prompt.lower()

    def test_build_prompts_with_strict_json_template(self):
        """Test building prompts with strict_json template."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(
            request, system_prompt_id="strict_json"
        )

        # Should use strict_json template
        assert "STRICT JSON MODE" in system_prompt
        assert "CRITICAL RULES" in system_prompt

    def test_build_prompts_with_verbose_template(self):
        """Test building prompts with verbose_explanation template."""
        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(
            request, system_prompt_id="verbose_explanation"
        )

        # Should use verbose_explanation template
        assert "YOUR TASK:" in system_prompt
        assert "Analyze the specifications" in system_prompt

    def test_build_prompts_with_unknown_template_falls_back(self, caplog):
        """Test that unknown template ID falls back to default when building prompts."""
        import logging

        caplog.set_level(logging.WARNING)

        spec = SpecInput(purpose="Test", vision="Test vision")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(
            request, system_prompt_id="nonexistent_template"
        )

        # Should fall back to default template
        assert "specification clarification assistant" in system_prompt.lower()

        # Should log a warning
        assert any("nonexistent_template" in record.message.lower() for record in caplog.records)


class TestBuildClarificationPrompts:
    """Tests for the build_clarification_prompts function."""

    def test_builds_prompts_for_simple_request(self):
        """Test building prompts for a simple request with one spec and no answers."""
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
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Verify system prompt contains key instructions
        assert "specification clarification" in system_prompt.lower()
        assert "purpose" in system_prompt
        assert "vision" in system_prompt
        assert "must" in system_prompt
        assert "dont" in system_prompt
        assert "nice" in system_prompt
        assert "assumptions" in system_prompt
        assert "open_questions" in system_prompt.lower()
        assert "valid JSON" in system_prompt or "JSON object" in system_prompt
        assert "no markdown" in system_prompt.lower() or "only" in system_prompt.lower()

        # Verify user prompt contains spec data
        assert "Build a web app" in user_prompt
        assert "Modern and user-friendly" in user_prompt
        assert "Authentication" in user_prompt
        assert "Database" in user_prompt
        assert "What database?" in user_prompt
        assert "Which auth provider?" in user_prompt

    def test_includes_indexed_answers_in_user_prompt(self):
        """Test that answers are properly indexed and included."""
        spec = SpecInput(
            purpose="Test",
            vision="Test vision",
            open_questions=["Question 1", "Question 2"],
        )
        plan = PlanInput(specs=[spec])

        answers = [
            QuestionAnswer(
                spec_index=0, question_index=0, question="Question 1", answer="Answer to question 1"
            ),
            QuestionAnswer(
                spec_index=0, question_index=1, question="Question 2", answer="Answer to question 2"
            ),
        ]

        request = ClarificationRequest(plan=plan, answers=answers)
        system_prompt, user_prompt = build_clarification_prompts(request)

        # Verify answers are in user prompt with proper indexing
        assert "spec_index" in user_prompt
        assert "question_index" in user_prompt
        assert "Answer to question 1" in user_prompt
        assert "Answer to question 2" in user_prompt

        # Verify JSON structure is present
        assert '"answers"' in user_prompt or "'answers'" in user_prompt

    def test_handles_multiple_specs(self):
        """Test building prompts with multiple specifications."""
        spec1 = SpecInput(purpose="Frontend", vision="Responsive UI", must=["React"])
        spec2 = SpecInput(purpose="Backend", vision="Scalable API", must=["FastAPI"])
        spec3 = SpecInput(purpose="DevOps", vision="Automated", nice=["CI/CD"])

        plan = PlanInput(specs=[spec1, spec2, spec3])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # All specs should be in user prompt
        assert "Frontend" in user_prompt
        assert "Backend" in user_prompt
        assert "DevOps" in user_prompt

    def test_handles_empty_open_questions(self):
        """Test that specs without open questions are handled correctly."""
        spec = SpecInput(
            purpose="No questions",
            vision="Clear vision",
            must=["Feature 1"],
            open_questions=[],  # Empty list
        )
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Should still contain open_questions field in JSON
        assert "open_questions" in user_prompt
        assert "No questions" in user_prompt

    def test_handles_missing_answers_for_questions(self):
        """Test handling when questions exist but no answers provided."""
        spec = SpecInput(
            purpose="Partial answers",
            vision="Test",
            open_questions=["Q1", "Q2", "Q3"],
        )
        plan = PlanInput(specs=[spec])

        # Only answer one question
        answers = [
            QuestionAnswer(spec_index=0, question_index=0, question="Q1", answer="Answer 1"),
        ]

        request = ClarificationRequest(plan=plan, answers=answers)
        system_prompt, user_prompt = build_clarification_prompts(request)

        # All questions should be in prompt
        assert "Q1" in user_prompt
        assert "Q2" in user_prompt
        assert "Q3" in user_prompt

        # Only one answer should be present
        assert "Answer 1" in user_prompt
        assert user_prompt.count('"answer"') == 1 or user_prompt.count("'answer'") == 1

    def test_handles_multiple_answers_for_same_spec(self):
        """Test multiple answers for different questions in same spec."""
        spec = SpecInput(
            purpose="Multi-answer",
            vision="Test",
            open_questions=["Q1", "Q2", "Q3"],
        )
        plan = PlanInput(specs=[spec])

        answers = [
            QuestionAnswer(spec_index=0, question_index=0, question="Q1", answer="A1"),
            QuestionAnswer(spec_index=0, question_index=1, question="Q2", answer="A2"),
            QuestionAnswer(spec_index=0, question_index=2, question="Q3", answer="A3"),
        ]

        request = ClarificationRequest(plan=plan, answers=answers)
        system_prompt, user_prompt = build_clarification_prompts(request)

        # All answers should be present with correct indexes
        assert "A1" in user_prompt
        assert "A2" in user_prompt
        assert "A3" in user_prompt

    def test_large_payload_handling(self):
        """Test that large plans don't cause truncation or escaping issues."""
        # Create a spec with large arrays
        large_must = [f"Requirement {i}" for i in range(100)]
        large_questions = [f"Question {i}?" for i in range(50)]

        spec = SpecInput(
            purpose="Large plan test",
            vision="Testing scalability",
            must=large_must,
            open_questions=large_questions,
        )
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Verify no truncation
        assert "Requirement 0" in user_prompt
        assert "Requirement 99" in user_prompt
        assert "Question 0?" in user_prompt
        assert "Question 49?" in user_prompt

        # Verify JSON is valid (can be parsed)
        # Extract JSON from user prompt
        start = user_prompt.find("{")
        end = user_prompt.rfind("}") + 1
        json_str = user_prompt[start:end]
        parsed = json.loads(json_str)
        assert len(parsed["specs"][0]["must"]) == 100

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in specs."""
        spec = SpecInput(
            purpose="Système de gestion 系统管理",
            vision="🚀 Modern & scalable <system>",
            must=["UTF-8 支持", "Émojis 👍"],
            dont=["Special \"quotes\" and 'apostrophes'"],
            open_questions=["Comment gérer les accents?"],
        )
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Verify unicode is preserved
        assert "Système de gestion 系统管理" in user_prompt
        assert "🚀" in user_prompt
        assert "UTF-8 支持" in user_prompt
        assert "Émojis 👍" in user_prompt

        # Verify JSON is valid
        start = user_prompt.find("{")
        end = user_prompt.rfind("}") + 1
        json_str = user_prompt[start:end]
        parsed = json.loads(json_str)  # Should not raise
        assert parsed["specs"][0]["purpose"] == "Système de gestion 系统管理"

    def test_no_secrets_in_prompts(self, monkeypatch):
        """Test that no environment secrets leak into prompts."""
        # Set some fake environment variables using monkeypatch for isolation
        monkeypatch.setenv("TEST_API_KEY", "secret-key-12345")
        monkeypatch.setenv("TEST_PASSWORD", "super-secret-pwd")

        spec = SpecInput(
            purpose="Test security",
            vision="No secrets",
            must=["Feature"],
        )
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Verify no secrets in prompts
        assert "secret-key-12345" not in system_prompt
        assert "secret-key-12345" not in user_prompt
        assert "super-secret-pwd" not in system_prompt
        assert "super-secret-pwd" not in user_prompt

    def test_provider_model_metadata_not_in_prompts(self):
        """Test that provider/model metadata doesn't leak into prompts."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        # Pass provider and model metadata
        system_prompt, user_prompt = build_clarification_prompts(
            request, provider="test-provider", model="test-model-123"
        )

        # Verify metadata is NOT in prompts (reserved for future use)
        assert "test-provider" not in system_prompt
        assert "test-provider" not in user_prompt
        assert "test-model-123" not in system_prompt
        assert "test-model-123" not in user_prompt

    def test_system_prompt_contains_schema_enumeration(self):
        """Test that system prompt enumerates the exact output schema."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Verify schema keys are explicitly mentioned
        assert "purpose" in system_prompt
        assert "vision" in system_prompt
        assert "must" in system_prompt
        assert "dont" in system_prompt
        assert "nice" in system_prompt
        assert "assumptions" in system_prompt

        # Verify it's clear that 6 keys are expected
        assert "6" in system_prompt or "six" in system_prompt.lower()

    def test_system_prompt_emphasizes_no_open_questions(self):
        """Test that system prompt emphasizes not including open_questions."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Should explicitly say not to include open_questions
        assert "NOT include" in system_prompt or "do not include" in system_prompt.lower()
        assert "open_questions" in system_prompt or "open questions" in system_prompt.lower()

    def test_system_prompt_requires_single_call(self):
        """Test that system prompt emphasizes single-call processing."""
        spec = SpecInput(purpose="Test", vision="Test")
        plan = PlanInput(specs=[spec])
        request = ClarificationRequest(plan=plan, answers=[])

        system_prompt, user_prompt = build_clarification_prompts(request)

        # Should indicate single call/response
        assert "single" in system_prompt.lower() or "one response" in system_prompt.lower()

    def test_raises_error_for_none_request(self):
        """Test that None request raises ValueError."""
        with pytest.raises(ValueError, match="request must not be None"):
            build_clarification_prompts(None)

    def test_raises_error_for_none_plan(self):
        """Test that request with None plan raises ValueError."""

        # Create a mock request without proper initialization
        class MockRequest:
            plan = None

        with pytest.raises((ValueError, TypeError)):
            # May raise TypeError due to isinstance check or ValueError for None plan
            build_clarification_prompts(MockRequest())

    def test_raises_error_for_invalid_request_type(self):
        """Test that non-ClarificationRequest types raise TypeError."""
        with pytest.raises(TypeError, match="must be a ClarificationRequest instance"):
            build_clarification_prompts({"plan": "not_a_request"})


class TestCleanupAndParseJSON:
    """Tests for the cleanup_and_parse_json function."""

    def test_parses_clean_json(self):
        """Test parsing already clean JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = cleanup_and_parse_json(json_str)

        assert result == {"key": "value", "number": 42}

    def test_removes_markdown_json_fences(self):
        """Test removing ```json fences."""
        json_str = '```json\n{"key": "value"}\n```'
        result = cleanup_and_parse_json(json_str)

        assert result == {"key": "value"}

    def test_removes_plain_markdown_fences(self):
        """Test removing plain ``` fences."""
        json_str = '```\n{"key": "value"}\n```'
        result = cleanup_and_parse_json(json_str)

        assert result == {"key": "value"}

    def test_removes_backticks(self):
        """Test removing backticks around JSON."""
        json_str = '`{"key": "value"}`'
        result = cleanup_and_parse_json(json_str)

        assert result == {"key": "value"}

    def test_extracts_json_from_prose(self):
        """Test extracting JSON embedded in prose."""
        json_str = 'Here is the result: {"key": "value"} Hope this helps!'
        result = cleanup_and_parse_json(json_str)

        assert result == {"key": "value"}

    def test_handles_leading_prose(self):
        """Test removing common prose patterns before JSON."""
        test_cases = [
            'Sure! {"key": "value"}',
            'Here is the JSON: {"key": "value"}',
            'Here\'s the result: {"key": "value"}',
            'The answer is: {"key": "value"}',
            'Certainly, {"key": "value"}',
        ]

        for json_str in test_cases:
            result = cleanup_and_parse_json(json_str)
            assert result == {"key": "value"}, f"Failed for: {json_str}"

    def test_preserves_whitespace_in_strings(self):
        """Test that whitespace within JSON strings is preserved."""
        json_str = '{"text": "  leading and trailing  "}'
        result = cleanup_and_parse_json(json_str)

        assert result["text"] == "  leading and trailing  "

    def test_handles_nested_json(self):
        """Test parsing nested JSON structures."""
        json_str = """```json
{
  "specs": [
    {
      "purpose": "Test",
      "nested": {"key": "value"}
    }
  ]
}
```"""
        result = cleanup_and_parse_json(json_str)

        assert result["specs"][0]["purpose"] == "Test"
        assert result["specs"][0]["nested"]["key"] == "value"

    def test_handles_arrays(self):
        """Test parsing JSON with arrays."""
        json_str = '{"items": ["one", "two", "three"]}'
        result = cleanup_and_parse_json(json_str)

        assert result["items"] == ["one", "two", "three"]

    def test_multiple_cleanup_attempts(self):
        """Test that multiple strategies are attempted."""
        # This should fail first attempt but succeed with extraction
        json_str = 'Some text before {"key": "value"} some text after'
        result = cleanup_and_parse_json(json_str)

        assert result == {"key": "value"}

    def test_max_attempts_parameter(self):
        """Test that max_attempts parameter is respected."""
        invalid_json = "This is not JSON at all"

        with pytest.raises(JSONCleanupError) as exc_info:
            cleanup_and_parse_json(invalid_json, max_attempts=2)

        assert exc_info.value.attempts == 2
        assert "2 attempt(s)" in str(exc_info.value)

    def test_raises_structured_error_on_failure(self):
        """Test that JSONCleanupError is raised with proper attributes."""
        invalid_json = "Not valid JSON {broken}"

        with pytest.raises(JSONCleanupError) as exc_info:
            cleanup_and_parse_json(invalid_json)

        error = exc_info.value
        assert error.message
        assert error.raw_content == invalid_json
        assert error.attempts >= 1

    def test_error_contains_last_parse_error(self):
        """Test that error message contains the last JSON parse error."""
        invalid_json = '{"key": incomplete'

        with pytest.raises(JSONCleanupError) as exc_info:
            cleanup_and_parse_json(invalid_json)

        assert "Last error:" in str(exc_info.value)

    def test_raises_error_for_none_input(self):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="raw_response must not be None"):
            cleanup_and_parse_json(None)

    def test_raises_error_for_empty_input(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="raw_response must not be empty"):
            cleanup_and_parse_json("")

        with pytest.raises(ValueError, match="raw_response must not be empty"):
            cleanup_and_parse_json("   ")

    def test_raises_error_for_invalid_max_attempts(self):
        """Test that invalid max_attempts raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            cleanup_and_parse_json('{"key": "value"}', max_attempts=0)

        with pytest.raises(ValueError, match="max_attempts must be at least 1"):
            cleanup_and_parse_json('{"key": "value"}', max_attempts=-1)

    def test_handles_unicode_in_json(self):
        """Test parsing JSON with unicode characters."""
        json_str = '{"text": "Système 系统 🚀"}'
        result = cleanup_and_parse_json(json_str)

        assert result["text"] == "Système 系统 🚀"

    def test_handles_escaped_quotes_in_json(self):
        """Test parsing JSON with escaped quotes."""
        json_str = r'{"text": "He said \"hello\""}'
        result = cleanup_and_parse_json(json_str)

        assert result["text"] == 'He said "hello"'

    def test_handles_multiline_json(self):
        """Test parsing pretty-printed multiline JSON."""
        json_str = """{
  "key1": "value1",
  "key2": "value2",
  "nested": {
    "inner": "value"
  }
}"""
        result = cleanup_and_parse_json(json_str)

        assert result["key1"] == "value1"
        assert result["nested"]["inner"] == "value"

    def test_real_world_llm_response_example_1(self):
        """Test with realistic LLM response example."""
        llm_response = """```json
{
  "specs": [
    {
      "purpose": "User authentication system",
      "vision": "Secure and user-friendly login",
      "must": ["JWT tokens", "Password hashing"],
      "dont": ["Plain text passwords"],
      "nice": ["Social login"],
      "assumptions": ["HTTPS enabled"]
    }
  ]
}
```"""
        result = cleanup_and_parse_json(llm_response)

        assert result["specs"][0]["purpose"] == "User authentication system"
        assert "JWT tokens" in result["specs"][0]["must"]

    def test_real_world_llm_response_example_2(self):
        """Test with LLM response containing leading explanation."""
        llm_response = """Here is the clarified specification:

```json
{
  "specs": [{"purpose": "Test", "vision": "Test"}]
}
```

Let me know if you need any changes!"""
        result = cleanup_and_parse_json(llm_response)

        assert result["specs"][0]["purpose"] == "Test"

    def test_complex_nested_structure(self):
        """Test parsing complex nested JSON structure matching ClarifiedPlan."""
        json_str = """```json
{
  "specs": [
    {
      "purpose": "Build authentication",
      "vision": "Secure system",
      "must": ["Feature 1", "Feature 2"],
      "dont": ["Antipattern 1"],
      "nice": ["Enhancement 1", "Enhancement 2"],
      "assumptions": ["Assumption 1"]
    },
    {
      "purpose": "Build API",
      "vision": "Fast and reliable",
      "must": [],
      "dont": [],
      "nice": [],
      "assumptions": []
    }
  ]
}
```"""
        result = cleanup_and_parse_json(json_str)

        assert len(result["specs"]) == 2
        assert result["specs"][0]["purpose"] == "Build authentication"
        assert len(result["specs"][0]["must"]) == 2
        assert result["specs"][1]["purpose"] == "Build API"

    def test_nested_braces_in_json(self):
        """Test extraction of JSON with nested braces."""
        json_str = 'Some text {"outer": {"inner": {"deep": "value"}}} trailing text'
        result = cleanup_and_parse_json(json_str)

        assert result["outer"]["inner"]["deep"] == "value"

    def test_braces_in_json_strings(self):
        """Test that braces inside JSON strings are handled correctly."""
        json_str = '{"text": "This {has} braces {in} it", "nested": {"key": "value"}}'
        result = cleanup_and_parse_json(json_str)

        assert result["text"] == "This {has} braces {in} it"
        assert result["nested"]["key"] == "value"

    def test_trailing_prose_removal(self):
        """Test removal of trailing prose after JSON."""
        test_cases = [
            ('{"key": "value"} Let me know if you need anything else!', {"key": "value"}),
            ('{"key": "value"} Hope this helps!', {"key": "value"}),
            ('{"key": "value"}, let me know if you need changes.', {"key": "value"}),
        ]

        for json_str, expected in test_cases:
            result = cleanup_and_parse_json(json_str)
            assert result == expected, f"Failed for: {json_str}"

    def test_multiple_json_objects_extracts_first(self):
        """Test that only the first JSON object is extracted when multiple exist."""
        json_str = '{"first": "object"} {"second": "object"}'
        result = cleanup_and_parse_json(json_str)

        # Should extract only the first complete object
        assert result == {"first": "object"}

    def test_escaped_quotes_in_json(self):
        """Test JSON with escaped quotes inside strings."""
        json_str = r'{"text": "She said \"hello\" to me"}'
        result = cleanup_and_parse_json(json_str)

        assert result["text"] == 'She said "hello" to me'

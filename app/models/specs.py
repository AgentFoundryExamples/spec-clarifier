"""Pydantic models for specification clarification."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class SpecInput(BaseModel):
    """Input specification with questions to be clarified.
    
    Attributes:
        purpose: The purpose of the specification
        vision: The vision statement
        must: List of must-have requirements
        dont: List of don't requirements (things to avoid)
        nice: List of nice-to-have features
        open_questions: List of questions needing clarification
        assumptions: List of assumptions made
    """
    
    model_config = ConfigDict(extra="forbid")
    
    purpose: str = Field(..., description="The purpose of the specification")
    vision: str = Field(..., description="The vision statement")
    must: List[str] = Field(default_factory=list, description="List of must-have requirements")
    dont: List[str] = Field(default_factory=list, description="List of don't requirements")
    nice: List[str] = Field(default_factory=list, description="List of nice-to-have features")
    open_questions: List[str] = Field(default_factory=list, description="List of questions needing clarification")
    assumptions: List[str] = Field(default_factory=list, description="List of assumptions made")


class PlanInput(BaseModel):
    """Input plan containing multiple specifications.
    
    Attributes:
        specs: List of specification inputs
    """
    
    model_config = ConfigDict(extra="forbid")
    
    specs: List[SpecInput] = Field(..., description="List of specification inputs")


class ClarifiedSpec(BaseModel):
    """Clarified specification after processing (no open questions).
    
    Attributes:
        purpose: The purpose of the specification
        vision: The vision statement
        must: List of must-have requirements
        dont: List of don't requirements (things to avoid)
        nice: List of nice-to-have features
        assumptions: List of assumptions made
    """
    
    model_config = ConfigDict(extra="forbid")
    
    purpose: str = Field(..., description="The purpose of the specification")
    vision: str = Field(..., description="The vision statement")
    must: List[str] = Field(default_factory=list, description="List of must-have requirements")
    dont: List[str] = Field(default_factory=list, description="List of don't requirements")
    nice: List[str] = Field(default_factory=list, description="List of nice-to-have features")
    assumptions: List[str] = Field(default_factory=list, description="List of assumptions made")


class ClarifiedPlan(BaseModel):
    """Plan containing clarified specifications.
    
    Attributes:
        specs: List of clarified specifications
    """
    
    model_config = ConfigDict(extra="forbid")
    
    specs: List[ClarifiedSpec] = Field(..., description="List of clarified specifications")


class QuestionAnswer(BaseModel):
    """Answer to a specific question in a specification.
    
    Attributes:
        spec_index: Index of the specification containing the question
        question_index: Index of the question within the specification
        question: The question text
        answer: The answer provided
    """
    
    model_config = ConfigDict(extra="forbid")
    
    spec_index: int = Field(..., ge=0, description="Index of the specification containing the question")
    question_index: int = Field(..., ge=0, description="Index of the question within the specification")
    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer provided")


class ClarificationRequest(BaseModel):
    """Request to clarify specifications with optional answers.
    
    Attributes:
        plan: The plan input with specifications
        answers: List of answers to open questions (optional, currently ignored)
    """
    
    model_config = ConfigDict(extra="forbid")
    
    plan: PlanInput = Field(..., description="The plan input with specifications")
    answers: List[QuestionAnswer] = Field(default_factory=list, description="List of answers to open questions")

"""Service for clarifying specifications."""

from app.models.specs import ClarifiedPlan, ClarifiedSpec, PlanInput


def clarify_plan(plan_input: PlanInput) -> ClarifiedPlan:
    """Transform a PlanInput to ClarifiedPlan by copying required fields.
    
    This function processes each specification in the plan, copying the required
    fields (purpose, vision, must, dont, nice, assumptions) while omitting
    open_questions. Currently, answers are ignored as per the requirements.
    
    Args:
        plan_input: The input plan containing specifications with potential questions
        
    Returns:
        ClarifiedPlan: A plan with clarified specifications (no open_questions)
    """
    clarified_specs = []
    
    for spec_input in plan_input.specs:
        clarified_spec = ClarifiedSpec(
            purpose=spec_input.purpose,
            vision=spec_input.vision,
            must=spec_input.must.copy(),
            dont=spec_input.dont.copy(),
            nice=spec_input.nice.copy(),
            assumptions=spec_input.assumptions.copy(),
        )
        clarified_specs.append(clarified_spec)
    
    return ClarifiedPlan(specs=clarified_specs)

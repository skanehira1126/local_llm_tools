from langchain_core.tools import tool
from pydantic import BaseModel
from pydantic import Field


class PlanInput(BaseModel):
    """Arguments for the plan tool."""

    objective: str = Field(..., description="The goal you are planning for")
    steps: list[str] = Field(..., description="Ordered steps to achieve the objective")


return_format = """\
## Plan
### Objective
{objective}

### Steps
{steps}
"""


@tool(
    name_or_callable="plan",
    args_schema=PlanInput,
    return_direct=True,
)
def plan(*, objective: str, steps: list[str]) -> str:
    """
    Draft a high‑level plan before taking action.
    Use when the user task requires multiple ordered steps.
    """
    steps_str = "\n".join([f"{idx}. {step}" for idx, step in enumerate(steps, 1)])
    return return_format.format(objective=objective, steps=steps_str)

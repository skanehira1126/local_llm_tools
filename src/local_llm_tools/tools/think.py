from langchain_core.tools import tool
from pydantic import BaseModel
from pydantic import Field


class ThinkInput(BaseModel):
    """Arguments for the think tool."""

    thought: str = Field(..., description="A short thought to remember.")


@tool(
    args_schema=ThinkInput,
    return_direct=True,
)
def think(*, thought: str) -> str:
    """
    Use the tool to think about something.
    It will not obtain new information or change the database,
    but just append the thought to the log.
    Use it when complex reasoning or some cache memory is needed.
    """
    return thought

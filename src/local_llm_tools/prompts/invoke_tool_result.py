from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field


# ツール実行結果のTemplate
class ParamsInvokeToolResult(BaseModel):
    tool_name: str = Field(description="The name of tool that is executed")
    arguments: str = Field(description="Arguments")
    tool_result: str = Field(description="The result of tool")


TOOL_RESULT_TEMPLATE = """\
[Tool:{tool_name}]
ARGS:
{arguments}
RESULT:
{tool_result}
"""

TEMPLATE_INVOKE_TOOL_RESULT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            TOOL_RESULT_TEMPLATE,
        )
    ]
)


# ツール実行失敗時のTemplate
class ParamsToolExecuteError(BaseModel):
    name: str = Field(description="The name of tool that failed to execute")


TEMPLATE_TOOL_EXECUTE_ERROR = ChatPromptTemplate.from_messages(
    [
        ("system", "Failed to execute tool: {name}"),
    ]
)

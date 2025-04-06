from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field


# ツール実行結果のTemplate
class ParamsInvokeToolResult(BaseModel):
    name: str = Field(description="The name of tool that is executed")
    tool_result: str = Field(description="The result of tool")


TEMPLATE_INVOKE_TOOL_RESULT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """\
Please use followed results to answer user querys.
These are only internal information for you to generate your answer, \
Please do not disclose every “memo” or “tool result” itself in your answer.

## Result of {name}

```
{tool_result}
```
""",
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

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from pydantic import BaseModel
from pydantic import Field

from local_llm_tools.prompts.invoke_tool_result import TOOL_RESULT_TEMPLATE


class ParamsJudgeUsingTool(BaseModel):
    rendered_tools: str = Field(description="enable tools expressions")
    history: list = Field(description="Recent chat histry")
    documents_description: str = Field(description="Whether the user provides documentation")


PROMPT_COMMON = """\
You are an assistant that has access to the following set of tools.
All tool outputs appear as **system** messages in the following format:

```text
{tool_result_format}
```
""".format(
    tool_result_format=TOOL_RESULT_TEMPLATE.format(
        tool_name="<tool_name>",
        arguments="<arguments>",
        tool_result="<tool_result>",
    )
)

PROMPT_JUDGE_USING_TOOL = """\
Here are the names and descriptions of the available tools:

{rendered_tools}

Based on the chat history, decide which tool—if any—is appropriate and respond **only** with a
valid JSON object containing exactly two keys: `"name"` and `"arguments"`.

- `"arguments"` must be a dictionary whose **keys** are the argument names and whose **values** are the corresponding inputs.
- Return *no* text, comments, or formatting beyond this JSON.
- If none of the tools apply, return:
  {{"name": "no_tool_needed", "arguments": {{}} }}

For reference, you also know the following context:
- {documents_description}

"""

TEMPLATE_JUDGE_USING_TOOL = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_COMMON),
        ("system", PROMPT_JUDGE_USING_TOOL),
        MessagesPlaceholder("history"),
    ],
)

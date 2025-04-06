from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pydantic import Field


class ParamsJudgeUsingTool(BaseModel):
    rendered_tools: str = Field(description="enable tools expressions")
    chat_history: str = Field(description="Recent chat histry")
    documents_description: str = Field(description="Whether the user provides documentation")


PROMPT_JUDGE_USING_TOOL = """\
You are an assistant that has access to the following set of tools.
Here are the names and descriptions for each tool:

{rendered_tools}

Recent conversation history:
{chat_history}

Given the user input, determine which tool to use and output ONLY a valid JSON object \
with exactly two keys: 'name' and 'arguments'.

The `arguments` should be a dictionary, with keys corresponding \
to the argument names and the values corresponding to the requested values.
Do not include any additional text, commentary, or formatting.
If none of the tools are applicable to the input, output the JSON object: \
{{"name": "no_tool_needed", "arguments": {{}} }}

For the purpose of determining the appropriate tool, you can use the following information.
- {documents_description}
"""

TEMPLATE_JUDGE_USING_TOOL = ChatPromptTemplate.from_messages(
    [("system", PROMPT_JUDGE_USING_TOOL)],
)

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field

from local_llm_tools.utils.llm import ExtractKeyParser


class InputState(BaseModel):
    thought: str = Field(description="A thought to think about.")


class State(InputState):
    output: str = Field(description="Internal thoudght process.")


SYSTEM_PROMPT = """\
Perform a detailed internal thought process on the user's input.

Do not include any meta instructions or tool usage details in the final output;\
focus only on key reasoning insights.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{thought}"),
    ]
)

# output format
"[Internal Thought Process Output]  \n"
"For the given user input, the following internal thought process was executed:\n"
"\n"
"{thought.text()}\n"
"\n"
"[Instructions]  \n"
"Using the above internal thought process as context,"
"generate the most appropriate and comprehensive response to the user's query."
"For answer to your prompt"


class ArgSchema(BaseModel):
    thought: str = Field(description="A thought to think about")


class ThinkGraph:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        self.graph = self.build()

    def build(self):
        graph_builder = StateGraph(State, input=InputState)

        graph_builder.add_node("think", self._think)

        graph_builder.add_edge(START, "think")
        graph_builder.add_edge("think", END)

        return graph_builder.compile()

    def as_tool(self):
        return (self.graph | ExtractKeyParser("output")).as_tool(
            name="think",
            description=(
                "Use the tool to think about something."
                "It will not obtain new information or change the "
                "database, but just append the thought to the log."
                "Use it when complex reasoning or some cache memory is needed."
            ),
            args_schema=ArgSchema,
        )

    def _think(self, state: State):
        chain = PROMPT_TEMPLATE | self.llm | StrOutputParser()
        output = chain.invoke({"thought": state.thought})

        return {"output": output}

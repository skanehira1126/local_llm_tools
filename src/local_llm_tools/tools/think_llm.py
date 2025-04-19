import re

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field

from local_llm_tools.utils.llm import ExtractKeyParser


class ThoughtExtractor(StrOutputParser):
    pattern: re.Pattern = re.compile(r"<THOUGHT>(.*?)</THOUGHT>", re.S)

    def parse(self, text: str) -> str:
        match = self.pattern.search(text)
        if not match:
            raise ValueError("No <THOUGHT> block found")
        return match.group(1).strip()


class InputState(BaseModel):
    thought: str = Field(description="A thought to think about.")
    history: str = Field(description="chat history")


class State(InputState):
    output: str = Field(description="Internal thought process.")


SYSTEM_PROMPT = """\
You are _THINK_, an internal scratch‑pad.  
Your entire output will be stored **only for the model’s private reasoning** and will never be shown to the end‑user.  
Therefore:

• Write down your reasoning steps as a concise bullet list (≤ 6 lines, ≤ 150 tokens).  
• Do **NOT** mention tool names, model names, or meta‑instructions.  
• No final answer—just the reasoning itself.  
• Wrap the whole content between the tags <THOUGHT> ... </THOUGHT>.


"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("system", "Conversation snapshot:\n{history}"),
        ("human", "{thought}"),
    ]
)


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
                # =============
                # "Internal scratch‑pad for the assistant. "
                # "Use this tool to jot down private reasoning, intermediate results, or a step‑by‑step plan. "
                # "It does NOT fetch new information or modify any data—just appends the note to an internal log. "
                # "Call it whenever complex reasoning, caching values, or task‑planning is helpful."
            ),
            args_schema=ArgSchema,
        )

    def _think(self, state: State):
        chain = PROMPT_TEMPLATE | self.llm | ThoughtExtractor()
        output = chain.invoke({"thought": state.thought, "history": state.history})

        return {"output": output}

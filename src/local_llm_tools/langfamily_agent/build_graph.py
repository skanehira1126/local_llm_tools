from logging import getLogger
from typing import Annotated
from typing import Literal
from typing import TypedDict

from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import render_text_description
from langchain_core.tools.base import BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# 多分これがメッセージ追加してくれるらしい
# from langgraph.graph.message import add_messages
from local_llm_tools.langfamily_agent.utils import get_role_of_message


logger = getLogger(__name__)


# class State(TypedDict):
#     # Messages have the type "list". The `add_messages` function
#     # in the annotation defines how this state key should be updated
#     # (in this case, it appends messages to the list, rather than overwriting them)
#     messages: Annotated[list, add_messages]


def should_continue(state: MessagesState) -> Literal["tools", END]:
    """
    Tool分岐するやつ
    """
    logger.info("Called should_continue node")
    messages = state["messages"]
    last_message = messages[-1]
    if get_role_of_message(last_message) != "system" and last_message.tool_calls:
        return "tools"
    else:
        return END


def build_graph(llm, tool_node: ToolNode):
    def chat(state: MessagesState):
        logger.info("Called chat node")
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder = StateGraph(MessagesState)

    # Nodes
    graph_builder.add_node("chat", chat)
    graph_builder.add_node("tools", tool_node)

    # Edge
    # 終了判定はshould_continueが持ってる
    graph_builder.add_edge(START, "chat")
    graph_builder.add_conditional_edges("chat", should_continue)
    graph_builder.add_edge("tools", "chat")

    # Memory
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


# ツール選択のSystem prompt
system_prompt = """\
You are an assistant that has access to the following set of tools. 
Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. 
Return your response as a JSON blob with 'name' and 'arguments' keys.

The `arguments` should be a dictionary, with keys corresponding 
to the argument names and the values corresponding to the requested values.

If you cannnot undertand to use which tools, please response JSON blob with 'name' key is 'unknown' and 'arguments' key is empty dictionary.
"""


def build_graph_no_tools_use_llm(llm, tools: list[BaseTool]):
    class MyMessageState(TypedDict):
        messages: Annotated[list, add_messages]
        tool_call_request: dict

    def chat(state: MyMessageState):
        logger.info("Called chat node")
        return {"messages": [llm.invoke(state["messages"])]}

    def judge_tool_use(state: MyMessageState) -> Literal["chat", "tools", END]:
        """
        ツールを選択する.
        """
        last_message = state["messages"][-1]

        if get_role_of_message(last_message) == "system":
            goto = END
            update = None
        else:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("user", "{input}")]
            )
            rendered_tools = render_text_description(tools)
            model = ChatOllama(model=llm.model, temperature=0, format="json")
            chain = prompt | model | JsonOutputParser()
            tool_call_request = chain.invoke(
                {"input": last_message.content, "rendered_tools": rendered_tools}
            )

            goto = "chat" if tool_call_request["name"] == "unknown" else "tools"
            update = (
                None
                if tool_call_request["name"] == "unknown"
                else {"tool_call_request": tool_call_request}
            )

        return Command(
            update=update,
            goto=goto,
        )

    def invoke_tool(state: MyMessageState, config: RunnableConfig | None = None):
        """A function that we can use the perform a tool invocation.

        Args:
            tool_call_request: a dict that contains the keys name and arguments.
                The name must match the name of a tool that exists.
                The arguments are the arguments to that tool.
            config: This is configuration information that LangChain uses that contains
                things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

        Returns:
            output from the requested tool
        """
        tool_call_request = state.get("tool_call_request")
        tool_name_to_tool = {tool.name: tool for tool in tools}
        name = tool_call_request["name"]
        requested_tool = tool_name_to_tool[name]
        try:
            tool_result = requested_tool.invoke(tool_call_request["arguments"], config=config)
            return {
                "messages": [
                    SystemMessage(
                        f"Result of {name} is {tool_result}. "
                        "Please use these results to answer user questions.",
                    )
                ]
            }
        except:
            logger.exception(f"Arguments: {tool_call_request['arguments']}")
            return {"messages": [SystemMessage(f"Failed to execute tool: {name}")]}

    # グラフ構築
    graph_builder = StateGraph(MyMessageState)

    # Nodes
    graph_builder.add_node("chat", chat)
    graph_builder.add_node("chat_end", chat)
    graph_builder.add_node("judge_tool_use", judge_tool_use)
    graph_builder.add_node("tools", invoke_tool)

    # Edge
    # 終了判定はshould_continueが持ってる
    graph_builder.add_edge(START, "judge_tool_use")
    graph_builder.add_edge("tools", "chat_end")
    graph_builder.add_edge("chat_end", END)

    # Memory
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph

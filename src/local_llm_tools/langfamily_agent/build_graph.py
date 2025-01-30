from logging import getLogger
from typing import Annotated
from typing import Literal
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# 多分これがメッセージ追加してくれるらしい
# from langgraph.graph.message import add_messages
from local_llm_tools.langfamily_agent.utils import get_role_of_message


logger = getLogger(__name__)


# class State(TypedDict):
#     # Messages have the type "list". The `add_messages` function
#     # in the annotation defines how this state key should be updated
#     # (in this case, it appends messages to the list, rather than overwriting them)
#     messages: Annotated[list, add_messages]


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    """
    継続判断するやつ？
    """
    logger.debug("Called should_continue node")
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    try:
        print(messages[-2], messages[-1])
    except:
        pass
    if get_role_of_message(last_message) != "system" and last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


def build_graph(llm, tool_node: ToolNode):
    def chat(state: MessagesState):
        logger.debug("Called chat node")
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

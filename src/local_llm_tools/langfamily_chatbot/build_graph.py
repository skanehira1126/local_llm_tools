from typing import Annotated
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

# 多分これがメッセージ追加してくれるらしい
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def build_graph(llm):
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # Memory
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph

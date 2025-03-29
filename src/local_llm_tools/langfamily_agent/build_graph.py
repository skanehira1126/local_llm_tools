from logging import getLogger
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypedDict

from langchain.schema import SystemMessage
from langchain.tools import tool
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import render_text_description
from langchain_core.tools.base import BaseTool
from langchain_openai import ChatOpenAI

# text splitter どう管理するかは要検討
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel

from local_llm_tools.langfamily_agent.utils import get_role_of_message
from local_llm_tools.tools.read_documents import SearchDocGraph


logger = getLogger(__name__)


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


@tool
def think() -> str:
    """
    Use the tool to think about something.
    It will not obtain new information or change the
    database, but just append the thought to the log.
    Use it when complex reasoning or some cache memory
    is needed.
    """
    # dummy
    pass


class RequestDocs(BaseModel):
    query: str


class MyMessageState(TypedDict):
    messages: Annotated[list, add_messages]
    docs: dict[str, str] | None
    tool_call_request: dict


class GemmaGraph:
    """
    Ollama Gemma用のGhraph
    """

    def __init__(
        self,
        llm_chat: ChatOpenAI,
        llm_structured_output: ChatOpenAI,
        tools: list[BaseTool],
        is_enable_think_tool: bool = False,
    ):
        # 利用するLLM
        self.llm_chat = llm_chat
        self.llm_structured_output = llm_structured_output

        # graphの作成
        self.tools = tools
        if is_enable_think_tool:
            self._tools.append(think)

        # Documentを読み込むための変数
        self.search_docs_graph = SearchDocGraph(self.llm_chat)

        self.graph = self.build_graph()

    def __getattr__(self, name):
        """
        StateGraphのWrapperとして活用する前提の実装
        """
        return getattr(self.graph, name)

    def build_graph(self):
        # グラフ構築
        graph_builder = StateGraph(MyMessageState)

        # Nodes
        # graph_builder.add_node("rooting_judge_tools", self._rooting_judge_tools)
        graph_builder.add_node("chat", self._chat)
        graph_builder.add_node("think", self._think)
        graph_builder.add_node("judge_tool_use", self._judge_tool_use)
        graph_builder.add_node("invoke_tools", self._invoke_tool)
        graph_builder.add_node("remove_messages", self._remove_messages)

        # Edge
        # toolがないときはただのchat
        graph_builder.add_conditional_edges(START, self._rooting_judge_tools)
        # graph_builder.add_edge("judge_tool_use", "invoke_tools")
        graph_builder.add_edge("invoke_tools", "chat")
        graph_builder.add_edge("think", "chat")
        graph_builder.add_edge("summarize_docs", "chat")
        graph_builder.add_edge("chat", "remove_messages")
        graph_builder.add_edge("remove_messages", END)

        # Memory
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        return graph

    def _rooting_judge_tools(self, state: MyMessageState):
        """
        toolの有無によって向き先を変える
        """
        print(state)
        if len(self.tools) or self.has_docs(state):
            return "judge_tool_use"
        else:
            return "chat"

    def _chat(self, state: MyMessageState):
        """
        chat関数
        """
        logger.info("Called chat node")
        return {"messages": [self.llm_chat.invoke(state["messages"])]}

    def _remove_messages(self, state: MyMessageState):
        """
        削除する必要のあるメッセージを削除する
        """
        remove_messages = [
            RemoveMessage(id=msg.id)
            for msg in state["messages"]
            if msg.response_metadata.get("is_delete", False)
        ]
        return {
            "messages": remove_messages,
        }

    def _think(self, state: MyMessageState):
        logger.info("Thinking....")
        internal_prompt = (
            "Perform a detailed internal thought process on the user's input. "
            "Do not include any meta instructions or tool usage details in the final output; "
            "focus only on key reasoning insights."
        )

        thought = self.llm_chat.invoke(
            state["messages"][-3:]
            + [
                SystemMessage(internal_prompt),
            ]
        )

        return {
            "messages": [
                SystemMessage(
                    "[Internal Thought Process Output]  \n"
                    "For the given user input, the following internal thought process was executed:\n"
                    "\n"
                    f"{thought.text()}\n"
                    "\n"
                    "[Instructions]  \n"
                    "Using the above internal thought process as context,"
                    "generate the most appropriate and comprehensive response to the user's query."
                    "For answer to your prompt",
                )
            ]
        }

    def has_docs(self, state: MyMessageState):
        return not (state["docs"] is None or len(state["docs"]) == 0)

    def get_tools(self, state: MyMessageState):
        if self.has_docs(state):
            return self.tools + [self.search_docs_graph.as_tool()]
        else:
            return self.tools

    def _judge_tool_use(self, state: MyMessageState) -> Literal["chat", "tools", END]:
        """
        Promptを元にToolを利用するか判断する
        """
        # ツール選択のSystem prompt
        system_prompt = (
            "You are an assistant that has access to the following set of tools.\n"
            "Here are the names and descriptions for each tool:\n"
            "\n"
            "{rendered_tools}\n"
            "\n"
            "Given the user input, determine which tool to use and output ONLY a valid JSON object "
            """with exactly two keys: "name" and "arguments".\n"""
            "The `arguments` should be a dictionary, with keys corresponding "
            "to the argument names and the values corresponding to the requested values."
            "Do not include any additional text, commentary, or formatting.\n"
            "If none of the tools are applicable to the input,"
            """output the JSON object: {{"name": "unknown", "arguments": {{}} }}.\n"""
            "{documents_description}"
        )
        # 変数を作成
        tools = self.get_tools(state)
        if self.has_docs(state):
            documents_description = "User provided documents are available."
        else:
            documents_description = "There are NO documents from user."

        rendered_tools = render_text_description(tools)
        tool_names = [t.name for t in tools] or ["dummy"]
        logger.info(f"Enable tools: {tool_names}")

        class ToolCall(BaseModel):
            """
            呼び出すツール判定のための型
            """

            name: Literal[*tool_names]
            arguments: dict[str, Any]

        # 最後がシステムメッセージの場合はChat終了
        if get_role_of_message(state["messages"][-1]) == "system":
            goto = END
            update = None
        else:
            prompt = ChatPromptTemplate.from_messages(
                state["messages"][-3:]
                + [
                    ("system", system_prompt),
                ]
            )

            logger.info(f"Tools: {rendered_tools}")
            logger.info(f"Document description: {documents_description}")

            chain = prompt | self.llm_structured_output | JsonOutputParser()
            tool_call_request = chain.invoke(
                {"rendered_tools": rendered_tools, "documents_description": documents_description},
                response_format=ToolCall,
            )

            if tool_call_request["name"] == "unknown":
                goto = "chat"
                update = None
            elif tool_call_request["name"] == "Search Documents":
                goto = "invoke_tools"
                tool_call_request.update(
                    {
                        "arguments": {
                            "query": state["messages"][-1].text(),
                            "docs": state["docs"],
                        }
                    }
                )
                update = {"tool_call_request": tool_call_request}
            elif tool_call_request["name"] == "think":
                goto = "think"
                update = {"query": state}
            else:
                goto = "invoke_tools"
                update = {"tool_call_request": tool_call_request}

        return Command(
            update=update,
            goto=goto,
        )

    def _invoke_tool(self, state: MyMessageState, config: RunnableConfig | None = None):
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

        tools = self.get_tools(state)
        tool_name_to_tool = {tool.name: tool for tool in tools}

        name = tool_call_request["name"]
        requested_tool = tool_name_to_tool[name]
        try:
            tool_result = requested_tool.invoke(tool_call_request["arguments"], config=config)
            return {
                "messages": [
                    SystemMessage(
                        "Please use followed results to answer user querys.  \n"
                        "These are only internal information for you to generate your answer, "
                        "Please do not disclose every “memo” or “tool result” itself in your answer."
                        f"\n\n## Result of {name} \n\n {tool_result['summary']} "
                    )
                ]
            }
        except:
            logger.exception(f"Arguments: {tool_call_request['arguments']}")
            return {"messages": [SystemMessage(f"Failed to execute tool: {name}")]}

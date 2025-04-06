from logging import getLogger
from typing import Annotated
from typing import Literal
from typing import TypedDict

from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
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
from pydantic import Field

from local_llm_tools.langfamily_agent.utils import get_role_of_message
from local_llm_tools.prompts.invoke_tool_result import TEMPLATE_INVOKE_TOOL_RESULT
from local_llm_tools.prompts.invoke_tool_result import TEMPLATE_TOOL_EXECUTE_ERROR
from local_llm_tools.prompts.judge_using_tool import TEMPLATE_JUDGE_USING_TOOL
from local_llm_tools.tools.read_documents import SearchDocGraph
from local_llm_tools.tools.think import ThinkGraph
from local_llm_tools.utils.llm import OllamaTokenCounter
from local_llm_tools.utils.llm import render_text_description


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


class RequestDocs(BaseModel):
    query: str


class MyMessageState(TypedDict):
    messages: Annotated[list, add_messages]
    docs: dict[str, str] | None
    tool_call_request: dict


class ToolCallRecuestState(BaseModel):
    name: str
    arguments: dict
    has_docs: bool


class GemmaGraph:
    """
    Ollama Gemma用のGhraph
    """

    def __init__(
        self,
        llm_chat: ChatOpenAI,
        tools: list[BaseTool],
        is_enable_think_tool: bool = False,
        llm_common_params: dict | None = None,
    ):
        # 利用するLLM
        self.llm_chat = llm_chat

        # graphの作成
        self.tools = tools
        if is_enable_think_tool:
            self.tools.append(
                ThinkGraph(ChatOpenAI(**llm_common_params, temperature=1.0)).as_tool()
            )

        # LLMを利用するTook
        self.search_docs_graph = SearchDocGraph(
            ChatOpenAI(**llm_common_params, temperature=1.0, max_tokens=1000),
            token_counter=OllamaTokenCounter(llm_common_params["model"]),
        )

        self.graph = self.build_graph()

        self.llm_common_params = llm_common_params or {}

    def build_graph(self):
        # グラフ構築
        graph_builder = StateGraph(MyMessageState)

        # Nodes
        graph_builder.add_node("chat", self._chat)
        graph_builder.add_node("judge_tool_use", self._judge_tool_use)
        graph_builder.add_node("invoke_tools", self._invoke_tool)
        graph_builder.add_node("remove_messages", self._remove_messages)

        # Edge
        # toolがないときはただのchat
        graph_builder.add_conditional_edges(START, self._rooting_judge_tools)
        graph_builder.add_edge("invoke_tools", "chat")
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

    def has_docs(self, state: MyMessageState):
        """
        ドキュメントがあるかどうかを判定する
        """
        return not (state["docs"] is None or len(state["docs"]) == 0)

    def get_tools(self, has_docs: bool):
        if has_docs:
            return self.tools + [self.search_docs_graph.as_tool()]
        else:
            return self.tools

    def _judge_tool_use(self, state: MyMessageState) -> Literal["chat", "tools", END]:
        """
        Promptを元にToolを利用するか判断する
        """
        # ツールリストの作成
        tools = self.get_tools(self.has_docs(state))
        rendered_tools = render_text_description(tools)

        # ドキュメントの有無の情報を取得
        if self.has_docs(state):
            documents_description = "User provided **TEXT** documents are available."
            # documents_description = "User documents are available, but should only be used if relevant to the question."
        else:
            documents_description = "There are NO text documents from user."

        # 直近の会話履歴を取得
        chat_history = "\n".join([f"- {msg.type}: {msg.text()}" for msg in state["messages"][-5:]])

        tool_names = [t.name for t in tools] or ["no_tool_needed"]
        logger.info(f"Enable tools: {tool_names}")

        class ToolCall(BaseModel):
            """
            呼び出すツール判定のための型
            """

            # no_tool_neededもかなり重要みたい。ちゃんとStructure outputの制限として機能していそう
            name: Literal[*tool_names, "no_tool_needed"] = Field(description="Tool name")
            arguments: dict[str, str | int | float] = Field(
                description="Arguments to execute tool"
            )

        # 最後がシステムメッセージの場合はChat終了
        if get_role_of_message(state["messages"][-1]) == "system":
            goto = END
            update = None
        else:
            logger.info(f"Tools: {rendered_tools}")
            logger.info(f"Document description: {documents_description}")

            llm_structured_output = ChatOpenAI(
                **self.llm_common_params,
                temperature=0,
            )
            chain = TEMPLATE_JUDGE_USING_TOOL | llm_structured_output.with_structured_output(
                ToolCall
            )
            tool_call_request = chain.invoke(
                {
                    "rendered_tools": rendered_tools,
                    "chat_history": chat_history,
                    "documents_description": documents_description,
                },
            )

            if tool_call_request.name == "no_tool_needed":
                goto = "chat"
                update = None
            else:
                goto = "invoke_tools"
                update = tool_call_request.model_dump()
                update["has_docs"] = self.has_docs(state)
                if tool_call_request.name == "Search Documents":
                    update["arguments"]["docs"] = state["docs"]

        return Command(
            update=update,
            goto=goto,
        )

    def _invoke_tool(self, state: ToolCallRecuestState, config: RunnableConfig | None = None):
        """A function that we can use the perform a tool invocation.

        Args:
            tool_call_request: a dict that contains the keys name and arguments.
                The name must match the name of a tool that exists.
                The arguments are the arguments to that tool.
            config: This is configuration information that LangChain uses that contains
                things like callbacks, metadata, etc.
                See LCEL documentation about RunnableConfig.

        Returns:
            output from the requested tool
        """
        tools = self.get_tools(state.has_docs)
        tool_name_to_tool = {tool.name: tool for tool in tools}

        requested_tool = tool_name_to_tool[state.name]
        try:
            tool_result = requested_tool.invoke(state.arguments, config=config)
            return {
                "messages": TEMPLATE_INVOKE_TOOL_RESULT.invoke(
                    {"name": state.name, "tool_result": tool_result}
                ).to_messages()
            }
        except:
            logger.exception(f"Arguments: {state.arguments}")
            return {
                "messages": TEMPLATE_TOOL_EXECUTE_ERROR.invoke({"name": state.name}).to_messages()
            }

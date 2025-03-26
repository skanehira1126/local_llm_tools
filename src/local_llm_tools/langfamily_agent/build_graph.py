from logging import getLogger
import operator
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypedDict

from langchain.schema import HumanMessage
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
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
def summarize_docs(query: str) -> str:
    """
    ユーザの問い合わせに対し回答をするためにファイルが必要であると考えられる場合に
    ファイルを読んで回答を生成する関数

    """
    # dummy
    pass


class RequestDocs(BaseModel):
    query: str


class MyMessageState(TypedDict):
    messages: Annotated[list, add_messages]
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
        is_enable_think_node: bool = False,
    ):
        # 利用するLLM
        self.llm_chat = llm_chat
        self.llm_structured_output = llm_structured_output

        # graphの作成
        self._tools = tools
        self.is_enable_think_node = is_enable_think_node

        # Documentを読み込むための変数
        self.query_doc_graph = None

        self.graph = self.build_graph()

        logger.info(f"Enable tools: {self.tool_names}")

    def __getattr__(self, name):
        """
        StateGraphのWrapperとして活用する前提の実装
        """
        return getattr(self.graph, name)

    @property
    def enable_read_docs(self):
        return self.query_doc_graph is not None

    @property
    def tools(self):
        if self.enable_read_docs:
            return self._tools + [summarize_docs]
        else:
            return self._tools

    @property
    def tool_names(self):
        return [t.name for t in self.tools] or ["dummy"]

    def register_docs(self, docs: dict[str, str] | None):
        # textを読み込んだ上での回答を生成するためのGraph
        if docs is None:
            self.query_doc_graph = None
        else:
            logger.info("Set docs: {}".format(", ".join(docs.keys())))
            self.query_doc_graph = QueryDocGraph(self.llm_chat, docs)

    def build_graph(self):
        # グラフ構築
        graph_builder = StateGraph(MyMessageState)

        # Nodes
        # graph_builder.add_node("rooting_judge_tools", self._rooting_judge_tools)
        graph_builder.add_node("chat", self._chat)
        graph_builder.add_node("think", self._think)
        graph_builder.add_node("judge_tool_use", self._judge_tool_use)
        graph_builder.add_node("tools", self._invoke_tool)
        graph_builder.add_node("summarize_docs", self._summarize_docs)
        graph_builder.add_node("remove_messages", self._remove_messages)

        # Edge
        # toolがないときはただのchat
        graph_builder.add_conditional_edges(START, self._rooting_judge_tools)
        graph_builder.add_edge("tools", "chat")
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
        if len(self.tools):
            return "judge_tool_use"
        elif self.is_enable_think_node:
            return "think"
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

        class ToolCall(BaseModel):
            """
            呼び出すツール判定のための型
            """

            name: Literal[*self.tool_names]
            arguments: dict[str, Any]

        # 最新のメッセージ
        last_message = state["messages"][-1]

        if get_role_of_message(last_message) == "system":
            goto = END
            update = None
        else:
            prompt = ChatPromptTemplate.from_messages(
                state["messages"][-3:]
                + [
                    ("system", system_prompt),
                ]
            )
            # 変数を作成
            rendered_tools = render_text_description(self.tools)
            if self.query_doc_graph is None:
                documents_description = "There are NO documents from user."
            else:
                documents_description = "There are documents from user."

            logger.info(f"Tools: {rendered_tools}")
            logger.info(f"Document description: {documents_description}")

            chain = prompt | self.llm_structured_output | JsonOutputParser()
            tool_call_request = chain.invoke(
                {"rendered_tools": rendered_tools, "documents_description": documents_description},
                response_format=ToolCall,
            )

            if tool_call_request["name"] == "unknown":
                goto = "think"
                update = None
            elif tool_call_request["name"] == "summarize_docs":
                goto = "summarize_docs"
                update = {"query": state["messages"][-1].text()}
            else:
                goto = "tools"
                update = {"tool_call_request": tool_call_request}

        return Command(
            update=update,
            goto=goto,
        )

    def _summarize_docs(self, state: RequestDocs):
        """
        ドキュメントを読み込んで要約する関数
        """
        logger.info(f"Summarize documents: query {state.query}")
        return {
            "messages": [
                SystemMessage(
                    self.query_doc_graph.invoke({"query": state.query})["summary"],
                    response_metadata={"is_delete": True},
                ),
            ],
        }

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
        tool_name_to_tool = {tool.name: tool for tool in self.tools}
        name = tool_call_request["name"]
        requested_tool = tool_name_to_tool[name]
        try:
            tool_result = requested_tool.invoke(tool_call_request["arguments"], config=config)
            return {
                "messages": [
                    SystemMessage(
                        f"Result of {name} is {tool_result}. "
                        "Please use these results to answer user querys."
                        "These are only internal information for you to generate your answer,"
                        "Please do not disclose every “memo” or “tool result” itself in your answer."
                    )
                ]
            }
        except:
            logger.exception(f"Arguments: {tool_call_request['arguments']}")
            return {"messages": [SystemMessage(f"Failed to execute tool: {name}")]}


class QueryState(BaseModel):
    content: str = Field(
        description=(
            "The target document from which to retrieve information necessary for "
            "answering the user's query."
        )
    )
    query: str = Field(description="The query provided by the user.")


class InputState(BaseModel):
    query: str = Field(description="Question related to the document")


class SplitDocs(BaseModel):
    """
    中間処理で利用する
    """

    doc_name: str = Field(description="Document name")
    query: str = Field(description="Question related to the document")
    contents: list[str] = Field(description="Segments of the document after splitting")


class OutputState(BaseModel):
    information: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="Document information used to answer the user query"
    )
    summary: str


class OverallState(InputState, OutputState):
    complete_docs: list[str] = Field(
        default_factory=list,
        description="Document list to complete searching",
    )


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """あなたは、与えられた文書の一部（チャンク）と、文書全体に対するユーザの質問を入力として受け取るエージェントです。以下のルールに従って出力を行ってください。

1. あなたの目的は、チャンク内から「ユーザの質問に関連する情報」を抜き出し、回答作成に役立つ要点をリスト化することです。回答そのものは作成しません。
2. 抽出した情報がある場合は箇条書き形式で要点をまとめてください。内容に引用元が分かるようであれば行番号や見出しなどを簡単に示してください。
3. 質問に関連する情報がチャンク内に全く含まれない場合は、`null`のみを出力してください。
4. 質問移管する情報が含まれる場合は出力の先頭には必ず「#### Relevance Extraction」という見出しをつけてください。
5. この抽出以外の目的や情報を付与しないでください。与えられたチャンク以外の知識や推測も付け加えないでください。
6. 冗長な説明や解釈、推測は行わず、チャンク内の情報のみを正確に反映してください。

これらの指示に反する出力は行わないでください。
""",
        ),
        (
            "human",
            """\
## 文書のチャンク
{document_chunk}

## ユーザの質問
{user_query}
""",
        ),
    ]
)


class QueryDocGraph:
    """
    Documentを読み込み、質問に回答する情報を探すグラフ
    """

    def __init__(self, llm: ChatOpenAI, docs: dict[str, str]):
        self.llm = llm
        self.docs = docs

        # text splitパラメータ
        chunk_size = 2000
        chunk_overlap = 400
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.length_function,
        )

        self.graph = self.build_graph()

    def length_function(self, documents: list[str]) -> int:
        """Get number of tokens for input contents."""
        return sum(self.llm.get_num_tokens(doc) for doc in documents)

    def __getattr__(self, name):
        """
        StateGraphのWrapperとして活用する前提の実装
        """
        return getattr(self.graph, name)

    def build_graph(self):
        graph = StateGraph(OverallState, input=InputState, output=OutputState)

        # node
        graph.add_node("split_docs", self._split_docs)
        graph.add_node("search_in_doc", self._search_in_doc)
        graph.add_node("generate_answer", self._generate_answer)

        # edge
        graph.add_edge(START, "split_docs")
        graph.add_edge("search_in_doc", "split_docs")
        graph.add_edge("generate_answer", END)

        return graph.compile()

    def _split_docs(self, state: OverallState):
        """
        Documentを調査を行う形に分割して、回答探索 or 回答生成に振り分ける
        """
        logger.info("START reading documents.")

        if len(self.docs) == len(state.complete_docs):
            goto = "generate_answer"
            update = None
        else:
            target_doc_name = [doc for doc in self.docs if doc not in state.complete_docs][0]
            target_doc_content = self.docs[target_doc_name]
            contents = self.text_splitter.split_text(target_doc_content)

            goto = "search_in_doc"
            update = {
                "doc_name": target_doc_name,
                "query": state.query,
                "contents": contents,
            }

        return Command(
            goto=goto,
            update=update,
        )

    def _search_in_doc(self, state: SplitDocs):
        """
        Documentごとに分割し質問に対する探索を行う
        """
        logger.info("Search information from documents")
        information = []
        for content in state.contents:
            chain = PROMPT_TEMPLATE | self.llm
            response = chain.invoke({"user_query": state.query, "document_chunk": content})
            logger.info(
                "Check text is null: {}".format(response.text().replace("\n", "") == "null")
            )
            logger.info(response.text())
            if response.text().replace("\n", "") == "null":
                pass
            else:
                information.append(response.text())

        return {
            "complete_docs": [state.doc_name],
            "information": information,
        }

    def _generate_answer(self, state: OverallState):
        """
        入力された文書を探索した結果を用いて回答を生成する
        """
        logger.info("Generate answer from infomation")
        logger.info(f"The number of information: {len(state.information)}")
        system_prompt = (
            "ユーザの質問に回答するためにファイルから回答に必要な情報を背景情報として抽出しました"
            "\n## 背景情報\n\n "
            + "-----\n\n".join(state.information)
            + "\n## 依頼\n"
            + "以上の背景情報を活用して、ユーザの質問に回答してください。\n"
        )
        return {"summary": system_prompt}

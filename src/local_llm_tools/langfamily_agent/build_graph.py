from logging import getLogger
import operator
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypedDict

from langchain.schema import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import render_text_description
from langchain_core.tools.base import BaseTool
from langchain_openai import ChatOpenAI

# text splitter どう管理するかは要検討
from langchain_text_splitters import CharacterTextSplitter
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

# 多分これがメッセージ追加してくれるらしい
# from langgraph.graph.message import add_messages
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


class MyMessageState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_request: dict


class GemmaGraph:
    """
    Ollama Gemma用のGhraph
    """

    def __init__(
        self, llm_chat: ChatOpenAI, llm_structured_output: ChatOpenAI, tools: list[BaseTool]
    ):
        # 利用するLLM
        self.llm_chat = llm_chat
        self.llm_structured_output = llm_structured_output

        # graphの作成
        self.tools = tools
        self.tool_names = [t.name for t in tools] or ["dummy"]
        logger.info(f"Enable tools: {self.tool_names}")

        # textを読み込んだ上での回答を生成するためのGraph
        self.query_doc_graph = QueryDocGraph(llm_chat)

        class ToolCall(BaseModel):
            """
            呼び出すツール判定のための型
            """

            name: Literal[*self.tool_names]
            arguments: dict[str, Any]

        self.ToolCall = ToolCall
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
        graph_builder.add_node("chat", self._chat)
        # graph_builder.add_node("chat_end", self._chat)
        graph_builder.add_node("think", self._think)
        graph_builder.add_node("judge_tool_use", self._judge_tool_use)
        graph_builder.add_node("tools", self._invoke_tool)

        # Edge
        # toolがないときはただのchat
        if len(self.tools):
            graph_builder.add_edge(START, "judge_tool_use")
            graph_builder.add_edge("tools", "chat")
            graph_builder.add_edge("think", "chat")
            graph_builder.add_edge("chat", END)
        else:
            graph_builder.add_edge(START, "think")
            graph_builder.add_edge("think", "chat")
            graph_builder.add_edge("chat", END)

        # Memory
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        return graph

    def _chat(self, state: MyMessageState):
        """
        chat関数
        """
        logger.info("Called chat node")
        return {"messages": [self.llm_chat.invoke(state["messages"])]}

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
                    "generate the most appropriate and comprehensive response to the user's question."
                    "For answer to your prompt",
                )
            ]
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
                        "Please use these results to answer user questions."
                        "These are only internal information for you to generate your answer,"
                        "Please do not disclose every “memo” or “tool result” itself in your answer."
                    )
                ]
            }
        except:
            logger.exception(f"Arguments: {tool_call_request['arguments']}")
            return {"messages": [SystemMessage(f"Failed to execute tool: {name}")]}

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
            """output the JSON object: {{"name": "unknown", "arguments": {{}} }}."""
        )

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
            rendered_tools = render_text_description(self.tools)
            chain = prompt | self.llm_structured_output | JsonOutputParser()
            tool_call_request = chain.invoke(
                {"rendered_tools": rendered_tools},
                response_format=self.ToolCall,
            )

            goto = "think" if tool_call_request["name"] == "unknown" else "tools"
            update = (
                None
                if tool_call_request["name"] == "unknown"
                else {"tool_call_request": tool_call_request}
            )

        return Command(
            update=update,
            goto=goto,
        )


class QueryState(BaseModel):
    content: str = Field(
        description=(
            "The target document from which to retrieve information necessary for "
            "answering the user's question."
        )
    )
    query: str = Field(description="The question provided by the user.")


class InputState(BaseModel):
    docs: dict[str, str] = Field(description="Input documents provided by the user")
    question: str = Field(description="Question related to the document")


class SplitDocs(BaseModel):
    """
    中間処理で利用する
    """

    doc_name: str = Field(description="Document name")
    question: str = Field(description="Question related to the document")
    contents: list[str] = Field(description="Segments of the document after splitting")


class OutputState(BaseModel):
    information: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="Document information used to answer the user question"
    )
    answer: str


class OverallState(InputState, OutputState):
    complete_docs: list[str] = Field(description="Document list to complete searching")


PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    (
        "human",
        """# System Prompt

あなたはGoogle DeepMindのgemma3モデル(4b～12b量子化)です。
ユーザから、以下の形式でMarkdown入力が与えられます:

---

## ユーザ入力 (例)

**User Query**:
{user_query}

**Document Chunk**:
{document_chunk}


---

## あなたの役割

1. **User Query** と **Document Chunk** をもとに、ユーザの質問に答えるために必要な情報を抽出してください。
2. 抽出した情報は **Markdown形式の文字列** で出力してください。
   - たとえば、見出しを `###` で始めたり、箇条書きを `- ` で書くなど、Markdownを自由に用いてください。
3. 該当する情報が全く見当たらない場合は `null` とだけ出力してください。

---

## 出力形式の具体例

- **有益な情報がある場合の例:**

```markdown
### Relevant Info
- この文書の主要なポイントは...
- (Markdownの書式でまとめる)
```

- **有益な情報がない場合の例

```markdown
null
```

以上の指示に従って、必ずMarkdown形式のテキストあるいは null のいずれかを出力してください。
それ以外の余計な文章や解説は不要です。
""",
    )
)


class QueryDocGraph:
    """
    Documentを読み込み、質問に回答する情報を探すグラフ
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        # text splitようパラメータ
        chunk_size = 500
        chunk_overlap = 50
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        self.graph = self.build_graph()

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

        if len(state.docs) == len(state.complete_docs):
            goto = "generate_answer"
            update = None
        else:
            target_doc_name = [doc for doc in state.docs if doc not in state.complete_docs][0]
            target_doc_content = state.docs[target_doc_name]
            contents = self.text_splitter.split_text(target_doc_content)

            goto = "_root_doc_analysis"
            update = {
                "doc_name": target_doc_name,
                "question": state.question,
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
        information = []
        for content in state.contents:
            chain = PROMPT_TEMPLATE | self.llm
            response = chain.invoke({"user_query": state.question, "document_chunk": content})
            if response.text() == "null":
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
        system_prompt = (
            "以下の背景情報を用いて、ユーザの質問に回答してください\n"
            "## 背景情報\n\n" + "-----\n\n".join(state.information)
        )
        return {
            "answer": self.invoke(
                [
                    ("system", system_prompt),
                    ("human", state.question),
                ]
            )
        }

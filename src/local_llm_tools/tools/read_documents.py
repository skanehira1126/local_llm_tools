from logging import getLogger
import operator
from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.types import Command
from pydantic import BaseModel
from pydantic import Field

from local_llm_tools.prompts.read_documents import TEMPLATE_READ_CHUNK
from local_llm_tools.prompts.read_documents import TEMPLATE_SUMMARIZE_CHUNK
from local_llm_tools.utils.llm import ExtractKeyParser
from local_llm_tools.utils.llm import OllamaTokenCounter


logger = getLogger(__name__)


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
    docs: dict[str, str] = Field(description="Supplied documents")


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
    complete_docs: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Document list to complete searching",
    )


class SearchDocGraph:
    """
    Documentを読み込み、質問に回答する情報を探すグラフ
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        token_counter: OllamaTokenCounter,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            llm (ChatOpenAI): 質問を回答するために利用するLLM
            output_max_tokens(int): チャンクの要約のTokenサイズ
            chunk_size (int): ドキュメントを分割する長さ
            chunk_overlap (int): ドキュメントを分割する時に隣のチャンクと重ねる部分
        """
        self.llm = llm

        def length_function(documents: list[str]) -> int:
            """
            Get number of tokens for input contents.
            """
            return sum(token_counter(doc) for doc in documents)

        # text splitパラメータ
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

        self.graph = self.build_graph()

    def as_tool(self):
        return (self.graph | ExtractKeyParser("summary")).as_tool(
            name="Search Documents",
            description=(
                "ユーザの質問がドキュメントの要約やドキュメント内の記載内容に関する場合に、"
                "ユーザの質問に関連する内容を取得する。"
                "このツールはテキストドキュメントに対してのみ利用する。"
            ),
        )

    def build_graph(self):
        """
        LangGraphを構築する
        """
        graph = StateGraph(OverallState, input=InputState, output=OutputState)

        # node
        graph.add_node("split_docs", self._split_docs)
        graph.add_node("search_in_doc", self._search_in_doc)
        graph.add_node("generate_prompt_for_answer", self._generate_prompt_for_answer)

        # edge
        graph.add_edge(START, "split_docs")
        graph.add_edge("search_in_doc", "split_docs")
        graph.add_edge("generate_prompt_for_answer", END)

        return graph.compile()

    def _split_docs(self, state: OverallState):
        """
        Documentを調査を行う形に分割して、回答探索 or 回答生成に振り分ける
        """
        logger.info("START reading documents.")

        if len(state.docs) == len(state.complete_docs):
            goto = "generate_prompt_for_answer"
            update = None
        else:
            target_doc_name = [doc for doc in state.docs if doc not in state.complete_docs][0]
            target_doc_content = state.docs[target_doc_name]
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
        chain = TEMPLATE_READ_CHUNK | self.llm | StrOutputParser()
        for idx, content in enumerate(state.contents, 1):
            response = chain.invoke(
                {"user_query": state.query, "document_chunk": content},
            ).strip()
            logger.info(response)
            if response.lower() == "null" or not response.startswith("#### Relevance Extraction"):
                pass
            else:
                information.append(f"### チャンク {idx}\n\n" + response)

        return {
            "complete_docs": [state.doc_name],
            "information": information,
        }

    def _generate_prompt_for_answer(self, state: OverallState):
        """
        入力された文書を探索した結果を用いて回答を生成する
        """
        logger.info("Generate answer from infomation")
        logger.info(f"The number of information: {len(state.information)}")

        chain = TEMPLATE_SUMMARIZE_CHUNK | self.llm | StrOutputParser()

        response = chain.invoke(
            {
                "chunks": "----------\n\n".join(state.information),
                "user_query": state.query,
            }
        )
        output = (
            "ユーザが入力したファイルから回答に必要な情報をまとめました.\n"
            "この情報をファイルに記載されている内容として扱ってください。\n"
            "```markdown\n"
            f"{response}"
            "```\n\n"
        )
        return {"summary": output}

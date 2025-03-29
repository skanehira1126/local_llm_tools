from logging import getLogger
import operator
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.types import Command
from pydantic import BaseModel
from pydantic import Field


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


CHUNK_SYSTEM_PROMPT = """\
あなたは、文書全体に対するユーザの回答とその文書全体の一部（チャンク）を入力として受け取るエージェントです。以下のルールに従って出力を行ってください。

1. あなたの目的は、チャンク内から「ユーザの質問に関連する情報」を抜き出し、\
回答作成に役立つ要点をリスト化することです。
2. あなたの作成した要点リストは最終的に他のチャンクの要点のリストとまとめられ、\
ユーザの入力への回答を利用するために利用されます。
3. 抽出した情報がある場合は箇条書き形式で要点をまとめてください。\
4. 質問に関連する情報がチャンク内に全く含まれない場合は、`null`のみを出力してください。
5. 質問移管する情報が含まれる場合は出力の先頭には必ず\
「#### Relevance Extraction」という見出しをつけてください。
6. この抽出以外の目的や情報を付与しないでください。\
与えられたチャンク以外の知識や推測も付け加えないでください。
7. 冗長な説明や解釈、推測は行わず、チャンク内の情報のみ正確に反映してください。

これらの指示に反する出力は行わないでください。
"""


CHUNK_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", CHUNK_SYSTEM_PROMPT),
        (
            "human",
            "## 文書のチャンク\n\n{document_chunk}\n## ユーザの質問\n\n{user_query}",
        ),
    ]
)

DOMAIN_SUMMARY_SYSTEM_PROMPT = """\
あなたは、雑多な情報を整理、整形し扱いやすい情報に修正するエージェントです。

以下のルールに従ってください。

1. あなたの役割は質問に回答するための雑多な背景情報を整理して整理、整形することです。\
2. 背景情報はユーザの質問に回答するために必要な情報を集めたものです。\
具体的にはユーザの入力したファイルを一定のサイズのチャンクに分割し、\
それぞれのチャンクから回答に必要だと思われる情報を抽出した情報を羅列したものです。
3. あなたの整理・整形した結果は、ユーザの入力への回答を利用するために利用されます。
4. 重複した城福や冗長な表現をわかりやすく読める形に整理してください。\
情報が過度に落ちないように注意してください。
5. 仮にあなたの知識を元に補足を加える場合、補足であることをわかりやすく記載してください。

これらの指示に反する出力は行わないでください。

## 質問

{query}

## 背景情報

{information}
"""
DOMAIN_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("human", DOMAIN_SUMMARY_SYSTEM_PROMPT),
    ]
)


class SearchDocGraph:
    """
    Documentを読み込み、質問に回答する情報を探すグラフ
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            llm (ChatOpenAI): 質問を回答するために利用するLLM
            docs (dict[str, str]): 質問回答のソースとなるドキュメント
            chunk_size (int): ドキュメントを分割する長さ
            chunk_overlap (int): ドキュメントを分割する時に隣のチャンクと重ねる部分
        """
        self.llm = llm

        def length_function(documents: list[str]) -> int:
            """
            Get number of tokens for input contents.
            """
            return sum(llm.get_num_tokens(doc) for doc in documents)

        # text splitパラメータ
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

        self.graph = self.build_graph()

    def as_tool(self):
        return self.graph.as_tool(
            name="Search Documents", description="Documentから質問の回答に必要な情報を抽出する"
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
        chain = CHUNK_PROMPT_TEMPLATE | self.llm
        for content in state.contents:
            response = chain.invoke({"user_query": state.query, "document_chunk": content})
            logger.info(
                "Check text is null: {}".format(response.text().replace("\n", "") == "null")
            )
            logger.info(response.text())
            text = response.text().strip()
            if text.lower() == "null" or not text.startswith("#### Relevance Extraction"):
                pass
            else:
                information.append(response.text())

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

        chain = DOMAIN_PROMPT_TEMPLATE | self.llm

        response = chain.invoke(
            {
                "information": "----------\n\n".join(state.information),
                "query": state.query,
            }
        )
        output_format = (
            "ユーザの質問に回答するためにファイルを読み、回答に必要な情報をまとめました.\n\n{}\n\n"
        )
        return {"summary": output_format.format(response.text())}

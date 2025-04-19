import chromadb
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from pydantic import Field
import torch


class InputSchema(BaseModel):
    query: str = Field(
        ...,
        description=(
            "コレクションから取得したい情報を表す自然言語クエリ。"
            "キーワード列または 1〜2 文程度の質問文など形式は自由ですが、"
            "余計な前置きや背景説明は省き、主題となる語句・問いを簡潔に記述してください。"
            "入力された文字列は `input_format` テンプレートにそのまま挿入され、"
            "埋め込み計算の後に最近傍探索に用いられます。"
        ),
    )


def make_rag_tool(collection: chromadb.Collection):
    """
    RAG検索を行うtool関数を返却する
    """

    # とりあえずクラスをインスタンス化
    rag_store = RagSearch(collection)

    @tool(
        name_or_callable="rag_search",
        args_schema=InputSchema,
    )
    def rag_search(query: str):
        """
        RAGで近傍探索を行う
        """

        return rag_store.rag_search(query)

    return rag_search


class RagSearch:
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

        # モデル読み込み
        self.device = "mps" if torch.mps.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=collection.metadata["embedding_model"],
            model_kwargs={"device": self.device},
        )

        # 検索後の前処理
        self.intput_format = collection.metadata["input"]

    def rag_search(
        self,
        query: str,
        n_result: int = 5,
    ):
        """
        RAGで近傍探索を行う
        """
        search_result = self.query(query)

        output = ""
        for chunk_idx, doc in enumerate(search_result["documents"][0], 1):
            output += f"<chunk:{chunk_idx}>  \n{doc}\n  </chunk:{chunk_idx}>\n\n"

        # おそらく検索結果が多いので、token数を食い潰しかねない
        header = "質問に回答するために以下の結果を取得しました。回答に必要な情報だけを利用してください。\n"
        return output

    def query(self, query):
        # queryのフォーマットに変換
        embed_query = self.embedding_model.embed_query(self.intput_format.format(query))

        # 検索
        search_result = self.collection.query(
            query_embeddings=[embed_query],
            n_results=5,
        )
        return search_result

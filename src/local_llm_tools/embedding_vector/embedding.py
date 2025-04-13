from typing import TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import Field
from sentence_transformers import SentenceTransformer
import torch


prompt_context_template = """
<document>
{document_content}
</document>

ここに、全体ドキュメントから抜粋したチャンクがあります:
<chunk>
{chunk_content}
</chunk>

上記チャンクがドキュメント全体のどの部分に属するかを示す、
簡潔なコンテキスト（2～3 文程度）を生成してください。
回答にコンテキスト以外を含めることは禁止です。
"""
TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("human", prompt_context_template),
    ],
)


class Chunk(BaseModel):
    chunk_id: str
    ordinal_index: int
    content: str
    metadata: dict


class ChunkedDocuments(BaseModel):
    doc_id: str
    content: str = Field(..., description="Whole document")
    chunks: list[Chunk]


class EmbedPromptFormat(TypedDict):
    input: str
    output: str


class EmbeddingFactory:
    def __init__(
        self,
        model_name: str,
        embed_prompt_format: EmbedPromptFormat,
        add_llm_context: bool,
        llm_params: dict | None = None,
    ):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        # Embedding モデル
        self.model_name = model_name
        self.embed_model = SentenceTransformer(model_name, device=device)
        self.embed_prompt_format = embed_prompt_format

        # コンテキスト付与用LLM
        self.add_llm_context = add_llm_context
        self.llm_params = llm_params or {}
        if self.add_llm_context:
            self.llm = TEMPLATE | ChatOpenAI(
                **llm_params,
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        else:
            self.llm = None

    def items(self):
        """
        主要パラメータを取得するための関数
        """
        for param in ["model_name", "embed_prompt_format", "add_llm_context", "llm_params"]:
            yield param, getattr(self, param)

    def run(
        self, chunked_documents: list[ChunkedDocuments], external_metadata: dict | None = None
    ):
        """
        ベクトル化する
        """
        chunk_contents = []
        text_list_to_embed = []
        metadata_list = []
        for doc in chunked_documents:
            doc_id = doc["doc_id"]
            document_content = doc["content"]
            document_metadata = {"doc_id": doc_id}
            if external_metadata is not None:
                document_metadata.update(external_metadata)

            # Embeddingのための整理
            doc_chunk_contens, doc_text_list_to_embed, doc_metadata_list = self.make_text_to_embed(
                document_content,
                doc["chunks"],
                document_metadata=document_metadata,
            )

            chunk_contents += doc_chunk_contens
            text_list_to_embed += doc_text_list_to_embed
            metadata_list += doc_metadata_list

        # embedding
        embedding_results = self.embed_model.encode(text_list_to_embed, convert_to_numpy=True)

        return chunk_contents, embedding_results, metadata_list

    def make_text_to_embed(
        self, document_content: str, chunks: list[Chunk], document_metadata: dict
    ):
        """
        Embeddingをするtextを作成する

        Args:
            document_content (str): ドキュメント全体の文字列
            chunks (list[Chunks]): チャンク分けした一覧
            document_metadata (dict): ドキュメントに付与されているMetadata
        """
        chunk_contents = []
        text_to_embed_list = []
        metadata_list = []
        for chunk in chunks:
            if self.add_llm_context:
                # contentを生成
                response = self.llm.invoke(
                    {
                        "document_content": document_content,
                        "chunk_content": chunk["content"],
                    }
                )
                context = response.text()
            else:
                context = ""
            text_to_embed = self.embed_prompt_format["input"].format(
                f"{chunk['content']}\n\n{context}"
            )

            # metadataとして、chunkの情報も追加する
            chunk_metadata = chunk["metadata"].copy()
            chunk_metadata["chunk_id"] = chunk["chunk_id"]
            chunk_metadata["chunk_ordinal_index"] = chunk["ordinal_index"]
            chunk_metadata.update(document_metadata)

            # NOTE: context情報を付与した後のテキストでいいのか要検討
            chunk_contents.append(text_to_embed)
            text_to_embed_list.append(text_to_embed)
            metadata_list.append(chunk_metadata)

        return chunk_contents, text_to_embed_list, metadata_list

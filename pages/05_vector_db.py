import json
from logging import getLogger
import pathlib

import chromadb
import streamlit as st

from local_llm_tools.embedding_vector.embedding import EmbeddingFactory
from local_llm_tools.utils import ollama as ollama_utils
from local_llm_tools.utils.streamlit_components import display_vector_configs


def embedding_conf():
    is_done_init = "embedding_factory" in st.session_state
    with st.expander("設定", expanded=not is_done_init):
        embed_model, embed_prompt_template, enable_contextual_embed, params_contextual_embed = (
            display_vector_configs(
                ollama_utils.enable_models(),
                enable_contextual_embed=(
                    is_done_init and st.session_state.embedding_factory.add_llm_context
                ),
            )
        )

        st.session_state.embedding_factory = EmbeddingFactory(
            model_name=embed_model,
            embed_prompt_format=embed_prompt_template,
            add_llm_context=enable_contextual_embed,
            llm_params=params_contextual_embed,
        )


# Config
def setup_sidebar():
    """
    ChromaDBのためのマネージャー
    """
    with st.form("ChromaDBマネージャー"):
        file_path = st.text_input("読み込むChromaDBのパス")
        if st.form_submit_button("読み込み"):
            st.session_state.preview_client = chromadb.PersistentClient(path=file_path)

    if client := st.session_state.get("preview_client", None):
        collection_name_list = [collection.name for collection in client.list_collections()]

        selected_collection = st.pills("表示するコレクション", options=collection_name_list)
        if selected_collection:
            collection = client.get_collection(name=selected_collection)
            st.markdown(f"- 件数: {collection.count()}")
            st.json(collection.peek())


logger = getLogger(__name__)


# ページ情報
st.set_page_config(page_title="Chatbot implemented by langfamily", layout="wide")


st.title("ドキュメントのデータベース作成")

###############
#
# Side bar
# Embeddingの設定
#
###############
with st.sidebar:
    setup_sidebar()

###############
#
# 初期設定
#
###############

# 対象のドキュメント
files = list(pathlib.Path("sample_docs").glob("*.json"))

target_docs = st.multiselect(
    "Documents",
    options=files,
)

# ドキュメントがない場合、先に進まない
if len(target_docs) == 0:
    st.stop()
else:
    document_chunks = []
    for doc in target_docs:
        with doc.open() as f:
            document_chunks.append(json.load(f))

    st.markdown("Documents List")
    st.markdown("\n".join([f"- {doc.name}" for doc in target_docs]))


# =====================
#
# ドキュメントの情報
#
# =====================
def trunc_content(content: str, max_length: int):
    if len(content) >= max_length + 1:
        return content[:max_length] + "\n ..."
    else:
        return content


# 基本的にcontentが長すぎるので切り取る
st.markdown("### Document preview")

max_length = st.number_input("コンテキストの最大長", min_value=1, value=50)
preview_json = []
for chunk in document_chunks:
    chunk_preview = chunk.copy()

    # 50文字以上ある場合、50文字までで切り取る
    chunk_preview["content"] = trunc_content(chunk_preview["content"], max_length)

    # チャンクの中身も同様の処理
    for i in range(len(chunk_preview["chunks"])):
        chunk_preview["chunks"][i]["content"] = trunc_content(
            chunk_preview["chunks"][i]["content"], max_length
        )

    preview_json.append(chunk_preview)

# 読みこんだDocumentの情報
n_docs = len(preview_json)
n_chunks = sum(len(doc["chunks"]) for doc in document_chunks)

st.markdown(f"- The number of Documents: {n_docs}\n- The number of chunks: {n_chunks}")

st.json(preview_json, expanded=2)


# ====================
#
# ベクトル化
#
# ====================

st.markdown("## ベクトル化")
embedding_conf()

st.markdown("### 設定")
if "embedding_factory" in st.session_state:
    st.markdown(
        "| 設定 | 値 |\n"
        + " | --- | --- |\n"
        + "\n".join(
            [
                f"| `{key}` | `{value}` |"
                for key, value in st.session_state.embedding_factory.items()
            ]
        )
    )
else:
    st.markdown("なし")
    st.stop()

if st.button("ベクトル化実行"):
    with st.spinner("処理中", show_time=True):
        chunk_contens, embedding_results, metadata_list = st.session_state.embedding_factory.run(
            document_chunks
        )

        st.session_state.chromadb_inputs = {
            "documents": chunk_contens,
            "embeddings": embedding_results,
            "metadatas": metadata_list,
            "ids": [metadata["chunk_id"] for metadata in metadata_list],
        }

if "chromadb_inputs" not in st.session_state:
    st.stop()

with st.form("ベクトルDBへ保存", border=True):
    file_path = st.text_input("ChromaDBの保存場所")
    collection_name = st.text_input("ChromaDBのコレクション名")

    if st.form_submit_button("保存"):
        client = chromadb.PersistentClient(path=file_path)

        embedding_factory = st.session_state.embedding_factory

        collection_metadata = {"embedding_model": embedding_factory.model_name}
        collection_metadata.update(embedding_factory.embed_prompt_format)
        if embedding_factory.add_llm_context:
            collection_metadata["context_llm_model"] = embedding_factory.llm_params["model"]

        collection = client.get_or_create_collection(
            collection_name,
            metadata=collection_metadata,
        )

        collection.add(**st.session_state.chromadb_inputs)

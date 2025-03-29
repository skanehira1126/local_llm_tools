from dataclasses import dataclass
from logging import getLogger

import markitdown
import streamlit as st

from local_llm_tools.langfamily_agent.agent import ChatBot
from local_llm_tools.utils import ollama as ollama_utils
from local_llm_tools.utils import setup_langsmith
from local_llm_tools.utils.image import encode_image
from local_llm_tools.utils.streamlit_components import display_llm_initial_configs
from local_llm_tools.utils.streamlit_components import show_images


@dataclass
class SourceFile:
    content: str
    is_enable: bool


def setup_sidebar():
    """
    ファイルを読み込むためのUIを追加
    """
    if "docs" not in st.session_state:
        st.session_state.docs = {}

    uploaded_files = st.file_uploader("ソースファイル", accept_multiple_files=True)

    # 読み込み
    md = markitdown.MarkItDown()
    for uploaded_file in uploaded_files:
        result = md.convert(uploaded_file)
        st.session_state.docs[uploaded_file.name] = SourceFile(
            content=result.text_content, is_enable=False
        )

    # 読み込んだファイルの有効化の制御
    active_source_flles = st.pills(
        "SourceFiles", st.session_state.docs.keys(), selection_mode="multi"
    )
    for file_name, doc in st.session_state.docs.items():
        doc.is_enable = file_name in active_source_flles


logger = getLogger(__name__)


# ============
# langsmithの認証のための処理
# ============
try:
    setup_langsmith.setup_langsminth_credentials("sandbox")
    logger.info("Langsmith connection environment variables have been set up successfully.")
except Exception as _:
    logger.exception("Failed to setup langsmith")


def reset_history():
    del st.session_state["chatbot"]


# ページ情報
st.set_page_config(page_title="Chatbot implemented by langfamily", layout="wide")


# Layout
col1, col2 = st.columns([8, 2])
col1.title("Chat Bot")
col1.markdown("""langgraphを使ってtoolを呼び出せるようにする""")
col2.button(
    "リセット",
    key="clear_all",
    icon=":material/clear_all:",
    on_click=reset_history,
)

config = {"configurable": {"thread_id": "1"}}


###############
#
# Side bar
# ファイル管理
#
###############
with st.sidebar:
    setup_sidebar()

###############
#
# 初期設定
#
###############
if "chatbot" not in st.session_state:
    (
        system_prompt,
        model_name,
        temperature,
        top_p,
        tools,
        is_tool_use_model,
        is_enable_think_tool,
    ) = display_llm_initial_configs(
        model_name_list=ollama_utils.enable_models(),
        add_tools=True,
    )
    client = ChatBot(
        model_name=model_name,
        params={
            "temperature": temperature,
            "top_p": top_p,
        },
        tools=tools,
        is_tool_use_model=is_tool_use_model,
        is_enable_think_tool=is_enable_think_tool,
    )

    history = []

    is_display_system_prompt = True
else:
    history = st.session_state.chatbot.history(config)
    is_display_system_prompt = False

    # toolを表示しておく
    tool_names = [t.name for t in st.session_state.chatbot.agent.tools]
    st.markdown("**Active Tools**")
    if len(tool_names):
        st.pills(
            "Active Tools",
            tool_names,
            default=tool_names,
            selection_mode="multi",
            disabled=True,
            label_visibility="collapsed",
        )


###############
#
# Display chat messages from history
#
###############
for cnt, (msg, model_name, role) in enumerate(history):
    # 表示
    with st.chat_message(role):
        if model_name is not None:
            st.markdown(f"`From {model_name}`")

        col1, col2 = st.columns([8, 1])

        # text部分
        col1.markdown(msg.text())

        # userの入力の場合、削除ボタンを実装
        if role == "user":
            col2.button(
                "",
                key=str(cnt),
                icon=":material/delete:",
                on_click=st.session_state.chatbot.delete_messages,
                kwargs={"message_idx": cnt, "config": config},
            )

        # 画像がある場合に表示
        if isinstance(msg.content, list):
            images = (
                content["image_url"]["url"]
                for content in msg.content
                if content["type"] == "image_url"
            )
            show_images(images, 3)


###############
#
# Handling user input
#
###############
if user_input := st.chat_input("何かお困りですか？", accept_file="multiple"):
    prompt = user_input.text
    files = user_input.files  # streamlitの一般的なファイルアップロード用の型

    images = [
        f"data:{f.type};base64,{encode_image(f.getvalue())}"
        for f in files
        if f.type.startswith("image")
    ]

    if is_display_system_prompt:
        # system_promptが入ってない
        st.session_state.chatbot = client
        st.session_state.chatbot.build()

    # Display user Message
    with st.chat_message("user"):
        st.markdown(prompt)
        if len(images):
            show_images(images, 3)

    # ユーザの入力があった場合に処理が分岐するための制御
    is_update_chat_log = True
else:
    is_update_chat_log = False

###############
#
# ユーザの入力に更新があった場合に更新
#
###############
if is_update_chat_log:
    # ユーザからのソースファイルの情報を反映
    enable_files = {
        name: file.content for name, file in st.session_state.docs.items() if file.is_enable
    }
    # if len(enable_files):
    #     st.session_state.chatbot.register_docs(enable_files)
    # else:
    #     st.session_state.chatbot.register_docs(None)

    with st.chat_message("assistant"):
        # st.markdown(f"`From {chatbot.model_name}`")
        if is_display_system_prompt:
            stream = st.session_state.chatbot.chat_stream(
                prompt, images, enable_files, config, system_prompt
            )
        else:
            stream = st.session_state.chatbot.chat_stream(prompt, images, enable_files, config)
        response = st.write_stream(stream)

    # chatbot.add_assistant_message(content=response, model_name=chatbot.model_name)

    # 再描画
    st.rerun()

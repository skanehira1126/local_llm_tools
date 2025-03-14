from logging import getLogger

import streamlit as st

from local_llm_tools.langfamily_agent.agent import ChatBot
from local_llm_tools.utils import ollama as ollama_utils
from local_llm_tools.utils import setup_langsmith
from local_llm_tools.utils.streamlit_components import display_llm_initial_configs


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

if "chatbot" not in st.session_state:
    system_prompt, model_name, temperature, top_p, tools, is_tool_use_model = (
        display_llm_initial_configs(
            model_name_list=ollama_utils.enable_models(),
            add_tools=True,
        )
    )
    client = ChatBot(
        model_name=model_name,
        params={
            "temperature": temperature,
            "top_p": top_p,
        },
        tools=tools,
        is_tool_use_model=is_tool_use_model,
    )

    history = []

    is_display_system_prompt = True
else:
    history = st.session_state.chatbot.history(config)
    is_display_system_prompt = False


# Display chat messages from history
for cnt, (msg, model_name, role) in enumerate(history):
    if msg.content == "":
        continue
    with st.chat_message(role):
        if model_name is not None:
            st.markdown(f"`From {model_name}`")

        col1, col2 = st.columns([8, 1])
        col1.markdown(msg.content)
        if role == "user":
            col2.button(
                "",
                key=str(cnt),
                icon=":material/delete:",
                on_click=st.session_state.chatbot.delete_messages,
                kwargs={"message_idx": cnt, "config": config},
            )


# Handling user input
if prompt := st.chat_input("何かお困りですか？"):
    if is_display_system_prompt:
        # system_promptが入ってない
        st.session_state.chatbot = client
        st.session_state.chatbot.build()

    # Display user Message
    with st.chat_message("user"):
        st.markdown(prompt)

    # ユーザの入力があった場合に処理が分岐するための制御
    is_update_chat_log = True
else:
    is_update_chat_log = False

# ユーザの入力があった場合に、返答を生成する
if is_update_chat_log:
    with st.chat_message("assistant"):
        # st.markdown(f"`From {chatbot.model_name}`")
        if is_display_system_prompt:
            stream = st.session_state.chatbot.chat_stream(prompt, config, system_prompt)
        else:
            stream = st.session_state.chatbot.chat_stream(prompt, config)
        response = st.write_stream(stream)

    # chatbot.add_assistant_message(content=response, model_name=chatbot.model_name)

    # 再描画
    st.rerun()

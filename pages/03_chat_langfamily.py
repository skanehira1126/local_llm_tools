import streamlit as st

from local_llm_tools.langfamily_chatbot.agent import ChatBot
from local_llm_tools.utils import ollama as ollama_utils
from local_llm_tools.utils.streamlit_components import display_llm_initial_configs


def reset_history():
    del st.session_state["chatbot"]


# ページ情報
st.set_page_config(page_title="Chatbot implemented by langfamily", layout="wide")


# Layout
col1, col2 = st.columns([8, 2])
col1.title("Chat Bot")
col1.markdown("""langfamilyを活かしたchatbot""")
col2.button(
    "リセット",
    key="clear_all",
    icon=":material/clear_all:",
    on_click=reset_history,
)

config = {"configurable": {"thread_id": "1"}}

if "chatbot" not in st.session_state:
    system_prompt, model_name, temperature, top_p = display_llm_initial_configs(
        model_name_list=ollama_utils.enable_models(),
    )
    client = ChatBot(
        model_name=model_name,
        params={
            "temperature": temperature,
            "top_p": top_p,
        },
    )

    history = []

    is_display_system_prompt = True
else:
    history = st.session_state.chatbot.history(config)
    is_display_system_prompt = False


# Display chat messages from history
for cnt, (msg, model_name, role) in enumerate(history):
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
                # on_click=chatbot.delete_messages,
                kwargs={"message_idx": cnt},
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
    # chatbot.add_user_message(prompt)

    # ユーザの入力があった場合に処理が分岐するための制御
    is_update_chat_log = True
else:
    is_update_chat_log = False

# ユーザの入力があった場合に、返答を生成する
if is_update_chat_log:
    with st.chat_message("assistant"):
        # st.markdown(f"`From {chatbot.model_name}`")
        stream = st.session_state.chatbot.chat_stream(prompt, config)
        response = st.write_stream(stream)

    # chatbot.add_assistant_message(content=response, model_name=chatbot.model_name)

    # 再描画
    st.rerun()

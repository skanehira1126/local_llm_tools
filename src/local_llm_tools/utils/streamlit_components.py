import streamlit as st

from local_llm_tools.tools import MATH_TOOLS
from local_llm_tools.tools import SEARCH_TOOLS
from local_llm_tools.utils import helps
from local_llm_tools.utils.llm_configs import SYSTEM_PROMPT
from local_llm_tools.utils.llm_configs import TEMPERATURE
from local_llm_tools.utils.llm_configs import TOP_P


def display_llm_initial_configs(
    model_name_list: list[str] | None = None,
    default_system_prompt: str = SYSTEM_PROMPT,
    add_tools: bool = False,
):
    with st.container(border=True):
        system_prompt = st.text_area("システムプロンプト", value=default_system_prompt)

        # モデル選択
        if model_name_list:
            model_name = st.selectbox(
                "model_name",
                options=model_name_list,
            )
        else:
            model_name = st.text_input("model_name")

        # ツール
        if add_tools:
            options = ["math", "search"]
            selection = st.pills("利用ツール", options, selection_mode="multi")
            tools = []
            if "math" in selection:
                tools += MATH_TOOLS
            if "search" in selection:
                tools += SEARCH_TOOLS

        # パラメータ
        temperature = st.slider(
            "temperature",
            help=helps.TEMPERATURE,
            **TEMPERATURE,
        )
        top_p = st.slider(
            "top_p",
            help=helps.TOP_P,
            **TOP_P,
        )

    if add_tools:
        return system_prompt, model_name, temperature, top_p, tools
    else:
        return system_prompt, model_name, temperature, top_p

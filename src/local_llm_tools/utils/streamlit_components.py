import streamlit as st

from local_llm_tools.utils import helps
from local_llm_tools.utils.llm_configs import SYSTEM_PROMPT, TEMPERATURE, TOP_P


def display_llm_initial_configs(
    model_name_list: list[str] | None = None,
    default_system_prompt: str = SYSTEM_PROMPT,
):
    with st.container(border=True):
        system_prompt = st.text_area("システムプロンプト", value=default_system_prompt)

        if model_name_list:
            model_name = st.selectbox(
                "model_name",
                options=model_name_list,
            )
        else:
            model_name = st.text_input("model_name")

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

    return system_prompt, model_name, temperature, top_p

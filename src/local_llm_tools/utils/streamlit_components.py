from logging import getLogger

import streamlit as st

from local_llm_tools.tools import MATH_TOOLS
from local_llm_tools.tools import SEARCH_TOOLS
from local_llm_tools.utils import helps
from local_llm_tools.utils.llm_configs import SYSTEM_PROMPT
from local_llm_tools.utils.llm_configs import TEMPERATURE
from local_llm_tools.utils.llm_configs import TOP_P


logger = getLogger(__name__)


def display_llm_initial_configs(
    model_name_list: list[str] | None = None,
    default_system_prompt: str = SYSTEM_PROMPT,
    add_tools: bool = False,
):
    """
    LLMのための初期化componentを作成
    """
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
            is_tool_use_model = st.checkbox("Tool対応モデル")
            options = ["math", "search", "think"]
            selection = st.pills("利用ツール", options, selection_mode="multi")
            tools = []
            if "math" in selection:
                tools += MATH_TOOLS
            if "search" in selection:
                tools += SEARCH_TOOLS
            is_enable_think_tool = "think" in selection

        # パラメータ
        temperature = st.slider(
            "temperature",
            help=helps.TEMPERATURE,
            **TEMPERATURE,
        )
        if st.toggle("TOP_Pの調整"):
            top_p = st.slider(
                "top_p",
                help=helps.TOP_P,
                **TOP_P,
            )
        else:
            top_p = None

    if add_tools:
        logger.info(f"Selected tools : {tools},")
        return (
            system_prompt,
            model_name,
            temperature,
            top_p,
            tools,
            is_tool_use_model,
            is_enable_think_tool,
        )
    else:
        return system_prompt, model_name, temperature, top_p


def show_images(images: str, n_cols: int):
    """
    columnsを利用して、画像を並べて表示する
    """
    columns = st.columns([1] * n_cols)
    for idx, img in enumerate(images):
        columns[idx % n_cols].image(img)

        # 3つグラフを描画したらcolumnsを更新
        if idx % n_cols == n_cols - 1:
            columns = st.columns([1] * n_cols)

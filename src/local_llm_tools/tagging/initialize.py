from dataclasses import dataclass

import pandas as pd
import streamlit as st

from local_llm_tools.utils.ollama import enable_models


@dataclass
class ConfigTagging:
    current_model: str
    temperature: float
    var_tagged_df: pd.DataFrame | None = None


def initialize():
    if "tagging" not in st.session_state:
        st.session_state.tagging = ConfigTagging(
            current_model=enable_models()[0],
            temperature=0.5,
        )


def sidebar():
    config: ConfigTagging = st.session_state.tagging

    with st.sidebar:
        model_list = enable_models()
        config.current_model = st.selectbox(
            "利用モデル",
            model_list,
            index=model_list.index(config.current_model),
        )

        config.temperature = st.slider(
            "Temperature",
            value=config.temperature,
        )

import pandas as pd
import streamlit as st

from local_llm_tools import tagging
from local_llm_tools.tagging import prompts
from local_llm_tools.tagging.files import read_uploaded_file
from local_llm_tools.tagging.ollama import add_tag_to_all_texts
from local_llm_tools.utils.files import download_dataframe_as_csv

tagging.initialize()
tagging.sidebar()


st.title("タグ付与ツール")


config = st.session_state.tagging

# data Preview
col1, col2 = st.columns([1, 1])
with col1:
    target_texts = read_uploaded_file()

    text_column = st.selectbox("タグ付与対象カラム", target_texts.columns, index=0)
    data_size = len(target_texts)
    st.metric("レコード数", data_size)

with col2:
    st.text("データプレビュー")
    st.dataframe(target_texts[[text_column]], hide_index=True)


st.markdown(f"利用するモデル: `{config.current_model}`")
tags_str = st.text_input(
    "付与するタグ", help="カンマ区切りで付与するタグの一覧をカンマ区切りで入力"
)
tag_list = [f"'{tag.strip()}'" for tag in tags_str.split(",")]
if len(tag_list) <= 1:
    st.error("タグを2種類以上入力してください")
    st.stop()

tag_descriptions = {}
with st.expander("タグ説明"):
    for tag_name in tag_list:
        tag_descriptions[tag_name] = st.text_input(tag_name)

dc_tagging_prompt = prompts.TaggingPrompt(
    tag_list=tag_list,
    tag_descriptions=tag_descriptions,
)

is_generate = st.button("タグ付与")

if is_generate:

    with st.status("Tagging to texts...", expanded=True) as status:
        pbar = st.progress(0)
        n_texts = len(target_texts)
        result = []

        for idx, tag_dict in enumerate(
            add_tag_to_all_texts(
                model=config.current_model,
                df=target_texts,
                text_column=text_column,
                dc_prompt=dc_tagging_prompt,
                params={
                    "temperature": config.temperature,
                },
            )
        ):

            status.update(label="Tagging : {}.".format(tag_dict[text_column]), expanded=True)
            result.append(tag_dict)
            pbar.progress((int(idx) + 1) / n_texts)
        status.update(label="Comleted !!", expanded=False, state="complete")

    config.var_tagged_df = pd.DataFrame(result)

if config.var_tagged_df is not None:
    st.markdown("### 付与結果")
    st.dataframe(config.var_tagged_df, hide_index=True)

    csv = download_dataframe_as_csv(config.var_tagged_df)

    st.download_button(
        "タグ付与結果のダウンロード",
        data=csv,
        file_name="tagged_result.csv",
        mime="text/csv",
    )

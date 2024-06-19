import streamlit as st


def if_file_upload(label: str):
    """
    ファイルアップロードインターフェイスを追加する
    """

    uploaded_file = st.file_uploader(label)
    if uploaded_file is None:
        st.stop()
    else:
        return uploaded_file.getvalue()


@st.cache_data
def download_dataframe_as_csv(df):
    return df.to_csv(index=False).encode("utf-8")

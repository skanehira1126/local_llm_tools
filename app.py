import streamlit as st

text = """
# Local LLM Tools
Ollamaを利用して、LLMを利用したツールを開発する.
"""

# streamlit global configuration
st.set_page_config(
    page_title="Local LLM Toolbox",
    layout="wide",
)

st.markdown(text)

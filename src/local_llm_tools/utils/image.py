import base64

from streamlit.runtime.uploaded_file_manager import UploadedFile


def encode_image(uploaded_file: UploadedFile):
    """
    画像を読み込んでbase64 encodingを行う
    """
    return base64.b64encode(uploaded_file).decode("utf-8")

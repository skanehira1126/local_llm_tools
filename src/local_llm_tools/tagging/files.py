import io

import pandas as pd

from local_llm_tools.utils import files


def read_uploaded_file():

    uploaded_file = files.if_file_upload("タグ付け対象ファイル")
    io_csv_text = io.StringIO(uploaded_file.decode())

    target_texts = pd.read_csv(io_csv_text)

    return target_texts

import json

import ollama

BASE_URL = "http://localhost:11434/v1/"


def enable_models() -> list:
    """
    ollama apiを利用してダウンロードされているモデル一覧を取得する
    """
    model_list = ollama.list()
    return list(map(lambda x: x.model, model_list["models"]))


def generate_json(model: str, prompt: str, options: dict | None = None):
    """ """
    response = ollama.generate(
        model=model,
        prompt=prompt,
        format="json",
        options=options,
    )

    return json.loads(response["response"])

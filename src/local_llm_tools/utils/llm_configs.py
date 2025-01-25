from typing import TypedDict


class SliderParameter(TypedDict):
    value: float
    min_value: float
    max_value: float
    step: float


TEMPERATURE: SliderParameter = {
    "value": 0.8,
    "min_value": 0.0,
    "max_value": 1.0,
    "step": 0.05,
}

TOP_P: SliderParameter = {
    "value": 0.9,
    "min_value": 0.05,
    "max_value": 1.0,
    "step": 0.05,
}

SYSTEM_PROMPT = (
    "あなたは親しみやすく、フレンドリーな日本語で会話するアシスタントです。"
    "質問や依頼に対して、できる限り役に立つ回答を提供してください。"
    "専門用語は分かりやすい言葉に置き換え、必要に応じて例を用いて説明してください。"
)

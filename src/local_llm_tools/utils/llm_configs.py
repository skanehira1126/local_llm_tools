from typing import TypedDict


class SliderParameter(TypedDict):
    value: float
    min_value: float
    max_value: float
    step: float


TEMPERATURE: SliderParameter = {
    "value": 0.1,
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
    "あなたは優秀なAIアシスタントです。\n"
    "ユーザの回答に対し、特に指定がない限り日本語で回答してください。"
)

from dataclasses import asdict, dataclass
from typing import Literal

ROLE = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: ROLE
    content: str
    model_name: str | None

    def as_dict(self, include_model_name: bool = True):
        """
        Messageを辞書に変換する
        """

        dict_message = asdict(self)

        if not include_model_name:
            del dict_message["model_name"]
        return dict_message

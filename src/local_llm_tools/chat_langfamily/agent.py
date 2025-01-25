from functools import partialmethod
from typing import Literal

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from local_llm_tools.chat_langfamily.utils import get_role_of_message


class ChatBot:
    def __init__(self, model_name: str, params: dict | None = None):
        self.model_name = model_name
        self.messages: list[AIMessage | HumanMessage | SystemMessage] = []
        self.messages_model: list[str | None] = []

        self.params: dict = {}
        if params is not None:
            self.params.update(params)

    def add_message(
        self,
        content: str,
        *,
        model_name: str | None,
        msg_type: Literal[
            type[AIMessage],
            type[HumanMessage],
            type[SystemMessage],
        ],
    ):
        """
        会話履歴としてメッセージを追加する
        """
        self.messages.append(msg_type(content=content))
        self.messages_model.append(model_name)

    add_system_message = partialmethod(
        add_message, model_name=None, msg_type=SystemMessage
    )
    add_user_message = partialmethod(
        add_message, model_name=None, msg_type=HumanMessage
    )
    add_assistant_message = partialmethod(add_message, msg_type=AIMessage)

    def set_params(self, **kwargs):
        """パラメータの更新"""
        if not kwargs:
            raise ValueError("One or more parameters are required.")

        self.params.update(kwargs)

    def delete_messages(self, message_idx: int):
        """
        指定したindexまでのMessageを削除する
        """
        self.messages = self.messages[:message_idx]
        self.messages_model = self.messages_model[:message_idx]

    def reset_message(self):
        """
        Messageの初期化
        """
        self.messages = []
        self.messages_model = []

    @property
    def history(self):
        for msg, model_name in zip(self.messages, self.messages_model):
            yield msg, model_name, get_role_of_message(msg)

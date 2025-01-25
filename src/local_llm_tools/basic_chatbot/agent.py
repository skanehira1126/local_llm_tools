from functools import partialmethod

from local_llm_tools.basic_chatbot.utils import ROLE, Message


class ChatBot:
    def __init__(self, model_name: str, params: dict | None = None):
        self.model_name = model_name
        self.messages: list[Message] = []

        self.params: dict = {}
        if params is not None:
            self.params.update(params)

    def add_message(
        self,
        content: str,
        *,
        model_name: str | None,
        role: ROLE,
    ):
        """
        会話履歴としてメッセージを追加する
        """
        self.messages.append(Message(content=content, role=role, model_name=model_name))

    add_system_message = partialmethod(add_message, model_name=None, role="system")
    add_user_message = partialmethod(add_message, model_name=None, role="user")
    add_assistant_message = partialmethod(add_message, role="assistant")

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

    def reset_message(self):
        """
        Messageの初期化
        """
        self.messages = []

    @property
    def history(self):
        for msg in self.messages:
            yield msg, msg.model_name, msg.role

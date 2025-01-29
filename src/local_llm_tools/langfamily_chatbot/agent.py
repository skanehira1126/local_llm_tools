from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_core.messages import RemoveMessage
from langchain_ollama import ChatOllama

from local_llm_tools.langfamily_chatbot.build_graph import build_graph
from local_llm_tools.langfamily_chatbot.utils import get_role_of_message


class ChatBot:
    def __init__(self, model_name: str, params: dict | None = None):
        self.model_name = model_name
        self.messages: list[AIMessage | HumanMessage | SystemMessage] = []
        self.messages_model: list[str | None] = []

        self.params: dict = {}
        if params is not None:
            self.params.update(params)

        self._agent = None

    @property
    def agent(self):
        if not self.is_build():
            raise ValueError("graph is not built.")
        return self._agent

    def is_build(self):
        return self._agent is not None

    def set_params(self, **kwargs):
        """パラメータの更新"""
        if not kwargs:
            raise ValueError("One or more parameters are required.")

        self.params.update(kwargs)

    def build(self):
        llm = ChatOllama(model=self.model_name, **self.params, stream=True)
        self._agent = build_graph(llm)

    def chat_stream(self, user_input: str, config: dict, system_promt: list[str] | None = None):
        if system_promt is None:
            messages = []
        else:
            messages = [{"role": "system", "content": system_promt}]
        messages.append({"role": "user", "content": user_input})

        for event in self.agent.stream(
            {"messages": messages},
            config,
            stream_mode="messages",
        ):
            # (AIMessageChunk, dict)
            yield event[0].content

    def delete_messages(self, message_idx: int, config: dict):
        """
        指定したindexまでのMessageを削除する
        """

        delete_messages = self.agent.get_state(config).values["messages"][message_idx:]
        _ = self.agent.update_state(
            config, {"messages": [RemoveMessage(id=msg.id) for msg in delete_messages]}
        )

    def reset_message(self):
        """
        Messageの初期化
        """
        self.build()

    def history(self, config):
        for msg in self._agent.get_state(config)[0]["messages"]:
            yield msg, msg.response_metadata.get("model", None), get_role_of_message(msg)

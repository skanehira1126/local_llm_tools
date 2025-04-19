from logging import getLogger

from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_core.messages import RemoveMessage
from langchain_core.tools.structured import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from local_llm_tools.langfamily_agent.build_graph import GemmaGraph
from local_llm_tools.langfamily_agent.build_graph import build_graph
from local_llm_tools.langfamily_agent.utils import get_role_of_message


logger = getLogger(__name__)


class ChatBot:
    def __init__(
        self,
        model_name: str,
        tools: list[StructuredTool],
        is_tool_use_model: bool,
        params: dict | None = None,
    ):
        self.model_name = model_name
        self.messages: list[AIMessage | HumanMessage | SystemMessage] = []
        self.messages_model: list[str | None] = []

        self.params: dict = {}
        if params is not None:
            self.params.update(params)

        # OllamaはTool対応していないモデルがある
        self.tools = tools
        self.is_tool_use_model = is_tool_use_model

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

    def register_docs(self, docs: dict[str, str]):
        self._agent.register_docs(docs)

    def build(self):
        llm = ChatOpenAI(
            model=self.model_name,
            **self.params,
            stream=True,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        if self.is_tool_use_model:
            llm = llm.bind_tools(self.tools)
            self._agent = build_graph(llm, ToolNode(self.tools))
        else:
            llm_common_params = {
                "model": self.model_name,
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
            }

            gemma_graph = GemmaGraph(
                llm_chat=llm,
                tools=self.tools,
                llm_common_params=llm_common_params,
            )
            self._agent = gemma_graph

    def chat_stream(
        self,
        user_input: str,
        images: list[str],
        docs: dict[str, str] | None,
        config: dict,
        system_promt: list[str] | None = None,
    ):
        # system prompt
        if system_promt is None:
            messages = []
        else:
            messages = [{"role": "system", "content": system_promt}]

        # ユーザの入力
        content = [{"type": "text", "text": user_input}]
        for image_url in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        messages.append(HumanMessage(content=content))

        for event in self.agent.graph.stream(
            {"messages": messages, "docs": docs},
            config,
            stream_mode="messages",
        ):
            # (AIMessageChunk, dict)
            # print(event)
            yield event[0].content

    def delete_messages(self, message_idx: int, config: dict):
        """
        指定したindexまでのMessageを削除する
        """

        delete_messages = self.agent.graph.get_state(config).values["messages"][message_idx:]
        _ = self.agent.graph.update_state(
            config, {"messages": [RemoveMessage(id=msg.id) for msg in delete_messages]}
        )

    def reset_message(self):
        """
        Messageの初期化
        """
        self.build()

    def history(self, config):
        for msg in self.agent.graph.get_state(config)[0]["messages"]:
            yield msg, msg.response_metadata.get("model", None), get_role_of_message(msg)

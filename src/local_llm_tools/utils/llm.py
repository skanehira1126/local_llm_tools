from inspect import signature

from langchain.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class ExtractKeyParser:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, output) -> str:
        if isinstance(output, BaseModel):
            return getattr(output, self.key)
        else:
            return output[self.key]

    def parse(self, output) -> str:
        return self.__call__(output)

    async def aparse(self, output) -> str:
        return self.__call__(output)


def render_text_description(tools: list[BaseTool]) -> str:
    """Render the tool name and description in plain text.

    Args:
        tools: The tools to render.

    Returns:
        The rendered text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search
        calculator: This tool is used for math
    """
    # Runnableから作ったtoolは対応していなさそう
    overwrite_sig = {
        #     "think": "(thought: str) -> str",
        "Search Documents": "(query: str) -> str",
    }
    descriptions = []
    for tool in tools:
        if hasattr(tool, "func") and tool.func:
            sig = overwrite_sig.get(tool.name, signature(tool.func))
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"

        descriptions.append(description)
    return "\n".join(descriptions)


class OllamaTokenCounter:
    def __init__(self, model_name: str):
        self.llm = ChatOllama(model=model_name)

    def __call__(self, text: str):
        return self.llm.get_num_tokens(text)


def build_chat_history(state, keep_last=5):
    msgs = state["messages"]
    start = max(len(msgs) - keep_last, 0)
    prefix = "...<snip earlier turns>...\n" if start > 0 else ""

    body = "\n".join(
        f"{start + idx}. {m.type.capitalize()}: {m.text()}" for idx, m in enumerate(msgs[start:])
    )
    return prefix + body

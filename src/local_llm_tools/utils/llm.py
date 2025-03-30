from inspect import signature

from langchain.tools import BaseTool


class ExtractKeyParser:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, text) -> str:
        return text[self.key]

    def parse(self, text) -> str:
        return self.__call__(text)

    async def aparse(self, text) -> str:
        return self.__call__(text)


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
    descriptions = []
    for tool in tools:
        if hasattr(tool, "func") and tool.func:
            sig = overwrite_sig.get(tool.name, signature(tool.func))
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"

        descriptions.append(description)
    return "\n".join(descriptions)


overwrite_sig = {
    "think": "(thought: 'str') -> 'str'",
    "Search Documents": "(query: 'str', docs: dict['str', 'str']) -> str",
}

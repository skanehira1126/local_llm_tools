__all__ = [
    "MATH_TOOLS",
    "MATH_TOOLS_DS",
    "SEARCH_TOOLS",
    "SEARCH_TOOLS_DS",
]
from langchain.tools import tool

from local_llm_tools.tools.math import add
from local_llm_tools.tools.math import divide
from local_llm_tools.tools.math import minus
from local_llm_tools.tools.math import multiply
from local_llm_tools.tools.search import get_weather
from local_llm_tools.tools.search import search_on_web


__MATH_TOOLS_FNS = [add, minus, multiply, divide]
__SEARCH_TOOLS_FNS = [get_weather, search_on_web]

MATH_TOOLS = [tool(fn) for fn in __MATH_TOOLS_FNS]
MATH_TOOLS_DS = [tool(parse_docstring=True)(fn) for fn in __MATH_TOOLS_FNS]
SEARCH_TOOLS = [tool(fn) for fn in __SEARCH_TOOLS_FNS]
SEARCH_TOOLS_DS = [tool(parse_docstring=True)(fn) for fn in __SEARCH_TOOLS_FNS]

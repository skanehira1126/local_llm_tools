__all__ = [
    "MATH_TOOLS",
    "SEARCH_TOOLS",
    "THINK_TOOLS",
    "PLAN_TOOLS",
    "make_rag_tool",
]

from local_llm_tools.tools.math import add
from local_llm_tools.tools.math import divide
from local_llm_tools.tools.math import minus
from local_llm_tools.tools.math import multiply
from local_llm_tools.tools.plan import plan
from local_llm_tools.tools.rag import make_rag_tool
from local_llm_tools.tools.search import get_weather
from local_llm_tools.tools.search import search_on_web
from local_llm_tools.tools.think import think


MATH_TOOLS = [add, minus, multiply, divide]
SEARCH_TOOLS = [get_weather, search_on_web]

THINK_TOOLS = [think]
PLAN_TOOLS = [plan]

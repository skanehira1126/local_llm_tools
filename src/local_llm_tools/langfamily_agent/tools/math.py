from logging import getLogger

from langchain.tools import tool


logger = getLogger(__name__)


@tool
def add(a: int | float, b: int | float) -> int | float:
    """add two input values"""
    logger.debug("Called add tool")
    return a + b

from langchain.tools import tool


@tool
def add(a: int | float, b: int | float) -> int | float:
    """add two input values"""
    return a + b

from logging import getLogger

from langchain.tools import tool
from pydantic import BaseModel
from pydantic import Field


logger = getLogger(__name__)


class NumbersInputSchema(BaseModel):
    """Add two numbers."""

    a: int | float = Field(..., description="First number")
    b: int | float = Field(..., description="Second number")


@tool(
    "add",
    args_schema=NumbersInputSchema,
)
def add(a: int | float, b: int | float) -> int | float:
    """
    2つの数の足し算を行う関数

    Args:
        a (int | float): 足し算を行う1つ目の値
        b (int | float): 足し算を行う2つ目の値

    Returns:
        int | float: 足し算の結果
    """
    return a + b


@tool(
    "minus",
    args_schema=NumbersInputSchema,
)
def minus(a: int | float, b: int | float) -> int | float:
    """
    2つの数の引き算を行う関数

    Args:
        a (int | float): 引き算を行う1つ目の値
        b (int | float): 引き算を行う2つ目の値

    Returns:
        int | float: 引き算の結果
    """
    return a - b


@tool(
    "multiply",
    args_schema=NumbersInputSchema,
)
def multiply(a: int | float, b: int | float) -> int | float:
    """
    2つの数の掛け算を行う関数

    Args:
        a (int | float): 掛け算を行う1つ目の値
        b (int | float): 掛け算を行う2つ目の値

    Returns:
        int | float: 掛け算の結果
    """
    return a * b


@tool(
    "divide",
    args_schema=NumbersInputSchema,
)
def divide(a: int | float, b: int | float) -> int | float:
    """
    2つの数の割り算を行う関数

    Args:
        a (int | float): 割り算で割られる側の値
        b (int | float): 割り算を割る側の値

    Returns:
        int | float: 割り算の結果
    """
    return a / b

from logging import getLogger


logger = getLogger(__name__)


def add(a: int | float, b: int | float) -> int | float:
    """
    足し算を行う関数

    Args:
        a (int | float): 足し算を行う1つ目の値
        b (int | float): 足し算を行う2つ目の値

    Returns:
        int | float: 足し算の結果
    """
    return a + b


def minus(a: int | float, b: int | float) -> int | float:
    """
    引き算を行う関数

    Args:
        a (int | float): 引き算を行う1つ目の値
        b (int | float): 引き算を行う2つ目の値

    Returns:
        int | float: 引き算の結果
    """
    return a - b


def multiply(a: int | float, b: int | float) -> int | float:
    """
    掛け算を行う関数

    Args:
        a (int | float): 掛け算を行う1つ目の値
        b (int | float): 掛け算を行う2つ目の値

    Returns:
        int | float: 掛け算の結果
    """
    return a * b


def divide(a: int | float, b: int | float) -> int | float:
    """
    割り算を行う関数

    Args:
        a (int | float): 割り算で割られる側の値
        b (int | float): 割り算を割る側の値

    Returns:
        int | float: 割り算の結果
    """
    return a / b

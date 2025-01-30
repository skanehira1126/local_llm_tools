from typing import Literal

from langchain_core.tools import tool


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """
    天気を調べるために利用する

    Parameters
    ----------
    city: {"nyc", "sf"}
        天気を調べる対象の町の名前
    """
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")

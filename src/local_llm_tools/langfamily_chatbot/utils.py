from typing import Literal

from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage


ROLE = Literal["system", "user", "assistant"]


def get_role_of_message(message: Literal[AIMessage, HumanMessage, SystemMessage]):
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "assistant"
    return role

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class Router(TypedDict):
    """Classify user query."""

    logic: str
    type: Literal["general", "retrieve", "more-info"]


class State(MessagesState):
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    documents: list[Document]

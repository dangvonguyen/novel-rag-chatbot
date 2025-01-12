from typing import TypedDict

from langchain_core.documents import Document


class State(TypedDict):
    question: str
    documents: list[Document]
    answer: str

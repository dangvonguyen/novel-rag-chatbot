import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.configs import Configuration


def format_docs(documents: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def get_embedding_dimension(embedding_model: Embeddings) -> int:
    return len(embedding_model.embed_query("."))


def get_literal_values(configuration: Configuration, attr: str) -> str:
    type_str = str(configuration.__annotations__[attr])
    match = re.search(r"Literal\[(.*?)\]", type_str)
    return match.group(1) if match else ""

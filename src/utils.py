import json
import re
from typing import cast

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.configs import Configuration


def _format_doc(document: Document) -> str:
    filtered_data = {k: document.model_dump()[k] for k in {"metadata", "page_content"}}
    return json.dumps(filtered_data)


def format_docs(documents: list[Document]) -> str:
    if not documents:
        return ""
    formatted = ", ".join(_format_doc(doc) for doc in documents)
    return formatted


def get_embedding_dimension(embedding_model: Embeddings) -> int:
    return len(embedding_model.embed_query("."))


def get_literal_values(configuration: Configuration, attr: str) -> str:
    type_str = str(configuration.__annotations__[attr])
    match = re.search(r"Literal\[(.*?)\]", type_str)
    return match.group(1) if match else ""


def load_embedding(fully_specified_name: str) -> Embeddings:
    """Load an embedding model from a fully specified name."""
    provider, model = fully_specified_name.split("/", maxsplit=1)
    embedding = cast(Embeddings, init_embeddings(model, provider=provider))
    return embedding


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name."""
    provider, model = fully_specified_name.split("/", maxsplit=1)
    if provider == "huggingface":
        return _load_huggingface_chat_model(model)
    else:
        return init_chat_model(model, model_provider=provider)


def _load_huggingface_chat_model(model: str) -> BaseChatModel:
    from langchain_huggingface.chat_models import ChatHuggingFace
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import BitsAndBytesConfig  # type: ignore

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    llm = HuggingFacePipeline.from_model_id(
        model_id=model,
        task="text-generation",
        pipeline_kwargs={"return_full_text": False},
        model_kwargs={
            "quantization_config": bnb_config,
            "low_cpu_mem_usage": True,
        },
    )
    return ChatHuggingFace(llm=llm)

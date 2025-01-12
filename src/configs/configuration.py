from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from src.components import prompts


@dataclass(kw_only=True)
class Configuration:

    embedding_model: Annotated[
        str, {"__template_metadata__": {"kind": "embeddings"}}
    ] = field(
        default="huggingface/hiieu/halong_embedding",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    # Retriever

    retriever_provider: Annotated[
        Literal["pinecone"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="pinecone",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'pinecone'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    # Model

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    # Prompts

    qa_system_prompt: str = field(
        default=prompts.QA_SYSTEM_PROMPT,
        metadata={
            "description": "The question-answering system prompt used for generating responses"
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=Configuration)

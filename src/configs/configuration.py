from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from src.components import prompts


@dataclass(kw_only=True)
class Configuration:

    embedding_model: Annotated[
        str, {"__template_metadata__": {"kind": "embeddings"}}
    ] = field(
        default="huggingface/hiieu/halong_embedding",
        metadata={
            "description": (
                "Name of the embedding model to use. "
                "Must be in the form: provider/model-name."
            )
        },
    )

    # Splitter

    splitter_type: Annotated[
        Literal["semantic", "delimiter"],
        {"__template_metadata__": {"kind": "splitter"}},
    ] = field(
        default="semantic",
        metadata={
            "description": (
                "Strategy used to divide documents into smaller chunks or sections "
                "for processing."
            )
        },
    )

    chunk_size: int = field(
        default=1500,
        metadata={"description": "The number of tokens for each chunk."},
    )

    chunk_overlap: int = field(
        default=300,
        metadata={
            "description": (
                "The maximum number of allowed tokens to overlap between chunks."
            )
        },
    )

    semantic_tokenizer: Annotated[
        str, {"__template_metadata__": {"kind": "tokenizer"}}
    ] = field(
        default="hiieu/halong_embedding",
        metadata={
            "description": (
                "Tokenizer in huggingface used to split text into meaningful "
                "semantic units."
            )
        },
    )

    # Retriever

    retriever_provider: Annotated[
        Literal["pinecone"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="pinecone",
        metadata={
            "description": (
                "The vector store provider to use for retrieval. "
                "Options are 'pinecone'."
            )
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Additional keyword arguments to pass to the search function "
                "of the retriever."
            )
        },
    )

    # Model

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-40-mini",
        metadata={
            "description": (
                "The language model used for processing and refining queries. "
                "Must be in the form: provider/model."
            )
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": (
                "The language model used for generating responses. "
                "Must be in the form: provider/model."
            )
        },
    )

    # Prompts

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": (
                "The system prompt used for classifying user questions to route "
                "them to the correct node."
            )
        },
    )

    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
        metadata={
            "description": (
                "The system prompt used for asking for more information from the user."
            )
        },
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for responding to general questions."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    summary_system_prompt: str = field(
        default=prompts.SUMMARY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for summarizing the conversation."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: type[T], config: RunnableConfig | None = None
    ) -> T:
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=Configuration)

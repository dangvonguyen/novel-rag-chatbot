import os
from collections.abc import Generator
from contextlib import contextmanager

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from src.configs import Configuration
from src.utils import (
    get_embedding_dimension,
    get_literal_values,
    load_embedding,
)


@contextmanager
def make_pinecone_retriever(
    configuration: Configuration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    client = Pinecone()

    index_name = os.getenv("PINECONE_INDEX_NAME", "default")
    if index_name not in client.list_indexes().names():
        client.create_index(
            name=index_name,
            dimension=get_embedding_dimension(embedding_model),
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name, embedding=embedding_model
    )
    yield vectorstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig | None = None,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = Configuration.from_runnable_config(config)
    embedding_model = load_embedding(configuration.embedding_model)
    match configuration.retriever_provider:
        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized vectorstore_provider in configuration. "
                "Expected one of: "
                f"{get_literal_values(configuration, 'retriever_provider')}\n"
                f"Got: {configuration.vectorstore_provider}"
            )

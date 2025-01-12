import os
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from src.components.embedders import make_embedder
from src.configs import Configuration
from src.utils import get_embedding_dimension, get_literal_values


def make_pinecone_retriever(
    configuration: Configuration, embedding_model: Embeddings
) -> VectorStoreRetriever:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    client = Pinecone()

    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in client.list_indexes().names():
        client.create_index(
            name=index_name,
            dimension=get_embedding_dimension(embedding_model),
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name, embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs=configuration.search_kwargs)


def make_retriever(config: Optional[RunnableConfig] = None) -> VectorStoreRetriever:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = Configuration.from_runnable_config(config)
    embedding_model = make_embedder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "pinecone":
            return make_pinecone_retriever(configuration, embedding_model)

        case _:
            raise ValueError(
                "Unrecognized vectorstore_provider in configuration. "
                f"Expected one of: {get_literal_values(configuration, 'retriever_provider')}\n"
                f"Got: {configuration.vectorstore_provider}"
            )

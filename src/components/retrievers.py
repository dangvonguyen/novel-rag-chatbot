import os
from typing import List

from langchain_core.embeddings import Embeddings


def make_pinecone_retriever(embedding_model: Embeddings):
    from langchain_pinecone import PineconeVectorStore

    vectorstore = PineconeVectorStore.from_existing_index(
        os.getenv("PINECONE_INDEX_NAME"), embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

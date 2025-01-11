import os
from typing import List

from src.components.embedders import make_embedder


def make_pinecone_vectorstore():
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    client = Pinecone()

    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in client.list_indexes().names():
        print("Create")
        client.create_index(
            name=index_name,
            dimension=768,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    embedding_model = make_embedder('huggingface/hiieu/halong_embedding')
    index = client.Index(index_name)
    return PineconeVectorStore(index, embedding_model)

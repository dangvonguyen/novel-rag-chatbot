from langchain_core.embeddings import Embeddings


def make_embedder(model: str) -> Embeddings:
    """Connect to the configured embedding model."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)

        case "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=model)

        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}.")

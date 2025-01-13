from src.components.document_loaders import DocumentManager
from src.components.retrievers import make_retriever


def add_docs_to_store(data_dir):
    """
    Add documents from the specified path to the vector store

    Args:
        data_dir (str): Path to the documents directory
    """
    # Initialize document manager
    doc_manager = DocumentManager()

    # Load and chunk documents
    docs = doc_manager.load_documents(data_dir)
    chunked_docs = doc_manager.split_documents(docs)

    # Add documents to vector store
    with make_retriever() as retriever:
        retriever.add_documents(chunked_docs)


if __name__ == "__main__":
    data_dir = "data"
    add_docs_to_store(data_dir)

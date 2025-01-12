from src.components.document_loaders import DocumentManager
from src.components.retrievers import make_retriever
from src.graph import build_graph


# Load data
doc_manager = DocumentManager()
docs = doc_manager.load_documents("data")
chunked_docs = doc_manager.split_documents(docs)

# Add to vector store
retriever = make_retriever()
retriever.add_documents(chunked_docs)

# Build graph
graph = build_graph()

# Ask a question
response = graph.invoke({"question": "Trần Bình An là ai?"})
print(response["answer"])

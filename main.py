import os

from src.components.document_loaders import DocumentManager
from src.components.vectorstores import make_pinecone_vectorstore
from src.graph import build_graph

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG"

# Load data
doc_manager = DocumentManager()
docs = doc_manager.load_documents("data")
chunked_docs = doc_manager.split_documents(docs)

# Add to vector store
vector_store = make_pinecone_vectorstore()
vector_store.add_documents(chunked_docs)

# Build graph
graph = build_graph()

# Ask a question
response = graph.invoke({"question": "Trần Bình An là ai?"})
print(response["answer"])

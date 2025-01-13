from src.components.document_loaders import DocumentManager
from src.components.retrievers import make_retriever
from src.graph.graph import graph as chat


# Load data
doc_manager = DocumentManager()
docs = doc_manager.load_documents("data")
chunked_docs = doc_manager.split_documents(docs)

# Add to vector store
with make_retriever() as retriever:
    retriever.add_documents(chunked_docs)

# Prepare the conversation
question = "Trong 10 chương đầu của Kiếm Lai, Trần Bình An được mô tả là một người như thế nào?"

messages = [
    {"role": "user", "content": question},
]

# Run the conversation
response = chat.invoke({"messages": messages})
print(response)

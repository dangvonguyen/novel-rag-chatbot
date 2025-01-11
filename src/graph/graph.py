from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from src.components import prompts
from src.components.llms import load_chat_model
from src.components.retrievers import make_embedder, make_pinecone_retriever
from src.graph.state import State


def retrieve(state: State) -> dict[str, list[Document]]:
    embedding_model = make_embedder("huggingface/hiieu/halong_embedding")
    retriever = make_pinecone_retriever(embedding_model)
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = PromptTemplate.from_template(prompts.QA_TEMPLATE)
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm = load_chat_model()
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_graph():
    # Define the graph
    builder = StateGraph(State)
    builder.add_sequence([retrieve, generate])
    builder.add_edge(START, "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()

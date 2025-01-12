from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.components.llms import load_chat_model
from src.components.retrievers import make_retriever
from src.configs import Configuration
from src.graph.state import State
from src.utils import format_docs


def retrieve(state: State, *, config: RunnableConfig) -> dict[str, list[Document]]:
    with make_retriever(config) as retriever:
        retrieved_docs = retriever.invoke(state["question"])
    return {"documents": retrieved_docs}


def generate(state: State, *, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state["documents"])
    prompt = PromptTemplate.from_template(configuration.qa_system_prompt)
    messages = prompt.invoke({"query": state["question"], "context": context})
    response = model.invoke(messages)
    return {"answer": response.content}


def build_graph():
    # Define the graph
    builder = StateGraph(State)
    builder.add_sequence([retrieve, generate])
    builder.add_edge(START, "retrieve")
    builder.add_edge("generate", END)

    return builder.compile()

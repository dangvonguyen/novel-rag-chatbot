from typing import Any, Literal, cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.components.retrievers import make_retriever
from src.configs import Configuration
from src.graph.state import Router, State
from src.utils import format_docs, load_chat_model


def retrieve(state: State, *, config: RunnableConfig) -> dict[str, list[Document]]:
    with make_retriever(config) as retriever:
        content = state["messages"][-1].content
        query = str(content) if content is not None else ""
        retrieved_docs = retriever.invoke(query)
    return {"documents": retrieved_docs}


def analyze_and_route_query(
    state: State, *, config: RunnableConfig
) -> dict[str, Router]:
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state["messages"]
    response = cast(Router, model.with_structured_output(Router).invoke(messages))
    return {"router": response}


def route_query(
    state: State,
) -> Literal["ask_for_more_info", "respond_to_general_query", "retrieve"]:
    _type = state["router"]["type"]
    match _type:
        case "more-info":
            return "ask_for_more_info"
        case "general":
            return "respond_to_general_query"
        case "retrieve":
            return "retrieve"
        case _:
            raise ValueError(f"Unknown router type: {_type}")


def ask_for_more_info(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate answer."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model).with_config(
        config={"run_name": "respond"}
    )
    prompt = configuration.more_info_system_prompt.format(
        logic=state["router"]["logic"]
    )
    messages = [{"role": "system", "content": prompt}] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def respond_to_general_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate answer."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model).with_config(
        config={"run_name": "respond"}
    )
    prompt = configuration.general_system_prompt.format(logic=state["router"]["logic"])
    messages = [{"role": "system", "content": prompt}] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def respond_with_context(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate answer."""
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model).with_config(
        {"run_name": "respond"}
    )
    context = format_docs(state["documents"])
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def summarize_conversation(state: State, *, config: RunnableConfig) -> dict[str, Any]:
    if len(state["messages"]) <= 4:
        return {}

    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    summary = state.get("summary", "")
    prompt = configuration.summary_system_prompt.format(summary=summary)
    summary_messages = state["messages"][:-4] + [{"role": "system", "content": prompt}]
    delete_messages = [
        RemoveMessage(id=m.id) for m in state["messages"][:-4] if m.id is not None
    ]
    response = model.invoke(summary_messages)
    return {"summary": response.content, "messages": delete_messages}


# Define the graph
workflow = StateGraph(State)
workflow.add_node(analyze_and_route_query)
workflow.add_node(retrieve)
workflow.add_node(ask_for_more_info)
workflow.add_node(respond_to_general_query)
workflow.add_node(respond_with_context)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "analyze_and_route_query")
workflow.add_conditional_edges("analyze_and_route_query", route_query)
workflow.add_edge("retrieve", "respond_with_context")
workflow.add_edge("ask_for_more_info", "summarize_conversation")
workflow.add_edge("respond_to_general_query", "summarize_conversation")
workflow.add_edge("respond_with_context", "summarize_conversation")
workflow.add_edge("summarize_conversation", END)

# Compile into a graph object
graph = workflow.compile()

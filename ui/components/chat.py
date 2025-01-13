import asyncio
import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.graph import graph as chat


def init_chat_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "is_responding" not in st.session_state:
        st.session_state.is_responding = False


def display_conversation():
    for message in st.session_state.conversation:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "ai"):
            formatted_content = message.content.replace("\n", "  \n")
            st.markdown(formatted_content, unsafe_allow_html=True)


def handle_user_input():
    user_query = st.chat_input(disabled=st.session_state.is_responding)

    if user_query:
        st.session_state.conversation.append(HumanMessage(content=user_query))
        st.session_state.is_responding = True
        st.rerun()


def handle_ai_response():
    if st.session_state.is_responding and st.session_state.conversation:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            def update_ui(text):
                response_placeholder.markdown(text)

            full_response = asyncio.run(_stream_response(update_ui))

            # Update final response without cursor
            response_placeholder.markdown(full_response)

        st.session_state.conversation.append(AIMessage(content=full_response))
        st.session_state.is_responding = False
        st.rerun()


async def _stream_response(update_ui):
    full_response = ""
    async for event in chat.astream_events(
        {"messages": st.session_state.conversation},
        version="v2",
        include_names=["respond"],
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            full_response += chunk
            update_ui(full_response + "▌")
            time.sleep(0.03)
    return full_response

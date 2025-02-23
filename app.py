import streamlit as st
from dotenv import find_dotenv, load_dotenv

from ui.components import display_conversation, render_header, render_sidebar
from ui.components.chat import handle_ai_response, handle_user_input, init_chat_state
from ui.utils import load_css

load_dotenv(find_dotenv(raise_error_if_not_found=True))


def main():
    # Set page config
    title = "Novel RAG Chatbot"
    st.set_page_config(
        page_title=title,
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Load and apply CSS
    css = load_css("ui/assets/css/styles.css")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Render components
    render_header(title)
    render_sidebar()

    # Initialize chat state
    init_chat_state()

    # Display chat interface
    display_conversation()
    handle_user_input()
    handle_ai_response()


if __name__ == "__main__":
    main()

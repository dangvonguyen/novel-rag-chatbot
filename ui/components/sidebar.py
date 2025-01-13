import streamlit as st


def render_sidebar():
    with st.sidebar:
        render_new_chat()


def render_new_chat(include_divider=False):
    if include_divider:
        st.divider()

    st.header("💭 **New Chat**")
    if st.button("Start New Chat", type="primary", use_container_width=True, icon="💭"):
        st.session_state.conversation = []
        st.session_state.is_responding = False
        st.rerun()

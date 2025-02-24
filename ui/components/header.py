import streamlit as st


def render_header(header: str = "MindEase") -> None:
    body = f"""<div class="header"><h3>{header}</h3></div>"""
    st.markdown(body, unsafe_allow_html=True)
